from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
import torch
from sklearn.metrics import accuracy_score

from mlops_project.data import NewsDataModule
from mlops_project.model import Model


class PyTorchPredictor:
    """PyTorch model inference wrapper."""

    def __init__(self, model: Model, device: str = "cpu") -> None:
        self.model = torch.compile(model, mode="reduce-overhead").to(device)
        self.model.eval()
        self.device = device

    def predict(self, batch_text: torch.Tensor) -> np.ndarray:
        """Return predicted class indices for a batch."""
        with torch.inference_mode():
            inputs = batch_text.to(self.device)
            logits = self.model(inputs)
            return torch.argmax(logits, dim=1).cpu().numpy()


class ONNXPredictor:
    """ONNX Runtime inference wrapper."""

    def __init__(self, onnx_path: Path) -> None:
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, batch_text: torch.Tensor | np.ndarray) -> np.ndarray:
        """Return predicted class indices for a batch."""
        if isinstance(batch_text, torch.Tensor):
            batch_text = batch_text.cpu().numpy()

        inputs = {self.input_name: batch_text}
        logits = self.session.run(None, inputs)[0]
        return np.argmax(logits, axis=1)


def benchmark(
    predictor: PyTorchPredictor | ONNXPredictor,
    dataloader: torch.utils.data.DataLoader,
    limit_batches: int | None = None,
) -> dict[str, float | list[int]]:
    """Benchmark a predictor over a dataloader."""
    all_preds = []
    all_labels = []
    latencies = []

    print("Warming up...")
    for i, batch in enumerate(dataloader):
        if i >= 5:
            break
        text, _ = batch
        predictor.predict(text)

    print("Starting benchmark...")
    start_global = time.perf_counter()

    for i, batch in enumerate(dataloader):
        if limit_batches and i >= limit_batches:
            break

        texts, labels = batch
        if labels.dim() > 1:
            labels = labels.squeeze()

        t0 = time.perf_counter()
        preds = predictor.predict(texts)
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)

        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    total_time = time.perf_counter() - start_global

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)

    return {
        "accuracy": acc,
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": p95_latency,
        "total_time_sec": total_time,
        "predictions": all_preds,
        "labels": all_labels,
    }


def _load_checkpoint_metadata(model_path: Path) -> tuple[dict[str, int], int, dict[str, object]]:
    """Load vocab and hyperparameters from a Lightning checkpoint."""
    checkpoint = torch.load(model_path, map_location="cpu")
    vocab = checkpoint.get("vocab")
    vocab_size = checkpoint.get("vocab_size")
    hyper_params = checkpoint.get("hyper_parameters", {})

    if vocab is None or vocab_size is None:
        raise ValueError("Checkpoint must contain 'vocab' and 'vocab_size'.")

    return vocab, int(vocab_size), hyper_params


def _load_pytorch_model(model_path: Path) -> tuple[Model, dict[str, int], int]:
    """Load a PyTorch Lightning model with checkpoint metadata."""
    vocab, vocab_size, hyper_params = _load_checkpoint_metadata(model_path)
    hyper_params_clean = {k: v for k, v in hyper_params.items() if k != "vocab_size"}
    model = Model.load_from_checkpoint(
        str(model_path),
        vocab_size=vocab_size,
        **hyper_params_clean,
        strict=False,
    )
    return model, vocab, vocab_size


def compare_models() -> None:
    """Compare PyTorch and ONNX accuracy and latency on the validation split."""
    model_path = Path("models/model.pt")
    onnx_path = Path("models/model_optimized.onnx")
    data_path = Path("data/processed")

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    pt_model, vocab, vocab_size = _load_pytorch_model(model_path)

    dm = NewsDataModule(data_path=data_path, batch_size=16)
    dm.vocab = vocab
    dm.vocab_size = vocab_size
    dm.setup("validate")
    loader = dm.val_dataloader()

    pt_predictor = PyTorchPredictor(pt_model, device="cpu")
    pt_results = benchmark(pt_predictor, loader)

    onnx_predictor = ONNXPredictor(onnx_path)
    onnx_results = benchmark(onnx_predictor, loader)

    print(f"PyTorch Accuracy: {pt_results['accuracy']:.4f} | Avg Latency: {pt_results['avg_latency_ms']:.2f}ms | ")
    print(f"ONNX    Accuracy: {onnx_results['accuracy']:.4f} | Avg Latency: {onnx_results['avg_latency_ms']:.2f}ms")

    speedup = pt_results["avg_latency_ms"] / onnx_results["avg_latency_ms"]
    print(f"Speedup Factor: {speedup:.2f}x")


if __name__ == "__main__":
    compare_models()
