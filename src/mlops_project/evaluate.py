from pathlib import Path

import torch
from loguru import logger
from sklearn.metrics import confusion_matrix
import typer
from lightning import Trainer

from mlops_project.model import Model
from mlops_project.data import NewsDataModule


def evaluate(
    model_path: Path = Path("models/model.pt"),
    data_path: Path = Path("data/processed"),
    batch_size: int = 32,
    accelerator: str = "auto",
    devices: int = 1,
) -> dict[str, float]:
    """Evaluate the model on test data using PyTorch Lightning."""
    checkpoint = torch.load(model_path, map_location="cpu")
    vocab = checkpoint.get("vocab")
    vocab_size = checkpoint.get("vocab_size")
    hyper_params = checkpoint.get("hyper_parameters", {})

    if vocab is None or vocab_size is None:
        raise ValueError("Checkpoint must contain 'vocab' and 'vocab_size'")

    data_module = NewsDataModule(data_path=data_path, batch_size=batch_size)
    data_module.vocab = vocab
    data_module.vocab_size = vocab_size
    data_module.setup("test")

    if hyper_params:
        hyper_params_clean = {k: v for k, v in hyper_params.items() if k != "vocab_size"}
        model = Model.load_from_checkpoint(
            model_path,
            vocab_size=vocab_size,
            **hyper_params_clean,
            strict=False,
        )
    else:
        model = Model.load_from_checkpoint(
            model_path,
            vocab_size=vocab_size,
            strict=False,
        )

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=True,
        logger=False,
    )

    results = trainer.test(model, data_module)

    test_metrics = results[0] if results else {}

    all_predictions = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in data_module.test_dataloader():
            texts, labels = batch
            outputs = model(texts)
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_predictions)

    metrics = {
        "loss": test_metrics.get("test_loss", 0.0),
        "accuracy": test_metrics.get("test_acc", 0.0),
        "precision": test_metrics.get("test_precision", 0.0),
        "recall": test_metrics.get("test_recall", 0.0),
        "f1_score": test_metrics.get("test_f1", 0.0),
        "confusion_matrix": cm.tolist(),
    }

    logger.info(f"Test Loss: {metrics['loss']:.4f}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")

    return metrics


if __name__ == "__main__":
    typer.run(evaluate)
