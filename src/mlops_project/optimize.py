from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.nn.utils import prune as pruning

from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process

from mlops_project.model import Model


MODEL_PATH: Path = Path("models/model.pt")


def load_model(model_path: Path = MODEL_PATH) -> Model:
    """Load a trained model checkpoint.

    Args:
        model_path: Path to the model checkpoint.

    Returns:
        Loaded model instance.
    """
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    vocab_size = checkpoint.get("vocab_size")
    hyper_params = checkpoint.get("hyper_parameters", {})
    hyper_params_clean = {key: value for key, value in hyper_params.items() if key != "vocab_size"}
    return Model.load_from_checkpoint(
        model_path,
        vocab_size=vocab_size,
        **hyper_params_clean,
        strict=False,
    )

def _parameters_to_prune(model: nn.Module) -> Tuple[Tuple[nn.Module, str], ...]:
    """Collect prunable parameters across the model.

    Args:
        model: PyTorch model to prune.

    Returns:
        Tuple of (module, parameter_name) pairs for pruning.
    """
    module_lookup = dict(model.named_modules())
    parameters: list[tuple[nn.Module, str]] = []
    for name, param in model.named_parameters():
        if param.ndim < 2:
            continue
        if "." not in name:
            continue
        module_name, param_name = name.rsplit(".", 1)
        module = module_lookup.get(module_name)
        if module is None:
            continue
        parameters.append((module, param_name))
    return tuple(parameters)

def prune_model(model: nn.Module, amount: float = 0.2) -> nn.Module:
    """Prune a model with global unstructured pruning.

    Args:
        model: PyTorch model to prune.
        amount: Fraction of weights to prune.

    Returns:
        Pruned model.
    """
    parameters_to_prune = _parameters_to_prune(model)[:-1]
    pruning.global_unstructured(
        parameters_to_prune,
        pruning_method=pruning.L1Unstructured,
        amount=amount,
    )
    for module, param_name in parameters_to_prune:
        pruning.remove(module, param_name)
    return model


def onnx_port(
    model: nn.Module,
    output_path: Path | None = None,
    seq_length: int = 128,
    batch_size: int = 1,
) -> Path:
    """Export a model to ONNX.

    Args:
        model: PyTorch model to export.
        output_path: Optional path for the exported ONNX file.
        opset_version: ONNX opset version to use.
        seq_length: Sequence length for dummy input.
        batch_size: Batch size for dummy input.

    Returns:
        Path to the exported ONNX file.
    """

    model.eval()
    onnx_raw = output_path.with_name(output_path.stem + "_raw.onnx")
    onnx_preprocessed = output_path.with_name(output_path.stem + "_preprocessed.onnx")

    dummy_input = torch.zeros((batch_size, seq_length), dtype=torch.long)

    torch.onnx.export(
        model=model,
        args=(dummy_input,),
        f=str(onnx_raw),
        input_names=["input"], 
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size", 1: "seq_len"}, "output": {0: "batch_size"}},
        dynamo=False,
    )

    quant_pre_process(
            input_model=str(onnx_raw),
            output_model_path=str(onnx_preprocessed),
            skip_optimization=False
        )
                                           
    quantize_dynamic(
        model_input=str(onnx_preprocessed),
        model_output=str(output_path),
        weight_type=QuantType.QInt8
    )

    onnx_preprocessed.unlink()
    onnx_raw.unlink()

if __name__ == "__main__":
    model = load_model()
    # pruned_model = prune_model(model, amount=0.3)
    onnx_port(model, output_path=Path("models/model_optimized.onnx"))
