"""
Export ONNX nativo con torch.onnx.export.

No se usa `optimum` porque sus series 1.x (única con `optimum.onnxruntime`)
son incompatibles con torch >= 2.9 (se eliminó `torch.onnx.symbolic_opset14._attention_scale`).
La exportación directa con torch produce el mismo resultado: un modelo .onnx
inferible con onnxruntime en CPU.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)

ONNX_OPSET_VERSION = 14


def _build_dummy_input(tokenizer: Any, max_seq_length: int) -> Dict[str, torch.Tensor]:
    """Construye un batch dummy (1, seq) usando el propio tokenizer."""
    enc = tokenizer(
        "This is a dummy input for ONNX export.",
        max_length=max_seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return {k: v for k, v in enc.items()}


def export_model_to_onnx(
    model: torch.nn.Module,
    tokenizer: Any,
    output_dir: Path,
    max_seq_length: int = 128,
    opset_version: int = ONNX_OPSET_VERSION,
) -> Path:
    """
    Exporta un modelo transformers a ONNX con ejes dinámicos (batch, sequence).

    Returns:
        Path del archivo `model.onnx` exportado.
    """
    model.eval()
    model = model.to("cpu")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"

    dummy = _build_dummy_input(tokenizer, max_seq_length)

    dynamic_axes: Dict[str, Dict[int, str]] = {
        name: {0: "batch", 1: "sequence"} for name in dummy.keys()
    }

    import inspect

    export_kwargs: Dict[str, Any] = {
        "input_names": list(dummy.keys()),
        "output_names": ["logits"],
        "dynamic_axes": dynamic_axes,
        "opset_version": opset_version,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    with torch.no_grad():
        torch.onnx.export(
            model,
            tuple(dummy.values()),
            str(onnx_path),
            **export_kwargs,
        )

    logger.info("Modelo ONNX exportado a %s", onnx_path)
    return onnx_path
