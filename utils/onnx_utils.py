#!/usr/bin/env python
# Made by: Jonathan Mikler on 2025-02-20
import torch
import torch.nn as nn
import onnx


from .logging_utils import get_logger_ready, announce
from .model_utils import format_model_name

logger = get_logger_ready(__name__)


def export_model(model: nn.Module, _x, report=False) -> torch.onnx.ONNXProgram:
    announce(logger, f"Exporting model '{model.name}' to ONNX format")

    with torch.no_grad():
        model.eval()
        onnx_program = torch.onnx.export(
            model=model,
            args=(_x),
            input_names=["image"],
            # output_names=["predictions"],
            report=report,
            dynamo=True,
            verbose=True,
        )

    return onnx_program


def export_and_save(model: nn.Module, _x: torch.Tensor, report: bool = False) -> str:
    """
    Export model to ONNX format and save it to disk.

    Args:
        model: The model to export
        _x: The input tensor
        report: Whether to print torch's report on exported ONNX model

    Returns:
        str: Path to the saved ONNX model
    """

    onnx_program = export_model(model, _x, report)
    onnx_filepath = (
        # f"./models/onnx/{format_model_name(model.name)}_{datetime.now().strftime('%H-%M')}.onnx"
        f"./models/onnx/{format_model_name(model.name)}.onnx"
    )
    onnx_program.save(onnx_filepath)
    logger.info(f"ONNX model saved at: {onnx_filepath}")

    return onnx_filepath


def load_onnx_model(onnx_filepath: str) -> onnx.ModelProto:
    announce(logger, f"Loading ONNX model from '{onnx_filepath}'")
    onnx_program = onnx.load(onnx_filepath)
    onnx.checker.check_model(onnx_program)

    return onnx_program
