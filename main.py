#!/usr/bin/env python
# Made by: Jonathan Mikler on 2024-12-03
import torch
import torch.nn as nn
from datetime import datetime

# local imports
import utils as my_utils
from utils.logging_utils import get_logger_ready

logger = get_logger_ready("main")


def announce(msg: str):
    logger.info(f"ℹ️  {msg}")


def run_model(model, x, print_output=False):
    announce("Running model")
    out = model(x)
    logger.info(f"model output shape: {out.shape}")

    if print_output:
        logger.info(f"Output: {out}")
    return out


def export_model(model: nn.Module, _x, onnx_filepath: str):
    announce(f"Exporting model '{model.name}' to ONNX format")

    onnx_program = torch.onnx.export(
        model=model,
        args=(_x),
        dynamo=True,
        report=True,
    )
    onnx_program.save(onnx_filepath)
    logger.info(f"✅ Model exported to '{onnx_filepath}'")


def main():
    args = my_utils.parse_config()

    # Dataset config
    dataset_config = args["dataset"]
    # ViT config
    model_config = args["model"]

    model = my_utils.get_model(model_config)

    x = my_utils.gen_data(
        data_shape=(
            2,
            dataset_config["channels_num"],
            dataset_config["image_size"],
            dataset_config["image_size"],
        )
    )
    out_pytorch = run_model(x=x, model=model)

    timestamp = datetime.now().strftime("%H-%M-%S")
    onnx_filepath = f"./models/onnx/{model.name}_{timestamp}.onnx"
    export_model(model=model, _x=x, onnx_filepath=onnx_filepath)

    out_ort = my_utils.load_and_run_onnx(onnx_filepath, x)

    # Compare the outputs
    assert torch.allclose(
        out_pytorch, torch.tensor(out_ort[0]), atol=1e-5
    ), "Outputs are not equal"

    logger.info("✅ Outputs are equal")


if __name__ == "__main__":
    main()
