#!/usr/bin/env python
# Made by: Jonathan Mikler on 2024-12-03
import torch
import torch.nn as nn
from datetime import datetime

# local imports
from utils import (
    parse_args,
    get_config,
    get_model,
    gen_data,
    load_and_run_onnx,
    check_conda_env,
)
from utils.arg_utils import parse_config_dict
from utils.logging_utils import get_logger_ready, announce

logger = get_logger_ready("main")


def run_model(model, x, print_output=False):
    announce(logger, "Running model")
    out = model(x)
    logger.info(f"model output shape: {out.shape}")

    if print_output:
        logger.info(f"Output: {out}")
    return out


def export_model(model: nn.Module, _x, onnx_filepath: str):
    announce(logger, f"Exporting model '{model.name}' to ONNX format")

    model.eval()
    onnx_program = torch.onnx.export(
        model=model,
        args=(_x),
        dynamo=True,
        report=True,
        verbose=True,
    )
    onnx_program.save(onnx_filepath)
    logger.info(f"Model exported to '{onnx_filepath}' ✅")


def main():
    # Check conda environment
    if not check_conda_env("eevit"):
        return

    args = parse_args()
    config = get_config(args.config_path)

    if args.dry_run:
        logger.info(f"🔍 Dry run. Config: {config}")
        return

    # Dataset config
    dataset_config = config["dataset"]  # noqa F841
    # ViT config
    model_config = parse_config_dict(config["model"].copy())

    model = get_model(model_config)

    x = gen_data(
        data_shape=(
            1,
            dataset_config["channels_num"],
            dataset_config["image_size"],
            dataset_config["image_size"],
        )
    )
    # x = add_fast_pass(gen_data(data_shape=(1, 197, 768)))

    out_pytorch = run_model(x=x, model=model)

    if args.export_onnx:
        timestamp = datetime.now().strftime("%H-%M-%S")
        onnx_filepath = f"./models/onnx/{model.name}_{timestamp}.onnx"
        export_model(model=model, _x=x, onnx_filepath=onnx_filepath)

        out_ort = load_and_run_onnx(onnx_filepath, x)

        # Compare the outputs
        assert torch.allclose(
            out_pytorch, torch.tensor(out_ort[0]), atol=1e-5
        ), "Outputs are not equal"

        logger.info("Outputs are equal ✅")


if __name__ == "__main__":
    main()
