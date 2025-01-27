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
from utils.logging_utils import get_logger_ready, announce, print_dict

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

    with torch.no_grad():
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


def init_checks(args):
    if not args.skip_conda_env_check:
        if not check_conda_env("eevit"):
            exit()
    else:
        logger.info("Skipping conda environment check")

    config = get_config(args.config_path)

    if args.dry_run:
        logger.info("🔍 Dry run. Config:")
        print_dict(config)
        exit()


def main():
    args = parse_args()

    init_checks(args)

    config = get_config(args.config_path)

    if args.dry_run:
        logger.info(f"🔍 Dry run. Config: {config}")
        return

    # Dataset config
    dataset_config = config["dataset"]  # noqa F841

    # ViT config
    model_config = parse_config_dict(config["model"].copy())

    model = get_model(model_config)

    # Generate random data
    # input_tensor = gen_data(model.patch_embedding.output_shape)

    input_tensor = gen_data(  # TODO: Currently shape is (1,c,w,h). I'm I sure the shape is not (1, w,h,c)? Check CIFAR10
        data_shape=(
            1,
            dataset_config["channels_num"],
            dataset_config["image_size"],
            dataset_config["image_size"],
        )
    )
    # input_tensor = add_fast_pass(gen_data(data_shape=(1, 197, 768)))

    out_pytorch = run_model(x=input_tensor, model=model)

    if args.export_onnx:
        model_name = (
            f"{model.name}_{args.onnx_filename_suffix}"
            if args.onnx_filename_suffix
            else model.name
        )
        timestamp = datetime.now().strftime("%H-%M")
        onnx_filepath = f"./models/onnx/{model_name}_{timestamp}.onnx"
        export_model(model=model, _x=input_tensor, onnx_filepath=onnx_filepath)

        out_ort = load_and_run_onnx(onnx_filepath, input_tensor)

        # Compare the outputs
        assert torch.allclose(
            out_pytorch, torch.tensor(out_ort[0]), atol=1e-5
        ), "Outputs are not equal"

        logger.info("Outputs are equal ✅")


if __name__ == "__main__":
    main()
