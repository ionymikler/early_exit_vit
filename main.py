#!/usr/bin/env python
# Made by: Jonathan Mikler on 2024-12-03
import os
import torch
import torch.nn as nn
from datetime import datetime

# local imports
from utils import (
    parse_args,
    get_config_dict,
    get_model,
    gen_data,
    load_and_run_onnx,
    check_conda_env,
)
from utils.arg_utils import parse_config_dict
from utils.logging_utils import get_logger_ready, announce, print_dict, yellow_txt

logger = get_logger_ready("main.py")
PRESS_ENTER_MSG = yellow_txt("Press Enter to continue...")


def run_model(model, data, print_output=False):
    announce(logger, "Running model")
    out = model(data)
    logger.info(f"model output shape: {out.shape}")

    if print_output:
        logger.info(f"Output: {out}")
    return out


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


def init_checks(args):
    if not args.skip_conda_env_check:
        if not check_conda_env("eevit"):
            exit()
    else:
        logger.info("Skipping conda environment check")

    config = get_config_dict(args.config_path)

    if args.dry_run:
        logger.info("🔍 Dry run. Config:")
        print_dict(config)
        exit()


def gen_random_input_data(dataset_config: dict):
    image_tensor = gen_data(  # TODO: Currently shape is (1,c,w,h). I'm I sure the shape is not (1, w,h,c)? Check CIFAR10
        data_shape=(
            1,
            dataset_config["channels_num"],
            dataset_config["image_size"],
            dataset_config["image_size"],
        )
    )
    # image_tensor = gen_data(model.patch_embedding.output_shape)
    # image_tensor = add_fast_pass(gen_data(data_shape=(1, 197, 768)))

    return image_tensor


def main():
    args = parse_args()

    init_checks(args)

    config = get_config_dict(args.config_path)

    if args.dry_run:
        logger.info(f"🔍 Dry run. Config: {config}")
        return

    # Dataset config
    dataset_config = config["dataset"]  # noqa F841

    # ViT config
    model_config = parse_config_dict(config["model"].copy())

    model = get_model(model_config)

    _ = input(PRESS_ENTER_MSG)

    # Generate random data
    dummy_image_tensor = gen_random_input_data(dataset_config)

    out_pytorch = run_model(data=dummy_image_tensor, model=model)
    _ = input(PRESS_ENTER_MSG)

    if args.export_onnx:
        model_name = (
            f"{model.name}_{args.onnx_filename_suffix}"
            if args.onnx_filename_suffix
            else model.name
        )
        timestamp = datetime.now().strftime("%H-%M")
        onnx_filepath = f"./models/onnx/{model_name}_{timestamp}.onnx"
        onnx_program = export_model(
            model=model, _x=dummy_image_tensor, report=args.report
        )

        onnx_program.save(onnx_filepath)
        logger.info(f"ONNX model saved at: {onnx_filepath}")

        announce(logger, "Loading and running the ONNX model...")
        out_ort = load_and_run_onnx(onnx_filepath, dummy_image_tensor)

        if not args.keep_onnx:
            os.remove(onnx_filepath)

        # Compare the outputs
        assert torch.allclose(
            out_pytorch, torch.tensor(out_ort[0]), atol=1e-5
        ), "Outputs are not equal"

        logger.info("Outputs are equal ✅")


if __name__ == "__main__":
    main()
