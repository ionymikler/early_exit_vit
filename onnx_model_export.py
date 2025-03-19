#!/usr/bin/env python
# Made by: Jonathan Mikler on 2024-12-03
import os
import torch

# local imports
from utils import (
    gen_data,
    load_and_run_onnx,
    check_conda_env,
)
from utils.model_utils import get_model
from utils.arg_utils import parse_config_dict, get_export_parser, get_config_dict
from utils.logging_utils import get_logger_ready, announce, print_dict, yellow_txt
from utils.onnx_utils import export_and_save

logger = get_logger_ready("onnx_model_export.py")
PRESS_ENTER_MSG = yellow_txt("Press Enter to continue...")


def run_model(model, data, print_output=False):
    announce(logger, "Running model")
    out = model(data)
    logger.info(f"model output shape: {out.shape}")

    if print_output:
        logger.info(f"Output: {out}")
    return out


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
    image_tensor = gen_data(
        # TODO: Currently shape is (1,c,w,h). I'm I sure the shape is not (1, w,h,c)? Check CIFAR10
        data_shape=(
            1,
            dataset_config["channels_num"],
            dataset_config["image_size"],
            dataset_config["image_size"],
        )
    )

    return image_tensor


def main():
    # NOTE: There might be some bug in the parser
    args = get_export_parser().parse_args()

    init_checks(args)

    config = get_config_dict(args.config_path)

    if args.dry_run:
        logger.info(f"🔍 Dry run. Config: {config}")
        return

    assert config["model"]["enable_export"] is True, yellow_txt(
        "export mode is disabled in config"
    )

    model_config = parse_config_dict(config["model"].copy())

    model = get_model(model_config)

    # Generate random data
    dataset_config_dict = config["dataset"]  # noqa F841
    dummy_image_tensor = gen_random_input_data(dataset_config_dict)

    out_pytorch = run_model(data=dummy_image_tensor, model=model)

    onnx_filepath = export_and_save(
        model, dummy_image_tensor, args.onnx_output_filepath, report=args.onnx_report
    )

    announce(logger, "Loading and running the ONNX model...")
    out_ort = load_and_run_onnx(onnx_filepath, dummy_image_tensor)

    if not args.onnx_keep:
        os.remove(onnx_filepath)

    # Compare the outputs
    assert torch.allclose(
        out_pytorch, torch.tensor(out_ort[0]), atol=1e-5
    ), "Outputs are not equal ❌"

    logger.info("Outputs are equal ✅")


if __name__ == "__main__":
    main()
