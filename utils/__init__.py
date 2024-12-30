import torch
import argparse
import yaml
import onnx
import onnxruntime

from .logging_utils import get_logger_ready
from vit import ViT

logger = get_logger_ready("utils")


def parse_config(from_argparse=True, **kwargs):
    default_config_path = kwargs.get("default_config_path", "./config/run_args.yaml")

    if not from_argparse:
        with open(default_config_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    parser = argparse.ArgumentParser(description="Process config file path.")
    parser.add_argument(
        "--config-path",
        type=str,
        default="./config/run_args.yaml",
        # required=True,
        help="Path to the configuration JSON file",
    )
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        default=False,
        help="Perform a dry run without making any changes",
    )
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    if args.dry_run:
        logger.info(f"🔍 Dry run. Config: {config}")
        exit(0)
    return config


def gen_data(data_shape: tuple):
    return torch.randn(data_shape)


def get_model(model_config: dict, verbose=False) -> torch.nn.Module:
    return ViT(config=model_config, verbose=True)


def load_and_run_onnx(onnx_filepath, _x, print_output=False):
    logger.info("Loading and running ONNX model")
    onnx_model = onnx.load(onnx_filepath)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(
        onnx_filepath, providers=["CPUExecutionProvider"]
    )

    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(_x)}
    ort_outs = ort_session.run(None, ort_inputs)

    if print_output:
        logger.info(f"[onnx] Output: {ort_outs}")

    return ort_outs
