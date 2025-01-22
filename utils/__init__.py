import torch
import argparse
import yaml
import onnx
import onnxruntime

from eevit.eevit import EEVIT
from utils.arg_utils import ModelConfig
from .logging_utils import get_logger_ready

logger = get_logger_ready("utils")


def parse_args(from_argparse=True, **kwargs):
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

    parser.add_argument(
        "--export-onnx",
        "-e",
        action="store_true",
        default=False,
        help="Export model to ONNX format",
    )
    args = parser.parse_args()

    return args


def get_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def gen_data(data_shape: tuple):
    return torch.randn(data_shape)


def get_model(model_config: ModelConfig, verbose=True) -> torch.nn.Module:
    return EEVIT(config=model_config, verbose=verbose)


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


__all__ = [
    "parse_args",
    "gen_data",
    "get_model",
    "load_and_run_onnx",
]
