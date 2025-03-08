# utils/__init__.py
import torch
import numpy as np
import onnx
import onnxruntime

from .logging_utils import get_logger_ready, announce, print_dict, yellow_txt

logger = get_logger_ready("utils")


def gen_data(data_shape: tuple):
    return torch.randn(data_shape)


def load_and_run_onnx(onnx_filepath, _x, print_output=False):
    announce(logger, "Loading and running ONNX model")
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


def check_conda_env(conda_env_required):
    import os

    active_env = os.environ.get("CONDA_DEFAULT_ENV")
    if active_env != conda_env_required:
        logger.error(
            yellow_txt(
                f"ERROR: Conda environment '{conda_env_required}' is required. Please activate it."
            )
        )
        return False
    return True


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


__all__ = [
    "gen_data",
    "load_and_run_onnx",
    "print_dict",
    "check_conda_env",
]
