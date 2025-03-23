#!/usr/bin/env python
# Made by: Jonathan Mikler on 2025-03-21
"""
Utility functions for model profiling and performance measurement.
"""

import torch
from tqdm import tqdm
from typing import Callable

from utils import logging_utils

logger = logging_utils.get_logger_ready(__name__, level="DEBUG")


def warmup_model(predictor_fn: Callable, test_loader):
    """
    Warmup the model by running a few batches of dummy data.
    This is done to ensure more accurate performance measurements by allowing
    various hardware optimizations (like caching, JIT compilation, etc.) to stabilize.

    Args:
        predictor_fn: Function that takes input data and returns predictions
        test_loader: DataLoader for test set (used to determine input shape)
    """
    _WARMUP_ITERS = 200
    logger.info(
        logging_utils.yellow_txt(
            f"Performing model warmup with {_WARMUP_ITERS} iterations..."
        )
    )

    # Generate dummy data with the same shape as the real inputs
    dummy_shape = next(iter(test_loader))["pixel_values"].shape
    dummy_input = torch.randn(dummy_shape)

    # Run warmup iterations with progress bar
    pbar = tqdm(range(_WARMUP_ITERS), desc="Warming up", unit="iter")
    for _ in pbar:
        _ = predictor_fn(dummy_input)

    logger.info("Warmup complete")


def check_before_profiling(args):
    """
    Check if profiling settings are appropriate and warn the user if not.

    Args:
        args: Command-line arguments object with profile_do and num_examples attributes
    """
    if args.profile_do and (
        (args.num_examples is not None and args.num_examples > 100)
        or args.num_examples is None
    ):
        logger.warning(
            logging_utils.yellow_txt(
                "Profiling is enabled and the number of examples is greater than 20. "
                "This may result in a large profiling file. "
                "Consider reducing the number of examples or disabling profiling."
            )
        )
        exit()


def save_profiler_output(profile_result, results_dir: str, profile_type: str):
    """
    Save profiling results to the specified directory.

    Args:
        profile_result: Profiling result object (PyTorch profile or ONNX profile path)
        results_dir: Directory to save profiling results
        profile_type: Type of profile ('pytorch' or 'onnx')
    """
    import os
    import shutil

    if profile_type == "pytorch":
        # For PyTorch profiler
        output_path = f"{results_dir}/pytorch_profiler_trace.json"
        profile_result.export_chrome_trace(output_path)
        logger.info(f"PyTorch profiler output saved to {output_path}")

    elif profile_type == "onnx":
        # For ONNX profiler (which produces a file path)
        profile_dest = f"{results_dir}/onnx_profiler_output.json"
        shutil.copy(profile_result, profile_dest)
        os.remove(profile_result)  # Remove temporary file
        logger.info(f"ONNX profiler output saved to {profile_dest}")

    else:
        logger.warning(f"Unknown profile type: {profile_type}")
