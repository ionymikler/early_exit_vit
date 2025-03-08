import torch
import logging
from colorama import Fore, Style


def color_txt(txt: str, color_name: str) -> str:
    color_constant = getattr(Fore, color_name.upper())
    return f"{color_constant}{txt}{Style.RESET_ALL}"


def yellow_txt(txt: str) -> str:
    return color_txt(txt, "yellow")


def green_txt(txt: str) -> str:
    return color_txt(txt, "green")


def get_logger_ready(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging with duplicate handler prevention.

    Args:
        name: Name of the logger
        level: Log level

    Returns:
        Configured logger
    """
    # Set up logging
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if logger already has handlers to avoid duplicates
    if not logger.handlers:
        # Create a stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)

        # Create a formatter and set it for the handler
        formatter = logging.Formatter(
            "[%(levelname)s] [%(asctime)s][%(name)s:%(lineno)d]: %(message)s"
        )
        formatter.default_time_format = "%H:%M:%S"
        formatter.default_msec_format = "%s.%03d"
        stream_handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(stream_handler)

        # Prevent propagation to the root logger to avoid duplicate messages
        logger.propagate = False

    return logger


def announce(logger, msg: str):
    logger.info(f"ℹ️  {msg}")


def print_dict(dictionary, ident="", braces=1):
    """Recursively prints nested dictionaries."""

    for key, value in dictionary.items():
        if isinstance(value, dict):
            print(f'{ident}{braces*"["}{key}{braces*"]"}')
            print_dict(value, ident + "  ", braces + 1)
        else:
            print(f"{ident}{key} = {value}")


def get_tensor_stats(tensor):
    return {
        "shape": tensor.shape,
        "sum": tensor.sum().item(),
        "mean": tensor.mean().item(),
        "std": tensor.std().item(),
        "min": tensor.min().item(),
        "max": tensor.max().item(),
        "norm": torch.norm(tensor).item(),
        # 'hash': hash(tensor.cpu().numpy().tobytes())
    }


def gts(tensor):
    return print_dict(get_tensor_stats(tensor))
