import logging
from colorama import Fore, Style


def yellow_txt(txt: str) -> str:
    return f"{Fore.YELLOW}{txt}{Style.RESET_ALL}"


def get_logger_ready(name: str) -> logging.Logger:
    # Set up logging
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create a stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter(
        "[%(levelname)s] [%(asctime)s][%(name)s]: %(message)s"
    )
    formatter.default_time_format = "%H:%M:%S"
    formatter.default_msec_format = "%s.%03d"
    stream_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(stream_handler)

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
