import logging


def get_logger_ready(name: str) -> logging.Logger:
    # Set up logging
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create a stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter("[%(levelname)s][%(name)s][%(asctime)s]: %(message)s")
    stream_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(stream_handler)

    return logger
