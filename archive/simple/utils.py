import argparse


def parse_args(from_argparse=True, **kwargs):
    # default_config_path = kwargs.get("default_config_path", "./config/run_args.yaml")

    # if not from_argparse:
    #     with open(default_config_path, "r") as f:
    #         config = yaml.safe_load(f)
    #     return config

    parser = argparse.ArgumentParser(description="Process config file path.")
    # parser.add_argument(
    #     "--config-path",
    #     type=str,
    #     default="./config/run_args.yaml",
    #     # required=True,
    #     help="Path to the configuration JSON file",
    # )
    # parser.add_argument(
    #     "-d",
    #     "--dry-run",
    #     action="store_true",
    #     default=False,
    #     help="Perform a dry run without making any changes",
    # )

    parser.add_argument(
        "--export-onnx",
        "-e",
        action="store_true",
        default=False,
        help="Export model to ONNX format",
    )

    parser.add_argument(
        "--save-onnx",
        "-s",
        action="store_true",
        default=False,
        help="Whether to save the exported model",
    )
    args = parser.parse_args()

    return args
