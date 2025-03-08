import argparse
import yaml
from dataclasses import dataclass
from typing import List, Tuple, Literal

from .logging_utils import get_logger_ready

logger = get_logger_ready(__name__)

DEFAULT_CONFIG_PATH = "./config/run_args.yaml"


@dataclass
class EarlyExitsConfig:
    """Configuration for early exits in the model."""

    # Required parameters (no defaults)
    embed_depth: int
    num_classes: int
    num_attn_heads: int
    confidence_threshold: float

    # Optional parameters with defaults
    general_dropout: float = 0.0
    exit_strategy: Literal["confidence"] = "confidence"  # Default strategy
    exits: List[Tuple[int, str, dict]] = (
        None  # Will be set to [(4, 'dummy', {})] in __post_init__
    )

    def __post_init__(self):
        if self.exits is None:
            self.exits = []
        # Ensure all exits have 3 elements with the correct type
        validated_exits = []
        for exit in self.exits:
            if len(exit) == 2:
                validated_exits.append((exit[0], exit[1], {}))
            elif len(exit) == 3:
                assert isinstance(exit[2], dict), "Third entry must be a dictionary"
                validated_exits.append(exit)
            else:
                raise ValueError("Each exit must have either 2 or 3 elements")
        self.exits = validated_exits

    @property
    def exit_list(self):
        return [exit[0] for exit in self.exits]


@dataclass
class ModelConfig:
    """Main configuration class for the vision transformer model."""

    # Required parameters (no defaults)
    channels_num: int
    image_size: int
    num_classes: int

    # Optional parameters with defaults
    pool: Literal["cls"] = "cls"
    embed_depth: int = 768  # 'hidden_size' in LGViT
    patch_size: int = 16
    num_attn_heads: int = 12  # 'num_attention_heads' in LGViT
    general_dropout: float = 0.0
    transformer_dropout: float = 0.0
    mlp_dim: int = 3072  # 'intermediate_size' in LGVIT
    dim_head: int = 64  # hardcoded here, computed as config.hidden_size / config.num_attention_heads in LGVIT
    num_layers_transformer: int = 12  # 'num_hidden_layers' in LGViT
    early_exit_config: EarlyExitsConfig = (
        None  # Will be set to default EarlyExitsConfig in __post_init__
    )
    enable_export: bool = False

    def __post_init__(self):
        """Validate configuration parameters."""
        # Set default EarlyExitsConfig if none provided
        if self.early_exit_config is None:
            self.early_exit_config = EarlyExitsConfig()

        # Validate numeric ranges
        if self.channels_num <= 0:
            raise ValueError("channels_num must be positive")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")

        # Validate architecture parameters
        if self.embed_depth <= 0:
            raise ValueError("embed_depth must be positive")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if self.num_attn_heads <= 0:
            raise ValueError("num_attn_heads must be positive")
        if not 0 <= self.general_dropout <= 1:
            raise ValueError("dropout_embedding must be between 0 and 1")
        if not 0 <= self.transformer_dropout <= 1:
            raise ValueError("dropout_transformer must be between 0 and 1")
        if self.mlp_dim <= 0:
            raise ValueError("mlp_dim must be positive")
        if self.dim_head <= 0:
            raise ValueError("dim_head must be positive")
        if self.num_layers_transformer <= 0:
            raise ValueError("num_layers_transformer must be positive")

        # Validate early exits
        for entry in self.early_exit_config.exits:
            layer = entry[0]
            if not 0 <= layer < self.num_layers_transformer:
                raise ValueError(
                    f"Error: Not possible to assign Exit after layer '{layer}'. Index be between 0 and {self.num_layers_transformer-1}"
                )


def get_config_dict(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def parse_config_dict(model_dict: dict) -> ModelConfig:
    """
    Parses a dictionary (loaded from YAML) into a ModelConfig object.

    Args:
        model_dict: Dictionary containing model configuration

    Returns:
        ModelConfig: Configured model configuration object

    Raises:
        KeyError: If required keys are missing from the config
    """
    try:
        # Handle early_exits separately if present
        if "early_exit_config" in model_dict:
            early_exits_dict = model_dict.pop("early_exit_config")
            # Create the EarlyExitsConfig and assign it back to model_dict
            model_dict["early_exit_config"] = EarlyExitsConfig(**early_exits_dict)

        # Create ModelConfig with all parameters
        return ModelConfig(**model_dict)

    except KeyError as e:
        raise KeyError(f"Missing required configuration key: {e}")


def get_argsparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and run an EEVIT model, as specified in the configuration file"
    )

    parser.add_argument(
        "--config-path",
        type=str,
        default="./config/run_args.yaml",
        # required=True,
        help="Path to the configuration JSON file. Default: './config/run_args.yaml'",
    )

    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        default=False,
        help="Perform a dry run without making any changes",
    )

    parser.add_argument(
        "--skip-conda-env-check",
        action="store_true",
        default=False,
        help="Skip the check for the required conda environment",
    )

    parser.add_argument(
        "-n",
        "--num-examples",
        type=int,
        default=None,
        help="Number of examples to evaluate. If not specified, uses all available examples.",
    )

    ### EVALUATION ARGUMENTS ###
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Run in interactive mode: evaluate one image at a time and show detailed results",
    )

    parser.add_argument(
        "-s",
        "--save-metrics",
        action="store_true",
        help="Save evaluation metrics to a JSON file",
    )

    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=False,
        help="Use GPU for model inference if available",
    )

    # ONNX evaluation arguments
    parser.add_argument(
        "--onnx-program-filepath",
        "-f",
        type=str,
        default="./models/onnx/EEVIT.onnx",
        help="Path to save the ONNX model file",
    )

    parser.add_argument(
        "--profile-do",
        "-p",
        action="store_true",
        default=False,
        help="Enable profiling for the ONNX model",
    )

    return parser


def get_export_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export the model to ONNX format")

    parser.add_argument(
        "--onnx-report",
        "-r",
        action="store_true",
        default=False,
        help="Print torch's report on exported ONNX model",
    )

    parser.add_argument(
        "--onnx-keep-file",
        "-k",
        dest="onnx_keep",
        action="store_true",
        help="Keep the ONNX model file after running it",
    )
    parser.set_defaults(onnx_keep=False)  # Default is True

    parser.add_argument(
        "--onnx-program-filepath",
        "-f",
        type=str,
        default="./models/onnx/default.onnx",
        help="Path to save the ONNX model file",
    )

    parser.add_argument(
        "--onnx-filename-suffix",
        type=str,
        default="",
        help="Suffix to append to the ONNX filename",
    )

    return parser
