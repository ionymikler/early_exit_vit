from dataclasses import dataclass
from typing import List, Tuple, Literal


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
    early_exits: EarlyExitsConfig = (
        None  # Will be set to default EarlyExitsConfig in __post_init__
    )

    def __post_init__(self):
        """Validate configuration parameters."""
        # Set default EarlyExitsConfig if none provided
        if self.early_exits is None:
            self.early_exits = EarlyExitsConfig()

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
        for entry in self.early_exits.exits:
            layer = entry[0]
            if not 0 <= layer < self.num_layers_transformer:
                raise ValueError(
                    f"Error: Not possible to assign Exit after layer '{layer}'. Index be between 0 and {self.num_layers_transformer-1}"
                )


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
        if "early_exits" in model_dict:
            early_exits_dict = model_dict.pop("early_exits")
            # Create the EarlyExitsConfig and assign it back to model_dict
            model_dict["early_exits"] = EarlyExitsConfig(**early_exits_dict)

        # Create ModelConfig with all parameters
        return ModelConfig(**model_dict)

    except KeyError as e:
        raise KeyError(f"Missing required configuration key: {e}")


# Example usage:
"""
# Minimal configuration (only required parameters)
minimal_config = ModelConfig(
    channels_num=3,
    image_size=224,
    num_classes=1000
)

# Custom configuration
custom_config = ModelConfig(
    channels_num=3,
    image_size=224,
    num_classes=1000,
    embed_depth=512,  # Override default
    num_attn_heads=8,  # Override default
    early_exits=EarlyExitsConfig(
        exits=[(4, 'conv1_1'), (5, 'conv1_1')]  # Custom exits
    )
)
"""
