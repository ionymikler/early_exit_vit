import torch
import json
from typing import Dict, Tuple, List, Any

from .arg_utils import ModelConfig
from .logging_utils import get_logger_ready
from eevit.eevit import EEVIT  # noqa F401

# Constants
PT_WEIGHTS_PATH = (
    "/home/iony/DTU/f24/thesis/code/lgvit/LGViT-ViT-Cifar100/pytorch_model.bin"
)
PT_CONFIG_PATH = "/home/iony/DTU/f24/thesis/code/lgvit/LGViT-ViT-Cifar100/config.json"

logger = get_logger_ready(__name__)


def get_model(model_config: ModelConfig, verbose=True) -> EEVIT:
    return EEVIT(config=model_config, verbose=verbose)


def load_pretrained_weights(
    model: EEVIT,
    model_config: ModelConfig,
    pretrained_path: str,
    config_pretrained_path: str,
    verbose: bool = False,
) -> Tuple[EEVIT, Dict[str, List[str]]]:
    """
    Load pretrained weights from LGVIT model into EEVIT model.

    Args:
        model: The EEVIT model to load weights into
        pretrained_path: Path to the pretrained LGVIT model weights
        config_path: Path to the LGVIT config file
        verbose: Whether to print debug information

    Returns:
        Tuple containing:
        - The model with loaded weights
        - Dictionary with incompatible keys info
    """
    # Load pretrained model and config
    saved_model_state_dict = torch.load(
        pretrained_path, map_location="cpu", weights_only=True
    )
    with open(config_pretrained_path, "r") as f:
        config_pretrained = json.load(f)

    # Create base architecture mapping
    keys_map = _create_base_architecture_mapping(config_pretrained)

    # Create highway mapping
    # Get EEVIT exit positions from model config
    eevit_exit_positions = model_config.early_exit_config.exit_list
    hw_keys = _create_highway_mapping(
        eevit_exit_positions, saved_model_state_dict, config_pretrained
    )
    keys_map.update(hw_keys)

    # Create the weight mapping dictionary
    lgvit_map = _make_weight_mapping(saved_model_state_dict, keys_map)

    # Load weights into model
    incompatible_keys = model.load_state_dict(lgvit_map, strict=False)

    return model, incompatible_keys


def _create_base_architecture_mapping(
    config_pretrained: Dict[str, Any],
) -> Dict[str, Any]:
    """Create the base architecture key mapping.
    The mapp is from LGVIT to EEVIT."""
    keys_map = {
        # Patch embeddings
        "patch_embedding.pos_embedding": "deit.embeddings.position_embeddings",
        "patch_embedding.cls_token": "deit.embeddings.cls_token",
        "patch_embedding.projection.weight": "deit.embeddings.patch_embeddings.projection.weight",
        "patch_embedding.projection.bias": "deit.embeddings.patch_embeddings.projection.bias",
        # Transformer post-layers norm
        "transformer.norm_post_layers.weight": "deit.layernorm.weight",
        "transformer.norm_post_layers.bias": "deit.layernorm.bias",
        # Classifier
        "last_exit.weight": "classifier.weight",
        "last_exit.bias": "classifier.bias",
    }

    # Add transformer layers mapping
    for i in range(config_pretrained["num_hidden_layers"]):
        # Layer norms
        keys_map[f"transformer.layers.{i}.norm_1.weight"] = (
            f"deit.encoder.layer.{i}.layernorm_before.weight"
        )
        keys_map[f"transformer.layers.{i}.norm_1.bias"] = (
            f"deit.encoder.layer.{i}.layernorm_before.bias"
        )

        # Attention
        keys_map[f"transformer.layers.{i}.W_QKV.weight"] = (
            f"deit.encoder.layer.{i}.attention.attention.query.weight",
            f"deit.encoder.layer.{i}.attention.attention.key.weight",
            f"deit.encoder.layer.{i}.attention.attention.value.weight",
        )
        keys_map[f"transformer.layers.{i}.W_QKV.bias"] = (
            f"deit.encoder.layer.{i}.attention.attention.query.bias",
            f"deit.encoder.layer.{i}.attention.attention.key.bias",
            f"deit.encoder.layer.{i}.attention.attention.value.bias",
        )

        # Attention output
        keys_map[f"transformer.layers.{i}.attention_output.0.weight"] = (
            f"deit.encoder.layer.{i}.attention.output.dense.weight"
        )
        keys_map[f"transformer.layers.{i}.attention_output.0.bias"] = (
            f"deit.encoder.layer.{i}.attention.output.dense.bias"
        )

        # MLPs
        keys_map[f"transformer.layers.{i}.mlps.norm_2.weight"] = (
            f"deit.encoder.layer.{i}.layernorm_after.weight"
        )
        keys_map[f"transformer.layers.{i}.mlps.norm_2.bias"] = (
            f"deit.encoder.layer.{i}.layernorm_after.bias"
        )

        keys_map[f"transformer.layers.{i}.mlps.mlp_intermediate.0.weight"] = (
            f"deit.encoder.layer.{i}.intermediate.dense.weight"
        )
        keys_map[f"transformer.layers.{i}.mlps.mlp_intermediate.0.bias"] = (
            f"deit.encoder.layer.{i}.intermediate.dense.bias"
        )
        keys_map[f"transformer.layers.{i}.mlps.mlp_output.0.weight"] = (
            f"deit.encoder.layer.{i}.output.dense.weight"
        )
        keys_map[f"transformer.layers.{i}.mlps.mlp_output.0.bias"] = (
            f"deit.encoder.layer.{i}.output.dense.bias"
        )

    return keys_map


def _create_highway_mapping(
    eevit_exit_positions: List[int],
    saved_model_state_dict: Dict[str, torch.Tensor],
    config_pretrained: Dict[str, Any],
) -> Dict[str, str]:
    """Create the highway layers key mapping."""
    hw_keys = {}

    # Parse exit positions
    s = config_pretrained["position_exits"].strip("][,")
    lgvit_exit_positions = list(map(int, s.split(",")))

    for idx in range(len(lgvit_exit_positions)):
        if idx > len(eevit_exit_positions) - 1:
            continue
        eevit_idx = eevit_exit_positions[idx]
        # lgvit_idx = lgvit_exit_positions[idx]

        eevit_prefix = f"transformer.layers.{eevit_idx}.highway"
        lgvit_hw_prefix = f"highway.{idx}"

        # Map highway keys
        for lgvit_key in saved_model_state_dict:
            if "highway" not in lgvit_key or lgvit_hw_prefix not in lgvit_key:
                continue

            if "classifier" in lgvit_key:
                length = lgvit_key.find("classifier.") + len("classifier.")
                eevit_key = f"{eevit_prefix}.classifier.classifier.{lgvit_key[length:]}"
            else:
                length = lgvit_key.find("mlp.") + len("mlp.")
                eevit_key = f"{eevit_prefix}.highway_head.{lgvit_key[length:]}"

            # if eevit_key in model.state_dict() and lgvit_key in saved_model_state_dict:
            hw_keys[eevit_key] = lgvit_key

    return hw_keys


def _make_weight_mapping(
    saved_model_state_dict: Dict[str, torch.Tensor], keys_map: Dict[str, Any]
) -> Dict[str, torch.Tensor]:
    """Create the final weight mapping dictionary."""
    values_dict = {}
    for k, v in keys_map.items():
        if "W_QKV" in k:
            dest_v = torch.cat([saved_model_state_dict[weight] for weight in v], dim=0)
        else:
            dest_v = saved_model_state_dict[v]
        values_dict[k] = dest_v
    return values_dict


def _print_incompatible_keys(
    incompatible_keys: Dict[str, List[str]], verbose: bool = False
) -> None:
    """Print information about incompatible keys."""
    logger.info(f"Unexpected Keys: {len(incompatible_keys.unexpected_keys)}")
    logger.info(f"Missing Keys: {len(incompatible_keys.missing_keys)}")
    if (
        len(incompatible_keys.unexpected_keys) > 0
        or len(incompatible_keys.missing_keys) > 0
    ):
        print(
            f"Unexpected Keys (Keys in LGVIT but not in EEVIT): Total: {len(incompatible_keys.unexpected_keys)}"
        )
        for uk in incompatible_keys.unexpected_keys:
            print(uk)

        print(
            f"Missing Keys (Keys in EEVIT but not in LGVIT) Total: {len(incompatible_keys.missing_keys)}"
        )
        for mk in incompatible_keys.missing_keys:
            print(mk)
        else:
            logger.info("✅ No incompatible keys found.")


def setup_model_for_evaluation(
    model_config: ModelConfig,
    pretrained_weights_path: str = PT_WEIGHTS_PATH,
    pretrained_config_path: str = PT_CONFIG_PATH,
    device: str = "cpu",
    verbose: bool = False,
) -> EEVIT:
    """
    Complete model setup function that:
    1. Creates the EEVIT model
    2. Loads pretrained weights
    3. Moves model to device
    4. Sets model to eval mode

    Args:
        config_path: Path to EEVIT config YAML
        pretrained_weights_path: Path to pretrained LGVIT weights
        pretrained_config_path: Path to pretrained LGVIT config
        device: Device to put model on
        verbose: Whether to print details

    Returns:
        EEVIT model ready for evaluation
    """
    # Create model
    model = get_model(model_config, verbose)

    # Load pretrained weights
    logger.info("Loading pretrained weights...")

    model, incompatible_keys = load_pretrained_weights(
        model=model,
        model_config=model_config,
        pretrained_path=pretrained_weights_path,
        config_pretrained_path=pretrained_config_path,
        verbose=verbose,
    )

    _print_incompatible_keys(incompatible_keys, verbose)

    # Prepare model for evaluation
    model = model.to(device)
    model.eval()

    return model


def format_model_name(model_name: str, suffix: str = None) -> str:
    return f"{model_name}_{suffix}" if suffix else model_name
