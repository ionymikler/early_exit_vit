import torch
import json
from typing import Dict, Tuple, List, Any


def load_pretrained_weights(
    model: torch.nn.Module,
    pretrained_path: str,
    config_pretrained_path: str,
    verbose: bool = False,
) -> Tuple[torch.nn.Module, Dict[str, List[str]]]:
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
    hw_keys = _create_highway_mapping(model, saved_model_state_dict, config_pretrained)
    keys_map.update(hw_keys)

    # Create the weight mapping dictionary
    lgvit_map = _make_weight_mapping(saved_model_state_dict, keys_map)

    # Load weights into model
    incompatible_keys = model.load_state_dict(lgvit_map, strict=False)

    if verbose:
        _print_incompatible_keys(incompatible_keys)

    return model, incompatible_keys


def _create_base_architecture_mapping(
    config_pretrained: Dict[str, Any],
) -> Dict[str, Any]:
    """Create the base architecture key mapping."""
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
        keys_map[f"transformer.layers.{i}.norm_2.weight"] = (
            f"deit.encoder.layer.{i}.layernorm_after.weight"
        )
        keys_map[f"transformer.layers.{i}.norm_2.bias"] = (
            f"deit.encoder.layer.{i}.layernorm_after.bias"
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

        # MLP
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
    model: torch.nn.Module,
    saved_model_state_dict: Dict[str, torch.Tensor],
    config_pretrained: Dict[str, Any],
) -> Dict[str, str]:
    """Create the highway layers key mapping."""
    hw_keys = {}

    # Parse exit positions
    s = config_pretrained["position_exits"].strip("][,")
    lgvit_exit_positions = list(map(int, s.split(",")))

    # Get EEVIT exit positions from model config
    model_config = model.config if hasattr(model, "config") else model.module.config
    eevit_exit_positions = model_config.early_exit_config.exit_list

    for idx in range(len(lgvit_exit_positions)):
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

            if eevit_key in model.state_dict() and lgvit_key in saved_model_state_dict:
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


def _print_incompatible_keys(incompatible_keys: Dict[str, List[str]]) -> None:
    """Print information about incompatible keys."""
    print(
        f"\nUnexpected Keys (Keys in LGVIT but not in EEVIT): Total: {len(incompatible_keys.unexpected_keys)}"
    )
    for uk in incompatible_keys.unexpected_keys:
        print(uk)

    print(
        f"\nMissing Keys (Keys in EEVIT but not in LGVIT) Total: {len(incompatible_keys.missing_keys)}"
    )
    for mk in incompatible_keys.missing_keys:
        print(mk)
