#!/usr/bin/env python
# EEVIT Model Parameter Analysis
# This script analyzes the parameter distribution of the EEVIT model

import json
import os
from collections import defaultdict
from datetime import datetime

from utils.arg_utils import get_config_dict, parse_config_dict
from utils.model_utils import get_model
from utils.logging_utils import get_logger_ready, green_txt, yellow_txt

# Initialize logger for external modules only
logger = get_logger_ready("param_analysis")


def count_parameters(model):
    """Count the total number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def analyze_model_parameters(
    model, show_details=True, export_json=False, json_path=None
):
    """Analyze the parameter distribution in the model by component"""
    # Create dictionaries to store parameters by component and layer
    param_counts = defaultdict(int)
    layer_param_counts = defaultdict(lambda: defaultdict(int))
    total_params = 0

    # Analyze parameters by component
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params

            # Parse layer index if in transformer layers
            layer_idx = None
            if "transformer.layers." in name:
                parts = name.split(".")
                if (
                    len(parts) >= 3
                    and parts[0] == "transformer"
                    and parts[1] == "layers"
                ):
                    layer_idx = int(parts[2])

            # Determine component
            if "patch_embedding" in name:
                param_counts["patch_embedding"] += num_params
            elif "transformer.layers" in name:
                if ".highway" in name:
                    param_counts["highway_exits"] += num_params
                    if layer_idx is not None:
                        layer_param_counts[layer_idx]["highway"] += num_params
                elif ".norm_1" in name:
                    param_counts["transformer_norm"] += num_params
                    if layer_idx is not None:
                        layer_param_counts[layer_idx]["norm"] += num_params
                elif ".W_QKV" in name:
                    param_counts["transformer_qkv"] += num_params
                    if layer_idx is not None:
                        layer_param_counts[layer_idx]["qkv"] += num_params
                elif ".attention_output" in name:
                    param_counts["transformer_attention_out"] += num_params
                    if layer_idx is not None:
                        layer_param_counts[layer_idx]["attention_out"] += num_params
                elif ".mlps.norm_2" in name:
                    param_counts["transformer_mlp_norm"] += num_params
                    if layer_idx is not None:
                        layer_param_counts[layer_idx]["mlp_norm"] += num_params
                elif ".mlps.mlp_intermediate" in name:
                    param_counts["transformer_mlp_intermediate"] += num_params
                    if layer_idx is not None:
                        layer_param_counts[layer_idx]["mlp_intermediate"] += num_params
                elif ".mlps.mlp_output" in name:
                    param_counts["transformer_mlp_output"] += num_params
                    if layer_idx is not None:
                        layer_param_counts[layer_idx]["mlp_output"] += num_params
                else:
                    param_counts["transformer_other"] += num_params
                    if layer_idx is not None:
                        layer_param_counts[layer_idx]["other"] += num_params
            elif "transformer.norm_post_layers" in name:
                param_counts["transformer_final_norm"] += num_params
            elif "last_exit" in name:
                param_counts["classifier"] += num_params
            else:
                param_counts["other"] += num_params

    # Calculate parameters per attention block
    num_layers = len(model.transformer.layers)
    attention_layers = [
        layer
        for layer in model.transformer.layers
        if "Attention" in layer.__class__.__name__
    ]
    num_attention_layers = len(attention_layers)

    # Calculate total parameters for the main components
    transformer_attention_total = (
        param_counts["transformer_norm"]
        + param_counts["transformer_qkv"]
        + param_counts["transformer_attention_out"]
    )

    transformer_mlp_total = (
        param_counts["transformer_mlp_norm"]
        + param_counts["transformer_mlp_intermediate"]
        + param_counts["transformer_mlp_output"]
    )

    transformer_params = (
        transformer_attention_total
        + transformer_mlp_total
        + param_counts["transformer_other"]
        + param_counts["transformer_final_norm"]
    )

    patch_emb = param_counts["patch_embedding"]
    highway_params = param_counts["highway_exits"]
    classifier_params = param_counts["classifier"]

    # Print results
    print(green_txt("=" * 60))
    print(green_txt("EEVIT Model Parameter Analysis"))
    print(green_txt("=" * 60))

    print(f"Total number of trainable parameters: {total_params:,}")
    print(f"Number of transformer layers: {num_layers}")
    print(f"Number of attention layers: {num_attention_layers}")

    # Summarize main components first
    print(yellow_txt("\nMain Component Summary:"))
    print(
        f"- Patch Embedding: {patch_emb:,} parameters ({patch_emb/total_params*100:.2f}%)"
    )
    print(
        f"- Transformer (all layers): {transformer_params:,} parameters ({transformer_params/total_params*100:.2f}%)"
    )
    print(f"  - Attention components: {transformer_attention_total:,} parameters")
    print(f"  - MLP components: {transformer_mlp_total:,} parameters")
    print(
        f"- Highway Exits: {highway_params:,} parameters ({highway_params/total_params*100:.2f}%)"
    )
    print(
        f"- Classifier: {classifier_params:,} parameters ({classifier_params/total_params*100:.2f}%)"
    )

    # Print per-layer analysis
    if show_details:
        print(green_txt("\n" + "=" * 60))
        print(green_txt("Layer-by-Layer Parameter Analysis"))
        print(green_txt("=" * 60))

        # For each layer
        for layer_idx in sorted(layer_param_counts.keys()):
            layer_counts = layer_param_counts[layer_idx]
            total_layer_params = sum(layer_counts.values())

            # Calculate attention and MLP parameters for this layer
            attention_params = (
                layer_counts["norm"]
                + layer_counts["qkv"]
                + layer_counts["attention_out"]
            )

            mlp_params = (
                layer_counts["mlp_norm"]
                + layer_counts["mlp_intermediate"]
                + layer_counts["mlp_output"]
            )

            highway_params = layer_counts["highway"]

            print(yellow_txt(f"\nLayer {layer_idx}:"))
            print(f"- Total parameters: {total_layer_params:,}")

            # Attention components with percentages
            attn_pct = (attention_params / total_layer_params) * 100
            print(
                f"- Attention parameters: {attention_params:,} ({attn_pct:.2f}% of layer)"
            )

            # Show attention subcomponents with percentages
            if attention_params > 0:
                norm_pct = (layer_counts["norm"] / attention_params) * 100
                qkv_pct = (layer_counts["qkv"] / attention_params) * 100
                attn_out_pct = (layer_counts["attention_out"] / attention_params) * 100

                print(
                    f"  - Norm: {layer_counts['norm']:,} ({norm_pct:.2f}% of attention)"
                )
                print(f"  - QKV: {layer_counts['qkv']:,} ({qkv_pct:.2f}% of attention)")
                print(
                    f"  - Attention output: {layer_counts['attention_out']:,} ({attn_out_pct:.2f}% of attention)"
                )

            # MLP components with percentages
            mlp_pct = (mlp_params / total_layer_params) * 100
            print(f"- MLP parameters: {mlp_params:,} ({mlp_pct:.2f}% of layer)")

            # Show MLP subcomponents with percentages
            if mlp_params > 0:
                mlp_norm_pct = (layer_counts["mlp_norm"] / mlp_params) * 100
                mlp_inter_pct = (layer_counts["mlp_intermediate"] / mlp_params) * 100
                mlp_out_pct = (layer_counts["mlp_output"] / mlp_params) * 100

                print(
                    f"  - MLP norm: {layer_counts['mlp_norm']:,} ({mlp_norm_pct:.2f}% of MLP)"
                )
                print(
                    f"  - MLP intermediate: {layer_counts['mlp_intermediate']:,} ({mlp_inter_pct:.2f}% of MLP)"
                )
                print(
                    f"  - MLP output: {layer_counts['mlp_output']:,} ({mlp_out_pct:.2f}% of MLP)"
                )

            # Highway parameters with percentages
            if highway_params > 0:
                hw_pct = (highway_params / total_layer_params) * 100
                print(
                    f"- Highway parameters: {highway_params:,} ({hw_pct:.2f}% of layer)"
                )

    print(green_txt("=" * 60))

    # Create JSON data structure for export
    if export_json:
        analysis_data = {
            "total_parameters": total_params,
            "num_layers": num_layers,
            "num_attention_layers": num_attention_layers,
            "main_components": {
                "patch_embedding": {
                    "parameters": patch_emb,
                    "percentage": patch_emb / total_params * 100,
                },
                "transformer": {
                    "parameters": transformer_params,
                    "percentage": transformer_params / total_params * 100,
                    "attention_components": transformer_attention_total,
                    "mlp_components": transformer_mlp_total,
                },
                "highway_exits": {
                    "parameters": highway_params,
                    "percentage": highway_params / total_params * 100,
                },
                "classifier": {
                    "parameters": classifier_params,
                    "percentage": classifier_params / total_params * 100,
                },
            },
            "component_breakdown": {k: v for k, v in param_counts.items()},
            "layers": {},
        }

        # Add detailed layer information
        for layer_idx in sorted(layer_param_counts.keys()):
            layer_counts = layer_param_counts[layer_idx]
            total_layer_params = sum(layer_counts.values())

            # Calculate attention and MLP parameters for this layer
            attention_params = (
                layer_counts["norm"]
                + layer_counts["qkv"]
                + layer_counts["attention_out"]
            )

            mlp_params = (
                layer_counts["mlp_norm"]
                + layer_counts["mlp_intermediate"]
                + layer_counts["mlp_output"]
            )

            highway_params = layer_counts["highway"]

            # Add to JSON data
            analysis_data["layers"][str(layer_idx)] = {
                "total_parameters": total_layer_params,
                "attention": {
                    "total": attention_params,
                    "percentage_of_layer": (attention_params / total_layer_params * 100)
                    if total_layer_params > 0
                    else 0,
                    "components": {
                        "norm": {
                            "parameters": layer_counts["norm"],
                            "percentage_of_attention": (
                                layer_counts["norm"] / attention_params * 100
                            )
                            if attention_params > 0
                            else 0,
                        },
                        "qkv": {
                            "parameters": layer_counts["qkv"],
                            "percentage_of_attention": (
                                layer_counts["qkv"] / attention_params * 100
                            )
                            if attention_params > 0
                            else 0,
                        },
                        "attention_output": {
                            "parameters": layer_counts["attention_out"],
                            "percentage_of_attention": (
                                layer_counts["attention_out"] / attention_params * 100
                            )
                            if attention_params > 0
                            else 0,
                        },
                    },
                },
                "mlp": {
                    "total": mlp_params,
                    "percentage_of_layer": (mlp_params / total_layer_params * 100)
                    if total_layer_params > 0
                    else 0,
                    "components": {
                        "norm": {
                            "parameters": layer_counts["mlp_norm"],
                            "percentage_of_mlp": (
                                layer_counts["mlp_norm"] / mlp_params * 100
                            )
                            if mlp_params > 0
                            else 0,
                        },
                        "intermediate": {
                            "parameters": layer_counts["mlp_intermediate"],
                            "percentage_of_mlp": (
                                layer_counts["mlp_intermediate"] / mlp_params * 100
                            )
                            if mlp_params > 0
                            else 0,
                        },
                        "output": {
                            "parameters": layer_counts["mlp_output"],
                            "percentage_of_mlp": (
                                layer_counts["mlp_output"] / mlp_params * 100
                            )
                            if mlp_params > 0
                            else 0,
                        },
                    },
                },
            }

            if highway_params > 0:
                analysis_data["layers"][str(layer_idx)]["highway"] = {
                    "parameters": highway_params,
                    "percentage_of_layer": (highway_params / total_layer_params * 100)
                    if total_layer_params > 0
                    else 0,
                }

        # Export to JSON file
        if json_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_path = f"eevit_param_analysis_{timestamp}.json"

        with open(json_path, "w") as f:
            json.dump(analysis_data, f, indent=2)

        print(f"Parameter analysis exported to: {os.path.abspath(json_path)}")

    return total_params, param_counts


def main():
    # Load configuration
    config_path = "./config/run_args.yaml"
    config = get_config_dict(config_path)
    model_config = parse_config_dict(config["model"].copy())

    # Create model
    print("Creating EEVIT model...")
    model = get_model(model_config, verbose=False)

    # Analyze parameters and export to JSON
    json_path = "eevit_param_analysis.json"
    total_params, param_counts = analyze_model_parameters(
        model, show_details=True, export_json=True, json_path=json_path
    )

    # Print warning if too many parameters for efficient ONNX export
    if total_params > 100_000_000:  # 100M parameter threshold
        print(
            yellow_txt(
                "Warning: Model has more than 100M parameters which might lead to slow ONNX export."
            )
        )


if __name__ == "__main__":
    main()
