#!/usr/bin/env python
# Enhanced EEVIT Model Parameter Analysis
# This script analyzes and visualizes the parameter distribution of the EEVIT model

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from pathlib import Path

from utils.arg_utils import get_config_dict, parse_config_dict
from utils.model_utils import get_model
from utils.logging_utils import get_logger_ready, green_txt, yellow_txt

# Initialize logger for external modules only
logger = get_logger_ready("param_analysis")


def _print_section_header(title):
    """Print a section header with the given title"""
    print(green_txt("=" * 60))
    print(green_txt(title))
    print(green_txt("=" * 60))


def count_parameters(model):
    """Count the total number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def analyze_model_parameters(model, show_details=False):
    """
    Analyze the parameter distribution in the model by component
    Returns a data structure with all analysis results
    """
    # Data structure to hold results
    analysis_data = {
        "total_parameters": 0,
        "num_layers": 0,
        "num_attention_layers": 0,
        "main_components": {},
        "layers": {},
        "has_highway_params": False,
    }

    # Parameter counters
    param_counts = defaultdict(int)
    layer_param_counts = defaultdict(lambda: defaultdict(int))
    total_params = 0

    # Analyze parameters by component
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        num_params = param.numel()
        total_params += num_params

        # Check for highway parameters
        if ".highway" in name:
            analysis_data["has_highway_params"] = True
            logger.debug(
                f"Found highway parameter: {name} with {num_params} parameters"
            )

        # Parse layer index if in transformer layers
        layer_idx = None
        if "transformer.layers." in name:
            parts = name.split(".")
            if len(parts) >= 3 and parts[0] == "transformer" and parts[1] == "layers":
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

    # Calculate derived metrics
    analysis_data["total_parameters"] = total_params
    analysis_data["num_layers"] = len(model.transformer.layers)
    analysis_data["num_attention_layers"] = len(
        [
            layer
            for layer in model.transformer.layers
            if "Attention" in layer.__class__.__name__
        ]
    )

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

    transformer_other_total = (
        param_counts["transformer_other"] + param_counts["transformer_final_norm"]
    )

    transformer_total = (
        transformer_attention_total + transformer_mlp_total + transformer_other_total
    )

    patch_emb = param_counts["patch_embedding"]
    highway_params = param_counts["highway_exits"]
    classifier_params = param_counts["classifier"]

    # Build main_components section
    analysis_data["main_components"] = {
        "patch_embedding": {
            "parameters": patch_emb,
            "percentage": patch_emb / total_params * 100,
        },
        "transformer": {
            "parameters": transformer_total,
            "percentage": transformer_total / total_params * 100,
            "components": {
                "attention": {
                    "parameters": transformer_attention_total,
                    "percentage": transformer_attention_total / transformer_total * 100,
                },
                "mlp": {
                    "parameters": transformer_mlp_total,
                    "percentage": transformer_mlp_total / transformer_total * 100,
                },
                "other": {
                    "parameters": transformer_other_total,
                    "percentage": transformer_other_total / transformer_total * 100,
                },
            },
        },
        "highway_exits": {
            "parameters": highway_params,
            "percentage": highway_params / total_params * 100,
        },
        "classifier": {
            "parameters": classifier_params,
            "percentage": classifier_params / total_params * 100,
        },
    }

    # Build layers section
    for layer_idx in sorted(layer_param_counts.keys()):
        layer_counts = layer_param_counts[layer_idx]
        total_layer_params = sum(layer_counts.values())

        # Calculate component parameters for this layer
        attention_params = (
            layer_counts["norm"] + layer_counts["qkv"] + layer_counts["attention_out"]
        )

        mlp_params = (
            layer_counts["mlp_norm"]
            + layer_counts["mlp_intermediate"]
            + layer_counts["mlp_output"]
        )

        highway_params = layer_counts["highway"]

        # Add to data structure
        layer_data = {
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
            layer_data["highway"] = {
                "parameters": highway_params,
                "percentage_of_layer": (highway_params / total_layer_params * 100)
                if total_layer_params > 0
                else 0,
            }

        analysis_data["layers"][str(layer_idx)] = layer_data

    # Print analysis results
    _print_section_header("EEVIT Model Parameter Analysis")

    print(
        f"Total number of trainable parameters: {total_params:,}\n"
        f"Number of transformer layers: {analysis_data['num_layers']}\n"
    )

    # Print main component summary
    print(yellow_txt("\nMain Component Summary:"))
    print(
        f"- Patch Embedding: {patch_emb:,} parameters ({analysis_data['main_components']['patch_embedding']['percentage']:.2f}%)\n"
        f"- Transformer (all layers): {transformer_total:,} parameters ({analysis_data['main_components']['transformer']['percentage']:.2f}%)\n"
        f"- Highway Exits: {highway_params:,} parameters ({analysis_data['main_components']['highway_exits']['percentage']:.2f}%)\n"
        f"- Last Classifier: {classifier_params:,} parameters ({analysis_data['main_components']['classifier']['percentage']:.2f}%)"
    )

    # Debug for highway detection
    if not analysis_data["has_highway_params"]:
        print(yellow_txt("\nWARNING: No highway parameters detected in the model!"))
        print("This might be because:")
        print("1. The model doesn't have early exits configured")
        print("2. The parameter naming pattern is different than expected")
        print("3. The early exits use different parameter structures")

    # Print per-layer analysis if requested
    if show_details:
        _print_section_header("Layer-by-Layer Analysis")

        for layer_idx in sorted(analysis_data["layers"].keys(), key=int):
            layer_data = analysis_data["layers"][layer_idx]

            print(yellow_txt(f"\nLayer {layer_idx}:"))
            print(f"- Total parameters: {layer_data['total_parameters']:,}")

            # Attention components
            attention_data = layer_data["attention"]
            print(
                f"- Attention parameters: {attention_data['total']:,} ({attention_data['percentage_of_layer']:.2f}% of layer)"
            )

            # MLP components
            mlp_data = layer_data["mlp"]
            print(
                f"- MLP parameters: {mlp_data['total']:,} ({mlp_data['percentage_of_layer']:.2f}% of layer)"
            )

            # Highway parameters if present
            if "highway" in layer_data:
                highway_data = layer_data["highway"]
                print(
                    f"- Highway parameters: {highway_data['parameters']:,} ({highway_data['percentage_of_layer']:.2f}% of layer)"
                )

    print(green_txt("=" * 60))

    return analysis_data


def plot_layer_detail_columns(analysis_data, model_config, layer_indices=[3, 7]):
    """Plot stacked column charts for multiple layers side by side,
    with the three main components (Attention, MLP, Highway) stacked together in each column,
    and showing the intermediate head type in the x-axis labels"""
    layers_data = analysis_data["layers"]

    # Get early exit types from model config
    ee_config = model_config.early_exit_config
    exit_types = {}
    for exit_entry in ee_config.exits:
        position = exit_entry[0]
        exit_type = exit_entry[1]
        exit_types[position] = exit_type

    # Map exit types to more descriptive names
    head_type_names = {
        "conv1_1": "Local Perception",
        "conv2_1": "Local Perception",
        "attention": "Global Aggregation",
        "dummy_mlp": "Dummy MLP",
    }

    # Create a single figure with one subplot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define colors for component types
    component_colors = {
        "Attention": "#4285F4",  # Blue for attention
        "MLP": "#EA4335",  # Red for MLP
        "Highway": "#FBBC05",  # Yellow for highway
    }

    # Define x positions for bars
    x_positions = np.arange(len(layer_indices))
    bar_width = 0.6

    # Store legend handles and labels
    legend_handles = []
    legend_labels = []
    used_labels = set()

    # Store x-tick labels
    x_tick_labels = []

    # Process each requested layer
    for i, layer_idx in enumerate(layer_indices):
        layer_key = str(layer_idx)

        # Create x-tick label including head type
        head_type = exit_types.get(layer_idx, "Unknown")
        head_type_desc = head_type_names.get(head_type, head_type)
        x_tick_label = f"Layer {layer_idx}\n({head_type_desc})"
        x_tick_labels.append(x_tick_label)

        if layer_key not in layers_data:
            ax.text(
                x_positions[i],
                0,
                f"Layer {layer_idx} data not available",
                ha="center",
                va="bottom",
                fontsize=10,
            )
            continue

        layer_data = layers_data[layer_key]

        # Collect the three main components
        main_components = [
            {
                "name": "Attention",
                "parameters": layer_data["attention"]["total"],
                "percentage": layer_data["attention"]["percentage_of_layer"],
            },
            {
                "name": "MLP",
                "parameters": layer_data["mlp"]["total"],
                "percentage": layer_data["mlp"]["percentage_of_layer"],
            },
        ]

        # Add Highway if it exists
        if "highway" in layer_data:
            main_components.append(
                {
                    "name": "Highway",
                    "parameters": layer_data["highway"]["parameters"],
                    "percentage": layer_data["highway"]["percentage_of_layer"],
                }
            )

        # Extract data for plotting
        component_names = [comp["name"] for comp in main_components]
        param_values = [comp["parameters"] for comp in main_components]
        percentages = [comp["percentage"] for comp in main_components]
        colors = [component_colors[comp["name"]] for comp in main_components]

        # Create stacked bar for this layer
        bottom = 0
        for j, (name, value, pct, color) in enumerate(
            zip(component_names, param_values, percentages, colors)
        ):
            # Only add to legend if this component name hasn't been used yet
            if name not in used_labels:
                bar = ax.bar(
                    x_positions[i],  # x-position
                    value,  # height
                    bottom=bottom,  # bottom position for stacking
                    width=bar_width,  # bar width
                    color=color,  # color based on component type
                    label=name,  # for legend
                )
                used_labels.add(name)
                legend_handles.append(bar)
                legend_labels.append(name)
            else:
                bar = ax.bar(
                    x_positions[i],  # x-position
                    value,  # height
                    bottom=bottom,  # bottom position for stacking
                    width=bar_width,  # bar width
                    color=color,  # color based on component type
                )

            # Add parameter count and percentage label in the middle of each segment
            # Only add if segment is large enough
            if value > layer_data["total_parameters"] * 0.05:
                ax.text(
                    x_positions[i],  # x-position (bar center)
                    bottom + value / 2,  # y-position (segment center)
                    f"{name}\n{value:,}\n({pct:.1f}%)",  # Component name, count, and percentage
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                    fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2),
                )

            bottom += value

        # Add total count on top of each stack
        ax.text(
            x_positions[i],  # x-position (bar center)
            bottom * 1.02,  # y-position (just above the stack)
            f"Total: {layer_data['total_parameters']:,}",  # Total layer parameters
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Set labels and title
    ax.set_title("Parameter Distribution by Layer and Head Type", fontsize=16)
    ax.set_ylabel("Number of Parameters", fontsize=14)

    # Set x-ticks with layer numbers and head types
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_tick_labels)

    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Format y-axis with commas for thousands
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))

    # Add a legend
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(used_labels))

    plt.tight_layout()
    return fig


def plot_main_components(analysis_data):
    """Plot the parameter distribution for the main components of EEVIT as a stacked column"""
    main_components = analysis_data["main_components"]

    # Prepare data for plotting
    component_names = []
    parameter_counts = []

    # Use specific order for components
    components_data = [
        ("Patch Embedding", main_components["patch_embedding"]["parameters"]),
        ("Transformer", main_components["transformer"]["parameters"]),
        ("Highway Exits", main_components["highway_exits"]["parameters"]),
        ("Last Classifier", main_components["classifier"]["parameters"]),
    ]

    # Extract names and values
    component_names = [item[0] for item in components_data]
    parameter_counts = [item[1] for item in components_data]

    # Create figure
    plt.figure(figsize=(10, 8))

    # Create color map
    colors = ["#4285F4", "#EA4335", "#FBBC05", "#34A853"]

    # Create stacked bar plot (single column with segments)
    bottom = 0
    bars = []
    for i, count in enumerate(parameter_counts):
        bar = plt.bar(
            ["EEVIT Model"],  # Single column
            [count],
            bottom=bottom,
            color=colors[i],
            label=component_names[i],
        )
        bottom += count
        bars.append(bar)

    # Add labels inside segments
    total_params = analysis_data["total_parameters"]
    bottom = 0
    for i, count in enumerate(parameter_counts):
        percentage = (count / total_params) * 100
        # Only add labels if the segment is large enough
        if percentage > 1.0:
            plt.text(
                0,  # x position (centered on the single column)
                bottom + count / 2,  # y position (middle of segment)
                f"{component_names[i]}\n{count:,} ({percentage:.1f}%)",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="black",  # Changed from white to black
                bbox=dict(
                    facecolor="white",
                    alpha=0.7,
                    boxstyle="round,pad=0.3",
                    edgecolor="none",
                ),
            )
        bottom += count

    # Add title and labels
    plt.title("EEVIT Parameter Distribution", fontsize=16)
    plt.ylabel("Number of Parameters", fontsize=14)
    plt.xticks([])  # Hide x-ticks since we have only one column
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(loc="upper right")

    # Format y-axis with commas for thousands
    plt.gca().yaxis.set_major_formatter(
        plt.matplotlib.ticker.StrMethodFormatter("{x:,.0f}")
    )

    plt.tight_layout()
    return plt.gcf()


def plot_layer_breakdown(analysis_data):
    """Plot the parameter distribution across the transformer layers as two vertically stacked plots"""
    layers_data = analysis_data["layers"]

    # Prepare data for plotting
    layer_indices = [int(idx) for idx in layers_data.keys()]
    layer_indices.sort()  # Ensure layers are in order

    attention_params = []
    mlp_params = []
    highway_params = []

    for idx in layer_indices:
        layer_data = layers_data[str(idx)]
        attention_params.append(layer_data["attention"]["total"])
        mlp_params.append(layer_data["mlp"]["total"])
        # Highway might not exist in all layers
        highway_params.append(layer_data.get("highway", {}).get("parameters", 0))

    # Split into two groups - first 6 and last 6 layers
    split_idx = 6
    first_half_indices = layer_indices[:split_idx]
    second_half_indices = layer_indices[split_idx:]

    # Create figure with two vertically-stacked subplots, but WITHOUT sharex
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=False)

    # Add main title to the figure
    fig.suptitle("EEVIT Parameter Distribution by Layer", fontsize=18, y=0.98)

    # Width for bars
    width = 0.25

    # First subplot - first 6 layers
    x1 = np.arange(len(first_half_indices))

    # Create bars for first half
    ax1.bar(
        x1 - width,
        attention_params[:split_idx],
        width,
        label="Attention",
        color="#4285F4",
    )
    ax1.bar(x1, mlp_params[:split_idx], width, label="MLP", color="#EA4335")
    ax1.bar(
        x1 + width, highway_params[:split_idx], width, label="Highway", color="#FBBC05"
    )

    # Add total count on top of each layer's combined bars
    for i, idx in enumerate(first_half_indices):
        total = attention_params[i] + mlp_params[i] + highway_params[i]
        max_height = max(attention_params[i], mlp_params[i], highway_params[i]) // 2
        ax1.text(
            i,
            max_height * 1.05,
            f"total params: {total:,}",
            ha="center",
            va="bottom",
            fontsize=10,
            bbox=dict(
                facecolor="white", alpha=0.7, boxstyle="round,pad=0.3", edgecolor="none"
            ),
        )

    ax1.set_title("Layers 0-5", fontsize=15)
    ax1.set_ylabel("Number of Parameters", fontsize=14)
    # Explicitly set x-ticks for top plot
    ax1.set_xticks(x1)
    ax1.set_xticklabels([str(idx) for idx in first_half_indices])
    ax1.grid(axis="y", linestyle="--", alpha=0.7)
    ax1.legend(loc="upper right")

    # Format y-axis with commas for thousands
    ax1.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))

    # Second subplot - last 6 layers
    x2 = np.arange(len(second_half_indices))

    # Create bars for second half
    ax2.bar(
        x2 - width,
        attention_params[split_idx:],
        width,
        label="Attention",
        color="#4285F4",
    )
    ax2.bar(x2, mlp_params[split_idx:], width, label="MLP", color="#EA4335")
    ax2.bar(
        x2 + width, highway_params[split_idx:], width, label="Highway", color="#FBBC05"
    )

    # Add total count on top of each layer's combined bars
    for i, idx in enumerate(second_half_indices):
        actual_idx = i + split_idx
        total = (
            attention_params[actual_idx]
            + mlp_params[actual_idx]
            + highway_params[actual_idx]
        )
        max_height = (
            max(
                attention_params[actual_idx],
                mlp_params[actual_idx],
                highway_params[actual_idx],
            )
            // 2
        )
        ax2.text(
            i,
            max_height * 1.05,
            f"total params: {total:,}",
            ha="center",
            va="bottom",
            fontsize=10,
            bbox=dict(
                facecolor="white", alpha=0.7, boxstyle="round,pad=0.3", edgecolor="none"
            ),
        )

    ax2.set_title("Layers 6-11", fontsize=15)
    ax2.set_xlabel("Layer Index", fontsize=14)
    ax2.set_ylabel("Number of Parameters", fontsize=14)
    # Explicitly set x-ticks for bottom plot
    ax2.set_xticks(x2)
    ax2.set_xticklabels([str(idx) for idx in second_half_indices])
    ax2.grid(axis="y", linestyle="--", alpha=0.7)
    ax2.legend(loc="upper right")

    # Format y-axis with commas for thousands
    ax2.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
    return fig


def main():
    # Load configuration
    config_path = Path("./config/run_args.yaml")
    config = get_config_dict(str(config_path))
    model_config = parse_config_dict(config["model"].copy())

    # Create model
    print("Creating EEVIT model...")
    model = get_model(model_config, verbose=False)

    # Analyze parameters
    analysis_data = analyze_model_parameters(model, show_details=False)

    # Create visualizations
    print("\nGenerating visualizations...")

    # Original visualizations
    main_components_fig = plot_main_components(analysis_data)  # Original column chart
    layers_fig = plot_layer_breakdown(analysis_data)  # Original layer breakdown

    # New layer detail visualization (passing model_config for exit type information)
    layer_detail_fig = plot_layer_detail_columns(analysis_data, model_config, [3, 7])

    # Show plots
    print("Displaying visualizations (close a figure to see the next one)...")

    # Show original plots
    plt.figure(main_components_fig.number)
    plt.show(block=False)

    plt.figure(layers_fig.number)
    plt.show(block=False)

    # Show new layer detail plot
    plt.figure(layer_detail_fig.number)
    plt.show(block=False)

    # Ask user if they want to save the results
    save_choice = input(
        "\nWould you like to save the results (JSON and figures)? (y/n): "
    ).lower()
    if save_choice.startswith("y"):
        save_results(analysis_data, main_components_fig, layers_fig, layer_detail_fig)

    # Keep figures open until user closes them
    plt.show()


def save_results(analysis_data, main_components_fig, layers_fig, layer_detail_fig=None):
    """Save the analysis results to JSON and figures to PNG files"""
    # Create paths using pathlib
    results_dir = Path("./results/model_analysis")
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "eevit_param_analysis.json"
    main_comp_path = results_dir / "eevit_main_components.png"
    layers_path = results_dir / "eevit_layer_breakdown.png"

    # Path for new layer detail plot
    layer_detail_path = results_dir / "eevit_layer_details.png"

    # Save JSON
    with json_path.open("w") as f:
        json.dump(analysis_data, f, indent=2)

    # Save figures
    main_components_fig.savefig(main_comp_path, dpi=300, bbox_inches="tight")
    layers_fig.savefig(layers_path, dpi=300, bbox_inches="tight")

    # Save layer detail figure if available
    if layer_detail_fig is not None:
        layer_detail_fig.savefig(layer_detail_path, dpi=300, bbox_inches="tight")

    print("Results saved to:")
    print(f"- {json_path.absolute()}")
    print(f"- {main_comp_path.absolute()}")
    print(f"- {layers_path.absolute()}")

    if layer_detail_fig is not None:
        print(f"- {layer_detail_path.absolute()}")


if __name__ == "__main__":
    main()
