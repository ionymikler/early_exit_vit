"""
Plotting utilities for visualization of model evaluation results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Tuple

from ..logging_utils import get_logger_ready

logger = get_logger_ready(__name__)

# Color schemes for different backends
COLOR_SCHEMES_BACKEND = {
    "teal": {
        "primary": "#57B4BA",  # Light Teal
        "secondary": "#B4EBE6",  # Lighter shade
        "scatter": "cool",  # Colormap for scatter plots
    },
    "onnx-cpu": {
        "primary": "#2878BD",  # Dark blue
        "secondary": "#8CC7FF",  # Lighter shade
        "scatter": "Blues",  # Colormap for scatter plots
    },
    "onnx-gpu": {
        "primary": "#4B0082",  # Indigo
        "secondary": "#8A2BE2",  # Blue Violet
        "scatter": "Purples",  # Colormap for scatter plots
    },
    "nvidia-onnx-cpu": {
        "primary": "#483D8B",  # Dark Slate Blue
        "secondary": "#66CDAA",  # Lighter shade
        "scatter": "Greens",  # Colormap for scatter plots
    },
    "nvidia-onnx-gpu": {
        "primary": "#22577A",
        "secondary": "#57CC99",
        "scatter": "Blues",  # Colormap for scatter plots
    },
    "pytorch": {
        "primary": "#8B0000",  # Dark Red
        "secondary": "#CD5C5C",  # Indian Red
        "scatter": "Reds",  # Colormap for scatter plots
    },
    "99Luftballons": {
        "primary": "#FF4500",  # Orange Red
        "secondary": "#FFA07A",  # Lighter shade
        "scatter": "autumn",  # Colormap for scatter plots
    },
    "custom": {
        "primary": "#57B4BA",  # Light Teal
        "secondary": "#B4EBE6",  # Lighter shade
        "scatter": "cool",  # Colormap for scatter plots
    },
}

HORIZONTAL_LINE_COLOR = "#8B0000"  # Dark Red

# Standardized font sizes for consistent visualizations
FONT_SIZE_FIGURE_TITLE = 20  # Main figure titles
FONT_SIZE_SUBPLOT_TITLE = 18  # Individual subplot titles
FONT_SIZE_AXIS_LABEL = 16  # Axis labels (x and y)
FONT_SIZE_TICK_LABEL = 14  # Tick labels on axes
FONT_SIZE_LEGEND = 14  # Legend text
FONT_SIZE_ANNOTATION = 14  # Text annotations (values on bars, etc)
FONT_SIZE_SMALL_ANNOTATION = 12  # Smaller annotations where space is limited


def save_figure(fig, results_dir: str, metric_name: str):
    """
    Save a figure to the results directory with standardized naming.

    Args:
        fig: The matplotlib figure to save
        results_dir: Path to the results directory
        metric_name: Name of the metric (will be used for filename)
    """
    save_path = f"{results_dir}/{metric_name}.png"
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    logger.info(f"Figure saved to: {save_path}")


def get_results_info(results_dir: str):
    """
    Extract model type and results identifier from the metadata YAML file in a results directory.

    Args:
        results_dir: Path to results directory

    Returns:
        Tuple of (model_type, results_identifier)
    """
    # Read the metadata.yaml file
    metadata_file = os.path.join(results_dir, "metadata.yaml")

    if not os.path.exists(metadata_file):
        logger.warning(f"Metadata file not found in directory: {metadata_file}")
        return "unknown_model", "unknown_results"

    try:
        import yaml

        with open(metadata_file, "r") as f:
            metadata = yaml.safe_load(f)

        # Extract model_type directly from metadata
        model_type = metadata["model_type"]

        # Get timestamp or suffix for the identifier
        if "args" in metadata and isinstance(metadata["args"], dict):
            identifier = metadata["args"].get("suffix")
        else:
            identifier = metadata.get("timestamp", "")

        return model_type, f"results_{identifier}"

    except Exception as e:
        logger.warning(f"Error reading metadata file: {e}")


def plot_metrics(metrics, results_dir: str, color_scheme_key="teal", top_n_classes=10):
    """
    Plot metrics for model evaluation, including both exit statistics and class statistics.

    Args:
        metrics: Dictionary containing evaluation metrics
        results_dir: Directory where results are stored
        color_scheme: Color scheme to use
        top_n_classes: Number of top classes to highlight in class-specific plots

    Returns:
        Tuple of figures: (exit_stats_figure, class_accuracy_stats_figure, class_speed_stats_figure)
    """
    # Get color scheme
    colors = COLOR_SCHEMES_BACKEND[color_scheme_key]

    # Extract model type and results identifier for title
    model_type, results_id = get_results_info(results_dir)

    title = f"{model_type} | {results_id}"
    logger.info(f"Using title for plots: {title}")

    # Create exit statistics visualization (now with column charts)
    exit_stats_fig = plot_exit_statistics(metrics, title, colors)

    # Create class statistics visualizations if available
    class_accuracy_fig = None
    class_speed_fig = None

    if "class_statistics" not in metrics and not metrics["class_statistics"]:
        raise ValueError("Class statistics not found in metrics")

    class_accuracy_fig = plot_class_statistics_unified(
        metrics, title, colors, sort_by="accuracy", top_n_classes=top_n_classes
    )

    class_speed_fig = plot_class_statistics_unified(
        metrics, title, colors, sort_by="speed", top_n_classes=top_n_classes
    )

    if "confusion_matrix" not in metrics:
        raise ValueError("confusion_matrix not found in metrics")

    confusion_fig = plot_confusion_matrix(
        metrics, title, normalize=True, top_n_classes=top_n_classes
    )
    return exit_stats_fig, class_accuracy_fig, class_speed_fig, confusion_fig


def plot_exit_statistics(metrics, title: str, colors):
    """
    Plot exit-related statistics including column charts for accuracy and inference time with standard deviation.

    Args:
        metrics: Dictionary containing evaluation metrics with distribution data
        title: Title for the plot
        colors: Color scheme to use

    Returns:
        Figure object
    """
    # Extract data for plotting
    exits = []
    counts = []
    accuracies = []
    accuracy_stds = []  # For standard error
    confidences = []
    inference_times = []
    inference_time_stds = []

    # Calculate overall average inference time across all exits
    total_inference_time = 0
    total_samples = 0

    # Sort exits by their position
    for exit_key, stats in sorted(
        metrics["exit_statistics"].items(),
        key=lambda x: float("inf")
        if x[0] == "final"
        else int(x[0].split("_")[1])
        if "_" in x[0]
        else float("inf"),
    ):
        exits.append(
            "Final" if exit_key == "final" else f"Exit {exit_key.split('_')[1]}"
        )
        counts.append(stats["count"])
        accuracies.append(stats["accuracy"])
        confidences.append(stats["average_confidence"])

        # Calculate standard error for accuracy using binomial statistics, since it's a proportion
        acc_values = np.array(stats["accuracy_values"])
        p = np.mean(acc_values)  # This is the accuracy as a proportion
        n = len(acc_values)
        std_error = np.sqrt((p * (1 - p)) / n) * 100
        accuracy_stds.append(std_error)

        # Get inference time data and standard deviations
        inference_times.append(stats["avg_inference_time_ms"])

        # Standard deviation of inference times
        time_values = np.array(stats["inference_time_values"])
        inference_time_stds.append(np.std(time_values))

        # Add to total for averaging
        total_inference_time += stats["avg_inference_time_ms"] * stats["count"]
        total_samples += stats["count"]

    # Calculate overall average inference time
    overall_avg_inference_time = (
        total_inference_time / total_samples if total_samples > 0 else 0
    )

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 12))  # Increased height from 10 to 12
    fig.suptitle(
        f"{title} - Exit Statistics Analysis", fontsize=FONT_SIZE_FIGURE_TITLE, y=0.98
    )

    # Increase spacing between subplots
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.4)  # Increased from 0.3 to 0.4

    # 1. Sample Distribution (Bar Chart)
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(exits, counts, color=colors["primary"])
    ax1.set_title(
        "Sample Distribution Across Exits", fontsize=FONT_SIZE_SUBPLOT_TITLE, pad=15
    )  # Added padding
    ax1.set_xlabel("Exit Point", fontsize=FONT_SIZE_AXIS_LABEL)
    ax1.set_ylabel("Number of Samples", fontsize=FONT_SIZE_AXIS_LABEL)
    ax1.tick_params(axis="x", rotation=45, labelsize=FONT_SIZE_TICK_LABEL)
    ax1.tick_params(axis="y", labelsize=FONT_SIZE_TICK_LABEL)

    # Add percentage labels
    total_samples = metrics["total_samples"]
    for bar in bars:
        height = bar.get_height()
        percentage = (height / total_samples) * 100
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}\n({percentage:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE_ANNOTATION,
            # bbox=dict(facecolor="white", alpha=0.8),
        )

    # 2. Accuracy Column Chart with Error Bars
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(
        exits,
        accuracies,
        color=colors["secondary"],
        yerr=accuracy_stds,
        capsize=5,
        # error_kw={"ecolor": "black", "capthick": 1.5},
    )

    # Add accuracy values on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + (accuracy_stds[i] if accuracy_stds[i] > 0 else 1.5),
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE_ANNOTATION,
            # fontweight="bold",
        )

    # Add overall accuracy line
    ax2.axhline(
        y=metrics["overall_accuracy"],
        color=HORIZONTAL_LINE_COLOR,
        linestyle="--",
        label=f"Overall Accuracy ({metrics['overall_accuracy']:.1f}%)",
    )

    ax2.set_title(
        "Accuracy by Exit Point", fontsize=FONT_SIZE_SUBPLOT_TITLE, pad=15
    )  # Added padding
    ax2.set_xlabel("Exit Point", fontsize=FONT_SIZE_AXIS_LABEL)
    ax2.set_ylabel("Accuracy (%)", fontsize=FONT_SIZE_AXIS_LABEL)
    ax2.set_ylim(0, max(accuracies) * 1.15)  # Give some headroom for error bars
    ax2.grid(axis="y", linestyle="--", alpha=0.7)
    ax2.legend(fontsize=FONT_SIZE_LEGEND)
    ax2.tick_params(axis="x", rotation=45, labelsize=FONT_SIZE_TICK_LABEL)
    ax2.tick_params(axis="y", labelsize=FONT_SIZE_TICK_LABEL)

    # 3. Inference Time Column Chart with Error Bars
    ax3 = fig.add_subplot(gs[1, 0])

    bars = ax3.bar(
        exits,
        inference_times,
        color=colors["primary"],
        yerr=inference_time_stds,
        capsize=5,
        error_kw={"ecolor": "black", "capthick": 1.5},
    )

    # Add time values on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        # Position label at the middle of the bar's height
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,  # Middle of the column
            f"{height:.1f}ms",
            ha="center",
            va="center",
            fontsize=FONT_SIZE_ANNOTATION,
            # fontweight="bold",
            color="black",  # Black text for contrast with white background
            bbox=dict(
                facecolor="white", alpha=0.8, pad=2, edgecolor="none"
            ),  # White background with slight transparency
        )

    # Add overall average latency line
    ax3.axhline(
        y=overall_avg_inference_time,
        color=HORIZONTAL_LINE_COLOR,
        linestyle="--",
        label=f"Overall Avg. Time ({overall_avg_inference_time:.1f}ms)",
    )

    ax3.set_title(
        "Inference Time by Exit Point", fontsize=FONT_SIZE_SUBPLOT_TITLE, pad=15
    )  # Added padding
    ax3.set_ylabel("Time (ms)", fontsize=FONT_SIZE_AXIS_LABEL)
    ax3.set_xlabel("Exit Point", fontsize=FONT_SIZE_AXIS_LABEL)
    ax3.set_ylim(0, max(inference_times) * 1.15)  # Give some headroom for error bars
    ax3.grid(axis="y", linestyle="--", alpha=0.7)
    ax3.legend(fontsize=FONT_SIZE_LEGEND)
    ax3.tick_params(axis="x", rotation=45, labelsize=FONT_SIZE_TICK_LABEL)
    ax3.tick_params(axis="y", labelsize=FONT_SIZE_TICK_LABEL)

    # 4. Accuracy vs Speed Scatter Plot
    ax4 = fig.add_subplot(gs[1, 1])

    # Create scatter plot of accuracy vs inference time (speed)
    scatter = ax4.scatter(  # noqa F841
        inference_times,
        accuracies,
        c=range(len(exits)),
        cmap=colors["scatter"],
        s=100,
    )

    # Add labels for each point
    for i, exit_label in enumerate(exits):
        ax4.annotate(
            exit_label,
            (inference_times[i], accuracies[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=FONT_SIZE_ANNOTATION,
        )

    ax4.set_title(
        "Accuracy vs Inference Time", fontsize=FONT_SIZE_SUBPLOT_TITLE, pad=15
    )  # Added padding
    ax4.set_xlabel("Inference Time (ms)", fontsize=FONT_SIZE_AXIS_LABEL)
    ax4.set_ylabel("Accuracy (%)", fontsize=FONT_SIZE_AXIS_LABEL)
    ax4.tick_params(axis="x", labelsize=FONT_SIZE_TICK_LABEL)
    ax4.tick_params(axis="y", labelsize=FONT_SIZE_TICK_LABEL)

    # Adjust layout with more space
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Slightly adjusted the rect parameter

    return fig


def plot_class_statistics_unified(
    metrics, title: str, colors, sort_by: str = "accuracy", top_n_classes: int = 10
):
    """
    Plot exit mode distribution for top and bottom classes by specified criteria.

    Args:
        metrics: Dictionary containing evaluation metrics with class_statistics
        title: Title for the plot
        colors: Color scheme to use
        sort_by: Criteria to sort by - either "accuracy" or "speed"
        top_n_classes: Total number of classes to highlight (half top, half bottom)

    Returns:
        Figure object
    """
    import matplotlib.patches as mpatches  # Import patches for legend

    class_stats = metrics["class_statistics"]

    # Calculate how many classes to show from top and bottom
    half_n = top_n_classes // 2
    remainder = top_n_classes % 2  # In case top_n_classes is odd

    # Sort based on criteria
    if sort_by == "accuracy":
        # High accuracy = better, so reverse=True
        all_sorted_classes = sorted(
            class_stats.items(), key=lambda x: x[1]["accuracy"], reverse=True
        )
        subtitle = "Exit Modes by Class Performance (Accuracy)"
    else:  # sort_by == "speed"
        # Low inference time = faster = better, so reverse=False
        all_sorted_classes = sorted(
            class_stats.items(),
            key=lambda x: x[1]["avg_inference_time_ms"],
            reverse=False,
        )
        subtitle = "Exit Modes by Class Performance (Speed)"

    # Get top and bottom classes
    top_classes = all_sorted_classes[: half_n + remainder]
    bottom_classes = all_sorted_classes[-half_n:]

    # Create a figure with a single subplot for exit modes
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle(f"{title} - {subtitle}", fontsize=16, y=0.98)

    # Prepare data for plotting
    class_names = []
    mode_exits = []
    bar_colors = []
    class_metrics = []  # Store the metric value (accuracy or inference time)

    # Add top classes
    for i, (class_id, stats) in enumerate(top_classes):
        class_names.append(stats["name"][:20])  # Truncate long names
        mode_exits.append(stats["mode_exit_layer"])
        bar_colors.append(colors["primary"])
        # Store the relevant metric
        if sort_by == "accuracy":
            class_metrics.append(f"{stats['accuracy']:.1f}%")
        else:
            class_metrics.append(f"{stats['avg_inference_time_ms']:.1f}ms")

    # Add bottom classes
    for i, (class_id, stats) in enumerate(bottom_classes):
        class_names.append(stats["name"][:20])  # Truncate long names
        mode_exits.append(stats["mode_exit_layer"])
        bar_colors.append("lightgray")
        # Store the relevant metric
        if sort_by == "accuracy":
            class_metrics.append(f"{stats['accuracy']:.1f}%")
        else:
            class_metrics.append(f"{stats['avg_inference_time_ms']:.1f}ms")

    # Create bar chart
    bars = ax.bar(range(len(class_names)), mode_exits, color=bar_colors)

    # Add a vertical line to separate top and bottom classes
    if half_n + remainder < len(class_names):
        ax.axvline(
            x=half_n + remainder - 0.5,
            color="black",
            linestyle="--",
            alpha=0.5,
            label="Divide between top and bottom classes",
        )

    # Add class metric values below class names
    for i, bar in enumerate(bars):
        # Add class metric (accuracy or latency) below the class name
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            -0.8,
            class_metrics[i],
            ha="center",
            va="top",
            fontsize=9,
            color=colors["primary"] if i < half_n + remainder else "dimgray",
            rotation=45,
        )

        # Add the exit layer value on top of the bar
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,  # Position slightly above the bar
            f"{mode_exits[i]}",  # Just the number, not "Exit X"
            ha="center",
            va="bottom",
            fontsize=9,
            # fontweight="bold",
        )

    # Set labels and title
    ax.set_title("Most Common Exit Layer by Class", fontsize=14)
    ax.set_ylabel("Exit Layer")
    ax.set_xlabel(
        f"Classes (sorted by {'accuracy' if sort_by == 'accuracy' else 'speed'})"
    )

    # Set x-ticks with class names
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")

    # Add gridlines
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Add legend for top vs bottom - use mpatches instead of plt.Patch
    legend_elements = [
        mpatches.Patch(
            facecolor=colors["primary"], label=f"Top {half_n + remainder} Classes"
        ),
        mpatches.Patch(facecolor="lightgray", label=f"Bottom {half_n} Classes"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    return fig


def plot_latency_accuracy_scatter(
    metrics, results_dir, colors, top_n_classes=10, title_override=None
):
    """
    Create a scatter plot showing the relationship between accuracy and latency for the
    top and bottom classes by accuracy and speed.

    Args:
        metrics (dict): Metrics dictionary containing class statistics
        results_dir (str): Directory where results are stored
        colors (dict): Color scheme to use
        top_n_classes (int): Total number of classes to display (half top, half bottom)
        title_override (str, optional): If provided, use this instead of generating a title

    Returns:
        matplotlib.figure.Figure: Scatter plot figure
    """
    # Import patches for legend
    import matplotlib.lines as mlines

    # Determine title to use
    if title_override is None:
        # Extract model type and results identifier for title (original behavior)
        model_type, results_id = get_results_info(results_dir)
        title = f"{model_type} | {results_id}"
    else:
        # Use the provided title
        title = title_override

    logger.info(f"Using title for latency-accuracy plot: {title}")

    # Rest of the function remains the same...

    # Extract class-level metrics
    class_stats = metrics.get("class_statistics", {})

    # Calculate how many classes to show from top and bottom
    half_n = top_n_classes // 2
    remainder = top_n_classes % 2  # In case top_n_classes is odd

    # Sort classes by accuracy (high to low)
    sorted_by_acc = sorted(
        class_stats.items(), key=lambda x: x[1]["accuracy"], reverse=True
    )

    # Sort classes by speed (fast to slow)
    sorted_by_speed = sorted(
        class_stats.items(), key=lambda x: x[1]["avg_inference_time_ms"], reverse=False
    )

    # Get top and bottom classes by accuracy
    top_acc_classes = sorted_by_acc[: half_n + remainder]
    bottom_acc_classes = sorted_by_acc[-half_n:]

    # Get top and bottom classes by speed
    top_speed_classes = sorted_by_speed[: half_n + remainder]
    bottom_speed_classes = sorted_by_speed[-half_n:]

    # Combine all selected classes and remove duplicates
    selected_classes = {}
    for class_id, stats in (
        top_acc_classes + bottom_acc_classes + top_speed_classes + bottom_speed_classes
    ):
        selected_classes[class_id] = stats

    # Prepare data for plotting
    class_ids = []  # Using IDs instead of names for annotations
    class_names = []  # Full names for the mapping legend
    accuracies = []
    latencies = []
    class_types = []  # To track if class is in top/bottom acc/speed

    # Keep track of which category each class belongs to
    top_acc_ids = []
    bottom_acc_ids = []
    top_speed_ids = []
    bottom_speed_ids = []

    for class_id, stats in selected_classes.items():
        class_ids.append(class_id)
        class_names.append(stats["name"])
        accuracies.append(stats["accuracy"])
        latencies.append(stats["avg_inference_time_ms"])

        # Determine class type for coloring and track which category it belongs to
        class_type = 0  # Default
        if (class_id, stats) in top_acc_classes:
            class_type = 1  # Top accuracy
            top_acc_ids.append((class_id, stats["name"], stats["accuracy"]))
        elif (class_id, stats) in bottom_acc_classes:
            class_type = 2  # Bottom accuracy
            bottom_acc_ids.append((class_id, stats["name"], stats["accuracy"]))
        elif (class_id, stats) in top_speed_classes:
            class_type = 3  # Top speed
            top_speed_ids.append(
                (class_id, stats["name"], stats["avg_inference_time_ms"])
            )
        elif (class_id, stats) in bottom_speed_classes:
            class_type = 4  # Bottom speed
            bottom_speed_ids.append(
                (class_id, stats["name"], stats["avg_inference_time_ms"])
            )
        class_types.append(class_type)

    # Create figure with adjusted size to make room for the class mapping
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle(
        f"{title} - Class Latency vs Accuracy (Top/Bottom Classes)",
        fontsize=FONT_SIZE_FIGURE_TITLE,
        y=0.98,
    )

    # Define colors for different class types
    color_map = plt.cm.get_cmap(colors["scatter"], 5)

    # Create scatter plot
    scatter = ax.scatter(  # noqa F841
        latencies,
        accuracies,
        c=class_types,  # Color by class type
        cmap=color_map,
        s=100,  # Marker size
        alpha=0.8,
    )

    # Add class IDs as labels for each point
    for i, (class_id, lat, acc) in enumerate(zip(class_ids, latencies, accuracies)):
        ax.annotate(
            str(class_id),
            (lat, acc),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=FONT_SIZE_ANNOTATION,
            fontweight="bold",
            alpha=0.9,
        )

    # Customize the plot
    ax.set_title("Class Latency vs Accuracy", fontsize=FONT_SIZE_SUBPLOT_TITLE, pad=15)
    ax.set_xlabel("Average Inference Time (ms)", fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_ylabel("Accuracy (%)", fontsize=FONT_SIZE_AXIS_LABEL)
    ax.tick_params(axis="x", labelsize=FONT_SIZE_TICK_LABEL)
    ax.tick_params(axis="y", labelsize=FONT_SIZE_TICK_LABEL)

    # Create legend for class types
    legend_elements = [
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map(1),
            markersize=10,
            label="Top Accuracy",
        ),
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map(2),
            markersize=10,
            label="Bottom Accuracy",
        ),
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map(3),
            markersize=10,
            label="Top Speed",
        ),
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map(4),
            markersize=10,
            label="Bottom Speed",
        ),
    ]
    # Placed legend in top right as requested
    ax.legend(handles=legend_elements, loc="upper right", fontsize=FONT_SIZE_LEGEND)

    # Add overall metrics
    overall_accuracy = metrics.get("overall_accuracy", "N/A")
    ax.axhline(
        y=overall_accuracy,
        color=HORIZONTAL_LINE_COLOR,
        linestyle="--",
        label=f"Overall Accuracy ({overall_accuracy}%)",
    )

    # Calculate and display correlation
    if latencies and accuracies:
        correlation = np.corrcoef(latencies, accuracies)[0, 1]
        ax.text(
            0.05,
            0.05,  # Moved to bottom left
            f"Correlation: {correlation:.2f}",
            transform=ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.8),
            fontsize=FONT_SIZE_ANNOTATION,
        )

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    # Create a textbox with class ID to name mapping on the right side
    # First adjust the main axes to make room for the textbox
    box = ax.get_position()
    ax.set_position(
        [box.x0, box.y0, box.width * 0.8, box.height]
    )  # Increased from 0.75 to 0.8 to bring text box closer

    # Create mapping text organized by category
    mapping_text = "Class ID Mapping:\n\n"

    # Add top accuracy classes
    mapping_text += "Top Accuracy:\n"
    for class_id, name, acc in sorted(top_acc_ids, key=lambda x: int(x[0])):
        mapping_text += f"{class_id}: {name} ({acc:.1f}%)\n"

    # Add bottom accuracy classes
    mapping_text += "\nBottom Accuracy:\n"
    for class_id, name, acc in sorted(bottom_acc_ids, key=lambda x: int(x[0])):
        mapping_text += f"{class_id}: {name} ({acc:.1f}%)\n"

    # Add top speed classes
    mapping_text += "\nTop Speed:\n"
    for class_id, name, speed in sorted(top_speed_ids, key=lambda x: int(x[0])):
        mapping_text += f"{class_id}: {name} ({speed:.1f}ms)\n"

    # Add bottom speed classes
    mapping_text += "\nBottom Speed:\n"
    for class_id, name, speed in sorted(bottom_speed_ids, key=lambda x: int(x[0])):
        mapping_text += f"{class_id}: {name} ({speed:.1f}ms)\n"

    # Add the mapping textbox - moved closer to main plot
    fig.text(
        0.82,  # x position (moved from 0.85 to 0.82 to be closer to plot)
        0.5,  # y position (center)
        mapping_text,
        fontsize=FONT_SIZE_SMALL_ANNOTATION,
        va="center",
        bbox=dict(
            facecolor="white", alpha=0.8, boxstyle="round,pad=0.5", edgecolor="gray"
        ),
    )

    plt.tight_layout(
        rect=[0, 0.05, 0.8, 0.95]
    )  # Adjusted from 0.75 to 0.8 to match the new position

    return fig


def plot_confusion_matrix(
    metrics: Dict[str, Any],
    title: str,
    normalize: bool = True,
    top_n_classes: Optional[int] = 4,  # Changed default to 4
    include_accuracy: bool = True,
    figsize: Tuple[int, int] = (12, 10),
) -> plt.Figure:
    """
    Create a visualization of the confusion matrix focusing on classes with highest participation.

    Args:
        metrics: Dictionary containing evaluation metrics including confusion_matrix
        title: Title for the plot
        normalize: Whether to normalize the confusion matrix by row (true class)
        top_n_classes: If provided, only show this many classes with the highest participation
        include_accuracy: Whether to include class accuracy in the y-axis labels
        figsize: Figure size as (width, height) tuple

    Returns:
        Figure object
    """
    # Extract confusion matrix and class statistics
    confusion_matrix = np.array(metrics["confusion_matrix"])
    class_stats = metrics["class_statistics"]

    # If class names are available, use them, otherwise use indices
    class_names = []
    for i in range(confusion_matrix.shape[0]):
        if str(i) in class_stats:
            class_names.append(class_stats[str(i)].get("name", f"Class {i}"))
        else:
            class_names.append(f"Class {i}")

    # Calculate participation - sum of each row (true class occurrences)
    class_participation = confusion_matrix.sum(axis=1)

    # Normalize if requested
    if normalize:
        # Normalize by row (sum of each row = 1)
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        norm_confusion_matrix = confusion_matrix / row_sums
    else:
        norm_confusion_matrix = confusion_matrix

    # Select top classes by participation (most frequent true classes)
    if top_n_classes and top_n_classes < len(class_names):
        # Get indices of top N classes by participation
        top_indices = np.argsort(class_participation)[-top_n_classes:]
        # Select only rows and columns for these classes
        norm_confusion_matrix = norm_confusion_matrix[top_indices][:, top_indices]
        class_names = [class_names[i] for i in top_indices]
        selected_indices = top_indices
    else:
        selected_indices = np.arange(len(class_names))

    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create the heatmap using seaborn
    sns.heatmap(
        norm_confusion_matrix,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="viridis" if normalize else "YlGnBu",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )

    # If including accuracy in labels and class stats are available
    if include_accuracy:
        # Create new yticklabels with accuracy information
        new_ylabels = []
        for i, label in enumerate(class_names):
            class_id = str(selected_indices[i])
            if class_id in class_stats:
                accuracy = class_stats[class_id].get("accuracy", 0)
                new_ylabels.append(f"{label} ({accuracy:.1f}%)")
            else:
                new_ylabels.append(label)

        ax.set_yticklabels(new_ylabels, fontsize=FONT_SIZE_TICK_LABEL)
    else:
        ax.set_yticklabels(class_names, fontsize=FONT_SIZE_TICK_LABEL)

    ax.set_xticklabels(
        class_names, fontsize=FONT_SIZE_TICK_LABEL, rotation=45, ha="right"
    )

    # Set title and axis labels
    ax.set_title(
        f"{title} - {'Normalized ' if normalize else ''}Confusion Matrix",
        fontsize=FONT_SIZE_SUBPLOT_TITLE,
        pad=20,
    )
    ax.set_xlabel("Predicted Class", fontsize=FONT_SIZE_AXIS_LABEL, labelpad=10)
    ax.set_ylabel("True Class", fontsize=FONT_SIZE_AXIS_LABEL, labelpad=10)

    # Add a text box with overall accuracy
    overall_accuracy = metrics.get("overall_accuracy", 0)
    ax.text(
        0.5,
        1.1,
        f"Overall Accuracy: {overall_accuracy:.2f}%",
        transform=ax.transAxes,
        ha="center",
        fontsize=FONT_SIZE_ANNOTATION,
        bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.5"),
    )

    plt.tight_layout()

    return fig


def plot_top_class_exit_distribution(metrics, title: str, colors):
    """
    Plot the exit distribution for the top fastest and most accurate classes and the bottom slowest and least accurate classes.

    Args:
        metrics: Dictionary containing evaluation metrics
        title: Title for the plot
        colors: Color scheme to use

    Returns:
        Figure object
    """
    class_stats = metrics.get("class_statistics", {})
    if not class_stats:
        raise ValueError("Class statistics not found in metrics")

    # Sort classes by accuracy and speed
    sorted_by_acc = sorted(
        class_stats.items(), key=lambda x: x[1].get("accuracy", 0), reverse=True
    )
    sorted_by_speed = sorted(
        class_stats.items(),
        key=lambda x: x[1].get("avg_inference_time_ms", float("inf")),
    )

    # Get top and bottom class from each category
    top_acc_class = sorted_by_acc[0] if sorted_by_acc else None
    bottom_acc_class = sorted_by_acc[-1] if len(sorted_by_acc) > 1 else None
    top_speed_class = sorted_by_speed[0] if sorted_by_speed else None
    bottom_speed_class = sorted_by_speed[-1] if len(sorted_by_speed) > 1 else None

    # Check if we have enough data
    if not (
        top_acc_class and bottom_acc_class and top_speed_class and bottom_speed_class
    ):
        raise ValueError("Could not find enough classes for plotting")

    # Create figure with subplots - 2x2 grid for top and bottom performers
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, figsize=(15, 12), gridspec_kw={"hspace": 0.4}
    )
    fig.suptitle(
        f"{title} - Exit Distribution for Top & Bottom Classes",
        fontsize=FONT_SIZE_FIGURE_TITLE,
        y=0.98,
    )

    # Define colormap for consistency with scatter plot
    color_map = plt.cm.get_cmap(colors["scatter"], 5)

    # Plot exit distribution for top classes
    _plot_class_exit_distribution(
        ax1,
        top_acc_class[1],
        "Most Accurate Class",
        color_map(1),  # Use accuracy color from scatter plot
        top_acc_class[1]["name"],
        show_accuracy=True,
    )

    _plot_class_exit_distribution(
        ax2,
        top_speed_class[1],
        "Fastest Class",
        color_map(3),  # Use speed color from scatter plot
        top_speed_class[1]["name"],
        show_speed=True,
    )

    # Plot exit distribution for bottom classes
    _plot_class_exit_distribution(
        ax3,
        bottom_acc_class[1],
        "Least Accurate Class",
        "#444444",  # Dark gray for bottom accuracy
        bottom_acc_class[1]["name"],
        show_accuracy=True,
    )

    _plot_class_exit_distribution(
        ax4,
        bottom_speed_class[1],
        "Slowest Class",
        "#AAAAAA",  # Light gray for bottom speed
        bottom_speed_class[1]["name"],
        show_speed=True,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def _plot_class_exit_distribution(
    ax,
    class_data,
    title_prefix,
    color,
    class_name,
    show_accuracy=False,
    show_speed=False,
):
    """
    Helper function to plot exit distribution for a class

    Args:
        ax: Matplotlib axes to plot on
        class_data: Class statistics
        title_prefix: Prefix for the plot title
        color: Color to use for the bars
        class_name: Name of the class
        show_accuracy: Whether to display the accuracy value in title
        show_speed: Whether to display the speed value in title
    """
    # Get exit distribution
    exit_dist = class_data.get("exit_distribution", {})
    if not exit_dist:
        ax.text(
            0.5,
            0.5,
            "No exit data available",
            ha="center",
            va="center",
            fontsize=FONT_SIZE_ANNOTATION,
        )
        ax.set_title(f"{title_prefix}: {class_name}", fontsize=FONT_SIZE_SUBPLOT_TITLE)
        return

    # Sort exit keys for consistent ordering
    def exit_sort_key(exit_name):
        if exit_name == "final":
            return float("inf")  # Final exit should appear last
        else:
            parts = exit_name.split("_")
            return int(parts[1]) if len(parts) > 1 else float("inf")

    # Prepare data for plotting
    exits = []
    counts = []

    for exit_key in sorted(exit_dist.keys(), key=exit_sort_key):
        # Format exit name
        if exit_key == "final":
            exit_name = "Final"
        else:
            parts = exit_key.split("_")
            exit_name = f"Exit {parts[1]}" if len(parts) > 1 else exit_key

        exits.append(exit_name)
        counts.append(exit_dist[exit_key])

    # Create bar chart
    bars = ax.bar(exits, counts, color=color)

    # Add count and percentage labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE_ANNOTATION,
        )

    # Create enhanced title with metrics if requested
    title = f"{title_prefix}: {class_name}"
    if show_accuracy and "accuracy" in class_data:
        title += f" (Accuracy: {class_data['accuracy']:.1f}%)"
    if show_speed and "avg_inference_time_ms" in class_data:
        title += f" (Latency: {class_data['avg_inference_time_ms']:.1f}ms)"

    # Set titles and labels
    ax.set_title(title, fontsize=0.85 * FONT_SIZE_SUBPLOT_TITLE, pad=15)
    ax.set_xlabel("Exit Point", fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_ylabel("Sample Count", fontsize=FONT_SIZE_AXIS_LABEL)
    ax.tick_params(axis="x", labelsize=FONT_SIZE_TICK_LABEL)
    ax.tick_params(axis="y", labelsize=FONT_SIZE_TICK_LABEL)

    # Add grid lines for readability
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Rotate x-tick labels if many exits
    if len(exits) > 4:
        ax.set_xticklabels(exits, rotation=45, ha="right")


def plot_class_statistics_combined(
    metrics, title: str, colors, n_per_category: int = 2
):
    """
    Plot exit mode distribution for top and bottom classes by both accuracy and speed in a single plot.

    Args:
        metrics: Dictionary containing evaluation metrics with class_statistics
        title: Title for the plot
        colors: Color scheme to use
        top_n_classes: Total number of classes to highlight (half top, half bottom) per category

    Returns:
        Figure object
    """
    import matplotlib.patches as mpatches  # Import patches for legend

    class_stats = metrics["class_statistics"]

    # Sort by accuracy (high to low)
    sorted_by_acc = sorted(
        class_stats.items(), key=lambda x: x[1]["accuracy"], reverse=True
    )

    # Sort by speed (fast to slow)
    sorted_by_speed = sorted(
        class_stats.items(),
        key=lambda x: x[1]["avg_inference_time_ms"],
        reverse=False,
    )

    # Get top and bottom classes by both criteria
    top_acc_classes = sorted_by_acc[:n_per_category]
    bottom_acc_classes = sorted_by_acc[-n_per_category:]
    top_speed_classes = sorted_by_speed[:n_per_category]
    bottom_speed_classes = sorted_by_speed[-n_per_category:]

    # Create a figure with a single subplot
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.suptitle(
        f"{title} - Exit Modes by Class Performance (Combined)",
        fontsize=FONT_SIZE_FIGURE_TITLE,
        y=0.98,
    )

    # Define color map for consistency with scatter plot
    color_map = plt.cm.get_cmap(colors["scatter"], 5)

    # Create a dictionary to track which classes we've already processed
    # We'll display classes multiple times if they appear in multiple categories
    processed_classes = {}

    # Prepare data for plotting - organize all classes
    class_data = []

    # Add top accuracy classes
    for i, (class_id, stats) in enumerate(top_acc_classes):
        category_key = f"acc_top_{class_id}"
        if category_key not in processed_classes:
            processed_classes[category_key] = True
            class_data.append(
                {
                    "name": stats["name"][:20],  # Truncate long names
                    "exit_mode": stats["mode_exit_layer"],
                    "category": "Top Accuracy",
                    "color": color_map(1),  # Color for top accuracy
                    "metric": f"{stats['accuracy']:.1f}%",
                }
            )

    # Add bottom accuracy classes
    for i, (class_id, stats) in enumerate(bottom_acc_classes):
        category_key = f"acc_bottom_{class_id}"
        if category_key not in processed_classes:
            processed_classes[category_key] = True
            class_data.append(
                {
                    "name": stats["name"][:20],
                    "exit_mode": stats["mode_exit_layer"],
                    "category": "Bottom Accuracy",
                    "color": color_map(2),  # Color for bottom accuracy
                    "metric": f"{stats['accuracy']:.1f}%",
                }
            )

    # Add top speed classes
    for i, (class_id, stats) in enumerate(top_speed_classes):
        category_key = f"speed_top_{class_id}"
        if category_key not in processed_classes:
            processed_classes[category_key] = True
            class_data.append(
                {
                    "name": stats["name"][:20],
                    "exit_mode": stats["mode_exit_layer"],
                    "category": "Top Speed",
                    "color": color_map(3),  # Color for top speed
                    "metric": f"{stats['avg_inference_time_ms']:.1f}ms",
                }
            )

    # Add bottom speed classes
    for i, (class_id, stats) in enumerate(bottom_speed_classes):
        category_key = f"speed_bottom_{class_id}"
        if category_key not in processed_classes:
            processed_classes[category_key] = True
            class_data.append(
                {
                    "name": stats["name"][:20],
                    "exit_mode": stats["mode_exit_layer"],
                    "category": "Bottom Speed",
                    "color": color_map(4),  # Color for bottom speed
                    "metric": f"{stats['avg_inference_time_ms']:.1f}ms",
                }
            )

    # Create bar chart - extract data from our prepared list
    class_names = [item["name"] for item in class_data]
    exit_modes = [item["exit_mode"] for item in class_data]
    bar_colors = [item["color"] for item in class_data]
    class_metrics = [item["metric"] for item in class_data]

    # Create bars
    bars = ax.bar(range(len(class_names)), exit_modes, color=bar_colors)

    # Add metrics and exit mode values
    for i, bar in enumerate(bars):
        # Add metric value below class name
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.4,
            class_metrics[i],
            ha="center",
            va="top",
            fontsize=FONT_SIZE_ANNOTATION,
            color="black",
        )

    # Set labels and title
    ax.set_title(
        "Most Common Exit Layer by Class Performance",
        fontsize=FONT_SIZE_SUBPLOT_TITLE,
        pad=15,
    )
    ax.set_ylabel("Exit Layer", fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_xlabel("Classes", fontsize=FONT_SIZE_AXIS_LABEL)

    # Set x-ticks with class names
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(
        class_names, rotation=45, ha="right", fontsize=FONT_SIZE_TICK_LABEL
    )
    ax.tick_params(axis="y", labelsize=FONT_SIZE_TICK_LABEL)

    # Add gridlines
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Add legend for categories
    legend_elements = [
        mpatches.Patch(facecolor=color_map(1), label="Top Accuracy"),
        mpatches.Patch(facecolor=color_map(2), label="Bottom Accuracy"),
        mpatches.Patch(facecolor=color_map(3), label="Top Speed"),
        mpatches.Patch(facecolor=color_map(4), label="Bottom Speed"),
    ]
    ax.legend(handles=legend_elements, loc="best", fontsize=FONT_SIZE_LEGEND)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    return fig


def choose_color_scheme_cli():
    """Command-line interface for selecting color scheme"""
    print("Available color schemes:")
    for i, scheme in enumerate(COLOR_SCHEMES_BACKEND.keys(), 1):
        print(f"{i}. {scheme}")

    while True:
        try:
            choice = input("Select a color scheme (number or name) [default: teal]: ")

            # If empty input, use default
            if not choice.strip():
                return "teal"

            # Try as a number first
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(COLOR_SCHEMES_BACKEND):
                    return list(COLOR_SCHEMES_BACKEND.keys())[idx]
            except ValueError:
                # Not a number, try as a name
                if choice in COLOR_SCHEMES_BACKEND:
                    return choice

            print("Invalid selection. Please try again.")
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled. Using default color scheme.")
            return "teal"
