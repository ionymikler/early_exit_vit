import json
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import shutil  # noqa F401
from datetime import datetime
from .logging_utils import get_logger_ready

logger = get_logger_ready(__name__)

# Color schemes for different backends
COLOR_SCHEMES = {
    "onnx-cpu": {
        "primary": "#2878BD",  # Dark blue
        "secondary": "#8CC7FF",  # Lighter shade
        "tertiary": "#5AA7FF",  # Medium shade
        "scatter": "Blues",  # Colormap for scatter plots
    },
    "onnx-gpu": {
        "primary": "#4B0082",  # Indigo
        "secondary": "#8A2BE2",  # Blue Violet
        "tertiary": "#9370DB",  # Medium Purple
        "scatter": "Purples",  # Colormap for scatter plots
    },
    "nvidia-onnx-cpu": {
        "primary": "#483D8B",  # Dark Slate Blue
        "secondary": "#66CDAA",  # Lighter shade
        "tertiary": "#228B22",  # Medium shade
        "scatter": "Greens",  # Colormap for scatter plots
    },
    "default": {
        "primary": "#808080",  # Gray
        "secondary": "#A9A9A9",  # Dark Gray
        "tertiary": "#D3D3D3",  # Light Gray
        "scatter": "Greys",  # Colormap for scatter plots
    },
}


def save_metadata(results_dir: str, model_type: str, args=None):
    """
    Save evaluation metadata to a YAML file.

    Args:
        results_dir: Directory to save metadata to
        model_type: Type of model ('pytorch' or 'onnx')
        args: Command-line arguments used for evaluation
    """
    if args is None:
        return

    try:
        import yaml

        # Convert args to dictionary if it's not already
        if hasattr(args, "__dict__"):
            args_dict = vars(args)
        else:
            args_dict = args

        # Create metadata object
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        metadata = {"timestamp": timestamp, "model_type": model_type, "args": args_dict}

        metadata_file = f"{results_dir}/metadata.yaml"
        with open(metadata_file, "w") as f:
            yaml.dump(metadata, f, default_flow_style=False)

        logger.info(f"Metadata saved to {metadata_file}")
    except Exception as e:
        logger.error(f"Failed to save metadata: {e}")


def save_metrics(metrics, file_prefix: str, args=None):
    """
    Save metrics to JSON file and create a symlink to the latest results.

    Args:
        metrics: Dictionary containing evaluation metrics
        file_prefix: Prefix for the output files
        args: Optional argparse.Namespace object containing the evaluation arguments
    """
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")

    # Create a subdirectory for results with datetime and model type
    model_type = "pytorch" if "pytorch" in file_prefix else "onnx"
    results_dir = f"results/{model_type}_{timestamp}"

    # Create directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    metrics_file = f"{results_dir}/{file_prefix}_metrics.json"

    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"Metrics saved to {metrics_file}")

    # Save metadata
    save_metadata(results_dir, model_type, args)

    # # Create a soft link to the latest results directory
    # latest_link = f"results/{model_type}_latest"

    # # Remove existing symlink if it exists
    # if os.path.islink(latest_link):
    #     os.unlink(latest_link)
    # elif os.path.exists(latest_link):
    #     # If it's a directory instead of a symlink, remove it
    #     shutil.rmtree(latest_link)

    # # Create the new symlink
    # os.symlink(results_dir, latest_link, target_is_directory=True)
    # logger.info(f"Created symlink: {latest_link} -> {results_dir}")


def load_metrics(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def plot_metrics(metrics, title: str, color_scheme="default", top_n_classes=10):
    """
    Plot metrics for model evaluation, including both exit statistics and class statistics.

    Args:
        metrics: Dictionary containing evaluation metrics
        title: Title for the plot
        color_scheme: Color scheme to use
        top_n_classes: Number of top classes to highlight in class-specific plots

    Returns:
        Tuple of figures: (exit_stats_figure, class_accuracy_stats_figure, class_speed_stats_figure)
    """
    # Get color scheme
    colors = COLOR_SCHEMES.get(color_scheme, COLOR_SCHEMES["default"])

    # Create exit statistics visualization (now with column charts)
    exit_fig = plot_exit_statistics(metrics, title, colors)

    # Create class statistics visualizations if available
    class_accuracy_fig = None
    class_speed_fig = None

    if "class_statistics" in metrics and metrics["class_statistics"]:
        # Use the unified function with different sorting criteria
        class_accuracy_fig = plot_class_statistics_unified(
            metrics, title, colors, sort_by="accuracy", top_n_classes=top_n_classes
        )
        class_speed_fig = plot_class_statistics_unified(
            metrics, title, colors, sort_by="speed", top_n_classes=top_n_classes
        )

    return exit_fig, class_accuracy_fig, class_speed_fig


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

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f"{title} - Exit Statistics Analysis", fontsize=16, y=0.98)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. Sample Distribution (Bar Chart)
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(exits, counts, color=colors["primary"])
    ax1.set_title("Sample Distribution Across Exits")
    ax1.set_xlabel("Exit Point")
    ax1.set_ylabel("Number of Samples")
    ax1.tick_params(axis="x", rotation=45)

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
        )

    # 2. Accuracy Column Chart with Error Bars
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(
        exits,
        accuracies,
        color=colors["secondary"],
        yerr=accuracy_stds,
        capsize=5,
        error_kw={"ecolor": "black", "capthick": 1.5},
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
            fontweight="bold",
        )

    # Add overall accuracy line
    ax2.axhline(
        y=metrics["overall_accuracy"],
        color=colors["tertiary"],
        linestyle="--",
        label=f"Overall Accuracy ({metrics['overall_accuracy']:.1f}%)",
    )

    ax2.set_title("Accuracy by Exit Point")
    ax2.set_xlabel("Exit Point")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, max(accuracies) * 1.15)  # Give some headroom for error bars
    ax2.grid(axis="y", linestyle="--", alpha=0.7)
    ax2.legend()

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
            fontweight="bold",
            color="black",  # Black text for contrast with white background
            bbox=dict(
                facecolor="white", alpha=0.8, pad=2, edgecolor="none"
            ),  # White background with slight transparency
        )

    ax3.set_title("Inference Time by Exit Point")
    ax3.set_ylabel("Time (ms)")
    ax3.set_xlabel("Exit Point")
    ax3.set_ylim(0, max(inference_times) * 1.15)  # Give some headroom for error bars
    ax3.grid(axis="y", linestyle="--", alpha=0.7)

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
        )

    ax4.set_title("Accuracy vs Inference Time")
    ax4.set_xlabel("Inference Time (ms)")
    ax4.set_ylabel("Accuracy (%)")

    # Add correlation coefficient
    correlation = np.corrcoef(inference_times, accuracies)[0, 1]
    ax4.text(
        0.05,
        0.95,
        f"Correlation: {correlation:.2f}",
        transform=ax4.transAxes,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


def plot_class_statistics_unified(
    metrics, title: str, colors, sort_by: str = "accuracy", top_n_classes: int = 10
):
    """
    Plot class-specific statistics showing top and bottom performers by specified criteria.

    Args:
        metrics: Dictionary containing evaluation metrics with class_statistics
        title: Title for the plot
        colors: Color scheme to use
        sort_by: Criteria to sort by - either "accuracy" or "speed"
        top_n_classes: Total number of classes to highlight (half top, half bottom)

    Returns:
        Figure object
    """
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
        subtitle = "Class Performance Analysis by Accuracy"
        legend_text = f"Showing top {half_n + remainder if remainder else half_n} and bottom {half_n} classes by accuracy"
    else:  # sort_by == "speed"
        # Low inference time = faster = better, so reverse=False
        all_sorted_classes = sorted(
            class_stats.items(),
            key=lambda x: x[1]["avg_inference_time_ms"],
            reverse=False,
        )
        subtitle = "Class Performance Analysis by Speed"
        legend_text = f"Showing fastest {half_n + remainder if remainder else half_n} and slowest {half_n} classes"

    # Get top and bottom classes
    top_classes = all_sorted_classes[: half_n + remainder]
    bottom_classes = all_sorted_classes[-half_n:]

    # Combine them, with top classes first
    sorted_classes = top_classes + bottom_classes

    # Extract data for plotting
    class_ids = [int(class_id) for class_id, _ in sorted_classes]
    class_names = [stats["name"] for _, stats in sorted_classes]
    class_accuracies = [stats["accuracy"] for _, stats in sorted_classes]
    std_exits = [stats["std_exit_layer"] for _, stats in sorted_classes]
    mode_exits = [stats["mode_exit_layer"] for _, stats in sorted_classes]
    avg_times = [stats["avg_inference_time_ms"] for _, stats in sorted_classes]
    std_times = [stats["std_inference_time_ms"] for _, stats in sorted_classes]
    avg_confidences = [stats["avg_confidence"] for _, stats in sorted_classes]

    # Create figure with subplots - 2x2 grid
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(f"{title} - {subtitle}", fontsize=16, y=0.98)
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)

    # Shortened class names for display
    short_names = [
        name[:15] + ("..." if len(name) > 15 else "") for name in class_names
    ]

    # Create colors for bars - different colors for top and bottom classes
    bar_colors = []
    for i in range(len(sorted_classes)):
        if i < half_n + remainder:  # Top classes
            bar_colors.append(colors["primary"])
        else:  # Bottom classes
            bar_colors.append("lightgray")

    x = range(len(class_ids))

    # 1. Class Accuracy (TOP LEFT)
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(x, class_accuracies, color=bar_colors)
    ax1.set_title("Accuracy by Class")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, rotation=45, ha="right")
    ax1.axhline(
        y=metrics["overall_accuracy"],
        color=colors["tertiary"],
        linestyle="--",
        label=f"Overall Accuracy ({metrics['overall_accuracy']:.1f}%)",
    )
    # Add a vertical separator between top and bottom classes
    if half_n + remainder < len(sorted_classes):
        ax1.axvline(
            x=half_n + remainder - 0.5, color="black", linestyle="--", alpha=0.3
        )

    # Set y-axis limit to 120 for better label visibility
    ax1.set_ylim(0, 120)

    # Add accuracy values on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
        )
    ax1.legend()

    # 2. Inference Time with Error Bars (TOP RIGHT)
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(x, avg_times, color=bar_colors, yerr=std_times, capsize=5)
    ax2.set_title("Inference Time by Class")
    ax2.set_ylabel("Time (ms)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, rotation=45, ha="right")
    # Add a vertical separator
    if half_n + remainder < len(sorted_classes):
        ax2.axvline(
            x=half_n + remainder - 0.5, color="black", linestyle="--", alpha=0.3
        )

    # Add time values on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{avg_times[i]:.1f}ms",
            ha="center",
            va="bottom",
        )

    # 3. Average Exit Layer with Error Bars (BOTTOM LEFT)
    ax3 = fig.add_subplot(gs[1, 0])
    bars = ax3.bar(x, mode_exits, color=bar_colors, yerr=std_exits, capsize=5)
    ax3.set_title("Most Common Exit Layer by Class")
    ax3.set_ylabel("Exit Layer")
    ax3.set_xticks(x)
    ax3.set_xticklabels(short_names, rotation=45, ha="right")
    # Add separator
    if half_n + remainder < len(sorted_classes):
        ax3.axvline(
            x=half_n + remainder - 0.5, color="black", linestyle="--", alpha=0.3
        )

    # Add exit layer values on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 3.0,
            height,
            f"{mode_exits[i]}",
            ha="center",
            va="bottom",
        )

    # 4. Confidence Distribution (BOTTOM RIGHT)
    ax4 = fig.add_subplot(gs[1, 1])
    bars = ax4.bar(x, avg_confidences, color=bar_colors)
    ax4.set_title("Average Confidence by Class")
    ax4.set_ylabel("Confidence")
    ax4.set_xticks(x)
    ax4.set_xticklabels(short_names, rotation=45, ha="right")
    # Add separator
    if half_n + remainder < len(sorted_classes):
        ax4.axvline(
            x=half_n + remainder - 0.5, color="black", linestyle="--", alpha=0.3
        )

    # Add confidence values on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{avg_confidences[i]:.2f}",
            ha="center",
            va="bottom",
        )

    # Add a legend explaining the classes
    ax4.text(
        0.95,
        0.05,
        legend_text,
        transform=ax4.transAxes,
        ha="right",
        bbox=dict(facecolor="white", alpha=0.8),
        fontsize=9,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


def choose_color_scheme_cli():
    """Command-line interface for selecting color scheme"""
    print("Available color schemes:")
    for i, scheme in enumerate(COLOR_SCHEMES.keys(), 1):
        print(f"{i}. {scheme}")

    while True:
        try:
            choice = input("Select a color scheme (number or name): ")
            # Try as a number first
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(COLOR_SCHEMES):
                    return list(COLOR_SCHEMES.keys())[idx]
            except ValueError:
                # Not a number, try as a name
                if choice in COLOR_SCHEMES:
                    return choice

            print("Invalid selection. Please try again.")
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled.")
            sys.exit(1)
