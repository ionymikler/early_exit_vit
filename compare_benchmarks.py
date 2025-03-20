#!/usr/bin/env python
# Compare EEVIT benchmark runs by visualizing exit distributions and metrics
# Created for comparing performance metrics of EEVIT across different benchmark runs

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple

# Define color palette for the runs (up to 4 distinct colors)
COLORS_PALETTES = [
    [
        "#57B4BA",
        "#015551",
        "#FE4F2D",
        "#FDFBEE",
    ],  # Light Teal, Dark Teal, Red Orange, Frosting Cream
    [
        "#FF6500",
        "#FF8A08",
        "#5CB338",
        "#16C47F",
    ],  # Orange, Light Orange, Green, Light Green
]

COLORS = COLORS_PALETTES[0]

# Constants for plotting
PLOT_TITLE_FONTSIZE = 16
AXIS_LABEL_FONTSIZE = 14
LEGEND_FONTSIZE = 12
BAR_WIDTH = 0.15  # Will be adjusted based on number of runs
FIGURE_SIZE = (15, 10)


def find_metric_files_in_directory(directory: str) -> List[str]:
    """
    Recursively find all directories containing advanced_metrics.json files
    in the given directory.

    Args:
        directory: The root directory to search in

    Returns:
        List of directories containing metric files
    """
    metric_dirs = []

    # Check if the provided directory itself contains metrics
    if os.path.isfile(os.path.join(directory, "advanced_metrics.json")):
        metric_dirs.append(directory)
        return metric_dirs

    # If not, look for subdirectories containing metrics
    for root, dirs, files in os.walk(directory):
        if "advanced_metrics.json" in files:
            metric_dirs.append(root)

    return metric_dirs


def load_metrics_from_dirs(
    directories: List[str], recursive: bool = False
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Load metrics from multiple directories.

    Args:
        directories: List of directory paths containing metrics files
        recursive: If True, search for metrics files in subdirectories

    Returns:
        Tuple of (list of metrics dictionaries, list of run names)
    """
    metrics_list = []
    run_names = []

    # Expand directories if recursive mode is enabled
    expanded_dirs = []
    if recursive:
        for directory in directories:
            if os.path.isdir(directory):
                subdirs = find_metric_files_in_directory(directory)
                expanded_dirs.extend(subdirs)
            else:
                expanded_dirs.append(directory)
    else:
        expanded_dirs = directories

    print(f"Found {len(expanded_dirs)} directories with metrics files")

    for dir_path in expanded_dirs:
        # Use the directory name as the run name
        run_name = os.path.basename(os.path.normpath(dir_path)).replace("_", " ")

        # Look only for advanced metrics
        metrics_path = os.path.join(dir_path, "advanced_metrics.json")

        if not os.path.exists(metrics_path):
            print(f"Warning: No advanced_metrics.json found in {dir_path}")
            continue

        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
                metrics_list.append(metrics)
                run_names.append(run_name)
                print(f"Loaded metrics from {metrics_path}")
        except Exception as e:
            print(f"Error loading metrics from {metrics_path}: {e}")

    # Sort both metrics_list and run_names alphabetically by run_names
    sorted_pairs = sorted(zip(run_names, metrics_list), key=lambda x: x[0])
    run_names = [pair[0] for pair in sorted_pairs]
    metrics_list = [pair[1] for pair in sorted_pairs]

    return metrics_list, run_names


def get_metrics_by_exit(
    metrics_list: List[Dict[str, Any]],
) -> Dict[str, Dict[str, List[Any]]]:
    """
    Organize metrics by exit point for comparison across runs.

    Args:
        metrics_list: List of metrics dictionaries from different runs

    Returns:
        Dictionary organized by metric type and exit point
    """
    # Create structure to organize metrics by exit points
    exit_metrics = {
        "sample_percentages": {},  # Exit -> [count from run1, count from run2, ...]
        "accuracies": {},  # Exit -> [accuracy from run1, accuracy from run2, ...]
        "inference_times": {},  # Exit -> [time from run1, time from run2, ...]
        "exit_indices": {},  # Sorted exit indices for consistent ordering
    }

    # Collect all unique exit points across all runs
    all_exits = set()
    for metrics in metrics_list:
        exit_stats = metrics.get("exit_statistics", {})
        all_exits.update(exit_stats.keys())

    # Initialize lists for each exit point
    for exit_point in all_exits:
        exit_metrics["sample_percentages"][exit_point] = []
        exit_metrics["accuracies"][exit_point] = []
        exit_metrics["inference_times"][exit_point] = []

    # Fill in data from each run
    for metrics in metrics_list:
        exit_stats = metrics.get("exit_statistics", {})

        # Handle each exit point for this run
        for exit_point in all_exits:
            if exit_point in exit_stats:
                # Advanced metrics use consistent field names
                exit_metrics["sample_percentages"][exit_point].append(
                    exit_stats[exit_point].get("percentage", 0)
                )

                exit_metrics["accuracies"][exit_point].append(
                    exit_stats[exit_point].get("avg_accuracy", 0)
                )

                exit_metrics["inference_times"][exit_point].append(
                    exit_stats[exit_point].get("avg_inference_time_ms", 0)
                )
            else:
                # Exit point not present in this run
                exit_metrics["sample_percentages"][exit_point].append(0)
                exit_metrics["accuracies"][exit_point].append(0)
                exit_metrics["inference_times"][exit_point].append(0)

    # Sort exit points to ensure consistent ordering
    # Convert 'exit_X' format to numbers for sorting
    def exit_sort_key(exit_name):
        if exit_name == "final":
            return float("inf")  # Final exit should appear last
        else:
            parts = exit_name.split("_")
            return int(parts[1]) if len(parts) > 1 else float("inf")

    sorted_exits = sorted(all_exits, key=exit_sort_key)
    exit_metrics["exit_indices"] = sorted_exits

    return exit_metrics


def plot_sample_distribution(
    exit_metrics: Dict[str, Dict[str, List[Any]]],
    run_names: List[str],
    title_suffix: str = "",
) -> plt.Figure:
    """
    Create a bar chart comparing sample distributions across runs.

    Args:
        exit_metrics: Dictionary with metrics organized by exit point
        run_names: Names of the runs to display in the legend
        title_suffix: Optional suffix for the plot title

    Returns:
        The matplotlib figure
    """
    exits = exit_metrics["exit_indices"]
    counts = exit_metrics["sample_percentages"]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Calculate bar width based on number of runs
    bar_width = min(BAR_WIDTH, 0.75 / len(run_names))

    # Determine x positions for each group of bars
    x = np.arange(len(exits))

    # Create bars for each run
    for i, run_name in enumerate(run_names):
        # Get counts for this run, ensuring the order matches sorted exits
        run_counts = [
            counts[exit_point][i] if i < len(counts[exit_point]) else 0
            for exit_point in exits
        ]

        # Position offset for this run's bars
        offset = bar_width * (i - len(run_names) / 2 + 0.5)

        # Create bars
        bars = ax.bar(
            x + offset,
            run_counts,
            bar_width,
            label=run_name,
            color=COLORS[i % len(COLORS)],
        )

        # Add count labels on top of bars
        for bar_idx, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:  # Only add labels to non-zero bars
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    rotation=0,
                )

    # Customize the plot
    ax.set_title(
        f"Sample Distribution Across Exit Points{title_suffix}",
        fontsize=PLOT_TITLE_FONTSIZE,
    )
    ax.set_xlabel("Exit Point", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Percentage of Total Samples (%)", fontsize=AXIS_LABEL_FONTSIZE)

    # Set x-tick labels to be more readable
    x_labels = []
    for exit_name in exits:
        if exit_name == "final":
            x_labels.append("Final")
        else:
            # Extract the index from exit_X
            parts = exit_name.split("_")
            x_labels.append(f"Exit {parts[1]}" if len(parts) > 1 else exit_name)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")

    # Add legend
    ax.legend(fontsize=LEGEND_FONTSIZE)

    # Add grid
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()

    return fig


def plot_accuracy_latency_scatter(
    exit_metrics: Dict[str, Dict[str, List[Any]]],
    metrics_list: List[Dict[str, Any]],
    run_names: List[str],
    title_suffix: str = "",
) -> plt.Figure:
    """
    Create a scatter plot comparing accuracy vs latency for each exit point across runs.

    Args:
        exit_metrics: Dictionary with metrics organized by exit point
        metrics_list: List of complete metrics dictionaries for overall stats
        run_names: Names of the runs to display in the legend
        title_suffix: Optional suffix for the plot title

    Returns:
        The matplotlib figure
    """
    exits = exit_metrics["exit_indices"]
    accuracies = exit_metrics["accuracies"]
    times = exit_metrics["inference_times"]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Marker styles for different runs
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "X", "h"]

    # Create a scatter plot for each run
    for i, run_name in enumerate(run_names):
        # Get data for this run
        run_accuracies = []
        run_times = []
        exit_labels = []

        for j, exit_name in enumerate(exits):
            if i < len(accuracies[exit_name]):
                acc = accuracies[exit_name][i]
                time_ms = times[exit_name][i]

                # Only add points with valid data
                if acc > 0 and time_ms > 0:
                    run_accuracies.append(acc)
                    run_times.append(time_ms)

                    # Create exit label (Exit 0, Exit 1, Final)
                    if exit_name == "final":
                        exit_labels.append("Final")
                    else:
                        parts = exit_name.split("_")
                        exit_labels.append(
                            f"Exit {parts[1]}" if len(parts) > 1 else exit_name
                        )

        # Skip if no valid data points
        if not run_accuracies:
            continue

        # Plot scatter points for this run (without label for legend)
        scatter = ax.scatter(  # noqa F841
            run_times,
            run_accuracies,
            color=COLORS[i % len(COLORS)],
            marker=markers[i % len(markers)],
            s=100,  # Marker size
            alpha=0.7,
        )

        # Add labels for each exit point
        for j, (x, y, label) in enumerate(zip(run_times, run_accuracies, exit_labels)):
            ax.annotate(
                label,
                (x, y),
                xytext=(7, 0),
                textcoords="offset points",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
                ha="left",
                va="center",
            )

        # Add dashed connecting lines between points (sorted by inference time)
        if len(run_times) > 1:
            # Sort points by inference time
            sorted_indices = np.argsort(run_times)
            sorted_times = [run_times[j] for j in sorted_indices]
            sorted_accs = [run_accuracies[j] for j in sorted_indices]

            # Draw connecting line
            ax.plot(
                sorted_times,
                sorted_accs,
                "--",
                color=COLORS[i % len(COLORS)],
                alpha=0.5,
                linewidth=1.5,
            )

    # Add overall average data points
    for i, run_name in enumerate(run_names):
        if i < len(metrics_list):
            # Get overall metrics
            overall_accuracy = metrics_list[i].get("overall_accuracy", 0)

            # Calculate weighted average inference time
            total_time = 0
            total_samples = 0
            exit_stats = metrics_list[i].get("exit_statistics", {})

            for exit_name, stats in exit_stats.items():
                exit_count = stats.get("count", 0)
                exit_time = stats.get("avg_inference_time_ms", 0)
                total_time += exit_count * exit_time
                total_samples += exit_count

            overall_time = total_time / total_samples if total_samples > 0 else 0

            # Skip if no valid data
            if overall_accuracy <= 0 or overall_time <= 0:
                continue

            # Plot overall point with a larger marker (without label for legend)
            ax.scatter(
                overall_time,
                overall_accuracy,
                color=COLORS[i % len(COLORS)],
                marker="*",  # Use star marker to distinguish
                s=250,  # Larger marker size
                alpha=1.0,
                zorder=0,  # Ensure it's on top
                edgecolors="black",
            )

    # Create custom legend elements
    legend_elements = []
    for i, run_name in enumerate(run_names):
        # Regular exit point marker
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker=markers[i % len(markers)],
                color="w",
                markerfacecolor=COLORS[i % len(COLORS)],
                markersize=10,
                label=run_name,
            )
        )

        # Overall marker for this run
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor=COLORS[i % len(COLORS)],
                markersize=15,
                markeredgecolor="black",
                markeredgewidth=1.5,
                label=f"{run_name} (Overall)",
            )
        )

    # Add a grid
    ax.grid(True, linestyle="--", alpha=0.7)

    # Customize the plot
    ax.set_title(
        f"Accuracy vs Latency by Exit Point{title_suffix}", fontsize=PLOT_TITLE_FONTSIZE
    )
    ax.set_xlabel("Inference Time (ms)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Accuracy (%)", fontsize=AXIS_LABEL_FONTSIZE)

    # Set axis limits with some padding
    if ax.get_xlim()[0] > 0:
        ax.set_xlim(left=0)
    if ax.get_ylim()[0] > 0:
        ax.set_ylim(bottom=0)

    # Add custom legend
    ax.legend(handles=legend_elements, fontsize=LEGEND_FONTSIZE)

    plt.tight_layout()

    return fig


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Compare metrics across multiple EEVIT benchmark runs"
    )

    parser.add_argument(
        "directories",
        type=str,
        nargs="+",
        help="Directories containing benchmark metrics files",
    )

    parser.add_argument(
        "--title-suffix",
        "-t",
        type=str,
        default="",
        help="Optional suffix to add to plot titles",
    )

    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Search for metrics in subdirectories",
    )

    parser.add_argument(
        "--save-dir",
        "-s",
        type=str,
        default="results/combined_benchmarks_plots",
        help="Directory to save plots in",
    )

    parser.add_argument(
        "--auto-save",
        "-a",
        action="store_true",
        help="Automatically save plots without asking",
    )

    parser.add_argument(
        "--color-palette",
        "-c",
        type=int,
        choices=[0, 1],
        default=0,
        help="Color palette to use (0 or 1)",
    )

    parser.add_argument(
        "--file-prefix",
        "-p",
        type=str,
        default="",
        help="Prefix to add to output filenames",
    )

    return parser.parse_args()


def main():
    """Main function to run the script"""
    args = parse_arguments()

    # Set the color palette
    global COLORS
    COLORS = COLORS_PALETTES[args.color_palette]

    # Load metrics from all directories
    metrics_list, run_names = load_metrics_from_dirs(args.directories, args.recursive)

    if not metrics_list:
        print("No valid metrics files found. Exiting.")
        return

    print(f"Loaded metrics from {len(metrics_list)} runs: {', '.join(run_names)}")

    # Process metrics by exit point
    exit_metrics = get_metrics_by_exit(metrics_list)

    # Plot sample distribution
    sample_plot = plot_sample_distribution(exit_metrics, run_names, args.title_suffix)

    # Plot accuracy vs latency scatter plot
    scatter_plot = plot_accuracy_latency_scatter(
        exit_metrics, metrics_list, run_names, args.title_suffix
    )

    # Show plots
    sample_plot.show()
    scatter_plot.show()

    # Determine whether to save plots
    save_images = args.auto_save
    if not save_images:
        save_images = (
            input("Do you want to save the plots as images? (y/n): ").strip().lower()
            == "y"
        )

    if save_images:
        # Create plots directory
        plots_dir = args.save_dir
        os.makedirs(plots_dir, exist_ok=True)
        file_prefix = args.file_prefix + "_" if args.file_prefix else ""
        sample_plot.savefig(
            os.path.join(plots_dir, f"{file_prefix}sample_distribution.png"), dpi=300
        )
        scatter_plot.savefig(
            os.path.join(plots_dir, f"{file_prefix}accuracy_latency_scatter.png"),
            dpi=300,
        )

        print(f"Plots saved in '{plots_dir}'")

    plt.show()


if __name__ == "__main__":
    main()
