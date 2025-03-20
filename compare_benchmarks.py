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
COLORS = [
    "#57B4BA",  # Light Teal
    "#015551",  # Dark Teal
    "#FE4F2D",  # Red Orange
    "#FDFBEE",  # Frosting Cream
]


# Constants for plotting
PLOT_TITLE_FONTSIZE = 16
AXIS_LABEL_FONTSIZE = 14
LEGEND_FONTSIZE = 12
BAR_WIDTH = 0.15  # Will be adjusted based on number of runs
FIGURE_SIZE = (15, 10)


def load_metrics_from_dirs(
    directories: List[str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Load metrics from multiple directories.

    Args:
        directories: List of directory paths containing metrics files

    Returns:
        Tuple of (list of metrics dictionaries, list of run names)
    """
    metrics_list = []
    run_names = []

    for dir_path in directories:
        # Use the directory name as the run name
        run_name = os.path.basename(os.path.normpath(dir_path))

        # Check for advanced metrics
        advanced_metrics_path = os.path.join(dir_path, "advanced_metrics.json")
        standard_metrics_path = os.path.join(dir_path, "result_metrics.json")

        metrics_path = (
            advanced_metrics_path
            if os.path.exists(advanced_metrics_path)
            else standard_metrics_path
        )

        if not os.path.exists(metrics_path):
            print(f"Warning: No metrics file found in {dir_path}")
            continue

        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
                metrics_list.append(metrics)
                run_names.append(run_name)
                print(f"Loaded metrics from {metrics_path}")
        except Exception as e:
            print(f"Error loading metrics from {metrics_path}: {e}")

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
                exit_metrics["sample_percentages"][exit_point].append(
                    exit_stats[exit_point]["percentage"]
                )

                exit_metrics["accuracies"][exit_point].append(
                    exit_stats[exit_point]["avg_accuracy"]
                )

                exit_metrics["inference_times"][exit_point].append(
                    exit_stats[exit_point]["avg_inference_time_ms"]
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
    run_names: List[str],
    title_suffix: str = "",
) -> plt.Figure:
    """
    Create a scatter plot comparing accuracy vs latency for each exit point across runs.

    Args:
        exit_metrics: Dictionary with metrics organized by exit point
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
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]

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

        # Plot scatter points for this run
        scatter = ax.scatter(  # noqa F841
            run_times,
            run_accuracies,
            label=run_name,
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

    # Calculate the Pareto frontier (optimal trade-offs)
    if len(run_names) > 1:
        # Collect all data points
        all_times = []
        all_accs = []
        all_labels = []

        for i, run_name in enumerate(run_names):
            for j, exit_name in enumerate(exits):
                if i < len(accuracies[exit_name]):
                    acc = accuracies[exit_name][i]
                    time_ms = times[exit_name][i]

                    if acc > 0 and time_ms > 0:
                        all_times.append(time_ms)
                        all_accs.append(acc)
                        all_labels.append(f"{run_name} - {exit_name}")

        # Find the Pareto frontier
        pareto_times = []
        pareto_accs = []
        pareto_labels = []

        # Convert to numpy arrays for easier manipulation
        points = np.column_stack([all_times, all_accs])

        if len(points) > 0:
            # Sort by time (x-axis)
            sorted_indices = np.argsort(points[:, 0])
            sorted_points = points[sorted_indices]
            sorted_labels = [all_labels[i] for i in sorted_indices]

            # Start with the first point
            max_acc = sorted_points[0, 1]
            pareto_times.append(sorted_points[0, 0])
            pareto_accs.append(max_acc)
            pareto_labels.append(sorted_labels[0])

            # Iterate through the sorted points
            for i in range(1, len(sorted_points)):
                # If this point has higher accuracy, it's part of the Pareto frontier
                if sorted_points[i, 1] > max_acc:
                    max_acc = sorted_points[i, 1]
                    pareto_times.append(sorted_points[i, 0])
                    pareto_accs.append(max_acc)
                    pareto_labels.append(sorted_labels[i])

            # Plot the Pareto frontier
            if len(pareto_times) > 1:
                ax.plot(
                    pareto_times,
                    pareto_accs,
                    "-",
                    color="black",
                    alpha=0.7,
                    linewidth=2,
                    label="Pareto Frontier",
                )

                # Add shaded area under Pareto frontier
                ax.fill_between(pareto_times, 0, pareto_accs, color="gray", alpha=0.1)

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

    # Add legend
    ax.legend(fontsize=LEGEND_FONTSIZE)

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

    return parser.parse_args()


def main():
    """Main function to run the script"""
    args = parse_arguments()

    # Load metrics from all directories
    metrics_list, run_names = load_metrics_from_dirs(args.directories)

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
        exit_metrics, run_names, args.title_suffix
    )

    # Prompt user to save images
    sample_plot.show()
    scatter_plot.show()

    save_images = (
        input("Do you want to save the plots as images? (y/n): ").strip().lower()
    )
    if save_images == "y":
        # Create plots directory inside the first result directory
        plots_dir = os.path.join("results", "combined_benchmarks_plots")
        os.makedirs(plots_dir, exist_ok=True)

        sample_plot.savefig(os.path.join(plots_dir, "combined_sample_distribution.png"))
        scatter_plot.savefig(
            os.path.join(plots_dir, "combined_accuracy_latency_scatter.png")
        )
        print(f"Plots saved in '{plots_dir}'")


if __name__ == "__main__":
    main()
