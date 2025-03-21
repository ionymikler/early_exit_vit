import os
import json
import argparse
import matplotlib.pyplot as plt
from utils import result_utils, logging_utils
from colorama import Fore, Style

logger = logging_utils.get_logger_ready(__name__)


def calculate_advanced_metrics(metrics):
    """
    Calculate advanced metrics from the evaluation results including speedup metric.

    Args:
        metrics: Dictionary containing original evaluation metrics

    Returns:
        Dictionary containing calculated advanced metrics
    """
    advanced_metrics = {
        "overall_accuracy": metrics.get("overall_accuracy", 0),
        "total_samples": metrics.get("total_samples", 0),
        "exit_statistics": {},
    }

    # Find the total number of layers from model metadata or config
    # For now, assuming the final layer is the max layer in the model
    all_exits = metrics.get("exit_statistics", {})
    max_layer = 0
    for exit_key, exit_data in all_exits.items():
        if exit_key == "final":
            # Assuming final exit represents the last layer
            # In practice, this should be extracted from the model config
            max_layer = metrics.get("model_config", {}).get(
                "num_layers_transformer", 12
            )
            break

    if max_layer == 0:
        # Default to 12 if we can't determine it
        max_layer = 12
        logger.warning("Could not determine maximum layer count, using default of 12")

    # Extract exit statistics for calculations
    total_samples = metrics.get("total_samples", 0)
    weighted_sum = 0
    total_computation = total_samples * max_layer

    # Calculate the weighted sum of samples by exit layer
    for exit_key, exit_data in all_exits.items():
        # Convert exit index to layer position (adding 1 to convert from 0-indexed to 1-indexed)
        exit_layer = (
            max_layer if exit_key == "final" else int(exit_key.split("_")[1]) + 1
        )

        # Add additional calculated metrics for this exit
        advanced_metrics["exit_statistics"][exit_key] = {
            "count": exit_data.get("count", 0),
            "avg_accuracy": exit_data.get("accuracy", 0),
            "avg_inference_time_ms": exit_data.get("avg_inference_time_ms", 0),
            "percentage": exit_data.get("percentage_samples", 0),
            "layer_position": exit_layer,  # 1-indexed layer position
            "layer_index": max_layer - 1
            if exit_key == "final"
            else int(exit_key.split("_")[1]),  # 0-indexed
        }

        # Calculate the weighted sum for speedup metric
        weighted_sum += (
            exit_layer * advanced_metrics["exit_statistics"][exit_key]["count"]
        )

    # Calculate the speedup metric as (total samples * max layers) / weighted sum
    if weighted_sum > 0:
        speedup = total_computation / weighted_sum
        expected_saving = 1 - (weighted_sum / total_computation)
    else:
        speedup = 1.0
        expected_saving = 0.0

    # Add to advanced metrics
    advanced_metrics["speedup"] = round(speedup, 4)
    advanced_metrics["expected_saving"] = round(
        expected_saving * 100, 2
    )  # as percentage
    advanced_metrics["total_computation"] = total_computation
    advanced_metrics["weighted_computation"] = weighted_sum
    advanced_metrics["max_layer"] = max_layer

    # Add per-class metrics if available
    if "class_statistics" in metrics:
        advanced_metrics["class_statistics"] = {}

        for class_id, class_data in metrics["class_statistics"].items():
            advanced_metrics["class_statistics"][class_id] = {
                "name": class_data.get("name", f"Class {class_id}"),
                "accuracy": class_data.get("accuracy", 0),
                "avg_inference_time_ms": class_data.get("avg_inference_time_ms", 0),
                "avg_exit_layer": class_data.get("avg_exit_layer", 0),
                "mode_exit_layer": class_data.get("mode_exit_layer", 0),
            }

    return advanced_metrics


def print_detailed_statistics(metrics, advanced_metrics):
    """
    Print detailed statistics to the terminal for each exit and the whole model.

    Args:
        metrics: Original metrics dictionary
        advanced_metrics: Advanced metrics dictionary with calculated values
    """
    # Define some color formatting for better terminal output
    HEADER = Fore.CYAN
    VALUE = Fore.GREEN
    RESET = Style.RESET_ALL
    BOLD = Style.BRIGHT

    # Print a header separator
    print("\n" + "=" * 80)
    print(f"{HEADER}{BOLD}EEVIT MODEL EVALUATION RESULTS{RESET}")
    print("=" * 80)

    # 1. Overall model statistics
    print(f"\n{HEADER}{BOLD}OVERALL MODEL STATISTICS:{RESET}")
    print(f"  - Overall Accuracy: {VALUE}{metrics['overall_accuracy']:.2f}%{RESET}")
    print(f"  - Total Samples: {VALUE}{metrics['total_samples']}{RESET}")
    print(f"  - Speedup Factor: {VALUE}{advanced_metrics['speedup']:.2f}x{RESET}")
    print(
        f"  - Computation Saved: {VALUE}{advanced_metrics['expected_saving']:.2f}%{RESET}"
    )

    # 2. Exit point statistics
    print(f"\n{HEADER}{BOLD}EXIT POINT STATISTICS:{RESET}")
    print("-" * 80)
    print(
        f"{'Exit Point':<15} {'Samples':<10} {'Percentage':<12} {'Accuracy':<12} {'Latency (ms)':<15} {'Exit Layer':<10}"
    )
    print("-" * 80)

    # Sort exit points for consistent display
    def exit_sort_key(exit_name):
        if exit_name == "final":
            return float("inf")  # Final exit should appear last
        else:
            parts = exit_name.split("_")
            return int(parts[1]) if len(parts) > 1 else float("inf")

    exit_stats = metrics.get("exit_statistics", {})
    for exit_key in sorted(exit_stats.keys(), key=exit_sort_key):
        stats = exit_stats[exit_key]
        adv_stats = advanced_metrics["exit_statistics"].get(exit_key, {})

        # Format exit name for display
        if exit_key == "final":
            exit_name = "Final Layer"
        else:
            parts = exit_key.split("_")
            exit_name = f"Exit {parts[1]}" if len(parts) > 1 else exit_key

        # Get statistics
        sample_count = stats.get("count", 0)
        percentage = stats.get("percentage_samples", 0)
        accuracy = stats.get("accuracy", 0)
        latency = stats.get("avg_inference_time_ms", 0)
        std_latency = stats.get("std_inference_time_ms", 0)
        layer_pos = adv_stats.get("layer_position", "N/A")

        # Print formatted row
        print(
            f"{exit_name:<15} {sample_count:<10} {percentage:>8.1f}%     {accuracy:>8.2f}%     {latency:>6.2f} ± {std_latency:<6.2f} {layer_pos:<10}"
        )

    print("-" * 80)

    # 3. Print sample distribution summary
    print(f"\n{HEADER}{BOLD}SAMPLE DISTRIBUTION SUMMARY:{RESET}")
    early_exit_samples = metrics["total_samples"] - (
        exit_stats.get("final", {}).get("count", 0)
    )
    early_exit_percentage = (
        (early_exit_samples / metrics["total_samples"]) * 100
        if metrics["total_samples"] > 0
        else 0
    )

    print(
        f"  - Early Exit Samples: {VALUE}{early_exit_samples}{RESET} ({early_exit_percentage:.1f}% of total)"
    )
    print(
        f"  - Final Layer Samples: {VALUE}{exit_stats.get('final', {}).get('count', 0)}{RESET} ({100 - early_exit_percentage:.1f}% of total)"
    )

    # 4. Classes overview (just summary, not per-class)
    if "class_statistics" in metrics:
        class_stats = metrics["class_statistics"]
        print(f"\n{HEADER}{BOLD}CLASS STATISTICS SUMMARY:{RESET}")
        print(f"  - Total Classes: {VALUE}{len(class_stats)}{RESET}")

        # Find classes with highest and lowest accuracy
        if class_stats:
            sorted_by_acc = sorted(
                class_stats.items(), key=lambda x: x[1].get("accuracy", 0), reverse=True
            )
            highest_acc_class = sorted_by_acc[0]
            lowest_acc_class = sorted_by_acc[-1]

            print(
                f"  - Highest Accuracy Class: {VALUE}{highest_acc_class[1]['name']}{RESET} ({highest_acc_class[1]['accuracy']:.2f}%)"
            )
            print(
                f"  - Lowest Accuracy Class: {VALUE}{lowest_acc_class[1]['name']}{RESET} ({lowest_acc_class[1]['accuracy']:.2f}%)"
            )

            # Find classes with earliest and latest exits on average
            sorted_by_exit = sorted(
                class_stats.items(), key=lambda x: x[1].get("avg_exit_layer", 0)
            )
            earliest_exit_class = sorted_by_exit[0]
            latest_exit_class = sorted_by_exit[-1]

            print(
                f"  - Earliest Average Exit: {VALUE}{earliest_exit_class[1]['name']}{RESET} (Layer {earliest_exit_class[1]['avg_exit_layer']:.2f})"
            )
            print(
                f"  - Latest Average Exit: {VALUE}{latest_exit_class[1]['name']}{RESET} (Layer {latest_exit_class[1]['avg_exit_layer']:.2f})"
            )

    # 5. Performance summary
    print(f"\n{HEADER}{BOLD}PERFORMANCE SUMMARY:{RESET}")
    print(
        f"  - Average Inference Time: {VALUE}{advanced_metrics.get('avg_inference_time_ms', 'N/A'):.2f} ms{RESET}"
    )
    print(
        f"  - Accuracy vs Speed Tradeoff: {VALUE}{advanced_metrics.get('accuracy_speed_tradeoff', 'N/A')}{RESET}"
    )

    print("\n" + "=" * 80)


def process_results_directory(
    results_dir,
    color_scheme=None,
    top_n_classes=10,
    save_figures=False,
    no_plots=False,
    auto_save=False,
):
    """
    Process a results directory to generate visualizations and calculate advanced metrics.

    Args:
        results_dir: Path to the results directory
        color_scheme: Color scheme to use (if None, will prompt)
        top_n_classes: Number of top classes to include in class visualizations
        save_figures: Whether to automatically save figures
        no_plots: If True, skip generating plots and only print statistics
        auto_save: If True, automatically save advanced metrics without prompting
    """
    # Verify the directory exists
    if not os.path.exists(results_dir) or not os.path.isdir(results_dir):
        logger.error(f"Results directory not found: {results_dir}")
        return

    # Check for metrics file
    metrics_file = os.path.join(results_dir, "result_metrics.json")
    if not os.path.exists(metrics_file):
        logger.error(f"Metrics file not found in directory: {metrics_file}")
        return

    # Load metrics
    try:
        metrics = result_utils.load_metrics_from_dir(results_dir)
    except Exception as e:
        logger.error(f"Error loading metrics file: {e}")
        return

    # Calculate advanced metrics
    advanced_metrics = calculate_advanced_metrics(metrics)

    # Calculate average inference time for the whole model
    total_time_weighted = 0
    total_samples = metrics.get("total_samples", 0)

    for exit_key, stats in metrics.get("exit_statistics", {}).items():
        count = stats.get("count", 0)
        avg_time = stats.get("avg_inference_time_ms", 0)
        total_time_weighted += count * avg_time

    if total_samples > 0:
        advanced_metrics["avg_inference_time_ms"] = total_time_weighted / total_samples

    # Calculate additional metrics
    if advanced_metrics.get("avg_inference_time_ms", 0) > 0:
        advanced_metrics["accuracy_speed_tradeoff"] = round(
            metrics.get("overall_accuracy", 0)
            / advanced_metrics["avg_inference_time_ms"],
            3,
        )

    # Print detailed statistics to terminal
    print_detailed_statistics(metrics, advanced_metrics)

    # Ask if user wants to save advanced metrics
    save_metrics = True  # Default behavior is to save
    if not auto_save:  # Use new auto_save flag
        save_choice = input(
            "Would you like to save the advanced metrics to JSON? (y/n): "
        ).lower()
        save_metrics = save_choice.startswith("y")

    # Save advanced metrics to a new JSON file if requested
    if save_metrics:
        advanced_metrics_file = os.path.join(results_dir, "advanced_metrics.json")
        try:
            with open(advanced_metrics_file, "w") as f:
                json.dump(advanced_metrics, f, indent=4)
            logger.info(f"Advanced metrics saved to {advanced_metrics_file}")
        except Exception as e:
            logger.error(f"Error saving advanced metrics: {e}")

    # If no_plots is True, skip all visualization steps
    if no_plots:
        logger.info("Skipping plot generation (--no-plots flag used)")
        return

    # Get color scheme if not provided
    if color_scheme is None:
        color_scheme = result_utils.choose_color_scheme_cli()

    # Generate plots
    exit_fig, class_accuracy_fig, class_speed_fig = result_utils.plot_metrics(
        metrics, results_dir, color_scheme, top_n_classes
    )

    # Show plots
    plt.figure(exit_fig.number)
    plt.show(block=False)

    if class_accuracy_fig:
        plt.figure(class_accuracy_fig.number)
        plt.show(block=False)

    if class_speed_fig:
        plt.figure(class_speed_fig.number)
        plt.show(block=False)

    # Ask if user wants to save figures
    if not save_figures:
        save_choice = input("Would you like to save the figures? (y/n): ").lower()
        save_figures = save_choice.startswith("y")

    if save_figures:
        result_utils.save_figure(exit_fig, results_dir, "exit_statistics")
        if class_accuracy_fig:
            result_utils.save_figure(class_accuracy_fig, results_dir, "class_accuracy")
        if class_speed_fig:
            result_utils.save_figure(class_speed_fig, results_dir, "class_speed")

    # Wait for user to close figures
    if exit_fig or class_accuracy_fig or class_speed_fig:
        plt.show()


def get_argument_parser():
    parser = argparse.ArgumentParser(
        description="Process model evaluation metrics and create visualizations"
    )

    parser.add_argument(
        "results_dir",
        help="Path to the results directory containing result_metrics.json",
    )

    parser.add_argument(
        "--color-scheme",
        "-c",
        choices=list(result_utils.COLOR_SCHEMES_BACKEND.keys()),
        help="Color scheme to use for visualizations",
    )

    parser.add_argument(
        "--top-classes",
        "-n",
        type=int,
        default=10,
        help="Number of top classes to display in class visualizations",
    )

    parser.add_argument(
        "--save",
        "-s",
        action="store_true",
        help="Automatically save figures without prompting",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots, only print statistics",
    )

    parser.add_argument(
        "--auto-save",
        "-a",
        action="store_true",
        help="Automatically save all outputs (figures and metrics) without prompting",
    )

    return parser


def main():
    args = get_argument_parser().parse_args()

    # Process the results directory
    process_results_directory(
        results_dir=args.results_dir,
        color_scheme=args.color_scheme,
        top_n_classes=args.top_classes,
        save_figures=args.save,
        no_plots=args.no_plots,
        auto_save=args.auto_save,
    )


if __name__ == "__main__":
    main()
