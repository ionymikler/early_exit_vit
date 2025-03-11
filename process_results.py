import os
import json
import argparse
import matplotlib.pyplot as plt
from utils import result_utils, logging_utils

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
        sample_count = exit_data.get("count", 0)
        weighted_sum += exit_layer * sample_count

        # Add this data to advanced metrics
        advanced_metrics["exit_statistics"][exit_key] = {
            "layer_position": exit_layer,  # 1-indexed layer position
            "layer_index": max_layer - 1
            if exit_key == "final"
            else int(exit_key.split("_")[1]),  # 0-indexed
            "count": sample_count,
            "percentage": exit_data.get("percentage_samples", 0),
        }

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


def process_results_directory(
    results_dir, color_scheme=None, top_n_classes=10, save_figures=False
):
    """
    Process a results directory to generate visualizations and calculate advanced metrics.

    Args:
        results_dir: Path to the results directory
        color_scheme: Color scheme to use (if None, will prompt)
        top_n_classes: Number of top classes to include in class visualizations
        save_figures: Whether to automatically save figures
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

    # Save advanced metrics to a new JSON file
    advanced_metrics_file = os.path.join(results_dir, "advanced_metrics.json")
    try:
        with open(advanced_metrics_file, "w") as f:
            json.dump(advanced_metrics, f, indent=4)
        logger.info(f"Advanced metrics saved to {advanced_metrics_file}")
    except Exception as e:
        logger.error(f"Error saving advanced metrics: {e}")

    # Get color scheme if not provided
    if color_scheme is None:
        color_scheme = result_utils.choose_color_scheme_cli()

    # Generate plots
    exit_fig, class_accuracy_fig, class_speed_fig = result_utils.plot_metrics(
        metrics, results_dir, color_scheme, top_n_classes
    )

    # Print summary of advanced metrics
    logger.info("\nAdvanced Metrics Summary:")
    logger.info(f"Overall Accuracy: {advanced_metrics['overall_accuracy']:.2f}%")
    logger.info(f"Speedup Factor: {advanced_metrics['speedup']:.2f}x")
    logger.info(f"Expected Saving: {advanced_metrics['expected_saving']:.2f}%")

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
        choices=list(result_utils.COLOR_SCHEMES.keys()),
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

    return parser


def main():
    args = get_argument_parser().parse_args()

    # Process the results directory
    process_results_directory(
        results_dir=args.results_dir,
        color_scheme=args.color_scheme,
        top_n_classes=args.top_classes,
        save_figures=args.save,
    )


if __name__ == "__main__":
    main()
