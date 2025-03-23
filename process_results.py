import argparse
import matplotlib.pyplot as plt

from utils import logging_utils
from utils.results_processing import result_utils
from utils.results_processing import print_utils as result_print_utils

logger = logging_utils.get_logger_ready(__name__)


def process_results_directory(
    results_dir,
    color_scheme=None,
    top_n_classes=10,
    save_figures=False,
    no_plots=False,
    auto_save=False,
    class_stats=False,
):
    """
    Process a results directory to generate visualizations from pre-computed metrics.

    Args:
        results_dir: Path to the results directory
        color_scheme: Color scheme to use (if None, will prompt)
        top_n_classes: Number of top classes to include in class visualizations
        save_figures: Whether to automatically save figures
        no_plots: If True, skip generating plots and only print statistics
        auto_save: If True, automatically save figures without prompting
        class_stats: If True, only generate class statistics and visualizations
    """
    # Load pre-computed metrics
    metrics = result_utils.load_metrics(results_dir)

    # Print appropriate statistics based on the mode
    if class_stats:
        result_print_utils.print_class_statistics(metrics, top_n=top_n_classes // 2)
    else:
        result_print_utils.print_statistics(metrics, top_n=top_n_classes // 2)

    # If no_plots is True, skip all visualization steps
    if no_plots:
        return

    # Get color scheme if not provided
    if color_scheme is None:
        color_scheme = result_utils.choose_color_scheme_cli()

    if class_stats:
        # Check if class statistics exist in the metrics
        if "class_statistics" not in metrics or not metrics["class_statistics"]:
            raise ValueError("Class statistics not found in metrics file")

        # Generate and show class-related visualizations only

        # 1. Latency vs accuracy scatter plot
        latency_accuracy_fig = result_utils.plot_latency_accuracy_scatter(
            metrics,
            results_dir,
            result_utils.COLOR_SCHEMES_BACKEND[color_scheme],
            top_n_classes,
        )
        plt.figure(latency_accuracy_fig.number)
        plt.show(block=False)

        # 2. Confusion matrix - using all classes with class IDs rather than names
        confusion_fig = result_utils.plot_confusion_matrix(
            metrics,
            title="Class Confusion Matrix",
            normalize=True,
            top_n_classes=top_n_classes // 2,
            include_accuracy=False,
        )
        plt.figure(confusion_fig.number)
        plt.show(block=False)

        # 3. Combined class statistics (both accuracy and speed)
        class_combined_fig = result_utils.plot_class_statistics_combined(
            metrics,
            "Class Statistics",
            result_utils.COLOR_SCHEMES_BACKEND[color_scheme],
            top_n_classes=top_n_classes,
        )
        plt.figure(class_combined_fig.number)
        plt.show(block=False)

        # 4. Exit distribution for top classes
        top_class_exit_fig = result_utils.plot_top_class_exit_distribution(
            metrics,
            "Class Statistics",
            result_utils.COLOR_SCHEMES_BACKEND[color_scheme],
        )
        plt.figure(top_class_exit_fig.number)
        plt.show(block=False)

        # Ask if user wants to save figures
        if not save_figures and not auto_save:
            save_choice = input("Would you like to save the figures? (y/n): ").lower()
            save_figures = save_choice.startswith("y")

        if save_figures or auto_save:
            result_utils.save_figure(
                latency_accuracy_fig, results_dir, "classes_latency_accuracy"
            )
            result_utils.save_figure(confusion_fig, results_dir, "confusion_matrix")
            result_utils.save_figure(
                class_combined_fig, results_dir, "class_combined_performance"
            )
            result_utils.save_figure(
                top_class_exit_fig, results_dir, "top_class_exit_distribution"
            )

    else:
        # Standard mode - Generate the original plots as before
        exit_fig, class_accuracy_fig, class_speed_fig, confusion_fig = (
            result_utils.plot_metrics(metrics, results_dir, color_scheme, top_n_classes)
        )

        # Replace the separate class_accuracy_fig and class_speed_fig with a combined one
        class_combined_fig = result_utils.plot_class_statistics_combined(
            metrics,
            "Class Statistics",
            result_utils.COLOR_SCHEMES_BACKEND[color_scheme],
            top_n_classes=top_n_classes,
        )

        # Add top class exit distribution visualization
        top_class_exit_fig = result_utils.plot_top_class_exit_distribution(
            metrics,
            "Class Statistics",
            result_utils.COLOR_SCHEMES_BACKEND[color_scheme],
        )

        # Show plots
        plt.figure(exit_fig.number)
        plt.show(block=False)

        plt.figure(class_combined_fig.number)
        plt.show(block=False)

        if confusion_fig:
            plt.figure(confusion_fig.number)
            plt.show(block=False)

        plt.figure(top_class_exit_fig.number)
        plt.show(block=False)

        # Ask if user wants to save figures
        if not save_figures and not auto_save:
            save_choice = input("Would you like to save the figures? (y/n): ").lower()
            save_figures = save_choice.startswith("y")

        if save_figures or auto_save:
            result_utils.save_figure(exit_fig, results_dir, "exit_statistics")
            result_utils.save_figure(
                class_combined_fig, results_dir, "class_combined_performance"
            )
            if confusion_fig:
                result_utils.save_figure(confusion_fig, results_dir, "confusion_matrix")
            result_utils.save_figure(
                top_class_exit_fig, results_dir, "top_class_exit_distribution"
            )

    # Wait for user to close figures
    plt.show()


def _get_argument_parser():
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

    parser.add_argument(
        "--class-stats",
        action="store_true",
        help="Generate only class-related statistics and visualizations",
    )

    return parser


def main():
    args = _get_argument_parser().parse_args()

    # Process the results directory
    process_results_directory(
        results_dir=args.results_dir,
        color_scheme=args.color_scheme,
        top_n_classes=args.top_classes,
        save_figures=args.save,
        no_plots=args.no_plots,
        auto_save=args.auto_save,
        class_stats=args.class_stats,
    )


if __name__ == "__main__":
    main()
