import argparse
import matplotlib.pyplot as plt

from utils import result_utils, logging_utils

logger = logging_utils.get_logger_ready(__name__)


def process_results_directory(
    results_dir,
    color_scheme=None,
    top_n_classes=10,
    save_figures=False,
    no_plots=False,
    auto_save=False,
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
    """
    # Load pre-computed metrics
    metrics = result_utils.load_metrics(results_dir)

    # Print detailed statistics to terminal
    result_utils.print_detailed_statistics(metrics)

    # If no_plots is True, skip all visualization steps
    if no_plots:
        return

    # Get color scheme if not provided
    if color_scheme is None:
        color_scheme = result_utils.choose_color_scheme_cli()

    # Generate plots
    exit_fig, class_accuracy_fig, class_speed_fig, confusion_fig = (
        result_utils.plot_metrics(metrics, results_dir, color_scheme, top_n_classes)
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

    if confusion_fig:
        plt.figure(confusion_fig.number)
        plt.show(block=False)

    # Ask if user wants to save figures
    if not save_figures and not auto_save:
        save_choice = input("Would you like to save the figures? (y/n): ").lower()
        save_figures = save_choice.startswith("y")

    if save_figures or auto_save:
        result_utils.save_figure(exit_fig, results_dir, "exit_statistics")
        if class_accuracy_fig:
            result_utils.save_figure(class_accuracy_fig, results_dir, "class_accuracy")
        if class_speed_fig:
            result_utils.save_figure(class_speed_fig, results_dir, "class_speed")
        if confusion_fig:
            result_utils.save_figure(confusion_fig, results_dir, "confusion_matrix")

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
    )


if __name__ == "__main__":
    main()
