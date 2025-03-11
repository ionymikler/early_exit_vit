import os
import argparse
import matplotlib.pyplot as plt
from utils import result_utils, logging_utils

logger = logging_utils.get_logger_ready(__name__)


def process_results_directory(results_dir, color_scheme=None, save_figures=False):
    """Process results directory to generate latency vs accuracy visualization."""
    # Verify the directory exists
    if not os.path.exists(results_dir) or not os.path.isdir(results_dir):
        logger.error(f"Results directory not found: {results_dir}")
        return

    # Check for metrics file
    metrics_file = os.path.join(results_dir, "result_metrics.json")
    if not os.path.exists(metrics_file):
        logger.error(f"Metrics file not found in directory: {metrics_file}")
        return

    try:
        metrics = result_utils.load_metrics_from_dir(results_dir)
    except Exception as e:
        logger.error(f"Error loading metrics file: {e}")
        return

    if color_scheme is None:
        color_scheme = result_utils.choose_color_scheme_cli()

    # Generate only the latency vs accuracy plot
    latency_accuracy_fig = result_utils.plot_latency_accuracy_scatter(
        metrics, results_dir, result_utils.COLOR_SCHEMES[color_scheme]
    )

    # Show plot
    plt.figure(latency_accuracy_fig.number)
    plt.show(block=False)

    if not save_figures:
        save_choice = input("Would you like to save the figure? (y/n): ").lower()
        save_figures = save_choice.startswith("y")

    if save_figures:
        result_utils.save_figure(
            latency_accuracy_fig, results_dir, "classes_latency_accuracy"
        )

    plt.show()


def get_argument_parser():
    parser = argparse.ArgumentParser(
        description="Generate latency vs accuracy visualization from metrics"
    )
    parser.add_argument(
        "results_dir",
        help="Path to the results directory containing result_metrics.json",
    )
    parser.add_argument(
        "--color-scheme",
        "-c",
        choices=list(result_utils.COLOR_SCHEMES.keys()),
        help="Color scheme to use for visualization",
    )
    parser.add_argument(
        "--save",
        "-s",
        action="store_true",
        help="Automatically save figure without prompting",
    )
    return parser


def main():
    args = get_argument_parser().parse_args()
    process_results_directory(
        results_dir=args.results_dir,
        color_scheme=args.color_scheme,
        save_figures=args.save,
    )


if __name__ == "__main__":
    main()
