import os
import argparse
import matplotlib.pyplot as plt
from utils import result_utils, logging_utils

logger = logging_utils.get_logger_ready(__name__)


def save_figure(fig, metrics_path, suffix=""):
    """Save figure as PNG in the same directory as the metrics file."""
    base_path = os.path.splitext(metrics_path)[0]
    save_path = f"{base_path}_plot{suffix}.png"

    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    logger.info(f"Figure saved to: {save_path}")


def process_metrics_file(file_path, color_scheme=None, save_figures=False):
    """Process metrics file to generate latency vs accuracy visualization."""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return

    try:
        metrics = result_utils.load_metrics(file_path)
    except Exception as e:
        logger.error(f"Error loading metrics file: {e}")
        return

    if color_scheme is None:
        color_scheme = result_utils.choose_color_scheme_cli()

    title = (
        f"EEVIT Model Evaluation - {os.path.basename(file_path).removesuffix('.json')}"
    )

    # Generate only the latency vs accuracy plot
    latency_accuracy_fig = result_utils.plot_latency_accuracy_scatter(
        metrics, title, result_utils.COLOR_SCHEMES[color_scheme]
    )

    # Show plot
    plt.figure(latency_accuracy_fig.number)
    plt.show(block=False)

    if not save_figures:
        save_choice = input("Would you like to save the figure? (y/n): ").lower()
        save_figures = save_choice.startswith("y")

    if save_figures:
        save_figure(latency_accuracy_fig, file_path, "_classes_latency_accuracy")

    plt.show()


def get_argument_parser():
    parser = argparse.ArgumentParser(
        description="Generate latency vs accuracy visualization from metrics"
    )
    parser.add_argument("file_path", help="Path to the metrics JSON file")
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
    process_metrics_file(
        file_path=args.file_path, color_scheme=args.color_scheme, save_figures=args.save
    )


if __name__ == "__main__":
    main()
