import os
import argparse
import matplotlib.pyplot as plt
from utils import result_utils, logging_utils

logger = logging_utils.get_logger_ready(__name__)


def save_figure(fig, metrics_path, suffix=""):
    """Save figure as PNG in the same directory as the metrics file."""
    # Get base path and create new filename
    base_path = os.path.splitext(metrics_path)[0]
    save_path = f"{base_path}_plot{suffix}.png"

    # Save the figure
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    logger.info(f"Figure saved to: {save_path}")


def process_metrics_file(
    file_path, color_scheme=None, top_n_classes=10, save_figures=False
):
    """
    Process a metrics file to generate visualizations.

    Args:
        file_path: Path to the metrics JSON file
        color_scheme: Color scheme to use (if None, will prompt)
        top_n_classes: Number of top classes to include in class visualizations
        save_figures: Whether to automatically save figures
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return

    # Load metrics
    try:
        metrics = result_utils.load_metrics(file_path)
    except Exception as e:
        logger.error(f"Error loading metrics file: {e}")
        return

    # Get color scheme if not provided
    if color_scheme is None:
        color_scheme = result_utils.choose_color_scheme_cli()

    # Create title from filename
    title = (
        f"EEVIT Model Evaluation - {os.path.basename(file_path).removesuffix('.json')}"
    )

    # Generate plots
    exit_fig, class_accuracy_fig, class_speed_fig = result_utils.plot_metrics(
        metrics, title, color_scheme, top_n_classes
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
        save_figure(exit_fig, file_path, "_exits")
        if class_accuracy_fig:
            save_figure(class_accuracy_fig, file_path, "_classes_accuracy")
        if class_speed_fig:
            save_figure(class_speed_fig, file_path, "_classes_speed")

    # Wait for user to close figures
    if exit_fig or class_accuracy_fig or class_speed_fig:
        plt.show()


def get_argument_parser():
    parser = argparse.ArgumentParser(
        description="Process model evaluation metrics and create visualizations"
    )

    parser.add_argument("file_path", help="Path to the metrics JSON file")

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

    # Process the metrics file
    process_metrics_file(
        file_path=args.file_path,
        color_scheme=args.color_scheme,
        top_n_classes=args.top_classes,
        save_figures=args.save,
    )


if __name__ == "__main__":
    main()
