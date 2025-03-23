import json
import yaml
import os
from pathlib import Path
from torch.profiler import profile as Profile
from datetime import datetime
from typing import Dict, Any

from . import plot_utils
from ..logging_utils import get_logger_ready

logger = get_logger_ready(__name__)

# Expose color schemes from plot_utils
COLOR_SCHEMES_BACKEND = plot_utils.COLOR_SCHEMES_BACKEND


def save_metadata(results_dir: str, model_type: str, args, config: dict = None):
    """
    Save evaluation metadata to a YAML file.

    Args:
        results_dir: Directory to save metadata to
        model_type: Type of model ('pytorch' or 'onnx')
        args: Command-line arguments used for evaluation
        config: Configuration dictionary
    """
    if args is None:
        return

    try:
        # Convert args to dictionary
        args_dict = vars(args)

        # Create metadata object
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        metadata = {
            "timestamp": timestamp,
            "model_type": model_type,
            "args": args_dict,
            "config": config,
        }

        metadata_file = f"{results_dir}/metadata.yaml"
        with open(metadata_file, "w") as f:
            yaml.dump(metadata, f, default_flow_style=False)

        logger.info(f"Metadata saved to {metadata_file}")
    except Exception as e:
        logger.error(f"Failed to save metadata: {e}")


def make_results_dir(model_type: str, profiling: bool, suffix: str = None) -> str:
    """
    Create a subdirectory for results with datetime and model type.
    If directory already exists, append a counter to create a unique one.

    Args:
        model_type: Type of model ('pytorch' or 'onnx')
        profiling: Whether profiling is enabled
        suffix: Optional suffix to use instead of timestamp

    Returns:
        Path to the results directory
    """
    base_dir = f"results/{model_type}"
    if profiling:
        base_dir += "_profiling"

    # Use provided suffix or generate timestamp
    if suffix:
        base_results_dir = f"{base_dir}_{suffix}"
    else:
        base_results_dir = f'{base_dir}_{datetime.now().strftime("%y%m%d_%H%M%S")}'

    # Create a unique directory if the initial one already exists
    results_dir = base_results_dir
    counter = 1

    while os.path.exists(results_dir):
        results_dir = f"{base_results_dir}_{counter}"
        counter += 1

    # Create the directory
    os.makedirs(results_dir)
    logger.info(f"Created results directory: {results_dir}")

    return results_dir


def save_metrics(metrics, results_dir: str):
    """
    Save metrics to JSON file with a standardized name.

    Args:
        metrics: Dictionary containing evaluation metrics
        results_dir: Directory to save results to
        files_prefix: Optional prefix for the output files (not used for standard metrics file)
    """
    # Save metrics to JSON file with standardized name
    metrics_file = f"{results_dir}/result_metrics.json"

    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"Metrics saved to {metrics_file}")


def save_figure(fig, results_dir: str, metric_name: str):
    """
    Save a figure to the results directory with standardized naming.

    Delegates to the plot_utils module.

    Args:
        fig: The matplotlib figure to save
        results_dir: Path to the results directory
        metric_name: Name of the metric (will be used for filename)
    """
    plot_utils.save_figure(fig, results_dir, metric_name)


def save_pytorch_profiler_output(profile: Profile, results_dir: str):
    """
    Save PyTorch profiler output to results directory.

    Args:
        profile: PyTorch profiler instance
        results_dir: Directory to save profiler output to
    """
    base_output_path = f"{results_dir}/pytorch_profiler_trace"
    output_path = f"{base_output_path}_1.json"
    counter = 2

    # Check if file exists and find the minimum available number
    while os.path.exists(output_path):
        output_path = f"{base_output_path}_{counter}.json"
        counter += 1

    profile.export_chrome_trace(output_path)
    logger.debug(f"PyTorch profiler output saved to {output_path}")


def load_metrics(results_dir: str) -> Dict[str, Any]:
    """
    Load metrics from the standard metrics file in a results directory.

    Args:
        results_dir: Path to results directory

    Returns:
        Dictionary containing metrics data

    Raises:
        FileNotFoundError: If the results directory or metrics file is not found
        ValueError: If there is an error loading the metrics file
    """
    results_path = Path(results_dir)
    if not results_path.exists() or not results_path.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    metrics_file = results_path / "result_metrics.json"
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found in directory: {metrics_file}")

    try:
        with open(metrics_file, "r") as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Error loading metrics file: {e}")


def get_results_info(results_dir: str):
    """
    Extract model type and results identifier from the metadata YAML file in a results directory.

    Delegates to the plot_utils module.

    Args:
        results_dir: Path to results directory

    Returns:
        Tuple of (model_type, results_identifier)
    """
    return plot_utils.get_results_info(results_dir)


def plot_metrics(metrics, results_dir: str, color_scheme_key="teal", top_n_classes=10):
    """
    Plot metrics for model evaluation, including both exit statistics and class statistics.

    Delegates to the plot_utils module.

    Args:
        metrics: Dictionary containing evaluation metrics
        results_dir: Directory where results are stored
        color_scheme: Color scheme to use
        top_n_classes: Number of top classes to highlight in class-specific plots

    Returns:
        Tuple of figures: (exit_stats_figure, class_accuracy_stats_figure, class_speed_stats_figure)
    """
    return plot_utils.plot_metrics(
        metrics, results_dir, color_scheme_key, top_n_classes
    )


def plot_exit_statistics(metrics, title: str, colors):
    """
    Plot exit-related statistics including column charts for accuracy and inference time.

    Delegates to the plot_utils module.
    """
    return plot_utils.plot_exit_statistics(metrics, title, colors)


def plot_class_statistics_combined(metrics, title, colors, top_n_classes=10):
    """
    Plot class-specific statistics showing top and bottom performers by both accuracy and speed in one plot.

    Delegates to the plot_utils module.
    """
    return plot_utils.plot_class_statistics_combined(
        metrics, title, colors, top_n_classes
    )


def plot_latency_accuracy_scatter(metrics, results_dir, colors, top_n_classes=10):
    """
    Create a scatter plot showing the relationship between accuracy and latency for top/bottom classes.

    Delegates to the plot_utils module.
    """
    return plot_utils.plot_latency_accuracy_scatter(
        metrics, results_dir, colors, top_n_classes
    )


def plot_confusion_matrix(
    metrics, title, normalize=True, top_n_classes=None, include_accuracy=True
):
    """
    Create a visualization of the confusion matrix.

    Delegates to the plot_utils module.
    """
    return plot_utils.plot_confusion_matrix(
        metrics, title, normalize, top_n_classes, include_accuracy
    )


def plot_top_class_exit_distribution(metrics, title, colors):
    """
    Plot the exit distribution for the top fastest and most accurate classes.

    Delegates to the plot_utils module.
    """
    return plot_utils.plot_top_class_exit_distribution(metrics, title, colors)


def choose_color_scheme_cli():
    """
    Command-line interface for selecting color scheme.

    Delegates to the plot_utils module.
    """
    return plot_utils.choose_color_scheme_cli()
