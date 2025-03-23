"""
Functions for calculating evaluation metrics from collected data.
"""

import numpy as np
from typing import Dict, Any
from .structures import EvaluationResults


def calculate_exit_statistics(
    evaluation_results: EvaluationResults,
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate statistics for each exit point from collected data.

    Args:
        evaluation_results: EvaluationResults object containing collected data

    Returns:
        Dictionary containing processed exit statistics
    """
    exit_stats = {}
    total_samples = evaluation_results.total_samples

    for exit_key, collector in evaluation_results.exit_collectors.items():
        if not collector.count > 0:
            continue

        # Calculate exit accuracy
        exit_accuracy = 100 * collector.correct / collector.count

        # Calculate additional statistics
        exit_stats[exit_key] = {
            "count": collector.count,
            "accuracy": round(exit_accuracy, 4),
            "percentage_samples": round(100 * collector.count / total_samples, 2),
            "average_confidence": round(np.mean(collector.confidences), 4),
            # Add inference time statistics
            "avg_inference_time_ms": round(np.mean(collector.inference_times), 4),
            "std_inference_time_ms": round(np.std(collector.inference_times), 4),
            "min_inference_time_ms": round(np.min(collector.inference_times), 4),
            "max_inference_time_ms": round(np.max(collector.inference_times), 4),
            # Store raw distributions for plotting
            "inference_time_values": collector.inference_times,
            "confidence_values": collector.confidences,
            "accuracy_values": collector.accuracies_per_sample,
            "batch_accuracy_values": collector.batch_accuracies,
        }

    return exit_stats


def calculate_class_statistics(
    evaluation_results: EvaluationResults, test_loader, config
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate per-class statistics from collected data.

    Args:
        evaluation_results: EvaluationResults object containing collected data
        test_loader: DataLoader containing dataset information for class names
        config: Model configuration

    Returns:
        Dictionary containing processed class-specific metrics
    """
    class_stats = {}
    max_layer = config.get("model", {}).get("num_layers_transformer", 12)

    for class_id, collector in evaluation_results.class_collectors.items():
        if collector.count == 0:
            continue  # Skip classes with no examples

        try:
            from utils import dataset_utils

            class_name = dataset_utils.get_label_name(test_loader.dataset, class_id)
        except (AttributeError, ValueError):
            class_name = f"Class {class_id}"

        # Calculate accuracy
        accuracy = 100 * collector.correct / collector.count

        # Calculate exit layer statistics from exit_by_layer
        exit_distribution = collector.exit_by_layer
        exit_values = []
        for exit_key, count in exit_distribution.items():
            # Convert exit_key to numeric value for calculations
            if exit_key == "final":
                value = max_layer
            else:
                value = int(exit_key.split("_")[1])
            # Add this exit value to the list count times
            exit_values.extend([value] * count)

        exit_values = np.array(exit_values)
        avg_exit = np.mean(exit_values).item() if len(exit_values) > 0 else 0
        mode_exit = (
            np.bincount(exit_values.astype(int)).argmax() if len(exit_values) > 0 else 0
        )
        std_exit = np.std(exit_values).item() if len(exit_values) > 0 else 0

        # Calculate statistics for inference times and confidences
        inference_times = np.array(collector.inference_times)
        confidences = np.array(collector.confidences)

        class_stats[class_id] = {
            "name": class_name,
            "count": collector.count,
            "accuracy": round(accuracy, 4),
            # Exit layer statistics
            "avg_exit_layer": round(avg_exit, 4),
            "std_exit_layer": round(std_exit, 4),
            "mode_exit_layer": int(mode_exit),
            "exit_distribution": exit_distribution,
            # Inference time statistics
            "avg_inference_time_ms": round(np.mean(inference_times).item(), 4),
            "std_inference_time_ms": round(np.std(inference_times).item(), 4),
            "min_inference_time_ms": round(np.min(inference_times).item(), 4)
            if len(inference_times) > 0
            else 0,
            "max_inference_time_ms": round(np.max(inference_times).item(), 4)
            if len(inference_times) > 0
            else 0,
            # Confidence statistics
            "avg_confidence": round(np.mean(confidences).item(), 4),
            "std_confidence": round(np.std(confidences).item(), 4),
            "min_confidence": round(np.min(confidences).item(), 4)
            if len(confidences) > 0
            else 0,
            "max_confidence": round(np.max(confidences).item(), 4)
            if len(confidences) > 0
            else 0,
            # Raw distributions for visualization
            "inference_times": inference_times.tolist(),
            "confidences": confidences.tolist(),
            "accuracies": collector.accuracies,
            "exit_values": exit_values.tolist(),
        }

        # Add confusion matrix data if needed
        if collector.predicted_as:
            class_stats[class_id]["predicted_as"] = collector.predicted_as

    return class_stats


def calculate_confusion_matrix(evaluation_results: EvaluationResults) -> np.ndarray:
    """
    Generate a confusion matrix from the collected class prediction data.

    Args:
        evaluation_results: EvaluationResults object containing collected data

    Returns:
        2D numpy array representing the confusion matrix
    """
    num_classes = evaluation_results.num_classes
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for class_id, collector in evaluation_results.class_collectors.items():
        for predicted_class, count in collector.predicted_as.items():
            confusion_matrix[class_id, predicted_class] = count

    return confusion_matrix


def calculate_advanced_metrics(metrics):
    """
    Calculate advanced metrics from the evaluation results including speedup metric.

    Args:
        metrics: Dictionary containing original evaluation metrics

    Returns:
        Dictionary containing calculated advanced metrics
    """
    advanced_metrics = {
        "overall_accuracy": metrics["overall_accuracy"],
        "total_samples": metrics["total_samples"],
        "average_inference_time_ms": metrics["avg_inference_time_ms"],
        "exit_statistics": {},
    }

    # Find the total number of layers from model metadata or config
    all_exits = metrics.get("exit_statistics", {})
    max_layer = 0
    for exit_key, exit_data in all_exits.items():
        if exit_key == "final":
            # Assuming final exit represents the last layer
            max_layer = metrics.get("model_config", {}).get(
                "num_layers_transformer", 12
            )
            break

    if max_layer == 0:
        # Default to 12 if we can't determine it
        max_layer = 12

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


def build_final_metrics(
    evaluation_results: EvaluationResults, test_loader, config
) -> Dict[str, Any]:
    """
    Build the final metrics dictionary from all collected data.

    Args:
        evaluation_results: EvaluationResults object with all collected data
        test_loader: DataLoader containing dataset information
        config: Model configuration

    Returns:
        Complete metrics dictionary with all calculated statistics
    """
    # TODO: This logic should be also possible to do with the results file
    # Calculate overall accuracy
    overall_accuracy = evaluation_results.current_accuracy_avg
    avg_inference_time_ms = evaluation_results.current_inference_avg_ms
    # Assemble basic metrics dictionary
    metrics = {
        "total_samples": evaluation_results.total_samples,
        "overall_accuracy": round(overall_accuracy, 2),
        "avg_inference_time_ms": round(avg_inference_time_ms, 1),
        "exit_statistics": calculate_exit_statistics(evaluation_results),
        "class_statistics": calculate_class_statistics(
            evaluation_results, test_loader, config
        ),
        "confusion_matrix": calculate_confusion_matrix(evaluation_results).tolist(),
    }

    # Add advanced metrics
    advanced_metrics = calculate_advanced_metrics(metrics)
    for key, value in advanced_metrics.items():
        if key not in metrics:
            metrics[key] = value

    return metrics
