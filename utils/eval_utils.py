# Made by: Jonathan Mikler on 2025-03-06
import torch
import numpy as np
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity

from utils import (
    logging_utils,
    result_utils,
    dataset_utils,
)

logger = logging_utils.get_logger_ready(__name__)


def _calculate_per_class_statistics(class_stats: dict, test_loader: DataLoader) -> dict:
    """
    Calculate statistics from the collected per-class data.
    Enhanced to include distribution data for visualization.

    Args:
        class_stats: Dictionary containing raw per-class data
        test_loader: DataLoader containing dataset information for class names

    Returns:
        Dictionary with processed per-class metrics including distribution data
    """
    class_metrics = {}
    for class_id, stats in class_stats.items():
        if stats["count"] == 0:
            continue  # Skip classes with no examples

        # Get class name using dataset_utils
        try:
            class_name = dataset_utils.get_label_name(test_loader.dataset, class_id)
        except (AttributeError, ValueError):
            class_name = f"Class {class_id}"

        # Calculate accuracy
        accuracy = 100 * stats["correct"] / stats["count"]

        # Calculate exit layer statistics from exit_by_layer
        exit_distribution = stats["exit_by_layer"]
        exit_values = []
        for exit_key, count in exit_distribution.items():
            # Convert exit_key to numeric value for calculations
            if exit_key == "final":
                value = -1  # Final layer is represented as -1
            else:
                value = int(exit_key.split("_")[1])
            # Add this exit value to the list count times
            exit_values.extend([value] * count)

        exit_values = np.array(exit_values)
        avg_exit = np.mean(exit_values).item() if len(exit_values) > 0 else 0
        std_exit = np.std(exit_values).item() if len(exit_values) > 0 else 0

        # Calculate statistics for inference times
        inference_times = np.array(stats["inference_times"])
        avg_time = np.mean(inference_times).item()
        std_time = np.std(inference_times).item()
        min_time = np.min(inference_times).item() if len(inference_times) > 0 else 0
        max_time = np.max(inference_times).item() if len(inference_times) > 0 else 0

        # Calculate statistics for confidences
        confidences = np.array(stats["confidences"])
        avg_conf = np.mean(confidences).item()
        std_conf = np.std(confidences).item()
        min_conf = np.min(confidences).item() if len(confidences) > 0 else 0
        max_conf = np.max(confidences).item() if len(confidences) > 0 else 0

        # Calculate statistics for accuracies (should be 0s and 1s)
        accuracies = np.array(stats["accuracies"])

        # Store processed metrics
        class_metrics[class_id] = {
            "name": class_name,
            "count": stats["count"],
            "accuracy": round(accuracy, 4),
            # Exit layer statistics
            "avg_exit_layer": round(avg_exit, 4),
            "std_exit_layer": round(std_exit, 4),
            "exit_distribution": exit_distribution,
            # Inference time statistics
            "avg_inference_time_ms": round(avg_time, 4),
            "std_inference_time_ms": round(std_time, 4),
            "min_inference_time_ms": round(min_time, 4),
            "max_inference_time_ms": round(max_time, 4),
            # Confidence statistics
            "avg_confidence": round(avg_conf, 4),
            "std_confidence": round(std_conf, 4),
            "min_confidence": round(min_conf, 4),
            "max_confidence": round(max_conf, 4),
            # Raw distributions for visualization
            "inference_times": inference_times.tolist(),  # Store raw values for violin plots
            "confidences": confidences.tolist(),
            "accuracies": accuracies.tolist(),
            "exit_values": exit_values.tolist(),
        }

    return class_metrics


def _warmup_model(model, test_loader, device):
    """
    Warmup the model by running a few batches of dummy data.
    This is done to ensure more accurate performance measurements.

    Args:
        model: PyTorch model
        test_loader: DataLoader for test set
        device: Device to run on
    """
    logger.info(logging_utils.yellow_txt("Performing model warmup..."))
    # Generate dummy data
    # Get input shape from first batch in test_loader
    dummy_shape = next(iter(test_loader))["pixel_values"].shape
    dummy_input = torch.randn(dummy_shape, device=device)

    # Run model for 20 iterations to warm up
    num_warmup_iterations = 20
    logger.info(f"Running {num_warmup_iterations} warmup iterations with dummy data...")

    # First run can often be much slower, so do it separately
    _ = model(dummy_input)

    # Then run the remaining warmup iterations
    for _ in range(num_warmup_iterations - 1):
        _ = model(dummy_input)

    logger.info("Warmup complete")


def _evaluate_model_generic(
    predictor_fn,
    test_loader: DataLoader,
    device: torch.device,
    interactive: bool = False,
    save_eval_metrics: bool = False,
    metrics_prefix: str = "evaluation",
) -> dict:
    """
    Generic evaluation function that works with both PyTorch and ONNX models.
    Enhanced to collect full distributions of accuracy and latency data.
    Includes a warmup phase to ensure more accurate performance measurements.

    Args:
        predictor_fn: A function that takes a batch of images and returns predictions and exit layer
                    The function should return a tuple (predictions, exit_layer)
        test_loader: DataLoader for test set
        device: Device to run on
        interactive: If True, shows detailed results for each image and waits for user input
        save_eval_metrics: Whether to save metrics to a file
        metrics_prefix: Prefix for saved metrics file

    Returns:
        dict: Dictionary containing detailed evaluation metrics including per-class statistics
    """
    # Perform warmup with dummy data to ensure fair performance measurement
    _warmup_model(predictor_fn, test_loader, device)

    # Start actual evaluation
    logger.info("Starting evaluation...")
    if interactive:
        logger.info("Press Enter to continue to next image, or 'q' to quit")

    # Use fixed number of classes for CIFAR-100
    num_classes = 100

    # Enhanced statistics tracking to include distributions
    total_samples = 0
    total_correct = 0

    # Dictionary to collect individual results for each exit
    exit_stats = {}

    # Initialize per-class statistics
    class_stats = {}
    for class_id in range(num_classes):
        class_stats[class_id] = {
            "count": 0,  # Total examples of this class
            "correct": 0,  # Correctly classified examples
            "inference_times": [],  # List to store inference time in milliseconds
            "confidences": [],  # List to store confidence for each example
            "exit_by_layer": {},  # Will track count of samples per exit layer
            "accuracies": [],  # List to store accuracy (0 or 1) for each example
        }

    # Create progress bar for non-interactive mode
    pbar = None if interactive else tqdm(test_loader, desc="Evaluating", unit="batch")
    iterator = test_loader if interactive else pbar

    for batch_idx, batch in enumerate(iterator):
        if interactive:
            user_input = input("\nPress Enter for next image, or 'q' to quit: ")
            if user_input.lower() == "q":
                print("\nExiting interactive evaluation...")
                break

        # Get images and labels
        images = batch["pixel_values"]
        labels = batch["labels"].to(device)
        batch_size = labels.size(0)
        total_samples += batch_size

        # Measure inference time
        start_time = time.perf_counter_ns()
        predictions, exit_layer = predictor_fn(images)
        end_time = time.perf_counter_ns()
        inference_time_ns = end_time - start_time
        inference_time_ms = inference_time_ns / 1_000_000  # Convert to milliseconds

        # Get predictions and confidence
        confidence, predicted_classes = torch.max(predictions, dim=1)
        correct_predictions = predicted_classes == labels
        num_correct = correct_predictions.sum().item()

        total_correct += num_correct

        # Determine exit key
        exit_key = "final" if exit_layer == -1 else f"exit_{int(exit_layer)}"

        # Initialize exit stats dictionary for this exit if it doesn't exist
        if exit_key not in exit_stats:
            exit_stats[exit_key] = {
                "count": 0,
                "correct": 0,
                "confidences": [],
                "inference_times": [],  # Store all inference times for this exit
                "batch_accuracies": [],  # Store accuracy for each batch
                "accuracies_per_sample": [],  # Store binary accuracy (0/1) for each sample
            }

        # Update per-exit statistics with raw distribution data
        exit_stats[exit_key]["count"] += batch_size
        exit_stats[exit_key]["correct"] += num_correct
        exit_stats[exit_key]["confidences"].extend(confidence.cpu().numpy().tolist())
        exit_stats[exit_key]["inference_times"].append(inference_time_ms)

        # Store batch accuracy
        batch_accuracy = 100 * num_correct / batch_size
        exit_stats[exit_key]["batch_accuracies"].append(batch_accuracy)

        # Store individual sample accuracies (1 for correct, 0 for incorrect)
        exit_stats[exit_key]["accuracies_per_sample"].extend(
            correct_predictions.cpu().numpy().tolist()
        )

        # Update per-class statistics
        for i in range(batch_size):
            true_class = labels[i].item()
            is_correct = correct_predictions[i].item()
            conf = confidence[i].item()

            # Update true class statistics
            class_stats[true_class]["count"] += 1
            class_stats[true_class]["correct"] += 1 if is_correct else 0
            class_stats[true_class]["inference_times"].append(inference_time_ms)
            class_stats[true_class]["confidences"].append(conf)
            class_stats[true_class]["accuracies"].append(
                1 if is_correct else 0
            )  # Store binary accuracy

            # Update exit layer counters (adding on the go)
            if exit_key not in class_stats[true_class]["exit_by_layer"]:
                class_stats[true_class]["exit_by_layer"][exit_key] = 0
            class_stats[true_class]["exit_by_layer"][exit_key] += 1

        if interactive:
            label_name = batch["label_names"][0]  # Since batch size is 1
            predicted_name = dataset_utils.get_label_name(
                test_loader.dataset, predicted_classes.item()
            )

            print("\n" + "=" * 50)
            print(f"Image {batch_idx + 1}")
            print(f"True label: {label_name} (class {labels.item()})")
            print(f"Predicted: {predicted_name} (class {predicted_classes.item()})")
            print(f"Confidence: {confidence.item():.2%}")
            print(f"Exit layer: {exit_layer if exit_layer != -1 else 'Final layer'}")
            print(f"Inference time: {inference_time_ms:.2f} ms")
            print("=" * 50)
        else:
            current_accuracy = 100 * total_correct / total_samples
            pbar.set_postfix({"acc": f"{current_accuracy:.2f}%"})

    overall_accuracy = 100 * total_correct / total_samples

    # Process per-exit statistics and compute aggregated metrics
    metrics = {
        "overall_accuracy": round(overall_accuracy, 4),
        "total_samples": total_samples,
        "exit_statistics": {},
    }

    for exit_key, stats in exit_stats.items():
        if not stats["count"] > 0:
            logger.info(
                logging_utils.yellow_txt(f"No samples found for exit layer {exit_key}")
            )
            continue

        # Calculate exit accuracy
        exit_accuracy = 100 * stats["correct"] / stats["count"]

        # Process and store distribution data
        metrics["exit_statistics"][exit_key] = {
            "count": stats["count"],
            "accuracy": round(exit_accuracy, 4),
            "percentage_samples": round(100 * stats["count"] / total_samples, 2),
            "average_confidence": round(np.mean(stats["confidences"]), 4),
            # Add inference time statistics
            "avg_inference_time_ms": round(np.mean(stats["inference_times"]), 4),
            "std_inference_time_ms": round(np.std(stats["inference_times"]), 4),
            "min_inference_time_ms": round(np.min(stats["inference_times"]), 4),
            "max_inference_time_ms": round(np.max(stats["inference_times"]), 4),
            # Store raw distributions for plotting
            "inference_time_values": stats["inference_times"],  # Raw inference times
            "confidence_values": stats["confidences"],  # Raw confidence values
            "accuracy_values": stats[
                "accuracies_per_sample"
            ],  # Per-sample accuracy values (0/1)
            "batch_accuracy_values": stats[
                "batch_accuracies"
            ],  # Per-batch accuracy values (%)
        }

    # Calculate and add per-class statistics
    class_metrics = _calculate_per_class_statistics(class_stats, test_loader)
    metrics["class_statistics"] = class_metrics

    # Printed summary
    print("")
    logger.info(logging_utils.yellow_txt("Evaluation Complete! ✅"))
    logger.info("Evaluation Summary:")
    logger.info(f"Overall Accuracy: {overall_accuracy:.2f}%")
    logger.info(f"Total Samples: {total_samples}")
    logger.info(f"Classes with samples: {len(class_metrics)}")

    print("")
    logger.info("Latency Summary by Exit:")
    for exit_key, stats in sorted(
        metrics["exit_statistics"].items(),
        key=lambda x: float("inf")
        if x[0] == "final"
        else int(x[0].split("_")[1])
        if "_" in x[0]
        else float("inf"),
    ):
        exit_name = (
            "Final Layer" if exit_key == "final" else f"Exit {exit_key.split('_')[1]}"
        )
        avg_latency = stats["avg_inference_time_ms"]
        std_latency = stats["std_inference_time_ms"]
        sample_count = stats["count"]
        percentage = stats["percentage_samples"]

        logger.info(
            f"  {exit_name}: {avg_latency:.2f} ms ± {std_latency:.2f} ms ({sample_count} samples, {percentage:.1f}%)"
        )

    if save_eval_metrics:
        result_utils.save_metrics(metrics, metrics_prefix)

    return metrics


def evaluate_pytorch_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    interactive: bool = False,
    save_eval_metrics: bool = False,
    profile_do: bool = False,
) -> dict:
    """
    Evaluate PyTorch model on test set with detailed per-exit and per-class statistics.

    Args:
        model: PyTorch model
        test_loader: DataLoader for test set
        device: Device to run model on
        interactive: If True, shows detailed results for each image and waits for user input
        save_eval_metrics: Whether to save metrics to a file
        profile_do: Whether to profile the model

    Returns:
        dict: Dictionary containing detailed evaluation metrics
    """
    logger.info("ℹ️  Starting PyTorch model evaluation...")
    model.eval()

    def predictor_fn(images):
        """Wrapper function for PyTorch model prediction"""
        with torch.no_grad():
            images = images.to(device)
            if profile_do:
                with profile(
                    activities=[ProfilerActivity.CPU],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                ) as prof:
                    with record_function("model_inference"):
                        outputs = model(images)
                print(
                    prof.key_averages(group_by_stack_n=5).table(
                        sort_by="cpu_time_total", row_limit=10
                    )
                )
            else:
                outputs = model(images)
            predictions = outputs[:, :-1]  # Remove exit layer index
            exit_layer = outputs[:, -1].item()  # Get exit layer
            return predictions, exit_layer

    return _evaluate_model_generic(
        predictor_fn=predictor_fn,
        test_loader=test_loader,
        interactive=interactive,
        save_eval_metrics=save_eval_metrics,
        metrics_prefix="pytorch_evaluation",
        device=device,
    )


def evaluate_onnx_model(
    onnx_session,
    test_loader: DataLoader,
    interactive: bool = False,
    save_eval_metrics: bool = False,
) -> dict:
    """
    Evaluate ONNX model on test set with detailed per-exit and per-class statistics.

    Args:
        onnx_session: ONNX Runtime InferenceSession
        test_loader: DataLoader for test set
        interactive: If True, shows detailed results for each image and waits for user input
        save_eval_metrics: Whether to save metrics to a file

    Returns:
        dict: Dictionary containing detailed evaluation metrics
    """
    logger.info("ℹ️  Starting ONNX model evaluation...")

    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    def predictor_fn(images):
        """Wrapper function for ONNX model prediction"""
        ort_inputs = {onnx_session.get_inputs()[0].name: to_numpy(images)}
        ort_outputs = onnx_session.run(None, ort_inputs)
        outputs = torch.from_numpy(ort_outputs[0])
        predictions = outputs[:, :-1]  # Remove exit layer index
        exit_layer = outputs[:, -1].item()  # Get exit layer
        return predictions, exit_layer

    return _evaluate_model_generic(
        predictor_fn=predictor_fn,
        test_loader=test_loader,
        interactive=interactive,
        save_eval_metrics=save_eval_metrics,
        metrics_prefix="onnx_evaluation",
        device=torch.device("cpu"),
    )
