"""
Main evaluation functions for EEVIT models.
"""

import time
import numpy as np
import torch
import onnxruntime
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Callable, Tuple

from utils import logging_utils, dataset_utils, to_numpy
from utils.arg_utils import get_config_dict
from utils.profiling import save_profiler_output

# Import the data structures and metric calculation functions
from .structures import BatchResult, EvaluationResults
from .metrics import build_final_metrics

logger = logging_utils.get_logger_ready(__name__, level="DEBUG")


def _evaluate_model_generic(
    predictor_fn: Callable[[torch.Tensor], Tuple[np.ndarray, float]],
    test_loader,
    interactive: bool = False,
    args=None,
) -> dict:
    """
    Generic evaluation function that works with both PyTorch and ONNX models.
    Collects comprehensive performance metrics including confusion matrix data.

    Args:
        predictor_fn: A function that takes a batch of images and returns predictions and exit layer.
                    The function should return a tuple (predictions, exit_layer)
        test_loader: DataLoader for test set
        interactive: If True, shows detailed results for each image and waits for user input
        args: Command-line arguments used for evaluation (for metadata)

    Returns:
        dict: Dictionary containing detailed evaluation metrics including:
            - overall_accuracy: Overall model accuracy as a percentage
            - total_samples: Total number of samples evaluated
            - exit_statistics: Per-exit point statistics (accuracy, latency, etc.)
            - class_statistics: Per-class statistics (accuracy, exit patterns, etc.)
            - confusion_matrix: Matrix showing class prediction patterns
            - advanced metrics: Speedup, computation savings, etc.
    """
    # Start actual evaluation
    logger.info("Starting evaluation...")
    if interactive:
        logger.info("Press Enter to continue to next image, or 'q' to quit")

    # Get number of classes from the dataset or model configuration
    num_classes = args.num_classes if hasattr(args, "num_classes") else 100

    # Initialize results collector
    results = EvaluationResults(num_classes=num_classes)

    # Get total number of batches for progress tracking
    total_batches = len(test_loader)

    # Create progress bar for non-interactive mode
    pbar = (
        None
        if interactive
        else tqdm(total=total_batches, desc="Evaluating", unit="batch")
    )

    logger.info("🐎 Go!")

    # Process each batch from the DataLoader
    for batch_idx, batch in enumerate(test_loader):
        # Get images and labels
        images = batch["pixel_values"]
        labels = batch["labels"]
        batch_size = labels.size(0)

        # Update total samples counter
        results.total_samples += batch_size

        # Measure inference time
        start_time = time.perf_counter_ns()
        predictions, exit_layer = predictor_fn(images)
        end_time = time.perf_counter_ns()
        inference_time_ms = (
            end_time - start_time
        ) / 1_000_000  # Convert to milliseconds

        # Get predictions and confidence scores
        predicted_classes = np.argmax(predictions, axis=1)
        confidence_scores = np.max(predictions, axis=1)

        # Convert labels to numpy for comparison
        labels_np = labels.numpy()

        # Determine correct predictions
        correct_predictions = predicted_classes == labels_np
        num_correct = np.sum(correct_predictions)

        # Update total correct counter
        results.total_correct += num_correct

        # Determine exit key
        exit_key = "final" if exit_layer == -1 else f"exit_{int(exit_layer)}"

        # Create batch result
        batch_result = BatchResult(
            batch_size=batch_size,
            true_labels=labels_np,
            predicted_classes=predicted_classes,
            confidences=confidence_scores,
            exit_layer=exit_layer,
            inference_time_ms=inference_time_ms,
            is_correct=correct_predictions,
        )

        # Update exit collector
        exit_collector = results.get_or_create_exit_collector(exit_key)
        exit_collector.update_from_batch(batch_result)

        # Update per-class collectors
        for i in range(batch_size):
            true_class = int(labels_np[i])
            predicted_class = int(predicted_classes[i])
            is_correct = bool(correct_predictions[i])
            confidence = float(confidence_scores[i])

            class_collector = results.get_or_create_class_collector(true_class)
            class_collector.update_from_sample(
                true_class=true_class,
                predicted_class=predicted_class,
                is_correct=is_correct,
                confidence=confidence,
                inference_time=inference_time_ms,
                exit_key=exit_key,
            )

        # Handle interactive mode display
        if interactive:
            _display_interactive_results(
                batch,
                batch_idx,
                total_batches,
                predicted_classes,
                confidence_scores,
                exit_layer,
                inference_time_ms,
                test_loader,
            )

            # Ask whether to continue
            user_input = input("\nPress Enter to continue, or 'q' to quit: ")
            if user_input.lower() == "q":
                break
        else:
            # Update progress bar for non-interactive mode
            if pbar:
                current_accuracy = 100 * results.total_correct / results.total_samples
                pbar.set_postfix({"acc": f"{current_accuracy:.2f}%"})
                pbar.update(1)

    # Close progress bar if it exists
    if pbar:
        pbar.close()

    # Process collected data into metrics
    config = (
        get_config_dict(args.config_path)
        if args and hasattr(args, "config_path")
        else {}
    )
    metrics = build_final_metrics(results, test_loader, config)

    # Print summary
    _print_evaluation_summary(metrics)

    return metrics


def evaluate_pytorch_model(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    args,
    interactive: bool = False,
    profile_do: bool = False,
    results_dir: str = None,
) -> dict:
    """
    Evaluate PyTorch model on test set with detailed per-exit and per-class statistics.

    Args:
        model: PyTorch model
        test_loader: DataLoader for test set
        device: Device to run model on
        args: Command-line arguments used for evaluation (for metadata)
        interactive: If True, shows detailed results for each image and waits for user input
        profile_do: Whether to profile the model
        results_dir: Directory to save results to

    Returns:
        dict: Dictionary containing detailed evaluation metrics
    """
    logger.info("ℹ️  Starting PyTorch model evaluation...")
    model.eval()

    def predictor_fn_with_profiling(images):
        """Prediction function with profiling"""
        with torch.no_grad():
            images = images.to(device)
            with profile(
                activities=[ProfilerActivity.CPU],
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            ) as prof:
                with record_function("model_inference"):
                    outputs = model(images)

            # Process profiling results
            if results_dir:
                save_profiler_output(prof, results_dir, "pytorch")

            predictions = outputs[:, :-1]  # Remove exit layer index
            exit_layer = outputs[:, -1].item()  # Get exit layer
            return to_numpy(predictions), exit_layer

    def predictor_fn_without_profiling(
        images: torch.Tensor,
    ) -> Tuple[np.ndarray, float]:
        """Prediction function without profiling"""
        with torch.no_grad():
            images = images.to(device)
            outputs: torch.Tensor = model(images)
            predictions = outputs[:, :-1]  # Remove exit layer index
            exit_layer = outputs[:, -1].item()  # Get exit layer

            return to_numpy(predictions), exit_layer

    # Use the appropriate predictor function based on profiling setting
    predictor_fn = (
        predictor_fn_with_profiling if profile_do else predictor_fn_without_profiling
    )

    # Call the generic evaluation function
    return _evaluate_model_generic(
        predictor_fn=predictor_fn,
        test_loader=test_loader,
        interactive=interactive,
        args=args,
    )


def evaluate_onnx_model(
    onnx_session: onnxruntime.InferenceSession,
    test_loader,
    interactive: bool = False,
    args=None,
    results_dir: str = None,
) -> dict:
    """
    Evaluate ONNX model on test set with detailed per-exit and per-class statistics.

    Args:
        onnx_session: ONNX Runtime InferenceSession
        test_loader: DataLoader for test set
        interactive: If True, shows detailed results for each image and waits for user input
        args: Command-line arguments used for evaluation (for metadata)
        results_dir: Directory to save results to

    Returns:
        dict: Dictionary containing detailed evaluation metrics
    """
    logger.info("ℹ️  Starting ONNX model evaluation...")

    def predictor_fn(images: torch.Tensor) -> Tuple[np.ndarray, float]:
        """
        Returns:
            Tuple of (predictions array, exit_layer value)
        """
        # Convert input to numpy for ONNX runtime
        images_np = to_numpy(images)

        # Use IO binding for potentially better performance
        io_binding = onnx_session.io_binding()
        io_binding.bind_cpu_input("image", images_np)
        io_binding.bind_output("getitem_24")
        onnx_session.run_with_iobinding(io_binding)
        ort_outputs = io_binding.copy_outputs_to_cpu()

        # Process outputs
        outputs = ort_outputs[0]
        predictions = outputs[:, :-1]  # Remove exit layer index
        exit_layer = float(outputs[0, -1])

        return predictions, exit_layer

    # Call the generic evaluation function
    metrics = _evaluate_model_generic(
        predictor_fn=predictor_fn,
        test_loader=test_loader,
        interactive=interactive,
        args=args,
    )

    # Handle profiling if enabled
    if args and hasattr(args, "profile_do") and args.profile_do and results_dir:
        profile_path = onnx_session.end_profiling()
        # Save the profile results
        save_profiler_output(profile_path, results_dir, "onnx")

    return metrics


def _display_interactive_results(
    batch,
    batch_idx,
    total_batches,
    predicted_classes,
    confidence_scores,
    exit_layer,
    inference_time_ms,
    test_loader,
):
    """Display detailed results for a single sample in interactive mode."""
    # Display details for interactive mode (assuming batch_size=1 for interactive)
    label_name = batch["label_names"][0]
    predicted_name = dataset_utils.get_label_name(
        test_loader.dataset, predicted_classes[0]
    )

    print("\n" + "=" * 50)
    print(f"Image {batch_idx} of {total_batches-1}")
    print(f"True label: {label_name} (class {batch['labels'][0].item()})")
    print(f"Predicted: {predicted_name} (class {predicted_classes[0]})")
    print(f"Confidence: {confidence_scores[0]:.2%}")
    print(f"Exit layer: {exit_layer if exit_layer != -1 else 'Final layer'}")
    print(f"Inference time: {inference_time_ms:.2f} ms")
    print("=" * 50)


def _print_evaluation_summary(metrics):
    """Print a summary of the evaluation results to the console."""
    print("")
    logger.info(logging_utils.yellow_txt("Evaluation Complete! ✅"))
    logger.info("Evaluation Summary:")
    logger.info(f"Overall Accuracy: {metrics['overall_accuracy']:.2f}%")
    logger.info(f"Total Samples: {metrics['total_samples']}")
    logger.info(f"Classes with samples: {len(metrics['class_statistics'])}")

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
