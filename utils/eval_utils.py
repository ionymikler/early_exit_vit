# Made by: Jonathan Mikler on 2025-03-06
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import defaultdict

from utils import (
    logging_utils,
    result_utils,
    dataset_utils,
)

logger = logging_utils.get_logger_ready("common_evaluation")


def evaluate_model_generic(
    predictor_fn,
    test_loader: DataLoader,
    interactive: bool = False,
    save_eval_metrics: bool = False,
    metrics_prefix: str = "evaluation",
) -> dict:
    """
    Generic evaluation function that works with both PyTorch and ONNX models.

    Args:
        predictor_fn: A function that takes a batch of images and returns predictions and exit layer
                    The function should return a tuple (predictions, exit_layer)
        test_loader: DataLoader for test set
        interactive: If True, shows detailed results for each image and waits for user input
        save_eval_metrics: Whether to save metrics to a file
        metrics_prefix: Prefix for saved metrics file

    Returns:
        dict: Dictionary containing detailed evaluation metrics
    """
    logger.info("Starting evaluation...")
    if interactive:
        logger.info("Press Enter to continue to next image, or 'q' to quit")

    # Statistics tracking
    total_samples = 0
    total_correct = 0
    exit_stats = defaultdict(lambda: {"count": 0, "correct": 0, "confidences": []})

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
        labels = batch["labels"]
        batch_size = labels.size(0)
        total_samples += batch_size

        # Apply the predictor function (different for PyTorch and ONNX)
        predictions, exit_layer = predictor_fn(images)

        # Get predictions and confidence
        confidence, predicted_classes = torch.max(predictions, dim=1)
        correct_predictions = predicted_classes == labels
        num_correct = correct_predictions.sum().item()

        total_correct += num_correct

        # Update per-exit statistics
        exit_key = "final" if exit_layer == -1 else f"exit_{int(exit_layer)}"
        exit_stats[exit_key]["count"] += batch_size
        exit_stats[exit_key]["correct"] += num_correct
        exit_stats[exit_key]["confidences"].append(confidence.cpu().numpy())

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
            print("=" * 50)
        else:
            current_accuracy = 100 * total_correct / total_samples
            pbar.set_postfix({"acc": f"{current_accuracy:.2f}%"})

    overall_accuracy = 100 * total_correct / total_samples

    # Process per-exit statistics
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
        exit_accuracy = 100 * stats["correct"] / stats["count"]
        exit_confidence = np.mean(np.concatenate(stats["confidences"])).item()

        metrics["exit_statistics"][exit_key] = {
            "count": stats["count"],
            "accuracy": round(exit_accuracy, 4),
            "percentage_samples": round(100 * stats["count"] / total_samples, 2),
            "average_confidence": round(exit_confidence, 4),
        }

    # Printed summary
    logger.info(logging_utils.yellow_txt("\nEvaluation Complete!"))
    logger.info("Evaluation Summary:")
    logger.info(f"Overall Accuracy: {overall_accuracy:.2f}%")
    logger.info(f"Total Samples: {total_samples}")

    if save_eval_metrics:
        result_utils.save_metrics(metrics, metrics_prefix)

    return metrics


def evaluate_pytorch_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    interactive: bool = False,
    save_eval_metrics: bool = False,
) -> dict:
    """
    Evaluate PyTorch model on test set with detailed per-exit statistics.

    Args:
        model: PyTorch model
        test_loader: DataLoader for test set
        device: Device to run model on
        interactive: If True, shows detailed results for each image and waits for user input
        save_eval_metrics: Whether to save metrics to a file

    Returns:
        dict: Dictionary containing detailed evaluation metrics
    """
    logger.info("ℹ️  Starting PyTorch model evaluation...")
    model.eval()

    def predictor_fn(images):
        """Wrapper function for PyTorch model prediction"""
        with torch.no_grad():
            images = images.to(device)
            outputs = model(images)
            predictions = outputs[:, :-1]  # Remove exit layer index
            exit_layer = outputs[:, -1].item()  # Get exit layer
            return predictions, exit_layer

    return evaluate_model_generic(
        predictor_fn=predictor_fn,
        test_loader=test_loader,
        interactive=interactive,
        save_eval_metrics=save_eval_metrics,
        metrics_prefix="pytorch_evaluation",
    )


def evaluate_onnx_model(
    onnx_session,
    test_loader: DataLoader,
    interactive: bool = False,
    save_eval_metrics: bool = False,
) -> dict:
    """
    Evaluate ONNX model on test set with detailed per-exit statistics.

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

    return evaluate_model_generic(
        predictor_fn=predictor_fn,
        test_loader=test_loader,
        interactive=interactive,
        save_eval_metrics=save_eval_metrics,
        metrics_prefix="onnx_evaluation",
    )
