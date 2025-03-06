# Made by: Jonathan Mikler on 2025-03-06

import torch
import numpy as np
import onnxruntime as ort

from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import defaultdict

import utils.logging_utils as logging_utils
import utils.dataset_utils as dataset_utils
import utils.arg_utils as arg_utils
import utils.result_utils as result_utils

from utils import check_conda_env

# _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DEVICE = torch.device("cpu")

logger = logging_utils.get_logger_ready("evaluation")


def evaluate_onnx_model(
    onnx_filepath: str,
    test_loader: DataLoader,
    interactive: bool = False,
    save_eval_metrics: bool = False,
) -> dict:
    """
    Evaluate ONNX model on test set with detailed per-exit statistics.

    Args:
        onnx_filepath: Path to the ONNX model file
        test_loader: DataLoader for test set
        interactive: If True, shows detailed results for each image and waits for user input
        save_eval_metrics: Whether to save metrics to a file

    Returns:
        dict: Dictionary containing detailed evaluation metrics
    """
    logger.info("ℹ️  Starting ONNX evaluation...")

    # Initialize ONNX Runtime session
    ort_session = ort.InferenceSession(
        onnx_filepath, providers=["CPUExecutionProvider"]
    )

    if interactive:
        logger.info("Press Enter to continue to next image, or 'q' to quit")

    # Statistics tracking
    total_samples = 0
    total_correct = 0
    exit_stats = defaultdict(lambda: {"count": 0, "correct": 0, "confidences": []})

    # Create progress bar for non-interactive mode
    pbar = None if interactive else tqdm(test_loader, desc="Evaluating", unit="batch")
    iterator = test_loader if interactive else pbar

    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    for batch_idx, batch in enumerate(iterator):
        if interactive:
            user_input = input("\nPress Enter for next image, or 'q' to quit: ")
            if user_input.lower() == "q":
                print("\nExiting interactive evaluation...")
                break

        # Prepare input
        images = batch["pixel_values"]
        labels = batch["labels"]
        batch_size = labels.size(0)
        total_samples += batch_size

        # Run inference
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(images)}
        ort_outputs = ort_session.run(None, ort_inputs)
        outputs = torch.from_numpy(ort_outputs[0])

        # Process outputs
        predictions = outputs[:, :-1]  # Remove exit layer index
        exit_layer = outputs[:, -1].item()  # Get exit layer

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
        result_utils.save_metrics(metrics, "onnx_evaluation")

    logger.info("Done evaluating ONNX model ✅")


def main():
    logger.info(logging_utils.yellow_txt("Starting evaluation..."))
    args = arg_utils.get_argsparser().parse_args()

    if not args.skip_conda_env_check and not check_conda_env("eevit"):
        exit()

    dataset = dataset_utils.get_cifar100_dataset()
    _, test_dataset = dataset_utils.prepare_dataset(dataset, args.num_examples)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset_utils.collate_fn,
    )

    # We either create a new ONNX model or use an existing one
    try:
        onnx_filepath = args.onnx_program_filepath
    except AttributeError as e:
        logger.error(f"Could not find ONNX model file. {e}")
        exit()

    # Evaluate the ONNX model
    evaluate_onnx_model(
        onnx_filepath=onnx_filepath,
        test_loader=test_dataloader,
        interactive=args.interactive,
        save_eval_metrics=args.save_metrics,
    )


if __name__ == "__main__":
    main()
