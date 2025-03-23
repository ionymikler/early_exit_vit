# Made by: Jonathan Mikler on 2025-02-10
import torch
from torch.utils.data import DataLoader

from utils import (
    logging_utils,
    dataset_utils,
    arg_utils,
    model_utils,
    check_conda_env,
)

from utils.evaluation import evaluate_pytorch_model
from utils.profiling import warmup_model, check_before_profiling
from typing import Tuple
import numpy as np

from utils.results_processing import result_utils


logger = logging_utils.get_logger_ready("evaluation")


def main():
    logger.info(logging_utils.yellow_txt("Starting evaluation..."))
    args = arg_utils.get_eval_argsparser().parse_args()
    device = torch.device("cuda" if args.use_gpu else "cpu")

    if not args.skip_conda_env_check and not check_conda_env("eevit"):
        exit()

    config = arg_utils.get_config_dict(args.config_path)
    model_config = arg_utils.parse_config_dict(config["model"].copy())

    dataset = dataset_utils.get_cifar100_dataset()
    _, test_dataset = dataset_utils.prepare_dataset(dataset, args.num_examples)

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=dataset_utils.collate_fn,
    )

    model = model_utils.setup_model_for_evaluation(
        model_config=model_config, device=device, verbose=True
    )

    check_before_profiling(args)

    # Create results directory before evaluation
    model_type = f"pytorch_{'gpu' if device.type == 'cuda' else 'cpu'}"
    if args.save_metrics:
        results_dir = result_utils.make_results_dir(
            model_type, profiling=args.profile_do, suffix=args.suffix
        )
        # Save metadata
        result_utils.save_metadata(results_dir, model_type, args, config)
    else:
        results_dir = None

    # warm up model
    def warmup_predictor_fn(images: torch.Tensor) -> Tuple[np.ndarray, float]:
        with torch.no_grad():
            images.to(device)
            outputs = model(images)
            predictions = outputs[:, :-1]  # Remove exit layer index
            exit_layer = outputs[:, -1].item()  # Get exit layer

            return predictions, exit_layer

    if args.profile_do:
        warmup_model(warmup_predictor_fn, test_loader)

    metrics = evaluate_pytorch_model(
        model=model,
        test_loader=test_loader,
        device=device,
        interactive=args.interactive,
        profile_do=args.profile_do,
        args=args,
        results_dir=results_dir,  # Pass results_dir to evaluate_pytorch_model
    )

    if args.save_metrics:
        result_utils.save_metrics(metrics, results_dir)


if __name__ == "__main__":
    main()
