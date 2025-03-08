# Made by: Jonathan Mikler on 2025-02-10
import torch
from torch.utils.data import DataLoader

from utils import (
    logging_utils,
    dataset_utils,
    arg_utils,
    model_utils,
    result_utils,
    check_conda_env,
)

from utils.eval_utils import evaluate_pytorch_model, check_before_profiling


logger = logging_utils.get_logger_ready("evaluation")


def main():
    logger.info(logging_utils.yellow_txt("Starting evaluation..."))
    args = arg_utils.get_argsparser().parse_args()
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
    model_type = f"pytorch_{device.type}"
    if args.save_metrics:
        results_dir = result_utils.make_results_dir(
            model_type, profiling=args.profile_do
        )
        # Save metadata
        result_utils.save_metadata(results_dir, model_type, args)
    else:
        results_dir = None

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
        prefix = f"pytorch_evaluation_{device.type}"
        result_utils.save_metrics(metrics, results_dir, prefix)


if __name__ == "__main__":
    main()
