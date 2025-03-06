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

from utils.eval_utils import evaluate_pytorch_model

# _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DEVICE = torch.device("cpu")

logger = logging_utils.get_logger_ready("evaluation")


def main():
    logger.info(logging_utils.yellow_txt("Starting evaluation..."))
    args = arg_utils.get_argsparser().parse_args()

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
        model_config=model_config, device=_DEVICE, verbose=True
    )

    evaluate_pytorch_model(
        model=model,
        test_loader=test_loader,
        device=_DEVICE,
        interactive=args.interactive,
        save_eval_metrics=args.save_metrics,
        profile_do=args.profile_do,
    )


if __name__ == "__main__":
    main()
