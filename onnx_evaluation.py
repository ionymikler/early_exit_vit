# Made by: Jonathan Mikler on 2025-03-06

import onnxruntime as ort
from torch.utils.data import DataLoader

import utils.logging_utils as logging_utils
import utils.dataset_utils as dataset_utils
import utils.arg_utils as arg_utils

from utils.eval_utils import evaluate_onnx_model
from utils import check_conda_env

logger = logging_utils.get_logger_ready("onnx_evaluation")


def make_inference_session(onnx_filepath: str, args) -> ort.InferenceSession:
    """
    Create an ONNX Runtime session for inference."
    """
    session_options = ort.SessionOptions()
    session_options.enable_profiling = args.onnx_enable_profile
    if session_options.enable_profiling and (
        (args.num_examples is not None and args.num_examples > 20)
        or args.num_examples is None
    ):
        logger.warning(
            logging_utils.yellow_txt(
                "Profiling is enabled and the number of examples is greater than 20. "
                "This may results in a large profiling file."
                "Consider reducing the number of examples or disabling profiling."
            )
        )
        exit()

    provider = "CUDAExecutionProvider" if args.use_gpu else "CPUExecutionProvider"
    logger.info(f"Using {provider} for inference.")
    ort_session = ort.InferenceSession(
        onnx_filepath, providers=[provider], sess_options=session_options
    )
    return ort_session


def main():
    logger.info(logging_utils.yellow_txt("Starting ONNX evaluation..."))
    args = arg_utils.get_argsparser().parse_args()

    if not args.skip_conda_env_check and not check_conda_env("onnx_eval"):
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

    ort_session = make_inference_session(onnx_filepath, args)

    # Evaluate the ONNX model
    evaluate_onnx_model(
        onnx_session=ort_session,
        test_loader=test_dataloader,
        interactive=args.interactive,
        save_eval_metrics=args.save_metrics,
        args=args,
    )


if __name__ == "__main__":
    main()
