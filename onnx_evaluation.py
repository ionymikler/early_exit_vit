# Made by: Jonathan Mikler on 2025-03-06

import torch
import onnxruntime as ort
from torch.utils.data import DataLoader

import utils.logging_utils as logging_utils
import utils.dataset_utils as dataset_utils
import utils.arg_utils as arg_utils
import utils.result_utils as result_utils

from utils.eval_utils import evaluate_onnx_model, check_before_profiling, warmup_model
from utils import check_conda_env, to_numpy

logger = logging_utils.get_logger_ready("onnx_evaluation")


def make_inference_session(
    onnx_filepath: str, profile_do: bool, use_gpu: bool
) -> ort.InferenceSession:
    """
    Create an ONNX Runtime session for inference."
    """
    session_options = ort.SessionOptions()
    session_options.enable_profiling = profile_do

    provider = "CUDAExecutionProvider" if use_gpu else "CPUExecutionProvider"
    logger.info(f"Using {provider} for inference.")
    ort_session = ort.InferenceSession(
        onnx_filepath, providers=[provider], sess_options=session_options
    )
    return ort_session


def main():
    logger.info(logging_utils.yellow_txt("Starting ONNX evaluation..."))
    args = arg_utils.get_argsparser().parse_args()
    device = torch.device("cuda" if args.use_gpu else "cpu")
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

    check_before_profiling(args)

    # Create results directory before evaluation
    model_type = f"onnx_{'gpu' if args.use_gpu else 'cpu'}"
    results_dir = result_utils.make_results_dir(
        model_type, profiling=args.profile_do, suffix=args.suffix
    )

    # Save metadata
    result_utils.save_metadata(results_dir, model_type, args)

    # warmup
    warmup_session = make_inference_session(onnx_filepath, False, args.use_gpu)

    def warmup_predictor_fn(images):
        ort_inputs = {warmup_session.get_inputs()[0].name: to_numpy(images)}
        ort_outputs = warmup_session.run(None, ort_inputs)
        outputs = torch.from_numpy(ort_outputs[0])
        predictions = outputs[:, :-1]
        exit_layer = outputs[:, -1].item()
        return predictions, exit_layer

    warmup_model(warmup_predictor_fn, test_dataloader, device)

    # evaluate
    ort_session = make_inference_session(onnx_filepath, args.profile_do, args.use_gpu)

    # Evaluate the ONNX model
    metrics = evaluate_onnx_model(
        onnx_session=ort_session,
        test_loader=test_dataloader,
        interactive=args.interactive,
        args=args,
        results_dir=results_dir,
    )

    # Save metrics if requested
    if args.save_metrics:
        result_utils.save_metrics(metrics, results_dir)


if __name__ == "__main__":
    main()
