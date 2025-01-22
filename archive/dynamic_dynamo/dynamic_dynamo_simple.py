# Made by: Jonathan Mikler on 2024-12-04

# V3: conditional using fc.cond. Did not work yet (13.11.24)

import torch
import torch.nn as nn
import onnx
import onnxruntime
from torch.onnx import ONNXProgram
import logging
from datetime import datetime

FILENAME = str(__file__).split("/")[-1].split(".")[0]  # get the name of the script


def get_logger():
    # Set up logging
    name = FILENAME
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create a stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter("[%(levelname)s][%(name)s][%(asctime)s]: %(message)s")
    formatter.default_time_format = "%H:%M:%S"
    formatter.default_msec_format = "%s.%03d"
    stream_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(stream_handler)

    logger.debug("Logger initialized with stream handler at DEBUG level")
    return logger


logger = get_logger()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TwoLayerNetDynamic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNetDynamic, self).__init__()
        self.model_name = "TwoLayerNetDynamic"

        self.true_module = SubBranch(1)
        self.false_module = SubBranch(2)

        self.threshold = torch.tensor([0.0], dtype=torch.float32).to(DEVICE)

        self.training_exits = False

        print("TwoLayerNetDynamic initialized")

    def forward(self, x: torch.Tensor):
        x = torch.cond(x.any(), self.true_module, self.false_module, (x,))
        return x


class SubBranch(nn.Module):
    def __init__(self, k):
        super(SubBranch, self).__init__()
        self.k = k

    def forward(self, x: torch.Tensor):
        return x * self.k


def run_model(x, model):
    logger.info("Running model")
    out = model(x)
    logger.info(f"model output shape: {out.shape}")


def export_model(model: nn.Module, _x):
    logger.info("Exporting model to ONNX format")

    ### Using TorchDynamo ###
    filename = FILENAME
    timestamp = datetime.now().strftime("%d_%m_%Y")
    onnx_filepath = f"./models/onnx/{filename}_{timestamp}.onnx"
    onnx_program: ONNXProgram = torch.onnx.export(
        model=model, args=(_x,), dynamo=True, report=True
    )
    onnx_program.save(onnx_filepath)
    logger.info(f"✅ Model exported to {onnx_filepath}")

    return onnx_filepath


def load_and_run_onnx(onnx_filepath, _x):
    logger.info("Loading and running ONNX model")
    logger.info(f"[onnx] Input: {_x}")

    onnx_model = onnx.load(onnx_filepath)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(
        onnx_filepath, providers=["CPUExecutionProvider"]
    )

    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(_x)}
    ort_outs = ort_session.run(None, ort_inputs)
    logger.info(f"[onnx] Output: {ort_outs}")


def main():
    model = TwoLayerNetDynamic(input_size=1, hidden_size=3, output_size=1)
    model.to(DEVICE)

    # _x = torch.normal(mean=0.0, std=2.0, size=(1, 1), dtype=torch.float32).to(DEVICE)
    _x = torch.tensor([1.0])
    logger.info(f"Input: {_x}")

    # run the model for sanity check
    run_model(_x, model)
    # return

    ## ONNX
    onnx_filepath = export_model(model, _x)

    load_and_run_onnx(onnx_filepath, _x)

    return


if __name__ == "__main__":
    main()
