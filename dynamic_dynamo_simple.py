# Made by: Jonathan Mikler on 2024-12-04

# V3: conditional using fc.cond. Did not work yet (13.11.24)

import torch
import torch.nn as nn
import onnx
import onnxruntime
from torch.onnx import ONNXProgram
import logging
from datetime import datetime

import utils as my_utils
from eevit.utils import (
    add_fast_pass,
    get_fast_pass,
    remove_fast_pass,
    flip_fast_pass_token,
)

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
    def __init__(self):
        super(TwoLayerNetDynamic, self).__init__()
        self.model_name = "TwoLayerNetDynamic"
        THRESHOLD = 3.0
        self.layers = nn.ModuleList(
            [
                SubBranch(1, THRESHOLD),
                SubBranch(1, THRESHOLD),
                SubBranch(1, THRESHOLD),
                SubBranch(1, THRESHOLD),
            ]
        )

        self.training_exits = False

        print("TwoLayerNetDynamic initialized")

    def fast_pass(self, x_with_fastpass: torch.Tensor):
        return x_with_fastpass

    def layer_forward(self, x_with_fastpass: torch.Tensor):
        module_i = self.layers[self.layer_idx]  # (attn or IC)
        x_with_fastpass = module_i(x_with_fastpass)

        return x_with_fastpass

    def forward(self, x: torch.Tensor):
        x_with_fastpass = add_fast_pass(x)
        for layer_idx in range(len(self.layers)):
            self.layer_idx = layer_idx
            fast_pass = get_fast_pass(x_with_fastpass).to(torch.bool)

            x_with_fastpass = torch.cond(
                fast_pass.any(), self.fast_pass, self.layer_forward, (x_with_fastpass,)
            )
        return x_with_fastpass


class SubBranch(nn.Module):
    def __init__(self, k, threshold):
        super(SubBranch, self).__init__()
        self.k = k
        self.threshold = threshold

    def decision(self, x: torch.Tensor):
        return x > self.threshold

    def forward(self, x_with_fastpass: torch.Tensor):
        x = remove_fast_pass(x_with_fastpass)
        x = x + self.k

        x_with_fastpass = torch.cond(
            self.decision(x), self.flip_token, self.do_nothing, (x_with_fastpass,)
        )

        return x_with_fastpass

    def do_nothing(self, x_with_fastpass):
        return x_with_fastpass

    def flip_token(self, x_with_fastpass):
        """Named function for true branch of cond"""
        return flip_fast_pass_token(x_with_fastpass)


def run_model(x, model):
    logger.info("Running model")
    out = model(x)
    logger.info(f"model output shape: {out.shape}")


def export_model(model: nn.Module, _x):
    logger.info("Exporting model to ONNX format")

    ### Using TorchDynamo ###
    filename = FILENAME
    timestamp = datetime.now().strftime("%d_%m_%y_%H_%M_%S")
    onnx_filepath = f"./models/onnx/dynamic_dynamo_simple/{filename}_{timestamp}.onnx"
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
    model = TwoLayerNetDynamic()
    model.to(DEVICE)

    # _x = torch.tensor([1.0])
    _x = my_utils.gen_data(data_shape=(1, 1, 1))
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
