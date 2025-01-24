import torch
import time
import torch.nn as nn

import mockup_eevit_utils as ee_utils
from utils import parse_args


def load_and_run_onnx(onnx_filepath: str, x: torch.Tensor):
    import onnxruntime as ort

    # Ensure the input tensor is in the correct shape
    x = x.unsqueeze(0) if x.dim() == 1 else x

    # Load ONNX model
    ort_session = ort.InferenceSession(onnx_filepath)
    ort_inputs = {ort_session.get_inputs()[0].name: x.detach().numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)

    return torch.tensor(ort_outputs[0])


class ExitEvaluatorMockup:
    def __init__(self, threshold):
        self.threshold = threshold

    def decision_tensor(self, logits):
        fast_pass = ee_utils.confidence(logits) > self.threshold
        return fast_pass


class HighwayMockup(nn.Module):
    def __init__(self):
        super().__init__()

        self.exit_evaluator = ExitEvaluatorMockup(0.6)

    def forward(self, x_fp):
        x = ee_utils.remove_fast_pass(x_fp)
        x_fp = ee_utils.set_fast_pass_token(
            x_fp, self.exit_evaluator.decision_tensor(x).any().float()
        )
        return x_fp


class AttentionMockup(nn.Module):
    def __init__(self):
        super().__init__()
        self.point_5 = nn.Parameter(torch.tensor([0.5, 0.0, 0.0]))

    def forward(self, x_fp):
        x = ee_utils.remove_fast_pass(x_fp)
        x = x + self.point_5
        x_fp = ee_utils.add_fast_pass(x)
        return x_fp


class EEVITMockup(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "EEVITMockup"
        # Create 3 simple modules that add 1
        self.layers = nn.ModuleList(
            [
                AttentionMockup(),
                HighwayMockup(),
                AttentionMockup(),
                HighwayMockup(),
                AttentionMockup(),
                HighwayMockup(),
                AttentionMockup(),
                HighwayMockup(),
                AttentionMockup(),
            ]
        )

    def fast_pass(self, x: torch.Tensor):
        return x.clone()

    def layer_forward(self, x_fp):
        """Regular computation through the current layer"""
        return self.layers[self.layer_idx](x_fp)

    def forward(self, x):
        x_fp = ee_utils.add_fast_pass(x)
        for layer_idx in range(len(self.layers)):
            self.layer_idx = layer_idx  # Keep track of current layer

            # Check if value > 3
            condition = ee_utils.get_fast_pass(x_fp).any()

            # x_fp = self.fast_pass(x_fp) if condition else self.layer_forward(x_fp)
            x_fp = torch.cond(condition, self.fast_pass, self.layer_forward, (x_fp,))

        return ee_utils.remove_fast_pass(x_fp)


def test_model(model: nn.Module, test_cases):
    print("Testing model:")

    for input_tensor, expected_output in test_cases:
        out = model(input_tensor)
        print(f"Input: {input_tensor}, Output: {out}")
        assert torch.allclose(
            out, expected_output, atol=1e-3
        ), "Test output mismatch ❗"


def export_to_onnx(model, x, test_cases, save=False):
    print("Exporting to ONNX:")

    current_time = time.strftime("%H_%M_%S")
    onnx_filename = f"{model.name}_{current_time}.onnx"
    try:
        onnx_program = torch.onnx.export(
            model,
            x,
            onnx_filename,
            input_names=["input"],
            output_names=["output"],
            # opset_version=17,
            dynamo=True,
            report=True,
            verbose=True,
        )

        if save:
            save(onnx_program, onnx_filename)
            print("ONNX program NOT saved")

    except Exception as e:
        print(f"Error exporting to ONNX: {e}")

    # Load and run ONNX model
    if save:
        test_onnx_export(onnx_filename, model, test_cases)
    else:
        print("ONNX test skipped")


def test_onnx_export(onnx_filename, model, test_cases):
    print("Testing ONNX export:")

    for input_tensor, expected_output in test_cases:
        out = model(input_tensor)
        onnx_out = load_and_run_onnx(onnx_filename, input_tensor)

        print(f"ONNX output: {onnx_out}")
        assert torch.allclose(
            out, onnx_out, atol=1e-3
        ), "ONNX vs Pytorch output mismatch ❗"
        print("Outputs match! ✅")


def make_test_cases():
    inputs = [
        (torch.tensor([[1.0, 1.1, 1.2]]), torch.tensor([[2.5, 1.1, 1.2]])),
        (torch.tensor([[1.5, 1.1, 1.2]]), torch.tensor([[2.5, 1.1, 1.2]])),
        (torch.tensor([[2.0, 1.1, 1.2]]), torch.tensor([[2.5, 1.1, 1.2]])),
    ]
    return inputs
    # return [(inp, ee_utils.confidence(inp)) for inp in inputs]


def main():
    args = parse_args()
    # Create model
    model = EEVITMockup()
    model.eval()

    # Test cases
    test_cases = make_test_cases()

    test_model(model, test_cases)

    # Export to ONNX
    if args.export_onnx:
        export_to_onnx(model, test_cases[-1][0], test_cases, save=args.save_onnx)


if __name__ == "__main__":
    main()
