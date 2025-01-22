import torch
import time
import torch.nn as nn


def add_fast_pass(x):
    return torch.cat([x, torch.zeros(x.shape[0], 1)], dim=1)


def remove_fast_pass(x_with_fastpass: torch.Tensor) -> torch.Tensor:
    return x_with_fastpass.clone()[:, :-1]


def get_fast_pass(x_with_fastpass: torch.Tensor) -> torch.Tensor:
    return x_with_fastpass.clone()[:, -1]


def set_fast_pass_token(x_with_fastpass: torch.Tensor, value: float) -> torch.Tensor:
    output = x_with_fastpass.clone()
    output[:, -1] = value
    return output


def load_and_run_onnx(onnx_filepath: str, x: torch.Tensor):
    import onnxruntime as ort

    # Load ONNX model
    ort_session = ort.InferenceSession(onnx_filepath)
    ort_inputs = {ort_session.get_inputs()[0].name: x.detach().numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)

    return torch.tensor(ort_outputs[0])


class CheckerNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_fp):
        x = remove_fast_pass(x_fp)
        check = x >= 3.0
        x_fp = add_fast_pass(x)
        x_fp = set_fast_pass_token(x_fp, check.any().float())
        return x_fp


class AddOneNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_fp):
        x = remove_fast_pass(x_fp)
        x = x + 1.0
        x_fp = add_fast_pass(x)
        return x_fp


class SimpleConditionalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "SimpleConditionalNet"
        # Create 3 simple modules that add 1
        self.layers = nn.ModuleList(
            [
                AddOneNet(),
                AddOneNet(),
                CheckerNet(),
                AddOneNet(),
                AddOneNet(),
                CheckerNet(),
                AddOneNet(),
            ]
        )

    def fast_pass(self, x: torch.Tensor):
        return x.clone()

    def layer_forward(self, x_fp):
        """Regular computation through the current layer"""
        return self.layers[self.layer_idx](x_fp)

    def forward(self, x):
        x_fp = add_fast_pass(x)
        for layer_idx in range(len(self.layers)):
            self.layer_idx = layer_idx  # Keep track of current layer

            # Check if value > 3
            fp = get_fast_pass(x_fp)
            condition = fp.any()

            # x_fp = self.fast_pass(x_fp) if condition else self.layer_forward(x_fp)
            x_fp = torch.cond(condition, self.fast_pass, self.layer_forward, (x_fp,))

        return remove_fast_pass(x_fp)


def test_model():
    # Create model
    model = SimpleConditionalNet()
    model.eval()

    # Test cases
    test_inputs = [
        torch.tensor([[0.0]]),
        torch.tensor([[1.0]]),
        torch.tensor([[4.0]]),
    ]

    print("Testing model:")
    for x in test_inputs:
        out = model(x)
        print(f"Input: {x.item():.1f}, Output: {out.item():.1f}")

    # Export to ONNX
    try:
        x = test_inputs[-1]
        current_time = time.strftime("%H_%M_%S")
        onnx_filename = f"simple_conditional_{current_time}.onnx"

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
        onnx_program.save(onnx_filename)
        print("\nModel exported to ONNX successfully")
    except Exception as e:
        print(f"\nError exporting to ONNX: {e}")

    # Load and run ONNX model
    try:
        for x in test_inputs:
            onnx_out = load_and_run_onnx(onnx_filename, x)
            print(f"ONNX output: {onnx_out.item():.1f}")
            out = model(x)
            assert torch.allclose(out, onnx_out, atol=1e-3), "Output mismatch ❗"
            print("Outputs match! ✅")
    except Exception as e:
        print(f"\nError loading and running ONNX model: {e}")


if __name__ == "__main__":
    test_model()
