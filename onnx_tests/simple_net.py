# simple_onnx_export.py
# A minimal example of exporting a PyTorch model to ONNX format

import torch
import torch.nn as nn
import onnx
import onnxruntime


# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.name = "SimpleModel"
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def export_to_onnx(model, input_tensor, output_path="model.onnx"):
    """Export PyTorch model to ONNX format"""
    print(f"Exporting model to ONNX format: {output_path}")

    # Using TorchDynamo (recommended for modern PyTorch)
    onnx_program = torch.onnx.export(  # noqa
        model=model,
        args=(input_tensor,),
        f=output_path,
        input_names=["input"],
        output_names=["output"],
        # do_constant_folding=True,
        # opset_version=17,
        # dynamo=True,
        verbose=True,
        report=True,
    )

    print("✅ Model exported successfully")

    return output_path


def verify_onnx_model(onnx_path, input_tensor):
    """Load and verify the ONNX model"""
    print(f"Verifying ONNX model: {onnx_path}")

    # Check the model structure
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # Run inference with ONNX Runtime
    ort_session = onnxruntime.InferenceSession(onnx_path)

    # Convert tensor to numpy for ONNX Runtime
    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    # Prepare inputs and run inference
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_tensor)}
    ort_outputs = ort_session.run(None, ort_inputs)

    print("✅ ONNX model verified successfully")
    return ort_outputs[0]


def main():
    # Create model instance
    model = SimpleModel(input_size=2, hidden_size=3, output_size=1)
    model.eval()  # Set to evaluation mode

    # Create sample input
    sample_input = torch.randn(1, 2)

    # Get PyTorch model prediction for comparison
    with torch.no_grad():
        torch_output = model(sample_input)
    print(f"PyTorch output: {torch_output}")

    # Export to ONNX
    onnx_path = export_to_onnx(model, sample_input, "simple_model.onnx")

    # Verify ONNX model
    onnx_output = verify_onnx_model(onnx_path, sample_input)
    print(f"ONNX output: {onnx_output}")

    # Compare outputs
    print(f"Outputs match: {torch.allclose(torch_output, torch.tensor(onnx_output))}")


if __name__ == "__main__":
    main()
