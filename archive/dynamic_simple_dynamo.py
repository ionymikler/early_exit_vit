# Made by: Jonathan Mikler on 2024-12-04

# V3: conditional using fc.cond. Did not work yet (13.11.24)

import torch
import torch.nn as nn
import onnx
import onnxruntime
from torch.onnx import ONNXProgram
import logging
import functorch.experimental.control_flow as fc

torch.cond


def get_logger():
    # Set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create a stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter("[%(levelname)s][%(name)s][%(asctime)s]: %(message)s")
    stream_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(stream_handler)

    logger.debug("Logger initialized with stream handler at DEBUG level")
    return logger


logger = get_logger()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNet, self).__init__()
        self.model_name = "TwoLayerNet"
        self.fully_connected_1 = nn.Linear(input_size, hidden_size)
        self.fully_connected_2 = nn.Linear(hidden_size, output_size)

        print("TwoLayerNet initialized")

    def forward(self, x):
        x = torch.relu(self.fully_connected_1(x))
        x = self.fully_connected_2(x)
        return x


class TwoLayerNetDynamic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNetDynamic, self).__init__()
        self.model_name = "TwoLayerNetDynamic"

        self.fully_connected_1 = nn.Linear(input_size, hidden_size)
        self.early_exit_head_1 = nn.Linear(hidden_size, output_size)

        self.threshold = torch.tensor([0.0], dtype=torch.float32).to(DEVICE)
        self.last_exit = nn.Linear(hidden_size, output_size)

        self.training_exits = False

        print("TwoLayerNetDynamic initialized")

    def forward(self, x: torch.Tensor):
        mean = x.mean()

        x = self.fully_connected_1(x)
        x = fc.cond(mean > 0.0, self.early_exit_head_1, self.last_exit, (x,))
        x = torch.cat([x, mean.reshape_as(x)], dim=1)

        # if torch.gt(mean, self.threshold):
        #     x = self.early_exit_head_1(x)
        # else:
        #     x = self.last_exit(x)
        # x = torch.cat([x, mean.reshape_as(x)], dim=1)
        return x


def train_model(model, x_train, y_train, batch_size=5, epochs=100, learning_rate=0.01):
    # pass data to device
    x_train = x_train.to(DEVICE)
    y_train = y_train.to(DEVICE)

    # Check if CUDA is available and move model to GPU if possible
    model.to(DEVICE)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(x_train.size(0))

        for i in range(0, x_train.size(0), batch_size):
            indices = permutation[i : i + batch_size]
            batch_x, batch_y = x_train[indices], y_train[indices]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            epoch_print = f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}"

            # Calculate accuracy
            model.eval()
            with torch.no_grad():
                outputs = model(x_train)
                predicted = outputs.round()
                accuracy = (predicted == y_train).sum().item() / y_train.size(0)
                epoch_print += f", Accuracy: {accuracy * 100:.2f}%"

            print(epoch_print)
            if accuracy > 0.90:
                print("Accuracy is above 90%, stopping training")
                break

    return model


def eval_model(model, x_test, y_test):
    x_test = x_test.to(DEVICE)
    y_test = y_test.to(DEVICE)

    # Calculate accuracy
    model.eval()
    with torch.no_grad():
        outputs = model(x_test)
        predicted = outputs.round()
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        print(f"Evaluation accuracy: {accuracy * 100:.2f}%")


def run_model(x, model: TwoLayerNetDynamic):
    model.eval()
    y_pred = model(x)
    return y_pred


def export_model(model: nn.Module, _x):
    logger.info("Exporting model to ONNX format")

    # ### Using tracing ###
    # filename = f"{model.model_name}_tracing"
    # onnx_filepath = f"./models/onnx/{filename}.onnx"
    # torch.onnx.export(
    #     model=model,
    #     args=_x,
    #     f=onnx_filepath
    # )
    # logger.info(f"Model exported to {onnx_filepath}")

    ### Using scripting ###
    # filename = f"{model.model_name}_scripting"
    # onnx_filepath = f"./models/onnx/{filename}.onnx"

    # script_module = torch.jit.script(model)
    # torch.onnx.export(
    #     model=script_module,
    #     args=_x,
    #     f=onnx_filepath,
    #     report=True
    # )
    # logger.info(f"✅ Model exported to {onnx_filepath}")

    ### Using TorchDynamo ###
    filename = f"{model.model_name}_dynamo"
    onnx_filepath = f"./models/onnx/{filename}.onnx"
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
    TASK = "export"  # 'train' or 'export'

    model = TwoLayerNet(input_size=2, hidden_size=3, output_size=1)
    # model = TwoLayerNetDynamic(input_size=1, hidden_size=3, output_size=1)
    model.to(DEVICE)

    if TASK == "train":
        # Prepare data
        # Generate training data
        train_range = 11
        # num_classes = 21 # 0-20. max is 10+10=20

        x_train = torch.tensor(
            [[i, j] for i in range(train_range) for j in range(train_range)],
            dtype=torch.float32,
        )
        y_train = torch.tensor([[i + j] for (i, j) in x_train], dtype=torch.float32)
        # y_one_hot = torch.nn.functional.one_hot(y_train.to(torch.int64), num_classes=21).to(torch.float32)
        # Train the model
        model = train_model(model, epochs=500, x_train=x_train, y_train=y_train)
        eval_model(model, x_test=x_train, y_test=y_train)
    elif TASK == "export":
        _x = torch.normal(mean=0.0, std=2.0, size=(1, 1), dtype=torch.float32).to(
            DEVICE
        )
        logger.info(f"Input: {_x}")

        # run the model for sanity check
        y_pred = run_model(_x, model)
        logger.info(f"Prediction: {y_pred}")

        ## ONNX
        onnx_filepath = export_model(model, _x)

        load_and_run_onnx(onnx_filepath, _x)

    return


if __name__ == "__main__":
    main()
