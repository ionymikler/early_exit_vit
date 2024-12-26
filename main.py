#!/usr/bin/env python
# Made by: Jonathan Mikler on 2024-12-03
import argparse
import yaml
import torch

# local imports
from utils.logging_utils import get_logger_ready
from vit import NamedModule, PatchEmbedding

logger = get_logger_ready("main")


def parse_config():
    parser = argparse.ArgumentParser(description="Process config file path.")
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to the configuration JSON file",
    )
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def run_model(x, model):
    logger.info("Running model")
    out = model(x)
    logger.info(f"model output shape: {out.shape}")


def export_model(model: NamedModule, _x, onnx_filepath: str):
    logger.info(f"Exporting model '{model.name}' to ONNX format")

    onnx_program = torch.onnx.export(
        model=model,
        args=(_x),
        dynamo=True,
        report=True,
    )
    onnx_program.save(onnx_filepath)
    logger.info(f"✅ Model exported to '{onnx_filepath}'")

    return onnx_filepath


def gen_data(data_shape: tuple):
    return torch.randn(data_shape)


def get_model(model_config: dict) -> NamedModule:
    # model = ViT(
    #     num_layers_transformer=model_config["transformer_layers"],
    #     image_size=model_config["image_size"],
    #     patch_size=model_config["patch_size"],
    #     num_classes=model_config["num_classes"],
    #     embed_depth=model_config["embed_depth"],
    #     num_attn_heads=model_config["heads"],
    #     mlp_dim=model_config["mlp_dim"],
    # )

    model = PatchEmbedding(config=model_config)

    return model


def main():
    args = parse_config()

    # Dataset config
    channels_num = args["dataset"]["channels_num"]
    img_size = args["dataset"]["image_size"]

    # ViT config
    model_config = args["model"]

    model = get_model(model_config)

    x = gen_data(data_shape=(2, channels_num, img_size, img_size))
    run_model(x=x, model=model)

    onnx_filepath = f"./models/onnx/{model.name}.onnx"
    export_model(model=model, _x=x, onnx_filepath=onnx_filepath)

    print("Model exported to model.onnx")


if __name__ == "__main__":
    main()
