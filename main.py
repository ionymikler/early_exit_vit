#!/usr/bin/env python
# Made by: Jonathan Mikler on 2024-12-03
import argparse
import yaml
import torch
import torch.nn as nn

# local imports
from utils.logging_utils import get_logger_ready
from vit import ViT

logger = get_logger_ready("main")


def parse_config():
    parser = argparse.ArgumentParser(description="Process config file path.")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the configuration JSON file",
    )
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def export_model(model: nn.Module, _x, onnx_filepath: str):
    logger.info("Exporting model to ONNX format")

    script_module = torch.jit.script(model)
    torch.onnx.export(model=script_module, args=_x, f=onnx_filepath, report=True)

    logger.info(f"✅ Model exported to {onnx_filepath}")

    return onnx_filepath


def main():
    args = parse_config()

    # Dataset config
    channels_num = args["dataset"]["channels_num"]
    img_size = args["dataset"]["image_size"]

    # ViT config
    vit_config = args["vit"]
    model = ViT(
        num_layers_transformer=vit_config["transformer_layers"],
        image_size=vit_config["image_size"],
        patch_size=vit_config["patch_size"],
        num_classes=vit_config["num_classes"],
        embed_depth=vit_config["embed_depth"],
        num_attn_heads=vit_config["heads"],
        mlp_dim=vit_config["mlp_dim"],
    )

    img = torch.randn(1, channels_num, img_size, img_size)
    pred = model(img)  # (1, 1000)
    logger.info(f"Prediction shape: {pred.shape}")

    model_name = "early_exit_vit"
    onnx_filepath = f"./models/onnx/{model_name}.onnx"
    export_model(model=model, _x=img, onnx_filepath=onnx_filepath)
    # torch.onnx.export(model=model, args=img, f=f"models/onnx/{model_name}.onnx", input_names=["image_batch"], output_names=["pred"])

    print("Model exported to model.onnx")


if __name__ == "__main__":
    main()
