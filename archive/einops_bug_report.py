import torch
from einops.layers.torch import Rearrange
from typing import Tuple

import torch.nn as nn


def pair(t: int) -> Tuple[int, int]:
    return t, t


class PatchEmbeddingSimple(nn.Module):
    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        embed_depth: int,
        pool: str,
        channels: int,
    ):
        super().__init__()
        self.name = "PatchEmbedding"

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        patch_dim = channels * patch_height * patch_width

        self.rearrange = Rearrange(
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=patch_height,
            p2=patch_width,
        )
        self.patch_embedding_linear = nn.Linear(patch_dim, embed_depth)

        self.seq = nn.Sequential(self.rearrange, self.patch_embedding_linear)

    def forward(self, image_batch: torch.Tensor):
        out = self.seq(image_batch)
        return out


def export_model(model: nn.Module, _x, onnx_filepath: str):
    print(f"Exporting model '{model.name}' to ONNX format")

    # model = torch.jit.script(model)
    torch.onnx.export(
        model=model,
        args=_x,
        f=onnx_filepath,
        verbose=False,
        opset_version=20,
        dynamo=False,
    )

    print(f"✅ Model exported to '{onnx_filepath}'")

    return onnx_filepath


def main():
    model_config = {
        "channels_num": 3,
        "image_size": 256,
        "pool": "cls",
        "embed_depth": 1024,
        "patch_size": 32,
    }
    model = PatchEmbeddingSimple(
        image_size=model_config["image_size"],
        patch_size=model_config["patch_size"],
        embed_depth=model_config["embed_depth"],
        pool=model_config["pool"],
        channels=model_config["channels_num"],
    )
    x = torch.randn(
        (
            2,
            model_config["channels_num"],
            model_config["image_size"],
            model_config["image_size"],
        )
    )
    onnx_filepath = f"./{model.name}_dynamo.onnx"
    export_model(model=model, _x=x, onnx_filepath=onnx_filepath)


if __name__ == "__main__":
    main()
