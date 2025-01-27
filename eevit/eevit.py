# Adapted from Github's lucidrains/vit-pytorch/vit_pytorch/vit.py (3.Dec.24)

import torch
from torch import nn

from .vit_classes import PatchEmbedding, TransformerEnconder
from utils.arg_utils import ModelConfig

import torch._dynamo.config


torch._dynamo.config.capture_scalar_outputs = True  # Not sure if needed


class EEVIT(nn.Module):
    def __init__(self, *, config: ModelConfig, verbose: bool = False):
        # * to enforce only keyword arguments
        """
        Initializes the Vision Transformer (ViT) model.

        Args:
            image_size (int or tuple): Size of the input image. If an int is provided, it is assumed to be the size of both dimensions.
            patch_size (int or tuple): Size of the patches to be extracted from the input image. If an int is provided, it is assumed to be the size of both dimensions.
            num_classes (int): Number of output classes.
            embed_depth (int): Dimension of the embeddings.
            transformer_layers (int): Number of transformer layers.
            heads (int): Number of attention heads.
            mlp_dim (int): Dimension of the MLP (Feed-Forward) layer.
            pool (str, optional): Pooling type, either 'cls' (class token) or 'mean' (mean pooling). Default is 'cls'.
            channels (int, optional): Number of input channels. Default is 3.
            dim_head (int, optional): Depth dimension of the attention matrices in each head. Default is 64.
            dropout (float, optional): Dropout rate for the transformer. Default is 0.
            dropout_embedding (float, optional): Dropout rate for the embedding layer. Default is 0.
        """
        super().__init__()
        self.name = "EEVIT"
        print("Initializing Vit model...")
        self.patch_embedding = PatchEmbedding(config, verbose=verbose)

        self.transformer = TransformerEnconder(config)

        self.pool = config.pool
        self.to_latent = nn.Identity()

        self.last_exit = nn.Linear(config.embed_depth, config.num_classes)

        print("ViT model initialized")

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(image_tensor)

        x, predictions = self.transformer(x)

        x = (
            x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        )  # take cls token or average all tokens (pooling)

        x = self.to_latent(x)  # TODO: Review why this is done in LGVIT
        x = self.last_exit(x)
        return x
