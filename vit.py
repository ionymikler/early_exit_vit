# Adapted from Github's lucidrains/vit-pytorch/vit_pytorch/vit.py (3.Dec.24)

from enum import Enum
import torch
from torch import nn

from einops.layers.torch import Rearrange

from eevit.classes import create_highway_network
from eevit.utils import add_fast_pass, remove_fast_pass
from utils.logging_utils import get_logger_ready
from utils.arg_utils import ModelConfig

import torch._dynamo.config


torch._dynamo.config.capture_scalar_outputs = True  # for debugging


logger = get_logger_ready("ViT.py")


# helpers
def ensure_tuple(t) -> tuple:
    return t if isinstance(t, tuple) else (t, t)


def print(f):
    """
    workaround for the lookup of the print function when exporting to ONNX
    """
    logger.info(f)


def real_print(f):
    print(f)


# classes
class PoolType(Enum):
    CLS = "cls"
    MEAN = "mean"


class PatchEmbedding(nn.Module):
    def __init__(self, config: ModelConfig, verbose: bool = False):
        super().__init__()
        print("Initializing PatchEmbeddings...")
        self.name = "PatchEmbedding"
        self.verbose = verbose
        self.config = config

        image_height, image_width = ensure_tuple(self.config.image_size)
        patch_height, patch_width = ensure_tuple(self.config.patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)

        self.projection = nn.Conv2d(
            self.config.channels_num,
            self.config.embed_depth,
            kernel_size=self.config.patch_size,
            stride=self.config.patch_size,
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, self.config.embed_depth)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.config.embed_depth))
        self.dropout = nn.Dropout(self.config.general_dropout)

        print(f"PatchEmbedding initialized with {num_patches} patches")

    def forward(self, image_batch: torch.Tensor):
        print(f"image_batch shape: {image_batch.shape}")

        # TODO: Understand this flattening and transpose
        embedded_patches = self.projection(image_batch).flatten(2).transpose(1, 2)
        b, _, _ = embedded_patches.shape

        cls_tokens_batch = self.cls_token.repeat(b, 1, 1)
        embedded_patches = torch.cat((cls_tokens_batch, embedded_patches), dim=1)
        embedded_patches += self.pos_embedding
        embedded_patches = self.dropout(embedded_patches)

        return embedded_patches


class AttentionMLP(nn.Module):
    def __init__(self, embed_depth, hidden_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(embed_depth)
        self.mlp = nn.Sequential(
            nn.Linear(embed_depth, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_depth),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.norm(x)
        return self.mlp(x)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        all_heads_size = config.dim_head * config.num_attn_heads
        project_out = not (
            config.num_attn_heads == 1 and config.dim_head == config.embed_depth
        )  # if more than one head, project out

        self.num_heads = config.num_attn_heads
        self.scale = config.dim_head**-0.5

        self.norm = nn.LayerNorm(
            config.embed_depth
        )  # maps to LGVIT's 'layernorm_before'

        self.scores_softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(config.transformer_dropout)

        self.W_QKV = nn.Linear(config.embed_depth, all_heads_size * 3, bias=True)

        # b: batch size, p: number of patches, h: number of heads, d: depth of head's output
        self.qkv_rearrage = Rearrange("b p (h d) -> b h p d", h=config.num_attn_heads)
        self.attn_rearrange = Rearrange("b h p d -> b p (h d)", h=config.num_attn_heads)

        self.attention_output = (
            nn.Sequential(
                nn.Linear(all_heads_size, config.embed_depth),
                nn.Dropout(config.transformer_dropout),
            )
            if project_out
            else nn.Identity()
        )

        self.norm_mlp = AttentionMLP(
            embed_depth=config.embed_depth,
            hidden_dim=config.mlp_dim,
            dropout=config.transformer_dropout,
        )

    def forward(self, x_with_fastpass):
        x = remove_fast_pass(x_with_fastpass)  # remove the fast-pass token
        x_norm = self.norm(x)
        embed_dim = x_norm.shape[-1]

        qkv = [
            self.W_QKV(x_norm)[:, :, embed_dim * i : embed_dim * (i + 1)]
            for i in range(3)
        ]
        print(f"qkv shape: {[qkv.shape for qkv in qkv]}")

        q = self.qkv_rearrage(qkv[0])
        k = self.qkv_rearrage(qkv[1])
        v = self.qkv_rearrage(qkv[2])
        print(f"Shapes after rearranging: q: {q.shape}, k: {k.shape}, v: {v.shape}")

        scaled_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.scores_softmax(scaled_scores)
        attn = self.dropout(attn)
        attn_by_head = torch.matmul(attn, v)

        # join all heads outputs into a single tensor of shape (batch, num_patches, embed_dim)
        attn_heads_combined = self.attn_rearrange(attn_by_head)

        # counterpart of LGVIT's DeiTSelfOutput.
        # QUESTION: This layer does not appear in ViT originally
        # 1st residual connection
        x = self.attention_output(attn_heads_combined) + x

        # second part of TransformerEnconder (after 1st res. connection),
        # added here for simplicity in iteration of its layers
        out = self.norm_mlp(x) + x  # 2nd residual connection

        out = add_fast_pass(out)  # add the fast-pass token
        return out


class TransformerEnconder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self._create_layers(config)

        self.norm_post_layers = nn.LayerNorm(config.embed_depth)

    def _create_layers(self, config: ModelConfig):
        self.layers = nn.ModuleList()
        ee_params_by_idx = {
            ee[0]: ee[1:] for ee in config.early_exits.exits
        }  # params := [type, kwargs]

        for idx in range(config.num_layers_transformer):
            self.layers.append(Attention(config))

            if idx in ee_params_by_idx.keys():
                ee_type = ee_params_by_idx[idx][0]
                ee_kwargs = ee_params_by_idx[idx][1]
                hw = create_highway_network(ee_type, config.early_exits, ee_kwargs)
                self.layers.append(hw)

    def forward(self, x):
        x_with_fastpass = add_fast_pass(x)

        for layer_idx in range(len(self.layers)):
            self.layer_idx = layer_idx
            print(f"[TransformerEnconder][forward]: Layer {layer_idx}")
            fast_pass_layer = x_with_fastpass[..., -1]

            x_with_fastpass = (
                self.fast_pass(x_with_fastpass)
                if fast_pass_layer.any()
                else self.layer_forward(x_with_fastpass)
            )
            # x_with_fastpass = torch.cond(
            #     fast_pass_layer.any(),  # if the fast pass layer has ones in it
            #     self.fast_pass,
            #     self.layer_forward,
            #     (x_with_fastpass,),
            # )

        return self.norm_post_layers(
            remove_fast_pass(x_with_fastpass)
        )  # Remove the fast-pass token before normalization

    def fast_pass(self, x_with_fastpass):
        # NOTE: Maybe break the graph trace here? tracing here qould be recursive and not sure how much it would pose a problem for dynamo
        return x_with_fastpass

    def layer_forward(self, x_with_fastpass):
        module_i = self.layers[self.layer_idx]  # (attn or IC)
        x_with_fastpass = module_i(x_with_fastpass)

        return x_with_fastpass


class ViT(nn.Module):
    default_config: dict = {
        "pool": PoolType.CLS,
    }

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
        self.name = "ViT"
        print("Initializing Vit model...")
        self.patch_embedding = PatchEmbedding(config, verbose=verbose)

        self.transformer = TransformerEnconder(config)

        self.pool = config.pool
        self.to_latent = nn.Identity()

        self.last_exit = nn.Linear(config.embed_depth, config.num_classes)

        print("ViT model initialized")

    def forward(self, x):
        x = self.patch_embedding(x)

        x = self.transformer(x)

        x = (
            x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        )  # take cls token or average all tokens (pooling)

        x = self.to_latent(x)  # identity, just for shape
        x = self.last_exit(x)
        return x
