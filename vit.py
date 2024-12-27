# Adapted from Github's lucidrains/vit-pytorch/vit_pytorch/vit.py (3.Dec.24)

from enum import Enum
import torch
from torch import nn

from einops.layers.torch import Rearrange

from utils.logging_utils import get_logger_ready


logger = get_logger_ready("ViT")


# helpers
def ensure_tuple(t) -> tuple:
    return t if isinstance(t, tuple) else (t, t)


def print(f):
    """
    workaround for the lookup of the print function when exporting to ONNX
    """
    logger.info(f)


# classes
class PoolType(Enum):
    CLS = "cls"
    MEAN = "mean"


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        config: dict,
    ):
        super().__init__()
        self.name = "PatchEmbedding"
        self._set_config_params(config)

        image_height, image_width = ensure_tuple(self.config["image_size"])
        patch_height, patch_width = ensure_tuple(self.config["patch_size"])
        print(f"image_height: {image_height}, image_width: {image_width}")
        print(f"patch_height: {patch_height}, patch_width: {patch_width}")

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = self.config["channels"] * patch_height * patch_width
        assert self.config["pool"] in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.patch_embedding_layers = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, self.config["embed_depth"]),
            nn.LayerNorm(self.config["embed_depth"]),
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, self.config["embed_depth"])
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.config["embed_depth"]))
        self.dropout = nn.Dropout(self.config["dropout_embedding"])

    def _set_config_params(self, config: dict):
        assert isinstance(config["image_size"], int) or isinstance(
            config["image_size"], tuple
        )
        assert isinstance(config["patch_size"], int) or isinstance(
            config["patch_size"], tuple
        )
        assert isinstance(config["embed_depth"], int)
        assert isinstance(config["pool"], str)
        assert isinstance(config["channels_num"], int)
        assert isinstance(config["dropout_embedding"], float)

        self.config = {}
        self.config["image_size"] = config["image_size"]
        self.config["patch_size"] = config["patch_size"]
        self.config["embed_depth"] = config["embed_depth"]
        self.config["pool"] = config["pool"]
        self.config["channels"] = config["channels_num"]
        self.config["dropout_embedding"] = config["dropout_embedding"]

    def __call__(self, *args, **kwds) -> torch.Tensor:
        return super().__call__(*args, **kwds)

    def forward(self, image_batch: torch.Tensor):
        print(f"image_batch shape: {image_batch.shape}")
        embedded_patches = self.patch_embedding_layers(image_batch)
        b, _, _ = embedded_patches.shape

        cls_tokens_batch = self.cls_token.repeat(b, 1, 1)
        embedded_patches = torch.cat((cls_tokens_batch, embedded_patches), dim=1)
        embedded_patches += self.pos_embedding

        embedded_patches = self.dropout(embedded_patches)
        return embedded_patches


class AttentionFeedForward(nn.Module):
    def __init__(self, embed_depth, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_depth),
            nn.Linear(embed_depth, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_depth),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, *, embed_depth, mlp_dim, num_heads=8, dim_head=64, dropout=0.0):
        super().__init__()

        inner_dim = dim_head * num_heads
        project_out = not (
            num_heads == 1 and dim_head == embed_depth
        )  # if more than one head, project out

        self.num_heads = num_heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(embed_depth)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(
            embed_depth, inner_dim * 3, bias=False
        )  # QUESTION: Why bias=False?
        self.qkv_rearrage = Rearrange("b n (h d) -> b h n d", h=num_heads)
        self.out_rearrange = Rearrange("b h n d -> b n (h d)", h=num_heads)

        self.to_ff = (
            nn.Sequential(nn.Linear(inner_dim, embed_depth), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

        self.feed_forward = AttentionFeedForward(
            embed_depth=embed_depth, hidden_dim=mlp_dim, dropout=dropout
        )

    def forward(self, x):
        x = self.norm(x)
        embed_dim = x.shape[-1]

        # qkv = self.to_qkv(x).chunk(3, dim = -1)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        qkv_i = [
            self.to_qkv(x)[:, :, embed_dim * i : embed_dim * (i + 1)] for i in range(3)
        ]
        print(f"Shapes after slicing: {[qkv.shape for qkv in qkv_i]}")
        # q, k, v = map(self.qkv_rearrage, qkv_i)

        # qkv_rearrage = Rearrange("b n (h d) -> b h n d", h=self.num_heads)
        q = self.qkv_rearrage(qkv_i[0])
        k = self.qkv_rearrage(qkv_i[1])
        v = self.qkv_rearrage(qkv_i[2])
        print(f"Shapes after rearranging: q: {q.shape}, k: {k.shape}, v: {v.shape}")

        scaled_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.softmax(scaled_scores)
        attn = self.dropout(attn)

        attn_out = torch.matmul(attn, v)

        # out = rearrange(
        #     out, "b h n d -> b n (h d)"
        # )
        attn_out = self.out_rearrange(
            attn_out
        )  # TODO: Review this again. Understand the rearrange function
        print(f"Shape after attention: {attn_out.shape}")

        x = self.to_ff(attn_out)

        out = self.feed_forward(x)

        return out


class Transformer(nn.Module):
    def __init__(
        self, *, embed_depth, num_layers, num_attn_heads, dim_head, mlp_dim, dropout=0.0
    ):
        super().__init__()
        self.norm = nn.LayerNorm(embed_depth)

        self.layers = nn.ModuleList(
            [
                Attention(
                    embed_depth=embed_depth,
                    num_heads=num_attn_heads,
                    dim_head=dim_head,
                    dropout=dropout,
                    mlp_dim=mlp_dim,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        _l_idx = 0
        for attn in self.layers:
            print(f"[forward]: Layer {_l_idx}")
            _l_idx += 1

            x = attn(x) + x

        return self.norm(x)


class ViT(nn.Module):
    default_config: dict = {
        "pool": PoolType.CLS,
    }

    def __init__(
        self,
        *,
        config: dict,
    ):
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
        self._set_config_params(config)

        self.patch_embedding = PatchEmbedding(config)

        self.transformer = Transformer(
            embed_depth=self.config["embed_depth"],
            num_layers=self.config["num_layers_transformer"],
            num_attn_heads=self.config["num_attn_heads"],
            dim_head=self.config["dim_head"],
            mlp_dim=self.config["mlp_dim"],
            dropout=self.config["dropout_transformer"],
        )

        self.pool = self.config["pool"]
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(
            self.config["embed_depth"], self.config["num_classes"]
        )

    def _set_config_params(self, config: dict):
        assert isinstance(config["image_size"], int) or isinstance(
            config["image_size"], tuple
        ), "image_size must be an int or a tuple"
        assert isinstance(config["patch_size"], int) or isinstance(
            config["patch_size"], tuple
        ), "patch_size must be an int or a tuple"

        assert isinstance(config["embed_depth"], int), "embed_depth must be an int"
        assert isinstance(config["pool"], str), "pool must be a string"
        assert isinstance(config["channels_num"], int)
        assert isinstance(config["dropout_embedding"], float)

        self.config = {}
        self.config["image_size"] = config["image_size"]
        self.config["patch_size"] = config["patch_size"]
        self.config["embed_depth"] = config["embed_depth"]
        self.config["pool"] = config.get("pool", ViT.default_config["pool"])
        self.config["channels"] = config["channels_num"]
        self.config["dropout_embedding"] = config["dropout_embedding"]

        self.config["num_layers_transformer"] = config["num_layers_transformer"]
        self.config["num_attn_heads"] = config["num_attn_heads"]
        self.config["dim_head"] = config["dim_head"]
        self.config["mlp_dim"] = config["mlp_dim"]
        self.config["dropout_transformer"] = config["dropout_transformer"]
        self.config["num_classes"] = config["num_classes"]

    def forward(self, x):
        x = self.patch_embedding(x)

        x = self.transformer(x)

        x = (
            x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        )  # take cls token or average all tokens (pooling)

        x = self.to_latent(x)  # identity, just for shape
        x = self.mlp_head(x)
        return x
