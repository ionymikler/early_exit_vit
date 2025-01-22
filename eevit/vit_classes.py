import torch
from torch import nn

from einops.layers.torch import Rearrange

from .ee_classes import create_highway_network
from eevit.utils import (
    add_fast_pass,
    remove_fast_pass,
    get_fast_pass,
    get_ee_indexed_params,
)
from utils.logging_utils import get_logger_ready
from utils.arg_utils import ModelConfig

logger = get_logger_ready("vit_classes.py")
debug = False


# helpers
def ensure_tuple(t) -> tuple:
    return t if isinstance(t, tuple) else (t, t)


# def print(f):
#     """
#     workaround for the lookup of the print function when exporting to ONNX
#     """
#     logger.info(f)


def real_print(f):
    print(f)


# classes
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

        q = self.qkv_rearrage(qkv[0])
        k = self.qkv_rearrage(qkv[1])
        v = self.qkv_rearrage(qkv[2])

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

        # second part of TransformerEnconder (after 1st res. connection).
        # Added here for simplicity in iteration of its layers
        out = self.norm_mlp(x) + x  # 2nd residual connection

        out = add_fast_pass(out)  # add the fast-pass token
        return out


class TransformerEnconder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self._create_layers(config)

        self.norm_post_layers = nn.LayerNorm(config.embed_depth)

        print(f"TransformerEnconder initialized with {len(self.layers)} layers")

    def _create_layers(self, config: ModelConfig):
        self.layers = nn.ModuleList()
        ee_params_by_idx = get_ee_indexed_params(config)

        for idx in range(config.num_layers_transformer):
            self.layers.append(Attention(config))

            if idx in ee_params_by_idx.keys():
                assert (
                    len(ee_params_by_idx[idx]) == 2
                ), "Early exit parameters must be a tuple with two elements, besides the ee position index"
                ee_type = ee_params_by_idx[idx][0]
                ee_kwargs = ee_params_by_idx[idx][1]
                hw = create_highway_network(
                    ee_type, config.early_exit_config, ee_kwargs
                )
                print(
                    f"Highway of type '{hw.highway_type}({ee_kwargs})' appended to location '{idx}'"
                )
                self.layers.append(hw)

    def forward(self, x: torch.Tensor):
        x_with_fastpass = add_fast_pass(x)

        for layer_idx in range(len(self.layers)):
            self.layer_idx = layer_idx
            fast_pass_layer = get_fast_pass(x_with_fastpass)

            x_with_fastpass = torch.cond(
                fast_pass_layer.any(),
                self.fast_pass,
                self.layer_forward,
                (x_with_fastpass,),
            )

        return self.norm_post_layers(remove_fast_pass(x_with_fastpass))

    def fast_pass(self, x_with_fastpass: torch.Tensor):
        return x_with_fastpass.clone()

    def layer_forward(self, x_with_fastpass: torch.Tensor):
        module_i = self.layers[self.layer_idx]  # (attn or IC)
        x_with_fastpass = module_i(x_with_fastpass)

        return x_with_fastpass
