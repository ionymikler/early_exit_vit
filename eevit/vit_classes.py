import torch
from torch import nn

from einops.layers.torch import Rearrange

from .ee_classes import Highway, IdentityHighway
from eevit.utils import (
    add_fast_pass,
    remove_fast_pass,
    get_fast_pass,
    get_ee_indexed_params,
    set_fast_pass_token,
)
from utils.logging_utils import get_logger_ready
from utils.arg_utils import ModelConfig

logger = get_logger_ready("vit_classes.py")
debug = False


def ensure_tuple(t) -> tuple:
    return t if isinstance(t, tuple) else (t, t)


# classes
class PatchEmbedding(nn.Module):
    def __init__(self, config: ModelConfig, verbose: bool = False):
        super().__init__()
        # print("Initializing PatchEmbeddings...")
        logger.info("Initializing PatchEmbeddings...")
        self.name = "PatchEmbedding"
        self.verbose = verbose
        self.config = config

        image_height, image_width = ensure_tuple(self.config.image_size)
        patch_height, patch_width = ensure_tuple(self.config.patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.num_patches = num_patches
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

        if self.verbose:
            logger.info(
                f"PatchEmbedding initialized with {num_patches + 1} patches (including the cls token)"
            )

    @property
    def input_shape(self):
        """
        Returns a sample input tensor for the PatchEmbedding layer.
        The sample input tensor has shape (batch_size, channels_num, image_height, image_width).
        """
        return (
            1,
            self.config.channels_num,
            self.config.image_size,
            self.config.image_size,
        )

    @property
    def output_shape(self):
        """
        Returns the output shape of the PatchEmbedding layer.
        The patch embedding layer outputs a tensor of shape (batch_size, num_patches + 1, embed_dim).
        """
        inpt = torch.randn(self.input_shape)
        return self.forward(inpt).shape

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        # TODO: Understand this flattening and transpose
        embedded_patches = self.projection(image_batch).flatten(2).transpose(1, 2)
        b, _, _ = embedded_patches.shape

        cls_tokens_batch = self.cls_token.repeat(b, 1, 1)
        embedded_patches = torch.cat((cls_tokens_batch, embedded_patches), dim=1)
        embedded_patches = embedded_patches + self.pos_embedding
        embedded_patches = self.dropout(embedded_patches)

        return embedded_patches


class AttentionMLPs(nn.Module):
    def __init__(self, embed_depth, hidden_dim, dropout=0.0):
        super().__init__()
        self.norm_2 = nn.LayerNorm(embed_depth)

        self.mlp_intermediate = nn.Sequential(
            nn.Linear(embed_depth, hidden_dim), nn.GELU()
        )

        self.mlp_output = nn.Sequential(
            nn.Linear(hidden_dim, embed_depth),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.norm_2(x)  # norm after attention

        x_int = self.mlp_intermediate(x_norm)
        x = self.mlp_output(x_int) + x  # 2nd residual connection
        return x


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, highway: Highway = IdentityHighway()):
        super().__init__()

        all_heads_size = config.dim_head * config.num_attn_heads
        project_out = not (
            config.num_attn_heads == 1 and config.dim_head == config.embed_depth
        )  # if more than one head, project out

        self.num_heads = config.num_attn_heads
        self.scale = config.dim_head**-0.5

        self.norm_1 = nn.LayerNorm(
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

        # This contains inside the MLP of LGVIT's 'DeiTIntermediate' and 'DeiTSelfOutput'
        self.mlps = AttentionMLPs(
            embed_depth=config.embed_depth,
            hidden_dim=config.mlp_dim,
            dropout=config.transformer_dropout,
        )

        self.highway = highway

    def forward(
        self,
        x_with_fastpass: torch.Tensor,
        predictions_placeholder_tensor: torch.Tensor,
    ):
        x = remove_fast_pass(x_with_fastpass)  # remove the fast-pass token
        x_norm = self.norm_1(x)
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
        # NOTE: 1st residual connection
        x = self.attention_output(attn_heads_combined) + x

        # second part of TransformerEnconder (after 1st res. connection).
        # NOTE: 2nd residual connection happens inside self.mlps
        x = self.mlps(x)

        # Highway
        x_with_fastpass = add_fast_pass(x)
        x_with_fastpass, predictions_placeholder_tensor = self.highway(
            x_with_fastpass, predictions_placeholder_tensor
        )

        return x_with_fastpass, predictions_placeholder_tensor


class TransformerEnconder(nn.Module):
    def __init__(self, config: ModelConfig, verbose: bool = False):
        super().__init__()
        self.verbose: bool = verbose
        self.early_exits_enabled = config.early_exit_config.enabled
        self.exportable = True if config.enable_export else False

        self._create_layers(config)
        self.num_classes = config.num_classes

        self.norm_post_layers = nn.LayerNorm(config.embed_depth)

        logger.info(
            f"TransformerEnconder initialized with {len(self.layers)} layers and {len(config.early_exit_config.exits)} early exits"
        )

    def _create_layers(self, config: ModelConfig):
        self.layers = nn.ModuleList()
        self.ee_params_by_idx = get_ee_indexed_params(config)

        for idx in range(config.num_layers_transformer):
            if idx in self.ee_params_by_idx.keys():
                hw = Highway.from_model_config(config, idx)

                if self.verbose:
                    logger.info(
                        f"Highway of type '{hw.highway_type}({hw.init_kwargs})' appended to location '{idx}'"
                    )

                self.layers.append(Attention(config, hw))
            else:
                self.layers.append(Attention(config))

    def fast_pass(
        self,
        x_with_fastpass: torch.Tensor,
        predictions_placeholder_tensor: torch.Tensor,
    ):
        return x_with_fastpass.clone(), predictions_placeholder_tensor.clone()

    def forward(self, x: torch.Tensor):
        x_with_fastpass = add_fast_pass(x)

        # initial predictions tensor with a uniform distribution
        predictions_placeholder_tensor = torch.cat(
            [
                torch.full((1, self.num_classes), 1.0 / self.num_classes),
                torch.zeros(1, 1),
            ],
            dim=1,
        )

        i = 0
        for layer in self.layers:
            fast_pass_layer = get_fast_pass(x_with_fastpass)

            #### CONDITIONAL ####
            if self.exportable:
                x_with_fastpass, predictions_placeholder_tensor = torch.cond(
                    fast_pass_layer.any(),
                    self.fast_pass,
                    layer,
                    (x_with_fastpass, predictions_placeholder_tensor),
                )
            else:
                x_with_fastpass, predictions_placeholder_tensor = (
                    self.fast_pass(x_with_fastpass, predictions_placeholder_tensor)
                    if (fast_pass_layer.any() and self.early_exits_enabled)
                    else layer(x_with_fastpass, predictions_placeholder_tensor)
                )
            # //CONDITIONAL
            i += 1
        fp = get_fast_pass(x_with_fastpass)
        x_norm = self.norm_post_layers(remove_fast_pass(x_with_fastpass))
        x_with_fastpass = set_fast_pass_token(add_fast_pass(x_norm), fp)

        return x_with_fastpass, predictions_placeholder_tensor
