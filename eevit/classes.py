# Made by: Jonathan Mikler on 2025-01-15

import torch.nn as nn
import math

from .utils import confidence, flip_fast_pass_token
from utils.arg_utils import EarlyExitsConfig
# class ConfidenceExitStrategy(nn.Module):
#     def __init__(self, confidence_threshold:float):
#         super().__init__()
#         self.confidence_threshold = confidence_threshold

#     def forward(self, logits):
#         pred_confidence = confidence(logits)

#         x_with_fastpass = torch.cond(
#             pred_confidence > self.confidence_threshold,
#             flip_fast_pass_token,
#             self.do_nothing,
#             (x_with_fastpass,)
#         )

#         return x_with_fastpass


class DummyIntermediateClassifier(nn.Module):
    def __init__(self, config: EarlyExitsConfig, kwargs: dict):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(config.embed_depth),
            nn.Linear(config.embed_depth, config.num_classes),
        )
        self.confidence_threshold = kwargs["confidence_threshold"]

    def do_nothing(x):
        return x

    def forward(self, x_with_fastpass):
        _logits = self.mlp(x_with_fastpass[:, 0])  # Using CLS token only

        # NOTE: This explicit addition of the confidence strategy is too rigid, but works for now.
        # Should be a class or something that gets configures at creation time
        pred_confidence = confidence(_logits)

        x_with_fastpass = (
            flip_fast_pass_token(x_with_fastpass)
            if pred_confidence > self.confidence_threshold
            else x_with_fastpass
        )
        # x_with_fastpass = torch.cond(
        #     pred_confidence > self.confidence_threshold,
        #     flip_fast_pass_token,
        #     self.do_nothing,
        #     (x_with_fastpass,)
        # )

        return x_with_fastpass


class highway_conv_normal(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, 1, 1),
            nn.GELU(),
            nn.BatchNorm2d(in_features, eps=1e-5),
        )

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv(x)
        x = x.flatten(2).permute(0, 2, 1)
        return x


# Local perception head
class highway_conv1_1(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(hidden_features, eps=1e-5),
        )

        self.proj = nn.Conv2d(
            hidden_features, hidden_features, 3, 1, 1, groups=hidden_features
        )
        self.proj_act = nn.GELU()
        self.proj_bn = nn.BatchNorm2d(hidden_features, eps=1e-5)
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True),
            nn.BatchNorm2d(out_features, eps=1e-5),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x0 = x
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv1(x)
        x = self.drop(x)
        x = self.proj(x)
        x = self.proj_act(x)
        x = self.proj_bn(x)
        x = self.conv2(x)
        x = x.flatten(2).permute(0, 2, 1) + x0
        x = self.drop(x)
        return x


# Local perception head
class highway_conv2_1(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(hidden_features, eps=1e-5),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True),
            nn.BatchNorm2d(out_features, eps=1e-5),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x0 = x
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv1(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = x.flatten(2).permute(0, 2, 1) + x0
        x = self.drop(x)
        return x


# Global aggregation head
class GlobalSparseAttn(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr = sr_ratio
        if self.sr > 1:
            self.sampler = nn.AvgPool2d(1, sr_ratio)
        else:
            self.sampler = nn.Identity()
            self.upsample = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x, H: int, W: int):
        B, N, C = x.shape
        if self.sr > 1.0:
            x = x.transpose(1, 2).reshape(B, C, H, W)
            x = self.sampler(x)
            x = x.flatten(2).transpose(1, 2)

        qkv = (
            self.qkv(x)
            .reshape(B, -1, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
    ):
        super().__init__()
        self.num_heads = num_heads
        # head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

    def forward(self, x):
        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, -1, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)
        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        return x


# Classifier
class HighwayClassifier(nn.Module):
    """Handles the classification part of the highway network"""

    def __init__(self, config: EarlyExitsConfig):
        super().__init__()
        self.config = config
        self.pooler = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(config.general_dropout)

        self.classifier = (
            nn.Linear(config.embed_depth, config.num_classes)
            if config.num_classes > 0
            else nn.Identity()
        )

    def forward(self, processed_embeddings, cls_embeddings):
        pooled_output = (
            self.pooler(processed_embeddings.transpose(1, 2)).transpose(1, 2).squeeze(1)
        )
        logits = self.classifier(pooled_output + cls_embeddings)
        return logits


class IntermediateHeadFactory:
    """Factory class for creating different types of highway networks"""

    @staticmethod
    def create_head(
        highway_type: str, config: EarlyExitsConfig, kwargs: dict
    ) -> nn.Module:
        return HIGHWAY_CLASSES[highway_type](config, kwargs)


class Highway(nn.Module):
    def __init__(self, type: str, config: EarlyExitsConfig, kwargs: dict) -> None:
        super().__init__()
        self.config = config
        self.highway_type = (
            type  # 'module.type' is a reserved attribute in PyTorch class
        )

        # Use factory to create highway network
        self.highway_head = IntermediateHeadFactory.create_head(type, config, kwargs)

        # Create classifier
        self.classifier = HighwayClassifier(config)

    def forward(self, encoder_outputs):
        hidden_states = encoder_outputs[0]
        cls_embeddings = hidden_states[:, 0, :]

        if self.config.backbone == "DeiT":
            distillation_embeddings = hidden_states[:, 1, :]
            patch_embeddings = hidden_states[:, 2:, :]
        elif self.config.backbone == "ViT":
            distillation_embeddings = None
            patch_embeddings = hidden_states[:, 1:, :]

        # Process patch embeddings through highway network
        if self.highway_type == "self_attention":
            processed_embeddings = self.highway_head(patch_embeddings)[0]
        else:
            h = w = int(math.sqrt(patch_embeddings.size()[1]))
            processed_embeddings = self.highway_head(patch_embeddings, h, w)

        # Get logits through classifier
        logits = self.classifier(
            processed_embeddings, cls_embeddings, distillation_embeddings
        )

        return logits, processed_embeddings


# Usage example:
def create_highway_network(
    type: str, config: EarlyExitsConfig, kwargs: dict
) -> Highway:
    """Helper function to create a highway network"""
    return Highway(type, config, kwargs)


HIGHWAY_CLASSES = {
    "dummy": DummyIntermediateClassifier,
    "conv_normal": highway_conv_normal,
    "conv1_1": highway_conv1_1,
    "conv2_1": highway_conv2_1,
    "attention": GlobalSparseAttn,  # TODO: Not sure how to retrieve this ones from the config entries which are not 'attention' but 'attention_r{x}'
}