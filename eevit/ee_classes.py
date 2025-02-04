# Made by: Jonathan Mikler on 2025-01-15
import torch
import torch.nn as nn
import math

from .utils import (
    confidence,
    set_fast_pass_token,
    remove_fast_pass,
    get_ee_indexed_params,
)
from utils.arg_utils import EarlyExitsConfig, ModelConfig


class ExitEvaluator:
    def __init__(self, ee_config: EarlyExitsConfig, kwargs: dict):
        self.confidence_theshold = ee_config.confidence_threshold

    def decision_tensor(self, logits) -> torch.Tensor:
        return confidence(logits) > self.confidence_theshold


class Highway_DummyMLP(torch.nn.Module):
    def __init__(self, ee_config: ModelConfig, kwargs: dict):
        super(Highway_DummyMLP, self).__init__()
        attention_size = ee_config.embed_depth
        hidden_size = 10

        self.layer1 = torch.nn.Linear(attention_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(hidden_size, attention_size)

    def forward(self, patch_embeddings, H: int, W: int):
        x = self.relu(self.layer1(patch_embeddings))
        x = self.layer2(x)
        return x


## Local Perception Heads
class HighwayConv1_1(nn.Module):
    def __init__(self, ee_config: EarlyExitsConfig, kwargs: dict):
        super().__init__()
        in_features = ee_config.embed_depth
        out_features = kwargs.get("out_feautes", in_features)
        hidden_features = kwargs.get("hidden_features", in_features)

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
        self.drop = nn.Dropout(ee_config.general_dropout)

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


class HighwayConv2_1(nn.Module):
    def __init__(self, ee_config: EarlyExitsConfig, kwargs: dict):
        super().__init__()
        in_features = ee_config.embed_depth
        out_features = kwargs.get("out_feautes", in_features)
        hidden_features = kwargs.get("hidden_features", in_features)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(hidden_features, eps=1e-5),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True),
            nn.BatchNorm2d(out_features, eps=1e-5),
        )
        self.drop = nn.Dropout(ee_config.general_dropout)

    def forward(self, x: torch.Tensor, H, W):
        B, N, C = x.shape
        x0 = x
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv1(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = x.flatten(2).permute(0, 2, 1) + x0
        x = self.drop(x)
        return x


# Global Aggregation Heads
class GlobalSparseAttn(nn.Module):
    # def __init__( self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, sr_ratio=1):
    def __init__(self, ee_config: EarlyExitsConfig, kwargs: dict):
        super().__init__()
        dim = ee_config.embed_depth
        qk_scale = kwargs.get("qk_scale", None)
        qkv_bias = kwargs.get("qkv_bias", False)
        attn_drop = kwargs.get("attn_drop", 0.0)
        proj_drop = kwargs.get("proj_drop", 0.0)
        sr_ratio = kwargs.get("sr_ratio", 1)

        self.num_heads = ee_config.num_attn_heads

        head_dim = dim // ee_config.num_attn_heads
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


# Classifier
class HighwayClassifier(nn.Module):
    """Handles the classification part of the highway network"""

    def __init__(self, ee_config: EarlyExitsConfig):
        super().__init__()
        self.pooler = nn.AdaptiveAvgPool1d(1)

        self.classifier = (
            nn.Linear(ee_config.embed_depth, ee_config.num_classes)
            if ee_config.num_classes > 0
            else nn.Identity()
        )

    def forward(self, processed_embeddings, cls_embeddings):
        pooled_output = (
            self.pooler(processed_embeddings.transpose(1, 2)).transpose(1, 2).squeeze(1)
        )
        logits = self.classifier(pooled_output + cls_embeddings)
        return logits


# Helping classes
class IntermediateHeadFactory:
    """Factory class for creating different types of highway networks"""

    @staticmethod
    def create_head(
        highway_type: str, config: EarlyExitsConfig, kwargs: dict
    ) -> nn.Module:
        return INTERMEDIATE_CLASSES[highway_type](config, kwargs)


# Wrapping class for (IntermediateHeads + Classifier)
class Highway(nn.Module):
    def __init__(
        self, type: str, ee_config: EarlyExitsConfig, kwargs: dict, layer_idx: int
    ) -> None:
        super().__init__()
        self.config = ee_config
        self.highway_type = type
        self.init_kwargs = kwargs
        self.layer_idx = layer_idx

        # Use factory to create highway network
        self.highway_head = IntermediateHeadFactory.create_head(type, ee_config, kwargs)

        # Create classifier
        self.classifier = HighwayClassifier(ee_config)

        # softmax early prediction
        self.softmax = nn.Softmax(dim=1)

        # Exit strategy Evaluation
        # NOTE: Maybe the Evaluator does not need to be here but at TransformerEnconder level,
        # for ease of implementing more complex strategies. For now here.
        self.exit_evaluator = ExitEvaluator(ee_config, kwargs)

    @staticmethod
    def from_model_config(config: ModelConfig, idx: int):
        ee_params_by_idx = get_ee_indexed_params(config)
        assert (
            len(ee_params_by_idx[idx]) == 2
        ), "Early exit parameters must be a tuple with two elements, besides the ee position index"
        ee_type = ee_params_by_idx[idx][0]
        ee_kwargs = ee_params_by_idx[idx][1]

        hw = Highway(ee_type, config.early_exit_config, ee_kwargs, idx)
        return hw

    def forward(
        self,
        x_with_fastpass: torch.Tensor,
        predictions_placeholder_tensor: torch.Tensor,
    ):
        hidden_states = remove_fast_pass(x_with_fastpass)

        cls_embeddings = hidden_states.clone()[:, 0, :]

        patch_embeddings = hidden_states.clone()[:, 1:, :]

        # Process patch embeddings through highway network
        h = w = int(math.sqrt(patch_embeddings.size()[1]))
        processed_embeddings = self.highway_head(patch_embeddings, h, w)

        # Get logits through classifier
        logits = self.classifier(processed_embeddings, cls_embeddings)

        predictions = self.softmax(logits)
        # predictions_placeholder_tensor = self.softmax(logits).clone()

        # Update predictions tensor with both predictions and layer index
        batch_size = predictions.shape[0]
        layer_idx_tensor = torch.full(
            (batch_size, 1), self.layer_idx, dtype=predictions.dtype
        )

        # Concatenate predictions with layer index
        predictions_with_idx = torch.cat([predictions, layer_idx_tensor], dim=1)

        x_with_fastpass = set_fast_pass_token(
            x_with_fastpass,
            value=self.exit_evaluator.decision_tensor(logits).to(
                dtype=x_with_fastpass.dtype
            ),
        )

        return x_with_fastpass, predictions_with_idx


class IdentityHighway(nn.Module):
    """A pass-through highway layer that maintains tensor shapes and types."""

    def __init__(self, layer_idx: int = -1):
        super().__init__()
        self.layer_idx = layer_idx

    def forward(
        self,
        x_with_fastpass: torch.Tensor,
        predictions_placeholder_tensor: torch.Tensor,
    ):
        return x_with_fastpass.clone(), predictions_placeholder_tensor.clone()


class HighwayWrapper(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.name = "HighwayWrapper"

        # Create EarlyExitsConfig from ModelConfig
        ee_config = EarlyExitsConfig(
            embed_depth=config.embed_depth,
            num_classes=config.num_classes,
            confidence_threshold=0.8,  # Default value
            general_dropout=config.general_dropout,
            num_attn_heads=config.num_attn_heads,
        )

        # Create Highway module
        self.highway = Highway(
            type="dummy_mlp",  # You can change this to any supported type
            ee_config=ee_config,
            kwargs={},
        )

    def forward(self, x):
        return self.highway(x)


INTERMEDIATE_CLASSES = {
    "dummy_mlp": Highway_DummyMLP,
    "conv1_1": HighwayConv1_1,
    "conv2_1": HighwayConv2_1,
    "attention": GlobalSparseAttn,
}
