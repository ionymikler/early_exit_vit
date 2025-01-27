#!/usr/bin/env python
# Made by: Jonathan Mikler on 2025-01-20
from typing import Dict
import torch
from utils.arg_utils import ModelConfig


def get_ee_indexed_params(model_config: ModelConfig) -> Dict[int, list]:
    return {
        ee[0]: ee[1:] for ee in model_config.early_exit_config.exits
    }  # int -> [type, kwargs]


def add_fast_pass(x):
    return torch.cat([x, torch.zeros(x.shape[0], 1, x.shape[-1])], dim=1)


def remove_fast_pass(x_with_fastpass: torch.Tensor) -> torch.Tensor:
    return x_with_fastpass.clone()[:, :-1, :]


def get_fast_pass(x_with_fastpass: torch.Tensor) -> torch.Tensor:
    return x_with_fastpass.clone()[:, -1, :]


def set_fast_pass_token(x_with_fastpass: torch.Tensor, value: float) -> torch.Tensor:
    # assert contains([0, 1],value), "Value must be either 0 or 1"
    output = x_with_fastpass.clone()
    output[:, -1, :] = value
    return output


def set_fast_pass(x: torch.Tensor, value: float = 0.0) -> torch.Tensor:
    """
    Adds a fast-pass token to the input tensor with the specified value.

    Args:
        x (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_dim]
        value (float, optional): Value for the fast-pass token. Defaults to 0.0

    Returns:
        torch.Tensor: Tensor with fast-pass token
    """
    # NOTE: Does not work yey
    return torch.cat([x, torch.full((x.shape[0], 1, x.shape[-1]), value)], dim=1)


def confidence(x: torch.Tensor) -> float:
    softmax = torch.softmax(x, dim=-1)
    # return 0.6
    return torch.max(softmax)
