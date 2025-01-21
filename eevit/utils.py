#!/usr/bin/env python
# Made by: Jonathan Mikler on 2025-01-20
import torch


def add_fast_pass(x):
    return torch.cat([x, torch.zeros(x.shape[0], 1, x.shape[-1])], dim=1)


def remove_fast_pass(x_with_fastpass):
    return x_with_fastpass[:, :-1, :]


def get_fast_pass(x_with_fastpass):
    return x_with_fastpass[:, -1, :]


def flip_fast_pass_token(x_with_fastpass):
    x_with_fastpass[:, -1, :] = 1.0
    return x_with_fastpass


def confidence(x):
    # x: torch.Tensor, logits BEFORE softmax
    softmax = torch.softmax(x, dim=-1)
    # print("WARNING: confidence is giving a dummy value")
    # return 0.6
    return torch.max(softmax)
