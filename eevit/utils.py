#!/usr/bin/env python
# Made by: Jonathan Mikler on 2025-01-20
import torch


def add_fast_pass(x):
    return torch.cat([x, torch.zeros(x.shape[0], 1, x.shape[-1])], dim=1)


def remove_fast_pass(x_with_fastpass: torch.Tensor):
    return x_with_fastpass.clone()[:, :-1, :]


def get_fast_pass(x_with_fastpass: torch.Tensor):
    return x_with_fastpass.clone()[:, -1, :]


def flip_fast_pass_token(x_with_fastpass: torch.Tensor):
    output = x_with_fastpass.clone()
    output[:, -1, :] = 1.0
    return output


def confidence(x):
    softmax = torch.softmax(x, dim=-1)
    # return 0.6
    return torch.max(softmax)
