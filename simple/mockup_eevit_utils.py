import torch


def add_fast_pass(x):
    return torch.cat([x, torch.zeros(x.shape[0], 1)], dim=1)


def remove_fast_pass(x_with_fastpass: torch.Tensor) -> torch.Tensor:
    return x_with_fastpass.clone()[:, :-1]


def get_fast_pass(x_with_fastpass: torch.Tensor) -> torch.Tensor:
    return x_with_fastpass.clone()[:, -1]


def set_fast_pass_token(x_with_fastpass: torch.Tensor, value: float) -> torch.Tensor:
    output = x_with_fastpass.clone()
    output[:, -1] = value
    return output


def confidence(logits):
    return torch.softmax(logits, dim=1).max()
