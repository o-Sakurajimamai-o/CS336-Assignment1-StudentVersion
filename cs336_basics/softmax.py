import torch

def SoftMax(x: torch.Tensor, dimension: int) -> torch.Tensor:

    value = x.max(dim=dimension, keepdim=True).values
    x = x - value
    exp = torch.exp(x).sum(dim=dimension, keepdim=True)

    softmax = torch.exp(x) / exp
    return softmax