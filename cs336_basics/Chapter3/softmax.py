import torch

def SoftMax(x: torch.Tensor, dimension: int) -> torch.Tensor:

    value = x.max(dim=dimension, keepdim=True).values
    x = x - value
    exp = torch.exp(x).sum(dim=dimension, keepdim=True)

    softmax = torch.exp(x) / exp
    return softmax

def SoftMax_with_Temperature(
    x: torch.Tensor, 
    dimension: int, 
    temperature: float
) -> torch.Tensor:

    if temperature == 0.0:
        index = x.argmax(dim=dimension, keepdim=True)
        return torch.zeros_like(x).scatter_(dimension, index, 1.0)

    value = x.max(dim=dimension, keepdim=True).values
    x = torch.exp((x - value) / temperature)

    exp = x.sum(dim=dimension, keepdim=True)

    softmax = x / exp
    return softmax