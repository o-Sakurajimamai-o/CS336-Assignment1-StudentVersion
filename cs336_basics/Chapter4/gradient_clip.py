import math
import torch
from typing import Iterable, Tuple

def Gradient_cliping(
    params: Iterable[torch.nn.Parameter],
    max_norm: float
):
    eps = 1e-6
    
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return

    g_sqare = sum(torch.sum(grad ** 2) for grad in grads)
    g = math.sqrt(g_sqare)

    if g > max_norm:
        scale = max_norm / (g + eps)
        for p in grads:
            p.data.mul_(scale)
            # p.data = p.data * scale