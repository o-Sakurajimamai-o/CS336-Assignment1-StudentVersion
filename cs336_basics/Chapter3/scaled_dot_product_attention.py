import math
import torch

from .softmax import SoftMax

def dot_product_attention(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        Mask: torch.Tensor | None = None,
        device = None,
        dtype = None
) -> torch.Tensor:
    score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.shape[-1])

    if Mask is not None:
        score = score.masked_fill(Mask==False, float('-inf'))
    
    Att = SoftMax(score, -1)
    return torch.matmul(Att, V)
