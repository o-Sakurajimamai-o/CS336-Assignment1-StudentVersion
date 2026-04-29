import math
import torch
from torch import nn

class Linear(nn.Module):
    """A bias-free linear layer used by the CS336 transformer components."""

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )

        # CS336 uses truncated normal initialization scaled by fan-in and fan-out.
        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(
            self.weight, 
            mean=0, 
            std=std, 
            a=-3 * std, 
            b=3 * std
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply x @ W.T while preserving any leading batch dimensions."""
        return torch.matmul(x, self.weight.transpose(-1, -2))
