import torch
from torch import nn

from .linear import Linear

class FFN(nn.Module):
    def __init__(self, d_model: int, mul: int = 64):
        super().__init__()

        self.d_model = d_model
        # compute d_ff. and round this to a nearby multiple of 64 for hardware efficiency
        hidden = int(8.0 / 3.0 * d_model)
        self.d_ff = mul * ((hidden + mul - 1) // mul)

        self.w1 = Linear(self.d_ff, self.d_model)
        self.w2 = Linear(self.d_model, self.d_ff)
        self.w3 = Linear(self.d_ff, self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FFN(x) = W2( SiLU(xW1) ⊙ xW3 )
        x1 = self.w1(x) # no bias in out Linear
        SiLU = x1 * torch.sigmoid(x1)
        x3 = self.w3(x)
        
        return self.w2(SiLU * x3)
