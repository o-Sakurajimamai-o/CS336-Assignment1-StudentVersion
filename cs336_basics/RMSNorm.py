import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(
            self, 
            d_model: int, 
            eps: float = 1e-5,
            device = None,
            dtype = None
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # here we get RMS(a)
        rms = (x ** 2).sum(dim=-1, keepdim=True) / self.d_model
        rms = torch.sqrt(rms + self.eps)
        rmsnorm = x / rms * self.weight

        return rmsnorm.to(in_dtype)


