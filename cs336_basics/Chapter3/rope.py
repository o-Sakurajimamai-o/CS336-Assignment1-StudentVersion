import torch
from torch import nn

"""_summary_

this algorithm is pretty hard to understand for the first time
here I want to explain something to it

first, we konw the rotation is (x, y) * (cos, sin // -sin cos) = (xcos-ysin, xsin+ycos)
we get rotate_half here, which means rotate [x, y] -> [-y, x]
so we can use rotate_half to forward: (x, y) dot (cos, cos) + rotate_half((x, y)) dot (sin, sin)

"""


class rope(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        power = torch.arange(0, d_k, 2, dtype=torch.float32, device=device)

        # freq shape: (d_k // 2, )
        freq = 1 / (theta ** (power / d_k))

        t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        # outter product -> pos * theta
        # freqs shape: (max_seq_len, d_k // 2)
        freqs = torch.outer(t, freq)
        freqs = torch.repeat_interleave(freqs, repeats=2, dim=-1)
        
        self.register_buffer("cos_cache", freqs.cos(), persistent=False)
        self.register_buffer("sin_cache", freqs.sin(), persistent=False)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x_paired = x.unflatten(-1, (-1, 2))
        x1, x2 = x_paired.unbind(dim=-1)
        x_rotated = torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)

        return x_rotated
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cache[token_positions].to(x.dtype)
        sin = self.sin_cache[token_positions].to(x.dtype)

        # Make cached angles broadcast across any extra attention-head dims
        # while keeping the sequence dimension aligned with x[..., seq_len, d_k].
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        return (x * cos) + (self.rotate_half(x) * sin)





