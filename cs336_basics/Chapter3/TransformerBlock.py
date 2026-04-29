import torch
from torch import nn

from .ffn import FFN
from .multihead_self_attention import MultiHeadSelfAttention
from .RMSNorm import RMSNorm

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        use_rope: bool,
        theta: float,
    ):
       super().__init__()

       self.d_ff = d_ff 
       self.d_model = d_model
       self.num_heads = num_heads
       self.RMSNorm1 = RMSNorm(d_model)
       self.MHA = MultiHeadSelfAttention(
           d_model,
           num_heads,
           max_seq_len,
           use_rope,
           theta
       )

       self.ffn = FFN(d_model)
       self.RMSNorm2 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        device = x.device
        # first we should get tokenpositions, because adapters' function doesn't have it
        base_positions = torch.arange(seq_len, device=device)        
        token_positions = base_positions.unsqueeze(0).expand(batch_size, seq_len)

        # rmsnorm = RMSNorm(self.d_model)
        y = x + self.MHA(self.RMSNorm1(x), token_positions)

        return y + self.ffn(self.RMSNorm2(y))
