import torch
from torch import nn
from cs336_basics.linear import Linear
from cs336_basics.softmax import SoftMax
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.embedding import Embedding
from cs336_basics.TransformerBlock import TransformerBlock

class TransformerLM(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
    ):
        super().__init__()

        self.embedding = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model,
                num_heads,
                d_ff,
                max_seq_len=context_length,
                use_rope=True,
                theta=rope_theta
            ) for _ in range(num_layers)
        ])

        self.Norm = RMSNorm(d_model)
        self.LM = Linear(d_model, vocab_size)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embedding(x)
        for block in self.layers:
            h = block(h)
    
        return self.LM(self.Norm(h))