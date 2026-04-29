import torch
from torch import nn

from .linear import Linear
from .rope import rope
from .scaled_dot_product_attention import dot_product_attention

class MultiHeadSelfAttention(nn.Module):

    def __init__(
        self, 
        d_model: int, 
        num_heads: int,
        max_seq_len: int,
        use_rope: bool,
        theta: float = 10000.0
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        # init weights
        self.w_q = Linear(d_model, d_model)
        self.w_k = Linear(d_model, d_model)
        self.w_v = Linear(d_model, d_model)
        self.w_o = Linear(d_model, d_model)

        self.use_rope = use_rope
        self.rope = rope(theta, self.d_k, max_seq_len) 

    def forward(
            self, 
            x: torch.Tensor, 
            token_position: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # shape Q[Batch, seq, d_model]
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
            
        # so we first get sliced Q, K, V, Q[b, seq, heads, dk]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k)  
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k)    
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k)    

        # second we transpose Q to [Batch, num_heads, seq_len, d_k]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        if self.use_rope:
            Q = self.rope(Q, token_position)
            K = self.rope(K, token_position)

        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        attention = dot_product_attention(Q, K, V, mask)
        attention = attention.transpose(1, 2)
        attention = attention.contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.w_o(attention)
