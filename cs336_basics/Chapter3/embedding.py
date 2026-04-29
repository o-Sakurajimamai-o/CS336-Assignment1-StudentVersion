import torch
from torch import nn

class Embedding(nn.Module):
    """A small embedding table wrapper with CS336-style initialization."""

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(
            self.weight,
            mean=0,
            std=1,
            a=-3,
            b=3
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return the embedding vectors indexed by token_ids."""
        return self.weight[token_ids]
