import torch

def cropss_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor
) -> torch.Tensor:
    value = logits.max(dim=-1, keepdim=True).values
    logits = logits - value

    exp = torch.exp(logits).sum(dim=-1, keepdim=True)
    logits = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1))

    loss = -logits.unsqueeze(-1) + torch.log(exp)
    return loss.mean()