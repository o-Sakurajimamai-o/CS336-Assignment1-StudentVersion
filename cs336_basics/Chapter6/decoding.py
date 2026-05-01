import torch
from cs336_basics.Chapter3.softmax import SoftMax_with_Temperature

"""
Deliverable: Implement a function to decode from your language model.
We recommend that you support the following features:
    • Generate completions for a user-provided prompt (i.e., take in some x1…t and
        sample a completion until you hit an <|endoftext|> token).
    • Allow the user to control the maximum number of generated tokens.
    • Given a desired temperature value, apply softmax temperature scaling to
        the predicted nexttoken distributions before sampling.
    • Top-p sampling ([A. Holtzman et al., 2020] also referred to as nucleus sampling),
        given a userspecified threshold value.


here, we should take prompt max_gen_tokens, temperature, top_p from user

"""

def decode(
    model: torch.nn.Module,
    prompt: torch.Tensor, #shape: [1, seq_len]
    max_gen_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 0.9,
    eos: int = 256
) -> torch.Tensor:

    model.eval()

    seq = prompt
    for _ in range(max_gen_tokens):
        # model(seq) return a shape [1, seq_len, vocab_size] of tensor
        logits = model(seq)[:, -1, :] #shape: [1, vocab_size]

        softmaxde_logits = SoftMax_with_Temperature(logits, -1, temperature)
        sorted_logits, sorted_index = torch.sort(softmaxde_logits, dim=-1, descending=True)


        # cumsum: [a, b, c] -> [a, a + b, a + b + c]
        # tensor1 > tensor2 -> return a tensor which dtype is bool, example: [1, 2, 3] > [0, 2, 1] -> [false, false, true]
        # [..., 1:] = [..., -1] means [F, F, T, T] -> [F, F, F, T], means left one iter
        sum_logits = torch.cumsum(sorted_logits, dim=-1)
        useless_logits = sum_logits > top_p
        useless_logits[..., 1:] = useless_logits[..., :-1].clone()
        useless_logits[..., 0] = False

        sorted_logits[useless_logits] = 0.0
        sorted_logits = sorted_logits / sorted_logits.sum(dim=-1, keepdim=True)

        logits = torch.zeros_like(logits).scatter_(-1, sorted_index, sorted_logits)

        # multinomial means sample 'num_samples' value by the probablity
        # not use argmax, because every token should have chance to generate
        next_token = torch.multinomial(logits, num_samples=1)

        seq = torch.cat([seq, next_token], dim=1)
        if next_token.item() == eos:
            break

    return seq
