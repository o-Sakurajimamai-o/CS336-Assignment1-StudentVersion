import math
import torch
from typing import Iterable, Tuple

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameters: {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                
                m, v = state['m'], state['v']
                state['step'] += 1
                t = state['step']

                # here I prepare two ways to compute adamw, which i use is more suitable and reasonable
                if wd > 0.0:
                    p.data.add_(p.data, alpha=-lr * wd)
                    # p.data = p.data - lr * wd * p.data

                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                # m = beta1 * m + (1 - beta1) * grad
                # state['m'] = m
                # v = beta2 * v + (1 - beta2) * (grad ** 2)
                # state['v'] = v

                alpha_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

                p.data.addcdiv_(m, v.sqrt().add_(eps), value=-alpha_t)
                # p.data = p.data - alpha_t * (m / (torch.sqrt(v) + eps)) 

        return loss