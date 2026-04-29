import math
import torch

def cos_learning_rate_schedule_with_warmup(
    t: int,
    alpha_max: float,
    alpha_min: float,
    t_w: int,
    t_c: int       
) -> float:
    
    if t < t_w: 
        lr = alpha_max * t / t_w
    elif t_w <= t and t <= t_c:
        lr = alpha_min + 0.50 * (1 + math.cos(math.pi * (t - t_w) / (t_c-t_w))) * (alpha_max - alpha_min)
    else: 
        lr = alpha_min

    return lr 
