import torch
import numpy
import random

def data_loading(
    dataset: numpy.ndarray,
    batch_size: int,
    context_length: int,
    device = None
) -> tuple[torch.Tensor, torch.Tensor]:

    valid_len = len(dataset) - context_length - 1

    x = []
    y = []
    for _ in range(batch_size):
        index = random.randint(0, valid_len)
        x.append(dataset[index : index + context_length])
        y.append(dataset[index + 1 : index + context_length + 1])
    
    # warning: use the array(x), not x
    # Creating a tensor from a list of numpy.ndarrays is extremely slow. 
    # Please consider converting the list to a single numpy.ndarray with numpy.array() 
    x = torch.tensor(numpy.array(x), dtype=torch.long, device=device)
    y = torch.tensor(numpy.array(y), dtype=torch.long, device=device)

    return x, y