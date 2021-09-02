import torch
import numpy as np


def num_parameters(module: torch.nn.Module):
    return sum([
        np.prod(x.shape) for x in module.parameters()
    ])


def to_cuda(elements):
    """
    Transfers every object in elements to GPU VRAM if available.
    elements can be a object or list/tuple of objects
    """
    if torch.cuda.is_available():
        if type(elements) == tuple or type(elements) == list:
            return [x.cuda() for x in elements]
        return elements.cuda()
    return elements


def format_params(module: torch.nn.Module):
    n = num_parameters(module)
    if n > 10**6:
        n /= 10**6
        return f"{n:.2f}M"
    if n > 10**3:
        n /= 10**3
        return f"{n:.1f}K"
    return n