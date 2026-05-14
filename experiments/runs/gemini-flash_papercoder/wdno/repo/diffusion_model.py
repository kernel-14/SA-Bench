```python
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Union, Tuple, List, Callable
import math

# Imports from project
from config import Config

# Helper functions for handling kernel/padding/stride (can be int or tuple)
def _handle_kernel_args(dim: int, kernel: Union[int, Tuple[int, ...]], padding: Union[int, Tuple[int, ...]], stride: Union[int, Tuple[int, ...]]):
    """Ensures kernel, padding, stride are tuples of length dim."""
    if isinstance(kernel, int):
        kernel = (kernel,) * dim
    if isinstance(padding, int):
        padding = (padding,) * dim
    if isinstance(stride, int):
        stride = (stride,) * dim
    
    if len(kernel) != dim or len(padding) != dim or len(stride) != dim:
        raise ValueError(f"Kernel, padding, stride must have {dim} elements or be a single int for {dim}D convolution.")
    
    return kernel, padding, stride

# Helper Layer Classes

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal position embeddings for time, following the original Transformer paper.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Residual(nn.Module):
    """
    Residual connection for modules.
    """
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.fn(x, *args, **kwargs) + x

class PreNorm(nn.Module):
    """
    Applies GroupNorm before passing through a function.
    """
    def __init__(self, dim: int, fn: nn.Module, num_groups: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.fn(self.norm