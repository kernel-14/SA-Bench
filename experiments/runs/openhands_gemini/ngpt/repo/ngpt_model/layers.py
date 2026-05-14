
import torch
import torch.nn as nn
from typing import Optional

def Norm(x: torch.Tensor) -> torch.Tensor:
    """
    Normalizes a vector to have unit norm.
    Equivalent to L2 normalization.
    """
    return x / x.norm(p=2, dim=-1, keepdim=True)

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Normalizes input to a norm of sqrt(d_model) then scales by a learnable vector.
    Used in baseline Transformer.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) as described in Su et al. (2024).
    Applies rotary transformations to input tensors for positional encoding.
    """
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def _update_cos_sin_tables(self, x: torch.Tensor, seq_len: int):
        if seq_len > self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, None, :, :].type_as(x)
            self.sin_cached = emb.sin()[None, None, :, :].type_as(x)
        return self.cos_cached[:seq_len, ...].to(x.device), self.sin_cached[:seq_len, ...].to(x.device)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        if seq_len is None:
            seq_len = x.shape[2] # Assume x has shape (batch, heads, seq_len, dim)

        cos, sin = self._update_cos_sin_tables(x, seq_len)
        x = (x * cos) + (self.rotate_half(x) * sin)
        return x

