import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_sinusoidal_embedding(timesteps: torch.Tensor, embedding_dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal timestep embedding used in diffusion models."""
    half = embedding_dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if embedding_dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def get_scale_sinusoidal_embedding(scale_idx: int, embedding_dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal embedding for scale index (e.g., 0 for phase 1, 1 for phase 2)."""
    half = embedding_dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half)
    t = torch.tensor([scale_idx], dtype=torch.float32)
    args = t.unsqueeze(1) * freqs.unsqueeze(0)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if embedding_dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding.squeeze(0)


class TimestepEmbedding(nn.Module):
    """MLP that maps sinusoidal timestep embedding to the desired dimension."""
    def __init__(self, embedding_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = get_sinusoidal_embedding(t, embedding_dim=256, max_period=10000)
        return self.mlp(emb.to(t.device))


class ScaleEmbedding(nn.Module):
    """MLP that maps scale sinusoidal embedding to a scale vector for AdaLN-Zero."""
    def __init__(self, embedding_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, scale_idx: int, device: torch.device) -> torch.Tensor:
        emb = get_scale_sinusoidal_embedding(scale_idx, self.embedding_dim).to(device)
        return self.mlp(emb)


class AdaLN(nn.Module):
    """Adaptive Layer Normalization: modulates scale and shift of LN based on condition.
    Returns LN(x) * scale + shift (scale is applied directly, not as 1+scale)."""
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(dim, elementwise_affine=False)
        self.linear = nn.Linear(cond_dim, dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(cond).chunk(2, dim=-1)
        x = self.ln(x)
        while scale.dim() < x.dim():
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        return x * scale + shift


class AdaLNZero(nn.Module):
    """AdaLN-Zero: initializes modulation parameters to zero for identity-like behavior."""
    def __init__(self, dim: int, cond_dim: int, num_modulations: int = 6):
        super().__init__()
        self.num_modulations = num_modulations
        self.ln1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.linear = nn.Linear(cond_dim, dim * num_modulations)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        params = self.linear(cond)  # [B, dim * num_modulations]
        alpha1, beta1, gamma1, alpha2, beta2, gamma2 = params.chunk(self.num_modulations, dim=-1)
        while alpha1.dim() < x.dim():
            alpha1 = alpha1.unsqueeze(1)
            beta1 = beta1.unsqueeze(1)
            gamma1 = gamma1.unsqueeze(1)
            alpha2 = alpha2.unsqueeze(1)
            beta2 = beta2.unsqueeze(1)
            gamma2 = gamma2.unsqueeze(1)
        x_ln1 = self.ln1(x)
        modulated_x = alpha1 * x_ln1 + beta1
        x_attn = x + gamma1 * modulated_x
        x_ln2 = self.ln2(x_attn)
        modulated_x2 = alpha2 * x_ln2 + beta2
        x_out = x_attn + gamma2 * modulated_x2
        return x_out


class MLPBlock(nn.Module):
    """Simple MLP block with two linear layers and SiLU activation."""
    def __init__(self, dim: int, expansion_ratio: float = 4.0):
        super().__init__()
        hidden_dim = int(dim * expansion_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention with optional causal mask."""
    def __init__(self, dim: int, num_heads: int = 12, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout.p if self.training else 0.0)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)
