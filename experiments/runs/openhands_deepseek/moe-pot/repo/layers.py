import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class Patchification(nn.Module):
    """Patchify spatiotemporal input with learned positional encoding.

    Following Eq (4): Z_p^t = P(u^t + p^t)

    Positional encoding is added to the input at full resolution BEFORE
    convolution/patchification. This matches the ViT-style approach where
    positional encoding is on the input grid.

    Input: (B, C, H, W, T)
    Output: (B, d, H/P, W/P, T)
    """

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 8):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size,
                               stride=patch_size)
        # W_p maps (x, y, t) -> C, added to input channels
        self.pos_embed = nn.Linear(3, in_channels)

    def _grid_coords(self, H: int, W: int, t: int, device: torch.device):
        xs = torch.linspace(-1.0, 1.0, H, device=device)
        ys = torch.linspace(-1.0, 1.0, W, device=device)
        y_grid, x_grid = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack([x_grid, y_grid, torch.full_like(x_grid, float(t))], dim=-1)
        return coords  # (H, W, 3)

    def forward(self, u: torch.Tensor):
        B, C, H, W, T = u.shape

        out = []
        for t in range(T):
            frame = u[..., t]  # (B, C, H, W)
            # Positional encoding at full input resolution
            coords = self._grid_coords(H, W, t, u.device)
            pe = self.pos_embed(coords).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
            # Add before convolution: P(u^t + p^t)
            patches = self.proj(frame + pe)  # (B, d, H/P, W/P)
            out.append(patches)
        return torch.stack(out, dim=-1)  # (B, d, H/P, W/P, T)


class TemporalAggregation(nn.Module):
    """Aggregate temporal features with learnable Fourier-inspired weights.

    Input: (B, d, H, W, T)
    Output: (B, d, H, W)
    """

    def __init__(self, dim: int, num_timesteps: int):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.W_t = nn.Linear(dim, dim, bias=False)
        self.gamma = nn.Parameter(torch.randn(dim) * 0.02)

    def forward(self, z_p: torch.Tensor):
        B, d, H, W, T = z_p.shape
        z_agg = torch.zeros(B, d, H, W, device=z_p.device, dtype=z_p.dtype)
        for t in range(T):
            z_t = z_p[..., t]  # (B, d, H, W)
            # Re-arrange to apply linear per spatial location
            z_t_flat = z_t.permute(0, 2, 3, 1).reshape(-1, d)  # (B*H*W, d)
            transformed = self.W_t(z_t_flat)  # (B*H*W, d)
            transformed = transformed.reshape(B, H, W, d).permute(0, 3, 1, 2)  # (B, d, H, W)
            phase = torch.exp(-1j * self.gamma * t)  # (d,)
            z_agg = z_agg + transformed * phase.real.to(z_p.device)
        return z_agg


class CNNExpert(nn.Module):
    """Convolutional expert network for MoE layer.

    Takes a feature map and returns a feature map of the same shape.
    """

    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RouterGating(nn.Module):
    """Router-gating network for MoE layer.

    Takes the full feature map, globally pools, and produces routing logits.
    Output: (B, num_experts)
    """

    def __init__(self, dim: int, num_experts: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(dim // 4, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, dim, H, W)
        h = self.conv(x)  # (B, dim//4, 1, 1)
        h = h.squeeze(-1).squeeze(-1)  # (B, dim//4)
        logits = self.fc(h)  # (B, num_experts)
        return logits


def top_k_routing(logits: torch.Tensor, k: int):
    """Top-K sparse routing with softmax normalization.

    Returns:
        weights: (B, num_experts) normalized top-K weights
        indices: (B, K) indices of selected experts
    """
    num_experts = logits.size(-1)
    topk_vals, topk_idx = torch.topk(logits, k, dim=-1)  # (B, K)
    # Mask: set non-top-K logits to -inf
    mask = torch.zeros_like(logits).scatter_(-1, topk_idx, 1.0)
    masked_logits = logits.masked_fill(mask == 0, float('-inf'))
    weights = F.softmax(masked_logits, dim=-1)
    # Zero out non-selected weights (softmax of -inf should be 0, but ensure)
    weights = weights * mask
    return weights, topk_idx


def load_balance_loss(weights: torch.Tensor, balancing_weight: float = 0.1) -> torch.Tensor:
    """Compute load balancing loss as squared coefficient of variation.

    weights: (B, N_r) routing weights after softmax and top-K

    Following DeepSeekMoE / Switch Transformer:
    Importance_i = sum_b w_{i,b}
    CV = std(Importance) / mean(Importance)
    Loss = w_bal * CV^2
    """
    B, N_r = weights.shape
    importance = weights.sum(dim=0)  # (N_r,)
    mean_imp = importance.mean()
    if mean_imp < 1e-8:
        return torch.tensor(0.0, device=weights.device)
    std_imp = importance.std()
    cv = std_imp / mean_imp
    return balancing_weight * (cv ** 2)
