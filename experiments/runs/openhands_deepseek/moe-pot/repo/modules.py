import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from layers import CNNExpert, RouterGating, top_k_routing, load_balance_loss


class MultiHeadFourierLayer(nn.Module):
    """Multi-head Fourier kernel integral layer.

    Splits the input feature map into h groups along the channel dimension,
    processes each group independently through the Fourier domain with a
    small MLP, and concatenates the results.

    Input: (B, d, H, W)
    Output: (B, d, H, W)
    """

    def __init__(self, dim: int, num_heads: int, modes: int = 16):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.modes = modes

        # Learnable frequency-domain transformations per head
        # W1, W2: (h, head_dim, head_dim) complex
        self.W1 = nn.Parameter(torch.randn(num_heads, self.head_dim, self.head_dim,
                                             dtype=torch.cfloat) * 0.02)
        self.W2 = nn.Parameter(torch.randn(num_heads, self.head_dim, self.head_dim,
                                             dtype=torch.cfloat) * 0.02)
        self.b1 = nn.Parameter(torch.zeros(num_heads, self.head_dim, dtype=torch.cfloat))
        self.b2 = nn.Parameter(torch.zeros(num_heads, self.head_dim, dtype=torch.cfloat))

    def _fourier_transform(self, x: torch.Tensor) -> torch.Tensor:
        """2D real FFT, keeping only low-frequency modes."""
        x_ft = torch.fft.rfft2(x, norm='ortho')
        # Truncate to modes x modes
        H_ft, W_ft = x_ft.shape[-2], x_ft.shape[-1]
        modes_h = min(self.modes, H_ft)
        modes_w = min(self.modes, W_ft)
        x_ft = x_ft[..., :modes_h, :modes_w]
        return x_ft

    def _inverse_fourier_transform(self, x_ft: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Pad back and apply inverse real FFT."""
        H_ft, W_ft = x_ft.shape[-2], x_ft.shape[-1]
        # Pad to expected frequency dimensions
        expected_H = H // 2 + 1
        pad_h = expected_H - H_ft
        pad_w = W - W_ft
        if pad_h > 0 or pad_w > 0:
            x_ft = F.pad(x_ft, (0, pad_w, 0, pad_h))
        return torch.fft.irfft2(x_ft, s=(H, W), norm='ortho')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, d, H, W = x.shape
        # Split into heads: (B, h, head_dim, H, W)
        x_heads = x.reshape(B, self.num_heads, self.head_dim, H, W)

        # Fourier transform per head
        x_ft_list = []
        for i in range(self.num_heads):
            x_i = x_heads[:, i]  # (B, head_dim, H, W)
            x_ft = self._fourier_transform(x_i)  # (B, head_dim, modes_h, modes_w)
            x_ft_list.append(x_ft)
        x_ft = torch.stack(x_ft_list, dim=1)  # (B, h, head_dim, modes_h, modes_w)

        # Spectral MLP: W2 · σ(W1 · z + b1) + b2
        # Permute to (B, h, modes_h, modes_w, head_dim) for matmul
        x_ft_t = x_ft.permute(0, 1, 3, 4, 2)  # (B, h, modes_h, modes_w, head_dim)
        # Apply W1
        x_ft_t = torch.matmul(x_ft_t, self.W1) + self.b1  # (B, h, modes_h, modes_w, head_dim)
        x_ft_t = F.gelu(x_ft_t.real.to(x.dtype).to(x.device)).to(torch.cfloat)
        # Apply W2
        x_ft_t = torch.matmul(x_ft_t, self.W2) + self.b2
        # Permute back
        x_ft = x_ft_t.permute(0, 1, 4, 2, 3)  # (B, h, head_dim, modes_h, modes_w)

        # Inverse Fourier per head
        out_heads = []
        for i in range(self.num_heads):
            x_i_ft = x_ft[:, i]  # (B, head_dim, modes_h, modes_w)
            x_i_out = self._inverse_fourier_transform(x_i_ft, H, W)
            out_heads.append(x_i_out)
        out = torch.cat(out_heads, dim=1)  # (B, d, H, W)
        return out


class MoELayer(nn.Module):
    """Mixture-of-Experts layer with shared and routed experts.

    Shared experts: always activated for every input.
    Routed experts: top-K selected dynamically via router-gating network.

    Input: (B, d, H, W)
    Output: (B, d, H, W), load_balance_loss
    """

    def __init__(self, dim: int, num_routed: int = 16, num_shared: int = 2,
                 top_k: int = 4, kernel_size: int = 3):
        super().__init__()
        self.num_routed = num_routed
        self.num_shared = num_shared
        self.top_k = top_k

        # Shared experts
        self.shared_experts = nn.ModuleList([
            CNNExpert(dim, kernel_size) for _ in range(num_shared)
        ])

        # Routed experts
        self.routed_experts = nn.ModuleList([
            CNNExpert(dim, kernel_size) for _ in range(num_routed)
        ])

        # Router-gating network
        self.router = RouterGating(dim, num_routed, kernel_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, d, H, W = x.shape

        # Shared experts
        shared_out = torch.zeros_like(x)
        for expert in self.shared_experts:
            shared_out = shared_out + expert(x)
        shared_out = shared_out / self.num_shared

        # Router
        logits = self.router(x)  # (B, num_routed)
        weights, indices = top_k_routing(logits, self.top_k)  # weights: (B, num_routed), indices: (B, K)

        # Routed experts
        routed_out = torch.zeros_like(x)
        for b in range(B):
            for k in range(self.top_k):
                expert_idx = indices[b, k].item()
                w = weights[b, expert_idx]
                expert_out = self.routed_experts[expert_idx](x[b:b+1])
                routed_out[b] = routed_out[b] + w * expert_out.squeeze(0)

        # Load balance loss
        lb_loss = load_balance_loss(weights)

        return shared_out + routed_out, lb_loss


class Block(nn.Module):
    """One MoE-POT block: Multi-Head Fourier Layer + MoE Layer.

    Input: (B, d, H, W)
    Output: (B, d, H, W), load_balance_loss
    """

    def __init__(self, dim: int, num_heads: int, num_routed: int = 16,
                 num_shared: int = 2, top_k: int = 4, fourier_modes: int = 16,
                 kernel_size: int = 3):
        super().__init__()
        self.fourier = MultiHeadFourierLayer(dim, num_heads, fourier_modes)
        self.moe = MoELayer(dim, num_routed, num_shared, top_k, kernel_size)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, d, H, W)
        # Fourier with residual
        identity = x
        z0 = x.permute(0, 2, 3, 1)  # (B, H, W, d)
        z0 = self.norm1(z0)
        z0 = z0.permute(0, 3, 1, 2)  # (B, d, H, W)
        z0 = self.fourier(z0)
        z0 = z0 + identity

        # MoE with residual
        identity = z0
        z0_norm = z0.permute(0, 2, 3, 1)
        z0_norm = self.norm2(z0_norm)
        z0_norm = z0_norm.permute(0, 3, 1, 2)
        z1, lb_loss = self.moe(z0_norm)
        z1 = z1 + identity

        return z1, lb_loss
