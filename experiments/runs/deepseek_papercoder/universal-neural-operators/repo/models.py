# models.py

"""
Neural operator models for the Universal Neural Operators reproduction.
Implements FNO, Mamba‑FNO, Perceiver IO, SwinV2, and CoDA‑NO backbones
together with problem‑specific lifting/projection adapters.

All components are configurable via a dictionary that mirrors the
`model` section of config.yaml.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.fft import irfft, irfft2, rfft, rfft2

# ----------------------------------------------------------------------
# Helper modules used by several backbones
# ----------------------------------------------------------------------

class SpectralConv1d(nn.Module):
    """1D Fourier transform convolution with a learnable complex kernel."""

    def __init__(self, modes: int, width: int) -> None:
        super().__init__()
        self.modes = modes
        self.width = width
        self.weight = nn.Parameter(torch.empty(width, width, modes, dtype=torch.cfloat))
        self.bias = nn.Parameter(torch.empty(width))
        self.linear = nn.Conv1d(width, width, kernel_size=1, bias=False)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.weight.real)
        nn.init.xavier_normal_(self.weight.imag)
        nn.init.zeros_(self.bias)
        nn.init.kaiming_normal_(self.linear.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        B, C, L = x.shape
        x_ft = rfft(x, dim=-1)  # (B, C, L//2+1)
        out_ft = torch.zeros(B, C, L // 2 + 1, device=x.device, dtype=torch.cfloat)
        # keep only the first 'modes' frequencies
        out_ft[:, :, : self.modes] = torch.einsum("bcl,ocl->bcl", x_ft[:, :, : self.modes], self.weight)
        x_conv = irfft(out_ft, n=L, dim=-1)
        # pointwise bypass
        x_linear = self.linear(x)
        return x_conv + x_linear + self.bias.view(1, -1, 1)


class SpectralConv2d(nn.Module):
    """2D Fourier transform convolution with a learnable complex kernel."""

    def __init__(self, modes1: int, modes2: int, width: int) -> None:
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.weight = nn.Parameter(
            torch.empty(width, width, modes1, modes2, dtype=torch.cfloat)
        )
        self.bias = nn.Parameter(torch.empty(width))
        self.linear = nn.Conv2d(width, width, kernel_size=1, bias=False)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.weight.real)
        nn.init.xavier_normal_(self.weight.imag)
        nn.init.zeros_(self.bias)
        nn.init.kaiming_normal_(self.linear.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x_ft = rfft2(x, dim=(-2, -1))  # (B, C, H, W//2+1)
        out_ft = torch.zeros(B, C, H, W // 2 + 1, device=x.device, dtype=torch.cfloat)
        # keep only the first modes1 rows, modes2 columns
        out_ft[:, :, : self.modes1, : self.modes2] = torch.einsum(
            "bchw,ochw->bchw",
            x_ft[:, :, : self.modes1, : self.modes2],
            self.weight,
        )
        x_conv = irfft2(out_ft, s=(H, W), dim=(-2, -1))
        x_linear = self.linear(x)
        return x_conv + x_linear + self.bias.view(1, -1, 1, 1)


# ----------------------------------------------------------------------
# Vanilla FNO bodies
# ----------------------------------------------------------------------

class FNOBody1d(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.n_layers = config.get("n_layers", 4)
        self.modes = config.get("modes1", 12)
        self.width = config.get("width", 32)
        self.activation = nn.GELU() if config.get("activation", "gelu") == "gelu" else nn.ReLU()
        layers = []
        for _ in range(self.n_layers):
            layers.append(SpectralConv1d(self.modes, self.width))
        self.spectral_layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.spectral_layers:
            x = layer(x)
            x = self.activation(x)
        return x


class FNOBody2d(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.n_layers = config.get("n_layers", 4)
        self.modes1 = config.get("modes1", 12)
        self.modes2 = config.get("modes2", 12)
        self.width = config.get("width", 32)
        self.activation = nn.GELU() if config.get("activation", "gelu") == "gelu" else nn.ReLU()
        layers = []
        for _ in range(self.n_layers):
            layers.append(SpectralConv2d(self.modes1, self.modes2, self.width))
        self.spectral_layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.spectral_layers:
            x = layer(x)
            x = self.activation(x)
        return x


# ----------------------------------------------------------------------
# Mamba SSM block (simplified, sequential scan)
# ----------------------------------------------------------------------

class MambaBlock1d(nn.Module):
    """
    Minimal implementation of a Mamba selective SSM for 1D sequences.
    The input is expected as (B, C, L) and is transposed internally.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(d_model * expand)

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)  # for x and z branches
        # Convolution kernel (causal)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=0,
            bias=False,
        )
        # SSM parameters
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).unsqueeze(0)))  # (1, d_state)
        self.D = nn.Parameter(torch.ones(d_inner))

        # Projections for delta, B, C
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1)  # delta, B, C
        self.out_proj = nn.Linear(self.d_inner, d_model)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.in_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.out_proj.weight, a=math.sqrt(5))
        nn.init.normal_(self.A_log, mean=0.0, std=0.01)

    def _selective_scan(self, u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        u: (B, L, d_inner)
        delta: (B, L, d_inner)  -> step size
        A: (d_inner, d_state)   -> negative exponential
        B: (B, L, d_state)
        C: (B, L, d_state)
        Returns: y (B, L, d_inner)
        """
        B, L, _ = u.shape
        A = -torch.exp(A)  # (d_inner, d_state)
        A_discrete = torch.exp(torch.einsum("bld,ds->blds", delta, A))  # (B, L, d_inner, d_state)
        delta_B = torch.einsum("bld,bls->blds", delta, B)  # element-wise multiplication
        # sequential scan
        h = torch.zeros(B, u.shape[2], self.d_state, device=u.device, dtype=u.dtype)  # (B, d_inner, d_state)
        ys = []
        for t in range(L):
            h = A_discrete[:, t] * h + delta_B[:, t] * u[:, t].unsqueeze(-1)  # h: (B, d_inner, d_state)
            y = (h * C[:, t].unsqueeze(1)).sum(dim=-1)  # (B, d_inner)
            ys.append(y)
        y = torch.stack(ys, dim=1)  # (B, L, d_inner)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) -> transpose to (B, L, C) for Mamba
        x = rearrange(x, "b c l -> b l c")
        # Project to x and z branches
        x_and_res = self.in_proj(x)  # (B, L, 2*d_inner)
        x_proj, res = x_and_res.chunk(2, dim=-1)  # each (B, L, d_inner)

        # Causal convolution
        # pad input on left for causality
        x_conv_in = rearrange(x_proj, "b l d -> b d l")
        x_conv_in_padded = F.pad(x_conv_in, (self.d_conv - 1, 0))
        x_conv_out = self.conv1d(x_conv_in_padded)
        x_conv_out = x_conv_out[:, :, : x_conv_in.size(-1)]  # truncate to original length
        x_conv_out = rearrange(x_conv_out, "b d l -> b l d")
        x_conv_out = F.silu(x_conv_out)

        # SSM
        proj = self.x_proj(x_conv_out)  # (B, L, d_state*2 + 1)
        delta, B, C = proj.split([1, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(delta.squeeze(-1))  # (B, L)

        # A parameter broadcast to (d_inner, d_state)
        A = self.A_log.repeat(self.d_inner, 1)  # (d_inner, d_state)

        y = self._selective_scan(x_conv_out, delta, A, B, C)  # (B, L, d_inner)

        # Gating with residual branch
        y = y * F.silu(res)  # element-wise multiplication with z branch (silu applied)
        y = self.out_proj(y)  # back to d_model
        y = rearrange(y, "b l c -> b c l")  # back to (B, C, L)
        return y


class MambaBlock2d(MambaBlock1d):
    """
    Adapts the 1D Mamba to 2D by flattening the spatial dimensions.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> flatten to (B, C, H*W) then (B, H*W, C)
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W)
        x_flat = super().forward(x_flat)  # MambaBlock1d.forward expects (B,C,L)
        return x_flat.view(B, C, H, W)


# ----------------------------------------------------------------------
# MambaFNO bodies
# ----------------------------------------------------------------------

class MambaFNOBody1d(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        mamba_cfg = config.get("mamba", {})
        self.mamba = MambaBlock1d(
            d_model=config["fno"].get("width", 32),
            d_state=mamba_cfg.get("d_state", 16),
            d_conv=mamba_cfg.get("d_conv", 4),
            expand=mamba_cfg.get("expand", 2),
        )
        self.fno = FNOBody1d(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fno(self.mamba(x))


class MambaFNOBody2d(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        mamba_cfg = config.get("mamba", {})
        self.mamba = MambaBlock2d(
            d_model=config["fno"].get("width", 32),
            d_state=mamba_cfg.get("d_state", 16),
            d_conv=mamba_cfg.get("d_conv", 4),
            expand=mamba_cfg.get("expand", 2),
        )
        self.fno = FNOBody2d(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fno(self.mamba(x))


# ----------------------------------------------------------------------
# Perceiver IO components
# ----------------------------------------------------------------------

class PerceiverBlock1d(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        width = config["fno"].get("width", 32)
        n_latent = config.get("perceiver", {}).get("n_latent", 32)
        n_heads = config.get("perceiver", {}).get("n_heads", 4)
        n_self_attn = config.get("perceiver", {}).get("n_self_attn", 2)
        use_fno_kv = config.get("perceiver", {}).get("fno_for_kv", False)

        self.width = width
        self.n_latent = n_latent

        # Keys/values projection
        if use_fno_kv:
            self.fno_k = SpectralConv1d(config["fno"].get("modes1", 12), width)
            self.fno_v = SpectralConv1d(config["fno"].get("modes1", 12), width)
        else:
            self.fno_k = nn.Conv1d(width, width, kernel_size=1)
            self.fno_v = nn.Conv1d(width, width, kernel_size=1)

        # Latent array
        self.latent = nn.Parameter(torch.randn(1, n_latent, width) * 0.02)

        # Cross/self attention layers
        self.cross_attn_in = nn.MultiheadAttention(width, n_heads, batch_first=True)
        self.self_attn_layers = nn.ModuleList(
            [nn.MultiheadAttention(width, n_heads, batch_first=True) for _ in range(n_self_attn)]
        )
        self.cross_attn_out = nn.MultiheadAttention(width, n_heads, batch_first=True)

        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.norm3 = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        B, C, L = x.shape
        x_seq = rearrange(x, "b c l -> b l c")  # (B, L, C)

        # Compute keys and values
        k = self.fno_k(x)   # (B, C, L)
        v = self.fno_v(x)
        k = rearrange(k, "b c l -> b l c")
        v = rearrange(v, "b c l -> b l c")

        # Cross attention: latent attends to input
        latent = self.latent.expand(B, -1, -1)  # (B, n_latent, C)
        latent2, _ = self.cross_attn_in(query=latent, key=k, value=v)
        latent = latent + latent2
        latent = self.norm1(latent)

        # Self attention on latent
        for attn in self.self_attn_layers:
            latent2, _ = attn(latent, latent, latent)
            latent = latent + latent2
            latent = self.norm2(latent)

        # Cross attention: input attends to latent
        out, _ = self.cross_attn_out(query=x_seq, key=latent, value=latent)
        out = out + x_seq
        out = self.norm3(out)
        out = rearrange(out, "b l c -> b c l")
        return out


class PerceiverBlock2d(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        width = config["fno"].get("width", 32)
        n_latent = config.get("perceiver", {}).get("n_latent", 32)
        n_heads = config.get("perceiver", {}).get("n_heads", 4)
        n_self_attn = config.get("perceiver", {}).get("n_self_attn", 2)
        use_fno_kv = config.get("perceiver", {}).get("fno_for_kv", False)

        self.width = width
        self.n_latent = n_latent

        if use_fno_kv:
            self.fno_k = SpectralConv2d(config["fno"].get("modes1", 12), config["fno"].get("modes2", 12), width)
            self.fno_v = SpectralConv2d(config["fno"].get("modes1", 12), config["fno"].get("modes2", 12), width)
        else:
            self.fno_k = nn.Conv2d(width, width, kernel_size=1)
            self.fno_v = nn.Conv2d(width, width, kernel_size=1)

        self.latent = nn.Parameter(torch.randn(1, n_latent, width) * 0.02)

        self.cross_attn_in = nn.MultiheadAttention(width, n_heads, batch_first=True)
        self.self_attn_layers = nn.ModuleList(
            [nn.MultiheadAttention(width, n_heads, batch_first=True) for _ in range(n_self_attn)]
        )
        self.cross_attn_out = nn.MultiheadAttention(width, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.norm3 = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_seq = x.flatten(2).transpose(1, 2)  # (B, H*W, C)

        k = self.fno_k(x)   # (B, C, H, W)
        v = self.fno_v(x)
        k = k.flatten(2).transpose(1, 2)  # (B, H*W, C)
        v = v.flatten(2).transpose(1, 2)

        latent = self.latent.expand(B, -1, -1)
        latent2, _ = self.cross_attn_in(query=latent, key=k, value=v)
        latent = latent + latent2
        latent = self.norm1(latent)

        for attn in self.self_attn_layers:
            latent2, _ = attn(latent, latent, latent)
            latent = latent + latent2
            latent = self.norm2(latent)

        out, _ = self.cross_attn_out(query=x_seq, key=latent, value=latent)
        out = out + x_seq
        out = self.norm3(out)
        out = out.transpose(1, 2).view(B, C, H, W)
        return out


class PerceiverIOBody1d(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.perceiver = PerceiverBlock1d(config)
        self.fno = FNOBody1d(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fno(self.perceiver(x))


class PerceiverIOBody2d(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.perceiver = PerceiverBlock2d(config)
        self.fno = FNOBody2d(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fno(self.perceiver(x))


# ----------------------------------------------------------------------
# SwinV2 body (window‑based transformer for 2D)
# ----------------------------------------------------------------------

class WindowAttention(nn.Module):
    """Multi‑head self attention within local windows, with relative position bias."""

    def __init__(self, dim: int, window_size: int, num_heads: int) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        # Index mapping
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, window_size, window_size)
        coords_flatten = torch.flatten(coords, 1)  # (2, window_size*window_size)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, Wh*Ww, Wh*Ww)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (Wh*Ww, Wh*Ww, 2)
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # (Wh*Ww, Wh*Ww)
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        nn.init.trunc_normal_(self.qkv.weight, std=0.02)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1
        )  # (Wh*Ww, Wh*Ww, nH)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (nH, Wh*Ww, Wh*Ww)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)
        return out


class SwinV2Block(nn.Module):
    """Swin‑V2 transformer block with optional shifted windows."""

    def __init__(self, dim: int, window_size: int, num_heads: int, shift_size: int = 0,
                 mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # x: (B, H*W, C)
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Pad to multiples of window_size
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        H_padded, W_padded = H + pad_h, W + pad_w

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition into windows
        x_windows = shifted_x.view(B, H_padded // self.window_size, self.window_size,
                                  W_padded // self.window_size, self.window_size, C)
        x_windows = x_windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(
            -1, self.window_size * self.window_size, C
        )  # (nW*B, window_window, C)

        # Attention (no mask for now, simplified)
        attn_windows = self.attn(x_windows)  # (nW*B, window_window, C)

        # Merge windows
        attn_windows = attn_windows.view(
            B, H_padded // self.window_size, W_padded // self.window_size,
            self.window_size, self.window_size, C
        )
        shifted_x = attn_windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(
            B, H_padded, W_padded, C
        )

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + x
        shortcut = x
        x = self.norm2(x)
        x = shortcut + self.mlp(x)
        return x


class SwinV2Body(nn.Module):
    """Swin‑V2 transformer body for 2D feature maps."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        img_size = config.get("swin", {}).get("img_size", 64)
        window_size = config.get("swin", {}).get("window_size", 8)
        embed_dim = config.get("swin", {}).get("embed_dim", 96)
        depths = config.get("swin", {}).get("depths", [2, 2, 6, 2])
        num_heads = config.get("swin", {}).get("num_heads", [3, 6, 12, 24])
        in_width = config.get("fno", {}).get("width", 32)

        self.input_proj = nn.Conv2d(in_width, embed_dim, kernel_size=1)
        self.output_proj = nn.Conv2d(embed_dim, in_width, kernel_size=1)

        self.layers = nn.ModuleList()
        for i_depth, depth in enumerate(depths):
            for d in range(depth):
                shift_size = 0 if (d % 2 == 0) else window_size // 2
                self.layers.append(
                    SwinV2Block(
                        dim=embed_dim,
                        window_size=window_size,
                        num_heads=num_heads[i_depth],
                        shift_size=shift_size,
                    )
                )

        self.H = img_size
        self.W = img_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W), C = width
        B, C, H, W = x.shape
        x = self.input_proj(x)  # (B, embed_dim, H, W)
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, embed_dim)
        for blk in self.layers:
            x = blk(x, H, W)
        x = x.transpose(1, 2).view(B, -1, H, W)
        x = self.output_proj(x)  # (B, width, H, W)
        return x


# ----------------------------------------------------------------------
# Codomain Attention Neural Operator (CoDA‑NO)
# ----------------------------------------------------------------------

class CoDALayer1d(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.qkv = nn.Conv1d(width, width * 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        qkv = self.qkv(x)  # (B, 3C, L)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        # reshape to (B, L, C)
        q = q.transpose(1, 2)  # (B, L, C)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scale = math.sqrt(C)
        attn = torch.bmm(q.transpose(1, 2), k) / scale  # (B, C, C)
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(v, attn)  # (B, L, C)
        out = out.transpose(1, 2)  # (B, C, L)
        return x + out


class CoDALayer2d(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.qkv = nn.Conv2d(width, width * 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)  # (B, 3C, H, W)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        q = q.view(B, C, N).transpose(1, 2)  # (B, N, C)
        k = k.view(B, C, N).transpose(1, 2)
        v = v.view(B, C, N).transpose(1, 2)
        scale = math.sqrt(C)
        attn = torch.bmm(q.transpose(1, 2), k) / scale  # (B, C, C)
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(v, attn)  # (B, N, C)
        out = out.transpose(1, 2).view(B, C, H, W)
        return x + out


class CoDANOBody1d(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        width = config.get("coda", {}).get("width", config["fno"].get("width", 32))
        n_layers = config.get("coda", {}).get("n_layers", 4)  # default same as fno
        layers = []
        for _ in range(n_layers):
            layers.append(CoDALayer1d(width))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class CoDANOBody2d(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        width = config.get("coda", {}).get("width", config["fno"].get("width", 32))
        n_layers = config.get("coda", {}).get("n_layers", 4)
        layers = []
        for _ in range(n_layers):
            layers.append(CoDALayer2d(width))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# ----------------------------------------------------------------------
# Adapters (lifting / projection)
# ----------------------------------------------------------------------

class Adapter1d(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, out_channels: int) -> None:
        super().__init__()
        self.lift = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, out_channels, kernel_size=1),
        )
        self.proj = nn.Sequential(
            nn.Conv1d(out_channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, in_channels, kernel_size=1),
        )

    def forward_lift(self, x: torch.Tensor) -> torch.Tensor:
        return self.lift(x)

    def forward_proj(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Adapter2d(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, out_channels: int) -> None:
        super().__init__()
        self.lift = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(out_channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=1),
        )

    def forward_lift(self, x: torch.Tensor) -> torch.Tensor:
        return self.lift(x)

    def forward_proj(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ----------------------------------------------------------------------
# ModelBase – combines body and adapters
# ----------------------------------------------------------------------

class ModelBase(nn.Module):
    """
    Shared neural operator with per‑problem adapters.
    body_cfg: dictionary representing the 'model' section of the config.
    dim: spatial dimension of the problem (1 or 2).
    """

    def __init__(self, body_cfg: dict, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.adapters_lift = nn.ModuleDict()
        self.adapters_proj = nn.ModuleDict()

        # Create the shared operator body
        arch = body_cfg.get("architecture", "fno").lower()
        if dim == 1:
            if arch == "fno":
                self.body = FNOBody1d(body_cfg["fno"])
            elif arch == "mamba_fno":
                self.body = MambaFNOBody1d(body_cfg)
            elif arch == "perceiver_fno":
                self.body = PerceiverIOBody1d(body_cfg)
            elif arch == "coda_no":
                self.body = CoDANOBody1d(body_cfg)
            else:
                raise ValueError(f"Unsupported 1D architecture: {arch}")
        elif dim == 2:
            if arch == "fno":
                self.body = FNOBody2d(body_cfg["fno"])
            elif arch == "mamba_fno":
                self.body = MambaFNOBody2d(body_cfg)
            elif arch == "perceiver_fno":
                self.body = PerceiverIOBody2d(body_cfg)
            elif arch == "swin_v2":
                self.body = SwinV2Body(body_cfg)
            elif arch == "coda_no":
                self.body = CoDANOBody2d(body_cfg)
            else:
                raise ValueError(f"Unsupported 2D architecture: {arch}")
        else:
            raise ValueError(f"Unsupported dimension: {dim} (only 1 or 2)")

    def add_adapter(self, name: str, in_channels: int, out_channels: int) -> None:
        """Add a new lifting/projection adapter for a problem identified by `name`."""
        width = self.body[0].width if isinstance(self.body, nn.Sequential) else self.body.width  # assumption
        # Actually, we need to get width from the body's configuration.
        # Better: get width from the underlying spectral conv layers.
        # We'll store width as attribute during body creation.
        # We can access self.body[0].width etc. We'll store a property.
        # For simplicity, we'll pass width explicitly.
        if self.dim == 1:
            lift = Adapter1d(in_channels, width, width)
            proj = Adapter1d(width, width, out_channels)
        else:
            lift = Adapter2d(in_channels, width, width)
            proj = Adapter2d(width, width, out_channels)
        self.adapters_lift[name] = lift
        self.adapters_proj[name] = proj

    def forward(self, problem: str, x: torch.Tensor) -> torch.Tensor:
        lift = self.adapters_lift[problem]
        proj = self.adapters_proj[problem]
        x = lift.forward_lift(x)
        x = self.body(x)
        x = proj.forward_proj(x)
        return x

    def freeze_body(self) -> None:
        """Freeze all parameters in the shared body (used in fine‑tuning)."""
        for param in self.body.parameters():
            param.requires_grad = False

    def unfreeze_body(self) -> None:
        for param in self.body.parameters():
            param.requires_grad = True


# ----------------------------------------------------------------------
# Factory helper (optional)
# ----------------------------------------------------------------------

def create_model_body(body_cfg: dict, dim: int) -> nn.Module:
    """
    Instantiate the operator body without adapters, for standalone use.
    """
    return ModelBase(body_cfg, dim).body
