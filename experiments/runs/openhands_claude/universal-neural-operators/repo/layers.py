from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Fourier spectral convolution layers
# ---------------------------------------------------------------------------

class SpectralConv1d(nn.Module):
    """1-D Fourier integral operator layer (FNO-1d)."""

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def _compl_mul1d(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bix,iox->box", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, N = x.shape
        x_ft = torch.fft.rfft(x, norm="ortho")
        out_ft = torch.zeros(B, self.out_channels, N // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, : self.modes] = self._compl_mul1d(x_ft[:, :, : self.modes], self.weights)
        return torch.fft.irfft(out_ft, n=N, norm="ortho")


class SpectralConv2d(nn.Module):
    """2-D Fourier integral operator layer (FNO-2d)."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def _compl_mul2d(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, : self.modes1, : self.modes2] = self._compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self._compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )
        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")


# ---------------------------------------------------------------------------
# FNO integral-operator blocks  (equation 1 in the paper)
# ---------------------------------------------------------------------------

class FNOBlock1d(nn.Module):
    """Single FNO layer: σ(A·v + K(v) + b) for 1-D spatial domains."""

    def __init__(self, channels: int, modes: int) -> None:
        super().__init__()
        self.spectral = SpectralConv1d(channels, channels, modes)
        self.bypass = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.InstanceNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.spectral(x) + self.bypass(x)))


class FNOBlock2d(nn.Module):
    """Single FNO layer for 2-D spatial domains."""

    def __init__(self, channels: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, modes1, modes2)
        self.bypass = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.InstanceNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.spectral(x) + self.bypass(x)))


# ---------------------------------------------------------------------------
# Mamba-SSM block (post-lifting, equation 2 in the paper)
# Implements the causal convolution approximation of the SSM recurrence.
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    """
    Simplified Mamba-SSM module inserted after the lifting map.

    The paper describes:
        ṽ₀(x,t) = (M_φ v₀)(x,t) = Σ_{τ≤t} K_τ v₀(x, t-τ)

    We implement this as a depthwise causal 1-D convolution over the
    temporal (or sequence) dimension, with a learnable selective gate
    that mimics the input-dependent state-space selection mechanism.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)

        # SSM parameters A, D
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.norm = nn.LayerNorm(d_model)

    def _ssm(self, x: torch.Tensor) -> torch.Tensor:
        """Selective state-space scan (simplified discretised version)."""
        B, L, d = x.shape
        d_state = self.A_log.shape[1]

        xz = self.x_proj(x)
        delta, B_ssm, C_ssm = xz.split([d, d_state, d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # Discretise: Ā = exp(Δ·A), B̄ = Δ·B
        delta_A = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, d, d_state)
        delta_B = delta.unsqueeze(-1) * B_ssm.unsqueeze(2)  # (B, L, d, d_state)

        # Sequential scan
        h = torch.zeros(B, d, d_state, device=x.device, dtype=x.dtype)
        ys = []
        for i in range(L):
            h = delta_A[:, i] * h + delta_B[:, i] * x[:, i].unsqueeze(-1)
            y = (h * C_ssm[:, i].unsqueeze(2)).sum(-1)
            ys.append(y)
        y = torch.stack(ys, dim=1)  # (B, L, d)
        return y + x * self.D

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, d_model, *spatial) — output of lifting layer.
               Spatial dims are flattened to a sequence for the SSM scan.
        Returns:
            Same shape as input.
        """
        shape = x.shape  # (B, C, ...)
        B, C = shape[0], shape[1]
        spatial = shape[2:]
        L = 1
        for s in spatial:
            L *= s

        # Flatten spatial → sequence: (B, L, C)
        x_seq = x.view(B, C, L).permute(0, 2, 1)
        residual = x_seq

        xz = self.in_proj(x_seq)
        x_inner, z = xz.chunk(2, dim=-1)

        # Causal depthwise conv over sequence
        x_conv = x_inner.permute(0, 2, 1)  # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[..., :L]  # causal truncation
        x_conv = F.silu(x_conv).permute(0, 2, 1)  # (B, L, d_inner)

        y = self._ssm(x_conv)
        y = y * F.silu(z)
        y = self.out_proj(y)
        y = self.norm(y + residual)

        return y.permute(0, 2, 1).view(shape)


# ---------------------------------------------------------------------------
# Local attention block (post-lifting LocalAttnFNO)
# ---------------------------------------------------------------------------

class LocalAttentionBlock(nn.Module):
    """
    Local self-attention over a fixed window size, applied after lifting.
    Operates on flattened spatial tokens.
    """

    def __init__(self, d_model: int, num_heads: int = 4, window_size: int = 16) -> None:
        super().__init__()
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        B, C = shape[0], shape[1]
        L = 1
        for s in shape[2:]:
            L *= s

        tokens = x.view(B, C, L).permute(0, 2, 1)  # (B, L, C)

        # Pad to multiple of window_size
        pad = (self.window_size - L % self.window_size) % self.window_size
        if pad > 0:
            tokens = F.pad(tokens, (0, 0, 0, pad))
        L_pad = tokens.shape[1]
        n_win = L_pad // self.window_size

        # Reshape into windows
        win = tokens.view(B * n_win, self.window_size, C)
        attn_out, _ = self.attn(win, win, win)
        win = self.norm(win + attn_out)
        win = self.norm2(win + self.ff(win))

        tokens = win.view(B, L_pad, C)[:, :L, :]
        return tokens.permute(0, 2, 1).view(shape)


# ---------------------------------------------------------------------------
# Codomain attention (CoDA-NO, from Rahman et al. 2024)
# ---------------------------------------------------------------------------

class CodomainAttention(nn.Module):
    """
    Codomain attention: dot-product similarity is computed between *features*
    (codomain channels) rather than between spatial samples.

    For input X of shape (B, C, *spatial):
      - Queries, keys, values are obtained by FNO-based mappings over spatial dims
      - Attention weights are computed across the channel (codomain) dimension
    """

    def __init__(
        self,
        channels: int,
        modes1: int,
        modes2: int,
        num_heads: int = 4,
        spatial_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0

        if spatial_dim == 1:
            self.fno_q = SpectralConv1d(channels, channels, modes1)
            self.fno_k = SpectralConv1d(channels, channels, modes1)
            self.fno_v = SpectralConv1d(channels, channels, modes1)
        else:
            self.fno_q = SpectralConv2d(channels, channels, modes1, modes2)
            self.fno_k = SpectralConv2d(channels, channels, modes1, modes2)
            self.fno_v = SpectralConv2d(channels, channels, modes1, modes2)

        self.out_proj = nn.Conv1d(channels, channels, 1) if spatial_dim == 1 else nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(num_heads, channels)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, *spatial = x.shape
        L = 1
        for s in spatial:
            L *= s

        Q = self.fno_q(x)  # (B, C, *spatial)
        K = self.fno_k(x)
        V = self.fno_v(x)

        # Flatten spatial: (B, C, L)
        Q = Q.view(B, C, L)
        K = K.view(B, C, L)
        V = V.view(B, C, L)

        # Codomain attention: attend over channels (C), aggregate over spatial (L)
        # Q: (B, heads, head_dim, L) → treat head_dim as "query tokens"
        Q = Q.view(B, self.num_heads, self.head_dim, L)
        K = K.view(B, self.num_heads, self.head_dim, L)
        V = V.view(B, self.num_heads, self.head_dim, L)

        # Attention over head_dim dimension (codomain)
        attn = torch.einsum("bhql,bhkl->bhqk", Q, K) / self.scale  # (B, heads, head_dim, head_dim)
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("bhqk,bhkl->bhql", attn, V)  # (B, heads, head_dim, L)

        out = out.reshape(B, C, L).view(B, C, *spatial)
        out = self.out_proj(out)
        return self.norm(x + out)


# ---------------------------------------------------------------------------
# Perceiver IO cross-attention and self-attention blocks
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    """
    Cross-attention: queries from one source, keys/values from another.
    Used in Perceiver IO encoder and decoder.
    """

    def __init__(self, q_dim: int, kv_dim: int, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(q_dim, num_heads, kdim=kv_dim, vdim=kv_dim,
                                          dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(q_dim)
        self.norm_kv = nn.LayerNorm(kv_dim)
        self.ff = nn.Sequential(
            nn.Linear(q_dim, q_dim * 4),
            nn.GELU(),
            nn.Linear(q_dim * 4, q_dim),
        )
        self.norm_out = nn.LayerNorm(q_dim)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        attn_out, _ = self.attn(q_norm, kv_norm, kv_norm)
        q = q + attn_out
        q = self.norm_out(q + self.ff(q))
        return q


class SelfAttentionBlock(nn.Module):
    """Standard transformer self-attention block."""

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = self.norm2(x + self.ff(x))
        return x


# ---------------------------------------------------------------------------
# Swin-v2 style window attention block
# ---------------------------------------------------------------------------

class WindowAttention(nn.Module):
    """
    Shifted-window multi-head self-attention (Swin-v2 style).
    Operates on 2-D feature maps.
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        shift: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.shift = shift
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # Relative position bias table (Swin-v2 uses log-spaced coords)
        self.rel_pos_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.rel_pos_bias_table, std=0.02)

        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing="ij"
        ))  # (2, W, W)
        coords_flat = coords.flatten(1)  # (2, W*W)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, W*W, W*W)
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        rel_idx = rel.sum(-1)  # (W*W, W*W)
        self.register_buffer("rel_idx", rel_idx)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        assert L == H * W

        ws = self.window_size
        shift = ws // 2 if self.shift else 0

        x2d = x.view(B, H, W, C)
        if self.shift:
            x2d = torch.roll(x2d, shifts=(-shift, -shift), dims=(1, 2))

        # Pad to multiple of window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x2d = F.pad(x2d, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = x2d.shape[1], x2d.shape[2]

        # Partition into windows
        x_win = x2d.view(B, Hp // ws, ws, Wp // ws, ws, C)
        x_win = x_win.permute(0, 1, 3, 2, 4, 5).contiguous()
        nW = (Hp // ws) * (Wp // ws)
        x_win = x_win.view(B * nW, ws * ws, C)

        # Attention
        qkv = self.qkv(x_win).reshape(B * nW, ws * ws, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Swin-v2: cosine attention
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        rel_bias = self.rel_pos_bias_table[self.rel_idx.view(-1)].view(ws * ws, ws * ws, -1)
        rel_bias = rel_bias.permute(2, 0, 1).unsqueeze(0)
        attn = attn + rel_bias

        if self.shift:
            # Mask for shifted windows
            img_mask = torch.zeros(1, Hp, Wp, 1, device=x.device)
            slices_h = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
            slices_w = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
            cnt = 0
            for sh in slices_h:
                for sw in slices_w:
                    img_mask[:, sh, sw, :] = cnt
                    cnt += 1
            mask_win = img_mask.view(1, Hp // ws, ws, Wp // ws, ws, 1)
            mask_win = mask_win.permute(0, 1, 3, 2, 4, 5).contiguous().view(nW, ws * ws)
            attn_mask = mask_win.unsqueeze(1) - mask_win.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
            attn = attn + attn_mask.unsqueeze(1)

        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B * nW, ws * ws, C)
        out = self.proj(out)

        # Reverse window partition
        out = out.view(B, Hp // ws, Wp // ws, ws, ws, C)
        out = out.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, C)
        if pad_h > 0 or pad_w > 0:
            out = out[:, :H, :W, :].contiguous()
        if self.shift:
            out = torch.roll(out, shifts=(shift, shift), dims=(1, 2))

        return out.view(B, H * W, C)


class SwinBlock(nn.Module):
    """Swin-v2 transformer block (W-MSA or SW-MSA + FFN)."""

    def __init__(self, dim: int, window_size: int, num_heads: int, shift: bool = False) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, shift=shift)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.ff(self.norm2(x))
        return x
