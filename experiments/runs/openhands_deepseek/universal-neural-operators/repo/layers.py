"""Core layers for neural operators: FNO spectral convolutions, Mamba SSM,
Perceiver IO blocks, codomain attention, and Swin Transformer V2 blocks."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
#  Fourier Neural Operator spectral convolution
# ---------------------------------------------------------------------------

class SpectralConv1d(nn.Module):
    """1D Fourier layer: FFT -> linear transform on modes -> IFFT."""

    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x):
        B, C, N = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(B, self.out_channels, x_ft.shape[-1],
                              dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = torch.einsum(
            'bix,iox->box', x_ft[:, :, :self.modes], self.weights
        )
        x = torch.fft.irfft(out_ft, n=N, dim=-1)
        return x


class SpectralConv2d(nn.Module):
    """2D Fourier layer for 2D problems (e.g., Navier–Stokes, reaction–diffusion)."""

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))
        out_ft = torch.zeros(B, self.out_channels, H, x_ft.shape[-1],
                              dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            'bixy,ioxy->boxy',
            x_ft[:, :, :self.modes1, :self.modes2],
            self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            'bixy,ioxy->boxy',
            x_ft[:, :, -self.modes1:, :self.modes2],
            self.weights2
        )
        x = torch.fft.irfft2(out_ft, s=(H, W), dim=(-2, -1))
        return x


class FNOBlock1d(nn.Module):
    """FNO integral operator block (1D)."""

    def __init__(self, hidden_channels, modes, activation=F.gelu):
        super().__init__()
        self.spectral = SpectralConv1d(hidden_channels, hidden_channels, modes)
        self.linear = nn.Conv1d(hidden_channels, hidden_channels, 1)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.spectral(x) + self.linear(x))


class FNOBlock2d(nn.Module):
    """FNO integral operator block (2D)."""

    def __init__(self, hidden_channels, modes1, modes2, activation=F.gelu):
        super().__init__()
        self.spectral = SpectralConv2d(hidden_channels, hidden_channels, modes1, modes2)
        self.linear = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.spectral(x) + self.linear(x))


# ---------------------------------------------------------------------------
#  Mamba SSM (post-lifting)  –  Section 3, Equation (2)
# ---------------------------------------------------------------------------

class MambaSSM(nn.Module):
    """
    Structured state-space model applied after the lifting map.
    Encodes long-range temporal and spatial dependencies via learnable
    causal convolution kernels K_tau.

    For lifted features v_0(x,t) computes:
        tilde{v}_0(x,t) = sum_{tau <= t} K_tau * v_0(x, t - tau)
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.expand = expand
        d_inner = int(expand * d_model)

        self.in_proj = nn.Linear(d_model, d_inner * 2)
        self.conv1d = nn.Conv1d(
            in_channels=d_inner, out_channels=d_inner,
            kernel_size=d_conv, groups=d_inner, padding=d_conv - 1
        )
        self.x_proj = nn.Linear(d_inner, d_state * 2 + d_inner)
        self.dt_proj = nn.Linear(d_state, d_inner)
        self.out_proj = nn.Linear(d_inner, d_model)
        self.d_state = d_state
        self.d_inner = d_inner

    def forward(self, x):
        # x: (B, N, d_model)  where N = spatial * temporal (flattened)
        B, N, D = x.shape
        x_and_res = self.in_proj(x)          # (B, N, 2 * d_inner)
        x_proj, z = x_and_res.chunk(2, dim=-1)

        x_conv = rearrange(x_proj, 'b n d -> b d n')
        x_conv = self.conv1d(x_conv)[..., :N]
        x_conv = F.silu(x_conv)
        x_conv = rearrange(x_conv, 'b d n -> b n d')

        x_db = self.x_proj(x_conv)
        dt, B_c, C = torch.split(x_db, [self.d_state, self.d_state, self.d_inner], dim=-1)
        dt = F.softplus(dt)

        # Selective scan (simplified parallel scan via convolution approximation)
        A = -torch.exp(torch.linspace(0, math.log(0.5), self.d_state, device=x.device))
        A = A[None, None, :, None]  # (1, 1, d_state, 1)

        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        ys = []
        for t in range(N):
            dt_t = dt[:, t:t + 1, :]          # (B, 1, d_state)
            B_t = B_c[:, t:t + 1, :]          # (B, 1, d_state)
            C_t = C[:, t:t + 1, :]            # (B, 1, d_inner)
            x_t = x_conv[:, t:t + 1, :]        # (B, 1, d_inner)
            h = h * torch.exp(A * dt_t.unsqueeze(-2)) + \
                (x_t.unsqueeze(-1) * B_t.unsqueeze(-2)) * dt_t.unsqueeze(-2)
            y = (h @ C_t.unsqueeze(-1)).squeeze(-1)  # (B, 1, d_inner)
            ys.append(y)
        y = torch.cat(ys, dim=1)               # (B, N, d_inner)

        y = y * F.silu(z)
        out = self.out_proj(y)
        return out


# ---------------------------------------------------------------------------
#  Codomain attention  (CoDA-NO)  –  Section 3 / Rahman et al.
# ---------------------------------------------------------------------------

class CodomainAttention(nn.Module):
    """
    Codomain attention: dot-product similarity computed between features
    (function-space channels) rather than between spatial samples.
    """

    def __init__(self, dim, num_heads=8, use_fno_proj=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0
        self.scale = self.head_dim ** -0.5
        self.use_fno_proj = use_fno_proj

        if use_fno_proj:
            self.q_proj = SpectralConv1d(dim, dim, min(dim // 4, 12))
            self.k_proj = SpectralConv1d(dim, dim, min(dim // 4, 12))
            self.v_proj = SpectralConv1d(dim, dim, min(dim // 4, 12))
        else:
            self.q_proj = nn.Linear(dim, dim)
            self.k_proj = nn.Linear(dim, dim)
            self.v_proj = nn.Linear(dim, dim)

        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        x_t = rearrange(x, 'b n d -> b d n')

        if self.use_fno_proj:
            Q = rearrange(self.q_proj(x_t), 'b d n -> b n d')
            K = rearrange(self.k_proj(x_t), 'b d n -> b n d')
            V = rearrange(self.v_proj(x_t), 'b d n -> b n d')
        else:
            Q = self.q_proj(x)
            K = self.k_proj(x)
            V = self.v_proj(x)

        Q = rearrange(Q, 'b n (h d) -> b h n d', h=self.num_heads)
        K = rearrange(K, 'b n (h d) -> b h n d', h=self.num_heads)
        V = rearrange(V, 'b n (h d) -> b h n d', h=self.num_heads)

        attn = torch.einsum('b h n d, b h m d -> b h n m', Q, K) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('b h n m, b h m d -> b h n d', attn, V)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.out_proj(out)


# ---------------------------------------------------------------------------
#  Perceiver IO blocks  –  Section 3
# ---------------------------------------------------------------------------

class PerceiverCrossAttention(nn.Module):
    """Cross-attention: queries from latent, keys/values from input data."""

    def __init__(self, query_dim, kv_dim, hidden_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0
        self.scale = self.head_dim ** -0.5
        self.to_q = nn.Linear(query_dim, hidden_dim)
        self.to_kv = nn.Linear(kv_dim, hidden_dim * 2)
        self.to_out = nn.Linear(hidden_dim, query_dim)

    def forward(self, x_latent, x_data):
        B, Nl, _ = x_latent.shape
        B, Nd, _ = x_data.shape

        Q = self.to_q(x_latent)
        K, V = self.to_kv(x_data).chunk(2, dim=-1)

        Q = rearrange(Q, 'b n (h d) -> b h n d', h=self.num_heads)
        K = rearrange(K, 'b n (h d) -> b h n d', h=self.num_heads)
        V = rearrange(V, 'b n (h d) -> b h n d', h=self.num_heads)

        attn = torch.einsum('b h n d, b h m d -> b h n m', Q, K) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('b h n m, b h m d -> b h n d', attn, V)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    def forward_kv(self, x_latent_q, x_latent_kv):
        """Cross-attention where keys/values come from transformed latents."""
        B, Nq, _ = x_latent_q.shape
        B, Nkv, _ = x_latent_kv.shape

        Q = self.to_q(x_latent_q)
        K, V = self.to_kv(x_latent_kv).chunk(2, dim=-1)

        Q = rearrange(Q, 'b n (h d) -> b h n d', h=self.num_heads)
        K = rearrange(K, 'b n (h d) -> b h n d', h=self.num_heads)
        V = rearrange(V, 'b n (h d) -> b h n d', h=self.num_heads)

        attn = torch.einsum('b h n d, b h m d -> b h n m', Q, K) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('b h n m, b h m d -> b h n d', attn, V)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class PerceiverSelfAttention(nn.Module):
    """Self-attention between latent representations."""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0
        self.scale = self.head_dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        Q, K, V = self.to_qkv(x).chunk(3, dim=-1)
        Q = rearrange(Q, 'b n (h d) -> b h n d', h=self.num_heads)
        K = rearrange(K, 'b n (h d) -> b h n d', h=self.num_heads)
        V = rearrange(V, 'b n (h d) -> b h n d', h=self.num_heads)
        attn = torch.einsum('b h n d, b h m d -> b h n m', Q, K) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('b h n m, b h m d -> b h n d', attn, V)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class PerceiverIOBlock(nn.Module):
    """
    Perceiver IO block as described in Section 3:
      1. Cross-attention: Q=latent, K/V = FNO(X)
      2. Self-attention on latent representations
      3. Cross-attention: Q=FNO(X), K/V = transformed latents -> output
    """

    def __init__(self, input_dim, latent_dim, num_latents, num_heads=8,
                 fno_modes=12, use_fno_proj=True):
        super().__init__()
        self.num_latents = num_latents
        self.latent_dim = latent_dim
        self.latent = nn.Parameter(torch.randn(1, num_latents, latent_dim) * 0.02)

        if use_fno_proj:
            self.fno_k1 = SpectralConv1d(input_dim, input_dim, fno_modes)
            self.fno_v1 = SpectralConv1d(input_dim, input_dim, fno_modes)
            self.fno_q2 = SpectralConv1d(input_dim, input_dim, fno_modes)
        else:
            self.fno_k1 = nn.Identity()
            self.fno_v1 = nn.Identity()
            self.fno_q2 = nn.Identity()

        self.cross_attn_in = PerceiverCrossAttention(latent_dim, input_dim, latent_dim, num_heads)
        self.self_attn = PerceiverSelfAttention(latent_dim, num_heads)
        self.cross_attn_out = PerceiverCrossAttention(input_dim, latent_dim, latent_dim, num_heads)

        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Linear(latent_dim * 4, latent_dim),
        )

    def forward(self, x):
        B, N, D = x.shape

        # Input cross-attention: Q = latent, K/V = FNO(x)
        x_t = rearrange(x, 'b n d -> b d n')
        k1 = rearrange(self.fno_k1(x_t), 'b d n -> b n d')
        v1 = rearrange(self.fno_v1(x_t), 'b d n -> b n d')
        kv_data = k1 + v1  # simple fusion

        lat = self.latent.expand(B, -1, -1)
        lat = lat + self.cross_attn_in(lat, kv_data)

        # Self-attention on latents
        lat_norm = self.norm1(lat)
        lat = lat + self.self_attn(lat_norm)

        # FFN
        lat_norm = self.norm2(lat)
        lat = lat + self.ffn(lat_norm)

        # Output cross-attention: Q = FNO(x), K/V = latents
        q2 = rearrange(self.fno_q2(x_t), 'b d n -> b n d')
        out = self.cross_attn_out.forward_kv(q2, lat)
        return out


# ---------------------------------------------------------------------------
#  Swin Transformer V2 blocks
# ---------------------------------------------------------------------------

class SwinV2WindowAttention(nn.Module):
    """Window-based multi-head self attention (Swin V2)."""

    def __init__(self, dim, window_size, num_heads=8):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.logit_scale = nn.Parameter(torch.log(10.0 * torch.ones(num_heads, 1, 1)))

    def forward(self, x):
        B, N, D = x.shape
        Q, K, V = self.qkv(x).chunk(3, dim=-1)
        Q = rearrange(Q, 'b n (h d) -> b h n d', h=self.num_heads)
        K = rearrange(K, 'b n (h d) -> b h n d', h=self.num_heads)
        V = rearrange(V, 'b n (h d) -> b h n d', h=self.num_heads)

        # Cosine attention
        Q = F.normalize(Q, dim=-1)
        K = F.normalize(K, dim=-1)
        attn = torch.einsum('b h n d, b h m d -> b h n m', Q, K)
        attn = attn * torch.exp(self.logit_scale)
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum('b h n m, b h m d -> b h n d', attn, V)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.proj(out)


class SwinV2Block(nn.Module):
    """Swin Transformer V2 block with window attention."""

    def __init__(self, dim, input_resolution, window_size, num_heads=8, shift_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = SwinV2WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def _window_partition(self, x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
        return windows

    def _window_reverse(self, windows, window_size, H, W):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def forward(self, x):
        B, N, D = x.shape
        H = W = int(N ** 0.5)
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, D)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        x_windows = self._window_partition(x, self.window_size)
        x_windows = self.attn(x_windows)
        x = self._window_reverse(x_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.view(B, H * W, D)
        x = shortcut + x

        shortcut = x
        x = shortcut + self.mlp(self.norm2(x))
        return x


class SwinV2Stage(nn.Module):
    """A Swin V2 stage with multiple blocks."""

    def __init__(self, dim, input_resolution, depth, window_size, num_heads=8):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            shift_size = 0 if (i % 2 == 0) else window_size // 2
            self.blocks.append(
                SwinV2Block(dim, input_resolution, window_size, num_heads, shift_size)
            )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


# ---------------------------------------------------------------------------
#  Local attention FNO
# ---------------------------------------------------------------------------

class LocalAttentionFNO(nn.Module):
    """Local attention block that can be used post-lifting, similar to MambaFNO."""

    def __init__(self, dim, window_size=16, num_heads=8):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, D = x.shape
        shortcut = x
        x = self.norm(x)

        # Pad to multiple of window_size
        pad_len = (self.window_size - N % self.window_size) % self.window_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))

        B2, N2, D2 = x.shape
        num_windows = N2 // self.window_size
        x = x.view(B2, num_windows, self.window_size, D2)

        Q, K, V = self.qkv(x).chunk(3, dim=-1)
        Q = rearrange(Q, 'b w n (h d) -> b w h n d', h=self.num_heads)
        K = rearrange(K, 'b w n (h d) -> b w h n d', h=self.num_heads)
        V = rearrange(V, 'b w n (h d) -> b w h n d', h=self.num_heads)

        attn = torch.einsum('b w h n d, b w h m d -> b w h n m', Q, K) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('b w h n m, b w h m d -> b w h n d', attn, V)
        out = rearrange(out, 'b w h n d -> b w n (h d)')
        out = out.reshape(B2, N2, D2)

        if pad_len > 0:
            out = out[:, :N, :]

        return shortcut + self.proj(out)


# ---------------------------------------------------------------------------
#  Lift and Project (adapters)  –  Section 3
# ---------------------------------------------------------------------------

class Lift(nn.Module):
    """Lifting map: input functions -> hidden representation.

    When mode='mlp': input (B, N, in_channels) -> (B, N, hidden_channels)
    When mode='conv1d': input (B, in_channels, N) -> (B, hidden_channels, N)
    When mode='conv2d': input (B, in_channels, H, W) -> (B, hidden_channels, H, W)
    """

    def __init__(self, in_channels, hidden_channels, mode='mlp'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = hidden_channels
        self.mode = mode
        if mode == 'mlp':
            self.net = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.GELU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
        elif mode == 'conv1d':
            self.net = nn.Conv1d(in_channels, hidden_channels, 1)
        elif mode == 'conv2d':
            self.net = nn.Conv2d(in_channels, hidden_channels, 1)
        else:
            raise ValueError(f"Unknown lift mode: {mode}")

    def forward(self, x):
        return self.net(x)


class Project(nn.Module):
    """Projection map: hidden representation -> output functions.

    When mode='mlp': input (B, N, hidden_channels) -> (B, N, out_channels)
    When mode='conv1d': input (B, hidden_channels, N) -> (B, out_channels, N)
    When mode='conv2d': input (B, hidden_channels, H, W) -> (B, out_channels, H, W)
    """

    def __init__(self, hidden_channels, out_channels, mode='mlp'):
        super().__init__()
        self.in_channels = hidden_channels
        self.out_channels = out_channels
        self.mode = mode
        if mode == 'mlp':
            self.net = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.GELU(),
                nn.Linear(hidden_channels, out_channels),
            )
        elif mode == 'conv1d':
            self.net = nn.Conv1d(hidden_channels, out_channels, 1)
        elif mode == 'conv2d':
            self.net = nn.Conv2d(hidden_channels, out_channels, 1)
        else:
            raise ValueError(f"Unknown project mode: {mode}")

    def forward(self, x):
        return self.net(x)
