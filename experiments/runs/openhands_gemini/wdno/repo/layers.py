
import torch
from torch import nn
from functools import partial
from einops import rearrange, reduce
import math

# Helper functions
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t):
    return t

# Residual module (from UNet in DDPM)
class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, groups=8):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(groups, dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(groups, dim_out),
            nn.SiLU(),
            nn.Conv2d(dim_out, dim_out, 3, padding=1)
        )
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else identity

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)

# Sinusoidal Positional Embeddings (for time encoding)
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# Attention Block
class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        sim = torch.einsum('b h d i, b h d j -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)

        out = torch.einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# Downsample and Upsample blocks
class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

# Helper for 3D operations (for 2D incompressible fluid)
class Block3D(nn.Module):
    def __init__(self, dim, dim_out, kernel_size=(3,3,3), padding=(1,1,1), stride=(1,1,1), groups=8):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(groups, dim),
            nn.SiLU(),
            nn.Conv3d(dim, dim_out, kernel_size, padding=padding, stride=stride)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(groups, dim_out),
            nn.SiLU(),
            nn.Conv3d(dim_out, dim_out, kernel_size, padding=padding, stride=stride)
        )
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else identity

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)

class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )
    def forward(self, x):
        return self.block(x)

class Downsample3D(nn.Module):
    def __init__(self, dim, kernel=(1,4,4), stride=(1,2,2), padding=(0,1,1)):
        super().__init__()
        self.conv = nn.Conv3d(dim, dim, kernel, stride=stride, padding=padding)

    def forward(self, x):
        return self.conv(x)

class Upsample3D(nn.Module):
    def __init__(self, dim, kernel=(1,4,4), stride=(1,2,2), padding=(0,1,1)):
        super().__init__()
        self.conv = nn.ConvTranspose3d(dim, dim, kernel, stride=stride, padding=padding)

    def forward(self, x):
        return self.conv(x)

