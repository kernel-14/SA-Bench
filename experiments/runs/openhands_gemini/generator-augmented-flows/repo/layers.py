
import torch
from torch import nn
import math

class SinusoidalPositionalEmbedding(nn.Module):
    """
    Sinusoidal Positional Embedding as used in Transformer models and diffusion models.
    From original Consistency Models paper implementation or similar diffusion model implementations.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class WeightStandardizedConv2d(nn.Conv2d):
    """
    Conv2d with Weight Standardization.
    Used in some advanced diffusion models to improve training stability.
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        weight = self.weight
        mean = weight.mean(dim=1, keepdim=True, prior=True)
        var = weight.var(dim=1, keepdim=True, prior=True, unbiased=False)
        normalized_weight = (weight - mean) * (var + eps).rsqrt()
        return nn.functional.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class ResidualBlock(nn.Module):
    """
    A standard Residual Block with GroupNorm and Swish activation.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(), # Swish activation
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb):
        h = self.conv1(x)
        h += self.time_mlp(time_emb)[:, :, None, None] # Add time embedding
        h = self.conv2(h)
        return h + self.residual_conv(x)

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class AttentionBlock(nn.Module):
    """
    Self-attention block.
    """
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.view(b, self.heads, t.shape[1] // self.heads, h * w).transpose(2, 3), qkv)

        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(2, 3).reshape(b, -1, h, w)
        return self.to_out(out)


