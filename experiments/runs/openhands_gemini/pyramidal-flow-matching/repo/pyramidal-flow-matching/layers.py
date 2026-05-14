
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) as described in Su et al., 2024.
    Applied to the temporal dimension.
    """
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, seq_len=None):
        # x: (batch, heads, seq_len, dim_head)
        if seq_len is None:
            seq_len = x.shape[-2]
        
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Apply sin and cos to the input
        # emb: (seq_len, dim)
        # x: (batch, heads, seq_len, dim_head)
        # rotated_x: (batch, heads, seq_len, dim_head)
        return self._rotate_queries_or_keys(x, emb)

    def _rotate_queries_or_keys(self, x, emb):
        # x: (batch, heads, seq_len, dim_head)
        # emb: (seq_len, dim)
        
        # Split x into two halves for complex number multiplication
        x_rot = x[..., :emb.shape[-1]]
        x_pass = x[..., emb.shape[-1]:]
        
        # Reshape emb for broadcasting
        emb = emb.view(1, 1, emb.shape[0], emb.shape[1])

        cos = emb.cos()
        sin = emb.sin()

        # Apply rotation (complex multiplication)
        x_rot_real = x_rot * cos - self._rotate_half(x_rot) * sin
        
        return torch.cat((x_rot_real, x_pass), dim=-1)

    def _rotate_half(self, x):
        # Rearranges the last dimension of x for rotation
        # [..., a, b] -> [..., b, -a]
        x = rearrange(x, '... (d r) -> ... d r', r=2)
        x1, x2 = x.unbind(dim=-1)
        return rearrange(torch.stack((-x2, x1), dim=-1), '... d r -> ... (d r)')

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None, rotary_pos_emb=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # B, num_heads, N, head_dim

        if rotary_pos_emb is not None:
            # Apply RoPE to queries and keys
            q = rotary_pos_emb(q)
            k = rotary_pos_emb(k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None, rotary_pos_emb=None):
        x = x + self.attn(self.norm1(x), mask=mask, rotary_pos_emb=rotary_pos_emb)
        x = x + self.mlp(self.norm2(x))
        return x

# VAE related layers
class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation=nn.ReLU, causal=False):
        super().__init__()
        self.causal = causal
        if causal:
            # For causal convolution, padding in temporal dimension needs to be handled carefully
            # A common way is to pad only on the left (past) side
            # Here we assume kernel_size is a tuple (T, H, W) and padding is (pad_T, pad_H, pad_W)
            # The actual convolution will be applied after padding
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, (0, padding[1], padding[2]))
            self.temporal_padding = (kernel_size[0] - 1) * stride[0] # Pad for causal convolution
        else:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = activation() if activation else nn.Identity()

    def forward(self, x):
        if self.causal:
            x = F.pad(x, (0, 0, 0, 0, self.temporal_padding, 0)) # Pad only in temporal dimension
        return self.activation(self.conv(x))

class Upsample3D(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)

class Downsample3D(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=1.0/self.scale_factor, mode=self.mode)

