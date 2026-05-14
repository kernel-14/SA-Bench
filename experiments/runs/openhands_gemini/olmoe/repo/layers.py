
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout_rate, use_qk_norm, layer_norm_epsilon, rope_theta, use_bias: bool = False): # Added use_bias
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=use_bias) # Use use_bias
        self.k_proj = nn.Linear(dim, dim, bias=use_bias) # Use use_bias
        self.v_proj = nn.Linear(dim, dim, bias=use_bias) # Use use_bias
        self.out_proj = nn.Linear(dim, dim, bias=use_bias) # Use use_bias

        self.dropout = nn.Dropout(dropout_rate)

        self.use_qk_norm = use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=layer_norm_epsilon)
            self.k_norm = RMSNorm(self.head_dim, eps=layer_norm_epsilon)
        
        self.rope = RotaryPositionEmbedding(self.head_dim, theta=rope_theta)

    def forward(self, x, mask=None):
        batch_size, seq_len, dim = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(q, 'b s (h d) -> b s h d', h=self.num_heads)
        k = rearrange(k, 'b s (h d) -> b s h d', h=self.num_heads)
        v = rearrange(v, 'b s (h d) -> b s h d', h=self.num_heads)
        
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply RoPE
        cos_freqs, sin_freqs = self.rope(seq_len, x.device) # Corrected call
        q, k = self.rope.apply_rotary_pos_emb(q, k, cos_freqs, sin_freqs) # Corrected call

        # Scaled Dot-Product Attention
        scores = torch.einsum('b s h d, b t h d -> b h s t', q, k) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('b h s t, b t h d -> b s h d', attn, v)
        out = rearrange(out, 'b s h d -> b s (h d)')
        return self.out_proj(out)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_hidden_dim, dropout_rate, use_bias: bool = False):
        super().__init__()
        # Paper implies SwiGLU activation. FFN dimension is (input_dim -> ffn_hidden_dim*2 -> ffn_hidden_dim -> output_dim)
        # For SwiGLU, input projection maps to 2*ffn_hidden_dim.
        self.w1 = nn.Linear(dim, ffn_hidden_dim * 2, bias=use_bias)
        self.w2 = nn.Linear(ffn_hidden_dim, dim, bias=use_bias)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = SwiGLU() # SwiGLU is applied after w1 projection

    def forward(self, x):
        return self.w2(self.dropout(self.activation(self.w1(x))))

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb_cos = freqs.cos()
        emb_sin = freqs.sin()
        return emb_cos, emb_sin

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q, k, cos_freqs, sin_freqs):
        cos_freqs_expanded = cos_freqs.unsqueeze(0).unsqueeze(2).repeat(1, 1, q.shape[2], 2)
        sin_freqs_expanded = sin_freqs.unsqueeze(0).unsqueeze(2).repeat(1, 1, q.shape[2], 2)

        q_embed = (q * cos_freqs_expanded) + (self.rotate_half(q) * sin_freqs_expanded)
        k_embed = (k * cos_freqs_expanded) + (self.rotate_half(k) * sin_freqs_expanded)

        return q_embed, k_embed

