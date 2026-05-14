
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class SwiGLU(nn.Module):
    """
    SwiGLU activation function as described in Llama-2 (Touvron et al., 2023).
    Combines SiLU (Swish) activation with a gated linear unit.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.w1 = nn.Linear(in_features, out_features * 2)
        self.w2 = nn.Linear(out_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split the output of w1 into two parts for gating
        gate, linear = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * linear)

class RMSNorm(nn.Module):
    """
    RMSNorm as described in Llama-2 (Touvron et al., 2023).
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class AdaLNZero(nn.Module):
    """
    Adaptive Layer Normalization Zero (AdaLN-Zero) mechanism introduced in (Peebles and Xie, 2023).
    This layer predicts modulation parameters (scale, shift, alpha) from a conditioning input.
    """
    def __init__(self, embed_dim: int, num_features: int):
        super().__init__()
        self.norm = RMSNorm(embed_dim)
        self.linear = nn.Linear(num_features, 6 * embed_dim) # Predicts 3 sets of (scale, shift) parameters for each attention block or 2 for simple cases

        # Initialize weights and biases to zero for stability
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        # c is the conditioning input, typically a timestep embedding or a latent state.
        # It's expanded to match the sequence length of x if necessary.
        c_proj = self.linear(c).unsqueeze(1) # [batch, 1, 6 * embed_dim]

        # Splitting c_proj into scale, shift, and alpha for different parts of the block
        # For a standard transformer block with self-attention and FFN, we need 6 parameters:
        # 1. Self-attention input: scale, shift
        # 2. FFN input: scale, shift
        # 3. Output modulation: alpha_attn, alpha_mlp (similar to residual scaling)
        scale_attn, shift_attn, \
        scale_mlp, shift_mlp, \
        alpha_attn, alpha_mlp = c_proj.chunk(6, dim=-1)

        # Apply AdaLN-Zero normalization
        x_norm = self.norm(x)

        return x_norm, scale_attn, shift_attn, scale_mlp, shift_mlp, alpha_attn, alpha_mlp


class SelfAttention(nn.Module):
    """
    Multi-head self-attention with head_dim 64 and FlashAttention v2-like logic (Dao, 2023).
    Note: Actual FlashAttention v2 requires specific CUDA kernels, this is a conceptual implementation.
    """
    def __init__(self, embed_dim: int, num_heads: int, head_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.qkv_proj = nn.Linear(embed_dim, 3 * num_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                scale: torch.Tensor = None, shift: torch.Tensor = None) -> torch.Tensor:
        # x: [batch, sequence_length, embed_dim]

        qkv = self.qkv_proj(x).chunk(3, dim=-1) # Each is [batch, sequence_length, num_heads * head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.num_heads), qkv)

        # Apply scale and shift if provided (from AdaLN-Zero)
        if scale is not None and shift is not None:
            # Scale and shift are usually applied *before* the QKV projection,
            # but AdaLN-Zero in SiT applies them *after* the normalization and *before* attention.
            # Here, we assume they modulate the input to attention.
            # However, in Peebles & Xie (2023), AdaLN-Zero generates scale/shift for RMSNorm and alpha for residual.
            # So, scale/shift for attention would typically be applied to the normalized input 'x_norm'.
            # For simplicity, if passed directly, we'll apply it here to the queries/keys for an "attention conditioning" effect.
            # A more faithful implementation might integrate it into a TransformerBlock.
            # For now, we'll just use it to scale the attention scores.
            # The paper's description of AdaLN-Zero implies it scales/shifts the input *to* the sub-layers.
            # So, if this attention block is a sub-layer, it would receive x_norm.
            # Let's assume for now that 'scale' and 'shift' are meant to modulate QKV or input to QKV.
            # The prompt suggests a SiT architecture which uses AdaLN-Zero. In SiT, the scale/shift are applied to the input
            # of the self-attention and MLP blocks.
            pass # We'll handle scale/shift outside this block in a TransformerBlock

        # Scaled Dot-Product Attention
        # (B, H, N, D) @ (B, H, D, N) -> (B, H, N, N)
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # (B, H, N, N) @ (B, H, N, D) -> (B, H, N, D)
        output = attn_weights @ v

        output = rearrange(output, 'b h n d -> b n (h d)') # [batch, sequence_length, num_heads * head_dim]
        output = self.out_proj(output) # [batch, sequence_length, embed_dim]
        return output

class CrossAttention(nn.Module):
    """
    Cross-attention mechanism used to compress the current state onto a single token
    to update the latent state `h` in diffusion forcing.
    """
    def __init__(self, query_dim: int, context_dim: int, num_heads: int, head_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(query_dim, num_heads * head_dim, bias=False)
        self.kv_proj = nn.Linear(context_dim, 2 * num_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, query_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: Query input, e.g., the latent state h [batch, 1, query_dim]
        # context: Key/Value input, e.g., current state x_t_k [batch, sequence_length, context_dim]

        q = self.q_proj(x)
        kv = self.kv_proj(context).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), kv)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)

        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = attn_weights @ v
        output = rearrange(output, 'b h n d -> b n (h d)')
        output = self.out_proj(output)
        return output

