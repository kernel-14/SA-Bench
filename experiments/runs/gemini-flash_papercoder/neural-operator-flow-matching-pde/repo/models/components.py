import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

try:
    import xformers.ops
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False
    print("xformers not available. Falling back to standard attention. Install xformers for faster attention.")

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization layer.
    As described in Section 4.1 of the paper (Llama-2 architecture components).
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initializes the RMSNorm layer.

        Args:
            dim (int): The feature dimension over which to normalize.
            eps (float, optional): A small epsilon value for numerical stability. Defaults to 1e-6.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the RMS normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        # (x * x).mean(-1, keepdim=True) is faster than x.norm(2, dim=-1, keepdim=True)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after RMS normalization.
        """
        output = self._norm(x.float()).type_as(x) # Ensure computation is in float32 for stability
        return output * self.weight


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit (SwiGLU) activation function.
    Used in the Feed-Forward Network (FFN) layers of the Transformer,
    as specified in Section 4.1 of the paper (Llama-2 architecture components).
    """

    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        """
        Initializes the SwiGLU layer.

        Args:
            dim (int): Input dimension.
            hidden_dim (Optional[int]): The intermediate dimension of the FFN.
                                        If None, defaults to `int(2 * dim * 2 / 3)`.
        """
        super().__init__()
        # Default hidden_dim calculation (common for SwiGLU/FFN)
        hidden_dim = hidden_dim or int(2 * dim * 2 / 3) # Example: 4*dim is common, Llama-2 uses a specific ratio.
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False) # Gate projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for SwiGLU.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after SwiGLU activation.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    """
    Multi-head self-attention mechanism, leveraging FlashAttention v2 if available,
    otherwise falling back to standard scaled dot-product attention.
    As specified in Section 4.1 of the paper.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initializes the Attention layer.

        Args:
            dim (int): The embedding dimension of the input tokens.
            heads (int, optional): Number of attention heads. Defaults to 8.
            dim_head (int, optional): Dimension per attention head. Defaults to 64.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
            dtype (torch.dtype, optional): Data type for computations. Defaults to torch.float16.
        """
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = dim_head * heads
        self.scale = dim_head**-0.5
        self.dropout = nn.Dropout(dropout)
        self.dtype = dtype

        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for Attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            mask (Optional[torch.Tensor], optional): Optional attention mask. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim).
        """
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(t.shape[0], -1, h, self.dim_head), qkv)

        # Move head dimension to be batch-like for xformers (B, N, H, D) -> (B*H, N, D)
        # q, k, v = map(lambda t: t.transpose(1, 2).reshape(t.shape[0] * h, -1, self.dim_head), (q, k, v))
        # No, xformers.ops.memory_efficient_attention expects (B, N, H, D)
        
        # FlashAttention v2 if available
        if XFORMERS_AVAILABLE and x.device.type == 'cuda':
            # xformers expects input to be float16 or bfloat16 for efficient computation
            # The paper specifies float16
            q = q.to(self.dtype)
            k = k.to(self.dtype)
            v = v.to(self.dtype)
            
            out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=mask)
            out = out.view(x.shape[0], -1, self.inner_dim) # Reshape from (B, N, H, D) to (B, N, H*D)
        else:
            # Standard Scaled Dot-Product Attention
            # (B, N, H, D) -> (B, H, N, D)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            sim = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale

            if mask is not None:
                sim = sim.masked_fill(mask == 0, -torch.finfo(sim.dtype).max)

            attn = sim.softmax(dim=-1)
            out = torch.einsum("bhij,bhjd->bhid", attn, v)

            out = out.transpose(1, 2).contiguous().view(x.shape[0], -1, self.inner_dim)

        return self.to_out(out)


class AdaLNZero(nn.Module):
    """
    Adaptive Layer Normalization with Zero-initialization.
    Used for conditioning Transformer blocks on external information (time 't' and history 'h'),
    as part of the AdaLN-Zero mechanism (Section 4.1).
    """

    def __init__(self, embed_dim: int, cond_dim: int):
        """
        Initializes the AdaLNZero layer.

        Args:
            embed_dim (int): The dimension of the features to be normalized/conditioned.
            cond_dim (int): The dimension of the input conditioning vector.
        """
        super().__init__()
        self.scale_proj = nn.Linear(cond_dim, embed_dim, bias=True)
        self.shift_proj = nn.Linear(cond_dim, embed_dim, bias=True)

        # Zero-initialize the projections as per AdaLN-Zero design
        nn.init.constant_(self.scale_proj.bias, 0.0)
        nn.init.constant_(self.shift_proj.bias, 0.0)
        nn.init.constant_(self.scale_proj.weight, 0.0)
        nn.init.constant_(self.shift_proj.weight, 0.0)

    def forward(self, conditioning_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for AdaLNZero.

        Args:
            conditioning_vector (torch.Tensor): Input conditioning vector of shape (batch_size, cond_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing (scale, shift) tensors,
                                               both of shape (batch_size, 1, embed_dim) for broadcasting.
        """
        scale = self.scale_proj(conditioning_vector).unsqueeze(1) # (B, 1, embed_dim)
        shift = self.shift_proj(conditioning_vector).unsqueeze(1) # (B, 1, embed_dim)
        return scale, shift


class TransformerBlock(nn.Module):
    """
    Represents a single layer of the Scalable Interpolant Transformer (SiT).
    It combines attention, FFN, and AdaLNZero conditioning.
    As specified in Section 4.1 of the paper.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        ffn_expansion_factor: int = 2, # Common default, 4 for original Llama. 2 for SiT in some implementations.
        dropout: float = 0.0,
        conditioning_dim: int = 0, # Dimension of the global conditioning vector
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initializes a TransformerBlock.

        Args:
            dim (int): The embedding dimension.
            num_heads (int): Number of attention heads.
            head_dim (int): Dimension per attention head.
            ffn_expansion_factor (int, optional): Expansion factor for the FFN hidden dimension. Defaults to 2.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
            conditioning_dim (int, optional): Dimension of the global conditioning vector. Defaults to 0 (no conditioning).
            dtype (torch.dtype, optional): Data type for computations. Defaults to torch.float16.
        """
        super().__init__()
        self.dtype = dtype

        self.attn_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

        self.attn = Attention(dim, heads, head_dim, dropout, dtype=dtype)
        
        # FFN hidden dimension calculation. Llama-2 uses a specific multiplier and aligns to nearest multiple of 256.
        # For simplicity, we use a factor.
        ffn_hidden_dim = int(dim * ffn_expansion_factor * 2 / 3) # Llama-2 style
        ffn_hidden_dim = 256 * ((ffn_hidden_dim + 256 - 1) // 256) # Align to nearest multiple of 256 for Llama

        self.ffn = SwiGLU(dim, hidden_dim=ffn_hidden_dim)
        self.dropout_ffn = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        if conditioning_dim > 0:
            self.adaln_zero_attn = AdaLNZero(dim, conditioning_dim)
            self.adaln_zero_ffn = AdaLNZero(dim, conditioning_dim)
        else:
            self.adaln_zero_attn = None
            self.adaln_zero_ffn = None

    def forward(
        self,
        x: torch.Tensor,
        conditioning_vector: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for a TransformerBlock.

        Args:
            x (torch.Tensor): Input tokens of shape (batch_size, seq_len, dim).
            conditioning_vector (Optional[torch.Tensor], optional): Global conditioning vector
                                                                     of shape (batch_size, conditioning_dim).
                                                                     Defaults to None.
            attn_mask (Optional[torch.Tensor], optional): Optional attention mask. Defaults to None.

        Returns:
            torch.Tensor: Output tokens of shape (batch_size, seq_len, dim).
        """
        h = x
        # Self-Attention Sub-layer
        attn_input = self.attn_norm(x)
        if self.adaln_zero_attn is not None and conditioning_vector is not None:
            scale_attn, shift_attn = self.adaln_zero_attn(conditioning_vector)
            attn_input = attn_input * (1 + scale_attn) + shift_attn
        
        attn_output = self.attn(attn_input, mask=attn_mask)
        x = h + self.dropout_attn(attn_output)

        # Feed-Forward Sub-layer
        h = x
        ffn_input = self.ffn_norm(x)
        if self.adaln_zero_ffn is not None and conditioning_vector is not None:
            scale_ffn, shift_ffn = self.adaln_zero_ffn(conditioning_vector)
            ffn_input = ffn_input * (1 + scale_ffn) + shift_ffn
        
        ffn_output = self.ffn(ffn_input)
        x = h + self.dropout_ffn(ffn_output)

        return x.to(self.dtype)


class CrossAttention(nn.Module):
    """
    Cross-attention mechanism used to compress a sequence of tokens (context)
    into a single representative token (query).
    Used for creating a single token from `x_t^k` to update the GRU's history state `h`,
    as described in Section 4.1.
    """

    def __init__(
        self,
        query_dim: int, # Dimension of the query token (e.g., Transformer's embedding_dim)
        context_dim: int, # Dimension of the context tokens (e.g., Transformer's embedding_dim for latent_y_tk)
        output_dim: int, # Dimension of the resulting single token
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initializes the CrossAttention layer.

        Args:
            query_dim (int): Dimension of the query token.
            context_dim (int): Dimension of the context tokens.
            output_dim (int): Dimension of the resulting single token.
            heads (int, optional): Number of attention heads. Defaults to 8.
            dim_head (int, optional): Dimension per attention head. Defaults to 64.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
            dtype (torch.dtype, optional): Data type for computations. Defaults to torch.float16.
        """
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim_q = dim_head * heads
        self.inner_dim_kv = dim_head * heads
        self.scale = dim_head**-0.5
        self.dtype = dtype

        self.norm_query = RMSNorm(query_dim)
        self.norm_context = RMSNorm(context_dim)

        self.to_q = nn.Linear(query_dim, self.inner_dim_q, bias=False)
        self.to_k = nn.Linear(context_dim, self.inner_dim_kv, bias=False)
        self.to_v = nn.Linear(context_dim, self.inner_dim_kv, bias=False)
        self.to_out = nn.Linear(self.inner_dim_q, output_dim, bias=False) # Output projection matches query inner_dim if Q/K/V dims align

        self.dropout = nn.Dropout(dropout)

        # Learned query token: (1, 1, query_dim)
        self.learned_query_token = nn.Parameter(torch.randn(1, 1, query_dim))


    def forward(self, context_tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for CrossAttention to compress context tokens into a single token.

        Args:
            context_tokens (torch.Tensor): Context tokens of shape (batch_size, num_context_tokens, context_dim).
                                           This typically represents the tokenized latent_y_tk.

        Returns:
            torch.Tensor: A single compressed token of shape (batch_size, output_dim).
        """
        B, N, D_context = context_tokens.shape
        H = self.heads

        # Expand learned query token for the batch
        query_token_expanded = self.learned_query_token.expand(B, -1, -1) # (B, 1, query_dim)

        # Normalize inputs
        query_norm = self.norm_query(query_token_expanded)
        context_norm = self.norm_context(context_tokens)

        # Project to Q, K, V
        q = self.to_q(query_norm).view(B, -1, H, self.dim_head)  # (B, 1, H, D_head)
        k = self.to_k(context_norm).view(B, -1, H, self.dim_head) # (B, N, H, D_head)
        v = self.to_v(context_norm).view(B, -1, H, self.dim_head) # (B, N, H, D_head)

        # Transpose for batch-head-seq-dim format
        q = q.transpose(1, 2) # (B, H, 1, D_head)
        k = k.transpose(1, 2) # (B, H, N, D_head)
        v = v.transpose(1, 2) # (B, H, N, D_head)

        # Compute attention scores
        sim = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale # (B, H, 1, N)

        attn = sim.softmax(dim=-1) # (B, H, 1, N)
        attn = self.dropout(attn)

        # Compute weighted sum of values
        out = torch.einsum("bhij,bhjd->bhid", attn, v) # (B, H, 1, D_head)

        # Concatenate heads and project to output dimension
        out = out.transpose(1, 2).contiguous().view(B, 1, self.inner_dim_q) # (B, 1, inner_dim_q)

        return self.to_out(out).squeeze(1).to(self.dtype) # (B, output_dim)


def downsample_latent(latent_y: torch.Tensor, factor: int) -> torch.Tensor:
    """
    Spatially downsamples a latent feature map by a given factor.
    Used to create the latent temporal pyramids (Section 3.3).

    Args:
        latent_y (torch.Tensor): Latent tensor of shape (B, C, H, W).
        factor (int): The downsampling factor.

    Returns:
        torch.Tensor: The spatially downsampled latent tensor.
    """
    if factor == 1:
        return latent_y
    
    # Use F.interpolate with 'area' mode for robust downsampling.
    # Align_corners=False is usually preferred for downsampling.
    # Output size is (H // factor, W // factor)
    output_size = (latent_y.shape[2] // factor, latent_y.shape[3] // factor)
    
    return F.interpolate(latent_y, size=output_size, mode='area', antialias=True)


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Generates sinusoidal positional embeddings for scalar values, e.g., time 't'.
    This converts a single scalar into a higher-dimensional embedding.
    """

    def __init__(self, dim: int, max_period: float = 10000.0):
        """
        Initializes the SinusoidalPositionalEmbedding layer.

        Args:
            dim (int): The output embedding dimension. Must be even.
            max_period (float, optional): The maximum period for the sinusoidal functions. Defaults to 10000.0.
        """
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Embedding dimension (dim) must be an even number, but got {dim}.")
        self.dim = dim
        self.max_period = max_period
        
        # Frequencies for the sinusoidal encoding
        # This calculates `1 / (max_period^(2i/dim))` for i from 0 to dim/2 - 1
        self.register_buffer('inv_freq', 1.0 / (max_period**(torch.arange(0, dim, 2).float() / dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute sinusoidal embeddings.

        Args:
            x (torch.Tensor): Input scalar or batch of scalars (e.g., time 't').
                              Shape can be (batch_size,) or (batch_size, 1).
                              Values are typically in [0, 1].

        Returns:
            torch.Tensor: Sinusoidal positional embedding of shape (batch_size, dim).
        """
        # Ensure x is 1D (batch_size,) for element-wise operations with inv_freq
        x = x.squeeze(-1) if x.dim() > 1 else x
        
        # outer product: (B,) * (dim/2,) -> (B, dim/2)
        emb = x.unsqueeze(-1) * self.inv_freq # (B, dim/2)

        # Concatenate sin and cos components
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1) # (B, dim)
        return emb

