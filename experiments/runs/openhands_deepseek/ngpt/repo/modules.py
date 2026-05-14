
"""Core modules for the Normalized Transformer (nGPT).

Implements:
- ScaledParameter: trainable parameters with init/scale decoupling (Section 2.5)
- NormalizedLinear: linear layer with column-wise L2 normalization
- Attention: multi-head self-attention with QK normalization
- MLP: SwiGLU MLP with rescaling
- TransformerBlock: GPT and nGPT block variants
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ScaledParameter(nn.Module):
    """Trainable parameter with init/scale decoupling for effective learning rate control.

    As described in Section 2.5: the parameter is stored with initial value s_scale,
    but during forward pass the actual value is restored by multiplying s_init / s_scale.
    This allows controlling the effective learning rate while keeping global LR unchanged.

    When s_scale = 1/√d_model and s_init = 1, the effective LR matches other normalized params.
    """

    def __init__(self, dim: int, s_init: float = 1.0, s_scale: float = 1.0):
        super().__init__()
        self.s_init = s_init
        self.s_scale = s_scale
        # Store parameter scaled: initial value = s_scale
        self.weight = nn.Parameter(torch.ones(dim) * s_scale)

    def forward(self) -> torch.Tensor:
        # Restore actual value: weight * (s_init / s_scale)
        return self.weight * (self.s_init / self.s_scale)

    def __repr__(self):
        return f"ScaledParameter(dim={self.weight.shape[0]}, s_init={self.s_init}, s_scale={self.s_scale})"


class NormalizedLinear(nn.Module):
    """Linear layer with column-wise L2 normalization along embedding dimension.

    W ∈ R^{d_in × d_out}: each column (embedding dimension) is normalized to unit norm.
    This makes matrix-vector multiplication a dot product (cosine similarity) bounded in [-1, 1].
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.reset_parameters()
        # No bias in normalized Transformer

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=1.0 / math.sqrt(self.in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize columns (embedding dim = dim 1 for weight stored as [out, in])
        w_norm = F.normalize(self.weight, p=2, dim=1, eps=1e-12)
        return F.linear(x, w_norm)

    def normalize_weights(self):
        """Post-training normalization: normalize along embedding dimension."""
        self.weight.data = F.normalize(self.weight.data, p=2, dim=1, eps=1e-12)


class NormalizedEmbedding(nn.Module):
    """Embedding layer with L2 normalized rows.

    Each embedding vector has unit norm, residing on the hypersphere.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=1.0 / math.sqrt(self.embedding_dim))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # Normalize rows
        w_norm = F.normalize(self.weight, p=2, dim=1, eps=1e-12)
        return F.embedding(idx, w_norm)

    def normalize_weights(self):
        self.weight.data = F.normalize(self.weight.data, p=2, dim=1, eps=1e-12)


class RotaryEmbedding(nn.Module):
    """RoPE (Rotary Position Embeddings) from Su et al. (2024).
    Applied pre-normalization to query and key in nGPT.
    """

    def __init__(self, dim: int, base: int = 10000, max_seq_len: int = 4096):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.register_buffer("freqs", self._compute_freqs())

    def _compute_freqs(self) -> torch.Tensor:
        theta = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        return theta

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        seq_len = x.shape[1]
        position = torch.arange(offset, offset + seq_len, device=x.device)
        freqs = torch.outer(position, self.freqs).float()
        freqs = torch.polar(torch.ones_like(freqs), freqs)  # complex numbers

        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_rotated = x_complex * freqs.unsqueeze(0).unsqueeze(0)
        x_out = torch.view_as_real(x_rotated).flatten(3)
        return x_out.type_as(x)


class Attention(nn.Module):
    """Multi-head self-attention for the Normalized Transformer.

    Key differences from baseline GPT:
    - Weight matrices are column-normalized (NormalizedLinear)
    - QK are normalized and scaled by s_qk (eqs 15, 16)
    - Softmax scaling factor is √d_k instead of 1/√d_k
    - Causal masking for autoregressive generation
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 4096,
        rope_base: int = 10000,
        use_qk_norm: bool = True,
        s_qk_init: float = 1.0,
        s_qk_scale: float = 1.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_qk_norm = use_qk_norm
        self.softmax_scale = math.sqrt(self.d_k)  # √d_k instead of 1/√d_k

        # Normalized projection matrices
        self.W_q = NormalizedLinear(d_model, d_model)
        self.W_k = NormalizedLinear(d_model, d_model)
        self.W_v = NormalizedLinear(d_model, d_model)
        self.W_o = NormalizedLinear(d_model, d_model)

        # QK scaling factor (per-head, d_k-dimensional)
        self.s_qk = ScaledParameter(self.d_k, s_qk_init, s_qk_scale)

        # Rotary embeddings
        self.rope = RotaryEmbedding(self.d_k, base=rope_base, max_seq_len=max_seq_len)

    def forward(
        self, h: torch.Tensor, causal_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, C = h.shape  # batch, seq_len, d_model

        # Project to Q, K, V
        q = self.W_q(h).view(B, T, self.n_heads, self.d_k)
        k = self.W_k(h).view(B, T, self.n_heads, self.d_k)
        v = self.W_v(h).view(B, T, self.n_heads, self.d_k)

        # Apply RoPE to Q and K
        q = self.rope(q)
        k = self.rope(k)

        # QK normalization (eqs 15, 16) - optional per ablation studies
        if self.use_qk_norm:
            q = F.normalize(q, p=2, dim=-1, eps=1e-12)
            k = F.normalize(k, p=2, dim=-1, eps=1e-12)

        # Apply QK scaling
        s_qk = self.s_qk()
        q = q * s_qk.view(1, 1, 1, -1)
        k = k * s_qk.view(1, 1, 1, -1)

        # Compute attention scores
        q = q.transpose(1, 2)  # (B, n_heads, T, d_k)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.softmax_scale

        # causal_mask shape: (1, 1, T, T), broadcasts over batch and heads
        if causal_mask is not None:
            if causal_mask.dim() == 2:
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores + causal_mask

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Concatenate heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.W_o(attn_output)

        return output

    def normalize_weights(self):
        """Normalize all weight matrices (Section 2.6, step 2)."""
        self.W_q.normalize_weights()
        self.W_k.normalize_weights()
        self.W_v.normalize_weights()
        self.W_o.normalize_weights()


class MLP(nn.Module):
    """SwiGLU MLP block for the Normalized Transformer.

    Key differences from baseline GPT:
    - Weight matrices are column-normalized (NormalizedLinear)
    - Intermediate states u and v are scaled by s_u and s_v (eqs 20, 21)
    - v is additionally rescaled by √d_model for SiLU non-linearity (Appendix A.1)
    """

    def __init__(
        self,
        d_model: int,
        d_mlp: int,
        s_u_init: float = 1.0,
        s_u_scale: float = 1.0,
        s_v_init: float = 1.0,
        s_v_scale: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_mlp = d_mlp

        self.W_u = NormalizedLinear(d_model, d_mlp)
        self.W_v = NormalizedLinear(d_model, d_mlp)
        self.W_o = NormalizedLinear(d_mlp, d_model)

        self.s_u = ScaledParameter(d_mlp, s_u_init, s_u_scale)
        self.s_v = ScaledParameter(d_mlp, s_v_init, s_v_scale)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # Project to u and v
        u = self.W_u(h)  # (B, T, d_mlp)
        v = self.W_v(h)  # (B, T, d_mlp)

        # Apply scaling factors (eqs 20, 21)
        u = u * self.s_u()
        # Rescale v by √d_model for SiLU non-linearity (Appendix A.1)
        v = v * self.s_v() * math.sqrt(self.d_model)

        # SwiGLU activation (eq 18)
        output = u * F.silu(v)

        # Final projection
        output = self.W_o(output)
        return output

    def normalize_weights(self):
        """Normalize all weight matrices."""
        self.W_u.normalize_weights()
        self.W_v.normalize_weights()
        self.W_o.normalize_weights()


class GPTAttention(nn.Module):
    """Standard multi-head self-attention for baseline GPT.

    Uses RoPE (Rotary Position Embeddings) and 1/√d_k scaling.
    No QK normalization, no weight normalization.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 4096,
        rope_base: int = 10000,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.softmax_scale = 1.0 / math.sqrt(self.d_k)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.rope = RotaryEmbedding(self.d_k, base=rope_base, max_seq_len=max_seq_len)

    def forward(self, h: torch.Tensor, causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = h.shape

        q = self.W_q(h).view(B, T, self.n_heads, self.d_k)
        k = self.W_k(h).view(B, T, self.n_heads, self.d_k)
        v = self.W_v(h).view(B, T, self.n_heads, self.d_k)

        q = self.rope(q)
        k = self.rope(k)

        q = q.transpose(1, 2)  # (B, n_heads, T, d_k)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.softmax_scale

        # causal_mask shape: (1, 1, T, T), broadcasts over batch and heads
        if causal_mask is not None:
            # Expand mask if needed
            if causal_mask.dim() == 2:
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores + causal_mask

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        return self.W_o(attn_output)


class GPTMLP(nn.Module):
    """SwiGLU MLP for baseline GPT (Section 2.4.1).

    Standard implementation with unconstrained weight matrices.
    """

    def __init__(self, d_model: int, d_mlp: int):
        super().__init__()
        self.W_u = nn.Linear(d_model, d_mlp, bias=False)
        self.W_v = nn.Linear(d_model, d_mlp, bias=False)
        self.W_o = nn.Linear(d_mlp, d_model, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        u = self.W_u(h)
        v = self.W_v(h)
        return self.W_o(u * F.silu(v))


class GPTBlock(nn.Module):
    """Standard GPT Transformer block.

    h = h + ATTN(RMSNorm(h))        (eq 4)
    h = h + MLP(RMSNorm(h))         (eq 5)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_mlp: int,
        max_seq_len: int = 4096,
        rope_base: int = 10000,
    ):
        super().__init__()
        self.ln1 = nn.RMSNorm(d_model)
        self.attn = GPTAttention(d_model, n_heads, max_seq_len, rope_base)
        self.ln2 = nn.RMSNorm(d_model)
        self.mlp = GPTMLP(d_model, d_mlp)

    def forward(self, h: torch.Tensor, causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = h + self.attn(self.ln1(h), causal_mask)
        h = h + self.mlp(self.ln2(h))
        return h


class nGPTBlock(nn.Module):
    """Normalized GPT Transformer block.

    h_A = Norm(ATTN(h))
    h = Norm(h + α_A ⊙ (h_A - h))

    h_M = Norm(MLP(h))
    h = Norm(h + α_M ⊙ (h_M - h))

    Where α_A, α_M are eigen learning rates (element-wise).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_mlp: int,
        max_seq_len: int = 4096,
        rope_base: int = 10000,
        use_qk_norm: bool = True,
        alpha_A_init: float = 0.05,
        alpha_A_scale: float = 1.0,
        alpha_M_init: float = 0.05,
        alpha_M_scale: float = 1.0,
        s_qk_init: float = 1.0,
        s_qk_scale: float = 1.0,
        s_u_init: float = 1.0,
        s_u_scale: float = 1.0,
        s_v_init: float = 1.0,
        s_v_scale: float = 1.0,
        use_lerp: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_lerp = use_lerp

        self.attn = Attention(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
            use_qk_norm=use_qk_norm,
            s_qk_init=s_qk_init,
            s_qk_scale=s_qk_scale,
        )
        self.mlp = MLP(
            d_model=d_model,
            d_mlp=d_mlp,
            s_u_init=s_u_init,
            s_u_scale=s_u_scale,
            s_v_init=s_v_init,
            s_v_scale=s_v_scale,
        )

        # Eigen learning rates (per-dimension)
        self.alpha_A = ScaledParameter(d_model, alpha_A_init, alpha_A_scale)
        self.alpha_M = ScaledParameter(d_model, alpha_M_init, alpha_M_scale)

    def _update_hidden_state(
        self, h: torch.Tensor, h_block: torch.Tensor, alpha: torch.Tensor
    ) -> torch.Tensor:
        """Update hidden state using LERP or SLERP.

        LERP: h = Norm(h + α ⊙ (h_block - h))            (eq 7)
        SLERP: h = Norm(sin((1-α)θ)/sin(θ) * h + sin(αθ)/sin(θ) * h_block)  (eq 6)

        Both include normalization as the retraction step (Appendix A.4).
        """
        if self.use_lerp:
            # Simple LERP approximation (eq 10, 11)
            h_new = h + alpha.unsqueeze(0).unsqueeze(0) * (h_block - h)
        else:
            # SLERP (eq 6)
            dot = (h * h_block).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
            theta = torch.acos(dot)
            sin_theta = torch.sin(theta).clamp(min=1e-12)

            a_expanded = alpha.unsqueeze(0).unsqueeze(0).expand_as(h)
            t1 = torch.sin((1.0 - a_expanded) * theta) / sin_theta
            t2 = torch.sin(a_expanded * theta) / sin_theta
            h_new = t1 * h + t2 * h_block

        # Normalize (retraction to hypersphere)
        return F.normalize(h_new, p=2, dim=-1, eps=1e-12)

    def forward(
        self, h: torch.Tensor, causal_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Attention block
        h_A = self.attn(h, causal_mask)
        h_A = F.normalize(h_A, p=2, dim=-1, eps=1e-12)
        alpha_A = torch.abs(self.alpha_A())  # Ensure positive (Section A.2)
        h = self._update_hidden_state(h, h_A, alpha_A)

        # MLP block
        h_M = self.mlp(h)
        h_M = F.normalize(h_M, p=2, dim=-1, eps=1e-12)
        alpha_M = torch.abs(self.alpha_M())  # Ensure positive
        h = self._update_hidden_state(h, h_M, alpha_M)

        return h

    def normalize_weights(self):
        """Normalize all internal weight matrices."""
        self.attn.normalize_weights()
        self.mlp.normalize_weights()
