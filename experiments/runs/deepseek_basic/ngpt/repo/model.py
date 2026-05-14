"""
nGPT: Normalized Transformer with Representation Learning on the Hypersphere

Core implementation of the normalized Transformer as described in the paper.
All vectors forming embeddings, MLP, attention matrices and hidden states are
unit norm normalized. The input stream of tokens travels on the surface of a
hypersphere, with each layer contributing a displacement towards the target
output predictions.

Based on: "nGPT: Normalized Transformer with Representation Learning on the Hypersphere"
by Loshchilov et al. (2024)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def norm(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize any vector x to have unit norm.
    Unlike RMSNorm or LayerNorm, does not introduce element-wise scaling factors.
    """
    return x / (x.norm(dim=-1, keepdim=True) + 1e-12)


class NormalizedLinear(nn.Module):
    """
    Linear layer where weight vectors are normalized along the embedding dimension.
    This ensures dot products represent cosine similarities bounded in [-1, 1].
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize with small values; normalization after each step handles the scale
        nn.init.normal_(self.weight, mean=0.0, std=1.0 / math.sqrt(self.in_features))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize weight vectors along embedding dimension
        weight_normed = norm(self.weight)
        return F.linear(x, weight_normed, self.bias)


class NormalizedEmbedding(nn.Module):
    """
    Embedding layer with normalized vectors along the embedding dimension.
    After each training step, embeddings are normalized to lie on unit hypersphere.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.0, std=1.0 / math.sqrt(self.embedding_dim))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # Normalize embedding vectors
        weight_normed = norm(self.weight)
        return F.embedding(idx, weight_normed)


class ScaledParameter(nn.Module):
    """
    A trainable scaling parameter with init/scale decomposition for controlling
    effective Adam learning rates.

    During initialization, the parameter's actual value is set to s_scale.
    During forward pass, the actual value is restored by multiplying s_init / s_scale.

    This allows controlling the effective learning rate by adjusting s_scale
    while keeping the global learning rate unchanged.
    """

    def __init__(self, shape, s_init: float, s_scale: float):
        super().__init__()
        self.s_init = s_init
        self.s_scale = s_scale
        # Store the raw parameter initialized at s_scale
        self.raw = nn.Parameter(torch.full(shape, s_scale))

    def forward(self) -> torch.Tensor:
        # Restore actual value: raw * (s_init / s_scale)
        return self.raw * (self.s_init / self.s_scale)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) as described in Su et al. (2024).
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Apply RoPE to input tensor x of shape (batch, n_heads, seq_len, d_k)
        """
        bsz, n_heads, seq_len, d_k = x.shape

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)

        # position_ids shape: (batch, seq_len)
        # We need frequencies for each position: use the first batch's positions
        pos = position_ids[0].float()  # (seq_len,)

        # Compute frequencies
        freqs = torch.outer(pos, self.inv_freq)  # (seq_len, d_k//2)
        emb = freqs.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, d_k//2)
        cos = torch.cos(emb)
        sin = torch.sin(emb)

        # Rotate
        x_rot = x.reshape(bsz, n_heads, seq_len, d_k // 2, 2)
        x_cos = x_rot[..., 0] * cos - x_rot[..., 1] * sin
        x_sin = x_rot[..., 0] * sin + x_rot[..., 1] * cos

        return torch.stack([x_cos, x_sin], dim=-1).reshape(bsz, n_heads, seq_len, d_k)


class AttentionBlock(nn.Module):
    """
    Normalized self-attention block for nGPT.

    Key differences from standard transformer:
    - All projection matrices (W_q, W_k, W_v, W_o) are normalized along embedding dim
    - q and k are additionally normalized and scaled by learnable s_qk
    - Softmax scaling factor is sqrt(d_k) instead of 1/sqrt(d_k)
    - No RMSNorm/LayerNorm before attention
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 4096,
        rope_base: float = 10000.0,
        s_qk_init: float = 1.0,
        s_qk_scale: float = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # s_qk_scale defaults to 1/sqrt(d_model)
        if s_qk_scale is None:
            s_qk_scale = 1.0 / math.sqrt(d_model)

        # Normalized projection matrices
        self.W_q = NormalizedLinear(d_model, d_model, bias=False)
        self.W_k = NormalizedLinear(d_model, d_model, bias=False)
        self.W_v = NormalizedLinear(d_model, d_model, bias=False)
        self.W_o = NormalizedLinear(d_model, d_model, bias=False)

        # Learnable scaling factors for q and k (per-head, per-dimension)
        self.s_qk = ScaledParameter((self.d_k,), s_qk_init, s_qk_scale)

        # RoPE
        self.rope = RotaryPositionalEmbedding(self.d_k, max_seq_len, rope_base)

        # Softmax scale: sqrt(d_k) in nGPT (instead of 1/sqrt(d_k))
        self.softmax_scale = math.sqrt(self.d_k)

    def forward(
        self,
        h: torch.Tensor,
        position_ids: torch.Tensor = None,
        causal_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            h: Hidden state of shape (batch, seq_len, d_model), already normalized
        Returns:
            h_A: Attention output, normalized
        """
        bsz, seq_len, _ = h.shape

        # Project to q, k, v using normalized weight matrices
        q = self.W_q(h)  # (batch, seq_len, d_model)
        k = self.W_k(h)
        v = self.W_v(h)

        # Reshape to multi-head: (batch, n_heads, seq_len, d_k)
        q = q.view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Apply RoPE
        q = self.rope(q, position_ids)
        k = self.rope(k, position_ids)

        # Normalize q and k, then scale by s_qk (equations 15 and 16)
        s_qk = self.s_qk()  # (d_k,)
        q = norm(q) * s_qk.view(1, 1, 1, -1)
        k = norm(k) * s_qk.view(1, 1, 1, -1)

        # Compute attention scores
        # q: (batch, n_heads, seq_len, d_k), k: (batch, n_heads, seq_len, d_k)
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # (batch, n_heads, seq_len, seq_len)

        # Scale by sqrt(d_k)
        attn_scores = attn_scores * self.softmax_scale

        # Apply causal mask
        if causal_mask is not None:
            attn_scores = attn_scores + causal_mask

        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Weighted sum of values
        h_A = torch.matmul(attn_weights, v)  # (batch, n_heads, seq_len, d_k)

        # Concatenate heads and project
        h_A = h_A.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        h_A = self.W_o(h_A)

        # Normalize the output (as per Table 1: h_A ← Norm(ATTN(h)))
        h_A = norm(h_A)

        return h_A


class MLPBlock(nn.Module):
    """
    Normalized MLP block for nGPT with SwiGLU activation.

    Key differences from standard transformer:
    - W_u, W_v, W_o_mlp are normalized along embedding dimension
    - u and v are scaled by learnable s_u and s_v
    - v is also scaled by sqrt(d_model) to benefit from SiLU non-linearity
    - No RMSNorm/LayerNorm before MLP
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

        # Normalized projection matrices
        self.W_u = NormalizedLinear(d_model, d_mlp, bias=False)
        self.W_v = NormalizedLinear(d_model, d_mlp, bias=False)
        self.W_o_mlp = NormalizedLinear(d_mlp, d_model, bias=False)

        # Learnable scaling factors for u and v
        self.s_u = ScaledParameter((d_mlp,), s_u_init, s_u_scale)
        self.s_v = ScaledParameter((d_mlp,), s_v_init, s_v_scale)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Hidden state of shape (batch, seq_len, d_model), already normalized
        Returns:
            h_M: MLP output, normalized
        """
        # Project using normalized weight matrices
        u = self.W_u(h)  # (batch, seq_len, d_mlp)
        v = self.W_v(h)  # (batch, seq_len, d_mlp)

        # Scale u and v (equations 20 and 21)
        s_u = self.s_u()  # (d_mlp,)
        s_v = self.s_v()  # (d_mlp,)
        u = u * s_u.view(1, 1, -1)
        # v is scaled by sqrt(d_model) to benefit from SiLU non-linearity
        v = v * s_v.view(1, 1, -1) * math.sqrt(self.d_model)

        # SwiGLU activation: u * SiLU(v)
        # SiLU(v) = v * sigmoid(v)
        h_M = u * F.silu(v)

        # Final linear projection
        h_M = self.W_o_mlp(h_M)

        # Normalize the output (as per Table 1: h_M ← Norm(MLP(h)))
        h_M = norm(h_M)

        return h_M


class nGPTBlock(nn.Module):
    """
    A single nGPT transformer block combining Attention and MLP with
    eigen learning rates and spherical interpolation.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_mlp: int,
        max_seq_len: int = 4096,
        rope_base: float = 10000.0,
        alpha_A_init: float = 0.05,
        alpha_M_init: float = 0.05,
        alpha_scale: float = None,
    ):
        super().__init__()
        if alpha_scale is None:
            alpha_scale = 1.0 / math.sqrt(d_model)

        self.attention = AttentionBlock(d_model, n_heads, max_seq_len, rope_base)
        self.mlp = MLPBlock(d_model, d_mlp)

        # Eigen learning rates (per embedding dimension)
        # alpha_A controls contribution of attention output to hidden state
        self.alpha_A = ScaledParameter((d_model,), alpha_A_init, alpha_scale)
        self.alpha_M = ScaledParameter((d_model,), alpha_M_init, alpha_scale)

    def forward(
        self,
        h: torch.Tensor,
        position_ids: torch.Tensor = None,
        causal_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Update equations (10 and 11 from paper):
        h ← Norm(h + α_A (h_A - h))
        h ← Norm(h + α_M (h_M - h))

        Where h_A = Norm(ATTN(h)) and h_M = Norm(MLP(h))
        """
        # Attention update
        h_A = self.attention(h, position_ids, causal_mask)
        alpha_A = self.alpha_A()  # (d_model,)
        # h ← Norm(h + α_A * (h_A - h))
        h = norm(h + alpha_A.view(1, 1, -1) * (h_A - h))

        # MLP update
        h_M = self.mlp(h)
        alpha_M = self.alpha_M()  # (d_model,)
        # h ← Norm(h + α_M * (h_M - h))
        h = norm(h + alpha_M.view(1, 1, -1) * (h_M - h))

        return h


class nGPT(nn.Module):
    """
    Normalized Transformer (nGPT) - full model.

    All vectors forming embeddings, MLP, attention matrices and hidden states
    are unit norm normalized. The input stream of tokens travels on the surface
    of a hypersphere.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_mlp: int = None,
        max_seq_len: int = 4096,
        rope_base: float = 10000.0,
        s_z_init: float = 1.0,
        s_z_scale: float = None,
        alpha_A_init: float = 0.05,
        alpha_M_init: float = 0.05,
        alpha_scale: float = None,
    ):
        super().__init__()
        if d_mlp is None:
            d_mlp = 4 * d_model
        if s_z_scale is None:
            s_z_scale = 1.0 / math.sqrt(d_model)
        if alpha_scale is None:
            alpha_scale = 1.0 / math.sqrt(d_model)

        self.d_model = d_model
        self.n_layers = n_layers
        self.vocab_size = vocab_size

        # Normalized input and output embeddings
        self.E_input = NormalizedEmbedding(vocab_size, d_model)
        self.E_output = NormalizedEmbedding(vocab_size, d_model)

        # Transformer blocks
        self.layers = nn.ModuleList([
            nGPTBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_mlp=d_mlp,
                max_seq_len=max_seq_len,
                rope_base=rope_base,
                alpha_A_init=alpha_A_init,
                alpha_M_init=alpha_M_init,
                alpha_scale=alpha_scale,
            )
            for _ in range(n_layers)
        ])

        # Learnable scaling for logits (equation 3)
        self.s_z = ScaledParameter((vocab_size,), s_z_init, s_z_scale)

        # Causal mask buffer
        causal_mask = torch.triu(
            torch.full((1, 1, max_seq_len, max_seq_len), float('-inf')),
            diagonal=1,
        )
        self.register_buffer('causal_mask', causal_mask, persistent=False)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor = None,
    ) -> tuple:
        """
        Args:
            idx: Input token indices of shape (batch, seq_len)
            targets: Target token indices of shape (batch, seq_len), optional
        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: scalar cross-entropy loss if targets provided, else None
        """
        bsz, seq_len = idx.shape
        position_ids = torch.arange(seq_len, device=idx.device).unsqueeze(0)

        # Get input embeddings (normalized)
        h = self.E_input(idx)  # (batch, seq_len, d_model)

        # Optionally normalize h explicitly (it's already normalized via E_input)
        # But we do it for safety
        h = norm(h)

        # Get causal mask for this sequence length
        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]

        # Pass through layers
        for layer in self.layers:
            h = layer(h, position_ids, causal_mask)

        # No final normalization needed - h is already normalized by the last layer

        # Compute logits: z_i = E_output @ h_i (equation 1)
        # Normalize output embeddings
        E_out_normed = norm(self.E_output.weight)  # (vocab_size, d_model)
        logits = torch.matmul(h, E_out_normed.T)  # (batch, seq_len, vocab_size)

        # Scale logits by s_z (equation 3)
        s_z = self.s_z()  # (vocab_size,)
        logits = logits * s_z.view(1, 1, -1)

        loss = None
        if targets is not None:
            # Cross-entropy loss
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    def normalize_weights(self):
        """
        Normalize all matrix parameters along their embedding dimension.
        This should be called after each optimizer step (and optionally during forward pass).

        Matrices to normalize:
        - E_input, E_output (embedding matrices)
        - W_q, W_k, W_v, W_o (attention matrices)
        - W_u, W_v, W_o_mlp (MLP matrices)
        """
        with torch.no_grad():
            # Normalize embedding matrices
            self.E_input.weight.data = norm(self.E_input.weight.data)
            self.E_output.weight.data = norm(self.E_output.weight.data)

            # Normalize matrices in each layer
            for layer in self.layers:
                # Attention matrices
                layer.attention.W_q.weight.data = norm(layer.attention.W_q.weight.data)
                layer.attention.W_k.weight.data = norm(layer.attention.W_k.weight.data)
                layer.attention.W_v.weight.data = norm(layer.attention.W_v.weight.data)
                layer.attention.W_o.weight.data = norm(layer.attention.W_o.weight.data)

                # MLP matrices
                layer.mlp.W_u.weight.data = norm(layer.mlp.W_u.weight.data)
                layer.mlp.W_v.weight.data = norm(layer.mlp.W_v.weight.data)
                layer.mlp.W_o_mlp.weight.data = norm(layer.mlp.W_o_mlp.weight.data)

    def configure_optimizers(self, learning_rate: float, betas=(0.9, 0.95), eps=1e-8):
        """
        Configure Adam optimizer (no weight decay, no warmup).
        nGPT uses Adam without weight decay (AdamW with weight_decay=0.0).
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=0.0,  # No weight decay in nGPT
        )
        return optimizer


# Helper function to create model configurations matching the paper
def create_ngpt_model(
    config: str = '0.5B',
    vocab_size: int = 32000,
    max_seq_len: int = 4096,
):
    """
    Create nGPT model with configurations matching the paper.

    Args:
        config: '0.5B' or '1B'
        vocab_size: Vocabulary size (paper uses LLaMA-2 tokenizer with 32k tokens)
        max_seq_len: Maximum sequence length
    """
    configs = {
        '0.5B': {
            'd_model': 1024,
            'n_heads': 16,
            'n_layers': 24,
            'd_mlp': 4096,  # 4 * d_model
        },
        '1B': {
            'd_model': 1280,
            'n_heads': 20,
            'n_layers': 36,
            'd_mlp': 5120,  # 4 * d_model
        },
    }

    cfg = configs[config]

    # alpha_A_init and alpha_M_init: 1/n_layers order of magnitude
    n_layers = cfg['n_layers']
    alpha_init = 1.0 / n_layers  # ~0.042 for 24 layers, ~0.028 for 36 layers

    # Paper uses 0.05 for alpha_init for both models
    # and alpha_scale = 1/sqrt(d_model)

    return nGPT(
        vocab_size=vocab_size,
        d_model=cfg['d_model'],
        n_heads=cfg['n_heads'],
        n_layers=n_layers,
        d_mlp=cfg['d_mlp'],
        max_seq_len=max_seq_len,
        alpha_A_init=0.05,
        alpha_M_init=0.05,
        alpha_scale=1.0 / math.sqrt(cfg['d_model']),
    )


# For comparison: Baseline GPT model
class BaselineGPTBlock(nn.Module):
    """Standard GPT transformer block for comparison."""

    def __init__(self, d_model: int, n_heads: int, d_mlp: int, max_seq_len: int = 4096):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # RMSNorm before attention and MLP
        self.rms_norm_1 = nn.RMSNorm(d_model)
        self.rms_norm_2 = nn.RMSNorm(d_model)

        # Standard (unconstrained) projection matrices
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # MLP with SwiGLU
        self.W_u_mlp = nn.Linear(d_model, d_mlp, bias=False)
        self.W_v_mlp = nn.Linear(d_model, d_mlp, bias=False)
        self.W_o_mlp = nn.Linear(d_mlp, d_model, bias=False)

        # RoPE
        self.rope = RotaryPositionalEmbedding(self.d_k, max_seq_len)

    def forward(self, h, position_ids=None, causal_mask=None):
        bsz, seq_len, _ = h.shape

        # Attention with pre-norm
        h_norm = self.rms_norm_1(h)
        q = self.W_q(h_norm).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(h_norm).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(h_norm).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        q = self.rope(q, position_ids)
        k = self.rope(k, position_ids)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if causal_mask is not None:
            attn_scores = attn_scores + causal_mask
        attn_weights = F.softmax(attn_scores, dim=-1)
        h_A = torch.matmul(attn_weights, v)
        h_A = h_A.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        h_A = self.W_o(h_A)
        h = h + h_A  # Residual connection

        # MLP with pre-norm
        h_norm = self.rms_norm_2(h)
        u = self.W_u_mlp(h_norm)
        v_mlp = self.W_v_mlp(h_norm)
        h_M = u * F.silu(v_mlp)
        h_M = self.W_o_mlp(h_M)
        h = h + h_M  # Residual connection

        return h


class BaselineGPT(nn.Module):
    """Standard GPT model for comparison with nGPT."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_mlp: int = None,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        if d_mlp is None:
            d_mlp = 4 * d_model

        self.d_model = d_model
        self.n_layers = n_layers

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights
        self.token_embedding.weight = self.lm_head.weight

        self.layers = nn.ModuleList([
            BaselineGPTBlock(d_model, n_heads, d_mlp, max_seq_len)
            for _ in range(n_layers)
        ])

        self.rms_norm_final = nn.RMSNorm(d_model)

        causal_mask = torch.triu(
            torch.full((1, 1, max_seq_len, max_seq_len), float('-inf')), diagonal=1
        )
        self.register_buffer('causal_mask', causal_mask, persistent=False)

    def forward(self, idx, targets=None):
        bsz, seq_len = idx.shape
        position_ids = torch.arange(seq_len, device=idx.device).unsqueeze(0)

        h = self.token_embedding(idx)
        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]

        for layer in self.layers:
            h = layer(h, position_ids, causal_mask)

        h = self.rms_norm_final(h)
        logits = self.lm_head(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    def configure_optimizers(self, learning_rate, weight_decay=0.1, betas=(0.9, 0.95)):
        """Standard AdamW optimizer with weight decay for baseline GPT."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
        )
        return optimizer
