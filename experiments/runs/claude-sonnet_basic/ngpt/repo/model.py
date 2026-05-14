"""
nGPT: Normalized Transformer with Representation Learning on the Hypersphere
Implementation based on Loshchilov et al. (2024)

Key architectural differences from standard GPT:
1. All weight matrices normalized along embedding dimension after each optimizer step
2. Hidden states travel on the unit hypersphere (normalized after each layer update)
3. LERP update: h = Norm(h + alpha * (h_block - h)) instead of h = h + h_block
4. Learnable eigen learning rates alpha_A, alpha_M (per d_model dimension)
5. Q/K normalized + scaled by learnable s_qk (per head, per d_head dimension)
6. Attention softmax scale is sqrt(d_k) instead of 1/sqrt(d_k)
7. MLP intermediate u scaled by s_u, v scaled by s_v * sqrt(d_model)
8. Logit scaling s_z (per vocab token)
9. No RMSNorm/LayerNorm anywhere; no weight decay; no LR warmup
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class nGPTConfig:
    # ── Architecture ──────────────────────────────────────────────────────────
    vocab_size: int = 32000       # LLaMA-2 tokenizer
    n_layers: int = 24            # 24 for 0.5B, 36 for 1B
    d_model: int = 1024           # 1024 for 0.5B, 1280 for 1B
    n_heads: int = 16             # 16 for 0.5B, 20 for 1B
    d_mlp: Optional[int] = None  # defaults to 4 * d_model
    max_seq_len: int = 4096
    dropout: float = 0.0
    rope_base: float = 10000.0

    # ── nGPT scaling-parameter init scheme (Section 2.5) ─────────────────────
    # Each learnable scalar s is stored as s_param = s_scale, and the actual
    # value used in the forward pass is s_param * (s_init / s_scale).
    # This lets Adam see a parameter of magnitude s_scale while the network
    # uses a value of s_init, decoupling the effective LR from the value.

    # Eigen learning rates  α_A, α_M  (one per d_model dimension)
    alpha_init: float = 0.05          # ≈ 1/n_layers
    alpha_scale: Optional[float] = None  # defaults to 1/sqrt(d_model)

    # QK scaling  s_qk  (one per d_head dimension, shared across heads)
    sqk_init: float = 1.0
    sqk_scale: Optional[float] = None   # defaults to 1/sqrt(d_model)

    # MLP gate scaling  s_u, s_v  (one per d_mlp dimension)
    su_init: float = 1.0
    su_scale: float = 1.0
    sv_init: float = 1.0
    sv_scale: float = 1.0

    # Logit scaling  s_z  (one per vocab token)
    sz_init: float = 1.0
    sz_scale: Optional[float] = None    # defaults to 1/sqrt(d_model)

    # ── Ablation flags ────────────────────────────────────────────────────────
    use_qk_norm: bool = True   # Section 2.3.2 / Appendix A.8

    def __post_init__(self):
        if self.d_mlp is None:
            self.d_mlp = 4 * self.d_model
        if self.alpha_scale is None:
            self.alpha_scale = 1.0 / math.sqrt(self.d_model)
        if self.sqk_scale is None:
            self.sqk_scale = 1.0 / math.sqrt(self.d_model)
        if self.sz_scale is None:
            self.sz_scale = 1.0 / math.sqrt(self.d_model)


# ── Helpers ───────────────────────────────────────────────────────────────────

def l2_norm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Unit-norm normalise along *dim*."""
    return F.normalize(x, p=2, dim=dim)


def build_rope_cache(
    seq_len: int,
    d_head: int,
    base: float = 10000.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute RoPE cos/sin tables."""
    half = d_head // 2
    theta = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=dtype) / half))
    pos = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.outer(pos, theta)          # (T, d_head/2)
    return freqs.cos(), freqs.sin()


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply RoPE to x of shape (B, T, n_heads, d_head).
    cos/sin have shape (T, d_head/2).
    """
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    # broadcast over batch and head dims
    cos = cos.unsqueeze(0).unsqueeze(2)   # (1, T, 1, d/2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# ── nGPT Attention ────────────────────────────────────────────────────────────

class nGPTAttention(nn.Module):
    """
    Multi-head self-attention for nGPT.

    Changes vs. standard attention (Section 2.3.2):
    • W_q, W_k, W_v, W_o normalised along embedding dim after each step
    • q, k normalised per-head then scaled by learnable s_qk
    • Softmax scale = sqrt(d_k)  (not 1/sqrt(d_k))
    """

    def __init__(self, config: nGPTConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.d_head  = config.d_model // config.n_heads
        self.use_qk_norm = config.use_qk_norm

        self.W_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_o = nn.Linear(config.d_model, config.d_model, bias=False)

        # s_qk: shape (d_head,) – shared across heads (one vector per layer)
        # Stored at sqk_scale; forward multiplies by sqk_init/sqk_scale
        sqk_scale = config.sqk_scale
        self.register_buffer('sqk_ratio',
                             torch.tensor(config.sqk_init / sqk_scale))
        self.sqk = nn.Parameter(torch.full((self.d_head,), sqk_scale))

        self.attn_drop = nn.Dropout(config.dropout)

    def forward(
        self,
        h: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = h.shape

        q = self.W_q(h).view(B, T, self.n_heads, self.d_head)
        k = self.W_k(h).view(B, T, self.n_heads, self.d_head)
        v = self.W_v(h).view(B, T, self.n_heads, self.d_head)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # QK normalisation (optional, Section 2.3.2 / Appendix A.8)
        if self.use_qk_norm:
            q = l2_norm(q, dim=-1)
            k = l2_norm(k, dim=-1)

        # Scale by s_qk  (actual value = stored * ratio)
        sqk = self.sqk * self.sqk_ratio          # (d_head,)
        q = q * sqk
        k = k * sqk

        # (B, n_heads, T, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention with scale = sqrt(d_k)  (Section 2.3.2)
        scale = math.sqrt(self.d_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale   # (B, H, T, T)

        if mask is not None:
            scores = scores + mask

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)                              # (B, H, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)


# ── nGPT MLP ──────────────────────────────────────────────────────────────────

class nGPTMLP(nn.Module):
    """
    SwiGLU MLP for nGPT.

    Changes vs. standard MLP (Section 2.4.2):
    • W_u, W_v, W_o normalised along embedding dim after each step
    • u  scaled by learnable s_u
    • v  scaled by learnable s_v * sqrt(d_model)  (restores SiLU non-linearity)
    """

    def __init__(self, config: nGPTConfig):
        super().__init__()
        self.sqrt_d = math.sqrt(config.d_model)

        self.W_u = nn.Linear(config.d_model, config.d_mlp, bias=False)
        self.W_v = nn.Linear(config.d_model, config.d_mlp, bias=False)
        self.W_o = nn.Linear(config.d_mlp,   config.d_model, bias=False)

        # s_u: stored at su_scale, actual = param * (su_init / su_scale)
        self.register_buffer('su_ratio',
                             torch.tensor(config.su_init / config.su_scale))
        self.su = nn.Parameter(torch.full((config.d_mlp,), config.su_scale))

        # s_v: stored at sv_scale, actual = param * (sv_init / sv_scale)
        self.register_buffer('sv_ratio',
                             torch.tensor(config.sv_init / config.sv_scale))
        self.sv = nn.Parameter(torch.full((config.d_mlp,), config.sv_scale))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        u = self.W_u(h) * (self.su * self.su_ratio)
        v = self.W_v(h) * (self.sv * self.sv_ratio * self.sqrt_d)
        return self.W_o(u * F.silu(v))


# ── nGPT Layer ────────────────────────────────────────────────────────────────

class nGPTLayer(nn.Module):
    """
    One nGPT transformer layer.

    Update equations (Section 2.2.2, Table 1):
        h_A = Norm(ATTN(h))
        h   = Norm(h + α_A ⊙ (h_A − h))
        h_M = Norm(MLP(h))
        h   = Norm(h + α_M ⊙ (h_M − h))

    α_A, α_M ∈ R^{d_model} are learnable eigen learning rates.
    """

    def __init__(self, config: nGPTConfig):
        super().__init__()
        self.attn = nGPTAttention(config)
        self.mlp  = nGPTMLP(config)

        # α stored at alpha_scale; actual = |param * ratio|  (kept positive, App. A.2)
        alpha_scale = config.alpha_scale
        self.register_buffer('alpha_ratio',
                             torch.tensor(config.alpha_init / alpha_scale))
        self.alpha_A = nn.Parameter(torch.full((config.d_model,), alpha_scale))
        self.alpha_M = nn.Parameter(torch.full((config.d_model,), alpha_scale))

    def forward(
        self,
        h: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # ── Attention sub-layer ───────────────────────────────────────────────
        h_A = l2_norm(self.attn(h, cos, sin, mask))
        alpha_A = torch.abs(self.alpha_A * self.alpha_ratio)
        h = l2_norm(h + alpha_A * (h_A - h))

        # ── MLP sub-layer ─────────────────────────────────────────────────────
        h_M = l2_norm(self.mlp(h))
        alpha_M = torch.abs(self.alpha_M * self.alpha_ratio)
        h = l2_norm(h + alpha_M * (h_M - h))

        return h


# ── Full nGPT Model ───────────────────────────────────────────────────────────

class nGPT(nn.Module):
    """
    Normalized Transformer (nGPT).

    All vectors (embeddings, weight rows/columns, hidden states) live on the
    unit hypersphere.  After every optimizer step call model.normalize_weights()
    to project matrices back onto the manifold.
    """

    def __init__(self, config: nGPTConfig):
        super().__init__()
        self.config = config

        # Separate input and output embeddings (not tied)
        self.E_input  = nn.Embedding(config.vocab_size, config.d_model)
        self.E_output = nn.Embedding(config.vocab_size, config.d_model)

        self.layers  = nn.ModuleList([nGPTLayer(config) for _ in range(config.n_layers)])
        self.drop    = nn.Dropout(config.dropout)

        # Logit scaling s_z (Section 2.1, eq. 3)
        sz_scale = config.sz_scale
        self.register_buffer('sz_ratio',
                             torch.tensor(config.sz_init / sz_scale))
        self.sz = nn.Parameter(torch.full((config.vocab_size,), sz_scale))

        self._init_weights()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_weights(self):
        """
        Initialise with N(0, 1/sqrt(d_model)) then immediately normalise.
        The paper notes that the exact initialisation doesn't matter for nGPT
        because weights are normalised afterwards (Appendix A.6).
        """
        std = 1.0 / math.sqrt(self.config.d_model)
        # Output projection matrices scaled by 1/sqrt(2*n_layers) (GPT-2 recipe)
        out_std = std / math.sqrt(2.0 * self.config.n_layers)

        nn.init.normal_(self.E_input.weight,  std=std)
        nn.init.normal_(self.E_output.weight, std=std)

        for layer in self.layers:
            nn.init.normal_(layer.attn.W_q.weight, std=std)
            nn.init.normal_(layer.attn.W_k.weight, std=std)
            nn.init.normal_(layer.attn.W_v.weight, std=std)
            nn.init.normal_(layer.attn.W_o.weight, std=out_std)
            nn.init.normal_(layer.mlp.W_u.weight,  std=std)
            nn.init.normal_(layer.mlp.W_v.weight,  std=std)
            nn.init.normal_(layer.mlp.W_o.weight,  std=out_std)

        self.normalize_weights()

    # ── Weight normalisation ──────────────────────────────────────────────────

    @torch.no_grad()
    def normalize_weights(self):
        """
        Project all weight matrices back onto the unit hypersphere.

        Called after every optimizer.step() during training (Section 2.6, step 2).

        Convention: for nn.Linear(in, out) the weight tensor has shape
        (out_features, in_features).  The "embedding dimension" the paper
        refers to is the *input* dimension, so we normalise along dim=1
        (each row becomes a unit vector in R^{in_features}).

        For embedding matrices (vocab × d_model) we normalise along dim=-1
        (each token embedding becomes a unit vector in R^{d_model}).
        """
        self.E_input.weight.data  = l2_norm(self.E_input.weight.data,  dim=-1)
        self.E_output.weight.data = l2_norm(self.E_output.weight.data, dim=-1)

        for layer in self.layers:
            a = layer.attn
            a.W_q.weight.data = l2_norm(a.W_q.weight.data, dim=1)
            a.W_k.weight.data = l2_norm(a.W_k.weight.data, dim=1)
            a.W_v.weight.data = l2_norm(a.W_v.weight.data, dim=1)
            a.W_o.weight.data = l2_norm(a.W_o.weight.data, dim=1)

            m = layer.mlp
            m.W_u.weight.data = l2_norm(m.W_u.weight.data, dim=1)
            m.W_v.weight.data = l2_norm(m.W_v.weight.data, dim=1)
            m.W_o.weight.data = l2_norm(m.W_o.weight.data, dim=1)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_ids.shape
        device = input_ids.device

        h = self.drop(self.E_input(input_ids))   # (B, T, d_model)

        cos, sin = build_rope_cache(
            T, self.config.d_model // self.config.n_heads,
            base=self.config.rope_base, device=device, dtype=h.dtype,
        )

        # Causal mask: upper-triangular = -inf
        mask = torch.full((T, T), float('-inf'), device=device, dtype=h.dtype)
        mask = torch.triu(mask, diagonal=1)

        for layer in self.layers:
            h = layer(h, cos, sin, mask)

        # Logits = E_output · h  (dot products bounded in [-1,1] since both normalised)
        logits = h @ self.E_output.weight.T          # (B, T, vocab_size)
        logits = logits * (self.sz * self.sz_ratio)  # scale by s_z  (eq. 3)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    # ── Generation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            ctx = input_ids[:, -self.config.max_seq_len:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_tok], dim=1)
        return input_ids


# ── Baseline GPT (for comparison) ────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt() * self.weight


class GPTAttention(nn.Module):
    def __init__(self, config: nGPTConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head  = config.d_model // config.n_heads
        self.W_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_o = nn.Linear(config.d_model, config.d_model, bias=False)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, h, cos, sin, mask=None):
        B, T, D = h.shape
        q = self.W_q(h).view(B, T, self.n_heads, self.d_head)
        k = self.W_k(h).view(B, T, self.n_heads, self.d_head)
        v = self.W_v(h).view(B, T, self.n_heads, self.d_head)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
        scale = 1.0 / math.sqrt(self.d_head)
        scores = torch.matmul(q, k.transpose(-2,-1)) * scale
        if mask is not None:
            scores = scores + mask
        attn = self.drop(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1,2).contiguous().view(B, T, D)
        return self.W_o(out)


class GPTMLP(nn.Module):
    def __init__(self, config: nGPTConfig):
        super().__init__()
        self.W_u = nn.Linear(config.d_model, config.d_mlp, bias=False)
        self.W_v = nn.Linear(config.d_model, config.d_mlp, bias=False)
        self.W_o = nn.Linear(config.d_mlp,   config.d_model, bias=False)

    def forward(self, h):
        return self.W_o(self.W_u(h) * F.silu(self.W_v(h)))


class GPTLayer(nn.Module):
    def __init__(self, config: nGPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn  = GPTAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.mlp   = GPTMLP(config)

    def forward(self, h, cos, sin, mask=None):
        h = h + self.attn(self.norm1(h), cos, sin, mask)
        h = h + self.mlp(self.norm2(h))
        return h


class GPT(nn.Module):
    """
    Baseline GPT with pre-norm (RMSNorm), SwiGLU MLP, RoPE.
    Uses separate (untied) input and output embeddings to match the
    parameter count reported in Table 2 of the paper (~468.2M for 0.5B).
    """

    def __init__(self, config: nGPTConfig):
        super().__init__()
        self.config = config
        # Separate input and output embeddings (untied) to match paper param count
        self.E_input    = nn.Embedding(config.vocab_size, config.d_model)
        self.E_output   = nn.Embedding(config.vocab_size, config.d_model)
        self.layers     = nn.ModuleList([GPTLayer(config) for _ in range(config.n_layers)])
        self.norm_final = RMSNorm(config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self._init_weights()

    def _init_weights(self):
        std     = 0.02
        out_std = std / math.sqrt(2.0 * self.config.n_layers)
        nn.init.normal_(self.E_input.weight,  std=std)
        nn.init.normal_(self.E_output.weight, std=std)
        for layer in self.layers:
            for W in [layer.attn.W_q, layer.attn.W_k, layer.attn.W_v,
                      layer.mlp.W_u, layer.mlp.W_v]:
                nn.init.normal_(W.weight, std=std)
            for W in [layer.attn.W_o, layer.mlp.W_o]:
                nn.init.normal_(W.weight, std=out_std)

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        device = input_ids.device
        h = self.drop(self.E_input(input_ids))
        cos, sin = build_rope_cache(
            T, self.config.d_model // self.config.n_heads,
            base=self.config.rope_base, device=device, dtype=h.dtype,
        )
        mask = torch.full((T, T), float('-inf'), device=device, dtype=h.dtype)
        mask = torch.triu(mask, diagonal=1)
        for layer in self.layers:
            h = layer(h, cos, sin, mask)
        h = self.norm_final(h)
        logits = h @ self.E_output.weight.T
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            ctx = input_ids[:, -self.config.max_seq_len:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_tok], dim=1)
        return input_ids


# ── Factory ───────────────────────────────────────────────────────────────────

# Paper model sizes (Table 2, Appendix A.6)
MODEL_CONFIGS = {
    '0.5B': dict(n_layers=24, d_model=1024, n_heads=16),
    '1B':   dict(n_layers=36, d_model=1280, n_heads=20),
}


def create_model(
    model_type: str = 'ngpt',
    size: str = '0.5B',
    **kwargs,
) -> nn.Module:
    """
    Convenience factory.

    Args:
        model_type: 'ngpt' or 'gpt'
        size:       '0.5B' or '1B'
        **kwargs:   override any nGPTConfig field
    """
    if size not in MODEL_CONFIGS:
        raise ValueError(f"size must be one of {list(MODEL_CONFIGS)}")
    cfg = {**MODEL_CONFIGS[size], **kwargs}
    config = nGPTConfig(**cfg)
    if model_type == 'ngpt':
        return nGPT(config)
    elif model_type == 'gpt':
        return GPT(config)
    raise ValueError(f"model_type must be 'ngpt' or 'gpt', got '{model_type}'")
