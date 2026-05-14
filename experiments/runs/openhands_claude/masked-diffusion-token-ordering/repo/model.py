"""
Transformer backbone shared by both MDM (bidirectional) and ARM (causal).

MDM uses full (non-causal) attention; ARM uses causal (masked) attention.
Both support learnable positional embeddings (used for π-learner experiments)
and RoPE (used for standard MDM/ARM experiments).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Rotary Positional Embedding (RoPE)
# ---------------------------------------------------------------------------

def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor,
                     freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:xq_.shape[1]].unsqueeze(0).unsqueeze(2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ---------------------------------------------------------------------------
# Multi-Head Attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 causal: bool = False, use_rope: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal
        self.use_rope = use_rope

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                freqs_cis: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)

        q = q.view(B, T, self.n_heads, self.d_head)
        k = k.view(B, T, self.n_heads, self.d_head)
        v = v.view(B, T, self.n_heads, self.d_head)

        if self.use_rope and freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis)

        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = math.sqrt(self.d_head)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale

        if self.causal:
            causal_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
            attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        if attn_mask is not None:
            attn = attn + attn_mask

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(out))


# ---------------------------------------------------------------------------
# Feed-Forward Network
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1, causal: bool = False,
                 use_rope: bool = True):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, causal, use_rope)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                freqs_cis: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), freqs_cis=freqs_cis, attn_mask=attn_mask)
        x = x + self.ff(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Transformer (shared backbone)
# ---------------------------------------------------------------------------

class Transformer(nn.Module):
    """
    Shared transformer backbone for both MDM (bidirectional) and ARM (causal).

    Args:
        vocab_size:         vocabulary size (including mask token 0)
        seq_len:            maximum sequence length
        d_model:            hidden dimension
        n_heads:            number of attention heads
        n_layers:           number of transformer blocks
        d_ff:               feed-forward hidden dimension
        dropout:            dropout probability
        causal:             if True, use causal (autoregressive) attention
        use_rope:           if True, use RoPE; otherwise use learnable pos embeddings
        tie_weights:        if True, tie input embedding and output projection weights
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float = 0.1,
        causal: bool = False,
        use_rope: bool = True,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.causal = causal
        self.use_rope = use_rope

        self.token_emb = nn.Embedding(vocab_size, d_model)

        if use_rope:
            self.pos_emb = None
            self.register_buffer(
                "freqs_cis",
                precompute_freqs_cis(d_model // n_heads, seq_len * 2),
                persistent=False,
            )
        else:
            # Learnable positional embeddings (used for π-learner experiments)
            self.pos_emb = nn.Embedding(seq_len, d_model)
            self.freqs_cis = None

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, causal, use_rope)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:          (B, T) token indices
            attn_mask:  optional additive attention mask (B, 1, T, T) or (T, T)

        Returns:
            logits:     (B, T, vocab_size)
        """
        B, T = x.shape
        h = self.token_emb(x)

        if self.use_rope:
            freqs_cis = self.freqs_cis[:T]
        else:
            positions = torch.arange(T, device=x.device)
            h = h + self.pos_emb(positions)
            freqs_cis = None

        for block in self.blocks:
            h = block(h, freqs_cis=freqs_cis, attn_mask=attn_mask)

        h = self.ln_f(h)
        return self.head(h)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def build_mdm_transformer(
    vocab_size: int,
    seq_len: int,
    model_config: dict,
    use_rope: bool = True,
) -> Transformer:
    """Bidirectional transformer for MDM (no causal mask)."""
    return Transformer(
        vocab_size=vocab_size,
        seq_len=seq_len,
        causal=False,
        use_rope=use_rope,
        **model_config,
    )


def build_arm_transformer(
    vocab_size: int,
    seq_len: int,
    model_config: dict,
    use_rope: bool = True,
) -> Transformer:
    """Causal transformer for ARM."""
    return Transformer(
        vocab_size=vocab_size,
        seq_len=seq_len,
        causal=True,
        use_rope=use_rope,
        **model_config,
    )
