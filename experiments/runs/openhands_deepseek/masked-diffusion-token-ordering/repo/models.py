import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
from config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, ndim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.dropout = cfg.dropout

        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln_1 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.wte = nn.Embedding(cfg.vocab_size + cfg.num_mask_tokens, cfg.n_embd)
        if cfg.use_learned_pos_emb:
            self.wpe = nn.Embedding(cfg.max_seq_len, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.h = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.n_embd, bias=cfg.bias)

    def forward(
        self,
        idx: torch.Tensor,
        causal: bool = False,
    ) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len

        tok_emb = self.wte(idx)
        if self.cfg.use_learned_pos_emb:
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
            pos_emb = self.wpe(pos)
        else:
            pos_emb = 0

        x = self.drop(tok_emb + pos_emb)

        attn_mask = None
        if causal:
            attn_mask = torch.triu(
                torch.ones(T, T, device=idx.device, dtype=torch.bool), diagonal=1
            )
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

        for block in self.h:
            x = block(x, attn_mask)

        x = self.ln_f(x)
        return x


class MDMHead(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size + cfg.num_mask_tokens, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden)


class MaskedDiffusionModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.transformer = Transformer(cfg)
        self.head = MDMHead(cfg)
        self.lm_head = self.head.lm_head
        # tie weights between embedding and head
        self.lm_head.weight = self.transformer.wte.weight

    def forward(
        self,
        x_t: torch.Tensor,
        causal: bool = False,
    ) -> torch.Tensor:
        hidden = self.transformer(x_t, causal=causal)
        logits = self.head(hidden)
        return logits


def get_num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
