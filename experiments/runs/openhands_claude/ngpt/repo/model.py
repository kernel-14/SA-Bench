"""
Full GPT and nGPT language models.

GPT  — baseline Transformer decoder with RMSNorm, SwiGLU, RoPE.
nGPT — normalized Transformer with all vectors on the unit hypersphere.

The critical post-optimizer-step operation for nGPT is normalize_weights(),
which projects every weight matrix back onto the hypersphere after each
gradient update (Section 2.6, step 2).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from config import ModelConfig
from layers import RMSNorm, l2_norm, normalize_matrix_rows, build_rope_cache
from modules import GPTLayer, NGPTLayer, ScaledParameter


# ---------------------------------------------------------------------------
# Baseline GPT
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    """Decoder-only Transformer (baseline).

    Architecture follows Section 2.2.1 / 2.3.1 / 2.4.1:
      - Pre-norm with RMSNorm
      - SwiGLU MLP
      - RoPE positional embeddings
      - Tied input/output embeddings are NOT used (separate E_input, E_output)
      - Final RMSNorm before logit projection
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.E_input = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(
            [GPTLayer(cfg.d_model, cfg.n_heads, cfg.d_mlp) for _ in range(cfg.n_layers)]
        )
        self.final_norm = RMSNorm(cfg.d_model)
        self.E_output = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        std = 0.02
        output_std = std / math.sqrt(2 * self.cfg.n_layers)

        nn.init.normal_(self.E_input.weight, mean=0.0, std=std)
        nn.init.normal_(self.E_output.weight, mean=0.0, std=std)

        for layer in self.layers:
            attn = layer.attn
            mlp = layer.mlp
            for proj in [attn.Wq, attn.Wk, attn.Wv]:
                nn.init.normal_(proj.weight, mean=0.0, std=std)
            nn.init.normal_(attn.Wo.weight, mean=0.0, std=output_std)
            for proj in [mlp.Wu, mlp.Wv]:
                nn.init.normal_(proj.weight, mean=0.0, std=std)
            nn.init.normal_(mlp.Wo.weight, mean=0.0, std=output_std)
            nn.init.ones_(attn.norm.weight)
            nn.init.ones_(mlp.norm.weight)

    def _build_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((T, T), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)   # (1, 1, T, T)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_ids.shape
        device = input_ids.device

        cos, sin = build_rope_cache(T, self.cfg.d_head, self.cfg.rope_base, device, dtype=torch.float32)
        mask = self._build_causal_mask(T, device)

        h = self.E_input(input_ids)

        for layer in self.layers:
            h = layer(h, cos, sin, mask)

        h = self.final_norm(h)
        logits = self.E_output(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.cfg.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids


# ---------------------------------------------------------------------------
# nGPT
# ---------------------------------------------------------------------------

class NGPT(nn.Module):
    """Normalized Transformer (nGPT).

    All embedding vectors and weight-matrix rows reside on the unit
    hypersphere.  The hidden state h travels on the hypersphere through
    LERP updates controlled by per-dimension eigen learning rates.

    Key differences from GPT (Section 2.6 recipe):
      1. No RMSNorm / LayerNorm anywhere.
      2. All weight matrices normalized after every optimizer step.
      3. LERP update: h <- Norm(h + alpha*(h_block - h)).
      4. QK normalization + sqrt(d_k) softmax scale.
      5. MLP intermediate scaling s_u, s_v*sqrt(d_model).
      6. Logit scaling s_z (per-vocabulary).
      7. No weight decay, no LR warmup.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.E_input = nn.Embedding(cfg.vocab_size, cfg.d_model)

        self.layers = nn.ModuleList([
            NGPTLayer(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                d_mlp=cfg.d_mlp,
                alpha_init=cfg.alpha_init,
                alpha_scale=cfg.alpha_scale,
                sqk_init=cfg.sqk_init,
                sqk_scale=cfg.sqk_scale,
                su_init=cfg.su_init,
                su_scale=cfg.su_scale,
                sv_init=cfg.sv_init,
                sv_scale=cfg.sv_scale,
            )
            for _ in range(cfg.n_layers)
        ])

        # Separate output embedding (not tied to E_input)
        self.E_output = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Logit scaling s_z ∈ R^V (Section 2.1, eq. 3)
        # s_z_init=1, s_z_scale=1/sqrt(d_model)
        self.sz = ScaledParameter((cfg.vocab_size,), cfg.sz_init, cfg.sz_scale)

        self._init_weights()

    def _init_weights(self):
        # Initialize from N(0, 1/sqrt(d_model)); output matrices scaled by
        # 1/sqrt(2*n_layers) as suggested by Radford et al. (2018).
        # Normalization is applied afterwards so exact init values matter less.
        std = 1.0 / math.sqrt(self.cfg.d_model)
        output_std = std / math.sqrt(2 * self.cfg.n_layers)

        nn.init.normal_(self.E_input.weight, mean=0.0, std=std)
        nn.init.normal_(self.E_output.weight, mean=0.0, std=std)

        for layer in self.layers:
            attn = layer.attn
            mlp = layer.mlp
            for proj in [attn.Wq, attn.Wk, attn.Wv]:
                nn.init.normal_(proj.weight, mean=0.0, std=std)
            nn.init.normal_(attn.Wo.weight, mean=0.0, std=output_std)
            for proj in [mlp.Wu, mlp.Wv]:
                nn.init.normal_(proj.weight, mean=0.0, std=std)
            nn.init.normal_(mlp.Wo.weight, mean=0.0, std=output_std)

        # Normalize all weight matrices immediately after init
        self.normalize_weights()

    def _build_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((T, T), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_ids.shape
        device = input_ids.device

        cos, sin = build_rope_cache(T, self.cfg.d_head, self.cfg.rope_base, device, dtype=torch.float32)
        mask = self._build_causal_mask(T, device)

        # Embed and normalize to unit sphere
        h = self.E_input(input_ids)
        h = l2_norm(h, dim=-1)

        for layer in self.layers:
            h = layer(h, cos, sin, mask)

        # Logits: dot products bounded in [-1, 1], scaled by s_z (eq. 1, 3)
        logits = h @ self.E_output.weight.t()
        logits = logits * self.sz()

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def normalize_weights(self):
        """Project all weight matrices back onto the unit hypersphere.

        Called after every optimizer step (Section 2.6, step 2).
        Normalizes each row of every weight matrix along the embedding
        (input) dimension so that dot products with unit-norm hidden states
        are cosine similarities bounded in [-1, 1].
        """
        # Input / output embeddings: rows are d_model-dimensional
        self.E_input.weight.data = normalize_matrix_rows(self.E_input.weight.data)
        self.E_output.weight.data = normalize_matrix_rows(self.E_output.weight.data)

        for layer in self.layers:
            attn = layer.attn
            mlp = layer.mlp

            # Attention projection matrices
            attn.Wq.weight.data = normalize_matrix_rows(attn.Wq.weight.data)
            attn.Wk.weight.data = normalize_matrix_rows(attn.Wk.weight.data)
            attn.Wv.weight.data = normalize_matrix_rows(attn.Wv.weight.data)
            attn.Wo.weight.data = normalize_matrix_rows(attn.Wo.weight.data)

            # MLP projection matrices
            mlp.Wu.weight.data = normalize_matrix_rows(mlp.Wu.weight.data)
            mlp.Wv.weight.data = normalize_matrix_rows(mlp.Wv.weight.data)
            mlp.Wo.weight.data = normalize_matrix_rows(mlp.Wo.weight.data)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.cfg.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(cfg: ModelConfig) -> nn.Module:
    if cfg.model_type == "ngpt":
        return NGPT(cfg)
    elif cfg.model_type == "gpt":
        return GPT(cfg)
    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
