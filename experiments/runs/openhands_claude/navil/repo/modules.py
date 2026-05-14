"""
High-level modules for NaViL:
  - VisualEncoderLayer  : single transformer block with bidirectional attention + 2D-RoPE
  - VisualEncoder       : stack of VisualEncoderLayers + PatchEmbedding
  - LLMLayerWithMoE     : single transformer block with modality-specific MoE
  - MoEExtendedLLM      : stack of LLMLayerWithMoE + embedding + LM head
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LLMConfig, MoEConfig, VisualEncoderConfig
from layers import (
    BidirectionalAttention,
    ModalityAttentionMoE,
    ModalityFFNMoE,
    PatchEmbedding,
    RMSNorm,
    SwiGLUFFN,
    precompute_freqs_1d,
    precompute_freqs_2d,
)


# ── Visual Encoder Layer ───────────────────────────────────────────────────────

class VisualEncoderLayer(nn.Module):
    """
    Single transformer block for the visual encoder.
    Uses bidirectional attention (no causal mask) and 2D-RoPE.
    Architecture mirrors LLM layers but with full attention.
    """

    def __init__(self, cfg: VisualEncoderConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.width)
        self.attn  = BidirectionalAttention(
            hidden_dim=cfg.width,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
        )
        self.norm2 = RMSNorm(cfg.width)
        self.ffn   = SwiGLUFFN(cfg.width, cfg.mlp_width)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin, attention_mask)
        x = x + self.ffn(self.norm2(x))
        return x


# ── Visual Encoder ─────────────────────────────────────────────────────────────

class VisualEncoder(nn.Module):
    """
    V_{d,w}(I) = C ∘ F_d^w ∘ ... ∘ F_1^w ∘ P(I)

    where P is PatchEmbedding, F_i are VisualEncoderLayers with bidirectional
    attention and 2D-RoPE, and C is the PixelShuffleConnector (defined in model.py).

    When depth=0, degenerates to a simple patch embedding.
    """

    def __init__(self, cfg: VisualEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbedding(
            in_channels=3,
            patch_size=cfg.patch_size,
            hidden_dim=cfg.width,
        )
        self.layers = nn.ModuleList([
            VisualEncoderLayer(cfg) for _ in range(cfg.depth)
        ])
        self.norm = RMSNorm(cfg.width)

        # Precompute 2D-RoPE for a large enough grid; resized dynamically if needed
        max_grid = cfg.image_size // cfg.patch_size
        cos, sin = precompute_freqs_2d(
            head_dim=cfg.width // cfg.num_heads,
            max_h=max_grid,
            max_w=max_grid,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _get_2d_rope(self, num_h: int, num_w: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return 2D-RoPE for a given spatial grid, recomputing if necessary."""
        max_grid = int(self.rope_cos.shape[0] ** 0.5)
        if num_h <= max_grid and num_w <= max_grid:
            # Slice the precomputed grid
            head_dim = self.cfg.width // self.cfg.num_heads
            cos, sin = precompute_freqs_2d(
                head_dim=head_dim,
                max_h=num_h,
                max_w=num_w,
                device=self.rope_cos.device,
            )
            return cos, sin
        # Recompute for larger grids
        head_dim = self.cfg.width // self.cfg.num_heads
        return precompute_freqs_2d(
            head_dim=head_dim,
            max_h=num_h,
            max_w=num_w,
            device=self.rope_cos.device,
        )

    def forward(
        self,
        images: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int, int]:
        """
        images: (B, 3, H, W)  — H and W must be multiples of patch_size
        Returns: visual_tokens (B, num_h * num_w, width), num_h, num_w
        """
        x, num_h, num_w = self.patch_embed(images)   # (B, N, width)

        cos, sin = self._get_2d_rope(num_h, num_w)
        cos = cos.to(x.device, x.dtype)
        sin = sin.to(x.device, x.dtype)

        for layer in self.layers:
            x = layer(x, cos, sin, attention_mask)

        x = self.norm(x)
        return x, num_h, num_w


# ── LLM Layer with Modality-specific MoE ──────────────────────────────────────

class LLMLayerWithMoE(nn.Module):
    """
    Single transformer block for the MoE-extended LLM.

    Implements Eq. (2) from the paper:
        x'_{i,m} = x^{l-1}_{i,m} + MHA-MMoE(RMSNorm(x^{l-1}_{i,m}))
        x^l_{i,m} = x'_{i,m}     + FFN-MMoE(RMSNorm(x'_{i,m}))

    Both attention and FFN use modality-specific experts.
    """

    def __init__(self, llm_cfg: LLMConfig, moe_cfg: MoEConfig):
        super().__init__()
        self.norm1 = RMSNorm(llm_cfg.width, eps=llm_cfg.rms_norm_eps)
        self.attn  = ModalityAttentionMoE(
            hidden_dim=llm_cfg.width,
            num_heads=llm_cfg.num_heads,
            num_kv_heads=llm_cfg.num_kv_heads,
            num_experts=moe_cfg.num_experts,
            dropout=llm_cfg.dropout,
        )
        self.norm2 = RMSNorm(llm_cfg.width, eps=llm_cfg.rms_norm_eps)
        self.ffn   = ModalityFFNMoE(
            hidden_dim=llm_cfg.width,
            mlp_dim=llm_cfg.mlp_width,
            num_experts=moe_cfg.num_experts,
        )

    def forward(
        self,
        x: torch.Tensor,
        modality_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, present = self.attn(
            self.norm1(x),
            modality_mask=modality_mask,
            cos=cos,
            sin=sin,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x), modality_mask)
        return x, present


# ── MoE-Extended LLM ──────────────────────────────────────────────────────────

class MoEExtendedLLM(nn.Module):
    """
    Full LLM with modality-specific MoE in every transformer layer.

    Supports:
    - Causal language modeling (next-token prediction)
    - Mixed visual + linguistic token sequences
    - KV-cache for inference
    - Initialization from a pre-trained LLM checkpoint (Observation 1)
    """

    def __init__(self, llm_cfg: LLMConfig, moe_cfg: MoEConfig):
        super().__init__()
        self.cfg = llm_cfg
        self.moe_cfg = moe_cfg

        self.embed_tokens = nn.Embedding(llm_cfg.vocab_size, llm_cfg.width)
        self.layers = nn.ModuleList([
            LLMLayerWithMoE(llm_cfg, moe_cfg) for _ in range(llm_cfg.depth)
        ])
        self.norm = RMSNorm(llm_cfg.width, eps=llm_cfg.rms_norm_eps)
        self.lm_head = nn.Linear(llm_cfg.width, llm_cfg.vocab_size, bias=False)

        if not llm_cfg.tie_word_embeddings:
            # Independent lm_head weights
            pass
        else:
            self.lm_head.weight = self.embed_tokens.weight

        # Precompute 1D-RoPE
        cos, sin = precompute_freqs_1d(
            head_dim=llm_cfg.width // llm_cfg.num_heads,
            max_seq_len=llm_cfg.max_seq_len,
            theta=llm_cfg.rope_theta,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        modality_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        return_logits: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        input_ids:      (B, T) — used when inputs_embeds is None
        inputs_embeds:  (B, T, D) — pre-computed embeddings (used during training
                        when visual tokens are already projected)
        modality_mask:  (B, T) — 0 for visual tokens, 1 for linguistic tokens
        """
        if inputs_embeds is None:
            x = self.embed_tokens(input_ids)
        else:
            x = inputs_embeds

        B, T, D = x.shape

        if modality_mask is None:
            # Default: all linguistic
            modality_mask = torch.ones(B, T, dtype=torch.long, device=x.device)

        if position_ids is None:
            past_len = past_key_values[0][0].shape[2] if past_key_values else 0
            position_ids = torch.arange(past_len, past_len + T, device=x.device).unsqueeze(0)

        cos = self.rope_cos.to(x.dtype)
        sin = self.rope_sin.to(x.dtype)

        # Build causal attention mask if needed
        if attention_mask is not None and attention_mask.dim() == 2:
            # Convert padding mask to additive attention mask
            attn_mask = _make_causal_mask(attention_mask, x.dtype, x.device)
        else:
            attn_mask = attention_mask

        new_past_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
            x, present = layer(
                x,
                modality_mask=modality_mask,
                cos=cos,
                sin=sin,
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_value=past_kv,
            )
            if use_cache:
                new_past_key_values.append(present)

        x = self.norm(x)

        output = {"hidden_states": x}
        if return_logits:
            output["logits"] = self.lm_head(x)
        if use_cache:
            output["past_key_values"] = new_past_key_values

        return output

    def load_pretrained_llm(self, state_dict: Dict[str, torch.Tensor], strict: bool = False):
        """
        Load weights from a pre-trained LLM checkpoint.
        Non-matching keys (MoE visual experts) are initialized randomly.
        """
        missing, unexpected = self.load_state_dict(state_dict, strict=strict)
        return missing, unexpected


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_causal_mask(
    attention_mask: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Convert a 2D padding mask (B, T) to a 4D additive causal attention mask
    (B, 1, T, T) suitable for scaled_dot_product_attention.
    """
    B, T = attention_mask.shape
    causal = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
    pad_mask = attention_mask.bool().unsqueeze(1).unsqueeze(2)   # (B, 1, 1, T)
    combined = causal.unsqueeze(0).unsqueeze(0) & pad_mask       # (B, 1, T, T)
    additive = torch.zeros(B, 1, T, T, device=device, dtype=dtype)
    additive = additive.masked_fill(~combined, torch.finfo(dtype).min)
    return additive
