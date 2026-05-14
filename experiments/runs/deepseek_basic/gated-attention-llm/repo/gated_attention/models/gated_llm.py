"""
Gated LLM models implementing the architectures from the paper.

Supports:
  - 15A2B MoE model (15B total, 2.54B activated)
  - 1.7B dense model
  - Configurable gating position, granularity, scope, mode, activation
  - Optional sandwich normalization
  - RoPE with configurable base frequency
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn

from ..modules.gating import (
    GatingPosition,
    GatingGranularity,
    GatingMode,
    GatingScope,
    ActivationType,
    GatedAttentionConfig,
)
from ..modules.transformer_block import GatedDecoderBlock, RMSNorm
from ..modules.gating import RotaryEmbedding


@dataclass
class GatedLLMConfig:
    """Configuration for a Gated LLM model.

    This captures model architectures from the paper:
      - 15A2B MoE: d_model=2048, num_layers=48, num_heads=32, num_kv_heads=4,
                    head_dim=128, ffn_type='moe', moe_num_experts=128, moe_top_k=8
      - 1.7B dense: d_model=2048, num_layers=28 (or 48), num_heads=16,
                    num_kv_heads=4, head_dim=128, ffn_type='swiglu'
    """
    # Model dimensions
    d_model: int = 2048
    vocab_size: int = 151936
    num_layers: int = 48
    num_heads: int = 32         # Query heads
    num_kv_heads: int = 4       # Key-value heads (for GQA)
    head_dim: int = 128

    # FFN configuration
    ffn_type: str = "moe"       # "swiglu" or "moe"
    d_ff: int = None            # FFN intermediate dim; if None, use 4*d_model
    moe_num_experts: int = 128
    moe_top_k: int = 8

    # Gating configuration
    gating_position: str = "g1"           # "g1", "g2", "g3", "g4", "g5", or None (baseline)
    gating_granularity: str = "elementwise"
    gating_mode: str = "multiplicative"
    gating_scope: str = "head_specific"
    gating_activation: str = "sigmoid"

    # RoPE
    rope_base: float = 10000.0
    max_seq_len: int = 4096

    # Regularization
    dropout: float = 0.0
    use_sandwich_norm: bool = False

    # Training
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model

    def to_attn_config(self) -> GatedAttentionConfig:
        """Convert to GatedAttentionConfig for building attention layers."""
        if self.gating_position is None:
            return GatedAttentionConfig(
                position=GatingPosition.NONE,
                d_model=self.d_model,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                max_seq_len=self.max_seq_len,
            )

        position_map = {
            "g1": GatingPosition.G1_SDPA_OUTPUT,
            "g2": GatingPosition.G2_VALUE,
            "g3": GatingPosition.G3_KEY,
            "g4": GatingPosition.G4_QUERY,
            "g5": GatingPosition.G5_DENSE_OUTPUT,
        }
        granularity_map = {
            "headwise": GatingGranularity.HEADWISE,
            "elementwise": GatingGranularity.ELEMENTWISE,
        }
        mode_map = {
            "multiplicative": GatingMode.MULTIPLICATIVE,
            "additive": GatingMode.ADDITIVE,
        }
        scope_map = {
            "head_specific": GatingScope.HEAD_SPECIFIC,
            "head_shared": GatingScope.HEAD_SHARED,
        }
        activation_map = {
            "sigmoid": ActivationType.SIGMOID,
            "silu": ActivationType.SILU,
            "identity": ActivationType.IDENTITY,
            "ns_sigmoid": ActivationType.NS_SIGMOID,
        }

        return GatedAttentionConfig(
            position=position_map[self.gating_position],
            granularity=granularity_map[self.gating_granularity],
            mode=mode_map[self.gating_mode],
            scope=scope_map[self.gating_scope],
            activation=activation_map[self.gating_activation],
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
        )


class GatedLLM(nn.Module):
    """Gated Large Language Model.

    Full decoder-only transformer with gated attention, following
    the architecture described in the paper (Sec 3.1).
    """

    def __init__(self, config: GatedLLMConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        # RoPE
        self.rotary_emb = RotaryEmbedding(
            dim=config.head_dim,
            max_position_embeddings=config.max_seq_len,
            base=config.rope_base,
        )

        # Attention config for all layers
        attn_config = config.to_attn_config()

        # Decoder layers
        self.layers = nn.ModuleList([
            GatedDecoderBlock(
                d_model=config.d_model,
                attn_config=attn_config,
                d_ff=config.d_ff,
                ffn_type=config.ffn_type,
                moe_num_experts=config.moe_num_experts,
                moe_top_k=config.moe_top_k,
                dropout=config.dropout,
                use_sandwich_norm=config.use_sandwich_norm,
            )
            for _ in range(config.num_layers)
        ])

        # Final norm
        self.norm = RMSNorm(config.d_model)

        # Output head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Initialize
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = self.config.d_model ** -0.5
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_cache: bool = False,
    ) -> Dict:
        batch, seq_len = input_ids.shape

        # Token embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Position IDs
        if position_ids is None:
            position_ids = torch.arange(
                seq_len, device=input_ids.device
            ).unsqueeze(0).expand(batch, -1)

        # RoPE embeddings
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        rope_cache = (cos, sin)

        # Set RoPE cache for all attention layers
        for layer in self.layers:
            layer.attn.rope_cache = rope_cache

        # Causal mask
        if attention_mask is None:
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device),
                diagonal=1,
            )
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        else:
            # Convert attention_mask to causal attention format
            causal_mask = (1.0 - attention_mask[:, None, None, :].float()) * float("-inf")

        # Run through layers
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        all_kv_caches = [] if use_cache else None
        total_aux_losses = {}

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            past_kv = past_key_values[i] if past_key_values is not None else None

            hidden_states, attn_weights, new_kv, aux_losses = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_kv,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            if output_attentions:
                all_attentions.append(attn_weights)
            if use_cache:
                all_kv_caches.append(new_kv)
            for k, v in aux_losses.items():
                total_aux_losses[k] = total_aux_losses.get(k, 0.0) + v

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        # LM head
        logits = self.lm_head(hidden_states)

        return {
            "logits": logits,
            "hidden_states": all_hidden_states,
            "attentions": all_attentions,
            "past_key_values": all_kv_caches,
            "aux_losses": total_aux_losses,
        }

    def get_num_params(self) -> Dict[str, int]:
        """Count parameters by component."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Count gating parameters
        gate_params = 0
        for name, param in self.named_parameters():
            if "gate_proj" in name and "ffn" not in name.lower():
                gate_params += param.numel()

        return {
            "total": total,
            "trainable": trainable,
            "gate_params": gate_params,
            "non_gate_params": total - gate_params,
        }


# ---------------------------------------------------------------------------
# Paper-specific model configurations
# ---------------------------------------------------------------------------

def create_model_from_paper_config(model_type: str, gating_variant: Optional[str] = None) -> GatedLLM:
    """Create a model matching the paper's experimental configurations.

    Args:
        model_type: One of:
            - "15A2B": 15B MoE with 2.54B activated parameters
            - "1.7B_28L": 1.7B dense model with 28 layers
            - "1.7B_48L": 1.7B dense model with 48 layers
        gating_variant: One of the paper's variants:
            - None or "baseline": No gating
            - "g1_elementwise": SDPA elementwise gating (Table 1, row 5)
            - "g1_headwise": SDPA headwise gating (Table 1, row 10)
            - "g1_head_shared": SDPA head-shared gating (Table 1, row 12)
            - "g1_additive": SDPA additive gating with SiLU (Table 1, row 14)
            - "g1_silu": SDPA SiLU activation gating (Table 1, row 15)
            - "g2_elementwise": Value elementwise gating (Table 1, row 6)
            - "g2_headwise": Value headwise gating (Table 1, row 11)
            - "g3_elementwise": Key elementwise gating (Table 1, row 7)
            - "g4_elementwise": Query elementwise gating (Table 1, row 8)
            - "g5": Dense output gating (Table 1, row 9)
            - "g1_ns_sigmoid": NS-sigmoid gating (Table 4, row 7)
            - "g1_input_independent": Input-independent gating (Sec 4.2)

    Returns:
        Configured GatedLLM model
    """
    # Base configurations
    if model_type == "15A2B":
        base_config = dict(
            d_model=2048,
            num_layers=48,
            num_heads=32,
            num_kv_heads=4,
            head_dim=128,
            ffn_type="moe",
            d_ff=1536,  # fine-grained experts
            moe_num_experts=128,
            moe_top_k=8,
            rope_base=10000.0,
        )
    elif model_type == "1.7B_28L":
        base_config = dict(
            d_model=2048,
            num_layers=28,
            num_heads=16,
            num_kv_heads=4,
            head_dim=128,
            ffn_type="swiglu",
            d_ff=5632,  # adjusted for parameter matching when gating
            rope_base=10000.0,
        )
    elif model_type == "1.7B_48L":
        base_config = dict(
            d_model=1536,
            num_layers=48,
            num_heads=16,
            num_kv_heads=4,
            head_dim=96,
            ffn_type="swiglu",
            d_ff=4096,
            rope_base=10000.0,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Gating variant decoding
    if gating_variant is None or gating_variant == "baseline":
        gating_config = dict(gating_position=None)
    elif gating_variant == "g1_elementwise":
        gating_config = dict(
            gating_position="g1",
            gating_granularity="elementwise",
            gating_mode="multiplicative",
            gating_scope="head_specific",
            gating_activation="sigmoid",
        )
    elif gating_variant == "g1_headwise":
        gating_config = dict(
            gating_position="g1",
            gating_granularity="headwise",
            gating_mode="multiplicative",
            gating_scope="head_specific",
            gating_activation="sigmoid",
        )
    elif gating_variant == "g1_head_shared":
        gating_config = dict(
            gating_position="g1",
            gating_granularity="elementwise",
            gating_mode="multiplicative",
            gating_scope="head_shared",
            gating_activation="sigmoid",
        )
    elif gating_variant == "g1_additive":
        gating_config = dict(
            gating_position="g1",
            gating_granularity="elementwise",
            gating_mode="additive",
            gating_scope="head_specific",
            gating_activation="silu",
        )
    elif gating_variant == "g1_silu":
        gating_config = dict(
            gating_position="g1",
            gating_granularity="elementwise",
            gating_mode="multiplicative",
            gating_scope="head_specific",
            gating_activation="silu",
        )
    elif gating_variant == "g2_elementwise":
        gating_config = dict(
            gating_position="g2",
            gating_granularity="elementwise",
            gating_mode="multiplicative",
            gating_scope="head_specific",
            gating_activation="sigmoid",
        )
    elif gating_variant == "g2_headwise":
        gating_config = dict(
            gating_position="g2",
            gating_granularity="headwise",
            gating_mode="multiplicative",
            gating_scope="head_specific",
            gating_activation="sigmoid",
        )
    elif gating_variant == "g3_elementwise":
        gating_config = dict(
            gating_position="g3",
            gating_granularity="elementwise",
            gating_mode="multiplicative",
            gating_scope="head_specific",
            gating_activation="sigmoid",
        )
    elif gating_variant == "g4_elementwise":
        gating_config = dict(
            gating_position="g4",
            gating_granularity="elementwise",
            gating_mode="multiplicative",
            gating_scope="head_specific",
            gating_activation="sigmoid",
        )
    elif gating_variant == "g5":
        gating_config = dict(
            gating_position="g5",
            gating_granularity="elementwise",
            gating_mode="multiplicative",
            gating_scope="head_specific",
            gating_activation="sigmoid",
        )
    elif gating_variant == "g1_ns_sigmoid":
        gating_config = dict(
            gating_position="g1",
            gating_granularity="elementwise",
            gating_mode="multiplicative",
            gating_scope="head_specific",
            gating_activation="ns_sigmoid",
        )
    elif gating_variant == "g1_input_independent":
        # Input-independent gating: zero-init gate params (Sec 4.2)
        gating_config = dict(
            gating_position="g1",
            gating_granularity="elementwise",
            gating_mode="multiplicative",
            gating_scope="head_specific",
            gating_activation="sigmoid",
        )
    else:
        raise ValueError(f"Unknown gating variant: {gating_variant}")

    config = GatedLLMConfig(**{**base_config, **gating_config})
    model = GatedLLM(config)

    # For input-independent gating, zero-init gate parameters
    if gating_variant == "g1_input_independent":
        for name, param in model.named_parameters():
            if "attn.gate_proj" in name:
                nn.init.zeros_(param)

    return model
