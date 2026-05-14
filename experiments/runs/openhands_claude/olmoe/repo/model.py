"""OLMoE: Open Mixture-of-Experts Language Model.

Full decoder-only transformer with sparse MoE layers.
Architecture matches OLMoE-1B-7B (Muennighoff et al., 2024):
  - 16 layers, d_model=2048, 16 heads
  - 64 experts per layer, 8 activated per token
  - Expert FFN dim=1024 (SwiGLU)
  - RMSNorm, QK-Norm, RoPE
  - Truncated normal initialization
  - No weight tying, no biases
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from config import ModelConfig
from layers import RMSNorm
from modules import TransformerBlock


class OLMoE(nn.Module):
    """OLMoE decoder-only language model with sparse MoE layers.

    Training loss (Equation 2 in paper):
        L = L_CE + alpha * L_LB + beta * L_RZ

    where:
        L_CE = cross-entropy language modeling loss
        L_LB = load balancing loss (alpha=0.01)
        L_RZ = router Z-loss (beta=0.001)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_experts=config.n_experts,
                n_experts_per_token=config.n_experts_per_token,
                ffn_dim=config.ffn_dim,
                max_seq_len=config.max_seq_len,
                rope_theta=config.rope_theta,
                use_qk_norm=config.use_qk_norm,
                norm_eps=config.norm_eps,
                use_bias=config.use_bias,
            )
            for _ in range(config.n_layers)
        ])

        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # No weight tying (tie_embeddings=False per paper)
        # OLMo uses weight tying but OLMoE does not (Table 10)

        self._init_weights()

    def _init_weights(self):
        """Truncated normal initialization (§4.2.2).

        std=0.02, truncated at ±3*std = ±0.06.
        This improves stability vs regular normal init.
        """
        std = self.config.init_std
        trunc = self.config.init_trunc_factor * std

        def _init_module(module: nn.Module):
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-trunc, b=trunc)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-trunc, b=trunc)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

        self.apply(_init_module)

    def get_num_params(self) -> Dict[str, int]:
        """Return parameter counts: total, active (per token), vocab."""
        total = sum(p.numel() for p in self.parameters())
        vocab_params = self.embed_tokens.weight.numel() + self.lm_head.weight.numel()

        # Active params = everything except inactive experts
        # Each layer has 64 experts but only 8 are active per token
        expert_params_per_layer = sum(
            p.numel() for p in self.layers[0].moe.experts.parameters()
        )
        inactive_expert_params = (
            self.config.n_layers
            * expert_params_per_layer
            * (1 - self.config.n_experts_per_token / self.config.n_experts)
        )
        active = total - int(inactive_expert_params)

        return {"total": total, "active": active, "vocab": vocab_params}

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_router_info: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (batch, seq_len) token IDs
            attention_mask: (batch, 1, seq_len, seq_len) additive mask (optional)
            labels: (batch, seq_len) target token IDs for LM loss
            return_router_info: collect routing metadata for analysis
        Returns:
            dict with 'logits', 'loss' (if labels), 'load_balance_loss',
            'router_z_loss', and optionally 'router_info'
        """
        x = self.embed_tokens(input_ids)

        total_lb_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        total_rz_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        router_info_per_layer: List[Dict] = []

        for layer in self.layers:
            x, aux = layer(x, attention_mask=attention_mask, return_router_info=return_router_info)
            total_lb_loss = total_lb_loss + aux["load_balance_loss"]
            total_rz_loss = total_rz_loss + aux["router_z_loss"]
            if return_router_info:
                router_info_per_layer.append({
                    k: v for k, v in aux.items()
                    if k not in ("load_balance_loss", "router_z_loss")
                })

        x = self.norm(x)
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)

        # Average auxiliary losses over layers
        avg_lb_loss = total_lb_loss / self.config.n_layers
        avg_rz_loss = total_rz_loss / self.config.n_layers

        out = {
            "logits": logits,
            "load_balance_loss": avg_lb_loss,
            "router_z_loss": avg_rz_loss,
        }

        if labels is not None:
            # Cross-entropy loss: shift labels left by 1
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            ce_loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            # Total loss: L = L_CE + alpha * L_LB + beta * L_RZ
            total_loss = (
                ce_loss
                + self.config.load_balance_loss_weight * avg_lb_loss
                + self.config.router_z_loss_weight * avg_rz_loss
            )

            out["ce_loss"] = ce_loss
            out["loss"] = total_loss

        if return_router_info:
            out["router_info"] = router_info_per_layer

        return out

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Simple autoregressive generation with temperature and top-p sampling."""
        self.eval()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # Truncate to max_seq_len
            context = generated[:, -self.config.max_seq_len:]
            out = self.forward(context)
            next_logits = out["logits"][:, -1, :]  # (batch, vocab)

            if temperature != 1.0:
                next_logits = next_logits / temperature

            if top_p < 1.0:
                next_logits = _top_p_filter(next_logits, top_p)

            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated


def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus (top-p) filtering."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs - torch.softmax(sorted_logits, dim=-1) > top_p
    sorted_logits[sorted_indices_to_remove] = float("-inf")
    return logits.scatter(1, sorted_indices, sorted_logits)


def build_olmoe_1b_7b() -> OLMoE:
    """Build OLMoE-1B-7B with the exact configuration from the paper."""
    config = ModelConfig(
        d_model=2048,
        n_heads=16,
        n_layers=16,
        vocab_size=50304,
        max_seq_len=4096,
        n_experts=64,
        n_experts_per_token=8,
        ffn_dim=1024,
        norm_eps=1e-5,
        use_qk_norm=True,
        rope_theta=10000.0,
        init_std=0.02,
        init_trunc_factor=3.0,
        use_bias=False,
        tie_embeddings=False,
        load_balance_loss_weight=0.01,
        router_z_loss_weight=0.001,
    )
    return OLMoE(config)
