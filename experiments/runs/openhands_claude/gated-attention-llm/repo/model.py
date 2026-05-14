"""
Full transformer language model supporting:
  - Dense 1.7B (28-layer and 48-layer variants)
  - MoE 15A2B (15B total / 2.54B activated)

Both architectures share the same GatedTransformerLM class; the difference
is entirely in the ModelConfig (use_moe flag, num_layers, d_model, etc.).
"""

from typing import Optional

import torch
import torch.nn as nn

from config import ModelConfig, GatingConfig
from modules import RMSNorm
from layers import TransformerBlock


class GatedTransformerLM(nn.Module):
    """Decoder-only transformer language model with configurable gating.

    Architecture:
      - Token embedding
      - N × TransformerBlock (pre-norm, optional sandwich norm)
      - Final RMSNorm
      - LM head (optionally tied to embedding)

    The gating configuration is embedded in ModelConfig.block.attention.gating
    and propagated to every attention layer.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(
            [TransformerBlock(cfg.block) for _ in range(cfg.num_layers)]
        )
        self.norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self._init_weights()

    def _init_weights(self):
        std = 0.02
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=std)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _make_causal_mask(
        self,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
        past_len: int = 0,
    ) -> torch.Tensor:
        """Upper-triangular additive causal mask (−inf above diagonal)."""
        total_len = past_len + seq_len
        mask = torch.full((seq_len, total_len), float("-inf"), dtype=dtype, device=device)
        mask = torch.triu(mask, diagonal=past_len + 1)
        return mask[None, None, :, :]  # (1, 1, seq_len, total_len)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            input_ids:       (batch, seq_len)
            attention_mask:  (batch, seq_len) boolean mask (1=attend, 0=ignore)
            past_key_values: list of (K, V) per layer for incremental decoding
            use_cache:       return updated KV caches
            labels:          (batch, seq_len) for computing cross-entropy loss

        Returns:
            dict with keys:
              'logits'          – (batch, seq_len, vocab_size)
              'loss'            – scalar CE loss (only if labels provided)
              'aux_loss'        – scalar MoE auxiliary loss (0 for dense)
              'past_key_values' – updated caches (only if use_cache=True)
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device
        dtype = next(self.parameters()).dtype

        past_len = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        # Token embeddings
        x = self.embed_tokens(input_ids)  # (batch, seq_len, d_model)

        # Build causal mask
        causal_mask = self._make_causal_mask(seq_len, dtype, device, past_len)

        # Combine with padding mask if provided
        if attention_mask is not None:
            # attention_mask: (batch, seq_len) → (batch, 1, 1, seq_len)
            pad_mask = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2)
            pad_mask = pad_mask * torch.finfo(dtype).min
            causal_mask = causal_mask + pad_mask

        new_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        total_aux_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, new_cache, aux_losses = layer(
                x,
                attention_mask=causal_mask,
                past_key_value=past_kv,
                use_cache=use_cache,
            )
            if use_cache and new_cache is not None:
                new_caches.append(new_cache)
            for v in aux_losses.values():
                total_aux_loss = total_aux_loss + v

        x = self.norm(x)
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)

        result: dict[str, torch.Tensor] = {
            "logits": logits,
            "aux_loss": total_aux_loss,
        }

        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            ce_loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.cfg.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = ce_loss + total_aux_loss

        if use_cache:
            result["past_key_values"] = new_caches

        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Simple greedy / top-p generation for evaluation."""
        past_key_values = None
        generated = input_ids

        for _ in range(max_new_tokens):
            out = self.forward(
                generated if past_key_values is None else generated[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = out["logits"][:, -1, :]  # (batch, vocab_size)
            past_key_values = out["past_key_values"]

            if temperature != 1.0:
                logits = logits / temperature

            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[sorted_mask] = float("-inf")
                logits = torch.scatter(logits, 1, sorted_idx, sorted_logits)

            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def build_model(cfg: ModelConfig) -> GatedTransformerLM:
    return GatedTransformerLM(cfg)


def count_parameters(model: GatedTransformerLM) -> dict[str, int]:
    """Return total and activated parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    # For MoE: activated params = non-expert params + top_k/num_experts * expert params
    expert_params = 0
    non_expert_params = 0
    for name, p in model.named_parameters():
        if "experts." in name:
            expert_params += p.numel()
        else:
            non_expert_params += p.numel()

    cfg = model.cfg
    if cfg.block.use_moe:
        moe_cfg = cfg.block.moe
        activated = non_expert_params + int(
            expert_params * moe_cfg.num_experts_per_tok / moe_cfg.num_experts
        )
    else:
        activated = total

    return {"total": total, "activated": activated, "expert": expert_params}
