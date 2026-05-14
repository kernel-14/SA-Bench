"""
Mixture-of-Experts Transformer with Gated Attention.

Implements the 15A2B MoE model architecture from the paper:
  - 128 total experts, top-8 softmax gating
  - Fine-grained experts (DeepSeekMoE style)
  - Group Query Attention (GQA)
  - Gated attention variants

The MoE architecture follows DeepSeekMoE (Dai et al., 2024) with:
  - Fine-grained expert segmentation
  - Global-batch load balancing loss (LBL)
  - Z-loss for router stability
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gated_attention import (
    GatedMultiHeadAttention,
    GatingActivation,
    GatingGranularity,
    GatingPosition,
    GatingType,
)
from .transformer import RMSNorm, TransformerConfig


@dataclass
class MoEConfig:
    """Configuration for the MoE transformer model.
    
    Default values match the 15A2B MoE model from the paper.
    """
    # Model dimensions
    d_model: int = 2048
    num_layers: int = 24
    num_heads: int = 32   # q heads
    num_kv_heads: int = 4  # kv heads (GQA)
    head_dim: int = 128
    vocab_size: int = 151936
    max_seq_len: int = 4096
    
    # MoE configuration
    num_experts: int = 128       # Total experts
    num_experts_per_tok: int = 8  # Top-k experts activated
    expert_intermediate_dim: int = 1408  # Fine-grained expert dim
    num_shared_experts: int = 1   # Shared (always-active) experts
    
    # Regularization
    dropout: float = 0.0
    router_z_loss_coef: float = 0.001  # Z-loss coefficient
    router_aux_loss_coef: float = 0.001  # Load balancing loss coefficient
    
    # RoPE
    rope_base: float = 10000.0
    
    # Gating configuration
    gating_position: Optional[str] = "sdpa_output"
    gating_granularity: str = "elementwise"
    head_specific: bool = True
    gating_type: str = "multiplicative"
    gating_activation: str = "sigmoid"
    
    use_sandwich_norm: bool = False
    
    def get_gating_position(self) -> Optional[GatingPosition]:
        if self.gating_position is None:
            return None
        mapping = {
            "sdpa_output": GatingPosition.G1_SDPA_OUTPUT,
            "value": GatingPosition.G2_VALUE,
            "key": GatingPosition.G3_KEY,
            "query": GatingPosition.G4_QUERY,
            "dense_output": GatingPosition.G5_DENSE_OUTPUT,
        }
        return mapping[self.gating_position]
    
    def get_gating_granularity(self) -> GatingGranularity:
        return GatingGranularity(self.gating_granularity)
    
    def get_gating_type(self) -> GatingType:
        return GatingType(self.gating_type)
    
    def get_gating_activation(self) -> GatingActivation:
        return GatingActivation(self.gating_activation)


class MoERouter(nn.Module):
    """Softmax router for Mixture-of-Experts.
    
    Implements top-k routing with:
      - Z-loss for router stability (Zoph et al., 2022)
      - Load balancing loss (Qiu et al., 2025)
    """
    
    def __init__(self, d_model: int, num_experts: int, num_experts_per_tok: int,
                 z_loss_coef: float = 0.001, aux_loss_coef: float = 0.001):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.z_loss_coef = z_loss_coef
        self.aux_loss_coef = aux_loss_coef
        
        self.gate = nn.Linear(d_model, num_experts, bias=False)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch * seq, d_model)
        
        Returns:
            routing_weights: (batch * seq, num_experts_per_tok) - normalized weights
            selected_experts: (batch * seq, num_experts_per_tok) - expert indices
            aux_loss: scalar auxiliary loss
        """
        # Router logits
        router_logits = self.gate(x)  # (tokens, num_experts)
        
        # Z-loss: penalizes large logits for numerical stability
        z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
        
        # Top-k routing with softmax
        routing_weights = F.softmax(router_logits, dim=-1)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.num_experts_per_tok, dim=-1
        )
        
        # Renormalize selected weights
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # Load balancing loss: encourage uniform expert utilization
        # f_i * P_i where f_i is fraction of tokens routed to expert i
        # and P_i is mean routing probability for expert i
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).float()
        # expert_mask: (tokens, top_k, num_experts)
        tokens_per_expert = expert_mask.sum(dim=1).mean(dim=0)  # (num_experts,)
        tokens_per_expert = tokens_per_expert / self.num_experts_per_tok
        
        mean_routing_prob = routing_weights.mean(dim=0)  # This is approximate
        # Better: use full softmax probs
        full_probs = F.softmax(router_logits, dim=-1)
        mean_routing_prob = full_probs.mean(dim=0)  # (num_experts,)
        
        aux_loss = (
            self.aux_loss_coef * self.num_experts * (tokens_per_expert * mean_routing_prob).sum()
            + self.z_loss_coef * z_loss
        )
        
        return routing_weights, selected_experts, aux_loss


class MoEExpert(nn.Module):
    """Single MoE expert: a SwiGLU FFN."""
    
    def __init__(self, d_model: int, intermediate_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseMoEFFN(nn.Module):
    """Sparse Mixture-of-Experts FFN layer.
    
    Implements fine-grained expert segmentation (DeepSeekMoE style):
      - Many small experts instead of few large ones
      - Top-k routing with softmax
      - Optional shared experts (always active)
    """
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.num_shared_experts = config.num_shared_experts
        
        # Router
        self.router = MoERouter(
            d_model=config.d_model,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            z_loss_coef=config.router_z_loss_coef,
            aux_loss_coef=config.router_aux_loss_coef,
        )
        
        # Sparse experts
        self.experts = nn.ModuleList([
            MoEExpert(config.d_model, config.expert_intermediate_dim)
            for _ in range(config.num_experts)
        ])
        
        # Shared experts (always active, like dense FFN)
        if config.num_shared_experts > 0:
            shared_dim = config.expert_intermediate_dim * config.num_shared_experts
            self.shared_expert = MoEExpert(config.d_model, shared_dim)
        else:
            self.shared_expert = None
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq, d_model)
        
        Returns:
            output: (batch, seq, d_model)
            aux_loss: scalar auxiliary loss
        """
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # (tokens, d_model)
        
        # Route tokens to experts
        routing_weights, selected_experts, aux_loss = self.router(x_flat)
        
        # Compute expert outputs
        final_hidden = torch.zeros_like(x_flat)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (selected_experts == expert_idx)  # (tokens, top_k)
            token_mask = expert_mask.any(dim=-1)  # (tokens,)
            
            if not token_mask.any():
                continue
            
            # Get tokens for this expert
            expert_tokens = x_flat[token_mask]  # (n_tokens, d_model)
            expert_output = self.experts[expert_idx](expert_tokens)
            
            # Get routing weights for this expert
            expert_weights = routing_weights[token_mask] * expert_mask[token_mask].float()
            expert_weights = expert_weights.sum(dim=-1, keepdim=True)  # (n_tokens, 1)
            
            # Accumulate weighted output
            final_hidden[token_mask] += expert_weights * expert_output
        
        # Add shared expert output
        if self.shared_expert is not None:
            final_hidden = final_hidden + self.shared_expert(x_flat)
        
        output = final_hidden.view(batch_size, seq_len, d_model)
        return output, aux_loss


class MoETransformerLayer(nn.Module):
    """Single MoE transformer layer with gated attention."""
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        
        # Pre-norm
        self.attn_norm = RMSNorm(config.d_model)
        self.ffn_norm = RMSNorm(config.d_model)
        
        # Sandwich norm
        self.use_sandwich_norm = config.use_sandwich_norm
        if config.use_sandwich_norm:
            self.attn_post_norm = RMSNorm(config.d_model)
            self.ffn_post_norm = RMSNorm(config.d_model)
        
        # Gated attention
        self.attn = GatedMultiHeadAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            dropout=config.dropout,
            gating_position=config.get_gating_position(),
            gating_granularity=config.get_gating_granularity(),
            head_specific=config.head_specific,
            gating_type=config.get_gating_type(),
            gating_activation=config.get_gating_activation(),
            max_seq_len=config.max_seq_len,
            rope_base=config.rope_base,
        )
        
        # MoE FFN
        self.ffn = SparseMoEFFN(config)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """
        Returns:
            x: Updated hidden states
            aux_loss: MoE auxiliary loss
            past_key_value: Updated KV cache
        """
        # Attention
        residual = x
        x_norm = self.attn_norm(x)
        attn_out, new_past_kv = self.attn(
            x_norm,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        if self.use_sandwich_norm:
            attn_out = self.attn_post_norm(attn_out)
        x = residual + attn_out
        
        # MoE FFN
        residual = x
        x_norm = self.ffn_norm(x)
        ffn_out, aux_loss = self.ffn(x_norm)
        if self.use_sandwich_norm:
            ffn_out = self.ffn_post_norm(ffn_out)
        x = residual + ffn_out
        
        return x, aux_loss, new_past_kv


class MoETransformerModel(nn.Module):
    """Full MoE transformer model with gated attention.
    
    Implements the 15A2B MoE model from the paper:
      - 128 total experts, top-8 routing
      - 15B total parameters, ~2.54B activated
      - GQA with 32 query heads, 4 KV heads
    """
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        
        self.layers = nn.ModuleList([
            MoETransformerLayer(config)
            for _ in range(config.num_layers)
        ])
        
        self.norm = RMSNorm(config.d_model)
        
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        batch_size, seq_len = input_ids.shape
        x = self.embed_tokens(input_ids)
        
        total_aux_loss = torch.tensor(0.0, device=x.device)
        new_past_key_values = [] if use_cache else None
        
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, aux_loss, new_past_kv = layer(
                x,
                attention_mask=attention_mask,
                past_key_value=past_kv,
                use_cache=use_cache,
            )
            total_aux_loss = total_aux_loss + aux_loss
            if use_cache:
                new_past_key_values.append(new_past_kv)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss = lm_loss + total_aux_loss / self.config.num_layers
        
        return {
            "logits": logits,
            "loss": loss,
            "aux_loss": total_aux_loss,
            "past_key_values": new_past_key_values,
        }
