"""
OLMoE: Open Mixture-of-Experts Language Model

Architecture implementation based on the paper:
"OLMoE: Open Mixture-of-Experts Language Models"

Key design choices for OLMoE-1B-7B:
- 16 transformer layers
- 2048 hidden dimension
- 16 attention heads
- 64 experts per MoE layer, 8 activated (top-k routing)
- FFN dimension of 1024 per expert
- Dropless token choice routing
- Load balancing loss (weight=0.01)
- Router Z-loss (weight=0.001)
- RMSNorm (parametric)
- QK-Norm
- RoPE positional embeddings
- SwiGLU activations
- Truncated normal initialization (std=0.02, clip at 3*std=0.06)
- AdamW with epsilon=1e-8
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class OLMoEConfig:
    """Configuration for OLMoE model."""
    # Model dimensions
    hidden_size: int = 2048
    num_hidden_layers: int = 16
    num_attention_heads: int = 16
    vocab_size: int = 50304

    # MoE configuration
    num_experts: int = 64
    num_experts_per_tok: int = 8
    expert_ffn_dim: int = 1024

    # Auxiliary loss weights
    load_balancing_loss_weight: float = 0.01
    router_z_loss_weight: float = 0.001

    # Normalization
    rms_norm_eps: float = 1e-5
    use_qk_norm: bool = True

    # Positional embeddings
    rope_theta: float = 10000.0
    max_position_embeddings: int = 4096

    # Initialization
    init_std: float = 0.02
    init_trunc_factor: float = 3.0

    # Training
    use_load_balancing_loss: bool = True
    use_router_z_loss: bool = True

    # Attention
    attention_dropout: float = 0.0

    # Whether to tie input/output embeddings
    tie_word_embeddings: bool = False

    # Sequence length
    max_seq_len: int = 4096


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (parametric).

    Used instead of non-parametric LayerNorm as in OLMo.
    The paper finds RMSNorm leads to better performance and fewer gradient spikes.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_position_embeddings: int = 4096, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class OLMoEAttention(nn.Module):
    """Multi-head attention with optional QK-Norm."""

    def __init__(self, config: OLMoEConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.use_qk_norm = config.use_qk_norm

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            theta=config.rope_theta,
        )
        self.attention_dropout = config.attention_dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        cos, sin = self.rotary_emb(q, seq_len=seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=attention_mask is None,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output


class OLMoEExpert(nn.Module):
    """Single FFN expert with SwiGLU activation."""

    def __init__(self, hidden_size: int, expert_ffn_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, expert_ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden_size, expert_ffn_dim, bias=False)
        self.down_proj = nn.Linear(expert_ffn_dim, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class OLMoEMoELayer(nn.Module):
    """Mixture-of-Experts layer with dropless token choice routing.

    MoE module(x) = sum_{i in Top-k(r(x))} softmax(r(x))_i * E_i(x)

    Auxiliary losses:
    - Load balancing loss: L_LB = N_E * sum_i f_i * P_i
    - Router Z-loss: L_RZ = (1/B) * sum_i (log sum_j exp(x_j^(i)))^2
    """

    def __init__(self, config: OLMoEConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.use_load_balancing_loss = config.use_load_balancing_loss
        self.use_router_z_loss = config.use_router_z_loss

        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([
            OLMoEExpert(config.hidden_size, config.expert_ffn_dim)
            for _ in range(config.num_experts)
        ])

    def compute_load_balancing_loss(
        self,
        router_logits: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        """Load balancing loss from Shazeer et al. (2017).

        L_LB = N_E * sum_{i=1}^{N_E} f_i * P_i

        f_i = fraction of tokens routed to expert i
        P_i = total routing probability allocated to expert i
        """
        num_tokens = router_logits.shape[0]
        num_experts = self.num_experts

        routing_probs = F.softmax(router_logits, dim=-1)

        expert_mask = torch.zeros(
            num_tokens, num_experts,
            device=router_logits.device,
            dtype=router_logits.dtype
        )
        expert_mask.scatter_(1, selected_experts, 1.0)

        tokens_per_expert = expert_mask.mean(dim=0)
        router_prob_per_expert = routing_probs.mean(dim=0)

        load_balancing_loss = num_experts * torch.sum(
            tokens_per_expert * router_prob_per_expert
        )
        return load_balancing_loss

    def compute_router_z_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """Router Z-loss from Zoph et al. (2022) ST-MoE.

        L_RZ(x) = (1/B) * sum_{i=1}^{B} (log sum_{j=1}^{N_E} exp(x_j^(i)))^2
        """
        log_z = torch.logsumexp(router_logits, dim=-1)
        z_loss = torch.mean(log_z ** 2)
        return z_loss

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        num_tokens = hidden_states_flat.shape[0]

        router_logits = self.router(hidden_states_flat)

        load_balancing_loss = None
        router_z_loss = None

        routing_weights_topk, selected_experts = torch.topk(
            router_logits, self.num_experts_per_tok, dim=-1
        )

        if self.use_load_balancing_loss and self.training:
            load_balancing_loss = self.compute_load_balancing_loss(
                router_logits, selected_experts
            )

        if self.use_router_z_loss and self.training:
            router_z_loss = self.compute_router_z_loss(router_logits)

        # Softmax over all experts, then gather top-k weights
        routing_weights_full = F.softmax(router_logits, dim=-1)
        routing_weights = routing_weights_full.gather(1, selected_experts)

        output = torch.zeros_like(hidden_states_flat)

        for expert_idx in range(self.num_experts):
            expert_mask = (selected_experts == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue

            expert_input = hidden_states_flat[expert_mask]
            expert_output = self.experts[expert_idx](expert_input)

            # Get routing weight for this expert
            expert_position = (selected_experts[expert_mask] == expert_idx)
            expert_weights = (routing_weights[expert_mask] * expert_position.float()).sum(dim=-1, keepdim=True)

            output[expert_mask] += expert_weights * expert_output

        output = output.view(batch_size, seq_len, hidden_size)
        return output, load_balancing_loss, router_z_loss


class OLMoEDecoderLayer(nn.Module):
    """Single transformer decoder layer with MoE FFN."""

    def __init__(self, config: OLMoEConfig):
        super().__init__()
        self.self_attn = OLMoEAttention(config)
        self.moe = OLMoEMoELayer(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_ids)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, load_balancing_loss, router_z_loss = self.moe(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, load_balancing_loss, router_z_loss


class OLMoEModel(nn.Module):
    """OLMoE transformer model (without LM head)."""

    def __init__(self, config: OLMoEConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            OLMoEDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.embed_tokens(input_ids)

        total_load_balancing_loss = torch.tensor(0.0, device=input_ids.device)
        total_router_z_loss = torch.tensor(0.0, device=input_ids.device)

        for layer in self.layers:
            hidden_states, lb_loss, rz_loss = layer(
                hidden_states, attention_mask, position_ids
            )
            if lb_loss is not None:
                total_load_balancing_loss = total_load_balancing_loss + lb_loss
            if rz_loss is not None:
                total_router_z_loss = total_router_z_loss + rz_loss

        hidden_states = self.norm(hidden_states)

        num_layers = self.config.num_hidden_layers
        avg_lb_loss = total_load_balancing_loss / num_layers
        avg_rz_loss = total_router_z_loss / num_layers

        aux_loss = (
            self.config.load_balancing_loss_weight * avg_lb_loss
            + self.config.router_z_loss_weight * avg_rz_loss
        )

        return hidden_states, aux_loss


class OLMoEForCausalLM(nn.Module):
    """OLMoE model with causal language modeling head.

    Training loss: L = L_CE + alpha * L_LB + beta * L_RZ
    - L_CE: cross-entropy loss
    - L_LB: load balancing loss (alpha=0.01)
    - L_RZ: router Z-loss (beta=0.001)
    """

    def __init__(self, config: OLMoEConfig):
        super().__init__()
        self.config = config
        self.model = OLMoEModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Truncated normal initialization: std=0.02, clip at +/-3*std=+/-0.06."""
        std = self.config.init_std
        trunc_val = self.config.init_trunc_factor * std

        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-trunc_val, b=trunc_val)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-trunc_val, b=trunc_val)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> dict:
        hidden_states, aux_loss = self.model(input_ids, attention_mask, position_ids)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            ce_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss = ce_loss + aux_loss

        return {
            "logits": logits,
            "loss": loss,
            "aux_loss": aux_loss,
        }

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def create_olmoe_1b_7b() -> OLMoEForCausalLM:
    """Create OLMoE-1B-7B with exact configuration from the paper (Table 10)."""
    config = OLMoEConfig(
        hidden_size=2048,
        num_hidden_layers=16,
        num_attention_heads=16,
        vocab_size=50304,
        num_experts=64,
        num_experts_per_tok=8,
        expert_ffn_dim=1024,
        load_balancing_loss_weight=0.01,
        router_z_loss_weight=0.001,
        rms_norm_eps=1e-5,
        use_qk_norm=True,
        rope_theta=10000.0,
        max_position_embeddings=4096,
        init_std=0.02,
        init_trunc_factor=3.0,
        use_load_balancing_loss=True,
        use_router_z_loss=True,
        tie_word_embeddings=False,
    )
    return OLMoEForCausalLM(config)


class OLMoEMoELayerWithSharedExpert(nn.Module):
    """MoE layer with a shared (always-active) expert.

    From Dai et al. (2024) DeepSeekMoE:
    "training with a shared/fixed expert that is always used in addition to the routed experts"

    From Section 4.1.3 of the paper:
    - Having a shared expert performs slightly worse than all-routed setup
    - Sharing reduces possible expert combinations significantly
    - Example: 1 shared + 31 routed with 3 activated = C(31,3) = 4,495 combinations
    - vs 32 routed with 4 activated = C(32,4) = 35,960 combinations

    The paper does NOT use shared experts in OLMoE-1B-7B.
    """

    def __init__(self, config: OLMoEConfig, num_shared_experts: int = 1):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        self.hidden_size = config.hidden_size
        self.use_load_balancing_loss = config.use_load_balancing_loss
        self.use_router_z_loss = config.use_router_z_loss

        # Shared experts (always active)
        self.shared_experts = nn.ModuleList([
            OLMoEExpert(config.hidden_size, config.expert_ffn_dim)
            for _ in range(num_shared_experts)
        ])

        # Routed experts
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.routed_experts = nn.ModuleList([
            OLMoEExpert(config.hidden_size, config.expert_ffn_dim)
            for _ in range(config.num_experts)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_size)

        # Process shared experts (always active)
        shared_output = torch.zeros_like(hidden_flat)
        for shared_expert in self.shared_experts:
            shared_output += shared_expert(hidden_flat)

        # Process routed experts (top-k selection)
        router_logits = self.router(hidden_flat)
        routing_weights_topk, selected_experts = torch.topk(
            router_logits, self.num_experts_per_tok, dim=-1
        )

        load_balancing_loss = None
        router_z_loss = None

        if self.use_load_balancing_loss and self.training:
            # Compute load balancing loss for routed experts only
            routing_probs = F.softmax(router_logits, dim=-1)
            num_tokens = hidden_flat.shape[0]
            expert_mask = torch.zeros(
                num_tokens, self.num_experts,
                device=router_logits.device,
                dtype=router_logits.dtype
            )
            expert_mask.scatter_(1, selected_experts, 1.0)
            tokens_per_expert = expert_mask.mean(dim=0)
            router_prob_per_expert = routing_probs.mean(dim=0)
            load_balancing_loss = self.num_experts * torch.sum(
                tokens_per_expert * router_prob_per_expert
            )

        if self.use_router_z_loss and self.training:
            log_z = torch.logsumexp(router_logits, dim=-1)
            router_z_loss = torch.mean(log_z ** 2)

        routing_weights_full = F.softmax(router_logits, dim=-1)
        routing_weights = routing_weights_full.gather(1, selected_experts)

        routed_output = torch.zeros_like(hidden_flat)
        for expert_idx in range(self.num_experts):
            expert_mask = (selected_experts == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue
            expert_input = hidden_flat[expert_mask]
            expert_output = self.routed_experts[expert_idx](expert_input)
            expert_position = (selected_experts[expert_mask] == expert_idx)
            expert_weights = (routing_weights[expert_mask] * expert_position.float()).sum(dim=-1, keepdim=True)
            routed_output[expert_mask] += expert_weights * expert_output

        # Combine shared and routed outputs
        output = (shared_output + routed_output).view(batch_size, seq_len, hidden_size)
        return output, load_balancing_loss, router_z_loss
