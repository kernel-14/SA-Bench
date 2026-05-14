"""
Expert Choice Routing for OLMoE Ablation Experiments

Implements Expert Choice (EC) routing as an alternative to Token Choice (TC).

From Section 4.1.4 of the paper:
- Token Choice (TC): each token selects top-k experts
- Expert Choice (EC): each expert selects top-c tokens from the sequence

Key findings:
- TC outperforms EC for the same token budget on all tasks
- EC runs ~20% faster (29,400 vs 24,400 tokens/sec/device)
- EC has perfect load balance by design (no need for load balancing loss)
- EC cannot be used for autoregressive generation (processes full sequence at once)
- EC can lead to token dropping (some tokens not selected by any expert)

The paper uses TC for OLMoE-1B-7B.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ExpertChoiceMoELayer(nn.Module):
    """
    Expert Choice (EC) MoE layer.

    Each expert selects a fixed number of tokens from the incoming sequence.
    This ensures perfect load balance but cannot be used for autoregressive generation.

    From Zhou et al. (2022) "Mixture-of-Experts with Expert Choice Routing".

    Args:
        config: OLMoE configuration
        capacity_factor: Number of tokens each expert processes = capacity_factor * (seq_len / num_experts)
    """

    def __init__(self, config, capacity_factor: float = 2.0):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.capacity_factor = capacity_factor

        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)

        from src.model import OLMoEExpert
        self.experts = nn.ModuleList([
            OLMoEExpert(config.hidden_size, config.expert_ffn_dim)
            for _ in range(config.num_experts)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, None, None]:
        """
        Expert Choice routing forward pass.

        Each expert selects top-c tokens where c = capacity_factor * (seq_len / num_experts).

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]

        Returns:
            output: [batch_size, seq_len, hidden_size]
            None (no load balancing loss needed - EC has perfect balance)
            None (no router z-loss)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_size)  # [B*T, H]
        num_tokens = hidden_flat.shape[0]

        # Compute router logits: [B*T, num_experts]
        router_logits = self.router(hidden_flat)

        # Transpose to [num_experts, B*T] for expert-centric view
        router_logits_T = router_logits.T  # [num_experts, B*T]

        # Each expert selects top-c tokens
        capacity = int(self.capacity_factor * num_tokens / self.num_experts)
        capacity = max(1, capacity)

        # Get top-c tokens for each expert
        # expert_weights: [num_experts, capacity]
        # expert_indices: [num_experts, capacity] - which tokens each expert processes
        expert_weights, expert_indices = torch.topk(
            router_logits_T, capacity, dim=-1
        )

        # Apply softmax to get routing probabilities
        expert_weights = F.softmax(expert_weights, dim=-1)

        # Initialize output
        output = torch.zeros_like(hidden_flat)
        output_counts = torch.zeros(num_tokens, device=hidden_flat.device)

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Get tokens for this expert
            token_indices = expert_indices[expert_idx]  # [capacity]
            weights = expert_weights[expert_idx]  # [capacity]

            # Get expert input
            expert_input = hidden_flat[token_indices]  # [capacity, H]

            # Process through expert
            expert_output = self.experts[expert_idx](expert_input)  # [capacity, H]

            # Accumulate weighted output
            for i, (tok_idx, weight) in enumerate(zip(token_indices, weights)):
                output[tok_idx] += weight * expert_output[i]
                output_counts[tok_idx] += 1

        # Note: tokens not selected by any expert get zero output (token dropping)
        # This is a known issue with EC routing

        output = output.view(batch_size, seq_len, hidden_size)

        # EC has perfect load balance by design, no auxiliary losses needed
        return output, None, None


class TokenChoiceMoELayer(nn.Module):
    """
    Token Choice (TC) MoE layer - the default for OLMoE.

    Each token selects top-k experts. This is the routing used in OLMoE-1B-7B.
    Requires load balancing loss to prevent expert collapse.

    This is the same as OLMoEMoELayer in model.py, included here for comparison.
    """

    def __init__(self, config):
        super().__init__()
        # Delegate to the main MoE implementation
        from src.model import OLMoEMoELayer
        self._moe = OLMoEMoELayer(config)

    def forward(self, hidden_states):
        return self._moe(hidden_states)


def compare_ec_vs_tc_config(
    base_config,
    num_experts: int = 8,
    num_experts_per_tok_tc: int = 2,
    capacity_factor_ec: float = 2.0,
):
    """
    Create configurations for EC vs TC comparison experiment.

    From Figure 7 in the paper:
    - Both models have 8-expert MoE in every 2nd layer
    - TC: 2 experts activated per token
    - EC: capacity factor = 2 (each expert processes 2x average tokens)
    - Both use same number of active parameters

    Returns:
        (tc_config, ec_config)
    """
    import copy

    tc_config = copy.deepcopy(base_config)
    tc_config.num_experts = num_experts
    tc_config.num_experts_per_tok = num_experts_per_tok_tc
    tc_config.use_load_balancing_loss = True  # TC needs load balancing

    ec_config = copy.deepcopy(base_config)
    ec_config.num_experts = num_experts
    ec_config.num_experts_per_tok = num_experts_per_tok_tc  # Same active params
    ec_config.use_load_balancing_loss = False  # EC doesn't need load balancing

    return tc_config, ec_config
