"""
This module defines the Router component for the Mixture-of-Experts (MoE) layer.

The Router is responsible for determining which experts will process each input token
by computing routing probabilities and selecting the top-k experts. It also provides
the necessary information for auxiliary losses like Load Balancing Loss and Router Z-loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Router(nn.Module):
    """
    Implements the routing mechanism for a Mixture-of-Experts (MoE) layer.

    The router learns to map input hidden states to a set of logit scores for each expert.
    It then uses these logits to determine routing probabilities and select the top-k
    experts for each token.

    Attributes:
        gate (nn.Linear): A linear layer that projects input hidden states to
                          expert logits.
        num_activated_experts (int): The number of experts (k) to activate for
                                     each input token.
        num_experts (int): The total number of available experts.
    """

    def __init__(
        self,
        d_model: int = 2048,  # Default from config.model.d_model
        num_experts: int = 64,  # Default from config.model.num_experts
        num_activated_experts: int = 8,  # Default from config.model.num_activated_experts
    ):
        """
        Initializes the Router module.

        Args:
            d_model: The dimensionality of the input hidden states.
            num_experts: The total number of experts in the MoE layer.
            num_activated_experts: The number of experts to activate for each token (k).
        """
        super().__init__()
        if not (0 < num_activated_experts <= num_experts):
            raise ValueError(
                f"num_activated_experts ({num_activated_experts}) must be "
                f"between 1 and num_experts ({num_experts})."
            )

        self.gate = nn.Linear(d_model, num_experts)
        self.num_activated_experts = num_activated_experts
        self.num_experts = num_experts

    def forward(
        self, hidden_states: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Performs the forward pass of the router.

        Args:
            hidden_states (torch.Tensor): Input tensor representing the hidden states
                                          of tokens. Shape: `[batch_size, sequence_length, d_model]`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - expert_weights (torch.Tensor): Routing probabilities for the
                  `num_activated_experts` selected experts for each token.
                  Shape: `[batch_size, sequence_length, num_activated_experts]`.
                - expert_indices (torch.Tensor): Indices (IDs) of the
                  `num_activated_experts` selected experts for each token.
                  Shape: `[batch_size, sequence_length, num_activated_experts]`.
                - router_logits (torch.Tensor): Raw logits produced by the gating
                  network for all experts. Used for Router Z-loss.
                  Shape: `[batch_size, sequence_length, num_experts]`.
        """
        # Compute router logits from hidden states
        # Shape: [batch_size, sequence_length, num_experts]
        router_logits = self.gate(hidden_states)

        # Compute routing probabilities using softmax
        # Shape: [batch_size, sequence_length, num_experts]
        router_probabilities = F.softmax(router_logits, dim=-1)

        # Select the top-k experts based on probabilities
        # expert_weights: probabilities of the selected experts
        # expert_indices: indices of the selected experts
        # Shapes: [batch_size, sequence_length, num_activated_experts]
        expert_weights, expert_indices = torch.topk(
            router_probabilities, k=self.num_activated_experts, dim=-1
        )

        return expert_weights, expert_indices, router_logits

