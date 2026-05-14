"""
This module defines the Mixture-of-Experts (MoE) layer for the OLMoE model.
It integrates a Router to select experts and multiple ExpertFFN modules for computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Callable, Optional

# Assuming Router is in the same 'model' package
from model.router import Router

def _get_activation_fn(activation_function_name: str) -> Callable[..., torch.Tensor]:
    """
    Retrieves the activation function based on its string name.

    Args:
        activation_function_name: The name of the activation function (e.g., "SwigGLU").

    Returns:
        A callable activation function.
    """
    if activation_function_name.lower() == "swigglu":
        # SwiGLU is (input * F.sigmoid(input)) * gelu_fn(input) in some contexts,
        # but often implemented as a gate mechanism: SwiGLU(x) = (xW_g + b_g) * activation(xW_v + b_v)
        # The paper refers to SwigGLU as an activation, so we interpret the FFN as:
        # FFN(x) = down_proj(SwiGLU_gate_proj(x) * SwiGLU_up_proj(x))
        # The common implementation for SwiGLU in FFN is `silu(gate_proj(x)) * up_proj(x)`
        # This implementation reflects the structure `gate * up`
        return F.silu
    elif activation_function_name.lower() == "relu":
        return F.relu
    elif activation_function_name.lower() == "gelu":
        return F.gelu
    else:
        raise ValueError(f"Unsupported activation function: {activation_function_name}")


class ExpertFFN(nn.Module):
    """
    A single Feed-Forward Network (FFN) expert module.
    It uses a SwiGLU-like activation structure.
    """
    def __init__(
        self,
        d_model: int,
        ffn_dim_expert: int,
        activation_function: str = "SwigGLU",
    ):
        """
        Initializes an ExpertFFN module.

        Args:
            d_model: The dimensionality of the input and output hidden states.
            ffn_dim_expert: The intermediate dimension for this expert's FFN.
            activation_function: The name of the activation function to use.
        """
        super().__init__()
        self.gate_proj = nn.Linear(d_model, ffn_dim_expert, bias=False)
        self.up_proj = nn.Linear(d_model, ffn_dim_expert, bias=False)
        self.down_proj = nn.Linear(ffn_dim_expert, d_model, bias=False)
        self.act_fn = _get_activation_fn(activation_function)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the ExpertFFN.

        Args:
            hidden_states (torch.Tensor): Input tensor. Shape: `[..., d_model]`.

        Returns:
            torch.Tensor: Output tensor after FFN computation. Shape: `[..., d_model]`.
        """
        gate = self.act_fn(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        hidden_states = gate * up
        hidden_states = self.down_proj(hidden_states)
        return hidden_states


class MoELayer(nn.Module):
    """
    Mixture-of-Experts (MoE) layer that replaces the FFN in a Transformer block.

    It uses a Router to select a subset of experts for each token and
    combines their outputs based on routing weights.
    """
    def __init__(
        self,
        d_model: int = 2048,
        ffn_dim_expert: int = 1024,
        num_experts: int = 64,
        num_activated_experts: int = 8,
        activation_function: str = "SwigGLU",
    ):
        """
        Initializes the MoELayer.

        Args:
            d_model: The dimensionality of the input/output hidden states.
            ffn_dim_expert: The intermediate dimension for each individual expert's FFN.
            num_experts: The total number of experts in this MoE layer.
            num_activated_experts: The number of experts activated for each token (k).
            activation_function: The name of the activation function to use within ExpertFFN.
        """
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_activated_experts = num_activated_experts

        self.router = Router(d_model, num_experts, num_activated_experts)
        self.experts = nn.ModuleList(
            [
                ExpertFFN(d_model, ffn_dim_expert, activation_function)
                for _ in range(num_experts)
            ]
        )

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass of the MoE layer.

        Args:
            hidden_states (torch.Tensor): Input tensor. Shape: `[batch_size, sequence_length, d_model]`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - combined_expert_output (torch.Tensor): The output of the MoE layer
                  after expert computation and weighted summation.
                  Shape: `[batch_size, sequence_length, d_model]`.
                - router_logits_all_experts (torch.Tensor): Raw logits from the router
                  for *all* experts. Used for Router Z-loss.
                  Shape: `[num_tokens, num_experts]`.
                - router_probs_all_experts (torch.Tensor): Softmax probabilities over
                  *all* experts. Used for Load Balancing Loss (for calculating P_i).
                  Shape: `[num_tokens, num_experts]`.
                - expert_mask_all_experts (torch.Tensor): A boolean mask indicating
                  for each token which of the `num_experts` were selected (for calculating f_i in Load Balancing Loss).
                  Shape: `[num_tokens, num_experts]`.
        """
        original_shape = hidden_states.shape
        batch_size, sequence_length, _ = original_shape

        # Reshape input to (num_tokens, d_model) for token-wise routing
        reshaped_hidden_states = hidden_states.view(-1, self.d_model)
        num_tokens = reshaped_hidden_states.shape[0]

        # Route tokens to experts
        # expert_weights_chosen_k: probabilities for the k chosen experts for each token (num_tokens, k)
        # expert_indices_chosen_k: indices of the k chosen experts for each token (num_tokens, k)
        # router_logits_all_experts: raw logits for all experts (num_tokens, num_experts)
        expert_weights_chosen_k, expert_indices_chosen_k, router_logits_all_experts = self.router(
            reshaped_hidden_states
        )
        
        # Calculate router probabilities for all experts for LBL
        router_probs_all_experts = F.softmax(router_logits_all_experts, dim=-1)

        # Prepare inputs for expert computation
        # Replicate hidden states k times for each token
        # Shape: (num_tokens * k, d_model)
        flat_expert_inputs = reshaped_hidden_states.repeat_interleave(self.num_activated_experts, dim=0)
        
        # Flatten chosen expert indices and weights
        # Shape: (num_tokens * k)
        flat_expert_indices = expert_indices_chosen_k.flatten()
        # Shape: (num_tokens * k, 1) - unsqueeze for element-wise multiplication later
        flat_expert_weights = expert_weights_chosen_k.flatten().unsqueeze(1)

        # Initialize a tensor to store outputs from experts
        # Shape: (num_tokens * k, d_model)
        expert_outputs_flat = torch.zeros_like(flat_expert_inputs, device=hidden_states.device)

        # Dispatch tokens to experts and collect outputs
        # This loop is for conceptual clarity. In production, optimized scatter/gather ops are used.
        for i in range(self.num_experts):
            # Create a mask for inputs destined for the current expert `i`
            expert_mask = (flat_expert_indices == i)
            
            if expert_mask.any():
                # Select inputs for expert `i`
                inputs_for_expert_i = flat_expert_inputs[expert_mask]
                
                # Process inputs through expert `i`
                outputs_from_expert_i = self.experts[i](inputs_for_expert_i)
                
                # Place the expert outputs back into the flattened output tensor
                expert_outputs_flat[expert_mask] = outputs_from_expert_i

        # Combine expert outputs by applying routing weights
        # Shape: (num_tokens * k, d_model)
        weighted_expert_outputs_flat = expert_outputs_flat * flat_expert_weights

        # Reshape and sum the weighted expert outputs for each token
        # Shape: (num_tokens, k, d_model) -> sum over k -> (num_tokens, d_model)
        combined_expert_output = weighted_expert_outputs_flat.view(
            num_tokens, self.num_activated_experts, self.d_model
        ).sum(dim=1)

        # Generate expert mask for Load Balancing Loss
        # Create a boolean mask indicating which experts were selected for each token
        # Shape: (num_tokens, num_experts)
        expert_mask_all_experts = torch.zeros(
            num_tokens, self.num_experts, dtype=torch.bool, device=hidden_states.device
        )
        # Use scatter_ to set True at the indices of chosen experts for each token
        expert_mask_all_experts.scatter_(
            1, expert_indices_chosen_k, True
        )

        # Reshape the final output back to the original batch and sequence dimensions
        # Shape: (batch_size, sequence_length, d_model)
        final_output = combined_expert_output.view(original_shape)

        return (
            final_output,
            router_logits_all_experts,
            router_probs_all_experts,
            expert_mask_all_experts,
        )

