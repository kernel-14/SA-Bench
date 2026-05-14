
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from layers import FeedForward

class Expert(nn.Module):
    """
    An Expert is simply a FeedForward network.
    """
    def __init__(self, dim: int, hidden_dim: int, dropout_rate: float, use_bias: bool = False):
        super().__init__()
        self.ffn = FeedForward(dim, hidden_dim, dropout_rate, use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)

class Router(nn.Module):
    """
    The router is a learned linear layer mapping from input logits to chosen k experts.
    It determines routing probabilities and selected expert indices.
    """
    def __init__(self, dim: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False) # Paper states 'learned linear layer' but not explicit about bias

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor to the router (batch_size, sequence_length, dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - gate_logits (torch.Tensor): Logits for each expert (batch_size, sequence_length, num_experts).
                - gate_probabilities (torch.Tensor): Softmax probabilities for each expert
                                                     (batch_size, sequence_length, num_experts).
        """
        gate_logits = self.gate(x)
        gate_probabilities = F.softmax(gate_logits, dim=-1)
        return gate_logits, gate_probabilities

class MoELayer(nn.Module):
    """
    Mixture-of-Experts (MoE) module consisting of N_E experts,
    from which a subset of k experts is activated for each input token.
    Uses dropless token choice routing.
    """
    def __init__(self,
                 dim: int,
                 ffn_dimension: int,
                 num_experts: int,
                 num_activated_experts: int,
                 dropout_rate: float,
                 use_bias: bool = False):
        super().__init__()
        self.dim = dim
        self.ffn_dimension = ffn_dimension
        self.num_experts = num_experts
        self.num_activated_experts = num_activated_experts
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias

        self.router = Router(dim, num_experts)
        self.experts = nn.ModuleList([
            Expert(dim, ffn_dimension, dropout_rate, use_bias) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor to the MoE layer (batch_size, sequence_length, dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - output (torch.Tensor): Output of the MoE layer (batch_size, sequence_length, dim).
                - router_logits (torch.Tensor): Raw logits from the router for Z-loss calculation.
                - expert_gate_probabilities (torch.Tensor): Softmax probabilities from the router for load balancing.
        """
        batch_size, seq_len, _ = x.shape

        router_logits, expert_gate_probabilities = self.router(x) # (B, S, N_E)

        # Select top-k experts
        # indices_of_top_k_experts: (B, S, k) - indices of the chosen experts
        # weights_of_top_k_experts: (B, S, k) - probabilities/weights for the chosen experts
        weights_of_top_k_experts, indices_of_top_k_experts = torch.topk(
            expert_gate_probabilities, self.num_activated_experts, dim=-1
        )

        # Normalize weights for the top-k experts to sum to 1
        # This is the softmax(r(x))_i part in the paper's MoE formula
        weights_of_top_k_experts = weights_of_top_k_experts / weights_of_top_k_experts.sum(dim=-1, keepdim=True)

        # Flatten inputs for experts (tokens become batch items for experts)
        # x_flat: (B * S, D)
        x_flat = x.view(-1, self.dim)
        
        # Initialize output tensor
        # output_tensor: (B * S, D)
        output_tensor = torch.zeros_like(x_flat)

        # Process each token
        for i in range(batch_size * seq_len):
            token_input = x_flat[i]
            # Get the top-k experts and their weights for the current token
            selected_expert_indices = indices_of_top_k_experts.view(-1, self.num_activated_experts)[i]
            selected_expert_weights = weights_of_top_k_experts.view(-1, self.num_activated_experts)[i]

            # Collect outputs from selected experts
            expert_outputs = []
            for j, expert_idx in enumerate(selected_expert_indices):
                expert_output = self.experts[expert_idx](token_input) # (D,)
                expert_outputs.append(expert_output * selected_expert_weights[j])
            
            # Sum the weighted outputs
            output_tensor[i] = torch.stack(expert_outputs).sum(dim=0)

        output = output_tensor.view(batch_size, seq_len, self.dim)

        return output, router_logits, expert_gate_probabilities

def compute_load_balancing_loss(expert_gate_probabilities: torch.Tensor, num_experts: int, num_activated_experts: int) -> torch.Tensor:
    """
    Computes the auxiliary load balancing loss (L_LB) as per Shazeer et al. [154].
    Penalizes unequal assignment to experts.
    L_LB = N_E * sum_i (f_i * P_i)
    where f_i is the fraction of tokens routed to expert E_i,
    and P_i is the total routing probability allocated to E_i.

    Args:
        expert_gate_probabilities (torch.Tensor): Softmax probabilities from the router
                                                  (batch_size, sequence_length, num_experts).
        num_experts (int): Total number of experts (N_E).
        num_activated_experts (int): Number of top-k experts activated per token (k).

    Returns:
        torch.Tensor: The scalar load balancing loss.
    """
    # Calculate fraction of tokens routed to each expert (f_i)
    # indicator_matrix: (B, S, N_E) where 1 if expert is among top-k, 0 otherwise
    _, top_k_indices = torch.topk(expert_gate_probabilities, num_activated_experts, dim=-1)
    
    # Create a one-hot tensor for selected experts
    # (B * S, N_E)
    flat_top_k_indices = top_k_indices.view(-1, num_activated_experts)
    
    # num_selected_tokens_per_expert: (N_E,) - count of how many tokens selected each expert
    num_selected_tokens_per_expert = torch.zeros(num_experts, device=expert_gate_probabilities.device, dtype=expert_gate_probabilities.dtype)
    for i in range(num_experts):
        num_selected_tokens_per_expert[i] = (flat_top_k_indices == i).sum().float()

    # f_i: fraction of tokens routed to expert E_i
    f_i = num_selected_tokens_per_expert / (expert_gate_probabilities.shape[0] * expert_gate_probabilities.shape[1]) # (N_E,)

    # Calculate total routing probability allocated to each expert (P_i)
    # P_i: (N_E,) - sum of probabilities for each expert across all tokens in the batch
    P_i = expert_gate_probabilities.sum(dim=(0,1)) # Sum across batch and sequence length

    # L_LB = N_E * sum(f_i * P_i)
    loss = num_experts * torch.sum(f_i * P_i)
    return loss

def compute_router_z_loss(router_logits: torch.Tensor, num_experts: int) -> torch.Tensor:
    """
    Computes the auxiliary router Z-loss (L_RZ) as per Zoph et al. [221].
    Penalizes large logits coming into the gating network to improve stability.
    L_RZ(x) = (1/B) * sum_i (log sum_j (exp(x_j^(i))))^2
    This is equivalent to log-softmax(x) - log-sum-exp(x) which simplifies to sum_batch (log(sum_experts(exp(logits))))^2 / B

    Args:
        router_logits (torch.Tensor): Raw logits from the router
                                      (batch_size, sequence_length, num_experts).
        num_experts (int): Total number of experts (N_E).

    Returns:
        torch.Tensor: The scalar router Z-loss.
    """
    # log_sum_exp_logits: (B, S)
    log_sum_exp_logits = torch.logsumexp(router_logits, dim=-1)
    # router_z_loss: scalar
    loss = torch.sum(log_sum_exp_logits ** 2) / (router_logits.shape[0] * router_logits.shape[1])
    return loss
