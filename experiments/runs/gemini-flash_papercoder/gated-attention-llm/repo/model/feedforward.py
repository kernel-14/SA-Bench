import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Any, List

# Import Config and get_activation_fn from utils
try:
    from config import Config
    from utils import get_activation_fn
except ImportError:
    # Fallback for testing or if imports are structured differently
    print("Warning: Could not import Config or get_activation_fn. Using dummy classes/functions.")

    class Config:  # Dummy Config for isolated testing
        def __init__(self):
            self.model = self  # Self-reference for model config
            self.d_model = 2048
            self.d_ff = 8192
            self.ffn_activation = "gelu"
            self.moe = self # Self-reference for moe config (for MoEFeedForward init)
            self.moe_num_experts = 8
            self.moe_top_k_experts = 2
            self.moe_router_bias = False
            self.moe_z_loss_coeff = 0.001
            self.moe_load_balancing_loss_coeff = 0.01

    def get_activation_fn(name: str) -> Any:  # Dummy get_activation_fn
        if name == "gelu":
            return nn.GELU()
        else:
            return lambda x: x


class FeedForward(nn.Module):
    """
    Implements a standard two-layer Feedforward Network (FFN) with an activation function.
    This FFN can also serve as an expert within an MoE layer.
    """

    def __init__(self, config: Config):
        """
        Initializes the FeedForward module.

        Args:
            config: Configuration object containing model hyperparameters like d_model, d_ff,
                    and the FFN activation function.
        """
        super().__init__()
        self.d_model: int = config.model.d_model
        self.d_ff: int = config.model.d_ff
        self.ffn_activation: nn.Module = get_activation_fn(config.model.ffn_activation)

        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            self.ffn_activation,
            nn.Linear(self.d_ff, self.d_model),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass through the FeedForward Network.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        return self.mlp(hidden_states)


class MoEFeedForward(nn.Module):
    """
    Implements a Mixture-of-Experts (MoE) Feedforward Network.
    This module routes tokens to a subset of experts, computes their outputs,
    and combines them, while also calculating MoE-specific regularization losses
    (load balancing and Z-loss).
    """

    def __init__(self, config: Config):
        """
        Initializes the MoEFeedForward module.

        Args:
            config: Configuration object containing MoE specific hyperparameters
                    (num_experts, top_k_experts, router_bias, z_loss_coeff, load_balancing_loss_coeff)
                    and FFN expert hyperparameters.
        """
        super().__init__()
        if config.model.type != "moe":
            raise ValueError("MoEFeedForward initialized but model type is not 'moe' in config.")

        self.d_model: int = config.model.d_model
        self.num_experts: int = config.moe_num_experts
        self.top_k_experts: int = config.moe_top_k_experts
        self.router_bias: bool = config.moe_router_bias
        self.z_loss_coeff: float = config.moe_z_loss_coeff
        self.load_balancing_loss_coeff: float = config.moe_load_balancing_loss_coeff

        if self.top_k_experts > self.num_experts:
            raise ValueError(
                f"top_k_experts ({self.top_k_experts}) cannot be greater than num_experts ({self.num_experts})."
            )

        # Router: projects input to scores for each expert
        self.router = nn.Linear(self.d_model, self.num_experts, bias=self.router_bias)

        # Experts: a list of FeedForward modules
        self.experts = nn.ModuleList([FeedForward(config) for _ in range(self.num_experts)])

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass for the MoE Feedforward Network.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            A tuple containing:
                - moe_output: The combined output of the experts (batch_size, seq_len, d_model).
                - moe_loss: The total MoE regularization loss (scalar tensor).
        """
        original_shape = hidden_states.shape  # (B, S, D)
        
        # Flatten (batch_size, seq_len) dimensions to process tokens independently
        # flat_hidden_states: (num_tokens, d_model), where num_tokens = B * S
        flat_hidden_states = hidden_states.view(-1, self.d_model)
        num_tokens = flat_hidden_states.shape[0]

        # 1. Router Computation
        # router_logits: (num_tokens, num_experts)
        router_logits = self.router(flat_hidden_states)
        
        # router_weights: (num_tokens, num_experts) - probabilities for each expert
        router_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)

        # 2. Top-k Expert Selection
        # top_k_weights: (num_tokens, top_k_experts) - weights for selected experts
        # top_k_indices: (num_tokens, top_k_experts) - indices of selected experts
        top_k_weights, top_k_indices = torch.topk(router_weights, self.top_k_experts, dim=-1)

        # Re-normalize top-k weights (sum to 1 for each token)
        # Add a small epsilon to avoid division by zero if all top_k_weights are zero
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # 3. MoE Loss Calculation
        moe_loss_components: List[torch.Tensor] = []

        # Z-loss (Encourages router logits to be small, preventing saturation)
        # L_z = mean(router_logits^2)
        z_loss = self.z_loss_coeff * torch.mean(router_logits ** 2)
        moe_loss_components.append(z_loss)

        # Load Balancing Loss (LBL)
        # As per design: num_experts * torch.sum(expert_load.mean(dim=0) * prob_per_expert)
        # expert_load: (num_tokens, num_experts) - binary mask indicating if an expert was selected for a token
        #   F.one_hot(top_k_indices, num_classes=self.num_experts) -> (num_tokens, top_k_experts, num_experts)
        #   .sum(dim=1) -> (num_tokens, num_experts)
        expert_load_mask = F.one_hot(top_k_indices, num_classes=self.num_experts).sum(dim=1).float()
        
        # P(expert_i selected for *any* token) = average over tokens if expert_i was chosen.
        avg_expert_selection_per_token = expert_load_mask.mean(dim=0) # (num_experts,)

        # Sum of probabilities for each expert across all tokens (total routing probability sent to expert)
        sum_router_prob_per_expert = router_weights.sum(dim=0) # (num_experts,)

        # LBL formula from design
        load_balancing_loss = self.load_balancing_loss_coeff * (
            self.num_experts * torch.sum(avg_expert_selection_per_token * sum_router_prob_per_expert)
        )
        moe_loss_components.append(load_balancing_loss)

        moe_loss = torch.sum(torch.stack(moe_loss_components))

        # 4. Expert Dispatch and Combination
        # Initialize buffer for combined expert outputs
        moe_output_flat = torch.zeros_like(flat_hidden_states, dtype=flat_hidden_states.dtype)

        # Iterate through each token to compute its weighted expert output
        # This approach explicitly handles dispatching to specific experts and combining outputs
        # from multiple experts for a single token, considering their respective weights.
        
        # Prepare for efficient expert computation by grouping tokens by experts
        # A list of lists, where each inner list contains indices of tokens routed to that expert
        expert_dispatch_tokens: List[List[int]] = [[] for _ in range(self.num_experts)]
        expert_dispatch_weights: List[List[torch.Tensor]] = [[] for _ in range(self.num_experts)]

        # Fill dispatch lists
        for token_idx in range(num_tokens):
            for k_idx in range(self.top_k_experts):
                expert_id = top_k_indices[token_idx, k_idx].item()
                weight = top_k_weights[token_idx, k_idx]
                
                expert_dispatch_tokens[expert_id].append(token_idx)
                expert_dispatch_weights[expert_id].append(weight)

        # Compute expert outputs and combine
        for expert_id, expert_module in enumerate(self.experts):
            if not expert_dispatch_tokens[expert_id]: # If no tokens routed to this expert
                continue
            
            # Gather tokens routed to this expert
            tokens_to_process_indices = torch.tensor(
                expert_dispatch_tokens[expert_id], 
                device=flat_hidden_states.device, 
                dtype=torch.long
            )
            tokens_for_expert = flat_hidden_states[tokens_to_process_indices]

            # Compute output from this expert
            expert_output = expert_module(tokens_for_expert) # (num_tokens_for_expert, d_model)

            # Gather weights for this expert for these specific tokens
            weights_for_expert = torch.stack(expert_dispatch_weights[expert_id]).to(flat_hidden_states.device).unsqueeze(-1)
            
            # Add weighted expert output to the combined output buffer
            moe_output_flat.index_add_(
                0, 
                tokens_to_process_indices, 
                expert_output * weights_for_expert
            )

        # Reshape the output back to original (batch_size, seq_len, d_model)
        moe_output = moe_output_flat.view(original_shape)

        return moe_output, moe_loss

