import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

# Import Config, GatedAttention, FeedForward, and MoEFeedForward
try:
    from config import Config
    from model.attention import GatedAttention
    from model.feedforward import FeedForward, MoEFeedForward
except ImportError:
    # Fallback for testing or if imports are structured differently
    print("Warning: Could not import Config, GatedAttention, FeedForward, or MoEFeedForward. Using dummy classes/functions.")

    class Config:  # Dummy Config for isolated testing
        def __init__(self):
            self.model = self  # Self-reference for model config
            self.d_model = 2048
            self.type = "dense" # Default for dummy
            self.ffn_activation = "gelu" # Dummy
            self.q_heads = 32 # Dummy
            self.kv_heads = 4 # Dummy
            self.head_dim = 64 # Dummy
            self.attn_dropout = 0.1 # Dummy
            self.rope_base = 10000.0 # Dummy
            self.d_ff = 8192 # Dummy
            self.gating_enabled = False # Dummy

            # MoE specific dummy
            self.moe_num_experts = 8
            self.moe_top_k_experts = 2
            self.moe_router_bias = False
            self.moe_z_loss_coeff = 0.001
            self.moe_load_balancing_loss_coeff = 0.01

    class GatedAttention(nn.Module): # Dummy GatedAttention
        def __init__(self, config: Config):
            super().__init__()
            self.proj = nn.Linear(config.model.d_model, config.model.d_model)
        def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            return self.proj(hidden_states)

    class FeedForward(nn.Module): # Dummy FeedForward
        def __init__(self, config: Config):
            super().__init__()
            self.mlp = nn.Linear(config.model.d_model, config.model.d_model) # Simplified
        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return self.mlp(hidden_states)

    class MoEFeedForward(nn.Module): # Dummy MoEFeedForward
        def __init__(self, config: Config):
            super().__init__()
            self.mlp = nn.Linear(config.model.d_model, config.model.d_model) # Simplified
        def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            return self.mlp(hidden_states), torch.tensor(0.0) # Dummy loss


class TransformerBlock(nn.Module):
    """
    A single Transformer block, composed of a GatedAttention sub-layer and a FeedForward (or MoEFeedForward)
    sub-layer, with pre-normalization and residual connections.
    """

    def __init__(self, config: Config):
        """
        Initializes a TransformerBlock.

        Args:
            config: Configuration object containing model hyperparameters.
        """
        super().__init__()
        self.config = config

        self.norm1 = nn.LayerNorm(config.model.d_model, eps=1e-5)
        self.attn = GatedAttention(config)

        self.norm2 = nn.LayerNorm(config.model.d_model, eps=1e-5)
        if config.model.type == "moe":
            self.ffn: Union[FeedForward, MoEFeedForward] = MoEFeedForward(config)
        else: # config.model.type == "dense"
            self.ffn = FeedForward(config)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Performs the forward pass through the TransformerBlock.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, d_model).
            attention_mask: Optional mask for attention scores (e.g., causal mask).

        Returns:
            A tuple containing:
                - The output hidden states of shape (batch_size, seq_len, d_model).
                - An optional MoE loss component (scalar tensor), or None if not an MoE block.
        """
        # First residual connection: Attention sub-layer
        residual_attn = hidden_states
        
        # Pre-normalization before attention
        normalized_hidden_states_attn = self.norm1(hidden_states)
        
        # Gated Attention sub-layer
        attn_output = self.attn(normalized_hidden_states_attn, attention_mask=attention_mask)
        
        # Add first residual
        hidden_states = residual_attn + attn_output

        # Second residual connection: FFN sub-layer
        residual_ffn = hidden_states
        
        # Pre-normalization before FFN
        normalized_hidden_states_ffn = self.norm2(hidden_states)
        
        # FFN sub-layer (could be MoE or standard)
        moe_loss_component: Optional[torch.Tensor] = None
        if isinstance(self.ffn, MoEFeedForward):
            ffn_output, moe_loss_component = self.ffn(normalized_hidden_states_ffn)
        else:
            ffn_output = self.ffn(normalized_hidden_states_ffn)
        
        # Add second residual
        hidden_states = residual_ffn + ffn_output

        return hidden_states, moe_loss_component

