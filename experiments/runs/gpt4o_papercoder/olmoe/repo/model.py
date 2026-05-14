## model.py
import torch
import torch.nn as nn
from typing import Dict, Tensor, Any
import numpy as np


class Model(nn.Module):
    """
    Implements the Mixture-of-Experts (MoE)-based Transformer architecture described in the paper.
    """

    def __init__(self, architecture: str, params: Dict[str, Any]) -> None:
        """
        Initializes the model based on the configuration parameters provided.
        
        Args:
            architecture (str): Type of model architecture (should be 'decoder_only').
            params (Dict[str, Any]): Configuration dictionary containing hyperparameters.
        """
        super(Model, self).__init__()
        
        # Validate architecture type
        assert architecture == "decoder_only", "Only 'decoder_only' architecture is supported."

        # Load parameters from the configuration file
        self.hidden_size = params["hidden_size"]
        self.num_layers = params["num_layers"]
        self.ffn_type = params["ffn_type"]
        self.num_experts = params["num_experts"]
        self.active_experts = params["active_experts"]
        self.ff_dim_per_expert = params["ff_dim_per_expert"]
        self.vocab_size = params["vocab_size"]
        self.attention_heads = params["attention_heads"]
        self.max_sequence_length = params["max_sequence_length"]
        self.norm_type = params["norm_type"]
        self.rotary_positional_embeddings = params["rotary_positional_embeddings"]
        self.router_z_loss_weight = params["router_z_loss_weight"]
        self.load_balancing_loss_weight = params["load_balancing_loss_weight"]

        # Initialize Transformer layers
        self.embedding_layer = nn.Embedding(self.vocab_size, self.hidden_size)
        self.positional_encoding = self._initialize_rotary_embeddings(self.max_sequence_length)

        # Define transformer layers with Mixture-of-Experts feedforward
        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_size=self.hidden_size,
                attention_heads=self.attention_heads,
                ff_dim=self.ff_dim_per_expert,
                num_experts=self.num_experts,
                active_experts=self.active_experts,
                norm_type=self.norm_type
            ) for _ in range(self.num_layers)
        ])

        # Final layer normalization
        self.final_norm = RMSNorm(self.hidden_size) if self.norm_type == "rms_norm" else nn.LayerNorm(self.hidden_size)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Processes input tensor through the transformer architecture.

        Args:
            inputs (Tensor): Input tensor of shape `(batch_size, sequence_length)`.

        Returns:
            Tensor: Output tensor of shape `(batch_size, sequence_length, hidden_size)`.
        """
        # Embed inputs
        embedded_inputs = self.embedding_layer(inputs)  # Shape: (batch_size, sequence_length, hidden_size)
        embedded_inputs += self.positional_encoding  # Add positional embeddings

        # Pass through transformer layers
        output = embedded_inputs
        for layer in self.layers:
            output = layer(output)
        
        # Apply final normalization
        return self.final_norm(output)

    def save_checkpoint(self, path: str) -> None:
        """
        Saves model parameters to a specified path.

        Args:
            path (str): Path to save the checkpoint.
        """
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path: str) -> None:
        """
        Loads model parameters from a saved checkpoint.

        Args:
            path (str): Path to load the checkpoint from.
        """
        try:
            self.load_state_dict(torch.load(path))
        except Exception as e:
            raise ValueError(f"Failed to load checkpoint from {path}. Error: {e}")

    def compute_auxiliary_losses(self, router_logits: Tensor, routing_probs: Tensor) -> Dict[str, Tensor]:
        """
        Computes the auxiliary losses for the Mixture-of-Experts module.

        Args:
            router_logits (Tensor): Router layer logits of shape `(batch_size, num_tokens, num_experts)`.
            routing_probs (Tensor): Routing probabilities of shape `(batch_size, num_tokens, num_experts)`.

        Returns:
            Dict[str, Tensor]: Auxiliary losses including load balancing loss and router z-loss.
        """
        # Compute Load-Balancing Loss (L_LB)
        mean_routing_probs = torch.mean(routing_probs, dim=1)  # Average across tokens
        lb_loss = self.num_experts * torch.sum(mean_routing_probs ** 2)
        lb_loss *= self.load_balancing_loss_weight

        # Compute Router Z-Loss (L_RZ)
        z_loss = torch.mean(torch.log(torch.sum(torch.exp(router_logits), dim=-1)) ** 2)
        z_loss *= self.router_z_loss_weight

        return {"load_balancing_loss": lb_loss, "router_z_loss": z_loss}

    def _initialize_rotary_embeddings(self, max_seq_len: int) -> Tensor:
        """
        Initialize rotary position embeddings.

        Args:
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tensor: Rotary positional embeddings of shape `(sequence_length, hidden_size)`.
        """
        if not self.rotary_positional_embeddings:
            return torch.zeros(max_seq_len, self.hidden_size)

        theta = 10_000.0
        positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.hidden_size, 2, dtype=torch.float32) * (-np.log(theta) / self.hidden_size))
        sinusoidal_embeddings = torch.zeros(max_seq_len, self.hidden_size)
        sinusoidal_embeddings[:, 0::2] = torch.sin(positions * div_term)
        sinusoidal_embeddings[:, 1::2] = torch.cos(positions * div_term)
        return sinusoidal_embeddings


class TransformerLayer(nn.Module):
    """
    Implements a single Transformer layer with Mixture-of-Experts integration.
    """

    def __init__(self, hidden_size: int, attention_heads: int, ff_dim: int, num_experts: int, active_experts: int, norm_type: str) -> None:
        """
        Initializes the Transformer layer.

        Args:
            hidden_size (int): Size of hidden representations.
            attention_heads (int): Number of attention heads.
            ff_dim (int): Dimension of feedforward layers per expert.
            num_experts (int): Total number of experts in the MoE module.
            active_experts (int): Number of experts activated per token.
            norm_type (str): Type of normalization technique ('rms_norm' or 'layer_norm').
        """
        super(TransformerLayer, self).__init__()

        # Self-attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=attention_heads, batch_first=True)

        # Mixture-of-Experts feedforward layer
        self.moe_ff_layer = MixtureOfExperts(num_experts, active_experts, ff_dim, hidden_size)

        # Layer normalization
        self.norm1 = RMSNorm(hidden_size) if norm_type == "rms_norm" else nn.LayerNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size) if norm_type == "rms_norm" else nn.LayerNorm(hidden_size)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Pass the inputs through the transformer layer.
        
        Args:
            inputs (Tensor): Input tensor of shape `(batch_size, sequence_length, hidden_size)`.
        
        Returns:
            Tensor: Output tensor of shape `(batch_size, sequence_length, hidden_size)`.
        """
        # Self-attention
        attention_output = self.attention(inputs, inputs, inputs)[0]
        attention_output = self.norm1(attention_output + inputs)

        # MoE feedforward
        ff_output = self.moe_ff_layer(attention_output)
        return self.norm2(ff_output + attention_output)


class MixtureOfExperts(nn.Module):
    """
    Implements the Mixture-of-Experts feedforward module.
    """

    def __init__(self, num_experts: int, active_experts: int, ff_dim: int, hidden_size: int) -> None:
        """
        Initializes the Mixture-of-Experts (MoE) module.

        Args:
            num_experts (int): Total number of experts available.
            active_experts (int): Number of activated experts.
            ff_dim (int): Feedforward dimension for each expert.
            hidden_size (int): Size of hidden representations.
        """
        super(MixtureOfExperts, self).__init__()
        self.router = nn.Linear(hidden_size, num_experts)
        self.experts = nn.ModuleList([nn.Linear(hidden_size, ff_dim) for _ in range(num_experts)])
        self.combine_layer = nn.Linear(active_experts * ff_dim, hidden_size)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Forward pass through the MoE.

        Args:
            inputs (Tensor): Input tensor of shape `(batch_size, sequence_length, hidden_size)`.

        Returns:
            Tensor: Output tensor of shape `(batch_size, sequence_length, hidden_size)`.
        """
        # Router logits -> softmax for routing probabilities
        router_logits = self.router(inputs)
        routing_probs = torch.softmax(router_logits, dim=-1)

        # Select top-k experts
        topk_indices = torch.topk(routing_probs, k=self.num_experts, dim=-1).indices  # Top `active_experts` per token
        expert_outputs = [self.experts[idx](inputs) * routing_probs[:, :, idx] for idx in topk_indices]

        # Combine expert outputs
        combined_output = torch.cat(expert_outputs, dim=-1)
        return self.combine_layer(combined_output)


class RMSNorm(nn.Module):
    """
    Implements RMSNorm as described in the paper.
    """

    def __init__(self, hidden_size: int) -> None:
        """
        Initializes the RMS normalization layer.

        Args:
            hidden_size (int): Size of hidden representations.
        """
        super(RMSNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(hidden_size))
        self.eps = 1e-6

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Normalize the inputs using RMS normalization.

        Args:
            inputs (Tensor): Input tensor of shape `(batch_size, sequence_length, hidden_size)`.

        Returns:
            Tensor: Normalized tensor of shape `(batch_size, sequence_length, hidden_size)`.
        """
        norm = inputs.norm(dim=-1, keepdim=True) / np.sqrt(inputs.size(-1))
        return self.scale * inputs / (norm + self.eps)
