# model.py

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoConfig, AutoModel
from typing import Dict, Any
from gating_mechanisms import GatedAttention

class TransformerModel(nn.Module):
    """
    Transformer model with integrated gated attention mechanisms.
    Extends a base Transformer architecture using Hugging Face's library, with custom
    attention layers featuring gated mechanisms at configurable positions.
    """

    def __init__(self, base_model: str, num_heads: int, gating_config: Dict[str, Any]):
        """
        Initialize the Transformer model with gated attention.
        
        Args:
            base_model (str): Pre-trained or custom backbone model for initialization.
            num_heads (int): Number of attention heads for multi-head attention.
            gating_config (dict): Configuration for applying gating 
                                  (positions, type, granularity, activation function).
        """
        super(TransformerModel, self).__init__()
        
        # Load base configuration and model
        config = AutoConfig.from_pretrained(base_model)
        self.backbone = AutoModel.from_pretrained(base_model, config=config)
        
        # Configuration: Attention heads, hidden size, gating settings
        self.num_heads = num_heads
        self.hidden_size = config.hidden_size  # Assumes the hidden size comes from the base model
        self.gating_config = gating_config
        self.gating_positions = gating_config.get("positions", ["G1", "G2"])  # Default to G1 and G2
        self.gating_server = GatedAttention(gating_config)  # GatedAttention module

        # Attention projection layers for Query, Key, and Value
        self.query_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.key_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.value_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Dense output projection layer
        self.output_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Shared parameters initialization
        self._initialize_parameters()

    def _initialize_parameters(self):
        """
        Initialize model and gating-specific parameters.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward_attention(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """
        Compute multi-head attention with gated mechanisms.
        
        Args:
            query (Tensor): Query projection from input data.
            key (Tensor): Key projection from input data.
            value (Tensor): Value projection from input data.
        
        Returns:
            Tensor: Attention output, with gating applied at configured positions.
        """
        # Scale factors
        scaling_factor = key.size(-1) ** 0.5
        
        # Compute Scaled Dot Product Attention (SDPA)
        scores = torch.matmul(query, key.transpose(-2, -1)) / scaling_factor
        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, value)
        
        # Apply gating at 'G1' position (post-SDPA)
        if "G1" in self.gating_positions:
            attention_output = self.gating_server.apply_gating(
                attention_output, position="G1", granularity=self.gating_config["granularity"]
            )
        
        return attention_output

    def apply_gating(self, x: Tensor, position: str) -> Tensor:
        """
        Apply gating to the input tensor at the specified position.
        
        Args:
            x (Tensor): Input tensor at the gating stage.
            position (str): Gating application stage (e.g., 'G1', 'G2').
        
        Returns:
            Tensor: Output tensor after gating is applied.
        """
        gated_output = self.gating_server.apply_gating(
            x, position=position, granularity=self.gating_config["granularity"]
        )
        return gated_output

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the Transformer model with gated attention.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: Output tensor after passing through transformer layers and gating.
        """
        # Query, Key, Value projections
        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)
        
        # Apply gating at 'G2' (Value projection)
        if "G2" in self.gating_positions:
            value = self.apply_gating(value, position="G2")
        
        # Compute attention
        attention_output = self.forward_attention(query, key, value)
        
        # Concatenate attention outputs across heads and project
        logits = self.output_proj(attention_output)
        
        return logits
