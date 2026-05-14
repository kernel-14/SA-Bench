## model.py
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, Tuple
import math


class Model(nn.Module):
    """
    Implements the NGPT model with detailed architectural changes, including hypersphere normalization,
    scaling factors, eigen learning rates, and hyperspherical weight representations across layers.
    """

    def __init__(self, params: Dict):
        """
        Initialize the NGPT model based on the provided parameters (from the configuration file).

        Args:
            params (Dict): Configuration parameters from config.yaml.
        """
        super(Model, self).__init__()
        # Extract configuration values
        self.d_model = params.get("d_model", 1024)
        self.n_layers = params.get("n_layers", 24)
        self.n_heads = params.get("n_heads", 16)
        self.d_mlp = params.get("d_mlp", 4 * self.d_model)
        self.vocab_size = params.get("vocabulary_size", 32000)
        self.softmax_scaling_factor = params.get("softmax_scaling_factor", math.sqrt(self.d_model / self.n_heads))
        self.alpha_a_init = params.get("scaling_factors", {}).get("alpha_a_init", 0.05 / self.n_layers)
        self.alpha_m_init = params.get("scaling_factors", {}).get("alpha_m_init", 0.05 / self.n_layers)
        self.alpha_a_scale = params.get("scaling_factors", {}).get("alpha_a_scale", 1 / math.sqrt(self.d_model))
        self.alpha_m_scale = params.get("scaling_factors", {}).get("alpha_m_scale", 1 / math.sqrt(self.d_model))
        self.sqk_scale = params.get("scaling_factors", {}).get("sqk_scale", 1 / math.sqrt(self.d_model))
        self.s_z_scale = params.get("scaling_factors", {}).get("sz_scale", 1 / math.sqrt(self.d_model))

        # Initialize embeddings
        self.embedding_input = nn.Embedding(self.vocab_size, self.d_model)
        self.embedding_output = nn.Embedding(self.vocab_size, self.d_model)

        # Layer normalization is replaced by hyperspherical normalization
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            self._create_attention_block() for _ in range(self.n_layers)
        ])

        # MLP layers
        self.mlp_layers = nn.ModuleList([
            self._create_mlp_block() for _ in range(self.n_layers)
        ])

        # Learnable scaling factors
        self.alpha_a = nn.Parameter(torch.full((self.d_model,), self.alpha_a_init))
        self.alpha_m = nn.Parameter(torch.full((self.d_model,), self.alpha_m_init))
        self.s_z = nn.Parameter(torch.full((self.vocab_size,), self.s_z_scale))

        # Weight initialization
        self.initialize_weights()

    def initialize_weights(self) -> None:
        """
        Initializes the embeddings and weight matrices for the model.
        """
        # Initialize embeddings with normal distribution
        nn.init.normal_(self.embedding_input.weight, mean=0.0, std=1.0 / math.sqrt(self.d_model))
        nn.init.normal_(self.embedding_output.weight, mean=0.0, std=1.0 / math.sqrt(self.d_model))

        # Normalize embeddings
        self.embedding_input.weight.data = F.normalize(self.embedding_input.weight.data, dim=-1)
        self.embedding_output.weight.data = F.normalize(self.embedding_output.weight.data, dim=-1)

        # Initialize attention and MLP layer parameters using normal distributions
        for layer in self.attention_layers + self.mlp_layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.normal_(param, mean=0.0, std=1.0 / math.sqrt(self.d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model. Processes the input tensor through embedding, attention, 
        MLP layers, and logits scaling.

        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, sequence_length].

        Returns:
            torch.Tensor: Logits tensor with shape [batch_size, sequence_length, vocab_size].
        """
        # Input embeddings
        h = self.embedding_input(x)
        h = F.normalize(h, dim=-1)  # Ensure input embedding normalization

        # Process through layers
        for i, (attention_layer, mlp_layer) in enumerate(zip(self.attention_layers, self.mlp_layers)):
            # Attention block
            h_attention = attention_layer(h)
            h = F.normalize(h + self.alpha_a * (h_attention - h), dim=-1)

            # MLP block
            h_mlp = mlp_layer(h)
            h = F.normalize(h + self.alpha_m * (h_mlp - h), dim=-1)

        # Output logits
        logits = torch.matmul(h, self.embedding_output.weight.T)
        logits *= self.s_z  # Apply scaling factor
        return logits

    def normalize_parameters(self) -> None:
        """
        Normalizes all weight matrices and embeddings to lie on the unit hypersphere.
        Call this method post-training batch for stability.
        """
        with torch.no_grad():
            # Normalize embeddings
            self.embedding_input.weight.data = F.normalize(self.embedding_input.weight.data, dim=-1)
            self.embedding_output.weight.data = F.normalize(self.embedding_output.weight.data, dim=-1)

            # Normalize attention and MLP weights
            for attention_layer, mlp_layer in zip(self.attention_layers, self.mlp_layers):
                self._normalize_layer(attention_layer)
                self._normalize_layer(mlp_layer)

    @staticmethod
    def _normalize_layer(layer: nn.Module) -> None:
        """
        Normalizes all linear layers in a block to maintain unit norm.

        Args:
            layer (nn.Module): The layer (attention or MLP) to normalize.
        """
        for module in layer.modules():
            if isinstance(module, nn.Linear):
                module.weight.data = F.normalize(module.weight.data, dim=-1)

    def _create_attention_block(self) -> nn.Module:
        """
        Creates a single multi-head attention block.

        Returns:
            nn.Module: The attention block.
        """
        return nn.Sequential(
            nn.Linear(self.d_model, self.d_model, bias=False),  # W_q
            nn.Linear(self.d_model, self.d_model, bias=False),  # W_k
            nn.Linear(self.d_model, self.d_model, bias=False),  # W_v
            nn.Linear(self.d_model, self.d_model, bias=False),  # W_o
        )

    def _create_mlp_block(self) -> nn.Module:
        """
        Creates a single MLP block consisting of two linear transformations and SwiGLU activation.

        Returns:
            nn.Module: The MLP block.
        """
        return nn.Sequential(
            nn.Linear(self.d_model, self.d_mlp, bias=False),  # W_u
            nn.Linear(self.d_model, self.d_mlp, bias=False),  # W_v
            nn.GELU(),
            nn.Linear(self.d_mlp, self.d_model, bias=False)  # W_o_MLP
        )
