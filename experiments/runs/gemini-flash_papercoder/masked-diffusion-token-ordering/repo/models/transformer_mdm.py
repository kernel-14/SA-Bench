import torch
from torch import nn
import math
import logging
from typing import Any, Dict, Optional

# Placeholder for Config to avoid circular imports.
# In main.py, the actual Config object will be imported.
# For this file's standalone integrity and type hinting, a placeholder is used.
class _ConfigPlaceholder:
    """
    A placeholder for the Config class. This ensures type hinting and method
    signatures are correctly defined without creating a direct import dependency
    that might lead to circular imports in a larger project structure.
    """
    def get(self, key: str, default: Any = None) -> Any:
        """Retrieves a configuration value from the underlying config dictionary."""
        # This method should ideally not be called on the placeholder itself,
        # but is needed for subclasses. If it's called on the placeholder, it
        # indicates an issue in the import chain or usage pattern.
        raise NotImplementedError("This is a placeholder for the Config object. "
                                  "Its 'get' method should not be called directly from here. "
                                  "Ensure the actual Config object is passed and used.")

# Re-assign for type hinting within this module.
# In the actual project, this would be: `from config import Config`
Config = _ConfigPlaceholder

# Import BaseMDMModel - this is a direct dependency
from models.base_mdm import BaseMDMModel

# Get logger instance. The logger is set up in utils/logger.py and retrieved here.
logger = logging.getLogger("MDM_Project_Logger")


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Implements fixed sinusoidal positional embeddings as described in "Attention Is All You Need".
    This module generates and stores sinusoidal position encodings.
    """
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        """
        Initializes the SinusoidalPositionalEmbedding module.

        Args:
            d_model (int): The dimensionality of the input embeddings.
            max_len (int): The maximum sequence length this positional embedding will support.
                           Positions beyond this will reuse the last embedding or require padding.
        """
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 'pe' has shape (max_len, d_model).
        # Register as a buffer: part of the module's state but not a trainable parameter.
        # No batch dimension added here; it's handled in forward if needed.
        self.register_buffer('pe', pe)

        logger.info(f"SinusoidalPositionalEmbedding initialized with d_model={d_model}, max_len={max_len}")

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Applies the sinusoidal positional embeddings for the given sequence length.

        Args:
            positions (torch.Tensor): A 1D tensor of shape (sequence_length,) representing
                                      the token indices (e.g., `torch.arange(seq_len)`).

        Returns:
            torch.Tensor: The positional embeddings for the given sequence length,
                          of shape (sequence_length, d_model).
        """
        seq_len: int = positions.size(0)

        if seq_len > self.pe.size(0):
            logger.warning(f"Requested sequence length {seq_len} exceeds max_len {self.pe.size(0)} for "
                           "SinusoidalPositionalEmbedding. Truncating positional embeddings.")
            # If `seq_len` exceeds `max_len`, return up to `max_len` embeddings.
            # This implicitly assumes that longer sequences might be truncated or that
            # positional information beyond `max_len` is less critical.
            return self.pe[:self.pe.size(0), :]
        
        # Return the positional embeddings up to the requested sequence length.
        # Shape: (seq_len, d_model)
        return self.pe[:seq_len, :]


class TransformerMDM(BaseMDMModel):
    """
    Implements a Masked Diffusion Model (MDM) using a Transformer architecture
    as its denoising network. It takes a partially masked sequence and predicts
    logits for the original tokens.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the TransformerMDM model.

        Args:
            config (Config): The global configuration object.
        """
        super().__init__(config) # This call populates self.model_params

        # Retrieve model parameters from the resolved config in BaseMDMModel
        vocab_size: int = self.model_params['vocab_size']
        max_sequence_length: int = self.model_params['max_sequence_length']
        num_layers: int = self.model_params['num_layers']
        num_heads: int = self.model_params['num_heads']
        hidden_dim: int = self.model_params['hidden_dim']
        ff_dim: int = self.model_params['ff_dim']
        dropout: float = self.model_params['dropout']
        use_learnable_pos_embeddings: bool = self.model_params['use_learnable_pos_embeddings']

        logger.info(f"Initializing TransformerMDM with parameters: "
                    f"vocab_size={vocab_size}, max_sequence_length={max_sequence_length}, "
                    f"num_layers={num_layers}, num_heads={num_heads}, hidden_dim={hidden_dim}, "
                    f"ff_dim={ff_dim}, dropout={dropout}, "
                    f"use_learnable_pos_embeddings={use_learnable_pos_embeddings}")

        # Token Embedding Layer
        self.token_embedding: nn.Embedding = nn.Embedding(vocab_size, hidden_dim)

        # Positional Encoder
        if use_learnable_pos_embeddings:
            # nn.Embedding for learnable positional embeddings
            self.positional_encoder: nn.Module = nn.Embedding(max_sequence_length, hidden_dim)
            logger.info("Using learnable positional embeddings.")
        else:
            # Fixed sinusoidal positional embeddings
            self.positional_encoder: nn.Module = SinusoidalPositionalEmbedding(hidden_dim, max_len=max_sequence_length)
            logger.info("Using sinusoidal positional embeddings.")

        # Transformer Encoder Blocks
        encoder_layer: nn.TransformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True # Input tensors will be (batch_size, sequence_length, hidden_dim)
        )
        self.transformer_blocks: nn.TransformerEncoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output Layer: maps hidden states to vocabulary logits
        self.output_layer: nn.Linear = nn.Linear(hidden_dim, vocab_size)

        logger.info("TransformerMDM initialization complete.")

    def forward(self, x_t: torch.Tensor, masked_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Performs the forward pass of the TransformerMDM.

        Args:
            x_t (torch.Tensor): The input sequence at a particular diffusion step `t`.
                                Shape: (batch_size, sequence_length). Contains token IDs,
                                including a mask token ID.
            masked_positions (Optional[torch.Tensor]): A boolean tensor (or indices)
                                representing the positions that were masked in `x_t`.
                                This is not used directly within the TransformerMDM's
                                forward pass as the model predicts for all positions.
                                It's passed for consistency with the BaseMDMModel
                                interface and can be used by external modules
                                (e.g., Trainer for loss calculation or Inferrer
                                for selecting tokens to unmask).

        Returns:
            torch.Tensor: Logits for the vocabulary over all positions.
                          Shape: (batch_size, sequence_length, vocab_size).
                          Each entry `[b, l, v]` is the unnormalized log-probability
                          of token `v` being the original token at position `l`
                          for batch item `b`.
        """
        # Ensure input is on the correct device
        x_t = x_t.to(self.token_embedding.weight.device)

        batch_size: int
        seq_len: int
        batch_size, seq_len = x_t.size()

        # 1. Token Embeddings
        token_embeddings: torch.Tensor = self.token_embedding(x_t) # (batch_size, seq_len, hidden_dim)

        # 2. Positional Embeddings
        position_ids: torch.Tensor = torch.arange(0, seq_len, dtype=torch.long, device=x_t.device)
        
        # positional_embeddings will have shape (seq_len, hidden_dim)
        position_embeddings: torch.Tensor = self.positional_encoder(position_ids)

        # Expand position_embeddings to (1, seq_len, hidden_dim) for broadcasting
        # during addition with token_embeddings (batch_size, seq_len, hidden_dim)
        position_embeddings = position_embeddings.unsqueeze(0) 

        # 3. Combine Embeddings
        input_embeds: torch.Tensor = token_embeddings + position_embeddings # (batch_size, seq_len, hidden_dim)

        # 4. Transformer Forward Pass
        # No `src_key_padding_mask` or `mask` arguments are typically passed
        # for a BERT-like encoder that computes representations for all tokens
        # without causal masking. Padding masks could be added here if pad tokens
        # are explicitly handled and should not contribute to attention.
        transformer_output: torch.Tensor = self.transformer_blocks(input_embeds) # (batch_size, seq_len, hidden_dim)

        # 5. Output Logits
        logits: torch.Tensor = self.output_layer(transformer_output) # (batch_size, seq_len, vocab_size)

        return logits

