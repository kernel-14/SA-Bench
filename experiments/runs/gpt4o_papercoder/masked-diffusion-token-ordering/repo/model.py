## model.py

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from typing import Dict, Tuple


class Model:
    def __init__(self, params: Dict[str, int]):
        """
        Initializes the Masked Diffusion Model (MDM) with transformer-based architecture.

        Args:
            params (dict): Dictionary containing model hyperparameters, which include:
                - transformer_layers (int): Number of transformer layers.
                - hidden_dim (int): Dimension of hidden layers.
                - num_attention_heads (int): Number of attention heads per transformer layer.
                - dropout_rate (float): Dropout probability for regularization.
                - max_sequence_length (int): Maximum length of input/output sequences.
                - positional_encoding (str): Type of positional encoding ('learnable' or 'fixed').
        """
        self.params = params

        # Define transformer configuration
        config = BertConfig(
            hidden_size=params.get("hidden_dim", 768),
            num_hidden_layers=params.get("transformer_layers", 12),
            num_attention_heads=params.get("num_attention_heads", 12),
            intermediate_size=params.get("hidden_dim", 768) * 4,  # Commonly 4x hidden size
            hidden_dropout_prob=params.get("dropout_rate", 0.1),
            max_position_embeddings=params.get("max_sequence_length", 2048),
        )

        # Initialize transformer model
        self.transformer = BertModel(config)

        # Additional learnable positional encoding (if required)
        if params.get("positional_encoding", "learnable") == "learnable":
            self.positional_embeddings = nn.Embedding(
                params.get("max_sequence_length", 2048), config.hidden_size
            )
        else:
            self.positional_embeddings = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass on the input masked sequence and generates logits.

        Args:
            x (torch.Tensor): Input tensor of masked sequences (shape: [batch_size, seq_length]).

        Returns:
            torch.Tensor: Logits output by the model (shape: [batch_size, seq_length, vocab_size]).
        """
        batch_size, seq_length = x.shape

        # Validate input sequence length
        if seq_length > self.params.get("max_sequence_length", 2048):
            raise ValueError(f"Input sequence length {seq_length} exceeds maximum allowed length {self.params['max_sequence_length']}.")

        # Generate positional embeddings if 'learnable' encoding is enabled
        if self.positional_embeddings:
            position_ids = torch.arange(seq_length, device=x.device).expand(batch_size, seq_length)
            positional_encoding = self.positional_embeddings(position_ids)
        else:
            positional_encoding = None

        # Pass through transformer model (with optional positional encoding)
        transformer_output = self.transformer(input_ids=x, position_ids=positional_encoding, output_hidden_states=False)
        logits = transformer_output.last_hidden_state  # Shape: [batch_size, seq_length, hidden_dim]
        return logits

    def generate_initial_sequence(self, length: int) -> torch.Tensor:
        """
        Generates an initial fully masked sequence.

        Args:
            length (int): The desired sequence length.

        Returns:
            torch.Tensor: Fully masked sequence (shape: [1, length]).
        """
        if length > self.params.get("max_sequence_length", 2048):
            raise ValueError(f"Requested sequence length {length} exceeds the maximum allowed length {self.params['max_sequence_length']}.")

        # Generate sequence filled with mask tokens (value = 0)
        initial_sequence = torch.zeros((1, length), dtype=torch.long)
        return initial_sequence

    def perform_reverse_sampling(self, strategy: str, sampling_steps: int) -> torch.Tensor:
        """
        Performs reverse sampling to reconstruct a masked sequence using specified inference strategy.

        Args:
            strategy (str): The inference strategy ('vanilla', 'adaptive_top_probability', or 'adaptive_top_probability_margin').
            sampling_steps (int): Number of sampling steps to perform.

        Returns:
            torch.Tensor: Fully reconstructed sequence (shape: [1, seq_length]).
        """
        if strategy not in ["vanilla", "adaptive_top_probability", "adaptive_top_probability_margin"]:
            raise ValueError(f"Invalid strategy '{strategy}'. Supported strategies: 'vanilla', 'adaptive_top_probability', 'adaptive_top_probability_margin'.")

        # Initialize with fully masked sequence
        seq_length = self.params.get("max_sequence_length", 2048)
        reconstructed_sequence = self.generate_initial_sequence(seq_length)

        # Sampling loop
        for step in range(sampling_steps):
            logits = self.forward(reconstructed_sequence)  # Predict logits for masked sequence

            # Determine masked positions
            mask_positions = (reconstructed_sequence == 0).nonzero(as_tuple=True)
            if mask_positions[0].numel() == 0:  # If no masked tokens remain, terminate
                break

            # Apply inference strategy
            if strategy == "vanilla":
                # Randomly unmask positions
                random_sample_indices = torch.randperm(mask_positions[0].numel())[:10]  # Example: unmask 10 positions
                unmask_positions = mask_positions[0][random_sample_indices]

            elif strategy == "adaptive_top_probability":
                # Top Probability strategy: based on max logits
                max_probs = logits.softmax(dim=-1).max(dim=-1).values
                top_position_indices = torch.topk(max_probs[mask_positions], k=10).indices  # Unmask top 10 positions
                unmask_positions = mask_positions[0][top_position_indices]

            elif strategy == "adaptive_top_probability_margin":
                # Top Probability Margin strategy: based on margin between top-2 probabilities
                probs = logits.softmax(dim=-1)
                sorted_probs = probs.sort(dim=-1, descending=True).values
                margin = sorted_probs[:, 0] - sorted_probs[:, 1]  # Difference between top-2 probabilities
                margin_scores = margin[mask_positions]
                top_position_indices = torch.topk(margin_scores, k=10).indices  # Unmask top 10 positions
                unmask_positions = mask_positions[0][top_position_indices]

            # Update reconstructed sequence
            for pos in unmask_positions:
                token = torch.argmax(logits[pos]).item()  # Sample most probable token
                reconstructed_sequence[0, pos] = token

        # Return reconstructed sequence
        return reconstructed_sequence
