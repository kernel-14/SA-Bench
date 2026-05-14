## transformer.py

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional

class TransformerGenerator(nn.Module):
    """Autoregressive Transformer model for frequency-guided token generation."""

    def __init__(self, vocab_size: int = 680, depth: int = 16, head_dim: int = 64):
        """
        Initializes the TransformerGenerator model.

        Args:
            vocab_size (int): Vocabulary size (number of unique tokens). Default is 680.
            depth (int): Number of Transformer decoder layers. Default is 16.
            head_dim (int): Dimension of attention head. Default is 64.
        """
        super(TransformerGenerator, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = head_dim * 8  # Set embedding dimension proportional to head_dim
        self.depth = depth

        # Embedding layers for tokens and positions
        self.token_embedding = nn.Embedding(vocab_size, self.embed_dim)
        self.position_embedding = nn.Embedding(1000, self.embed_dim)  # Set max seq length to 1000

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=8,  # Number of attention heads fixed to 8
            dim_feedforward=4 * self.embed_dim,
            dropout=0.1,
            activation='relu'
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)

        # Token output head
        self.output_linear = nn.Linear(self.embed_dim, vocab_size)

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Forward pass through the Transformer.

        Args:
            tokens (Tensor): Input tensor of shape (batch_size, seq_len).

        Returns:
            Tensor: Predicted logits of shape (batch_size, seq_len, vocab_size).
        """
        # Step 1: Embed tokens and positions
        batch_size, seq_len = tokens.shape
        token_embeddings = self.token_embedding(tokens)  # Shape: (batch_size, seq_len, embed_dim)
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)  # Shape: (1, seq_len)
        position_embeddings = self.position_embedding(positions)  # Shape: (1, seq_len, embed_dim)

        inputs = token_embeddings + position_embeddings  # Combine token and position embeddings

        # Step 2: Apply Transformer Decoder
        # Transformer expects inputs with shape (seq_len, batch_size, embed_dim)
        inputs = inputs.permute(1, 0, 2)  # Transpose to adjust dimensions
        causal_mask = self._generate_causal_mask(seq_len, tokens.device)  # Generate attention mask
        outputs = self.transformer_decoder(inputs, inputs, tgt_mask=causal_mask)  # Shape: (seq_len, batch_size, embed_dim)

        # Step 3: Project Transformer outputs to logits
        logits = self.output_linear(outputs.permute(1, 0, 2))  # Shape: (batch_size, seq_len, vocab_size)

        return logits

    def generate(self, tokens: Tensor, num_steps: int, top_k: Optional[int] = None, cfg: Optional[float] = None) -> Tensor:
        """
        Autoregressively generates tokens band-by-band.

        Args:
            tokens (Tensor): Input tensor for initial frequency band, shape (batch_size, seq_len).
            num_steps (int): Number of autoregressive steps (frequency bands).
            top_k (Optional[int]): Top-k sampling during generation. Default is None.
            cfg (Optional[float]): Classifier-Free Guidance (CFG) value. Default is None.

        Returns:
            Tensor: Generated tokens of shape (batch_size, seq_len + num_steps).
        """
        batch_size, seq_len = tokens.shape
        generated_tokens = tokens.clone()

        for step in range(num_steps):
            # Step 1: Forward pass to get logits for the current step
            logits = self.forward(generated_tokens)  # Shape: (batch_size, seq_len + step, vocab_size)
            logits_step = logits[:, -1, :]  # Extract logits for the last token only

            # Step 2: Apply CFG if specified
            if cfg:
                # For simplicity, CFG assumes guidance logits are included in the input sequence
                logits_step = logits_step * cfg

            # Step 3: Perform top-k sampling if specified
            if top_k:
                logits_step = self._apply_top_k_sampling(logits_step, top_k)

            # Step 4: Sample next token from logits
            probabilities = F.softmax(logits_step, dim=-1)  # Convert logits to probabilities
            next_token = torch.multinomial(probabilities, num_samples=1)  # Shape: (batch_size, 1)

            # Step 5: Append the sampled token to the output sequence
            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)  # Append new token

        return generated_tokens

    @staticmethod
    def _generate_causal_mask(seq_len: int, device: torch.device) -> Tensor:
        """
        Generates a causal mask to ensure autoregressive behavior.
        Prevents each position from attending to future positions.

        Args:
            seq_len (int): Sequence length.
            device (torch.device): Torch device for tensor allocation.

        Returns:
            Tensor: A causal mask of shape (seq_len, seq_len).
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask

    @staticmethod
    def _apply_top_k_sampling(logits: Tensor, k: int) -> Tensor:
        """
        Applies top-k sampling by zeroing out logits below the top-k threshold.

        Args:
            logits (Tensor): Logits tensor of shape (batch_size, vocab_size).
            k (int): Number of top tokens to retain.

        Returns:
            Tensor: Logits tensor with only top-k values retained.
        """
        values, indices = torch.topk(logits, k=k, dim=-1)
        min_values = values[:, -1, None]  # Minimum value in the top-k logits
        logits[logits < min_values] = float("-inf")  # Zero out logits below top-k threshold
        return logits
