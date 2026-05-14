"""
Autoregressive Model (ARM) for comparison
==========================================
Implements autoregressive (causal) language models used as baselines
in the paper (Sections 2.1.1, 3.2, 4.3-4.5).

Two variants:
1. ARM without ordering: Standard left-to-right autoregressive training
2. ARM with ordering: Trained via teacher forcing on the correct token 
   generation order (Shah et al., 2024; Lehnert et al., 2024)

Configuration:
- Causal transformer with learnable positional embeddings
- Same architecture as MDM but with causal attention mask
- Trained with standard next-token prediction loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class CausalTransformer(nn.Module):
    """
    Causal (autoregressive) transformer for ARM training.
    
    Uses a decoder-only architecture with causal self-attention.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 512,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Learnable positional embedding (not RoPE)
        self.pos_embedding = nn.Embedding(max_seq_length, d_model)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Causal mask
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(max_seq_length, max_seq_length) * float('-inf'), diagonal=1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with causal attention.
        
        Args:
            x: (batch, seq_len) input token ids
            
        Returns:
            logits: (batch, seq_len, vocab_size) prediction logits
        """
        batch_size, seq_len = x.shape
        
        # Embed tokens
        h = self.token_embedding(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        h = h + self.pos_embedding(positions)
        
        # Apply causal mask
        attn_mask = self.causal_mask[:seq_len, :seq_len]
        
        # Transformer decoder (self-attention only, no cross-attention)
        # Using memory=None since we only need self-attention
        h = self.transformer(h, h, tgt_mask=attn_mask)
        
        # Output projection
        logits = self.output_proj(h)
        
        return logits


class AutoregressiveModel:
    """
    Autoregressive Model wrapper.
    
    Supports:
    - Standard left-to-right training (identity permutation)
    - Order-aware training with arbitrary permutation
    - Teacher forcing on ground-truth prefix
    """
    
    def __init__(self, model: CausalTransformer, vocab_size: int):
        self.model = model
        self.vocab_size = vocab_size
    
    def compute_loss(self, x_0: torch.Tensor, pi: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute autoregressive loss.
        
        If pi is None, uses standard left-to-right order.
        If pi is provided, permutes the input according to pi before
        applying causal attention.
        
        Args:
            x_0: (batch, seq_len) clean sequences
            pi: Optional (batch, seq_len) or (seq_len,) permutation
            
        Returns:
            Cross-entropy loss
        """
        batch_size, seq_len = x_0.shape
        
        if pi is not None:
            # Apply permutation before feeding to model
            if pi.dim() == 1:
                pi = pi.unsqueeze(0).expand(batch_size, -1)
            
            # Create permuted input
            x_permuted = torch.zeros_like(x_0)
            for b in range(batch_size):
                x_permuted[b] = x_0[b, pi[b]]
            
            # Forward pass
            logits = self.model(x_permuted)  # (batch, seq_len, vocab_size)
            
            # For loss, we need to align predictions with correct targets
            # The model predicts x_{t+1} given x_{≤t}
            # After permutation, target at position i is π(i+1)
            targets = torch.zeros_like(x_0)
            for b in range(batch_size):
                # Shift left: predict token at π(i+1) given tokens up to π(i)
                targets[b, :-1] = x_permuted[b, 1:]
                targets[b, -1] = 0  # Don't care about last position
            
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=0,
            )
        else:
            # Standard left-to-right
            logits = self.model(x_0)  # (batch, seq_len, vocab_size)
            
            # Shift: predict next token
            targets = torch.zeros_like(x_0)
            targets[:, :-1] = x_0[:, 1:]
            targets[:, -1] = 0
            
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=0,
            )
        
        return loss
    
    def generate(self, prefix: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """
        Autoregressive generation.
        
        Args:
            prefix: (batch, prefix_len) starting tokens
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            
        Returns:
            Full sequence including prefix
        """
        self.model.eval()
        batch_size = prefix.size(0)
        device = prefix.device
        
        generated = prefix.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Truncate if exceeding max length
                if generated.size(1) >= self.model.max_seq_length:
                    break
                
                logits = self.model(generated)
                next_logits = logits[:, -1, :] / temperature  # (batch, vocab_size)
                
                if top_k is not None:
                    top_k_vals, _ = next_logits.topk(top_k, dim=-1)
                    min_top_k = top_k_vals[:, -1].unsqueeze(-1)
                    next_logits[next_logits < min_top_k] = float('-inf')
                
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)  # (batch, 1)
                
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated


def create_arm_model(
    vocab_size: int,
    seq_length: int,
    d_model: int = 512,
    n_heads: int = 8,
    n_layers: int = 6,
    d_ff: int = 2048,
    dropout: float = 0.1,
) -> AutoregressiveModel:
    """Create an ARM model with given configuration."""
    model = CausalTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        max_seq_length=seq_length,
    )
    return AutoregressiveModel(model, vocab_size)
