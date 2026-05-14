"""
Pi-Learner: Order-Aware Training Experiments
=============================================
Implements the pi-learner experiments from Section 3.2 of the paper.

A pi-learner is a likelihood model that predicts tokens in a specific order
defined by permutation pi. The MDM loss is equivalent to the average loss
of pi-learners over all permutations.

Key insight: As pi deviates from the identity (left-to-right), the scaling
law of the pi-learner gets worse, demonstrating that MDMs train on harder
subproblems than ARMs.

Experimental setup:
- Dataset: SlimPajama (Soboleva et al., 2023)
- Model: Transformer with causal attention + learnable positional embeddings
- Permutations: identity (ARM), closer, much-closer, random (MDM)
- Metric: IsoFLOP scaling law analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple


def generate_permutation(L: int, mode: str = 'random', 
                          n_swaps: int = None,
                          seed: int = 42) -> np.ndarray:
    """
    Generate a permutation of indices {0, ..., L-1}.
    
    Args:
        L: sequence length
        mode: 'identity', 'random', 'closer', 'much_closer'
              - 'identity': identity permutation (ARM)
              - 'random': uniform random permutation (MDM)
              - 'closer': L/10 random swaps from identity
              - 'much_closer': sqrt(L) random swaps from identity
        n_swaps: number of random swaps (overrides mode if provided)
        seed: random seed
    
    Returns:
        permutation: array of shape (L,) with values in {0, ..., L-1}
    """
    rng = np.random.RandomState(seed)
    
    if mode == 'identity':
        return np.arange(L)
    elif mode == 'random':
        return rng.permutation(L)
    elif mode == 'closer':
        # L/10 random swaps from identity
        n_swaps = L // 10
    elif mode == 'much_closer':
        # sqrt(L) random swaps from identity
        n_swaps = int(np.sqrt(L))
    
    # Start from identity and perform n_swaps random transpositions
    perm = np.arange(L)
    for _ in range(n_swaps):
        i, j = rng.choice(L, size=2, replace=False)
        perm[i], perm[j] = perm[j], perm[i]
    
    return perm


def permute_sequence(x: torch.Tensor, perm: np.ndarray) -> torch.Tensor:
    """
    Permute a sequence according to permutation pi.
    
    For a pi-learner, we permute the input sequence so that the
    autoregressive model predicts tokens in the order defined by pi.
    
    Args:
        x: input sequences (B, L)
        perm: permutation array of shape (L,)
    
    Returns:
        x_permuted: permuted sequences (B, L)
    """
    perm_tensor = torch.tensor(perm, dtype=torch.long, device=x.device)
    return x[:, perm_tensor]


def compute_pi_learner_loss(model: nn.Module, x: torch.Tensor, 
                             perm: np.ndarray) -> torch.Tensor:
    """
    Compute the pi-learner loss.
    
    The pi-learner predicts token at position pi(i) given tokens at
    positions pi(0), ..., pi(i-1). This is equivalent to:
    1. Permuting the sequence according to pi
    2. Computing the standard autoregressive loss on the permuted sequence
    
    Args:
        model: causal transformer model
        x: input sequences (B, L)
        perm: permutation array
    
    Returns:
        loss: scalar loss value
    """
    # Permute the sequence
    x_perm = permute_sequence(x, perm)
    
    # Compute autoregressive loss on permuted sequence
    # Input: x_perm[:, :-1], Target: x_perm[:, 1:]
    logits = model(x_perm[:, :-1])  # (B, L-1, vocab_size)
    
    B, L_minus_1, V = logits.shape
    loss = F.cross_entropy(
        logits.reshape(B * L_minus_1, V),
        x_perm[:, 1:].reshape(B * L_minus_1),
    )
    
    return loss


@torch.no_grad()
def compute_pi_learner_likelihood(model: nn.Module, x: torch.Tensor,
                                   perm: np.ndarray) -> torch.Tensor:
    """
    Compute the pi-learner log-likelihood.
    
    log p_theta(x_0) = sum_{i=0}^{L-1} log p_theta(x_0^{pi(i)} | x_0[pi{i,...,L-1}])
    
    Args:
        model: causal transformer model
        x: input sequences (B, L)
        perm: permutation array
    
    Returns:
        log_likelihood: (B,) log-likelihood for each sequence
    """
    model.eval()
    
    # Permute the sequence
    x_perm = permute_sequence(x, perm)
    
    # Compute log-probabilities
    logits = model(x_perm[:, :-1])  # (B, L-1, vocab_size)
    log_probs = F.log_softmax(logits, dim=-1)  # (B, L-1, vocab_size)
    
    # Gather log-probabilities for the actual tokens
    B, L_minus_1, V = log_probs.shape
    targets = x_perm[:, 1:]  # (B, L-1)
    
    token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # (B, L-1)
    
    # Sum over sequence length
    log_likelihood = token_log_probs.sum(dim=-1)  # (B,)
    
    return log_likelihood


class CausalTransformer(nn.Module):
    """
    Causal (autoregressive) transformer for pi-learner experiments.
    
    Uses causal attention mask and learnable positional embeddings
    (instead of RoPE, to avoid inductive bias towards left-to-right ordering).
    """
    
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int,
                 d_ff: int = None, max_seq_len: int = 2048, dropout: float = 0.1,
                 use_learnable_pos: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        if d_ff is None:
            d_ff = 4 * d_model
        
        self.token_emb = nn.Embedding(vocab_size, d_model)
        
        if use_learnable_pos:
            self.pos_emb = nn.Embedding(max_seq_len, d_model)
        else:
            self.pos_emb = None
        
        self.emb_dropout = nn.Dropout(dropout)
        
        # Causal transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def _make_causal_mask(self, L: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
        return mask  # True = masked (not attended to)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: input token ids (B, L)
        
        Returns:
            logits: (B, L, vocab_size)
        """
        B, L = x.shape
        
        h = self.token_emb(x)
        
        if self.pos_emb is not None:
            positions = torch.arange(L, device=x.device).unsqueeze(0)
            h = h + self.pos_emb(positions)
        
        h = self.emb_dropout(h)
        
        # Causal mask
        causal_mask = self._make_causal_mask(L, x.device)
        
        h = self.transformer(h, mask=causal_mask, is_causal=True)
        h = self.norm(h)
        logits = self.lm_head(h)
        
        return logits


class PiLearnerExperiment:
    """
    Manages the pi-learner scaling law experiments.
    
    Trains multiple pi-learners with different permutations and
    measures their scaling laws (IsoFLOP analysis).
    """
    
    def __init__(self, L: int = 2048, seed: int = 42):
        self.L = L
        self.seed = seed
        
        # Generate permutations for different conditions
        self.permutations = {
            'identity': generate_permutation(L, 'identity', seed=seed),
            'much_closer': generate_permutation(L, 'much_closer', seed=seed),
            'closer': generate_permutation(L, 'closer', seed=seed),
            'random_1': generate_permutation(L, 'random', seed=seed),
            'random_2': generate_permutation(L, 'random', seed=seed + 1),
            'random_3': generate_permutation(L, 'random', seed=seed + 2),
        }
    
    def get_permutation(self, name: str) -> np.ndarray:
        """Get a named permutation."""
        return self.permutations[name]
    
    def compute_permutation_distance(self, perm: np.ndarray) -> float:
        """
        Compute the distance of a permutation from the identity.
        
        Uses Kendall tau distance (number of inversions).
        """
        n = len(perm)
        inversions = 0
        for i in range(n):
            for j in range(i + 1, n):
                if perm[i] > perm[j]:
                    inversions += 1
        return inversions / (n * (n - 1) / 2)  # Normalize to [0, 1]
