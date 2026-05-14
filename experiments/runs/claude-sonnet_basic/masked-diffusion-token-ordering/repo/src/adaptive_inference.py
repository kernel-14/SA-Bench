"""
Adaptive MDM Inference
======================
Implements the adaptive inference strategies for Masked Diffusion Models
as described in Section 4 of the paper.

Two key strategies:
1. Top Probability: Select positions with highest max probability
2. Top Probability Margin: Select positions with highest difference between
   top-2 probabilities (more robust to uncertainty)

The key insight: instead of randomly selecting which tokens to unmask (vanilla MDM),
we adaptively choose positions where the model is most confident, avoiding
hard subproblems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


MASK_TOKEN = 0


def top_probability_oracle(probs: torch.Tensor, 
                            masked_positions: torch.Tensor,
                            K: int) -> torch.Tensor:
    """
    Top Probability oracle: select K positions with highest max probability.
    
    For each masked position i, certainty = max_j p_theta(x^i = j | x_t)
    Select the K positions with highest certainty.
    
    Args:
        probs: token probabilities (B, L, vocab_size)
        masked_positions: boolean mask of currently masked positions (B, L)
        K: number of positions to select
    
    Returns:
        selected: boolean tensor of selected positions (B, L)
    """
    B, L, V = probs.shape
    device = probs.device
    
    # Compute max probability at each position
    max_probs = probs.max(dim=-1).values  # (B, L)
    
    # Set non-masked positions to -inf so they won't be selected
    max_probs = max_probs.masked_fill(~masked_positions, float('-inf'))
    
    # Select top-K positions per sequence
    selected = torch.zeros(B, L, dtype=torch.bool, device=device)
    
    if K > 0:
        for i in range(B):
            n_masked_i = masked_positions[i].sum().item()
            if n_masked_i > 0:
                k_i = min(K, int(n_masked_i))
                _, top_k_idx = max_probs[i].topk(k_i)
                selected[i, top_k_idx] = True
        # Only keep positions that are actually masked
        selected = selected & masked_positions
    
    return selected


def top_probability_margin_oracle(probs: torch.Tensor,
                                   masked_positions: torch.Tensor,
                                   K: int,
                                   gumbel_noise: float = 0.0) -> torch.Tensor:
    """
    Top Probability Margin oracle: select K positions with highest margin
    between top-2 probabilities.
    
    For each masked position i, certainty = |p(x^i = j1 | x_t) - p(x^i = j2 | x_t)|
    where j1, j2 are the two most probable values.
    
    This is more robust than top probability when multiple values have similar
    high probabilities (e.g., in Sudoku where multiple digits might be plausible).
    
    Args:
        probs: token probabilities (B, L, vocab_size)
        masked_positions: boolean mask of currently masked positions (B, L)
        K: number of positions to select
        gumbel_noise: coefficient for Gumbel noise (for diversity in text generation)
    
    Returns:
        selected: boolean tensor of selected positions (B, L)
    """
    B, L, V = probs.shape
    device = probs.device
    
    # Get top-2 probabilities at each position
    top2_probs, _ = probs.topk(min(2, V), dim=-1)  # (B, L, 2)
    
    if top2_probs.shape[-1] >= 2:
        # Margin = difference between top-1 and top-2 probabilities
        margin = top2_probs[..., 0] - top2_probs[..., 1]  # (B, L)
    else:
        margin = top2_probs[..., 0]  # Only one token in vocab
    
    # Add optional Gumbel noise for diversity (used in text generation)
    if gumbel_noise > 0.0:
        noise = -torch.log(-torch.log(torch.rand_like(margin) + 1e-10) + 1e-10)
        margin = margin + gumbel_noise * noise
    
    # Set non-masked positions to -inf
    margin = margin.masked_fill(~masked_positions, float('-inf'))
    
    # Select top-K positions per sequence
    selected = torch.zeros(B, L, dtype=torch.bool, device=device)
    
    if K > 0:
        for i in range(B):
            n_masked_i = masked_positions[i].sum().item()
            if n_masked_i > 0:
                k_i = min(K, int(n_masked_i))
                _, top_k_idx = margin[i].topk(k_i)
                selected[i, top_k_idx] = True
        selected = selected & masked_positions
    
    return selected


def vanilla_oracle(masked_positions: torch.Tensor, K: int) -> torch.Tensor:
    """
    Vanilla (random) oracle: randomly select K masked positions.
    
    This is the standard MDM inference that selects positions randomly.
    
    Args:
        masked_positions: boolean mask of currently masked positions (B, L)
        K: number of positions to select
    
    Returns:
        selected: boolean tensor of selected positions (B, L)
    """
    B, L = masked_positions.shape
    device = masked_positions.device
    
    selected = torch.zeros(B, L, dtype=torch.bool, device=device)
    
    for i in range(B):
        masked_idx = masked_positions[i].nonzero(as_tuple=True)[0]
        n_masked = len(masked_idx)
        if n_masked > 0 and K > 0:
            k_actual = min(K, n_masked)
            perm = torch.randperm(n_masked, device=device)[:k_actual]
            selected[i, masked_idx[perm]] = True
    
    return selected


@torch.no_grad()
def mdm_sample(model: nn.Module, 
               x_init: torch.Tensor,
               n_steps: int = 50,
               strategy: str = 'vanilla',
               gumbel_noise: float = 0.0,
               temperature: float = 1.0,
               noise_schedule: str = 'linear') -> torch.Tensor:
    """
    Generate samples using MDM reverse process.
    
    Starting from a fully (or partially) masked sequence, iteratively unmask
    tokens using the specified strategy.
    
    Args:
        model: trained MDM denoising network
        x_init: initial sequence (B, L), with 0 for masked tokens
        n_steps: number of reverse diffusion steps
        strategy: 'vanilla', 'top_prob', or 'top_prob_margin'
        gumbel_noise: Gumbel noise coefficient for diversity
        temperature: sampling temperature
        noise_schedule: 'linear' or 'cosine'
    
    Returns:
        x_final: generated sequences (B, L)
    """
    model.eval()
    device = x_init.device
    B, L = x_init.shape
    
    x = x_init.clone()
    
    # Compute noise schedule alpha_t values
    # alpha_t goes from ~1 (no masking) to ~0 (fully masked)
    # We go from t=1 (fully masked) to t=0 (fully unmasked)
    if noise_schedule == 'linear':
        alphas = torch.linspace(0, 1, n_steps + 1, device=device)
    elif noise_schedule == 'cosine':
        t_vals = torch.linspace(0, 1, n_steps + 1, device=device)
        alphas = torch.cos(t_vals * np.pi / 2) ** 2
        alphas = alphas / alphas[0]  # normalize
    else:
        alphas = torch.linspace(0, 1, n_steps + 1, device=device)
    
    # Reverse: from alpha_1 (=0, fully masked) to alpha_0 (=1, fully unmasked)
    # At each step t -> s, we unmask some tokens
    
    for step in range(n_steps):
        # Current and next alpha values
        alpha_t = alphas[n_steps - step]      # current (higher t = more masked)
        alpha_s = alphas[n_steps - step - 1]  # next (lower t = less masked)
        
        # Identify currently masked positions
        masked_positions = (x == MASK_TOKEN)  # (B, L)
        
        if not masked_positions.any():
            break
        
        # Get model predictions
        logits = model(x)  # (B, L, vocab_size)
        
        if temperature != 1.0:
            logits = logits / temperature
        
        probs = F.softmax(logits, dim=-1)  # (B, L, vocab_size)
        
        # Compute number of tokens to unmask at this step
        # Expected number: (alpha_s - alpha_t) / (1 - alpha_t) * n_masked
        n_masked_per_seq = masked_positions.sum(dim=-1).float()  # (B,)
        
        if alpha_t < 1.0:
            unmask_ratio = (alpha_s - alpha_t) / (1.0 - alpha_t)
        else:
            unmask_ratio = 1.0 / max(n_steps - step, 1)
        
        K_per_seq = (n_masked_per_seq * unmask_ratio).round().long()
        K = max(K_per_seq.max().item(), 1)
        
        # Select positions to unmask using the oracle
        if strategy == 'vanilla':
            selected = vanilla_oracle(masked_positions, K)
        elif strategy == 'top_prob':
            selected = top_probability_oracle(probs, masked_positions, K)
        elif strategy == 'top_prob_margin':
            selected = top_probability_margin_oracle(probs, masked_positions, K, gumbel_noise)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Sample token values for selected positions
        for i in range(B):
            sel_idx = selected[i].nonzero(as_tuple=True)[0]
            if len(sel_idx) > 0:
                # Sample from the predicted distribution
                sel_probs = probs[i, sel_idx].clone()  # (n_sel, vocab_size)
                # Don't sample the mask token
                sel_probs[:, MASK_TOKEN] = 0.0
                sel_probs = sel_probs / sel_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
                
                sampled = torch.multinomial(sel_probs, num_samples=1).squeeze(-1)
                x[i, sel_idx] = sampled
    
    return x


@torch.no_grad()
def mdm_sample_greedy(model: nn.Module,
                       x_init: torch.Tensor,
                       n_steps: int = 50,
                       strategy: str = 'top_prob_margin',
                       gumbel_noise: float = 0.5) -> torch.Tensor:
    """
    Generate samples using greedy MDM inference (argmax decoding).
    
    Similar to mdm_sample but uses argmax instead of sampling.
    This is used for logic puzzle solving where we want deterministic outputs.
    
    The number of tokens to unmask at each step is:
    K = ceil(n_masked / remaining_steps)
    
    This ensures all tokens are unmasked by the end.
    
    Args:
        model: trained MDM denoising network
        x_init: initial sequence (B, L), with 0 for masked tokens
        n_steps: number of reverse diffusion steps
        strategy: 'vanilla', 'top_prob', or 'top_prob_margin'
        gumbel_noise: Gumbel noise coefficient for oracle selection
    
    Returns:
        x_final: generated sequences (B, L)
    """
    model.eval()
    device = x_init.device
    B, L = x_init.shape
    
    x = x_init.clone()
    
    for step in range(n_steps):
        masked_positions = (x == MASK_TOKEN)
        
        if not masked_positions.any():
            break
        
        # Get model predictions
        logits = model(x)
        probs = F.softmax(logits, dim=-1)
        
        # Compute K: number of tokens to unmask
        # Use ceiling to ensure we make progress
        n_masked_max = masked_positions.sum(dim=-1).max().item()
        remaining_steps = n_steps - step
        K = max(int(np.ceil(n_masked_max / remaining_steps)), 1)
        
        # Select positions using oracle
        if strategy == 'vanilla':
            selected = vanilla_oracle(masked_positions, K)
        elif strategy == 'top_prob':
            selected = top_probability_oracle(probs, masked_positions, K)
        elif strategy == 'top_prob_margin':
            selected = top_probability_margin_oracle(probs, masked_positions, K, gumbel_noise)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Assign tokens: use argmax (greedy) for selected positions
        for i in range(B):
            sel_idx = selected[i].nonzero(as_tuple=True)[0]
            if len(sel_idx) > 0:
                sel_probs = probs[i, sel_idx].clone()
                # Don't assign mask token
                sel_probs[:, MASK_TOKEN] = 0.0
                best_tokens = sel_probs.argmax(dim=-1)
                x[i, sel_idx] = best_tokens
    
    return x


def compute_K_for_step(n_masked: int, alpha_s: float, alpha_t: float) -> int:
    """
    Compute the number of tokens to unmask at a given step.
    
    Based on the MDM reverse process: at each step t -> s,
    the expected number of tokens to unmask is:
    K = n_masked * (alpha_s - alpha_t) / (1 - alpha_t)
    
    Args:
        n_masked: current number of masked tokens
        alpha_s: alpha value at next step (lower noise)
        alpha_t: alpha value at current step (higher noise)
    
    Returns:
        K: number of tokens to unmask
    """
    if alpha_t >= 1.0:
        return max(1, n_masked // 10)
    ratio = (alpha_s - alpha_t) / (1.0 - alpha_t)
    return max(1, round(n_masked * ratio))
