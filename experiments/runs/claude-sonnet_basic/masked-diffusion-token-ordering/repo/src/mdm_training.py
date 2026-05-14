"""
MDM Training
============
Implements the masked diffusion model training objective:

L_theta = integral_0^1 (alpha_t' / (1 - alpha_t)) * E_{x_t ~ p_data} 
          sum_{i: x_t^i = 0} -log p_theta(x_0^i | x_t, t) dt

Which is equivalent to (Proposition 2.1):

L_theta = -sum_{M subset [L], i in M} (1/|M|) * (1/C(L,|M|)) * E[log p_theta(x_0^i | x_0[M])]

In practice, we use the simplified form: randomly mask tokens and compute
cross-entropy loss on masked positions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Tuple
import numpy as np


MASK_TOKEN = 0  # Token ID for the mask token


def mask_tokens(x: torch.Tensor, mask_prob: float = None, 
                min_mask: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly mask tokens in a sequence for MDM training.
    
    For each sequence, independently mask each token with probability
    sampled uniformly from [0, 1] (or a fixed mask_prob if provided).
    
    Args:
        x: input token ids (B, L), values in {1, ..., vocab_size-1}
        mask_prob: if None, sample uniformly from [0, 1] per sequence
        min_mask: minimum number of tokens to mask per sequence
    
    Returns:
        x_masked: masked sequence with 0 for masked tokens
        mask: boolean tensor indicating masked positions (B, L)
    """
    B, L = x.shape
    device = x.device
    
    if mask_prob is None:
        # Sample mask probability uniformly for each sequence
        # This corresponds to the MDM training objective
        probs = torch.rand(B, 1, device=device)
    else:
        probs = torch.full((B, 1), mask_prob, device=device)
    
    # Create mask: True where tokens should be masked
    mask = torch.rand(B, L, device=device) < probs
    
    # Ensure at least min_mask tokens are masked per sequence
    for i in range(B):
        if mask[i].sum() < min_mask:
            # Randomly select min_mask positions to mask
            idx = torch.randperm(L, device=device)[:min_mask]
            mask[i, idx] = True
    
    # Apply mask
    x_masked = x.clone()
    x_masked[mask] = MASK_TOKEN
    
    return x_masked, mask


def mdm_loss(model: nn.Module, x: torch.Tensor, 
             mask_prob: float = None,
             ignore_index: int = -100) -> torch.Tensor:
    """
    Compute the MDM training loss.
    
    The loss is the cross-entropy on masked positions:
    L = -E[sum_{i: x_t^i = 0} log p_theta(x_0^i | x_t)]
    
    Args:
        model: MDM denoising network
        x: clean token sequences (B, L)
        mask_prob: masking probability (None = sample uniformly)
        ignore_index: index to ignore in cross-entropy
    
    Returns:
        loss: scalar loss value
    """
    x_masked, mask = mask_tokens(x, mask_prob)
    
    # Forward pass
    logits = model(x_masked)  # (B, L, vocab_size)
    
    # Compute loss only on masked positions
    # Reshape for cross-entropy
    B, L, V = logits.shape
    
    # Create targets: original tokens at masked positions, ignore_index elsewhere
    targets = torch.full_like(x, ignore_index)
    targets[mask] = x[mask]
    
    loss = F.cross_entropy(
        logits.reshape(B * L, V),
        targets.reshape(B * L),
        ignore_index=ignore_index,
    )
    
    return loss


class MDMTrainer:
    """
    Trainer for Masked Diffusion Models.
    
    Implements the training loop with the MDM objective.
    """
    
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 device: torch.device, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
    
    def train_epoch(self, dataloader: DataLoader, 
                    mask_prob: float = None) -> float:
        """
        Train for one epoch.
        
        Args:
            dataloader: data loader
            mask_prob: masking probability (None = sample uniformly)
        
        Returns:
            average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            x = x.to(self.device)
            
            self.optimizer.zero_grad()
            loss = mdm_loss(self.model, x, mask_prob)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, 
                 mask_prob: float = None) -> float:
        """
        Evaluate on a dataset.
        
        Returns:
            average loss
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            x = x.to(self.device)
            
            loss = mdm_loss(self.model, x, mask_prob)
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches


def create_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, 
                                        num_training_steps: int,
                                        min_lr_ratio: float = 0.1):
    """Create cosine learning rate schedule with warmup."""
    from torch.optim.lr_scheduler import LambdaLR
    
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)
