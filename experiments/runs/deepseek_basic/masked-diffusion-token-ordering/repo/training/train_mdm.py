"""
MDM Training Script
====================
Implements the training procedures described in the paper:

1. Standard MDM training (Section 2, Equation 1):
   Trains a denoising network on the ELBO-based loss for all masking patterns.

2. π-learner training (Section 3.2):
   Trains a causal transformer on a specific permutation order.
   Used for analyzing the hardness of different token orders.

3. ARM training (Section 2.1.1):
   Standard autoregressive (left-to-right) training as baseline.

Configuration follows the paper's experimental setup:
- AdamW optimizer with β1=0.9, β2=0.95, weight_decay=0.1
- Cosine learning rate schedule (max 4e-4, min 4e-5)
- Learnable positional embeddings (not RoPE, to avoid left-to-right bias)
- IsoFLOP analysis support (Hoffmann et al., 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Callable, Dict
import math
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mdm import MDMTransformer, MDMConfig, MaskedDiffusionModel
from utils.permutations import (
    random_permutation, identity_permutation, interpolated_permutation,
    sample_permutations_for_interpolation
)


class SequenceDataset(Dataset):
    """Dataset wrapper for token sequences."""
    
    def __init__(self, sequences: torch.Tensor):
        """
        Args:
            sequences: (N, L) tensor of token ids
        """
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]


class TokenizedTextDataset(Dataset):
    """
    Simple text dataset that tokenizes on-the-fly using character-level tokens.
    For real experiments, this would use a proper tokenizer (e.g., GPT-2 tokenizer).
    """
    
    def __init__(self, texts: list, seq_length: int, vocab_size: int,
                 mask_token_id: int = 0):
        self.texts = texts
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        
        # Simple character-level tokenization (for demo)
        self.char_to_id = {}
        for i, c in enumerate(set(''.join(texts))):
            if i + 1 < vocab_size:
                self.char_to_id[c] = i + 1  # 0 is mask
        
        self.mask_token_id = mask_token_id
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = []
        for c in text[:self.seq_length]:
            tokens.append(self.char_to_id.get(c, len(self.char_to_id) + 1))
        # Pad
        while len(tokens) < self.seq_length:
            tokens.append(self.mask_token_id)
        return torch.tensor(tokens[:self.seq_length], dtype=torch.long)


def create_mdm_model(vocab_size: int, seq_length: int, 
                     d_model: int = 512, n_heads: int = 8, n_layers: int = 6,
                     d_ff: int = 2048, dropout: float = 0.1,
                     max_seq_length: int = 512) -> MaskedDiffusionModel:
    """Create an MDM model with given configuration."""
    config = MDMConfig(
        vocab_size=vocab_size,
        seq_length=seq_length,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        max_seq_length=max_seq_length,
        noise_schedule='cosine',
        T=1000,
        mask_token_id=0,
    )
    denoiser = MDMTransformer(config)
    return MaskedDiffusionModel(denoiser, config)


class MDMTrainer:
    """
    Trainer for Masked Diffusion Models.
    
    Implements:
    - Standard MDM training (order-agnostic)
    - π-learner training (order-aware, for analysis)
    - ARM training (left-to-right autoregressive)
    """
    
    def __init__(
        self,
        model: MaskedDiffusionModel,
        device: str = 'cpu',
        learning_rate: float = 4e-4,
        min_learning_rate: float = 4e-5,
        weight_decay: float = 0.1,
        beta1: float = 0.9,
        beta2: float = 0.95,
        warmup_steps: int = 2000,
    ):
        self.model = model
        self.device = device
        self.model.denoiser.to(device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.denoiser.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
        )
        
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_steps = warmup_steps
        
        self.step_count = 0
        
        # Track losses
        self.train_losses = []
        self.val_losses = []
    
    def cosine_lr_schedule(self, step: int, total_steps: int) -> float:
        """Cosine learning rate schedule with warmup."""
        if step < self.warmup_steps:
            return self.learning_rate * step / self.warmup_steps
        
        progress = (step - self.warmup_steps) / max(1, total_steps - self.warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_learning_rate + (self.learning_rate - self.min_learning_rate) * cosine_decay
    
    def train_step_mdm(self, batch: torch.Tensor) -> float:
        """Single training step for standard MDM."""
        batch = batch.to(self.device)
        self.optimizer.zero_grad()
        
        loss = self.model.compute_loss(batch)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.denoiser.parameters(), 1.0)
        
        self.optimizer.step()
        return loss.item()
    
    def train_step_pi_learner(self, batch: torch.Tensor, pi: torch.Tensor) -> float:
        """Single training step for π-learner (Section 3.2)."""
        batch = batch.to(self.device)
        pi = pi.to(self.device)
        self.optimizer.zero_grad()
        
        loss = self.model.compute_pi_learner_loss(batch, pi)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.denoiser.parameters(), 1.0)
        
        self.optimizer.step()
        return loss.item()
    
    def train_epoch(self, dataloader: DataLoader, mode: str = 'mdm',
                    pi_generator: Optional[Callable] = None):
        """Train for one epoch."""
        self.model.denoiser.train()
        total_loss = 0.0
        
        for batch in dataloader:
            self.step_count += 1
            
            if mode == 'mdm':
                loss = self.train_step_mdm(batch)
            elif mode == 'pi_learner':
                if pi_generator is not None:
                    pi = pi_generator(batch.size(0))  # Generate permutation for batch
                else:
                    pi = torch.arange(batch.size(1)).unsqueeze(0).expand(batch.size(0), -1)
                loss = self.train_step_pi_learner(batch, pi)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            total_loss += loss
            
            # Update LR
            lr = self.cosine_lr_schedule(self.step_count, len(dataloader) * 100)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        
        avg_loss = total_loss / len(dataloader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, dataloader: DataLoader, mode: str = 'mdm') -> float:
        """Compute validation loss."""
        self.model.denoiser.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                if mode == 'mdm':
                    loss = self.model.compute_loss(batch)
                else:
                    loss = self.model.compute_pi_learner_loss(
                        batch, torch.arange(batch.size(1)).unsqueeze(0).expand(batch.size(0), -1).to(self.device)
                    )
                total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.denoiser.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.denoiser.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step_count = checkpoint['step_count']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])


def run_isoflop_analysis(
    model_sizes: list,
    flops_budget: float,
    train_dataset: Dataset,
    val_dataset: Dataset,
    device: str = 'cpu',
    batch_size: int = 32,
):
    """
    Run IsoFLOP analysis as described in Section 3.2 (Hoffmann et al., 2022).
    
    For a given FLOPs budget C, vary model size N and train for
    C / (6 * N) tokens observed.
    
    Args:
        model_sizes: List of (d_model, n_layers) tuples
        flops_budget: Total FLOPs budget
        train_dataset, val_dataset: Datasets
        device: Device
        batch_size: Batch size
    
    Returns:
        results: Dict mapping model_size -> best_val_loss
    """
    results = {}
    seq_length = train_dataset[0].size(0)
    vocab_size = 50257  # GPT-2 vocab size
    
    for d_model, n_layers in model_sizes:
        # Calculate number of non-embedding parameters
        # Approximate: 12 * d_model^2 * n_layers
        N_params = 12 * d_model * d_model * n_layers
        
        # Total tokens to observe: C / (6 * N)
        total_tokens = flops_budget / (6 * N_params)
        total_steps = int(total_tokens / (batch_size * seq_length))
        
        # Create model
        config = MDMConfig(
            vocab_size=vocab_size,
            seq_length=seq_length,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=d_model // 64,
            d_ff=4 * d_model,
            max_seq_length=seq_length,
        )
        denoiser = MDMTransformer(config)
        mdm = MaskedDiffusionModel(denoiser, config)
        
        # Train
        trainer = MDMTrainer(mdm, device=device)
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Train for total_steps
        best_val_loss = float('inf')
        steps_done = 0
        
        while steps_done < total_steps:
            for batch in dataloader:
                trainer.train_step_mdm(batch)
                steps_done += 1
                
                if steps_done >= total_steps:
                    break
            
            # Validate
            val_loss = trainer.validate(val_dataloader)
            best_val_loss = min(best_val_loss, val_loss)
        
        results[(d_model, n_layers)] = best_val_loss
    
    return results
