"""
Training utilities for NFIG: FR-VAE tokenizer and NFIG Transformer.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Tuple, List
import os
import time
from tqdm import tqdm

from .fr_vae import FRVAE
from .nfig_transformer import NFIGTransformer


class FRVAETrainer:
    """
    Trainer for the FR-VAE image tokenizer.
    
    Trains the frequency-guided residual-quantized VAE using:
    - Reconstruction loss (pixel + feature)
    - LPIPS perceptual loss
    - GAN loss with DINO discriminator
    """
    
    def __init__(
        self,
        model: FRVAE,
        device: torch.device,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.5, 0.9),
        disc_lr: float = 1e-4,
    ):
        self.model = model.to(device)
        self.device = device
        
        # Optimizers
        self.optimizer_g = optim.Adam(
            list(model.encoder.parameters()) + 
            list(model.residual_quantizer.parameters()) +
            list(model.decoder.parameters()),
            lr=lr, betas=betas
        )
        
        if model.discriminator is not None:
            self.optimizer_d = optim.Adam(
                model.discriminator.parameters(),
                lr=disc_lr, betas=betas
            )
        else:
            self.optimizer_d = None
    
    def train_step(self, x: torch.Tensor) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        x = x.to(self.device)
        B = x.shape[0]
        
        # Forward pass
        x_recon, token_list, vq_loss = self.model(x)
        
        # Get feature maps for feature reconstruction loss
        f_orig = self.model.encoder(x)
        _, f_combined, _ = self.model.encode(x)
        
        # Train discriminator
        if self.model.discriminator is not None:
            d_loss = self.model.compute_loss(x, x_recon.detach(), vq_loss, 
                                              f_orig, f_combined, optimizer_idx=1)
            self.optimizer_d.zero_grad()
            d_loss.backward()
            self.optimizer_d.step()
        
        # Train generator
        g_loss = self.model.compute_loss(x, x_recon, vq_loss, 
                                          f_orig, f_combined, optimizer_idx=0)
        self.optimizer_g.zero_grad()
        g_loss.backward()
        self.optimizer_g.step()
        
        return {
            'g_loss': g_loss.item(),
            'vq_loss': vq_loss.item() if isinstance(vq_loss, torch.Tensor) else vq_loss,
            'd_loss': d_loss.item() if self.model.discriminator is not None else 0.0,
        }
    
    @torch.no_grad()
    def evaluate_reconstruction(self, x: torch.Tensor) -> Dict[str, float]:
        """Evaluate reconstruction quality."""
        self.model.eval()
        x = x.to(self.device)
        
        x_recon, _, _ = self.model(x)
        mse = nn.functional.mse_loss(x, x_recon).item()
        
        return {'mse': mse}
    
    def save_checkpoint(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_g': self.optimizer_g.state_dict(),
            'optimizer_d': self.optimizer_d.state_dict() if self.optimizer_d else None,
        }, path)
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        if self.optimizer_d and checkpoint['optimizer_d']:
            self.optimizer_d.load_state_dict(checkpoint['optimizer_d'])


class NFIGTrainer:
    """
    Trainer for the NFIG autoregressive transformer.
    
    Uses:
    - Adam optimizer with lr=8e-5 (from paper)
    - Cross-entropy loss
    - Classifier-free guidance training with cond_drop_prob
    - Batch size 768 (from paper)
    """
    
    def __init__(
        self,
        model: NFIGTransformer,
        device: torch.device,
        lr: float = 8e-5,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.grad_clip = grad_clip
        
        # Adam optimizer (paper uses Adam with lr=8e-5)
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=lr, betas=betas, weight_decay=weight_decay
        )
        
        # Store the n_scales for splitting token sequences
        self.scales = model.scales
        self.n_scales = model.n_scales
    
    def split_tokens_to_scales(self, flat_tokens: torch.Tensor) -> List[torch.Tensor]:
        """Split flat token sequence into per-scale token grids."""
        token_seqs = []
        start = 0
        for h, w in self.scales:
            n = h * w
            token_seqs.append(flat_tokens[:, start:start + n])
            start += n
        return token_seqs
    
    def train_step(self, flat_tokens: torch.Tensor, class_ids: torch.Tensor) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            flat_tokens: (B, total_tokens) token indices from FR-VAE
            class_ids: (B,) class labels
        
        Returns:
            dict of loss values
        """
        self.model.train()
        flat_tokens = flat_tokens.to(self.device)
        class_ids = class_ids.to(self.device)
        
        # Split into per-scale sequences
        token_seqs = self.split_tokens_to_scales(flat_tokens)
        
        # Forward pass
        logits, loss = self.model(token_seqs, class_ids)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
        self.optimizer.step()
        
        return {'loss': loss.item()}
    
    @torch.no_grad()
    def generate(
        self,
        class_ids: torch.Tensor,
        top_k: int = 990,
        cfg_scale: float = 4.5,
        temperature: float = 1.0,
    ) -> List[torch.Tensor]:
        """Generate image tokens."""
        self.model.eval()
        return self.model.generate(class_ids, top_k, cfg_scale, temperature)
    
    def save_checkpoint(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
