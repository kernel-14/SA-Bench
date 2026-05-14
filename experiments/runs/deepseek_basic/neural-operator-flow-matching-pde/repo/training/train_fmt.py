"""Training script for FMT (Stage 2).

Two-stage training recipe:
1. Train P2VAE for 100k steps
2. Freeze P2VAE, train FMT for 100k steps with conditional flow marching loss

Optimizer: AdamW with β1=0.9, β2=0.95
Learning rate: cosine schedule with 10% linear warmup
Weight decay: 0.01
Base lr: 1e-4 for batch size 256

The training iterates over 4 consecutive states (x_0, x_1, x_2, x_3)
and computes the flow marching loss using the k-free objective.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any, List
import logging

from p2vae import P2VAE
from fmt import FlowMarchingTransformer, FMTConfig

logger = logging.getLogger(__name__)


def train_fmt(
    model: FlowMarchingTransformer,
    p2vae: P2VAE,
    train_dataloader,
    val_dataloader=None,
    total_steps: int = 100000,
    batch_size: int = 256,
    base_lr: float = 1e-4,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.95,
    warmup_ratio: float = 0.1,
    device: str = 'cuda',
    use_fp16: bool = True,
    log_interval: int = 100,
    save_interval: int = 5000,
    checkpoint_dir: Optional[str] = None,
):
    """Train FMT (Stage 2) with frozen P2VAE.
    
    Args:
        model: FMT model instance
        p2vae: Pre-trained P2VAE (frozen)
        train_dataloader: DataLoader yielding 5 consecutive frames
                         (x_0, x_1, x_2, x_3, x_4)
        val_dataloader: Optional validation dataloader
        total_steps: Total training steps (100k per paper)
        batch_size: Batch size
        base_lr: Base learning rate
        weight_decay: Weight decay (0.01 per paper)
        beta1: AdamW beta1 (0.9 per paper)
        beta2: AdamW beta2 (0.95 per paper)
        warmup_ratio: Fraction of steps for warmup
        device: Training device
        use_fp16: Use mixed precision
        log_interval: Steps between logging
        save_interval: Steps between checkpoint saves
        checkpoint_dir: Directory for saving checkpoints
    """
    model = model.to(device)
    p2vae = p2vae.to(device)
    p2vae.eval()  # Freeze P2VAE
    
    for param in p2vae.parameters():
        param.requires_grad = False
    
    model.train()
    
    # Adjust learning rate
    lr = base_lr * (batch_size / 256) ** 0.5
    
    # AdamW with different betas than P2VAE
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
    )
    
    warmup_steps = int(total_steps * warmup_ratio)
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    
    global_step = 0
    data_iter = iter(train_dataloader)
    losses = []
    
    while global_step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dataloader)
            batch = next(data_iter)
        
        # Batch should contain 5 consecutive frames
        # (x_0, x_1, x_2, x_3, x_4) or shape (B, 5, C, H, W)
        if isinstance(batch, torch.Tensor) and batch.dim() == 5:
            # (B, T, C, H, W)
            x0, x1, x2, x3, x4 = [batch[:, i].to(device) for i in range(5)]
        elif isinstance(batch, (list, tuple)) and len(batch) == 5:
            x0, x1, x2, x3, x4 = [b.to(device) for b in batch]
        else:
            raise ValueError(f"Expected 5 frames, got format: {type(batch)}")
        
        # Encode all frames to latent space with frozen P2VAE
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_fp16):
                y0 = p2vae.encode(x0)[0]  # mu only (no sampling)
                y1 = p2vae.encode(x1)[0]
                y2 = p2vae.encode(x2)[0]
                y3 = p2vae.encode(x3)[0]
                y4 = p2vae.encode(x4)[0]
        
        # Update learning rate
        current_lr = lr
        if global_step < warmup_steps:
            current_lr = lr * global_step / warmup_steps
        else:
            progress = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
            current_lr = lr * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=use_fp16):
            loss_dict = model.compute_flow_marching_loss(y0, y1, y2, y3, y4)
            loss = loss_dict['loss']
        
        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        losses.append(loss.item())
        
        if global_step % log_interval == 0:
            avg_loss = sum(losses[-log_interval:]) / log_interval
            logger.info(
                f"Step {global_step}/{total_steps} | "
                f"LR: {current_lr:.2e} | "
                f"Loss: {avg_loss:.6f}"
            )
        
        if checkpoint_dir and global_step % save_interval == 0:
            checkpoint_path = f"{checkpoint_dir}/fmt_step{global_step}.pt"
            torch.save({
                'step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        global_step += 1
    
    logger.info("FMT training completed!")
    return model


def train_finetune_kolmogorov(
    model: FlowMarchingTransformer,
    p2vae: P2VAE,
    train_dataloader,
    val_dataloader=None,
    total_steps: int = 5000,
    lambda_vae: float = 1.0,
    batch_size: int = 64,
    base_lr: float = 1e-5,
    device: str = 'cuda',
    use_fp16: bool = True,
    log_interval: int = 50,
):
    """Few-shot finetuning on Kolmogorov turbulence (REPA-E style).
    
    Following the paper:
    - Finetune FMT-B-42M on 200 training trajectories for 5k steps
    - λ_VAE = 1
    - Stop-gradient after latent generation to preserve autoencoder
    
    The combined loss:
    L(θ, φ, ω) = L_CFM(θ, φ) + λ_VAE * L_VAE(ω)
    
    Where θ = FMT parameters, φ = GRU parameters, ω = P2VAE parameters
    """
    model = model.to(device)
    p2vae = p2vae.to(device)
    
    # During finetuning, we do NOT freeze P2VAE entirely
    # Instead, we use stop-gradient after latent generation
    # This means: encode -> stop_gradient(latent) -> pass to FMT
    # But P2VAE decoder loss still backprops through encoder
    
    model.train()
    
    # Optimizer includes both FMT + P2VAE parameters
    trainable_params = list(model.parameters())
    # P2VAE decoder is trainable for reconstruction
    # Encoder receives gradient from decoder loss only
    for param in p2vae.parameters():
        param.requires_grad = True
    
    all_params = trainable_params + list(p2vae.parameters())
    
    optimizer = torch.optim.AdamW(all_params, lr=base_lr, weight_decay=0.01)
    
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    
    global_step = 0
    data_iter = iter(train_dataloader)
    losses = []
    
    while global_step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dataloader)
            batch = next(data_iter)
        
        if isinstance(batch, torch.Tensor) and batch.dim() == 5:
            x0, x1, x2, x3, x4 = [batch[:, i].to(device) for i in range(5)]
        elif isinstance(batch, (list, tuple)) and len(batch) == 5:
            x0, x1, x2, x3, x4 = [b.to(device) for b in batch]
        else:
            raise ValueError(f"Expected 5 frames")
        
        with torch.cuda.amp.autocast(enabled=use_fp16):
            # Encode to latent (stop gradient on latent for FMT path)
            y0_mu, y0_logvar = p2vae.encode(x0)
            y1_mu, y1_logvar = p2vae.encode(x1)
            y2_mu, y2_logvar = p2vae.encode(x2)
            y3_mu, y3_logvar = p2vae.encode(x3)
            y4_mu, y4_logvar = p2vae.encode(x4)
            
            # Stop gradient for FMT input (REPA-E style)
            y0 = y0_mu.detach()
            y1 = y1_mu.detach()
            y2 = y2_mu.detach()
            y3 = y3_mu.detach()
            y4 = y4_mu.detach()
            
            # Flow marching loss
            fmt_loss_dict = model.compute_flow_marching_loss(y0, y1, y2, y3, y4)
            loss_cfm = fmt_loss_dict['loss']
            
            # VAE reconstruction loss (backprops through encoder)
            vae_loss_0 = p2vae.compute_loss(x0)['loss']
            vae_loss_1 = p2vae.compute_loss(x1)['loss']
            vae_loss_2 = p2vae.compute_loss(x2)['loss']
            vae_loss_3 = p2vae.compute_loss(x3)['loss']
            loss_vae = (vae_loss_0 + vae_loss_1 + vae_loss_2 + vae_loss_3) / 4
            
            # Combined loss
            loss = loss_cfm + lambda_vae * loss_vae
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        losses.append(loss.item())
        
        if global_step % log_interval == 0:
            avg_loss = sum(losses[-log_interval:]) / log_interval
            logger.info(
                f"Finetune Step {global_step}/{total_steps} | "
                f"Loss: {avg_loss:.6f} | "
                f"CFM: {loss_cfm.item():.6f} | "
                f"VAE: {loss_vae.item():.6f}"
            )
        
        global_step += 1
    
    logger.info("Finetuning completed!")
    return model, p2vae
