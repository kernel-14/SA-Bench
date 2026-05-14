"""Training script for P2VAE (Stage 1).

Two-stage training recipe:
1. Train P2VAE for 100k steps with β_KL = 1e-3
2. Freeze P2VAE, train FMT for another 100k steps

Optimizer: AdamW with β1=0.9, β2=0.995
Learning rate: cosine schedule with 10% linear warmup
Weight decay: 1e-4
Base lr: 1e-4 for batch size 256
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Dict, Any
import logging

from p2vae import P2VAE, P2VAEConfig

logger = logging.getLogger(__name__)


def adjust_learning_rate(base_lr: float, batch_size: int, 
                          base_batch_size: int = 256) -> float:
    """Adjust learning rate based on batch size.
    
    Following the paper: base learning rates of 1e-4 for a 256 batch size
    are adjusted according to batch sizes and model sizes.
    """
    return base_lr * (batch_size / base_batch_size) ** 0.5


def cosine_schedule_with_warmup(optimizer, 
                                 current_step: int,
                                 total_steps: int,
                                 warmup_steps: int,
                                 base_lr: float,
                                 min_lr: float = 0.0):
    """Cosine learning rate schedule with linear warmup.
    
    10% of total steps used for linear warmup (as per paper).
    """
    if current_step < warmup_steps:
        # Linear warmup
        lr = base_lr * current_step / warmup_steps
    else:
        # Cosine decay
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr


def train_p2vae(
    model: P2VAE,
    train_dataloader,
    val_dataloader=None,
    total_steps: int = 100000,
    batch_size: int = 256,
    base_lr: float = 1e-4,
    weight_decay: float = 1e-4,
    beta1: float = 0.9,
    beta2: float = 0.995,
    warmup_ratio: float = 0.1,
    device: str = 'cuda',
    use_fp16: bool = True,
    log_interval: int = 100,
    save_interval: int = 5000,
    checkpoint_dir: Optional[str] = None,
):
    """Train P2VAE (Stage 1 of two-stage recipe).
    
    Args:
        model: P2VAE model instance
        train_dataloader: DataLoader for training data
        val_dataloader: Optional DataLoader for validation
        total_steps: Total training steps (100k per paper)
        batch_size: Batch size
        base_lr: Base learning rate for batch_size=256
        weight_decay: Weight decay for AdamW
        beta1: AdamW beta1
        beta2: AdamW beta2
        warmup_ratio: Fraction of steps for warmup
        device: Training device
        use_fp16: Use mixed precision training
        log_interval: Steps between logging
        save_interval: Steps between checkpoint saves
        checkpoint_dir: Directory for saving checkpoints
    """
    model = model.to(device)
    model.train()
    
    # Adjust learning rate for batch size
    lr = adjust_learning_rate(base_lr, batch_size)
    
    # AdamW optimizer as specified in paper
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
    )
    
    # Cosine schedule with warmup
    warmup_steps = int(total_steps * warmup_ratio)
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    
    # Training loop
    global_step = 0
    data_iter = iter(train_dataloader)
    
    losses = {'total': [], 'recon': [], 'kl': []}
    
    while global_step < total_steps:
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dataloader)
            batch = next(data_iter)
        
        # Extract frames from batch
        # Batch format: (B, T, C, H, W) or list of frames
        if isinstance(batch, torch.Tensor):
            # Single frame per batch item
            x = batch.to(device)
        elif isinstance(batch, (list, tuple)):
            x = batch[0].to(device)  # Use first frame
        else:
            raise ValueError(f"Unexpected batch format: {type(batch)}")
        
        # Update learning rate
        current_lr = cosine_schedule_with_warmup(
            optimizer, global_step, total_steps, warmup_steps, lr
        )
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=use_fp16):
            loss_dict = model.compute_loss(x)
            loss = loss_dict['loss']
        
        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Logging
        losses['total'].append(loss_dict['loss'].item())
        losses['recon'].append(loss_dict['recon_loss'].item())
        losses['kl'].append(loss_dict['kl_loss'].item())
        
        if global_step % log_interval == 0:
            avg_loss = sum(losses['total'][-log_interval:]) / log_interval
            avg_recon = sum(losses['recon'][-log_interval:]) / log_interval
            avg_kl = sum(losses['kl'][-log_interval:]) / log_interval
            logger.info(
                f"Step {global_step}/{total_steps} | "
                f"LR: {current_lr:.2e} | "
                f"Loss: {avg_loss:.6f} | "
                f"Recon: {avg_recon:.6f} | "
                f"KL: {avg_kl:.6f}"
            )
        
        # Save checkpoint
        if checkpoint_dir and global_step % save_interval == 0:
            checkpoint_path = f"{checkpoint_dir}/p2vae_step{global_step}.pt"
            torch.save({
                'step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        global_step += 1
    
    logger.info("P2VAE training completed!")
    return model


def compute_reconstruction_error(
    model: P2VAE,
    dataloader,
    device: str = 'cuda',
    use_fp16: bool = True,
) -> Dict[str, float]:
    """Compute L2 relative error and VRMSE for P2VAE.
    
    L2RE = ||x - x_hat||_2 / ||x||_2
    VRMSE = RMSE(x, x_hat) / std(x)
    """
    model.eval()
    total_l2re = 0.0
    total_vrmse = 0.0
    count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, torch.Tensor):
                x = batch.to(device)
            else:
                x = batch[0].to(device)
            
            with torch.cuda.amp.autocast(enabled=use_fp16):
                output = model.forward(x, sample_posterior=False)
                x_hat = output['reconstruction']
            
            B = x.shape[0]
            
            # L2RE
            l2re = torch.norm(x - x_hat, p=2, dim=[1, 2, 3]) / \
                   torch.norm(x, p=2, dim=[1, 2, 3])
            total_l2re += l2re.sum().item()
            
            # VRMSE
            std_x = torch.std(x, dim=[1, 2, 3])
            rmse = torch.sqrt(torch.mean((x - x_hat) ** 2, dim=[1, 2, 3]))
            vrmse = rmse / (std_x + 1e-8)
            total_vrmse += vrmse.sum().item()
            
            count += B
    
    return {
        'L2RE': total_l2re / count,
        'VRMSE': total_vrmse / count,
    }
