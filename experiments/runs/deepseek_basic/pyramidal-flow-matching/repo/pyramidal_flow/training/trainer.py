"""
Pyramidal Flow Trainer.

Implements the three-stage training procedure described in Section 4.1 and
Appendix B. Supports:
- Stage 1: Image pre-training (50k steps, 1536 A100 GPU hours)
- Stage 2: Low-resolution video training (200k steps, 11,520 GPU hours)
- Stage 3: High-resolution video fine-tuning (50k steps, 7,680 GPU hours)

Uses Patch n' Pack for length-balanced batching and supports
joint image-video training with 12.5% image data in video stages.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional, Dict, Any, Iterator
import math

from ..pyramidal_flow import PyramidalFlowMatching
from .config import TrainingConfig


def constant_with_warmup_schedule(step: int, warmup_steps: int) -> float:
    """Learning rate schedule: constant with linear warmup."""
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    return 1.0


class PyramidalFlowTrainer:
    """
    Trainer for Pyramidal Flow Matching.
    
    Manages the training loop across three stages with appropriate
    hyperparameter adjustments.
    
    Args:
        model: The PyramidalFlowMatching model
        config: Training configuration
    """
    
    def __init__(
        self,
        model: PyramidalFlowMatching,
        config: TrainingConfig,
    ):
        self.model = model
        self.config = config
        self.current_stage = 1
        self.global_step = 0
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.epsilon,
            weight_decay=config.weight_decay,
        )
        
        # LR scheduler
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: constant_with_warmup_schedule(
                step % config.max_steps, config.warmup_steps
            ),
        )
        
        # Automatic mixed precision
        self.scaler = torch.amp.GradScaler() if config.numerical_precision == 'float16' else None
        
        # Training statistics
        self.stats = {
            'loss': [],
            'grad_norm': [],
        }
    
    def train_step(
        self,
        x1: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
        past_frames: Optional[list] = None,
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            x1: Clean data latent
            conditioning: Text embeddings
            past_frames: Previous frames for autoregressive training
            
        Returns:
            Dict with loss and gradient norm
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Compute flow matching loss
        with torch.amp.autocast(
            device_type='cuda',
            dtype=torch.bfloat16 if self.config.numerical_precision == 'bfloat16' else torch.float32,
        ) if self.scaler is not None else torch.enable_grad():
            loss_dict = self.model.compute_loss(
                x1=x1,
                conditioning=conditioning,
                past_frames=past_frames,
                noise_strength_range=(
                    self.config.history_noise_min,
                    self.config.history_noise_max,
                ),
            )
            loss = loss_dict['loss']
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clipping
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clipping
            )
            self.optimizer.step()
        
        self.scheduler.step()
        self.global_step += 1
        
        # Log stats
        self.stats['loss'].append(loss.item())
        self.stats['grad_norm'].append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
        
        return {
            'loss': loss.item(),
            'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            'lr': self.scheduler.get_last_lr()[0],
        }
    
    def train_stage(
        self,
        stage: int,
        dataloader: Iterator,
        total_steps: int,
        log_every: int = 100,
        save_every: int = 5000,
        checkpoint_dir: Optional[str] = None,
    ):
        """
        Train a single stage.
        
        Args:
            stage: Stage number (1, 2, or 3)
            dataloader: Data iterator
            total_steps: Total training steps for this stage
            log_every: Logging interval
            save_every: Checkpoint interval
            checkpoint_dir: Directory for saving checkpoints
        """
        # Update config for this stage
        stage_config = self.config.get_stage_config(stage)
        self.current_stage = stage
        
        # Reinitialize optimizer with stage-specific settings
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=stage_config.learning_rate,
            betas=(stage_config.beta1, stage_config.beta2),
            eps=stage_config.epsilon,
            weight_decay=stage_config.weight_decay,
        )
        
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: constant_with_warmup_schedule(
                step, stage_config.warmup_steps
            ),
        )
        
        print(f"Starting stage {stage} training: {total_steps} steps")
        print(f"  Learning rate: {stage_config.learning_rate}")
        print(f"  Batch size: {stage_config.global_batch_size}")
        print(f"  Beta2: {stage_config.beta2}")
        print(f"  Estimated GPU hours: {stage_config.get_gpu_hours(stage)}")
        
        for step in range(total_steps):
            batch = next(dataloader)
            
            x1 = batch.get('latent')
            conditioning = batch.get('text_embeddings')
            past_frames = batch.get('past_frames')
            
            stats = self.train_step(x1, conditioning, past_frames)
            
            if step % log_every == 0:
                print(f"  Step {step}/{total_steps}: loss={stats['loss']:.6f}, "
                      f"grad_norm={stats['grad_norm']:.4f}, lr={stats['lr']:.2e}")
            
            if checkpoint_dir and step % save_every == 0 and step > 0:
                self.save_checkpoint(f"{checkpoint_dir}/stage{stage}_step{step}.pt")
        
        if checkpoint_dir:
            self.save_checkpoint(f"{checkpoint_dir}/stage{stage}_final.pt")
        
        print(f"Stage {stage} complete.")
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'current_stage': self.current_stage,
            'config': self.config,
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.current_stage = checkpoint.get('current_stage', 1)
        print(f"Checkpoint loaded from {path} (step {self.global_step})")
    
    def get_average_loss(self, window: int = 100) -> float:
        """Get average loss over last N steps."""
        if not self.stats['loss']:
            return 0.0
        recent = self.stats['loss'][-window:]
        return sum(recent) / len(recent)
