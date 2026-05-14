"""
Training utilities for Hi-MAR.

Implements the training loop with:
- AdamW optimizer with β1=0.9, β2=0.95
- Weight decay
- Constant lr schedule with linear warmup
- EMA (exponential moving average)
- Classifier-free guidance dropout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import time
import os
from typing import Optional, Dict, Any
from collections import OrderedDict


class EMA:
    """Exponential Moving Average for model parameters."""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register()

    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                    self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply EMA weights to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original model weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self):
        return {'shadow': self.shadow, 'decay': self.decay}

    def load_state_dict(self, state_dict):
        self.shadow = state_dict['shadow']
        self.decay = state_dict['decay']


class WarmupCosineSchedule:
    """Learning rate schedule with linear warmup and constant lr."""
    def __init__(self, optimizer, warmup_steps, total_steps, base_lr, min_lr=0.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def get_lr(self):
        if self.current_step < self.warmup_steps:
            return self.base_lr * self.current_step / self.warmup_steps
        else:
            return self.base_lr


class ConstantLRSchedule:
    """Constant learning rate with linear warmup (as used in Hi-MAR paper)."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, steps_per_epoch):
        self.optimizer = optimizer
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
        self.base_lr = base_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def get_lr(self):
        if self.current_step < self.warmup_steps:
            return self.base_lr * self.current_step / max(1, self.warmup_steps)
        else:
            return self.base_lr


class HiMARTrainer:
    """
    Trainer for Hi-MAR model.
    
    Supports both class-conditional (ImageNet) and text-to-image (MS-COCO) training.
    """
    def __init__(
        self,
        model,
        # Optimizer
        learning_rate=1e-4,
        weight_decay=0.02,
        beta1=0.9,
        beta2=0.95,
        # Schedule
        warmup_epochs=100,
        total_epochs=800,
        # EMA
        ema_decay=0.9999,
        # CFG
        cfg_drop_prob=0.1,
        # Logging
        log_interval=100,
        save_interval=50,
        output_dir='./checkpoints',
        # Mixed precision
        use_amp=False,
        # Gradient accumulation
        gradient_accumulation_steps=1,
        # Max grad norm
        max_grad_norm=1.0,
    ):
        self.model = model
        self.device = next(model.parameters()).device
        self.cfg_drop_prob = cfg_drop_prob
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.output_dir = output_dir
        self.use_amp = use_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        
        # Optimizer
        if weight_decay > 0:
            param_groups = model.get_param_groups(weight_decay=weight_decay)
        else:
            param_groups = model.parameters()
        
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=learning_rate,
            betas=(beta1, beta2),
            weight_decay=weight_decay if weight_decay > 0 else 0.0,
        )
        
        self.lr_schedule = None  # Set when training starts
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = learning_rate
        
        # EMA
        self.ema = EMA(model, decay=ema_decay) if ema_decay > 0 else None
        
        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        # Stats
        self.global_step = 0
        self.current_epoch = 0
        
        os.makedirs(output_dir, exist_ok=True)

    def train_epoch(self, dataloader, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_loss1 = 0.0
        total_loss2 = 0.0
        num_batches = 0
        
        if self.lr_schedule is None:
            steps_per_epoch = len(dataloader)
            self.lr_schedule = ConstantLRSchedule(
                self.optimizer,
                self.warmup_epochs,
                self.total_epochs,
                self.base_lr,
                steps_per_epoch,
            )
            # Advance to current epoch
            for _ in range(epoch * steps_per_epoch):
                self.lr_schedule.step()
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(dataloader):
            # Parse batch
            if len(batch) == 2:
                # Class-conditional: (x_low, x_high), class_idx
                (x_low, x_high), class_idx = batch
                context_embeds = None
            elif len(batch) == 3:
                # Text-to-image: (x_low, x_high), class_idx, text_embeds
                (x_low, x_high), class_idx, context_embeds = batch
            else:
                raise ValueError(f"Unexpected batch format: {len(batch)} items")
            
            x_low = x_low.to(self.device)
            x_high = x_high.to(self.device)
            class_idx = class_idx.to(self.device) if class_idx is not None else None
            if context_embeds is not None:
                context_embeds = context_embeds.to(self.device)
            
            # CFG dropout: randomly drop class labels
            if class_idx is not None and self.cfg_drop_prob > 0:
                drop_mask = torch.rand(class_idx.shape[0], device=self.device) < self.cfg_drop_prob
                class_idx = class_idx.clone()
                class_idx[drop_mask] = -1  # Use -1 as unconditional indicator
                # Replace -1 with None handling in model (model ignores class_idx if -1)
            
            # Forward pass
            with torch.cuda.amp.autocast() if self.use_amp else torch.enable_grad():
                total_loss_batch, loss_dict = self.model(
                    x_low=x_low,
                    x_high=x_high,
                    class_idx=class_idx if class_idx is not None else None,
                    context_embeds=context_embeds,
                )
                loss = total_loss_batch / self.gradient_accumulation_steps
            
            # Backward
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Step
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Update EMA
                if self.ema is not None:
                    self.ema.update()
                
                # Update LR
                self.lr_schedule.step()
                self.global_step += 1
            
            total_loss += total_loss_batch.item()
            total_loss1 += loss_dict['loss_phase1'].item()
            total_loss2 += loss_dict['loss_phase2'].item()
            num_batches += 1
            
            if batch_idx % self.log_interval == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                      f"Loss: {total_loss_batch.item():.4f} "
                      f"(P1: {loss_dict['loss_phase1'].item():.4f}, "
                      f"P2: {loss_dict['loss_phase2'].item():.4f}) "
                      f"LR: {lr:.2e}")
        
        avg_loss = total_loss / num_batches
        avg_loss1 = total_loss1 / num_batches
        avg_loss2 = total_loss2 / num_batches
        
        return {
            'loss': avg_loss,
            'loss_phase1': avg_loss1,
            'loss_phase2': avg_loss2,
        }

    def save_checkpoint(self, epoch, metrics=None, filename=None):
        """Save training checkpoint."""
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'metrics': metrics,
        }
        
        if self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        path = os.path.join(self.output_dir, filename)
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        
        if 'ema_state_dict' in checkpoint and self.ema is not None:
            self.ema.load_state_dict(checkpoint['ema_state_dict'])
        
        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Loaded checkpoint from {path}")
        return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})

    def train(self, dataloader, start_epoch=0, num_epochs=None):
        """Full training loop."""
        if num_epochs is None:
            num_epochs = self.total_epochs
        
        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            metrics = self.train_epoch(dataloader, epoch)
            
            print(f"Epoch {epoch} completed. "
                  f"Avg Loss: {metrics['loss']:.4f} "
                  f"(P1: {metrics['loss_phase1']:.4f}, P2: {metrics['loss_phase2']:.4f})")
            
            # Save checkpoint
            if (epoch + 1) % self.save_interval == 0 or epoch == num_epochs - 1:
                self.save_checkpoint(epoch + 1, metrics)
        
        # Save final model
        self.save_checkpoint(num_epochs, filename="final_model.pt")


def create_trainer(config, model):
    """Create a HiMARTrainer from configuration."""
    trainer_kwargs = {
        'learning_rate': config.get('learning_rate', 1e-4),
        'weight_decay': config.get('weight_decay', 0.02),
        'beta1': config.get('beta1', 0.9),
        'beta2': config.get('beta2', 0.95),
        'warmup_epochs': config.get('warmup_epochs', 100),
        'total_epochs': config.get('total_epochs', 800),
        'ema_decay': config.get('ema_decay', 0.9999),
        'cfg_drop_prob': config.get('cfg_drop_prob', 0.1),
        'log_interval': config.get('log_interval', 100),
        'save_interval': config.get('save_interval', 50),
        'output_dir': config.get('output_dir', './checkpoints'),
        'use_amp': config.get('use_amp', False),
        'gradient_accumulation_steps': config.get('gradient_accumulation_steps', 1),
        'max_grad_norm': config.get('max_grad_norm', 1.0),
    }
    
    return HiMARTrainer(model, **trainer_kwargs)
