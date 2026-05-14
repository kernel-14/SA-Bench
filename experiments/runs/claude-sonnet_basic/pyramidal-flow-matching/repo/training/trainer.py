"""
Training framework for Pyramidal Flow Matching.

Implements the three-stage training procedure:
1. Image Training (50k steps, ~1536 A100 GPU hours)
2. Low-Resolution Video Training (200k steps, ~11520 A100 GPU hours)
3. High-Resolution Video Training (50k steps, ~7680 A100 GPU hours)

Key training details from the paper:
- AdamW optimizer with different beta values per stage
- Constant learning rate with warmup
- Gradient clipping at 1.0
- bfloat16 precision
- 128 NVIDIA A100 GPUs
- Patch n' Pack for length-balanced batches
- Uniform sampling of pyramid stages per iteration
- 12.5% image data mixed in video training batches
"""

import os
import math
import logging
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler, autocast

from ..models.pyramid_dit import PyramidDiT
from ..models.pyramidal_flow import PyramidalFlowMatching, TemporalPyramidCondition

logger = logging.getLogger(__name__)


def get_warmup_constant_schedule(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
) -> LambdaLR:
    """
    Constant learning rate schedule with linear warmup.
    
    As described in the paper: "Constant with warmup" for all stages.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0
    
    return LambdaLR(optimizer, lr_lambda)


class PyramidFlowTrainer:
    """
    Trainer for Pyramidal Flow Matching video generation model.
    
    Implements the unified training objective that jointly optimizes
    all pyramid stages in a single DiT model.
    """
    
    def __init__(
        self,
        model: PyramidDiT,
        vae: nn.Module,
        config: Dict[str, Any],
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
    ):
        """
        Args:
            model: The Pyramid DiT model
            vae: The 3D VAE for encoding/decoding
            config: Training configuration dictionary
            device: Training device
            rank: Process rank for distributed training
            world_size: Total number of processes
        """
        self.model = model
        self.vae = vae
        self.config = config
        self.device = device
        self.rank = rank
        self.world_size = world_size
        
        # Initialize pyramidal flow matching algorithm
        self.pyramid_flow = PyramidalFlowMatching(
            num_stages=config.get('num_pyramid_stages', 3),
        )
        
        # Initialize temporal pyramid condition
        self.temporal_pyramid = TemporalPyramidCondition(
            noise_strength_range=(0.0, 1/3),  # As per paper
        )
        
        # Setup optimizer
        self._setup_optimizer()
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.global_step = 0
        self.current_stage_name = config.get('training_stage', 'image')
        
        # Classifier-free guidance dropout probability
        self.cfg_dropout_prob = config.get('cfg_dropout_prob', 0.1)
    
    def _setup_optimizer(self):
        """Setup AdamW optimizer with stage-specific hyperparameters."""
        stage = self.config.get('training_stage', 'image')
        lr = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        # Different beta values for different stages (from Table 4)
        if stage == 'image':
            betas = (0.9, 0.999)
        else:
            betas = (0.9, 0.95)
        
        eps = 1e-6
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        
        # Learning rate scheduler
        warmup_steps = self.config.get('warmup_steps', 1000)
        self.scheduler = get_warmup_constant_schedule(self.optimizer, warmup_steps)
    
    def encode_latents(
        self,
        videos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode videos to latent space using the 3D VAE.
        
        Args:
            videos: (B, C, T, H, W) video tensor
        
        Returns:
            Latent tensor (B, latent_channels, T//8, H//8, W//8)
        """
        with torch.no_grad():
            latents = self.vae.encode_video(videos)
        return latents
    
    def prepare_text_embeddings(
        self,
        text_embeds_t5: torch.Tensor,
        text_embeds_clip: torch.Tensor,
        dropout: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare text embeddings with optional classifier-free guidance dropout.
        
        Args:
            text_embeds_t5: T5 embeddings (B, L, D)
            text_embeds_clip: CLIP embeddings (B, D)
            dropout: Whether to apply CFG dropout
        
        Returns:
            Processed (t5_embeds, clip_embeds)
        """
        if dropout and self.training:
            # Randomly drop text conditioning for CFG training
            mask = torch.rand(text_embeds_t5.shape[0]) > self.cfg_dropout_prob
            mask = mask.to(text_embeds_t5.device)
            
            text_embeds_t5 = text_embeds_t5 * mask.view(-1, 1, 1)
            text_embeds_clip = text_embeds_clip * mask.view(-1, 1)
        
        return text_embeds_t5, text_embeds_clip
    
    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        is_video: bool = False,
    ) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Implements the unified pyramidal flow matching objective:
        E_{k,t,(x_ek, x_sk)} || v_t(x_t) - (x_ek - x_sk) ||^2
        
        Args:
            batch: Dictionary containing:
                - 'latents': Pre-encoded latents or raw video/images
                - 'text_embeds_t5': T5 text embeddings
                - 'text_embeds_clip': CLIP text embeddings
                - 'history_latents': Optional history latents for video
            is_video: Whether this is a video training batch
        
        Returns:
            Dictionary of loss values
        """
        self.model.train()
        
        latents = batch['latents'].to(self.device)
        text_embeds_t5 = batch['text_embeds_t5'].to(self.device)
        text_embeds_clip = batch['text_embeds_clip'].to(self.device)
        
        # Apply CFG dropout
        text_embeds_t5, text_embeds_clip = self.prepare_text_embeddings(
            text_embeds_t5, text_embeds_clip, dropout=True
        )
        
        # Uniformly sample a pyramid stage for this iteration
        # (as described in Section 3.4: "different pyramidal stages are uniformly sampled")
        stage = torch.randint(0, self.pyramid_flow.num_stages, (1,)).item()
        
        # Sample training pair for this stage
        x_sk, x_ek, x_t, t_prime, target_velocity = self.pyramid_flow.sample_training_pair(
            latents, stage
        )
        
        # Prepare history condition for video training
        history_tokens = None
        if is_video and 'history_latents' in batch:
            history_latents = batch['history_latents']
            if isinstance(history_latents, list):
                history_latents = [h.to(self.device) for h in history_latents]
            
            # Prepare temporal pyramid history
            compressed_history = self.temporal_pyramid.prepare_history_condition(
                history_latents,
                current_stage=stage,
                num_pyramid_stages=self.pyramid_flow.num_stages,
                training=True,
            )
            
            # Patchify history frames (would need model's patchify method)
            # For simplicity, we pass the compressed latents directly
            history_tokens = compressed_history
        
        # Compute timestep for this stage
        # t_prime is in [0, 1] within the stage, convert to global timestep
        s_k, e_k = self.pyramid_flow.stage_time_windows[stage]
        t_global = s_k + t_prime * (e_k - s_k)
        
        # Forward pass with mixed precision
        with autocast(enabled=self.use_amp, dtype=torch.bfloat16):
            predicted_velocity = self.model(
                x_t,
                t_global,
                text_embeds_t5,
                text_embeds_clip,
                pyramid_stage=stage,
                history_tokens=history_tokens,
                use_causal_attention=is_video,
            )
            
            # Compute flow matching loss
            loss = self.pyramid_flow.compute_flow_loss(predicted_velocity, target_velocity)
        
        # Backward pass
        self.optimizer.zero_grad()
        
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get('gradient_clip', 1.0)
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get('gradient_clip', 1.0)
            )
            self.optimizer.step()
        
        self.scheduler.step()
        self.global_step += 1
        
        return {
            'loss': loss.item(),
            'stage': stage,
            'lr': self.scheduler.get_last_lr()[0],
        }
    
    def save_checkpoint(self, save_dir: str, step: Optional[int] = None):
        """Save model checkpoint."""
        step = step or self.global_step
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        path = os.path.join(save_dir, f'checkpoint_{step:08d}.pt')
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['step']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path} (step {self.global_step})")
