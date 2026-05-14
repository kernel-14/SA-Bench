"""Training loop for pyramidal flow matching.

Implements the three-stage training procedure:
1. Image training (50k steps, 1536 batch size)
2. Low-resolution video training (200k steps, 768 batch size)
3. High-resolution video training (50k steps, 384 batch size)

Key training details:
- AdamW optimizer with stage-specific hyperparameters
- Constant LR with warmup
- Gradient clipping at 1.0
- bfloat16 precision
- 128 NVIDIA A100 GPUs
- CFG dropout at 10% probability
- Patch n' Pack for variable-length batching
"""

import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from config import ModelConfig, TrainingConfig
from data.dataset import MixedDataset, build_dataloader
from data.text_encoder import TextEncoder
from model.dit import MMDiT
from model.vae import VideoVAE
from pyramid_flow.spatial_pyramid import SpatialPyramidFlow
from pyramid_flow.temporal_pyramid import TemporalPyramid

logger = logging.getLogger(__name__)


def get_warmup_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Constant LR with linear warmup."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


class PyramidFlowTrainer:
    """Trainer for pyramidal flow matching.

    Handles the full training pipeline including:
    - VAE encoding of input images/videos
    - Pyramid stage sampling
    - Flow matching loss computation
    - Gradient updates with mixed precision
    - Checkpoint saving and loading
    """

    def __init__(
        self,
        config: ModelConfig,
        dit: MMDiT,
        vae: VideoVAE,
        text_encoder: TextEncoder,
        rank: int = 0,
        world_size: int = 1,
        device: torch.device = None,
    ):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = device or torch.device(f"cuda:{rank}")

        self.dit = dit.to(self.device)
        self.vae = vae.to(self.device)
        self.text_encoder = text_encoder.to(self.device)

        # Freeze VAE and text encoder
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

        # Pyramid algorithms
        self.spatial_pyramid = SpatialPyramidFlow(
            num_stages=config.pyramid.num_stages,
            stage_range=config.pyramid.stage_range,
            upsample_mode=config.pyramid.upsample_mode,
            downsample_mode=config.pyramid.downsample_mode,
        )
        self.temporal_pyramid = TemporalPyramid(
            num_stages=config.pyramid.num_stages,
            history_noise_max=config.pyramid.history_noise_max,
            downsample_mode=config.pyramid.downsample_mode,
        )

        # Wrap DiT in DDP if distributed
        if world_size > 1:
            self.dit = DDP(self.dit, device_ids=[rank], find_unused_parameters=False)

        self.global_step = 0
        self.scaler = GradScaler(enabled=True)

    def setup_optimizer(
        self,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
        warmup_steps: int,
        total_steps: int,
    ) -> Tuple[AdamW, LambdaLR]:
        """Setup AdamW optimizer and LR scheduler."""
        # Separate weight decay for different parameter groups
        decay_params = []
        no_decay_params = []

        model = self.dit.module if hasattr(self.dit, "module") else self.dit
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "embedding" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = AdamW(
            param_groups,
            lr=lr,
            betas=(beta1, beta2),
            eps=eps,
        )
        scheduler = get_warmup_schedule(optimizer, warmup_steps, total_steps)
        return optimizer, scheduler

    @torch.no_grad()
    def encode_to_latent(self, pixel_values: torch.Tensor, is_video: bool) -> torch.Tensor:
        """Encode pixel values to latent space using the frozen VAE."""
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if is_video:
                # pixel_values: (B, C, T, H, W)
                latent = self.vae.encode(pixel_values, sample=False)
            else:
                # pixel_values: (B, C, H, W) -> add temporal dim
                pixel_values = pixel_values.unsqueeze(2)  # (B, C, 1, H, W)
                latent = self.vae.encode(pixel_values, sample=False)
                latent = latent.squeeze(2)  # (B, C, H, W)
        return latent

    def compute_loss(
        self,
        latent: torch.Tensor,
        t5_embeds: torch.Tensor,
        clip_pooled: torch.Tensor,
        is_video: bool,
        stage: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute the pyramidal flow matching loss.

        For each training step:
        1. Sample a pyramid stage uniformly
        2. Sample training pair (x_start, x_end) with coupled noise
        3. Interpolate to get x_t at random timestep within stage
        4. Predict velocity and compute MSE loss

        Args:
            latent: (B, C, H, W) or (B, C, T, H, W) clean latent
            t5_embeds: (B, L, 4096) T5 text embeddings
            clip_pooled: (B, 2048) CLIP pooled embeddings
            is_video: whether input is video
            stage: optional fixed stage (for ablation); if None, sample uniformly

        Returns:
            loss: scalar flow matching loss
        """
        B = latent.shape[0]

        if is_video:
            return self._compute_video_loss(latent, t5_embeds, clip_pooled, stage)
        else:
            return self._compute_image_loss(latent, t5_embeds, clip_pooled, stage)

    def _compute_image_loss(
        self,
        latent: torch.Tensor,
        t5_embeds: torch.Tensor,
        clip_pooled: torch.Tensor,
        stage: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute flow matching loss for image generation."""
        B = latent.shape[0]

        # Sample pyramid stage uniformly
        if stage is None:
            stage = torch.randint(0, self.spatial_pyramid.num_stages, (1,)).item()

        # Sample training pair with coupled noise
        x_start, x_end, target_velocity, t_prime = (
            self.spatial_pyramid.sample_training_pair(latent, stage)
        )

        # Interpolate to get x_t
        x_t = self.spatial_pyramid.interpolate_within_stage(x_start, x_end, t_prime)

        # Rescale timestep to absolute value within stage
        s_k, e_k = self.spatial_pyramid.stage_range[stage]
        t_abs = s_k + t_prime * (e_k - s_k)
        t_batch = t_abs.to(latent.device)

        # Predict velocity
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            velocity = self._get_dit_module()(
                x_t,
                t_batch,
                t5_embeds,
                clip_pooled,
                num_frames=1,
            )

        # Flow matching loss
        loss = F.mse_loss(velocity, target_velocity)
        return loss

    def _compute_video_loss(
        self,
        latent: torch.Tensor,
        t5_embeds: torch.Tensor,
        clip_pooled: torch.Tensor,
        stage: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute flow matching loss for video generation with temporal pyramid."""
        B, C, T, H, W = latent.shape

        # Sample pyramid stage uniformly
        if stage is None:
            stage = torch.randint(0, self.spatial_pyramid.num_stages, (1,)).item()

        # For video, we generate a chunk of frames autoregressively
        # Split latent into history and current frames
        # Use first T//2 frames as history, last T//2 as current generation
        split = max(1, T // 2)
        history_latents = [latent[:, :, t] for t in range(split)]
        current_latents = latent[:, :, split:]  # (B, C, T_curr, H, W)

        T_curr = current_latents.shape[2]
        total_loss = torch.tensor(0.0, device=latent.device)

        # Process each current frame
        for frame_idx in range(T_curr):
            frame_latent = current_latents[:, :, frame_idx]  # (B, C, H, W)

            # Sample training pair for this frame
            x_start, x_end, target_velocity, t_prime = (
                self.spatial_pyramid.sample_training_pair(frame_latent, stage)
            )
            x_t = self.spatial_pyramid.interpolate_within_stage(x_start, x_end, t_prime)

            # Build temporal pyramid history
            all_history = history_latents + [current_latents[:, :, i] for i in range(frame_idx)]
            history_compressed, hist_indices = (
                self.temporal_pyramid.build_pyramid_history_sequence(
                    all_history, current_stage=stage, training=True
                )
            )

            s_k, e_k = self.spatial_pyramid.stage_range[stage]
            t_abs = s_k + t_prime * (e_k - s_k)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                velocity = self._get_dit_module()(
                    x_t,
                    t_abs.to(latent.device),
                    t5_embeds,
                    clip_pooled,
                    num_frames=1,
                    history_frames=history_compressed,
                    history_frame_indices=hist_indices,
                )

            total_loss = total_loss + F.mse_loss(velocity, target_velocity)

        return total_loss / T_curr

    def _get_dit_module(self) -> MMDiT:
        """Get the underlying DiT module (unwrap DDP if needed)."""
        if hasattr(self.dit, "module"):
            return self.dit.module
        return self.dit

    def apply_cfg_dropout(
        self,
        t5_embeds: torch.Tensor,
        clip_pooled: torch.Tensor,
        dropout_prob: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply classifier-free guidance dropout during training.

        Randomly replaces text embeddings with null embeddings.
        """
        B = t5_embeds.shape[0]
        mask = torch.rand(B, device=t5_embeds.device) < dropout_prob

        null_t5 = torch.zeros_like(t5_embeds)
        null_clip = torch.zeros_like(clip_pooled)

        t5_embeds = torch.where(mask.view(B, 1, 1), null_t5, t5_embeds)
        clip_pooled = torch.where(mask.view(B, 1), null_clip, clip_pooled)

        return t5_embeds, clip_pooled

    def train_step(
        self,
        batch: Dict,
        optimizer: AdamW,
        scheduler: LambdaLR,
        grad_clip: float = 1.0,
        cfg_dropout_prob: float = 0.1,
    ) -> Dict[str, float]:
        """Single training step."""
        pixel_values = batch["pixel_values"].to(self.device)
        captions = batch["captions"]
        is_video_list = batch["is_video"]
        is_video = any(is_video_list)

        # Encode text
        with torch.no_grad():
            t5_embeds, clip_pooled = self.text_encoder.encode_text(captions, self.device)

        # Apply CFG dropout
        t5_embeds, clip_pooled = self.apply_cfg_dropout(
            t5_embeds, clip_pooled, cfg_dropout_prob
        )

        # Encode to latent space
        with torch.no_grad():
            latent = self.encode_to_latent(pixel_values, is_video)

        # Compute loss
        optimizer.zero_grad()
        loss = self.compute_loss(latent, t5_embeds, clip_pooled, is_video)

        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self._get_dit_module().parameters(), grad_clip
        )
        self.scaler.step(optimizer)
        self.scaler.update()
        scheduler.step()

        self.global_step += 1

        return {
            "loss": loss.item(),
            "grad_norm": grad_norm.item(),
            "lr": scheduler.get_last_lr()[0],
        }

    def save_checkpoint(self, output_dir: str, step: int):
        """Save model checkpoint."""
        if self.rank != 0:
            return

        output_path = Path(output_dir) / f"checkpoint-{step}"
        output_path.mkdir(parents=True, exist_ok=True)

        model = self._get_dit_module()
        torch.save(
            {
                "step": step,
                "model_state_dict": model.state_dict(),
                "config": self.config,
            },
            output_path / "model.pt",
        )
        logger.info(f"Saved checkpoint at step {step} to {output_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model = self._get_dit_module()
        model.load_state_dict(checkpoint["model_state_dict"])
        self.global_step = checkpoint.get("step", 0)
        logger.info(f"Loaded checkpoint from {checkpoint_path}, step={self.global_step}")
