"""
Training script for Ca2-VDM.

Implements the two-stage training procedure:
1. Stage 1: Train causal modeling ability without clean prefix on shorter videos
2. Stage 2: Train with clean prefix on longer videos, with cyclic TPEs

Training configurations follow the paper:
- Text-to-Video (T2V): InternVid subset, 4.9M videos, 256x256
  Stage 1: 32-frame videos, batch size 288, 32k steps
  Stage 2: 65-frame videos (P_max=49, l=16), batch size 144, 21k steps
- Video Prediction: SkyTimelapse, 256x256
  l=8, P_max=25, L_train=33, batch size 8, 11k steps

Hyperparameters:
- DDPM schedule: T=1000, beta_1=1e-4, beta_T=0.02
- Optimizer: AdamW, learning rate 2e-5
- Classifier-free guidance scale: 7.5 (for T2V)
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from typing import Optional, Dict, Any
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ca2_vdm.model import Ca2VDM
from ca2_vdm.diffusion import DiffusionProcess
from ca2_vdm.tpe import CyclicTPE


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(name: str):
    """Get a logger instance."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


class Ca2VDMTrainer:
    """
    Trainer for Ca2-VDM.
    
    Implements the training procedure with:
    - Partial noising: randomly sampled prefix length P
    - Cyclic TPEs: randomly sampled offset
    - Distinct timestep embeddings for clean prefix and denoising target
    - Combined loss: L_simple + L_vlb
    """
    
    def __init__(
        self,
        model: Ca2VDM,
        diffusion: DiffusionProcess,
        device: torch.device,
        lr: float = 2e-5,
        weight_decay: float = 0.0,
        use_amp: bool = True,
        logger=None,
    ):
        self.model = model.to(device)
        self.diffusion = diffusion
        self.device = device
        self.use_amp = use_amp
        self.logger = logger or get_logger(__name__)
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )
        
        # Gradient scaler for mixed precision
        self.scaler = GradScaler() if use_amp else None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
    
    def _sample_prefix_length(
        self,
        l: int,
        P_max: int,
    ) -> int:
        """
        Randomly sample prefix length P.
        
        P ∈ {1, 1+l, ..., 1+n*l} where P_max = 1+n*l
        
        Args:
            l: chunk length
            P_max: maximum prefix length
        
        Returns:
            P: sampled prefix length
        """
        # Valid P values: 1, 1+l, 1+2l, ..., 1+n*l
        n_max = (P_max - 1) // l
        possible_P = [1 + i * l for i in range(n_max + 1)]
        
        # Ensure at least 1
        if len(possible_P) == 0:
            possible_P = [1]
        
        return random.choice(possible_P)
    
    def _sample_cyclic_offset(self, L_train: int) -> int:
        """Sample random offset for cyclic TPE."""
        return random.randint(0, L_train - 1)
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        stage: int = 2,
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: dict with 'video' (B, C, L, H, W) and optionally 'text_emb'
            stage: training stage (1 or 2)
        
        Returns:
            dict of loss values
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        video = batch['video'].to(self.device)  # (B, C, L, H, W)
        text_emb = batch.get('text_emb', None)
        if text_emb is not None:
            text_emb = text_emb.to(self.device)
        
        B, C, L, H, W = video.shape
        l = self.model.l
        P_max = self.model.P_max
        L_train = self.model.L_train
        
        if stage == 1:
            # Stage 1: No prefix, train causal modeling on shorter clips
            P = 0
            cyclic_offset = 0
        else:
            # Stage 2: Random prefix length, cyclic TPEs
            P = self._sample_prefix_length(l, P_max)
            cyclic_offset = self._sample_cyclic_offset(L_train)
        
        # Sample random timestep
        t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=self.device)
        
        # Compute loss
        with autocast(enabled=self.use_amp):
            loss_dict = self.diffusion.compute_loss(
                model=self.model,
                z_0=video,
                P=P,
                t=t,
                text_emb=text_emb,
                cyclic_offset=cyclic_offset,
                learn_sigma=self.model.learn_sigma,
                use_vlb_loss=self.model.use_vb_loss,
            )
        
        loss = loss_dict['loss']
        
        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        
        self.global_step += 1
        
        return {
            'loss': loss.item(),
            'mse_loss': loss_dict['mse_loss'].item(),
            'vlb_loss': loss_dict['vlb_loss'].item(),
        }
    
    def save_checkpoint(
        self,
        save_path: str,
        extra_info: Optional[Dict[str, Any]] = None,
    ):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
        }
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        if extra_info:
            checkpoint.update(extra_info)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(checkpoint, save_path)
        self.logger.info(f"Checkpoint saved to {save_path}")
    
    def load_checkpoint(self, load_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.logger.info(f"Checkpoint loaded from {load_path}")


def create_t2v_model():
    """Create Ca2-VDM model for text-to-video generation."""
    return Ca2VDM(
        in_channels=4,
        H=32, W=32,
        dim=1152,
        num_heads=16,
        num_layers=28,
        mlp_ratio=4.0,
        l=16,
        P_max=49,
        L_train=65,
        prefix_len=3,
        use_text_cond=True,
        text_dim=4096,
        dropout=0.0,
        learn_sigma=True,
        use_vb_loss=True,
    )


def create_vp_model():
    """Create Ca2-VDM model for video prediction (no text)."""
    return Ca2VDM(
        in_channels=4,
        H=32, W=32,
        dim=1152,
        num_heads=16,
        num_layers=28,
        mlp_ratio=4.0,
        l=8,
        P_max=25,
        L_train=33,
        prefix_len=3,
        use_text_cond=False,
        dropout=0.0,
        learn_sigma=True,
        use_vb_loss=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Train Ca2-VDM")
    parser.add_argument('--mode', type=str, choices=['t2v', 'vp'], default='t2v',
                        help='Training mode: t2v (text-to-video) or vp (video prediction)')
    parser.add_argument('--stage', type=int, choices=[1, 2], default=1,
                        help='Training stage: 1 (no prefix) or 2 (with prefix)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to training data directory')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides default)')
    parser.add_argument('--max_steps', type=int, default=None,
                        help='Maximum training steps (overrides default)')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained weights (e.g., Open-Sora)')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = get_logger('train')
    
    # Create model
    if args.mode == 't2v':
        model = create_t2v_model()
        default_batch_size = 288 if args.stage == 1 else 144
        default_max_steps = 32000 if args.stage == 1 else 21000
    else:
        model = create_vp_model()
        default_batch_size = 8
        default_max_steps = 11000
    
    batch_size = args.batch_size or default_batch_size
    max_steps = args.max_steps or default_max_steps
    
    # Load pretrained weights if provided
    if args.pretrained:
        logger.info(f"Loading pretrained weights from {args.pretrained}")
        state_dict = torch.load(args.pretrained, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    
    # Create diffusion process
    diffusion = DiffusionProcess(
        num_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        schedule='linear',
    )
    
    # Create trainer
    trainer = Ca2VDMTrainer(
        model=model,
        diffusion=diffusion,
        device=device,
        lr=args.lr,
        logger=logger,
    )
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Training loop (simplified - actual data loading depends on dataset)
    logger.info(f"Starting training: mode={args.mode}, stage={args.stage}")
    logger.info(f"Model: l={model.l}, P_max={model.P_max}, L_train={model.L_train}")
    logger.info(f"Batch size: {batch_size}, Max steps: {max_steps}")
    
    # Note: This is a skeleton. In practice, you would:
    # 1. Load the dataset (InternVid, SkyTimelapse, etc.)
    # 2. Create DataLoader with appropriate video preprocessing
    # 3. Run the training loop
    # 
    # The paper uses:
    # - InternVid subset with 4.9M video-text pairs
    # - SkyTimelapse with 997 training videos
    # 
    # Training expects video tensors of shape (B, C, L, H, W)
    # with L = L_train (stage 2) or L = 32 (stage 1 for T2V)


if __name__ == '__main__':
    main()
