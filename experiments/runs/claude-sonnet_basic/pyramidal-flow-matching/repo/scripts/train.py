"""
Main training script for Pyramidal Flow Matching.

Supports distributed training across multiple GPUs using PyTorch DDP.
Implements the three-stage training procedure from the paper.

Usage:
    # Single GPU
    python scripts/train.py --config configs/train_stage1_image.yaml
    
    # Multi-GPU (128 GPUs as in paper)
    torchrun --nproc_per_node=8 --nnodes=16 scripts/train.py \
        --config configs/train_stage1_image.yaml
"""

import os
import sys
import argparse
import logging
import yaml
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.pyramid_dit import PyramidDiT
from models.vae_3d import VideoVAE
from training.trainer import PyramidFlowTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_model(model_config: dict) -> PyramidDiT:
    """Build the Pyramid DiT model."""
    model = PyramidDiT(
        in_channels=model_config.get('in_channels', 16),
        hidden_dim=model_config.get('hidden_dim', 1536),
        num_layers=model_config.get('num_layers', 24),
        num_heads=model_config.get('num_heads', 24),
        mlp_ratio=model_config.get('mlp_ratio', 4.0),
        patch_size=model_config.get('patch_size', 2),
        text_dim=model_config.get('text_dim', 4096),
        clip_dim=model_config.get('clip_dim', 768),
        dropout=model_config.get('dropout', 0.0),
        num_pyramid_stages=model_config.get('num_pyramid_stages', 3),
    )
    return model


def build_vae(vae_config: dict) -> VideoVAE:
    """Build the 3D VAE."""
    vae = VideoVAE(
        in_channels=vae_config.get('in_channels', 3),
        latent_channels=vae_config.get('latent_channels', 16),
        base_channels=vae_config.get('base_channels', 128),
        channel_multipliers=tuple(vae_config.get('channel_multipliers', [1, 2, 4, 8])),
        encoder_num_res_blocks=vae_config.get('encoder_num_res_blocks', 2),
        decoder_num_res_blocks=vae_config.get('decoder_num_res_blocks', 3),
        kl_weight=vae_config.get('kl_weight', 1e-6),
    )
    return vae


def main():
    parser = argparse.ArgumentParser(description='Train Pyramidal Flow Matching')
    parser.add_argument('--config', type=str, required=True, help='Training config YAML')
    parser.add_argument('--model_config', type=str, default='configs/model_config.yaml',
                        help='Model config YAML')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint to resume from')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    args = parser.parse_args()
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    if rank == 0:
        logger.info(f"Starting training with {world_size} processes")
    
    # Load configs
    train_config = load_config(args.config)
    model_config_path = args.model_config
    if os.path.exists(model_config_path):
        model_config = load_config(model_config_path)
    else:
        model_config = {'model': {}, 'vae': {}}
    
    # Build models
    model = build_model(model_config.get('model', {}))
    vae = build_vae(model_config.get('vae', {}))
    
    # Move to device
    model = model.to(device)
    vae = vae.to(device)
    
    # Freeze VAE (it's pre-trained)
    for param in vae.parameters():
        param.requires_grad = False
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {total_params / 1e9:.2f}B")
    
    # Wrap with DDP for distributed training
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Build trainer
    trainer = PyramidFlowTrainer(
        model=model.module if world_size > 1 else model,
        vae=vae,
        config=train_config,
        device=device,
        rank=rank,
        world_size=world_size,
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training loop
    training_steps = train_config.get('training_steps', 50000)
    log_every = train_config.get('log_every', 100)
    save_every = train_config.get('save_every', 5000)
    is_video = train_config.get('training_stage', 'image') != 'image'
    
    logger.info(f"Starting training for {training_steps} steps")
    
    # Note: In a real implementation, you would have a proper DataLoader here
    # This is a placeholder showing the training loop structure
    for step in range(trainer.global_step, training_steps):
        # In practice, get batch from DataLoader
        # batch = next(data_iter)
        
        # Placeholder batch for demonstration
        batch = create_dummy_batch(train_config, device, is_video)
        
        # Training step
        metrics = trainer.training_step(batch, is_video=is_video)
        
        if rank == 0 and step % log_every == 0:
            logger.info(
                f"Step {step}/{training_steps} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"Stage: {metrics['stage']} | "
                f"LR: {metrics['lr']:.2e}"
            )
        
        if rank == 0 and step % save_every == 0 and step > 0:
            trainer.save_checkpoint(args.output_dir, step)
    
    # Save final checkpoint
    if rank == 0:
        trainer.save_checkpoint(args.output_dir, training_steps)
        logger.info("Training complete!")
    
    if world_size > 1:
        dist.destroy_process_group()


def create_dummy_batch(config: dict, device: torch.device, is_video: bool) -> dict:
    """
    Create a dummy batch for testing the training loop.
    In practice, this would be replaced by a real DataLoader.
    """
    B = 2  # Small batch for testing
    
    if is_video:
        # Video batch: (B, C, T, H, W) in latent space
        # With 8x compression: 384p video -> 48x48 latent
        latents = torch.randn(B, 16, 4, 48, 48, device=device)
    else:
        # Image batch: (B, C, H, W) in latent space
        # With 8x compression: 512x512 image -> 64x64 latent
        latents = torch.randn(B, 16, 64, 64, device=device)
    
    # Text embeddings (T5 and CLIP)
    text_embeds_t5 = torch.randn(B, 77, 4096, device=device)
    text_embeds_clip = torch.randn(B, 768, device=device)
    
    batch = {
        'latents': latents,
        'text_embeds_t5': text_embeds_t5,
        'text_embeds_clip': text_embeds_clip,
    }
    
    if is_video:
        # Add history latents for autoregressive training
        history_latents = [
            torch.randn(B, 16, 4, 48, 48, device=device),
            torch.randn(B, 16, 4, 48, 48, device=device),
        ]
        batch['history_latents'] = history_latents
    
    return batch


if __name__ == '__main__':
    main()
