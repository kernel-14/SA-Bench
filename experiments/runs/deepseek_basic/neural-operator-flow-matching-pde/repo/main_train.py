"""Main training script for the PDE Foundation Model.

Two-stage training:
1. Train P2VAE on individual frames
2. Freeze P2VAE, train FMT on latent trajectories

Usage:
    # Stage 1: Train P2VAE
    python main_train.py --stage 1 --config configs/p2vae_16m.yaml
    
    # Stage 2: Train FMT
    python main_train.py --stage 2 --config configs/fmt_base.yaml
"""

import os
import sys
import argparse
import logging
import yaml
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from p2vae import P2VAE, P2VAEConfig
from fmt import FlowMarchingTransformer, FMTConfig
from training.train_p2vae import train_p2vae, compute_reconstruction_error
from training.train_fmt import train_fmt
from data.dataset import create_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_p2vae_from_config(config: dict) -> P2VAE:
    """Create P2VAE model from configuration dict."""
    model_cfg = config['model']
    p2vae_config = P2VAEConfig(
        in_channels=model_cfg['in_channels'],
        out_channels=model_cfg['out_channels'],
        spatial_size=model_cfg['spatial_size'],
        latent_channels=model_cfg['latent_channels'],
        latent_size=model_cfg['latent_size'],
        base_dim=model_cfg['base_dim'],
        ch_mult=tuple(model_cfg.get('ch_mult', (1, 2, 4, 4))),
        num_res_blocks=model_cfg.get('num_res_blocks', 2),
        z_channels=model_cfg.get('z_channels', 16),
        kl_weight=model_cfg.get('kl_weight', 1e-3),
        use_fp16=model_cfg.get('use_fp16', True),
    )
    return P2VAE(p2vae_config)


def create_fmt_from_config(config: dict) -> FlowMarchingTransformer:
    """Create FMT model from configuration dict."""
    model_cfg = config['model']
    fmt_config = FMTConfig(
        embed_dim=model_cfg['embed_dim'],
        num_heads=model_cfg['num_heads'],
        head_dim=model_cfg['head_dim'],
        num_layers=model_cfg['num_layers'],
        latent_channels=model_cfg['latent_channels'],
        latent_size=model_cfg['latent_size'],
        num_diffusion_steps=model_cfg.get('num_diffusion_steps', 100),
        dt=model_cfg.get('dt', 0.01),
        temporal_pyramid_factors=tuple(
            model_cfg.get('temporal_pyramid_factors', (8, 4, 2, 1))
        ),
        gru_hidden_dim=model_cfg.get('gru_hidden_dim'),
        use_adaln=model_cfg.get('use_adaln', True),
        use_fp16=model_cfg.get('use_fp16', True),
        dropout=model_cfg.get('dropout', 0.0),
    )
    return FlowMarchingTransformer(fmt_config)


def stage1_train_p2vae(config: dict, args):
    """Stage 1: Train P2VAE."""
    logger.info("=" * 60)
    logger.info(f"Stage 1: Training {config['model']['name']}")
    logger.info("=" * 60)
    
    # Create model
    model = create_p2vae_from_config(config)
    
    # Create dataloaders
    data_cfg = config['data']
    dataloaders = create_dataloaders(
        data_root=data_cfg['data_root'],
        batch_size=config['training']['batch_size'],
        num_workers=data_cfg.get('num_workers', 4),
        use_equal_sampling=data_cfg.get('use_equal_sampling', True),
    )
    
    # Train
    train_cfg = config['training']
    checkpoint_dir = args.checkpoint_dir or f"checkpoints/{config['model']['name']}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    trained_model = train_p2vae(
        model=model,
        train_dataloader=dataloaders['train'],
        val_dataloader=dataloaders.get('val'),
        total_steps=train_cfg['total_steps'],
        batch_size=train_cfg['batch_size'],
        base_lr=train_cfg['base_lr'],
        weight_decay=train_cfg['weight_decay'],
        beta1=train_cfg.get('beta1', 0.9),
        beta2=train_cfg.get('beta2', 0.995),
        warmup_ratio=train_cfg.get('warmup_ratio', 0.1),
        device=args.device,
        use_fp16=train_cfg.get('use_fp16', True),
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        checkpoint_dir=checkpoint_dir,
    )
    
    # Compute reconstruction error on test set
    if 'test' in dataloaders:
        metrics = compute_reconstruction_error(
            trained_model,
            dataloaders['test'],
            device=args.device,
        )
        logger.info(f"Test reconstruction metrics: {metrics}")
    
    # Save final model
    final_path = os.path.join(checkpoint_dir, 'p2vae_final.pt')
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'config': config,
    }, final_path)
    logger.info(f"Final model saved to {final_path}")


def stage2_train_fmt(config: dict, args):
    """Stage 2: Train FMT with frozen P2VAE."""
    logger.info("=" * 60)
    logger.info(f"Stage 2: Training {config['model']['name']}")
    logger.info("=" * 60)
    
    # Load pretrained P2VAE
    p2vae_path = config.get('pretrained_p2vae', args.p2vae_checkpoint)
    if not p2vae_path or not os.path.exists(p2vae_path):
        raise ValueError(f"P2VAE checkpoint not found: {p2vae_path}")
    
    p2vae_checkpoint = torch.load(p2vae_path, map_location='cpu')
    p2vae_config_dict = p2vae_checkpoint.get('config', {})
    
    # Create P2VAE
    if p2vae_config_dict:
        p2vae = create_p2vae_from_config(p2vae_config_dict)
    else:
        # Default to 16M config
        p2vae = P2VAE(P2VAEConfig(base_dim=64))
    
    p2vae.load_state_dict(p2vae_checkpoint['model_state_dict'])
    logger.info(f"Loaded pretrained P2VAE from {p2vae_path}")
    
    # Create FMT
    model = create_fmt_from_config(config)
    
    # Create dataloaders
    data_cfg = config['data']
    dataloaders = create_dataloaders(
        data_root=data_cfg['data_root'],
        batch_size=config['training']['batch_size'],
        num_workers=data_cfg.get('num_workers', 4),
        use_equal_sampling=data_cfg.get('use_equal_sampling', True),
    )
    
    # Train
    train_cfg = config['training']
    checkpoint_dir = args.checkpoint_dir or f"checkpoints/{config['model']['name']}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    trained_model = train_fmt(
        model=model,
        p2vae=p2vae,
        train_dataloader=dataloaders['train'],
        val_dataloader=dataloaders.get('val'),
        total_steps=train_cfg['total_steps'],
        batch_size=train_cfg['batch_size'],
        base_lr=train_cfg['base_lr'],
        weight_decay=train_cfg['weight_decay'],
        beta1=train_cfg.get('beta1', 0.9),
        beta2=train_cfg.get('beta2', 0.95),
        warmup_ratio=train_cfg.get('warmup_ratio', 0.1),
        device=args.device,
        use_fp16=train_cfg.get('use_fp16', True),
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        checkpoint_dir=checkpoint_dir,
    )
    
    # Save final model
    final_path = os.path.join(checkpoint_dir, 'fmt_final.pt')
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'config': config,
    }, final_path)
    logger.info(f"Final model saved to {final_path}")


def stage3_finetune(config: dict, args):
    """Stage 3: Few-shot finetuning on downstream task (Kolmogorov)."""
    from training.train_fmt import train_finetune_kolmogorov
    
    logger.info("=" * 60)
    logger.info("Stage 3: Few-shot finetuning on Kolmogorov turbulence")
    logger.info("=" * 60)
    
    # Load pretrained models
    raise NotImplementedError("Full finetuning pipeline - see training/train_fmt.py")


def main():
    parser = argparse.ArgumentParser(description='Train PDE Foundation Model')
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2, 3],
                       help='Training stage (1=P2VAE, 2=FMT, 3=Finetune)')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML config file')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to train on')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                       help='Directory for saving checkpoints')
    parser.add_argument('--p2vae_checkpoint', type=str, default=None,
                       help='Path to pretrained P2VAE checkpoint (for stage 2)')
    parser.add_argument('--log_interval', type=int, default=100,
                       help='Steps between logging')
    parser.add_argument('--save_interval', type=int, default=5000,
                       help='Steps between checkpoint saves')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    if args.stage == 1:
        stage1_train_p2vae(config, args)
    elif args.stage == 2:
        stage2_train_fmt(config, args)
    elif args.stage == 3:
        stage3_finetune(config, args)


if __name__ == '__main__':
    main()
