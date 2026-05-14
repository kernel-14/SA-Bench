"""
Main training script for consistency models with generator-augmented flows.

Usage:
    python train.py --config configs/cifar10.yaml
    python train.py --config configs/cifar10.yaml --mode GC --mu 0.5
    python train.py --config configs/imagenet32.yaml --mode IC
"""

import os
import math
import argparse
import yaml
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from pathlib import Path

from models import SongUNet, ConsistencyModel
from training import ConsistencyTrainer, NoiseSchedule, TimestepSchedule
from training.schedules import TimestepSampler
from utils.lion_optimizer import Lion


def setup_logging(log_dir):
    """Setup logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def get_dataset(config):
    """Get training dataset based on config."""
    dataset_name = config['dataset']['name']
    data_dir = config['dataset'].get('data_dir', './data')
    img_size = config['dataset']['img_resolution']
    
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),  # Scale to [-1, 1]
    ])
    
    if dataset_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform
        )
    elif dataset_name == 'imagenet':
        dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(data_dir, 'train'), transform=transform
        )
    elif dataset_name == 'celeba':
        dataset = torchvision.datasets.CelebA(
            root=data_dir, split='train', download=True, transform=transform
        )
    elif dataset_name == 'lsun_church':
        dataset = torchvision.datasets.LSUN(
            root=data_dir, classes=['church_outdoor_train'], transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset


def build_model(config, device):
    """Build the consistency model from config."""
    model_config = config['model']
    dataset_config = config['dataset']
    
    img_resolution = dataset_config['img_resolution']
    in_channels = dataset_config.get('in_channels', 3)
    out_channels = in_channels
    label_dim = dataset_config.get('label_dim', 0)
    
    # Build backbone network
    network = SongUNet(
        img_resolution=img_resolution,
        in_channels=in_channels,
        out_channels=out_channels,
        label_dim=label_dim,
        model_channels=model_config.get('model_channels', 128),
        channel_mult=model_config.get('channel_mult', [1, 2, 2, 2]),
        num_blocks=model_config.get('num_blocks', 4),
        attn_resolutions=model_config.get('attn_resolutions', [16]),
        dropout=model_config.get('dropout', 0.0),
        embedding_type=model_config.get('embedding_type', 'positional'),
    )
    
    # Wrap with consistency model parametrization
    sigma_data = model_config.get('sigma_data', 0.5)
    sigma_min = config['training'].get('sigma_min', 0.002)
    sigma_max = config['training'].get('sigma_max', 80.0)
    
    model = ConsistencyModel(
        network=network,
        sigma_data=sigma_data,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )
    
    return model.to(device)


def build_optimizer(model, config):
    """Build optimizer from config."""
    opt_config = config['training']
    lr = opt_config.get('learning_rate', 1e-4)
    optimizer_name = opt_config.get('optimizer', 'lion')
    
    if optimizer_name == 'lion':
        optimizer = Lion(model.parameters(), lr=lr, weight_decay=0.0)
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def train(config, args):
    """Main training function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup logging
    log_dir = config.get('log_dir', './logs')
    logger = setup_logging(log_dir)
    logger.info(f"Training on device: {device}")
    logger.info(f"Config: {config}")
    
    # Build dataset
    dataset = get_dataset(config)
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True,
        drop_last=True,
    )
    
    # Build model
    model = build_model(config, device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Build optimizer
    optimizer = build_optimizer(model, config)
    
    # Build schedules
    train_config = config['training']
    noise_schedule = NoiseSchedule(
        sigma_min=train_config.get('sigma_min', 0.002),
        sigma_max=train_config.get('sigma_max', 80.0),
        rho=train_config.get('rho', 7.0),
    )
    
    total_steps = train_config.get('training_steps', 100000)
    timestep_schedule = TimestepSchedule(
        s0=train_config.get('s0', 10),
        s1=train_config.get('s1', 1280),
        total_steps=total_steps,
    )
    
    timestep_sampler = TimestepSampler(
        P_mean=train_config.get('P_mean', -1.1),
        P_std=train_config.get('P_std', 2.0),
    )
    
    # Build trainer
    mode = args.mode if args.mode else train_config.get('mode', 'GC')
    mu = args.mu if args.mu is not None else train_config.get('mu', 0.5)
    
    trainer = ConsistencyTrainer(
        model=model,
        optimizer=optimizer,
        noise_schedule=noise_schedule,
        timestep_schedule=timestep_schedule,
        timestep_sampler=timestep_sampler,
        device=device,
        mu=mu,
        ema_decay=train_config.get('ema_decay', 0.9999),
        use_ema_for_gc=train_config.get('use_ema_for_gc', True),
        loss_type=train_config.get('loss_type', 'pseudo_huber'),
    )
    
    # Training loop
    logger.info(f"Starting training with mode={mode}, mu={mu}")
    
    data_iter = iter(dataloader)
    save_dir = config.get('save_dir', './checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    
    for step in range(total_steps):
        # Get next batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        if isinstance(batch, (list, tuple)):
            x_star = batch[0].to(device)
        else:
            x_star = batch.to(device)
        
        # Training step
        loss = trainer.train_step(x_star, mode=mode)
        
        # Logging
        if step % 100 == 0:
            logger.info(f"Step {step}/{total_steps}, Loss: {loss:.6f}, N: {trainer.current_N}")
        
        # Save checkpoint
        save_interval = train_config.get('save_interval', 10000)
        if (step + 1) % save_interval == 0 or step == total_steps - 1:
            checkpoint = {
                'step': step,
                'model_state_dict': model.state_dict(),
                'ema_model_state_dict': trainer.ema_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
            }
            checkpoint_path = os.path.join(save_dir, f'checkpoint_{step+1:06d}.pt')
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    logger.info("Training complete!")
    return model, trainer.ema_model


def main():
    parser = argparse.ArgumentParser(description='Train consistency models with GC')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['IC', 'OT', 'GC'], 
                        help='Training mode (overrides config)')
    parser.add_argument('--mu', type=float, help='Joint learning factor (overrides config)')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    train(config, args)


if __name__ == '__main__':
    main()
