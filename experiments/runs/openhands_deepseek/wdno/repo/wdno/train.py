import os
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import time

from .model import WDNO1D, WDNO2D, SuperResolutionModel
from data.dataset import (
    get_dataloader, get_multiresolution_dataloader,
    BurgersDataset, NavierStokes1DDataset, Fluid2DDataset, ERA5Dataset
)


def train_brm(config, dataset_name='burgers', task='simulation', device='cuda'):
    """Train the Base-Resolution Model (BRM).
    
    Following Algorithm 1 in the paper.
    """
    # Setup
    learning_rate = config['training']['learning_rate']
    training_steps = config['training']['training_steps']
    batch_size = config['training']['batch_size']
    
    # Create dataloader
    dataloader = get_dataloader(
        dataset_name, batch_size=batch_size, split='train', task=task,
        data_path=config.get('data', {}).get(f'{dataset_name}_dir', None)
    )
    
    # Create model
    if dataset_name in ['burgers', 'advection', 'navier_stokes']:
        model = WDNO1D(config, task=task)
    elif dataset_name in ['fluid_2d', 'era5']:
        model = WDNO2D(config, task=task)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    model = model.to(device)
    model.diffusion.to(device)
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=training_steps)
    
    # Training loop
    model.train()
    step = 0
    dataloader_iter = iter(dataloader)
    
    pbar = tqdm(total=training_steps, desc=f"Training BRM ({dataset_name}, {task})")
    
    while step < training_steps:
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
        
        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else 
                 {k2: v2.to(device) if isinstance(v2, torch.Tensor) else v2 
                  for k2, v2 in v.items()} if isinstance(v, dict) else v
                for k, v in batch.items()}
        
        optimizer.zero_grad()
        loss = model.training_step(batch)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        step += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        pbar.update(1)
        
        if step % 10000 == 0:
            # Save checkpoint
            ckpt_path = f'checkpoints/brm_{dataset_name}_{task}_step{step}.pt'
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, ckpt_path)
    
    pbar.close()
    return model


def train_srm(config, dataset_name='burgers', device='cuda'):
    """Train the Super-Resolution Model (SRM).
    
    Trains p(W_h | W_l, W_{a_h}) using multi-resolution data.
    """
    learning_rate = config['training']['learning_rate']
    training_steps = config['training']['training_steps']
    batch_size = config['training']['batch_size']
    num_levels = config['multires']['num_levels']
    
    # Create multi-resolution dataloader
    dataloader = get_multiresolution_dataloader(
        dataset_name, batch_size=batch_size, split='train',
        num_levels=num_levels,
        data_path=config.get('data', {}).get(f'{dataset_name}_dir', None)
    )
    
    # Create SRM
    if dataset_name in ['burgers', 'advection', 'navier_stokes']:
        srm = SuperResolutionModel(config, experiment_type='1d')
    elif dataset_name in ['fluid_2d']:
        srm = SuperResolutionModel(config, experiment_type='2d')
    else:
        srm = SuperResolutionModel(config, experiment_type='1d')
    
    srm = srm.to(device)
    srm.diffusion.to(device)
    
    optimizer = Adam(srm.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=training_steps)
    
    # Training loop
    srm.train()
    step = 0
    
    pbar = tqdm(total=training_steps, desc=f"Training SRM ({dataset_name})")
    
    while step < training_steps:
        batch = next(iter(dataloader))  # Simplified for demonstration
        
        if isinstance(batch, dict) and 'high' not in batch:
            # Standard batch from dataset - need to create multi-res pair
            high = batch['data']
            low = torch.nn.functional.interpolate(
                high, scale_factor=0.5, mode='bilinear' if dataset_name in ['burgers', 'advection', 'navier_stokes'] else 'trilinear',
                align_corners=False
            ) if high.dim() >= 4 else high
            continue
        
        if 'high' not in batch:
            continue
        
        high = batch['high'].to(device)
        low = batch['low'].to(device)
        
        # Encode to wavelet domain
        w_high = srm.wavelet_2d.decompose(high) if dataset_name in ['burgers', 'advection', 'navier_stokes'] else srm.wavelet_3d.decompose(high)
        w_low = srm.wavelet_2d.decompose(low) if dataset_name in ['burgers', 'advection', 'navier_stokes'] else srm.wavelet_3d.decompose(low)
        
        w_cond_high = w_high  # Same as high-res target for conditioning
        
        # Duplicate low-res to match high-res
        if w_low.shape[2:] != w_high.shape[2:]:
            from .wavelet_utils import duplicate_low_res_to_high_res
            w_low = duplicate_low_res_to_high_res(w_low, w_high.shape[2:])
        
        optimizer.zero_grad()
        loss = srm.forward_diffusion_loss(w_high, w_low, w_cond_high)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(srm.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1
        
        if step % 10000 == 0:
            ckpt_path = f'checkpoints/srm_{dataset_name}_step{step}.pt'
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'step': step,
                'model_state_dict': srm.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, ckpt_path)
    
    pbar.close()
    return srm


def train_control(config, dataset_name='burgers', device='cuda'):
    """Train WDNO for control tasks.
    
    Uses classifier-free guidance and energy-based optimization.
    Following equations (4) and (5) from the paper.
    """
    learning_rate = config['training']['learning_rate']
    training_steps = config['training']['training_steps']
    batch_size = config['training']['batch_size']
    
    dataloader = get_dataloader(
        dataset_name, batch_size=batch_size, split='train', task='control',
        data_path=config.get('data', {}).get(f'{dataset_name}_dir', None)
    )
    
    model = WDNO1D(config, task='control')
    model = model.to(device)
    model.diffusion.to(device)
    
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=training_steps)
    
    model.train()
    step = 0
    
    pbar = tqdm(total=training_steps, desc=f"Training Control ({dataset_name})")
    
    while step < training_steps:
        try:
            batch = next(iter(dataloader))
        except StopIteration:
            break
        
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}
        
        # For control, we learn p(f | u0, uT)
        # Reformat conditioning
        cond = {
            'u0': batch['cond']['u0'],
            'uT': batch['cond'].get('uT', batch['cond']['u0'].clone()),
            'target_shape': (batch['data'].shape[2] - 1, batch['data'].shape[3]),
        }
        
        optimizer.zero_grad()
        # Use f (control) as target data
        x_start = model.encode(batch['cond']['f'])
        w_cond = model.encode_cond(cond)
        loss = model.forward_diffusion_loss(x_start, w_cond)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1
    
    pbar.close()
    return model


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Return default config
        return {
            'wavelet': {'type_1d': 'bior2.4', 'type_2d': 'bior1.3',
                       'mode_1d': 'periodization', 'mode_2d': 'zero'},
            'diffusion': {'num_timesteps': 1000, 'beta_start': 1e-4,
                         'beta_end': 0.02, 'schedule': 'linear'},
            'ddim': {'sampling_steps': 50, 'eta': 1.0},
            'unet_1d': {'init_dim': 128, 'down_up_layers': 4, 'kernel_size': 3,
                       'dim_mult_phi': [1, 2, 4, 8], 'dim_mult_theta': [1, 2, 4, 8],
                       'resnet_groups': 8, 'attn_hidden_dim': 32, 'attn_heads': 4},
            'unet_3d': {'init_dim': 100, 'kernel_size': [3, 3, 3],
                       'kernel_padding': [1, 1, 1], 'attn_heads': 4},
            'multires': {'num_levels': 3, 'sr_steps': [1, 2, 3]},
            'training': {'batch_size': 16, 'optimizer': 'adam',
                        'learning_rate': 1e-4, 'training_steps': 190000,
                        'lr_scheduler': 'cosine'},
            'control': {'guidance_weight': 120000, 'guidance_scheduler': 'cosine'},
        }
