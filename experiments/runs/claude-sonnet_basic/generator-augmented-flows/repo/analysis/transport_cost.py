"""
Analysis of transport costs for different couplings.

This script reproduces Figure 3 from the paper, comparing:
- c(0): transport cost for IC (E[||x_star - z||^2])
- c_OT(0): transport cost for batch-OT
- c(t): transport cost for GC (E[||f(x_t, sigma_t) - z||^2])

The transport cost measures how well the data-noise coupling aligns
the predicted data point with the noise vector.
"""

import os
import sys
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_ic_transport_cost(dataset, num_samples=10000):
    """
    Compute IC transport cost: c(0) = E[||x_star - z||^2]
    
    For IC, x_star and z are independent, so:
    c(0) = E[||x_star||^2] + E[||z||^2] = Var(x_star) + d
    """
    dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
    batch = next(iter(dataloader))
    if isinstance(batch, (list, tuple)):
        x_star = batch[0]
    else:
        x_star = batch
    x_star = x_star[:num_samples]
    
    z = torch.randn_like(x_star)
    
    # IC transport cost
    cost = ((x_star - z) ** 2).sum(dim=[1, 2, 3]).mean().item()
    return cost


def compute_ot_transport_cost(dataset, num_samples=1000):
    """
    Compute batch-OT transport cost using Hungarian matching.
    """
    from scipy.optimize import linear_sum_assignment
    
    dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
    batch = next(iter(dataloader))
    if isinstance(batch, (list, tuple)):
        x_star = batch[0]
    else:
        x_star = batch
    x_star = x_star[:num_samples]
    
    z = torch.randn_like(x_star)
    
    # OT matching
    x_flat = x_star.reshape(num_samples, -1).float()
    z_flat = z.reshape(num_samples, -1).float()
    cost_matrix = torch.cdist(x_flat, z_flat, p=2).numpy()
    
    _, col_ind = linear_sum_assignment(cost_matrix)
    z_ot = z[col_ind]
    
    # OT transport cost
    cost = ((x_star - z_ot) ** 2).sum(dim=[1, 2, 3]).mean().item()
    return cost


def compute_gc_transport_cost(dataset, consistency_model, sigmas, device, num_samples=1000):
    """
    Compute GC transport cost: c(t) = E[||f(x_t, sigma_t) - z||^2]
    
    This measures how well the predicted endpoint aligns with the noise.
    """
    dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
    batch = next(iter(dataloader))
    if isinstance(batch, (list, tuple)):
        x_star = batch[0].to(device)
    else:
        x_star = batch.to(device)
    x_star = x_star[:num_samples]
    
    costs = []
    with torch.no_grad():
        for sigma in sigmas:
            sigma_t = torch.tensor(sigma, dtype=x_star.dtype, device=device)
            z = torch.randn_like(x_star)
            x_t = x_star + sigma_t * z
            
            # Predict endpoint
            x_hat_t = consistency_model(x_t, sigma_t.expand(x_star.shape[0]))
            
            # GC transport cost
            cost = ((x_hat_t - z) ** 2).sum(dim=[1, 2, 3]).mean().item()
            costs.append(cost)
    
    return np.array(costs)


def plot_transport_costs(sigmas, cost_ic, cost_ot, costs_gc, save_path=None):
    """Plot comparison of transport costs."""
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(8, 5))
        plt.axhline(y=cost_ic, color='blue', linestyle='--', label='IC')
        plt.axhline(y=cost_ot, color='orange', linestyle='--', label='batch-OT')
        plt.plot(sigmas, costs_gc, color='green', label='GC')
        plt.xlabel('sigma_t')
        plt.ylabel('Transport Cost')
        plt.title('Transport Cost Comparison')
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
    except ImportError:
        print("matplotlib not available, skipping plot")
        print(f"IC cost: {cost_ic}")
        print(f"OT cost: {cost_ot}")
        print(f"GC costs: {costs_gc}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gc_checkpoint', type=str, help='Path to GC model checkpoint')
    parser.add_argument('--gc_config', type=str, help='Path to GC model config')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--save_path', type=str, default='transport_cost.png')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=True, transform=transform
    )
    
    from training.schedules import NoiseSchedule
    noise_schedule = NoiseSchedule(sigma_min=0.002, sigma_max=80.0, rho=7.0)
    sigmas = noise_schedule.get_sigmas(50)[1:]
    
    print("Computing IC transport cost...")
    cost_ic = compute_ic_transport_cost(dataset)
    print(f"IC transport cost: {cost_ic:.4f}")
    
    print("Computing OT transport cost...")
    cost_ot = compute_ot_transport_cost(dataset)
    print(f"OT transport cost: {cost_ot:.4f}")
    
    if args.gc_checkpoint and args.gc_config:
        from models import SongUNet, ConsistencyModel
        import yaml
        
        with open(args.gc_config, 'r') as f:
            config = yaml.safe_load(f)
        
        model_config = config['model']
        dataset_config = config['dataset']
        
        network = SongUNet(
            img_resolution=dataset_config['img_resolution'],
            in_channels=dataset_config.get('in_channels', 3),
            out_channels=dataset_config.get('in_channels', 3),
            model_channels=model_config.get('model_channels', 128),
            channel_mult=model_config.get('channel_mult', [1, 2, 2]),
            num_blocks=model_config.get('num_blocks', 3),
            attn_resolutions=model_config.get('attn_resolutions', []),
            dropout=0.0,
        )
        
        model = ConsistencyModel(
            network=network,
            sigma_data=model_config.get('sigma_data', 0.5),
            sigma_min=config['training'].get('sigma_min', 0.002),
            sigma_max=config['training'].get('sigma_max', 80.0),
        )
        
        checkpoint = torch.load(args.gc_checkpoint, map_location=device)
        if 'ema_model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['ema_model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        model = model.to(device)
        model.eval()
        
        print("Computing GC transport costs...")
        costs_gc = compute_gc_transport_cost(dataset, model, sigmas, device)
    else:
        print("No GC checkpoint provided, using placeholder values")
        costs_gc = np.ones(len(sigmas)) * cost_ic * 0.5
    
    plot_transport_costs(sigmas, cost_ic, cost_ot, costs_gc, save_path=args.save_path)
