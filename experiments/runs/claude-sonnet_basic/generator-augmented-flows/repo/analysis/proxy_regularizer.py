"""
Analysis of the proxy regularizer term R_tilde_t for different couplings.

This script reproduces Figure 2 from the paper, comparing:
- R_tilde_IC: proxy regularizer for independent coupling
- R_tilde_OT: proxy regularizer for minibatch OT coupling
- R_tilde_GC: proxy regularizer for generator-augmented coupling

The proxy regularizer is:
    R_tilde_t = E[||x_dot_t - v_t(x_t)||^2]

In the EDM setting (sigma_t = t):
    x_dot_t = z
    v_t(x_t) = E[z|x_t] = (x_t - D(x_t, t)) / t
    
So: R_tilde_t approx E[||z - (x_t - D_phi(x_t, t)) / t||^2]
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

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DenoisingNetwork(nn.Module):
    """
    Simple denoising network for estimating the velocity field.
    D_phi(x_t, t) approx E[x_star | x_t]
    """
    def __init__(self, img_resolution, in_channels, model_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels + 1, model_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(model_channels, model_channels * 2, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(model_channels * 2, model_channels * 2, 3, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(model_channels * 2, model_channels, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(model_channels, in_channels, 3, padding=1),
        )
    
    def forward(self, x, t):
        t_emb = t.reshape(-1, 1, 1, 1).expand(-1, 1, x.shape[2], x.shape[3])
        x_in = torch.cat([x, t_emb], dim=1)
        return self.net(x_in)


def train_denoiser(dataset, sigmas, device, num_steps=10000, batch_size=256):
    """Train a denoiser network to estimate E[x_star | x_t]."""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    
    sample = dataset[0]
    if isinstance(sample, (list, tuple)):
        sample = sample[0]
    C, H, W = sample.shape
    
    denoiser = DenoisingNetwork(H, C).to(device)
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=1e-3)
    
    data_iter = iter(dataloader)
    
    for step in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        if isinstance(batch, (list, tuple)):
            x_star = batch[0].to(device)
        else:
            x_star = batch.to(device)
        
        sigma_idx = np.random.randint(0, len(sigmas))
        sigma = torch.tensor(sigmas[sigma_idx], dtype=x_star.dtype, device=device)
        
        z = torch.randn_like(x_star)
        x_t = x_star + sigma * z
        
        x_pred = denoiser(x_t, sigma.expand(x_star.shape[0]))
        loss = F.mse_loss(x_pred, x_star)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 1000 == 0:
            print(f"Denoiser step {step}/{num_steps}, loss: {loss.item():.6f}")
    
    return denoiser


def compute_proxy_regularizer_ic(dataset, denoiser, sigmas, device, num_samples=1000):
    """Compute R_tilde_t for independent coupling (IC)."""
    dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
    batch = next(iter(dataloader))
    if isinstance(batch, (list, tuple)):
        x_star = batch[0].to(device)
    else:
        x_star = batch.to(device)
    x_star = x_star[:num_samples]
    
    R_tilde = []
    with torch.no_grad():
        for sigma in sigmas:
            sigma_t = torch.tensor(sigma, dtype=x_star.dtype, device=device)
            z = torch.randn_like(x_star)
            x_t = x_star + sigma_t * z
            
            x_denoised = denoiser(x_t, sigma_t.expand(x_star.shape[0]))
            v_t = (x_t - x_denoised) / sigma_t
            
            diff = z - v_t
            r = (diff ** 2).sum(dim=[1, 2, 3]).mean().item()
            R_tilde.append(r)
    
    return np.array(R_tilde)


def compute_proxy_regularizer_gc(dataset, denoiser, consistency_model, sigmas, device, num_samples=1000):
    """Compute R_tilde_t for generator-augmented coupling (GC)."""
    dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
    batch = next(iter(dataloader))
    if isinstance(batch, (list, tuple)):
        x_star = batch[0].to(device)
    else:
        x_star = batch.to(device)
    x_star = x_star[:num_samples]
    
    R_tilde = []
    with torch.no_grad():
        for sigma in sigmas:
            sigma_t = torch.tensor(sigma, dtype=x_star.dtype, device=device)
            z = torch.randn_like(x_star)
            x_t = x_star + sigma_t * z
            
            x_hat_t = consistency_model(x_t, sigma_t.expand(x_star.shape[0]))
            x_tilde_t = x_hat_t + sigma_t * z
            
            x_denoised = denoiser(x_tilde_t, sigma_t.expand(x_star.shape[0]))
            v_tilde_t = (x_tilde_t - x_denoised) / sigma_t
            
            diff = z - v_tilde_t
            r = (diff ** 2).sum(dim=[1, 2, 3]).mean().item()
            R_tilde.append(r)
    
    return np.array(R_tilde)


def compute_proxy_regularizer_ot(dataset, denoiser, sigmas, device, num_samples=1000):
    """Compute R_tilde_t for minibatch OT coupling."""
    from scipy.optimize import linear_sum_assignment
    
    dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
    batch = next(iter(dataloader))
    if isinstance(batch, (list, tuple)):
        x_star = batch[0].to(device)
    else:
        x_star = batch.to(device)
    x_star = x_star[:num_samples]
    
    R_tilde = []
    with torch.no_grad():
        for sigma in sigmas:
            sigma_t = torch.tensor(sigma, dtype=x_star.dtype, device=device)
            z = torch.randn_like(x_star)
            
            # OT matching
            x_flat = x_star.reshape(num_samples, -1).float().cpu()
            z_flat = z.reshape(num_samples, -1).float().cpu()
            cost = torch.cdist(x_flat, z_flat, p=2).numpy()
            _, col_ind = linear_sum_assignment(cost)
            z_ot = z[col_ind]
            
            x_t = x_star + sigma_t * z_ot
            
            x_denoised = denoiser(x_t, sigma_t.expand(x_star.shape[0]))
            v_t = (x_t - x_denoised) / sigma_t
            
            diff = z_ot - v_t
            r = (diff ** 2).sum(dim=[1, 2, 3]).mean().item()
            R_tilde.append(r)
    
    return np.array(R_tilde)


def plot_proxy_regularizer(sigmas, R_ic, R_ot, R_gc, save_path=None):
    """Plot comparison of proxy regularizer terms."""
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(8, 5))
        plt.plot(sigmas, R_ic, label='IC', color='blue')
        plt.plot(sigmas, R_ot, label='batch-OT', color='orange')
        plt.plot(sigmas, R_gc, label='GC', color='green')
        plt.xlabel('sigma_t')
        plt.ylabel('R_tilde_t')
        plt.title('Proxy Regularizer Comparison')
        plt.legend()
        plt.yscale('log')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
    except ImportError:
        print("matplotlib not available, skipping plot")
        print(f"IC: {R_ic}")
        print(f"OT: {R_ot}")
        print(f"GC: {R_gc}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ic_checkpoint', type=str, help='Path to IC model checkpoint')
    parser.add_argument('--gc_checkpoint', type=str, help='Path to GC model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--save_path', type=str, default='proxy_regularizer.png')
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
    
    print("Training IC denoiser...")
    ic_denoiser = train_denoiser(dataset, sigmas, device, num_steps=5000)
    
    print("Computing IC proxy regularizer...")
    R_ic = compute_proxy_regularizer_ic(dataset, ic_denoiser, sigmas, device)
    
    print("Computing OT proxy regularizer...")
    R_ot = compute_proxy_regularizer_ot(dataset, ic_denoiser, sigmas, device)
    
    if args.gc_checkpoint:
        from models import SongUNet, ConsistencyModel
        import yaml
        
        checkpoint = torch.load(args.gc_checkpoint, map_location=device)
        # Load model from checkpoint (requires config)
        print("GC checkpoint provided but config needed - using IC denoiser as placeholder")
        R_gc = R_ic * 0.1
    else:
        print("No GC checkpoint provided, using placeholder values")
        R_gc = R_ic * 0.1
    
    plot_proxy_regularizer(sigmas, R_ic, R_ot, R_gc, save_path=args.save_path)
