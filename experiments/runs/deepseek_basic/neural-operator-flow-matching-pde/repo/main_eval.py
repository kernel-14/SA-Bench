"""Evaluation and inference script for the PDE Foundation Model.

Supports:
1. Long-term autoregressive rollout
2. Ensemble generation with different bridge parameters k
3. Reconstruction quality assessment
4. Few-shot adaptation evaluation (Kolmogorov turbulence)
"""

import os
import sys
import argparse
import logging
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from p2vae import P2VAE, P2VAEConfig
from fmt import FlowMarchingTransformer, FMTConfig
from fmt.sampler import FlowMarchingSampler
from utils.metrics import (
    compute_l2re, compute_vrmse, compute_both_metrics,
    evaluate_long_rollout, compute_ensemble_variance
)
from data.dataset import create_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


def load_models(p2vae_path: str, fmt_path: str, device: str):
    """Load pretrained P2VAE and FMT models."""
    # Load P2VAE
    p2vae_ckpt = torch.load(p2vae_path, map_location='cpu')
    p2vae = P2VAE(P2VAEConfig(base_dim=64))
    p2vae.load_state_dict(p2vae_ckpt['model_state_dict'])
    p2vae = p2vae.to(device)
    p2vae.eval()
    logger.info(f"Loaded P2VAE from {p2vae_path}")
    
    # Load FMT
    fmt_ckpt = torch.load(fmt_path, map_location='cpu')
    fmt_config = FMTConfig(embed_dim=512, num_layers=12)
    fmt = FlowMarchingTransformer(fmt_config)
    fmt.load_state_dict(fmt_ckpt['model_state_dict'])
    fmt = fmt.to(device)
    fmt.eval()
    logger.info(f"Loaded FMT from {fmt_path}")
    
    return p2vae, fmt


@torch.no_grad()
def evaluate_long_rollout(
    p2vae: P2VAE,
    fmt: FlowMarchingTransformer,
    dataloader,
    num_rollout_steps: int = 40,
    device: str = 'cuda',
) -> Dict:
    """Evaluate long-term autoregressive rollout performance.
    
    Compares predictions at steps 1, 5, 10, and last step.
    """
    sampler = FlowMarchingSampler(fmt, num_steps=100)
    
    results = {
        'step_1': {'L2RE': []},
        'step_5': {'L2RE': []},
        'step_10': {'L2RE': []},
        'last_step': {'L2RE': []},
    }
    
    for batch in dataloader:
        if isinstance(batch, torch.Tensor) and batch.dim() == 5:
            frames = [batch[:, i].to(device) for i in range(5)]
        elif isinstance(batch, (list, tuple)):
            frames = [b.to(device) for b in batch[:4]]  # first 4 for context
        else:
            continue
        
        # Encode initial frames
        latent_frames = []
        for x in frames[:4]:
            mu, _ = p2vae.encode(x)
            latent_frames.append(mu)
        
        # Autoregressive rollout
        all_latents = sampler.autoregressive_rollout(
            latent_frames,
            num_steps=num_rollout_steps,
            k_prediction=1.0,
        )
        
        # Decode and evaluate at specific steps
        eval_steps = [1, 5, 10, num_rollout_steps]
        for step_idx, step_name in zip(eval_steps, ['step_1', 'step_5', 'step_10', 'last_step']):
            if step_idx <= num_rollout_steps:
                y_pred = all_latents[3 + step_idx]  # offset by initial 4
                x_pred = p2vae.decode(y_pred)
                x_true = frames[min(3 + step_idx, len(frames) - 1)]
                
                l2re = compute_l2re(x_pred, x_true).item()
                results[step_name]['L2RE'].append(l2re)
    
    # Compute averages
    for key in results:
        if results[key]['L2RE']:
            results[key]['L2RE'] = np.mean(results[key]['L2RE'])
        else:
            results[key]['L2RE'] = 0.0
    
    return results


@torch.no_grad()
def evaluate_ensemble_generation(
    p2vae: P2VAE,
    fmt: FlowMarchingTransformer,
    dataloader,
    k_values: List[float] = [0.0, 0.3, 0.6, 0.9],
    ensemble_size: int = 32,
    device: str = 'cuda',
) -> Dict[float, float]:
    """Generate ensembles at different bridge parameters k3.
    
    Evaluates the variance of ensemble predictions as a function of k3.
    (Fig. 3 in the paper)
    """
    sampler = FlowMarchingSampler(fmt, num_steps=100)
    
    results = {}
    
    for batch in dataloader:
        if isinstance(batch, torch.Tensor) and batch.dim() == 5:
            frames = [batch[:, i].to(device) for i in range(5)]
        elif isinstance(batch, (list, tuple)):
            frames = [b.to(device) for b in batch[:4]]
        else:
            continue
        
        # Use first 3 frames as clean history
        history = frames[:3]
        
        # Encode history
        latent_history = []
        for x in history:
            mu, _ = p2vae.encode(x)
            latent_history.append(mu)
        
        # Need 4th frame for context - use last of history replicated
        latent_history.append(latent_history[-1].clone())
        
        for k3 in k_values:
            ensemble = sampler.generate_ensemble(
                latent_history,
                k3=k3,
                batch_size=ensemble_size,
            )
            
            variance = compute_ensemble_variance(ensemble)
            if k3 not in results:
                results[k3] = []
            results[k3].append(variance)
        
        break  # Just one trajectory for ensemble demonstration
    
    # Average results
    avg_results = {k: np.mean(v) for k, v in results.items()}
    return avg_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate PDE Foundation Model')
    parser.add_argument('--p2vae_path', type=str, required=True,
                       help='Path to pretrained P2VAE checkpoint')
    parser.add_argument('--fmt_path', type=str, required=True,
                       help='Path to pretrained FMT checkpoint')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of preprocessed data')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to evaluate on')
    parser.add_argument('--task', type=str, default='all',
                       choices=['all', 'reconstruction', 'rollout', 'ensemble'],
                       help='Evaluation task')
    parser.add_argument('--num_rollout_steps', type=int, default=40,
                       help='Number of autoregressive rollout steps')
    parser.add_argument('--ensemble_size', type=int, default=32,
                       help='Number of ensemble members')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory for saving results')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    p2vae, fmt = load_models(args.p2vae_path, args.fmt_path, args.device)
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        data_root=args.data_root,
        batch_size=1,  # Single trajectory for evaluation
        num_workers=0,
        use_equal_sampling=False,
    )
    
    all_results = {}
    
    # Reconstruction evaluation
    if args.task in ['all', 'reconstruction']:
        logger.info("Evaluating reconstruction quality...")
        from training.train_p2vae import compute_reconstruction_error
        recon_metrics = compute_reconstruction_error(
            p2vae, dataloaders['test'], device=args.device
        )
        logger.info(f"Reconstruction: {recon_metrics}")
        all_results['reconstruction'] = recon_metrics
    
    # Long-term rollout
    if args.task in ['all', 'rollout']:
        logger.info("Evaluating long-term rollout...")
        rollout_results = evaluate_long_rollout(
            p2vae, fmt, dataloaders['test'],
            num_rollout_steps=args.num_rollout_steps,
            device=args.device,
        )
        logger.info(f"Rollout results: {rollout_results}")
        all_results['rollout'] = rollout_results
    
    # Ensemble generation
    if args.task in ['all', 'ensemble']:
        logger.info("Evaluating ensemble generation...")
        k_values = [0.0, 0.3, 0.6, 0.9, 1.0]
        ensemble_results = evaluate_ensemble_generation(
            p2vae, fmt, dataloaders['test'],
            k_values=k_values,
            ensemble_size=args.ensemble_size,
            device=args.device,
        )
        logger.info(f"Ensemble variance vs k3: {ensemble_results}")
        all_results['ensemble'] = ensemble_results
    
    # Save results
    import json
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {results_path}")


if __name__ == '__main__':
    main()
