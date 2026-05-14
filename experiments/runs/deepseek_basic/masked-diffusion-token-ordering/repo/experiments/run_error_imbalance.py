"""
Error Imbalance Experiment (Section 3.3)
=========================================
Reproduces the analysis of performance imbalance across masking subproblems.

Two settings:
1. L&O-NAE-SAT: Measures error on latent vs. observation positions
   Error = E_x0 ||log p_θ(x_0|x_0[M]) - log p_data(x_0|x_0[M])||^2
   
2. Text data: Measures π-learner likelihood for different permutations
   Accumulated error across subproblems

Experimental details from Appendix C.2:
- L&O-NAE-SAT: (N,P) = (20,280), pad to 512, 19M MDM, 2K iters (proxy: 50K iters)
- Text: 170M MDM pretrained, measure over 1024 samples
"""

import torch
import numpy as np
import os
import sys
import json
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mdm import MDMTransformer, MDMConfig, MaskedDiffusionModel
from data_generation.lo_distribution import (
    LODistribution, create_lo_nae_sat, make_nae_observation
)
from utils.permutations import (
    random_permutation, identity_permutation,
    interpolated_permutation, sample_permutations_for_interpolation
)
from evaluation.metrics import compute_task_error_imbalance, evaluate_pi_learner_likelihood


def run_lo_nae_sat_error_experiment(
    N: int = 20,
    P: int = 280,
    m: int = 2,
    output_dir: str = 'results/error_imbalance',
    device: str = 'cpu',
):
    """
    Run error imbalance experiment on L&O-NAE-SAT distribution.
    
    Measures error separately for latent and observation positions
    across different mask sizes (Section 3.3, Figure 2 right).
    
    From Appendix C.2.1:
    - For each l in [1, N-1], randomly mask l latent tokens 
      and l*(P/N) observation tokens
    - Measure error for each masked prediction position
    """
    os.makedirs(output_dir, exist_ok=True)
    
    rng = np.random.RandomState(42)
    
    # Create L&O-NAE-SAT distribution
    lo_dist = create_lo_nae_sat(N=N, P=P, m=m, rng=rng)
    L = N + P  # 300
    max_seq_length = 512
    
    # Pad to 512 as in paper
    pad_length = max_seq_length - L
    
    # Create MDM model (19M)
    config = MDMConfig(
        vocab_size=m + 2,  # 0=mask, 1-m=values, m+1=padding
        seq_length=max_seq_length,
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        dropout=0.1,
        max_seq_length=max_seq_length,
        noise_schedule='cosine',
        T=1000,
        mask_token_id=0,
    )
    denoiser = MDMTransformer(config)
    mdm = MaskedDiffusionModel(denoiser, config)
    mdm.denoiser.to(device)
    
    # Train MDM (2K iterations for main model)
    optimizer = torch.optim.AdamW(mdm.denoiser.parameters(), lr=4e-4)
    batch_size = 32
    
    for iteration in range(2000):
        # Sample batch from L&O distribution
        x_batch = []
        for _ in range(batch_size):
            x = lo_dist.sample(rng)
            # Pad
            x_padded = np.zeros(max_seq_length, dtype=int)
            x_padded[:L] = x
            x_padded[L:] = m + 1  # Padding token
            x_batch.append(x_padded)
        
        x_tensor = torch.tensor(np.stack(x_batch), dtype=torch.long, device=device)
        
        optimizer.zero_grad()
        loss = mdm.compute_loss(x_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mdm.denoiser.parameters(), 1.0)
        optimizer.step()
    
    # Create proxy model (train for 50K iterations)
    # For static repo, we simulate this by training more
    proxy_config = MDMConfig(
        vocab_size=m + 2,
        seq_length=max_seq_length,
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        dropout=0.1,
        max_seq_length=max_seq_length,
        noise_schedule='cosine',
        T=1000,
        mask_token_id=0,
    )
    proxy_denoiser = MDMTransformer(proxy_config)
    proxy_mdm = MaskedDiffusionModel(proxy_denoiser, proxy_config)
    proxy_mdm.denoiser.to(device)
    
    proxy_optimizer = torch.optim.AdamW(proxy_denoiser.parameters(), lr=4e-4)
    
    for iteration in range(10000):  # Reduced for demo
        x_batch = []
        for _ in range(batch_size):
            x = lo_dist.sample(rng)
            x_padded = np.zeros(max_seq_length, dtype=int)
            x_padded[:L] = x
            x_padded[L:] = m + 1
            x_batch.append(x_padded)
        
        x_tensor = torch.tensor(np.stack(x_batch), dtype=torch.long, device=device)
        
        proxy_optimizer.zero_grad()
        loss = proxy_mdm.compute_loss(x_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(proxy_denoiser.parameters(), 1.0)
        proxy_optimizer.step()
    
    # Measure errors for different mask sizes
    results = {}
    
    for l in range(1, N):
        # Mask l latent tokens and l*(P/N) observation tokens
        num_latent_mask = l
        num_obs_mask = int(l * (P / N))
        
        latent_errors = []
        obs_errors = []
        
        for _ in range(100):  # 1000 in paper, reduced for demo
            x = lo_dist.sample(rng)
            x_padded = np.zeros(max_seq_length, dtype=int)
            x_padded[:L] = x
            x_padded[L:] = m + 1
            
            # Randomly select latent positions to mask
            latent_positions = list(range(N))
            rng.shuffle(latent_positions)
            mask_latent = set(latent_positions[:num_latent_mask])
            
            # Randomly select observation positions to mask
            obs_positions = list(range(N, L))
            rng.shuffle(obs_positions)
            mask_obs = set(obs_positions[:num_obs_mask])
            
            mask = np.zeros(max_seq_length, dtype=bool)
            for pos in mask_latent:
                mask[pos] = True
            for pos in mask_obs:
                mask[pos] = True
            
            x_masked = x_padded.copy()
            x_masked[mask] = 0  # Mask token
            
            # Get model predictions
            x_t = torch.tensor(x_masked, dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad():
                probs = mdm.denoiser.get_probs(x_t).squeeze(0).cpu().numpy()
                proxy_probs = proxy_mdm.denoiser.get_probs(x_t).squeeze(0).cpu().numpy()
            
            # Compute error for each masked position
            for pos in mask_latent:
                p_theta = probs[pos]
                p_proxy = proxy_probs[pos]
                # MSE between log probabilities
                log_p_theta = np.log(p_theta + 1e-10)
                log_p_proxy = np.log(p_proxy + 1e-10)
                error = np.mean((log_p_theta - log_p_proxy) ** 2)
                latent_errors.append(error)
            
            for pos in mask_obs:
                p_theta = probs[pos]
                p_proxy = proxy_probs[pos]
                log_p_theta = np.log(p_theta + 1e-10)
                log_p_proxy = np.log(p_proxy + 1e-10)
                error = np.mean((log_p_theta - log_p_proxy) ** 2)
                obs_errors.append(error)
        
        results[f'l_{l}'] = {
            'latent_mean_error': float(np.mean(latent_errors)) if latent_errors else 0,
            'latent_std_error': float(np.std(latent_errors)) if latent_errors else 0,
            'observation_mean_error': float(np.mean(obs_errors)) if obs_errors else 0,
            'observation_std_error': float(np.std(obs_errors)) if obs_errors else 0,
        }
    
    with open(os.path.join(output_dir, 'lo_nae_sat_errors.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def run_text_error_imbalance_experiment(
    model: MaskedDiffusionModel,
    val_loader: DataLoader,
    output_dir: str = 'results/error_imbalance',
    device: str = 'cpu',
):
    """
    Run error imbalance experiment on text data (Section 3.3, Figure 2 top right).
    
    Measures accumulated error across subproblems for different permutations:
    E_x0 [Σ_i log p_θ(x_0^{π(i)} | x_0[π{i,...,L-1}])]
    
    Compares:
    - Identity permutation (ARM-like)
    - Closer permutation
    - Uniform random permutation
    """
    os.makedirs(output_dir, exist_ok=True)
    
    L = model.config.seq_length
    
    results = {}
    
    for perm_name, perm in [
        ('identity', identity_permutation(L)),
        ('much_closer', interpolated_permutation(L, int(np.sqrt(L)))),
        ('closer', interpolated_permutation(L, L // 10)),
        ('uniform', random_permutation(L)),
    ]:
        avg_loss = evaluate_pi_learner_likelihood(model, val_loader, perm, device)
        results[perm_name] = float(avg_loss)
    
    with open(os.path.join(output_dir, 'text_error_imbalance.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='results/error_imbalance')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--N', type=int, default=20)
    parser.add_argument('--P', type=int, default=280)
    args = parser.parse_args()
    
    results = run_lo_nae_sat_error_experiment(
        N=args.N, P=args.P,
        output_dir=args.output_dir,
        device=args.device,
    )
    print("Error imbalance experiment complete.")
