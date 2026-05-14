"""
Error Imbalance Analysis
=========================
Analyzes the imbalance in MDM performance across different masking subproblems.

This reproduces the experiments in Section 3.3 and Figure 2 (right) of the paper.

For L&O-NAE-SAT:
- Measures error for latent positions vs. observation positions
- Shows that MDM performs well on observation positions but struggles with latents

For text data:
- Measures pi-learner likelihood for different permutations
- Shows that performance degrades as permutation deviates from identity
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lo_nae_sat import LONAESATDistribution, LONAESATDataset
from mdm_model import MDMTransformer
from pi_learner import generate_permutation, permute_sequence


@torch.no_grad()
def measure_error_by_position(model: torch.nn.Module,
                                dist: LONAESATDistribution,
                                n_samples: int = 1000,
                                ell: int = 11,
                                device: torch.device = None,
                                seed: int = 42) -> Dict[str, np.ndarray]:
    """
    Measure MDM prediction error for each position type (latent vs. observation).
    
    For each sample:
    1. Randomly mask ell latent tokens and ell * (P/N) observation tokens
    2. Measure prediction error at each masked position
    
    Args:
        model: trained MDM model
        dist: L&O-NAE-SAT distribution
        n_samples: number of samples to evaluate
        ell: number of latent tokens to mask
        device: computation device
        seed: random seed
    
    Returns:
        dict with 'latent_errors' and 'obs_errors' arrays
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    rng = np.random.RandomState(seed)
    
    # Generate samples
    samples = dist.sample(n_samples, rng)
    
    latent_errors = []
    obs_errors = []
    
    batch_size = 64
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = samples[start:end]
        B = end - start
        
        # Create masked versions
        x_masked = batch.copy()
        latent_mask = np.zeros((B, dist.N), dtype=bool)
        obs_mask = np.zeros((B, dist.P), dtype=bool)
        
        for i in range(B):
            # Mask ell latent tokens
            lat_idx = rng.choice(dist.N, size=ell, replace=False)
            latent_mask[i, lat_idx] = True
            x_masked[i, lat_idx] = 0  # mask token
            
            # Mask ell * (P/N) observation tokens
            n_obs_mask = int(ell * dist.P / dist.N)
            obs_idx = rng.choice(dist.P, size=n_obs_mask, replace=False)
            obs_mask[i, obs_idx] = True
            x_masked[i, dist.N + obs_idx] = 0  # mask token
        
        # Get model predictions
        x_tensor = torch.tensor(x_masked, dtype=torch.long, device=device)
        logits = model(x_tensor)  # (B, L, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Compute errors for latent positions
        for i in range(B):
            lat_masked_idx = np.where(latent_mask[i])[0]
            for j in lat_masked_idx:
                true_token = batch[i, j]
                pred_log_prob = log_probs[i, j, true_token].item()
                latent_errors.append(-pred_log_prob)  # negative log-likelihood
            
            # Compute errors for observation positions
            obs_masked_idx = np.where(obs_mask[i])[0]
            for j in obs_masked_idx:
                true_token = batch[i, dist.N + j]
                pred_log_prob = log_probs[i, dist.N + j, true_token].item()
                obs_errors.append(-pred_log_prob)
    
    return {
        'latent_errors': np.array(latent_errors),
        'obs_errors': np.array(obs_errors)
    }


@torch.no_grad()
def measure_pi_learner_performance(model: torch.nn.Module,
                                    data: torch.Tensor,
                                    permutations: Dict[str, np.ndarray],
                                    device: torch.device = None) -> Dict[str, float]:
    """
    Measure pi-learner performance for different permutations.
    
    Computes the average log-likelihood under each pi-learner.
    
    Args:
        model: trained causal transformer
        data: text sequences (n_samples, L)
        permutations: dict mapping name -> permutation array
        device: computation device
    
    Returns:
        dict mapping permutation name -> average log-likelihood
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    results = {}
    
    for name, perm in permutations.items():
        total_ll = 0.0
        n_samples = 0
        
        batch_size = 32
        for start in range(0, len(data), batch_size):
            end = min(start + batch_size, len(data))
            x = data[start:end].to(device)
            
            # Permute sequence
            perm_tensor = torch.tensor(perm, dtype=torch.long, device=device)
            x_perm = x[:, perm_tensor]
            
            # Compute log-likelihood
            logits = model(x_perm[:, :-1])
            log_probs = F.log_softmax(logits, dim=-1)
            
            targets = x_perm[:, 1:]
            token_ll = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
            seq_ll = token_ll.sum(dim=-1)
            
            total_ll += seq_ll.sum().item()
            n_samples += end - start
        
        results[name] = total_ll / n_samples
    
    return results


def plot_error_imbalance(latent_errors: np.ndarray, obs_errors: np.ndarray,
                          save_path: str = None):
    """
    Plot the error imbalance between latent and observation positions.
    
    Reproduces Figure 2 (right, bottom) from the paper.
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Plot histograms
        ax.hist(latent_errors, bins=50, alpha=0.7, label='Latent positions', 
                color='darkblue', density=True)
        ax.hist(obs_errors, bins=50, alpha=0.7, label='Observation positions',
                color='lightblue', density=True)
        
        ax.set_xlabel('Negative log-likelihood (error)')
        ax.set_ylabel('Density')
        ax.set_title('Error Imbalance: Latent vs. Observation Positions\n(L&O-NAE-SAT)')
        ax.legend()
        
        # Add statistics
        ax.axvline(latent_errors.mean(), color='darkblue', linestyle='--', alpha=0.8,
                   label=f'Latent mean: {latent_errors.mean():.2f}')
        ax.axvline(obs_errors.mean(), color='lightblue', linestyle='--', alpha=0.8,
                   label=f'Obs mean: {obs_errors.mean():.2f}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    except ImportError:
        print("matplotlib not available. Skipping plot.")
        print(f"Latent errors: mean={latent_errors.mean():.4f}, std={latent_errors.std():.4f}")
        print(f"Obs errors: mean={obs_errors.mean():.4f}, std={obs_errors.std():.4f}")


def run_lo_nae_sat_error_analysis(model_path: str, N: int = 20, P: int = 280,
                                    m: int = 3, n_samples: int = 1000,
                                    ell: int = 11, device: str = 'auto',
                                    save_dir: str = '../experiments/error_analysis'):
    """
    Run the L&O-NAE-SAT error imbalance analysis.
    
    Args:
        model_path: path to trained MDM model checkpoint
        N: number of latent tokens
        P: number of observation tokens
        m: vocabulary size
        n_samples: number of test samples
        ell: number of latent tokens to mask
        device: computation device
        save_dir: directory to save results
    """
    from mdm_model import create_mdm_19m
    
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model
    pad_to = 512
    vocab_size = m + 3  # 0=mask, 1..m=latent, m+1=obs, m+2=pad
    model = create_mdm_19m(vocab_size=vocab_size, max_seq_len=pad_to, use_rope=True)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Create distribution
    dist = LONAESATDistribution(N=N, P=P, m=m, seed=42)
    
    print(f"Measuring error imbalance for L&O-NAE-SAT (N={N}, P={P})...")
    errors = measure_error_by_position(model, dist, n_samples=n_samples, ell=ell, device=device)
    
    print(f"Latent errors: mean={errors['latent_errors'].mean():.4f}")
    print(f"Obs errors: mean={errors['obs_errors'].mean():.4f}")
    
    # Save results
    np.save(os.path.join(save_dir, 'latent_errors.npy'), errors['latent_errors'])
    np.save(os.path.join(save_dir, 'obs_errors.npy'), errors['obs_errors'])
    
    # Plot
    plot_error_imbalance(
        errors['latent_errors'], errors['obs_errors'],
        save_path=os.path.join(save_dir, 'error_imbalance.png')
    )
    
    return errors


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Error Imbalance Analysis')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained MDM model')
    parser.add_argument('--N', type=int, default=20)
    parser.add_argument('--P', type=int, default=280)
    parser.add_argument('--m', type=int, default=3)
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--ell', type=int, default=11)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--save_dir', type=str, default='../experiments/error_analysis')
    args = parser.parse_args()
    
    run_lo_nae_sat_error_analysis(
        model_path=args.model_path,
        N=args.N, P=args.P, m=args.m,
        n_samples=args.n_samples, ell=args.ell,
        device=args.device, save_dir=args.save_dir
    )
