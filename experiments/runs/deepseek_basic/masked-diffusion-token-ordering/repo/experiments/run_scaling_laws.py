"""
Scaling Laws Experiment (Section 3.2)
======================================
Reproduces the scaling law analysis from Figure 2 (left).

Compares:
- ARM (identity permutation, left-to-right)
- MDM (order-agnostic training)
- pi-learners with different distances from identity:
  - Much-closer (sqrt(L) swaps)
  - Closer (L/10 swaps)
  - Uniform random permutations

Uses IsoFLOP analysis: for each FLOPs budget, vary model size
and find the optimal validation loss.

Training configuration from Appendix C.1:
- AdamW: beta1=0.9, beta2=0.95, weight_decay=0.1
- Cosine LR: max 4e-4, min 4e-5
- Learnable positional embeddings
- L = 2048
"""

import torch
import numpy as np
import os
import sys
import json
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mdm import MDMTransformer, MDMConfig, MaskedDiffusionModel
from models.arm import CausalTransformer, AutoregressiveModel
from training.train_mdm import MDMTrainer, SequenceDataset
from utils.permutations import (
    random_permutation, identity_permutation,
    interpolated_permutation, sample_permutations_for_interpolation
)
from evaluation.metrics import evaluate_pi_learner_likelihood


def run_scaling_laws_experiment(
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    output_dir: str,
    device: str = 'cpu',
    L: int = 2048,
    vocab_size: int = 50257,
):
    """Run scaling laws experiment from Section 3.2."""
    os.makedirs(output_dir, exist_ok=True)
    
    train_dataset = SequenceDataset(train_data)
    val_dataset = SequenceDataset(val_data)
    
    model_sizes = [(256, 4), (384, 6), (512, 8), (640, 10), (768, 12)]
    flops_budgets = [1e15, 3e15, 1e16, 3e16, 1e17]
    
    results = {}
    
    for perm_name, perm_fn in [
        ('identity', lambda: identity_permutation(L)),
        ('much_closer', lambda: interpolated_permutation(L, int(np.sqrt(L)))),
        ('closer', lambda: interpolated_permutation(L, L // 10)),
        ('uniform', lambda: random_permutation(L)),
    ]:
        perm_results = {}
        for C in flops_budgets:
            best_val_loss = float('inf')
            for d_model, n_layers in model_sizes:
                N_params = 12 * d_model * d_model * n_layers
                total_tokens = C / (6 * N_params)
                total_steps = int(total_tokens / (32 * L))
                if total_steps < 100:
                    continue
                
                config = MDMConfig(
                    vocab_size=vocab_size, seq_length=L,
                    d_model=d_model, n_layers=n_layers,
                    n_heads=d_model // 64, d_ff=4 * d_model,
                    max_seq_length=L,
                )
                denoiser = MDMTransformer(config)
                mdm = MaskedDiffusionModel(denoiser, config)
                
                pi = perm_fn()
                trainer = MDMTrainer(mdm, device=device)
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
                pi_tensor = torch.tensor(pi, device=device)
                
                for step in range(total_steps):
                    try:
                        batch = next(iter(train_loader))
                    except StopIteration:
                        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                        batch = next(iter(train_loader))
                    trainer.train_step_pi_learner(batch, pi_tensor)
                
                val_loss = evaluate_pi_learner_likelihood(mdm, val_loader, pi, device)
                best_val_loss = min(best_val_loss, val_loss)
            
            perm_results[C] = best_val_loss
        results[perm_name] = perm_results
    
    with open(os.path.join(output_dir, 'scaling_laws_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='results/scaling_laws')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--L', type=int, default=128)
    parser.add_argument('--vocab_size', type=int, default=1000)
    parser.add_argument('--n_samples', type=int, default=10000)
    args = parser.parse_args()
    
    rng = np.random.RandomState(42)
    train_data = torch.tensor(
        rng.randint(1, args.vocab_size, (args.n_samples, args.L)), dtype=torch.long
    )
    val_data = torch.tensor(
        rng.randint(1, args.vocab_size, (args.n_samples // 10, args.L)), dtype=torch.long
    )
    
    results = run_scaling_laws_experiment(
        train_data, val_data, output_dir=args.output_dir,
        device=args.device, L=args.L, vocab_size=args.vocab_size,
    )
