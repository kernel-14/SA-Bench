"""
Train MDM on L&O-NAE-SAT Distribution
=======================================
Trains a Masked Diffusion Model on the L&O-NAE-SAT distribution
and evaluates vanilla vs. adaptive inference.

This reproduces the experiments in Section 3.3 and Table 1 of the paper.

Paper details (Appendix D.1.1):
- 19M MDM with RoPE, max sequence length 512
- Learning rate 0.001, batch size 128
- 300 epochs (same as Sudoku)

For error imbalance analysis (Appendix C.2.1):
- (N, P) = (20, 280), trained for 2000 iterations

Usage:
    python train_mdm_lo_nae_sat.py --N 25 --P 275
    python train_mdm_lo_nae_sat.py --N 50 --P 250
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lo_nae_sat import LONAESATDistribution, LONAESATDataset, create_lo_nae_sat_datasets
from mdm_model import MDMTransformer, create_mdm_19m
from mdm_training import MDMTrainer, mdm_loss, create_cosine_schedule_with_warmup
from adaptive_inference import mdm_sample_greedy, MASK_TOKEN


def evaluate_lo_nae_sat(model, dist, n_test=1000, strategy='vanilla', 
                          n_steps=50, gumbel_noise=0.5, device=None, seed=100):
    """
    Evaluate MDM on L&O-NAE-SAT by measuring accuracy on observation tokens.
    
    The paper measures accuracy in predicting observation tokens (not latent tokens),
    since the observation tokens have a deterministic relationship to the latents.
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    rng = np.random.RandomState(seed)
    
    # Generate test sequences
    test_seqs = dist.sample(n_test, rng)
    
    # Create fully masked versions (start from fully masked)
    x_init = torch.zeros(n_test, dist.L, dtype=torch.long, device=device)
    solutions = torch.tensor(test_seqs, dtype=torch.long, device=device)
    
    # Run inference in batches
    batch_size = 64
    n_correct = 0
    n_total = 0
    
    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        x_batch = x_init[start:end]
        sol_batch = solutions[start:end]
        
        generated = mdm_sample_greedy(
            model, x_batch, n_steps=n_steps,
            strategy=strategy, gumbel_noise=gumbel_noise
        )
        
        # Evaluate accuracy on observation tokens only
        obs_start = dist.N
        obs_correct = (generated[:, obs_start:] == sol_batch[:, obs_start:]).all(dim=-1)
        n_correct += obs_correct.sum().item()
        n_total += end - start
    
    return n_correct / n_total


def train_by_iterations(model, optimizer, scheduler, train_loader, device, 
                         n_iters, log_every=100):
    """Train for a fixed number of iterations."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    data_iter = iter(train_loader)
    
    for iteration in range(n_iters):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        x = x.to(device)
        
        optimizer.zero_grad()
        loss = mdm_loss(model, x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        n_batches += 1
        
        if (iteration + 1) % log_every == 0:
            avg_loss = total_loss / n_batches
            print(f"  Iter {iteration+1}/{n_iters}: loss={avg_loss:.4f}")
            total_loss = 0.0
            n_batches = 0
    
    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description='Train MDM on L&O-NAE-SAT')
    parser.add_argument('--N', type=int, default=25, help='Number of latent tokens')
    parser.add_argument('--P', type=int, default=275, help='Number of observation tokens')
    parser.add_argument('--m', type=int, default=3, help='Vocabulary size')
    parser.add_argument('--n_train', type=int, default=50000, help='Training samples')
    parser.add_argument('--n_val', type=int, default=5000, help='Validation samples')
    parser.add_argument('--n_test', type=int, default=1000, help='Test samples')
    parser.add_argument('--n_epochs', type=int, default=None, 
                        help='Training epochs (use instead of n_iters)')
    parser.add_argument('--n_iters', type=int, default=None,
                        help='Training iterations (paper uses 2000 for error analysis)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--n_steps', type=int, default=50, help='Inference steps')
    parser.add_argument('--gumbel_noise', type=float, default=0.5, help='Gumbel noise')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='auto', help='Device')
    parser.add_argument('--save_dir', type=str, default='../experiments/lo_nae_sat',
                        help='Directory to save results')
    args = parser.parse_args()
    
    # Default: use iterations if neither specified
    if args.n_epochs is None and args.n_iters is None:
        args.n_iters = 50000  # ~300 epochs with batch_size=128, n_train=50000
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"N={args.N}, P={args.P}, m={args.m}")
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create datasets
    # Paper uses pad_to=512 for the 19M model with max_seq_len=512
    # Paper pads with token value 2 (for (N,P)=(20,280), pads 212 tokens with value 2)
    pad_to = 512
    train_ds, val_ds, test_ds = create_lo_nae_sat_datasets(
        N=args.N, P=args.P, m=args.m,
        n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
        pad_to=pad_to, seed=args.seed
    )
    
    print(f"Vocab size: {train_ds.vocab_size}")
    print(f"Sequence length: {train_ds.L} (padded to {pad_to})")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                               num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0)
    
    # Create model (19M as per paper)
    model = create_mdm_19m(
        vocab_size=train_ds.vocab_size,
        max_seq_len=pad_to,
        use_rope=True
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Training
    print("\nStarting training...")
    
    if args.n_iters is not None:
        # Train by iterations
        n_iters = args.n_iters
        scheduler = create_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=n_iters // 20,
            num_training_steps=n_iters,
            min_lr_ratio=0.1
        )
        train_by_iterations(model, optimizer, scheduler, train_loader, device, n_iters)
    else:
        # Train by epochs
        n_steps_total = args.n_epochs * len(train_loader)
        scheduler = create_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=n_steps_total // 20,
            num_training_steps=n_steps_total,
            min_lr_ratio=0.1
        )
        
        trainer = MDMTrainer(model, optimizer, device, scheduler)
        best_val_loss = float('inf')
        
        for epoch in range(args.n_epochs):
            train_loss = trainer.train_epoch(train_loader)
            
            if (epoch + 1) % max(1, args.n_epochs // 20) == 0:
                val_loss = trainer.evaluate(val_loader)
                print(f"Epoch {epoch+1}/{args.n_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), 
                              os.path.join(args.save_dir, f'best_model_N{args.N}_P{args.P}.pt'))
    
    # Save final model
    torch.save(model.state_dict(), 
              os.path.join(args.save_dir, f'model_N{args.N}_P{args.P}.pt'))
    
    # Evaluate with different strategies
    dist = LONAESATDistribution(N=args.N, P=args.P, m=args.m, seed=args.seed)
    
    print("\nEvaluating inference strategies...")
    results = {}
    for strategy in ['vanilla', 'top_prob', 'top_prob_margin']:
        acc = evaluate_lo_nae_sat(
            model, dist, n_test=args.n_test,
            strategy=strategy, n_steps=args.n_steps,
            gumbel_noise=args.gumbel_noise if strategy != 'vanilla' else 0.0,
            device=device
        )
        results[strategy] = acc
        print(f"  {strategy}: {acc*100:.2f}%")
    
    # Save results
    import json
    results_path = os.path.join(args.save_dir, f'results_N{args.N}_P{args.P}.json')
    with open(results_path, 'w') as f:
        json.dump({
            'N': args.N, 'P': args.P, 'm': args.m,
            'results': results
        }, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    print("\nSummary (Table 1 from paper):")
    print(f"  (N={args.N}, P={args.P}): Vanilla={results['vanilla']*100:.2f}%, "
          f"Adaptive={results['top_prob_margin']*100:.2f}%")


if __name__ == '__main__':
    main()
