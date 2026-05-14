"""
Train MDM on Sudoku Puzzles
============================
Trains a 6M parameter MDM on Sudoku puzzles and evaluates
vanilla vs. adaptive inference strategies.

This reproduces the experiments in Section 4.2 and Table 2 of the paper.

The paper uses the dataset from Shah et al. (2024), which filters puzzles
from Radcliffe (2020) that can be solved using 7 fixed strategies.

Usage:
    python train_mdm_sudoku.py --data_path /path/to/sudoku_data.csv
    python train_mdm_sudoku.py --use_synthetic  # For testing without real data
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sudoku import SudokuDataset, evaluate_sudoku_accuracy, generate_synthetic_sudoku_data
from mdm_model import MDMTransformer, create_mdm_6m
from mdm_training import MDMTrainer, create_cosine_schedule_with_warmup
from adaptive_inference import mdm_sample_greedy, MASK_TOKEN


class SudokuMDMDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper for MDM training on Sudoku.
    
    For MDM training, we use the solution as the clean sequence.
    The model learns to predict any masked subset of the solution.
    """
    
    def __init__(self, sudoku_dataset: SudokuDataset):
        self.dataset = sudoku_dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Return only the solution for MDM training
        _, solution = self.dataset[idx]
        return solution


def evaluate_sudoku_with_puzzle(model, dataset, strategy='top_prob_margin',
                                  n_steps=50, gumbel_noise=0.5,
                                  batch_size=64, device=None, max_samples=None):
    """
    Evaluate MDM on Sudoku puzzles.
    
    Unlike unconditional generation, here we start from the puzzle
    (with empty cells masked) and fill in the missing digits.
    """
    from adaptive_inference import mdm_sample_greedy
    
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    n_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    n_correct = 0
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        
        puzzles = []
        solutions = []
        for i in range(start, end):
            p, s = dataset[i]
            puzzles.append(p)
            solutions.append(s)
        
        puzzles = torch.stack(puzzles).to(device)
        solutions = torch.stack(solutions).to(device)
        
        # Run inference starting from the puzzle (partially masked)
        generated = mdm_sample_greedy(
            model, puzzles, n_steps=n_steps,
            strategy=strategy, gumbel_noise=gumbel_noise
        )
        
        # Check if the generated solution matches the ground truth
        correct = (generated == solutions).all(dim=-1)
        n_correct += correct.sum().item()
    
    return n_correct / n_samples


def main():
    parser = argparse.ArgumentParser(description='Train MDM on Sudoku')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to Sudoku dataset CSV')
    parser.add_argument('--use_synthetic', action='store_true',
                        help='Use synthetic Sudoku data for testing')
    parser.add_argument('--n_train', type=int, default=None,
                        help='Number of training samples (None = use all)')
    parser.add_argument('--n_epochs', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--n_steps', type=int, default=50,
                        help='Number of inference steps')
    parser.add_argument('--gumbel_noise', type=float, default=0.5,
                        help='Gumbel noise coefficient')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto/cpu/cuda)')
    parser.add_argument('--save_dir', type=str, default='../experiments/sudoku',
                        help='Directory to save results')
    parser.add_argument('--eval_only', type=str, default=None,
                        help='Path to model checkpoint for evaluation only')
    args = parser.parse_args()
    
    # Setup
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"Device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load data
    if args.use_synthetic or args.data_path is None:
        print("Using synthetic Sudoku data...")
        n_total = 10000
        puzzles, solutions = generate_synthetic_sudoku_data(n_total, seed=args.seed)
        
        n_train = int(0.8 * n_total)
        n_val = int(0.1 * n_total)
        
        train_dataset = SudokuDataset(
            puzzles=puzzles[:n_train], solutions=solutions[:n_train]
        )
        val_dataset = SudokuDataset(
            puzzles=puzzles[n_train:n_train+n_val],
            solutions=solutions[n_train:n_train+n_val]
        )
        test_dataset = SudokuDataset(
            puzzles=puzzles[n_train+n_val:],
            solutions=solutions[n_train+n_val:]
        )
    else:
        print(f"Loading Sudoku data from {args.data_path}...")
        full_dataset = SudokuDataset(data_path=args.data_path, max_samples=args.n_train)
        
        n_total = len(full_dataset)
        n_val = min(1000, int(0.1 * n_total))
        n_test = min(1000, int(0.1 * n_total))
        n_train = n_total - n_val - n_test
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(args.seed)
        )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create MDM training dataset (uses solutions only)
    train_mdm = SudokuMDMDataset(train_dataset if hasattr(train_dataset, '__getitem__') 
                                   else train_dataset)
    
    train_loader = DataLoader(train_mdm, batch_size=args.batch_size, shuffle=True,
                               num_workers=0)
    
    # Create model (6M as per paper for Sudoku)
    # Vocab: 0=mask, 1-9=digits
    SUDOKU_VOCAB_SIZE = 10  # 0-9
    model = create_mdm_6m(
        vocab_size=SUDOKU_VOCAB_SIZE,
        max_seq_len=81,
        use_rope=True
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    if args.eval_only:
        print(f"Loading checkpoint from {args.eval_only}")
        model.load_state_dict(torch.load(args.eval_only, map_location=device))
    else:
        # Training
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
        
        n_steps_total = args.n_epochs * len(train_loader)
        scheduler = create_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=n_steps_total // 20,
            num_training_steps=n_steps_total,
            min_lr_ratio=0.1
        )
        
        trainer = MDMTrainer(model, optimizer, device, scheduler)
        
        print("\nTraining...")
        best_val_acc = 0.0
        
        for epoch in range(args.n_epochs):
            train_loss = trainer.train_epoch(train_loader)
            
            if (epoch + 1) % 50 == 0:
                # Quick validation
                val_acc = evaluate_sudoku_with_puzzle(
                    model, val_dataset, strategy='top_prob_margin',
                    n_steps=args.n_steps, gumbel_noise=args.gumbel_noise,
                    batch_size=args.batch_size, device=device,
                    max_samples=200
                )
                print(f"Epoch {epoch+1}/{args.n_epochs}: loss={train_loss:.4f}, "
                      f"val_acc={val_acc*100:.2f}%")
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(),
                              os.path.join(args.save_dir, 'best_model.pt'))
        
        # Load best model
        best_path = os.path.join(args.save_dir, 'best_model.pt')
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location=device))
    
    # Evaluation
    print("\nEvaluating inference strategies...")
    results = {}
    
    for strategy in ['vanilla', 'top_prob', 'top_prob_margin']:
        acc = evaluate_sudoku_with_puzzle(
            model, test_dataset, strategy=strategy,
            n_steps=args.n_steps,
            gumbel_noise=args.gumbel_noise if strategy != 'vanilla' else 0.0,
            batch_size=args.batch_size, device=device
        )
        results[strategy] = acc
        print(f"  {strategy}: {acc*100:.2f}%")
    
    # Save results
    import json
    results_path = os.path.join(args.save_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    print("\nTable 2 comparison:")
    print(f"  MDM (vanilla):          {results['vanilla']*100:.2f}%  (paper: 6.88%)")
    print(f"  MDM (Top probability):  {results['top_prob']*100:.2f}%  (paper: 18.51%)")
    print(f"  MDM (Top prob. margin): {results['top_prob_margin']*100:.2f}%  (paper: 89.49%)")


if __name__ == '__main__':
    main()
