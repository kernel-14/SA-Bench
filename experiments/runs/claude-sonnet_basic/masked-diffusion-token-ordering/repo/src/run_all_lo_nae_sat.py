"""
Run All L&O-NAE-SAT Experiments
=================================
Runs all L&O-NAE-SAT experiments from Table 1 of the paper.

Usage:
    python run_all_lo_nae_sat.py --n_epochs 2000
"""

import argparse
import os
import sys
import subprocess
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# Table 1 configurations
TABLE1_CONFIGS = [
    {'N': 25, 'P': 275},
    {'N': 30, 'P': 270},
    {'N': 40, 'P': 260},
    {'N': 50, 'P': 250},
    {'N': 100, 'P': 200},
]

# Expected results from paper (Table 1)
EXPECTED_RESULTS = {
    (25, 275): {'vanilla': 78.06, 'adaptive': 93.76},
    (30, 270): {'vanilla': 75.70, 'adaptive': 93.54},
    (40, 260): {'vanilla': 74.60, 'adaptive': 92.21},
    (50, 250): {'vanilla': 67.94, 'adaptive': 90.01},
    (100, 200): {'vanilla': 62.84, 'adaptive': 88.91},
}


def main():
    parser = argparse.ArgumentParser(description='Run all L&O-NAE-SAT experiments')
    parser.add_argument('--n_epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_steps', type=int, default=50)
    parser.add_argument('--gumbel_noise', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--save_dir', type=str, default='../experiments/lo_nae_sat')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    all_results = {}
    
    for config in TABLE1_CONFIGS:
        N, P = config['N'], config['P']
        print(f"\n{'='*60}")
        print(f"Running experiment: N={N}, P={P}")
        print(f"{'='*60}")
        
        # Run training script
        cmd = [
            sys.executable, 'train_mdm_lo_nae_sat.py',
            '--N', str(N),
            '--P', str(P),
            '--n_epochs', str(args.n_epochs),
            '--batch_size', str(args.batch_size),
            '--lr', str(args.lr),
            '--n_steps', str(args.n_steps),
            '--gumbel_noise', str(args.gumbel_noise),
            '--seed', str(args.seed),
            '--device', args.device,
            '--save_dir', args.save_dir,
        ]
        
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        # Load results
        results_path = os.path.join(args.save_dir, f'results_N{N}_P{P}.json')
        if os.path.exists(results_path):
            with open(results_path) as f:
                exp_results = json.load(f)
            all_results[(N, P)] = exp_results['results']
    
    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY: Table 1 Reproduction")
    print("="*70)
    print(f"{'(N, P)':<15} {'Vanilla (paper)':<20} {'Vanilla (ours)':<20} {'Adaptive (paper)':<20} {'Adaptive (ours)':<20}")
    print("-"*70)
    
    for config in TABLE1_CONFIGS:
        N, P = config['N'], config['P']
        key = (N, P)
        
        paper_vanilla = EXPECTED_RESULTS[key]['vanilla']
        paper_adaptive = EXPECTED_RESULTS[key]['adaptive']
        
        if key in all_results:
            our_vanilla = all_results[key].get('vanilla', 0) * 100
            our_adaptive = all_results[key].get('top_prob_margin', 0) * 100
        else:
            our_vanilla = our_adaptive = float('nan')
        
        print(f"({N}, {P}){'':<8} {paper_vanilla:.2f}%{'':<13} {our_vanilla:.2f}%{'':<13} "
              f"{paper_adaptive:.2f}%{'':<13} {our_adaptive:.2f}%")
    
    # Save all results
    summary_path = os.path.join(args.save_dir, 'table1_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            str(k): v for k, v in all_results.items()
        }, f, indent=2)
    
    print(f"\nSummary saved to {summary_path}")


if __name__ == '__main__':
    main()
