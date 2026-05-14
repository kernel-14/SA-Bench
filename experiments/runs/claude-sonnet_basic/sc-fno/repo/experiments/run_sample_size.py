"""
Training data volume experiment (Section 3.3 of the paper).
Tests how model performance varies with training data size.

From the paper:
- Models trained with different numbers of samples
- SC-FNO maintains higher accuracy with fewer samples
- FNO performance degrades faster as training data decreases
"""

import torch
import numpy as np
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.run_pde1 import prepare_pde1_data, build_pde1_model
from experiments.sc_fno_experiment import train_sc_fno, evaluate_model


def run_sample_size_experiment(
    sample_sizes=[100, 200, 500, 1000, 2000],
    modes=["fno", "sc_fno"],
    device="cpu",
    save_dir="results/sample_size"
):
    """
    Run experiment varying training data size.
    
    For each sample size, train FNO and SC-FNO and evaluate on a fixed test set.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate full dataset
    print("Generating full PDE1 dataset...")
    max_samples = max(sample_sizes)
    train_data_full, val_data_full, test_data = prepare_pde1_data(n_samples=max_samples + 500)
    
    results = {mode: {} for mode in modes}
    
    for n_samples in sample_sizes:
        print(f"\n{'='*60}")
        print(f"Training with {n_samples} samples")
        print(f"{'='*60}")
        
        # Subsample training data
        train_data = {
            k: v[:n_samples] if isinstance(v, np.ndarray) else v
            for k, v in train_data_full.items()
        }
        
        for mode in modes:
            print(f"  Mode: {mode}")
            
            model = build_pde1_model()
            
            config = {
                "mode": mode,
                "n_epochs": 500,
                "batch_size": min(4, n_samples),
                "lr": 1e-3,
                "c1": 1.0,
                "c2": 1.0,
                "n_sample_points": 50,
            }
            
            history = train_sc_fno(model, train_data, val_data_full, config, device=device, verbose=False)
            metrics = evaluate_model(model, test_data, device=device)
            
            results[mode][n_samples] = {
                "u_r2": metrics["u_r2"],
                "u_relative_l2": metrics["u_relative_l2"],
                "avg_epoch_time": history["avg_epoch_time"]
            }
            
            # Add Jacobian metrics
            for key, val in metrics.items():
                if "jac" in key:
                    results[mode][n_samples][key] = val
            
            print(f"    u R2: {metrics['u_r2']:.4f}, u L2: {metrics['u_relative_l2']:.4f}")
    
    # Save results
    with open(os.path.join(save_dir, "sample_size_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary table
    print("\n" + "="*80)
    print("Sample Size Experiment Results (PDE1)")
    print("="*80)
    print(f"{'N':>8} {'FNO u R2':>12} {'SC-FNO u R2':>14} {'FNO Jac R2':>12} {'SC-FNO Jac R2':>15}")
    print("-"*80)
    
    for n in sample_sizes:
        fno_u_r2 = results.get("fno", {}).get(n, {}).get("u_r2", float("nan"))
        sc_u_r2 = results.get("sc_fno", {}).get(n, {}).get("u_r2", float("nan"))
        
        # Average Jacobian R2 across parameters
        fno_jac_r2 = np.mean([v for k, v in results.get("fno", {}).get(n, {}).items() if "jac" in k and "r2" in k] or [float("nan")])
        sc_jac_r2 = np.mean([v for k, v in results.get("sc_fno", {}).get(n, {}).items() if "jac" in k and "r2" in k] or [float("nan")])
        
        print(f"{n:>8} {fno_u_r2:>12.4f} {sc_u_r2:>14.4f} {fno_jac_r2:>12.4f} {sc_jac_r2:>15.4f}")
    
    return results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    run_sample_size_experiment(
        sample_sizes=[100, 200, 500, 1000, 2000],
        modes=["fno", "sc_fno"],
        device=device
    )
