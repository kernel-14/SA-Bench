"""
Main script to run all SC-FNO experiments from the paper.

Usage:
    python run_all_experiments.py [--experiment {ode1,ode2,pde1,pde2,pde3,pde4,all}]
                                  [--mode {fno,sc_fno,fno_pinn,sc_fno_pinn,all}]
                                  [--device {cpu,cuda}]
"""

import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_ode1(mode="all", device="cpu"):
    """Run ODE1 experiments."""
    from experiments.run_ode1 import run_ode1_experiment, run_all_ode1_experiments
    
    if mode == "all":
        return run_all_ode1_experiments(device=device)
    else:
        model, metrics, history = run_ode1_experiment(mode=mode, device=device)
        return {mode: metrics}


def run_pde1(mode="all", device="cpu"):
    """Run PDE1 experiments."""
    from experiments.run_pde1 import run_pde1_experiment
    
    modes = ["fno", "sc_fno", "fno_pinn", "sc_fno_pinn"] if mode == "all" else [mode]
    results = {}
    for m in modes:
        _, metrics, _ = run_pde1_experiment(mode=m, device=device)
        results[m] = metrics
    return results


def run_pde2(mode="all", device="cpu"):
    """Run PDE2 experiments."""
    from experiments.run_pde2_pde3 import run_pde2_experiment, run_pde2_zoned_experiment
    
    modes = ["fno", "sc_fno"] if mode == "all" else [mode]
    results = {}
    for m in modes:
        _, metrics, _ = run_pde2_experiment(mode=m, device=device)
        results[m] = metrics
    return results


def run_pde3(mode="all", device="cpu"):
    """Run PDE3 experiments."""
    from experiments.run_pde2_pde3 import run_pde3_experiment
    
    modes = ["fno", "sc_fno"] if mode == "all" else [mode]
    results = {}
    for m in modes:
        _, metrics, _ = run_pde3_experiment(mode=m, device=device)
        results[m] = metrics
    return results


def run_pde4(mode="all", device="cpu"):
    """Run PDE4 experiments."""
    from experiments.run_pde4 import run_pde4_experiment
    
    modes = ["fno", "sc_fno"] if mode == "all" else [mode]
    results = {}
    for n_samples in [500, 100]:
        for m in modes:
            _, metrics, _ = run_pde4_experiment(mode=m, n_samples=n_samples, device=device)
            results[f"{m}_N{n_samples}"] = metrics
    return results


def run_high_dim(mode="all", device="cpu"):
    """Run high-dimensional parameter space experiment (zoned PDE2)."""
    from experiments.run_pde2_pde3 import run_pde2_zoned_experiment
    
    modes = ["fno", "sc_fno"] if mode == "all" else [mode]
    results = {}
    for n_samples in [500, 100]:
        for m in modes:
            _, metrics, _ = run_pde2_zoned_experiment(mode=m, n_samples=n_samples, device=device)
            results[f"{m}_N{n_samples}"] = metrics
    return results


def main():
    parser = argparse.ArgumentParser(description="Run SC-FNO experiments")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["ode1", "ode2", "pde1", "pde2", "pde3", "pde4", "high_dim", "all"],
                        help="Which experiment to run")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["fno", "sc_fno", "fno_pinn", "sc_fno_pinn", "all"],
                        help="Which model configuration to use")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cpu or cuda). Auto-detected if not specified.")
    
    args = parser.parse_args()
    
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    print(f"Running experiment: {args.experiment}")
    print(f"Model mode: {args.mode}")
    
    os.makedirs("results", exist_ok=True)
    
    if args.experiment in ["ode1", "all"]:
        print("\n" + "="*60)
        print("ODE1: Composite Harmonic Oscillator")
        print("="*60)
        run_ode1(mode=args.mode, device=device)
    
    if args.experiment in ["pde1", "all"]:
        print("\n" + "="*60)
        print("PDE1: Generalized Nonlinear Damped Wave Equation")
        print("="*60)
        run_pde1(mode=args.mode, device=device)
    
    if args.experiment in ["pde2", "all"]:
        print("\n" + "="*60)
        print("PDE2: Forced Burgers Equation")
        print("="*60)
        run_pde2(mode=args.mode, device=device)
    
    if args.experiment in ["pde3", "all"]:
        print("\n" + "="*60)
        print("PDE3: Navier-Stokes (Vorticity)")
        print("="*60)
        run_pde3(mode=args.mode, device=device)
    
    if args.experiment in ["pde4", "all"]:
        print("\n" + "="*60)
        print("PDE4: Allen-Cahn Equation")
        print("="*60)
        run_pde4(mode=args.mode, device=device)
    
    if args.experiment in ["high_dim", "all"]:
        print("\n" + "="*60)
        print("High-Dimensional: Zoned PDE2 (82 parameters)")
        print("="*60)
        run_high_dim(mode=args.mode, device=device)
    
    print("\nAll experiments completed!")


if __name__ == "__main__":
    main()
