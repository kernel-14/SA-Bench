#!/usr/bin/env python3
"""Run all experiments and verifications from the paper.

Usage:
    python run_all.py [--quick] [--skip-mscoco]

Options:
    --quick         Run with fewer trials for faster execution.
    --skip-mscoco   Skip the MS-COCO experiment (which uses synthetic data).
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser(description="Run all experiments.")
    parser.add_argument("--quick", action="store_true",
                        help="Run with fewer trials.")
    parser.add_argument("--skip-mscoco", action="store_true",
                        help="Skip MS-COCO experiment.")
    args = parser.parse_args()

    n_trials = 200 if args.quick else 10000

    print("=" * 70)
    print("Conformal Prediction as Bayesian Quadrature")
    print("Reproduction Experiments")
    print("=" * 70)

    # 1. Theoretical Verification
    print("\n" + "=" * 70)
    print("PART 1: Theoretical Verification")
    print("=" * 70)
    from src import theoretical_verification

    r1 = theoretical_verification.verify_quantile_spacing_distribution(
        n=10, n_simulations=10000 if args.quick else 100000
    )
    print(f"Lemma 4.2: Quantile spacings mean MAE: {r1['mean_absolute_error_mean']:.6f}")

    r2 = theoretical_verification.verify_E_L_plus_formula(
        n=10, n_simulations=10000 if args.quick else 100000
    )
    print(f"E[L^+] formula diff: {r2['difference']:.8f}")

    r4 = theoretical_verification.verify_scp_recovery(n=100, alpha=0.1)
    print(f"SCP recovery: E[L^+] <= alpha: {r4['E_L_plus_le_alpha']}")

    # 2. Synthetic Binomial Experiment (Table 1)
    print("\n" + "=" * 70)
    print("PART 2: Synthetic Binomial Experiment (Section 5.1, Table 1)")
    print("=" * 70)
    from experiments.synthetic_binomial import run_experiment as run_binomial
    results_binomial = run_binomial(
        n_trials=n_trials,
        n=10,
        K=4,
        alpha=0.4,
        B=1.0,
        beta=0.95,
        n_dirichlet_samples=500 if args.quick else 1000,
        seed=42,
    )
    from experiments.synthetic_binomial import print_results as print_binomial
    print_binomial(results_binomial)

    # 3. Synthetic Heteroskedastic Experiment (Table 2)
    print("\n" + "=" * 70)
    print("PART 3: Synthetic Heteroskedastic Experiment (Section 5.2, Table 2)")
    print("=" * 70)
    from experiments.synthetic_heteroskedastic import run_experiment as run_hetero
    results_hetero = run_hetero(
        n_trials=n_trials,
        n=200,
        alpha=0.1,
        B=1.0,
        beta=0.95,
        n_dirichlet_samples=500 if args.quick else 1000,
        seed=42,
    )
    from experiments.synthetic_heteroskedastic import print_results as print_hetero
    print_hetero(results_hetero)

    # 4. MS-COCO Experiment (Table 3)
    if not args.skip_mscoco:
        print("\n" + "=" * 70)
        print("PART 4: MS-COCO Experiment (Section 5.3, Table 3)")
        print("=" * 70)
        print("Note: Using synthetic data to demonstrate methodology.")
        print("For real MS-COCO results, download the dataset and use a pre-trained model.")
        from experiments.mscoco import run_synthetic_mscoco_experiment, print_results as print_mscoco
        results_mscoco = run_synthetic_mscoco_experiment(
            n_trials=n_trials,
            n_cal=1000,
            n_test=3952,
            n_classes=80,
            alpha=0.1,
            beta=0.95,
            seed=42,
        )
        print_mscoco(results_mscoco)

    print("\n" + "=" * 70)
    print("All experiments completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
