"""
Run all experiments from:
  "Conformal Prediction as Bayesian Quadrature"
  Jake C. Snell, Thomas L. Griffiths (ICML 2025)

Usage
-----
  python run_all_experiments.py                  # all three experiments
  python run_all_experiments.py --experiments 1  # only binomial
  python run_all_experiments.py --experiments 1 2
"""

import argparse
import os

os.makedirs("results", exist_ok=True)


def run_experiment_1():
    """Section 5.1 — Synthetic Binomial Data (Table 1, Figure 3, Figure 4)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Synthetic Binomial Data (Section 5.1)")
    print("=" * 70)

    from experiment_binomial import (
        run_binomial_experiment,
        plot_lambda_histograms,
        plot_L_plus_density,
        print_table,
    )

    print("Parameters: n=10, K=4, alpha=0.4, beta=0.95, M=10000")
    results = run_binomial_experiment(
        n=10, K=4, alpha=0.4, beta=0.95, M=10000,
        B=1.0, n_bq_samples=1000, seed=42
    )

    print_table(results)
    print("Mean risks (paper: CRC=0.3363, Ours=0.1758):")
    for name, res in results.items():
        print(f"  {name}: {res['mean_risk']:.4f}")

    plot_lambda_histograms(results, alpha=0.4,
                           save_path="results/figure3_binomial_histograms.png")
    plot_L_plus_density(n=10, K=4, B=1.0, n_samples=100000,
                        lambda_values=[0.7, 0.8, 0.9], seed=42,
                        save_path="results/figure4_L_plus_density.png")
    return results


def run_experiment_2():
    """Section 5.2 — Synthetic Heteroskedastic Data (Table 2)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Synthetic Heteroskedastic Data (Section 5.2)")
    print("=" * 70)

    from experiment_heteroskedastic import (
        run_heteroskedastic_experiment,
        print_table,
    )

    print("Parameters: n=200, alpha=0.1, beta=0.95, M=10000")
    results = run_heteroskedastic_experiment(
        n=200, alpha=0.1, beta=0.95, M=10000,
        B=1.0, n_bq_samples=1000, seed=42
    )

    print_table(results)
    return results


def run_experiment_3():
    """Section 5.3 — MS-COCO False Negative Rate (Table 3)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: MS-COCO False Negative Rate (Section 5.3)")
    print("=" * 70)

    from experiment_coco import load_coco_data, run_coco_experiment, print_table

    try:
        scores, labels = load_coco_data("data/coco")
        print(f"Loaded COCO data: {scores.shape[0]} examples, {scores.shape[1]} classes")
        print("Parameters: n_cal=1000, alpha=0.1, beta=0.95, M=10000")

        results = run_coco_experiment(
            scores, labels,
            n_cal=1000, alpha=0.1, beta=0.95, M=10000,
            B=1.0, n_bq_samples=1000, seed=42
        )
        print_table(results)
        return results

    except FileNotFoundError as e:
        print(f"Skipping: {e}")
        print("Run 'python download_coco_data.py' to download the required data.")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce experiments from 'Conformal Prediction as Bayesian Quadrature'"
    )
    parser.add_argument(
        "--experiments", nargs="+", type=int, default=[1, 2, 3],
        choices=[1, 2, 3],
        help="Which experiments to run (1=binomial, 2=heteroskedastic, 3=coco)"
    )
    args = parser.parse_args()

    print("Conformal Prediction as Bayesian Quadrature — Experiments")
    print("=" * 70)

    all_results = {}
    if 1 in args.experiments:
        all_results["binomial"] = run_experiment_1()
    if 2 in args.experiments:
        all_results["heteroskedastic"] = run_experiment_2()
    if 3 in args.experiments:
        all_results["coco"] = run_experiment_3()

    print("\n" + "=" * 70)
    print("All experiments complete. Results saved to results/")
    print("=" * 70)
    return all_results


if __name__ == "__main__":
    main()
