"""
Low-data regime experiment: FNO trained on 25 trajectories of 1D PDEs.

Reproduces Table 1 (Burgers'), Table 4 (Hyper Diffusion), Table 5 (KS conservative)
from the LUNO paper.

Methods evaluated:
  - Input Perturbations
  - Deep Ensemble (10 members)
  - Sample-Iso
  - LUNO-Iso
  - Sample-LA
  - LUNO-LA

Usage:
  python experiments/run_low_data.py --pde burgers
  python experiments/run_low_data.py --pde hyper_diffusion
  python experiments/run_low_data.py --pde ks_conservative
"""

import argparse
import os
import sys
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.ensemble import ensemble_mean_std
from baselines.input_perturbations import input_perturbation_mean_std
from calibrate import (
    calibrate_input_perturbation,
    calibrate_luno_iso,
    calibrate_luno_la,
    calibrate_sample_iso,
    calibrate_sample_la,
)
from config import DEFAULT_CONFIG
from data.apebench_data import (
    PDEDataset,
    generate_apebench_data,
    make_input_output_pairs,
)
from evaluate import compute_all_metrics
from luno.linearization import luno_predictive_std
from luno.sampling import sample_pushforward_mean_std
from luno.weight_uncertainty import (
    IsotropicGaussian,
    LowRankLaplace,
    compute_ggn_low_rank_streaming,
    get_flat_params,
)
from models.fno import FNO1d
from train import train, train_ensemble


def make_fno_1d(d_in: int, seed: int = 0) -> FNO1d:
    """Create a 1D FNO with paper hyperparameters."""
    cfg = DEFAULT_CONFIG.fno
    rngs = nnx.Rngs(params=jax.random.PRNGKey(seed))
    return FNO1d(
        d_in=d_in,
        d_out=1,
        n_modes=cfg.n_modes,
        d_v=cfg.d_v,
        n_layers=cfg.n_layers,
        padding=cfg.padding,
        projection_hidden=cfg.projection_hidden,
        activation=cfg.activation,
        rngs=rngs,
    )


def run_low_data_experiment(pde_name: str, verbose: bool = True) -> Dict[str, Dict]:
    """
    Run the full low-data regime experiment for a given PDE.

    Args:
        pde_name: "burgers", "hyper_diffusion", or "ks_conservative"
        verbose: print progress
    Returns:
        results: dict mapping method name to metrics dict
    """
    cfg = DEFAULT_CONFIG
    key = jax.random.PRNGKey(cfg.train.seed)

    # -------------------------------------------------------------------------
    # 1. Generate data
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n{'='*60}")
        print(f"Low-data experiment: {pde_name}")
        print(f"{'='*60}")
        print("Generating data...")

    data = generate_apebench_data(
        pde_name,
        n_train=cfg.low_data.n_train,
        n_val=cfg.low_data.n_val,
        n_test=cfg.low_data.n_test,
        seed=cfg.train.seed,
    )

    train_inputs, train_outputs = make_input_output_pairs(
        data["train"], n_history=cfg.low_data.n_history
    )
    val_inputs, val_outputs = make_input_output_pairs(
        data["val"], n_history=cfg.low_data.n_history
    )
    test_inputs, test_outputs = make_input_output_pairs(
        data["test"], n_history=cfg.low_data.n_history
    )

    d_in = train_inputs.shape[-1]  # n_history

    train_loader = PDEDataset(train_inputs, train_outputs, batch_size=1, shuffle=True)
    val_loader = PDEDataset(val_inputs, val_outputs, batch_size=1, shuffle=False)
    test_loader = PDEDataset(test_inputs, test_outputs, batch_size=1, shuffle=False)

    if verbose:
        print(f"Train: {train_inputs.shape[0]} pairs, Val: {val_inputs.shape[0]}, Test: {test_inputs.shape[0]}")

    # -------------------------------------------------------------------------
    # 2. Train main FNO
    # -------------------------------------------------------------------------
    if verbose:
        print("\nTraining main FNO...")

    model = make_fno_1d(d_in, seed=cfg.train.seed)
    train(
        model,
        train_loader,
        val_loader=val_loader,
        n_epochs=cfg.train.low_data_epochs,
        learning_rate=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        warmup_steps=cfg.train.warmup_steps,
        verbose=verbose,
    )

    # -------------------------------------------------------------------------
    # 3. Compute GGN low-rank approximation
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\nComputing GGN low-rank approximation (rank={cfg.laplace.rank})...")

    flat_params = get_flat_params(model)
    V, n_ggn = compute_ggn_low_rank_streaming(
        model,
        flat_params,
        iter(train_loader),
        rank=cfg.laplace.rank,
        n_data=cfg.laplace.n_data_ggn_low_data,
        key=jax.random.PRNGKey(1),
    )

    if verbose:
        print(f"GGN computed using {n_ggn} data points, V shape: {V.shape}")

    # -------------------------------------------------------------------------
    # 4. Train ensemble
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\nTraining ensemble ({cfg.eval.n_ensemble} members)...")

    def model_factory(seed):
        return make_fno_1d(d_in, seed=seed)

    ensemble_models = train_ensemble(
        model_factory,
        train_loader,
        val_loader=val_loader,
        n_ensemble=cfg.eval.n_ensemble,
        n_epochs=cfg.train.low_data_epochs,
        learning_rate=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        warmup_steps=cfg.train.warmup_steps,
        base_seed=100,
        verbose=verbose,
    )

    # -------------------------------------------------------------------------
    # 5. Calibrate all methods
    # -------------------------------------------------------------------------
    if verbose:
        print("\nCalibrating uncertainty methods...")

    # Input Perturbations
    sigma_perturb = calibrate_input_perturbation(
        model, val_loader,
        n_samples=cfg.eval.n_samples,
        n_val=cfg.eval.n_test_pairs,
        verbose=verbose,
    )

    # LUNO-Iso
    sigma2_luno_iso = calibrate_luno_iso(
        model, val_loader,
        n_val=cfg.eval.n_test_pairs,
        verbose=verbose,
    )

    # Sample-Iso
    sigma2_sample_iso = calibrate_sample_iso(
        model, val_loader,
        n_samples=cfg.eval.n_samples,
        n_val=cfg.eval.n_test_pairs,
        verbose=verbose,
    )

    # LUNO-LA
    sigma2_luno_la = calibrate_luno_la(
        model, V, n_ggn, val_loader,
        n_val=cfg.eval.n_test_pairs,
        verbose=verbose,
    )

    # Sample-LA
    sigma2_sample_la = calibrate_sample_la(
        model, V, n_ggn, val_loader,
        n_samples=cfg.eval.n_samples,
        n_val=cfg.eval.n_test_pairs,
        verbose=verbose,
    )

    # -------------------------------------------------------------------------
    # 6. Evaluate all methods on test set
    # -------------------------------------------------------------------------
    if verbose:
        print("\nEvaluating on test set...")

    results = {}
    key = jax.random.PRNGKey(99)

    # Collect test data
    test_a_list, test_u_list = [], []
    for a_b, u_b in test_loader:
        for i in range(a_b.shape[0]):
            if len(test_a_list) >= cfg.eval.n_test_pairs:
                break
            test_a_list.append(np.array(a_b[i:i + 1]))
            test_u_list.append(np.array(u_b[i:i + 1]))
        if len(test_a_list) >= cfg.eval.n_test_pairs:
            break

    def evaluate_method(predict_fn, name):
        means, stds = [], []
        for a_i in test_a_list:
            m, s = predict_fn(jnp.array(a_i))
            means.append(np.array(m))
            stds.append(np.array(s))
        mean_pred = np.concatenate(means, axis=0)
        std_pred = np.concatenate(stds, axis=0)
        target = np.concatenate(test_u_list, axis=0)
        metrics = compute_all_metrics(mean_pred, std_pred, target)
        if verbose:
            print(f"  {name:25s}: RMSE={metrics['rmse']:.3e}, χ²={metrics['chi2']:.3f}, NLL={metrics['nll']:.4f}")
        return metrics

    # Input Perturbations
    def predict_input_perturb(a):
        nonlocal key
        key, sk = jax.random.split(key)
        return input_perturbation_mean_std(model, a, sigma_perturb, cfg.eval.n_samples, sk)
    results["Input Perturbations"] = evaluate_method(predict_input_perturb, "Input Perturbations")

    # Ensemble
    def predict_ensemble(a):
        return ensemble_mean_std(ensemble_models, a)
    results["Ensemble"] = evaluate_method(predict_ensemble, "Ensemble")

    # Sample-Iso
    wu_sample_iso = IsotropicGaussian(mean=flat_params, sigma2=sigma2_sample_iso)
    def predict_sample_iso(a):
        nonlocal key
        key, sk = jax.random.split(key)
        return sample_pushforward_mean_std(model, wu_sample_iso, a, cfg.eval.n_samples, sk)
    results["Sample-Iso"] = evaluate_method(predict_sample_iso, "Sample-Iso")

    # LUNO-Iso
    wu_luno_iso = IsotropicGaussian(mean=flat_params, sigma2=sigma2_luno_iso)
    def predict_luno_iso(a):
        mean = model(a)
        std = luno_predictive_std(model, wu_luno_iso, a)
        return mean, std
    results["LUNO-Iso"] = evaluate_method(predict_luno_iso, "LUNO-Iso")

    # Sample-LA
    wu_sample_la = LowRankLaplace(
        mean=flat_params, V=jnp.array(V), n_data=n_ggn, sigma2_prior=sigma2_sample_la
    )
    def predict_sample_la(a):
        nonlocal key
        key, sk = jax.random.split(key)
        return sample_pushforward_mean_std(model, wu_sample_la, a, cfg.eval.n_samples, sk)
    results["Sample-LA"] = evaluate_method(predict_sample_la, "Sample-LA")

    # LUNO-LA
    wu_luno_la = LowRankLaplace(
        mean=flat_params, V=jnp.array(V), n_data=n_ggn, sigma2_prior=sigma2_luno_la
    )
    def predict_luno_la(a):
        mean = model(a)
        std = luno_predictive_std(model, wu_luno_la, a)
        return mean, std
    results["LUNO-LA"] = evaluate_method(predict_luno_la, "LUNO-LA")

    # -------------------------------------------------------------------------
    # 7. Print summary table
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n{'='*60}")
        print(f"Results for {pde_name} (low-data regime, 25 training trajectories)")
        print(f"{'='*60}")
        print(f"{'Method':<25} {'RMSE':>12} {'χ²':>10} {'NLL':>10}")
        print("-" * 60)
        for method, metrics in results.items():
            print(f"{method:<25} {metrics['rmse']:>12.3e} {metrics['chi2']:>10.3f} {metrics['nll']:>10.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Low-data regime LUNO experiment")
    parser.add_argument(
        "--pde",
        type=str,
        default="burgers",
        choices=["burgers", "hyper_diffusion", "ks_conservative"],
        help="PDE to use for the experiment",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    DEFAULT_CONFIG.train.seed = args.seed
    results = run_low_data_experiment(args.pde, verbose=not args.quiet)
