"""
Out-of-distribution experiment: FNO trained on 2D advection-diffusion (Base),
evaluated on OOD variants (Flip, Pos, Pos-Neg, Pos-Neg-Flip).

Reproduces Table 2 and Table 6 from the LUNO paper.

Methods evaluated:
  - Input Perturbations
  - Deep Ensemble (10 members)
  - Sample-Iso
  - LUNO-Iso
  - Sample-LA
  - LUNO-LA

Usage:
  python experiments/run_ood.py
"""

import os
import sys
from typing import Dict, List

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
from data.advection_diffusion import OODDataset, generate_ood_dataset
from evaluate import compute_all_metrics
from luno.linearization import luno_predictive_std
from luno.sampling import sample_pushforward_mean_std
from luno.weight_uncertainty import (
    IsotropicGaussian,
    LowRankLaplace,
    compute_ggn_low_rank_streaming,
    get_flat_params,
)
from models.fno import FNO2d
from train import train, train_ensemble


OOD_VARIANTS = ["base", "flip", "pos", "pos_neg", "pos_neg_flip"]


def make_fno_2d(d_in: int, seed: int = 0) -> FNO2d:
    """Create a 2D FNO with paper hyperparameters."""
    cfg = DEFAULT_CONFIG.fno
    rngs = nnx.Rngs(params=jax.random.PRNGKey(seed))
    return FNO2d(
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


def run_ood_experiment(verbose: bool = True) -> Dict[str, Dict[str, Dict]]:
    """
    Run the full OOD experiment.

    Returns:
        results: dict mapping variant -> method -> metrics
    """
    cfg = DEFAULT_CONFIG
    key = jax.random.PRNGKey(cfg.train.seed)

    # -------------------------------------------------------------------------
    # 1. Generate Base training data
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n{'='*60}")
        print("OOD Experiment: 2D Advection-Diffusion")
        print(f"{'='*60}")
        print("Generating Base training data...")

    base_data = generate_ood_dataset(
        "base",
        n_train=cfg.ood.n_train,
        n_val=cfg.ood.n_val,
        n_test=cfg.ood.n_test,
        seed=cfg.train.seed,
        n_x=cfg.ood.spatial_res,
        n_y=cfg.ood.spatial_res,
        n_history=cfg.ood.n_history,
    )

    d_in = base_data["train"]["inputs"].shape[-1]  # n_history + 3

    train_loader = OODDataset(
        base_data["train"]["inputs"],
        base_data["train"]["outputs"],
        batch_size=1, shuffle=True,
    )
    val_loader = OODDataset(
        base_data["val"]["inputs"],
        base_data["val"]["outputs"],
        batch_size=1, shuffle=False,
    )

    if verbose:
        print(f"Train: {base_data['train']['inputs'].shape[0]} pairs, d_in={d_in}")

    # -------------------------------------------------------------------------
    # 2. Train main FNO on Base
    # -------------------------------------------------------------------------
    if verbose:
        print("\nTraining main FNO on Base dataset...")

    model = make_fno_2d(d_in, seed=cfg.train.seed)
    train(
        model,
        train_loader,
        val_loader=val_loader,
        n_epochs=cfg.train.ood_epochs,
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
        n_data=cfg.laplace.n_data_ggn_ood,
        key=jax.random.PRNGKey(1),
    )

    if verbose:
        print(f"GGN computed using {n_ggn} data points")

    # -------------------------------------------------------------------------
    # 4. Train ensemble on Base
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\nTraining ensemble ({cfg.eval.n_ensemble} members) on Base...")

    def model_factory(seed):
        return make_fno_2d(d_in, seed=seed)

    ensemble_models = train_ensemble(
        model_factory,
        train_loader,
        val_loader=val_loader,
        n_ensemble=cfg.eval.n_ensemble,
        n_epochs=cfg.train.ood_epochs,
        learning_rate=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        warmup_steps=cfg.train.warmup_steps,
        base_seed=100,
        verbose=verbose,
    )

    # -------------------------------------------------------------------------
    # 5. Calibrate all methods on Base validation set
    # -------------------------------------------------------------------------
    if verbose:
        print("\nCalibrating uncertainty methods on Base validation set...")

    sigma_perturb = calibrate_input_perturbation(
        model, val_loader, n_samples=cfg.eval.n_samples,
        n_val=cfg.eval.n_test_pairs, verbose=verbose,
    )
    sigma2_luno_iso = calibrate_luno_iso(
        model, val_loader, n_val=cfg.eval.n_test_pairs, verbose=verbose,
    )
    sigma2_sample_iso = calibrate_sample_iso(
        model, val_loader, n_samples=cfg.eval.n_samples,
        n_val=cfg.eval.n_test_pairs, verbose=verbose,
    )
    sigma2_luno_la = calibrate_luno_la(
        model, V, n_ggn, val_loader, n_val=cfg.eval.n_test_pairs, verbose=verbose,
    )
    sigma2_sample_la = calibrate_sample_la(
        model, V, n_ggn, val_loader, n_samples=cfg.eval.n_samples,
        n_val=cfg.eval.n_test_pairs, verbose=verbose,
    )

    # Build weight uncertainty objects
    wu_luno_iso = IsotropicGaussian(mean=flat_params, sigma2=sigma2_luno_iso)
    wu_sample_iso = IsotropicGaussian(mean=flat_params, sigma2=sigma2_sample_iso)
    wu_luno_la = LowRankLaplace(
        mean=flat_params, V=jnp.array(V), n_data=n_ggn, sigma2_prior=sigma2_luno_la
    )
    wu_sample_la = LowRankLaplace(
        mean=flat_params, V=jnp.array(V), n_data=n_ggn, sigma2_prior=sigma2_sample_la
    )

    # -------------------------------------------------------------------------
    # 6. Generate OOD test data and evaluate
    # -------------------------------------------------------------------------
    all_results = {}

    for variant in OOD_VARIANTS:
        if verbose:
            print(f"\n--- Evaluating on variant: {variant} ---")

        # Generate test data for this variant
        ood_data = generate_ood_dataset(
            variant,
            n_train=0,
            n_val=0,
            n_test=cfg.ood.n_test,
            seed=cfg.train.seed + 1,
            n_x=cfg.ood.spatial_res,
            n_y=cfg.ood.spatial_res,
            n_history=cfg.ood.n_history,
        )

        test_loader = OODDataset(
            ood_data["test"]["inputs"],
            ood_data["test"]["outputs"],
            batch_size=1, shuffle=False,
        )

        # Collect test pairs
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

        variant_results = {}
        eval_key = jax.random.PRNGKey(99)

        # Input Perturbations
        def predict_input_perturb(a):
            nonlocal eval_key
            eval_key, sk = jax.random.split(eval_key)
            return input_perturbation_mean_std(model, a, sigma_perturb, cfg.eval.n_samples, sk)
        variant_results["Input Perturbations"] = evaluate_method(predict_input_perturb, "Input Perturbations")

        # Ensemble
        def predict_ensemble(a):
            return ensemble_mean_std(ensemble_models, a)
        variant_results["Ensemble"] = evaluate_method(predict_ensemble, "Ensemble")

        # Sample-Iso
        def predict_sample_iso(a):
            nonlocal eval_key
            eval_key, sk = jax.random.split(eval_key)
            return sample_pushforward_mean_std(model, wu_sample_iso, a, cfg.eval.n_samples, sk)
        variant_results["Sample-Iso"] = evaluate_method(predict_sample_iso, "Sample-Iso")

        # LUNO-Iso
        def predict_luno_iso(a):
            mean = model(a)
            std = luno_predictive_std(model, wu_luno_iso, a)
            return mean, std
        variant_results["LUNO-Iso"] = evaluate_method(predict_luno_iso, "LUNO-Iso")

        # Sample-LA
        def predict_sample_la(a):
            nonlocal eval_key
            eval_key, sk = jax.random.split(eval_key)
            return sample_pushforward_mean_std(model, wu_sample_la, a, cfg.eval.n_samples, sk)
        variant_results["Sample-LA"] = evaluate_method(predict_sample_la, "Sample-LA")

        # LUNO-LA
        def predict_luno_la(a):
            mean = model(a)
            std = luno_predictive_std(model, wu_luno_la, a)
            return mean, std
        variant_results["LUNO-LA"] = evaluate_method(predict_luno_la, "LUNO-LA")

        all_results[variant] = variant_results

    # -------------------------------------------------------------------------
    # 7. Print summary table (NLL only, matching Table 2)
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n{'='*80}")
        print("Expected marginal NLL across OOD datasets (lower is better)")
        print(f"{'='*80}")
        methods = list(all_results["base"].keys())
        header = f"{'Method':<25}" + "".join(f"{v.upper():>15}" for v in ["base", "flip", "pos_neg_flip"])
        print(header)
        print("-" * 80)
        for method in methods:
            row = f"{method:<25}"
            for variant in ["base", "flip", "pos_neg_flip"]:
                nll = all_results[variant][method]["nll"]
                row += f"{nll:>15.3f}"
            print(row)

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OOD LUNO experiment")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    DEFAULT_CONFIG.train.seed = args.seed
    results = run_ood_experiment(verbose=not args.quiet)
