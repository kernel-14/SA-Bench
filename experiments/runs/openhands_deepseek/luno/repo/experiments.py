"""Main experiment runner to reproduce LUNO paper results.

Section 5 experiments:
1. Low-data regime (Burgers', Hyper Diffusion, Kuramoto-Sivashinsky)
2. Out-of-distribution (Advection-Diffusion variants)

Method comparison:
- Input Perturbations
- Deep Ensemble (10 members)
- Sample-Iso / Sample-LA
- LUNO-Iso / LUNO-LA
"""

import os
import argparse
import json
from typing import Dict, List
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from config import (
    ExperimentConfig,
    get_low_data_burgers_config,
    get_low_data_hyper_diffusion_config,
    get_low_data_kuramoto_sivashinsky_config,
    get_ood_advection_config,
)
from data import (
    generate_dataset,
    prepare_fno_data,
    split_train_val_test,
)
from train import train, train_ensemble
from uq import (
    compute_marginal_metrics,
    calibrate_sigma2,
    get_laplace_posterior,
    luno_predict_last_layer,
    luno_predict_isotropic,
    sample_predict,
    input_perturbation_predict,
    ensemble_predict,
)


def run_low_data_experiment(config: ExperimentConfig, output_dir: str):
    """Run low-data regime experiment on a 1D PDE.

    Reproduces Tables 1, 4, 5 and Figure 2 from the paper.
    """
    os.makedirs(output_dir, exist_ok=True)
    pde = config.data.pde_name

    print(f"\n{'='*60}")
    print(f"Low-Data Experiment: {pde}")
    print(f"{'='*60}")

    # Generate data
    print("Generating dataset...")
    trajectories, _ = generate_dataset(
        pde_name=pde,
        n_trajectories=config.data.n_train_trajectories + config.data.n_val_trajectories + config.data.n_test_trajectories,
        n_steps=config.data.temporal_resolution,
        spatial_resolution=config.data.spatial_resolution,
        domain_size=config.data.domain_size,
        key=jax.random.PRNGKey(config.seed),
    )

    X, y = prepare_fno_data(
        trajectories,
        n_input_steps=config.data.n_input_steps,
        n_output_steps=config.data.n_output_steps,
        spatial_dim=config.data.spatial_dim,
    )

    n_pairs = X.shape[0]
    n_pairs_per_traj = config.data.temporal_resolution - config.data.n_input_steps - config.data.n_output_steps + 1
    n_train_pairs = config.data.n_train_trajectories * n_pairs_per_traj
    n_val_pairs = config.data.n_val_trajectories * n_pairs_per_traj
    n_test_pairs = min(config.data.n_test_trajectories * n_pairs_per_traj, n_pairs - n_train_pairs - n_val_pairs)

    X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(
        X, y, n_train_pairs, n_val_pairs, n_test_pairs,
        key=jax.random.PRNGKey(config.seed + 1),
    )

    print(f"Train pairs: {X_train.shape[0]}, Val pairs: {X_val.shape[0]}, Test pairs: {X_test.shape[0]}")

    # Train single FNO
    print("\nTraining FNO...")
    model = train(config, X_train, y_train, X_val, y_val)

    # Get MAP parameters
    params = nnx.state(model)
    flat_params, _ = jax.flatten_util.ravel_pytree(params)

    # Train ensemble (10 members)
    print("\nTraining ensemble (10 members)...")
    ensemble_models = train_ensemble(config, X_train, y_train, X_val, y_val, n_members=10)

    # Compute Laplace posterior (low-rank GGN)
    print("\nComputing Laplace approximation (low-rank GGN)...")
    weight_mean, weight_cov = get_laplace_posterior(
        model, X_train, y_train,
        rank=500,
        prior_precision=1.0,
        n_data=-1,  # all data for low-data regime
        last_layer_only=True,
        key=jax.random.PRNGKey(config.seed + 2),
    )

    # Calibration: grid search over sigma2 for each method
    print("\nCalibrating hyperparameters...")
    sigma2_grid = jnp.logspace(-6, 6, 500)

    # Calibrate LUNO-Iso
    sigma2_luno_iso, _ = calibrate_sigma2(
        model, X_val, y_val, "luno_iso", sigma2_grid,
        n_modes=config.fno.n_modes[0] if isinstance(config.fno.n_modes, tuple) else config.fno.n_modes,
        key=jax.random.PRNGKey(config.seed + 3),
    )
    print(f"LUNO-Iso sigma2: {sigma2_luno_iso:.6f}")

    # Calibrate Sample-Iso
    sigma2_sample_iso, _ = calibrate_sigma2(
        model, X_val, y_val, "sample_iso", sigma2_grid,
        key=jax.random.PRNGKey(config.seed + 4),
    )
    print(f"Sample-Iso sigma2: {sigma2_sample_iso:.6f}")

    # Calibrate Input Perturbations
    sigma2_input_pert, _ = calibrate_sigma2(
        model, X_val, y_val, "input_perturbations", sigma2_grid,
        key=jax.random.PRNGKey(config.seed + 5),
    )
    print(f"Input Perturbations sigma2: {sigma2_input_pert:.6f}")

    # Evaluate all methods
    print("\nEvaluating uncertainty quantification methods...")
    methods = [
        "input_perturbations",
        "ensemble",
        "sample_iso",
        "luno_iso",
        "sample_la",
        "luno_la",
    ]
    sigma2_values = {
        "input_perturbations": sigma2_input_pert,
        "sample_iso": sigma2_sample_iso,
        "luno_iso": sigma2_luno_iso,
        "sample_la": 1.0,  # Laplace uses posterior covariance, not sigma2
        "luno_la": 1.0,
    }

    results = {}
    for method in methods:
        sig = sigma2_values.get(method, 1.0)
        method_results = evaluate_single_method(
            model, X_test, y_test, method, sig,
            n_modes=config.fno.n_modes[0] if isinstance(config.fno.n_modes, tuple) else config.fno.n_modes,
            weight_cov=weight_cov if "la" in method else None,
            weight_mean_flat=flat_params,
            ensemble_models=ensemble_models,
            key=jax.random.PRNGKey(config.seed + 6),
        )
        results[method] = method_results
        print(f"  {method}: RMSE={method_results['rmse']:.6f}, "
              f"χ²={method_results['chi2']:.3f}, NLL={method_results['nll']:.4f}")

    # Save results
    results_path = os.path.join(output_dir, f"results_{pde}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


def evaluate_single_method(
    model, X_test, y_test, method, sigma2, n_modes=None,
    weight_cov=None, weight_mean_flat=None, ensemble_models=None, key=None,
):
    """Evaluate one UQ method on the test set, sample by sample."""
    if key is None:
        key = jax.random.PRNGKey(0)

    total_rmse = 0.0
    total_chi2 = 0.0
    total_nll = 0.0
    n_test = X_test.shape[0]

    for i in range(n_test):
        x_i = X_test[i:i+1] if X_test.ndim >= 3 else X_test[i]
        y_i = y_test[i:i+1] if y_test.ndim >= 3 else y_test[i]

        if method == "luno_iso":
            pred = luno_predict_isotropic(model, x_i, sigma2=sigma2, n_modes=n_modes)
        elif method == "luno_la":
            n_modes_val = n_modes if n_modes is not None else 12
            pred = luno_predict_last_layer(
                model, x_i, model.last_layer_params(), weight_cov, n_modes_val,
            )
        elif method == "sample_iso":
            pred = sample_predict(
                model, x_i, n_samples=200, weight_distribution="isotropic",
                sigma2=sigma2, weight_mean_flat=weight_mean_flat, key=key,
            )
        elif method == "sample_la":
            pred = sample_predict(
                model, x_i, n_samples=200, weight_distribution="laplace",
                weight_cov=weight_cov, weight_mean_flat=weight_mean_flat, key=key,
            )
        elif method == "ensemble":
            pred = ensemble_predict(ensemble_models, x_i)
        elif method == "input_perturbations":
            pred = input_perturbation_predict(
                model, x_i, n_samples=200, noise_sigma=jnp.sqrt(sigma2), key=key,
            )
        else:
            pred_mean = model(x_i)
            pred = {"mean": pred_mean, "variance": jnp.ones_like(pred_mean) * sigma2}

        metrics = compute_marginal_metrics(pred["mean"], pred["variance"], y_i)
        total_rmse += metrics["rmse"]
        total_chi2 += metrics["chi2"]
        total_nll += metrics["nll"]

    return {
        "rmse": total_rmse / n_test,
        "chi2": total_chi2 / n_test,
        "nll": total_nll / n_test,
    }


def run_ood_experiment(config: ExperimentConfig, output_dir: str):
    """Run out-of-distribution experiment on 2D Advection-Diffusion equation.

    Reproduces Tables 2, 6-11 and Figures 3, 4 from the paper.

    Evaluates on multiple OOD variants: Base, Flip, Pos, Pos-Neg, Pos-Neg-Flip
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"OOD Experiment: Advection-Diffusion")
    print(f"{'='*60}")

    # Generate training data (Base variant only)
    print("Generating training dataset (Base variant)...")
    train_trajectories, train_aux = generate_dataset(
        pde_name=config.data.pde_name,
        n_trajectories=config.data.n_train_trajectories,
        n_steps=config.data.temporal_resolution,
        spatial_resolution=config.data.spatial_resolution,
        domain_size=config.data.domain_size,
        variant="base",
        key=jax.random.PRNGKey(config.seed),
    )

    X_train_full, y_train_full = prepare_fno_data(
        train_trajectories,
        n_input_steps=config.data.n_input_steps,
        n_output_steps=config.data.n_output_steps,
        spatial_dim=config.data.spatial_dim,
        aux_fields=train_aux,
    )

    # Generate validation data (Base variant)
    print("Generating validation dataset...")
    val_trajectories, val_aux = generate_dataset(
        pde_name=config.data.pde_name,
        n_trajectories=config.data.n_val_trajectories,
        n_steps=config.data.temporal_resolution,
        spatial_resolution=config.data.spatial_resolution,
        domain_size=config.data.domain_size,
        variant="base",
        key=jax.random.PRNGKey(config.seed + 100),
    )

    X_val_full, y_val_full = prepare_fno_data(
        val_trajectories,
        n_input_steps=config.data.n_input_steps,
        n_output_steps=config.data.n_output_steps,
        spatial_dim=config.data.spatial_dim,
        aux_fields=val_aux,
    )

    n_val_pairs = min(250, X_val_full.shape[0])
    X_val, y_val = X_val_full[:n_val_pairs], y_val_full[:n_val_pairs]

    print(f"Train pairs: {X_train_full.shape[0]}, Val pairs: {X_val.shape[0]}")

    # Train single FNO
    print("\nTraining FNO for 1000 epochs...")
    model = train(config, X_train_full, y_train_full, X_val, y_val)
    params = nnx.state(model)
    flat_params, _ = jax.flatten_util.ravel_pytree(params)

    # Train ensemble
    print("\nTraining ensemble (10 members)...")
    ensemble_models = train_ensemble(config, X_train_full, y_train_full, X_val, y_val, n_members=10)

    # Compute Laplace posterior on subset of data
    print("\nComputing Laplace approximation (minibatch of 1000)...")
    n_ggn = min(1000, X_train_full.shape[0])
    weight_mean, weight_cov = get_laplace_posterior(
        model, X_train_full, y_train_full,
        rank=500,
        prior_precision=1.0,
        n_data=n_ggn,
        last_layer_only=True,
        key=jax.random.PRNGKey(config.seed + 2),
    )

    # Evaluate on each OOD variant
    variants = ["base", "flip", "pos", "pos_neg", "pos_neg_flip"]

    all_results = {}
    for variant in variants:
        print(f"\n--- Evaluating on {variant} variant ---")

        test_trajectories, test_aux = generate_dataset(
            pde_name=config.data.pde_name,
            n_trajectories=config.data.n_test_trajectories,
            n_steps=config.data.temporal_resolution,
            spatial_resolution=config.data.spatial_resolution,
            domain_size=config.data.domain_size,
            variant=variant,
            key=jax.random.PRNGKey(config.seed + 200 + variants.index(variant)),
        )

        X_test_full, y_test_full = prepare_fno_data(
            test_trajectories,
            n_input_steps=config.data.n_input_steps,
            n_output_steps=config.data.n_output_steps,
            spatial_dim=config.data.spatial_dim,
            aux_fields=test_aux,
        )

        n_test_pairs = min(250, X_test_full.shape[0])
        X_test, y_test = X_test_full[:n_test_pairs], y_test_full[:n_test_pairs]

        # Calibrate on val set (Base)
        print("  Calibrating...")
        sigma2_grid = jnp.logspace(-6, 6, 500)

        sigma2_values = {}
        for calib_method in ["luno_iso", "sample_iso", "input_perturbations"]:
            sigma2_val, _ = calibrate_sigma2(
                model, X_val, y_val, calib_method, sigma2_grid,
                n_modes=config.fno.n_modes if isinstance(config.fno.n_modes, int) else config.fno.n_modes[0],
                key=jax.random.PRNGKey(config.seed + 3),
            )
            sigma2_values[calib_method] = sigma2_val

        # Evaluate
        methods_list = [
            "input_perturbations", "ensemble", "sample_iso",
            "luno_iso", "sample_la", "luno_la",
        ]

        variant_results = {}
        for method in methods_list:
            sig = sigma2_values.get(method, 1.0)
            method_results = evaluate_single_method(
                model, X_test, y_test, method, sig,
                n_modes=config.fno.n_modes if isinstance(config.fno.n_modes, int) else config.fno.n_modes[0],
                weight_cov=weight_cov if "la" in method else None,
                weight_mean_flat=flat_params,
                ensemble_models=ensemble_models,
                key=jax.random.PRNGKey(config.seed + 6),
            )
            variant_results[method] = method_results
            print(f"    {method}: NLL={method_results['nll']:.4f}")

        all_results[variant] = variant_results

    # Save results
    results_path = os.path.join(output_dir, "ood_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Reproduce LUNO paper experiments")
    parser.add_argument(
        "--experiment",
        type=str,
        default="low_data_burgers",
        choices=[
            "low_data_burgers",
            "low_data_hyper_diffusion",
            "low_data_kuramoto_sivashinsky",
            "ood_advection",
        ],
        help="Which experiment to run",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    # Select config
    config_map = {
        "low_data_burgers": get_low_data_burgers_config,
        "low_data_hyper_diffusion": get_low_data_hyper_diffusion_config,
        "low_data_kuramoto_sivashinsky": get_low_data_kuramoto_sivashinsky_config,
        "ood_advection": get_ood_advection_config,
    }

    config_fn = config_map[args.experiment]
    config = config_fn()
    config.seed = args.seed
    config.output_dir = args.output_dir

    if args.experiment.startswith("low_data"):
        run_low_data_experiment(config, args.output_dir)
    elif args.experiment == "ood_advection":
        run_ood_experiment(config, args.output_dir)


if __name__ == "__main__":
    main()
