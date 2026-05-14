"""
Uncertainty quantification evaluation script.

Implements all UQ methods from Section 5 and Appendix D.3:
  1. Input Perturbations (Pathak et al., 2022)
  2. Deep Ensembles (Lakshminarayanan et al., 2017)
  3. Sample-Iso: Sample-based with isotropic Gaussian weight-space belief
  4. LUNO-Iso: Linearized with isotropic Gaussian weight-space belief
  5. Sample-LA: Sample-based with Laplace approximation
  6. LUNO-LA: Linearized with Laplace approximation (main contribution)

Evaluation metrics (Appendix D.4):
  - RMSE: Root Mean Squared Error
  - NLL: Marginal Negative Log-Likelihood
  - chi2: Chi-squared statistic
"""

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fno import FNO1d, FNO2d
from luno.luno import LUNO, LUNOPrediction
from luno.weight_space import IsotropicGaussian, LaplaceApproximation, compute_ggn_low_rank
from luno.metrics import evaluate_predictions, calibrate_sigma
from data.dataset import PDEDataset, batch_iterator


# ============================================================================
# Input Perturbations (Appendix D.3.1)
# ============================================================================

def predict_input_perturbations(
    model: nnx.Module,
    x: jnp.ndarray,
    sigma: float = 0.01,
    n_samples: int = 200,
    key: Optional[jax.Array] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Input perturbation uncertainty quantification.
    
    From Appendix D.3.1: Augment input with small random perturbations.
    Perturbations are sampled as epsilon_{x,t} ~ N(0, sigma^2).
    
    Args:
        model: Trained FNO model
        x: Input function, shape (n_x, in_ch) or (batch, n_x, in_ch)
        sigma: Perturbation standard deviation (calibrated on validation set)
        n_samples: Number of perturbed samples (200 in paper)
        key: JAX random key
    
    Returns:
        mean: Predictive mean
        std: Predictive standard deviation
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    batched = x.ndim > 2
    if not batched:
        x = x[None]

    batch_size = x.shape[0]

    # Generate perturbed inputs
    perturbations = jax.random.normal(key, (n_samples, *x.shape)) * sigma
    x_perturbed = x[None] + perturbations  # (n_samples, batch, n_x, in_ch)

    # Forward pass for each perturbed input
    all_preds = []
    for i in range(n_samples):
        pred = jax.vmap(model)(x_perturbed[i])  # (batch, n_x, out_ch)
        all_preds.append(pred)

    all_preds = jnp.stack(all_preds)  # (n_samples, batch, n_x, out_ch)

    mean = jnp.mean(all_preds, axis=0)  # (batch, n_x, out_ch)
    std = jnp.std(all_preds, axis=0)    # (batch, n_x, out_ch)

    if not batched:
        mean = mean[0]
        std = std[0]

    return mean, std


# ============================================================================
# Deep Ensembles (Appendix D.3.2)
# ============================================================================

def predict_ensemble(
    ensemble: List[nnx.Module],
    x: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Deep ensemble prediction.
    
    From Appendix D.3.2: Compute mean and std over ensemble predictions.
    
    Args:
        ensemble: List of trained FNO models
        x: Input function, shape (n_x, in_ch) or (batch, n_x, in_ch)
    
    Returns:
        mean: Ensemble mean prediction
        std: Ensemble standard deviation
    """
    batched = x.ndim > 2
    if not batched:
        x = x[None]

    all_preds = []
    for model in ensemble:
        pred = jax.vmap(model)(x)  # (batch, n_x, out_ch)
        all_preds.append(pred)

    all_preds = jnp.stack(all_preds)  # (n_members, batch, n_x, out_ch)

    mean = jnp.mean(all_preds, axis=0)
    std = jnp.std(all_preds, axis=0)

    if not batched:
        mean = mean[0]
        std = std[0]

    return mean, std


def get_ensemble_covariance(
    ensemble: List[nnx.Module],
    x: jnp.ndarray,
) -> jnp.ndarray:
    """Compute empirical covariance matrix from ensemble predictions.
    
    Used for Figure 3 in the paper (comparing ensemble vs LUNO-LA covariance structure).
    
    Args:
        ensemble: List of trained FNO models
        x: Input function, shape (n_x, in_ch)
    
    Returns:
        Covariance matrix, shape (n_x * out_ch, n_x * out_ch)
    """
    if x.ndim == 2:
        x = x[None]

    all_preds = []
    for model in ensemble:
        pred = model(x)[0]  # (n_x, out_ch)
        all_preds.append(pred.ravel())

    all_preds = jnp.stack(all_preds)  # (n_members, n_x * out_ch)
    mean = jnp.mean(all_preds, axis=0)
    centered = all_preds - mean[None]
    cov = centered.T @ centered / (len(ensemble) - 1)  # (n_x * out_ch, n_x * out_ch)

    return cov


# ============================================================================
# Sample-based methods (Appendix D.3.5)
# ============================================================================

def predict_sample_based(
    model: nnx.Module,
    x: jnp.ndarray,
    weight_belief,
    n_samples: int = 200,
    key: Optional[jax.Array] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample-based uncertainty quantification.
    
    From Appendix D.3.5: Draw 200 samples from weight-space belief,
    propagate through the (nonlinear) model, compute empirical mean and std.
    
    Args:
        model: Trained FNO model
        x: Input function, shape (n_x, in_ch) or (batch, n_x, in_ch)
        weight_belief: IsotropicGaussian or LaplaceApproximation
        n_samples: Number of weight samples (200 in paper)
        key: JAX random key
    
    Returns:
        mean: Predictive mean
        std: Predictive standard deviation
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    batched = x.ndim > 2
    if not batched:
        x = x[None]

    # Get model parameters
    graphdef, params = nnx.split(model)
    params_flat, unravel_fn = jax.flatten_util.ravel_pytree(params)
    n_params = params_flat.shape[0]

    # Sample weight perturbations
    delta_w_samples = weight_belief.sample_weight_perturbation(key, n_params, n_samples)

    # Forward pass for each weight sample
    all_preds = []
    for i in range(n_samples):
        perturbed_params_flat = params_flat + delta_w_samples[i]
        perturbed_params = unravel_fn(perturbed_params_flat)
        perturbed_model = nnx.merge(graphdef, perturbed_params)
        pred = jax.vmap(perturbed_model)(x)  # (batch, n_x, out_ch)
        all_preds.append(pred)

    all_preds = jnp.stack(all_preds)  # (n_samples, batch, n_x, out_ch)

    mean = jnp.mean(all_preds, axis=0)
    std = jnp.std(all_preds, axis=0)

    if not batched:
        mean = mean[0]
        std = std[0]

    return mean, std


# ============================================================================
# LUNO methods (Section 3, Appendix D.3.6)
# ============================================================================

def predict_luno(
    model: nnx.Module,
    x: jnp.ndarray,
    weight_belief,
    last_layer_only: bool = True,
    is_2d: bool = False,
) -> LUNOPrediction:
    """LUNO prediction.
    
    From Section 3.2 and Appendix D.3.6: Linearized uncertainty quantification.
    
    Args:
        model: Trained FNO model
        x: Input function, shape (n_x, in_ch) or (batch, n_x, in_ch)
        weight_belief: IsotropicGaussian or LaplaceApproximation
        last_layer_only: Whether to use last-layer LUNO (Appendix C.1)
        is_2d: Whether the model is a 2D FNO
    
    Returns:
        LUNOPrediction with mean, variance, std
    """
    luno = LUNO(model, weight_belief, last_layer_only=last_layer_only, is_2d=is_2d)
    return luno.predict_marginal_variance(x)


# ============================================================================
# Calibration (Appendix D.5)
# ============================================================================

def calibrate_uq_method(
    predict_fn,
    val_dataset: PDEDataset,
    method_name: str,
    n_val: int = 250,
    n_grid: int = 500,
) -> float:
    """Calibrate UQ method hyperparameters on validation set.
    
    From Appendix D.5: Calibrate sigma^2 using 250 validation pairs,
    minimizing marginal NLL via grid search over 500 log-spaced points.
    
    Args:
        predict_fn: Function that takes x and returns (mean, std_unnormalized)
        val_dataset: Validation dataset
        method_name: Name of the method (for logging)
        n_val: Number of validation samples to use
        n_grid: Number of grid points for sigma search
    
    Returns:
        Optimal sigma value
    """
    # Collect validation predictions
    all_means = []
    all_stds = []
    all_targets = []

    indices = np.arange(min(n_val, len(val_dataset)))
    x_val, y_val = val_dataset.get_batch(indices)

    x_val_jax = jnp.array(x_val, dtype=jnp.float32)
    mean, std = predict_fn(x_val_jax)

    all_means.append(np.array(mean))
    all_stds.append(np.array(std))
    all_targets.append(y_val)

    means = np.concatenate(all_means, axis=0)
    stds = np.concatenate(all_stds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Grid search for optimal sigma
    optimal_sigma = calibrate_sigma(
        y_true=jnp.array(targets),
        y_pred_mean=jnp.array(means),
        y_pred_std_unnormalized=jnp.array(stds),
        n_grid=n_grid,
    )

    print(f"{method_name}: optimal sigma = {optimal_sigma:.6f}")
    return optimal_sigma


# ============================================================================
# Full evaluation pipeline
# ============================================================================

def evaluate_all_methods(
    model: nnx.Module,
    ensemble: Optional[List[nnx.Module]],
    train_dataset: PDEDataset,
    val_dataset: PDEDataset,
    test_dataset: PDEDataset,
    ggn_rank: int = 500,
    n_samples: int = 200,
    seed: int = 42,
    is_2d: bool = False,
) -> Dict[str, Dict]:
    """Evaluate all UQ methods on test data.
    
    Implements the full evaluation pipeline from Section 5.
    
    Args:
        model: Trained FNO model
        ensemble: List of ensemble models (or None to skip ensemble)
        train_dataset: Training dataset (for GGN computation)
        val_dataset: Validation dataset (for calibration)
        test_dataset: Test dataset (for evaluation)
        ggn_rank: Rank for low-rank GGN approximation (500 in paper)
        n_samples: Number of samples for sample-based methods (200 in paper)
        seed: Random seed
    
    Returns:
        Dictionary mapping method names to metric dictionaries
    """
    key = jax.random.PRNGKey(seed)
    results = {}

    # Get test data
    n_test = min(250, len(test_dataset))
    test_indices = np.arange(n_test)
    x_test, y_test = test_dataset.get_batch(test_indices)
    x_test_jax = jnp.array(x_test, dtype=jnp.float32)
    y_test_jax = jnp.array(y_test, dtype=jnp.float32)

    # ---- Compute GGN for Laplace approximation ----
    print("Computing GGN low-rank approximation...")
    graphdef, params = nnx.split(model)
    params_flat, unravel_fn = jax.flatten_util.ravel_pytree(params)

    # Use last-layer parameters only for efficiency
    last_block = model.fourier_blocks[-1]
    last_block_graphdef, last_block_params = nnx.split(last_block)
    last_params_flat, last_unravel_fn = jax.flatten_util.ravel_pytree(last_block_params)

    # Get training data for GGN
    n_ggn = min(1000, len(train_dataset))
    ggn_indices = np.arange(n_ggn)
    x_ggn, y_ggn = train_dataset.get_batch(ggn_indices)
    x_ggn_jax = jnp.array(x_ggn, dtype=jnp.float32)

    def last_layer_model_fn(last_params_flat, x_batch):
        """Model function using only last-layer parameters."""
        # Get v_prev for the batch
        v_prev = jax.vmap(model.get_last_layer_input)(x_batch)

        last_block_params = last_unravel_fn(last_params_flat)
        last_block = nnx.merge(last_block_graphdef, last_block_params)

        def forward_single(v_prev_single):
            z = last_block(v_prev_single[None])[0]
            h = model.activation(model.proj1(z))
            return model.proj2(h)

        return jax.vmap(forward_single)(v_prev)

    try:
        eigenvectors, eigenvalues = compute_ggn_low_rank(
            model_fn=last_layer_model_fn,
            params_flat=last_params_flat,
            data_inputs=x_ggn_jax,
            data_targets=jnp.array(y_ggn, dtype=jnp.float32),
            rank=min(ggn_rank, last_params_flat.shape[0] // 2),
            batch_size=32,
        )
        la_available = True
    except Exception as e:
        print(f"GGN computation failed: {e}. Skipping LA methods.")
        la_available = False

    # ---- Input Perturbations ----
    print("\nEvaluating Input Perturbations...")
    key, subkey = jax.random.split(key)

    def predict_perturb_uncalibrated(x, sigma=0.01):
        return predict_input_perturbations(model, x, sigma=sigma, n_samples=n_samples, key=subkey)

    # Calibrate sigma
    sigma_perturb = calibrate_uq_method(
        lambda x: predict_perturb_uncalibrated(x),
        val_dataset,
        "Input Perturbations",
    )

    mean_perturb, std_perturb = predict_input_perturbations(
        model, x_test_jax, sigma=sigma_perturb, n_samples=n_samples, key=subkey
    )
    results['Input Perturbations'] = evaluate_predictions(y_test_jax, mean_perturb, std_perturb)

    # ---- Deep Ensembles ----
    if ensemble is not None:
        print("\nEvaluating Deep Ensembles...")
        mean_ens, std_ens = predict_ensemble(ensemble, x_test_jax)
        results['Ensemble'] = evaluate_predictions(y_test_jax, mean_ens, std_ens)

    # ---- Isotropic Gaussian methods ----
    print("\nEvaluating Isotropic Gaussian methods...")
    iso_belief = IsotropicGaussian(sigma2=1.0)

    # Sample-Iso
    key, subkey = jax.random.split(key)

    def predict_sample_iso_uncalibrated(x):
        mean, std = predict_sample_based(model, x, iso_belief, n_samples=n_samples, key=subkey)
        return mean, std

    sigma_sample_iso = calibrate_uq_method(
        predict_sample_iso_uncalibrated, val_dataset, "Sample-Iso"
    )
    iso_belief_calibrated = IsotropicGaussian(sigma2=sigma_sample_iso**2)
    mean_sample_iso, std_sample_iso = predict_sample_based(
        model, x_test_jax, iso_belief_calibrated, n_samples=n_samples, key=subkey
    )
    results['Sample-Iso'] = evaluate_predictions(y_test_jax, mean_sample_iso, std_sample_iso)

    # LUNO-Iso
    def predict_luno_iso_uncalibrated(x):
        pred = predict_luno(model, x, iso_belief, is_2d=is_2d)
        return pred.mean, pred.std

    sigma_luno_iso = calibrate_uq_method(
        predict_luno_iso_uncalibrated, val_dataset, "LUNO-Iso"
    )
    iso_belief_luno = IsotropicGaussian(sigma2=sigma_luno_iso**2)
    luno_iso_pred = predict_luno(model, x_test_jax, iso_belief_luno, is_2d=is_2d)
    results['LUNO-Iso'] = evaluate_predictions(y_test_jax, luno_iso_pred.mean, luno_iso_pred.std)

    # ---- Laplace Approximation methods ----
    if la_available:
        print("\nEvaluating Laplace Approximation methods...")
        la_belief = LaplaceApproximation(
            eigenvectors=eigenvectors,
            eigenvalues=eigenvalues,
            n_data=n_ggn,
            prior_precision=1.0,
            sigma2_scale=1.0,
        )

        # Sample-LA
        key, subkey = jax.random.split(key)

        def predict_sample_la_uncalibrated(x):
            mean, std = predict_sample_based(model, x, la_belief, n_samples=n_samples, key=subkey)
            return mean, std

        sigma_sample_la = calibrate_uq_method(
            predict_sample_la_uncalibrated, val_dataset, "Sample-LA"
        )
        la_belief_sample = LaplaceApproximation(
            eigenvectors=eigenvectors,
            eigenvalues=eigenvalues,
            n_data=n_ggn,
            prior_precision=1.0,
            sigma2_scale=sigma_sample_la**2,
        )
        mean_sample_la, std_sample_la = predict_sample_based(
            model, x_test_jax, la_belief_sample, n_samples=n_samples, key=subkey
        )
        results['Sample-LA'] = evaluate_predictions(y_test_jax, mean_sample_la, std_sample_la)

        # LUNO-LA
        def predict_luno_la_uncalibrated(x):
            pred = predict_luno(model, x, la_belief, is_2d=is_2d)
            return pred.mean, pred.std

        sigma_luno_la = calibrate_uq_method(
            predict_luno_la_uncalibrated, val_dataset, "LUNO-LA"
        )
        la_belief_luno = LaplaceApproximation(
            eigenvectors=eigenvectors,
            eigenvalues=eigenvalues,
            n_data=n_ggn,
            prior_precision=1.0,
            sigma2_scale=sigma_luno_la**2,
        )
        luno_la_pred = predict_luno(model, x_test_jax, la_belief_luno, is_2d=is_2d)
        results['LUNO-LA'] = evaluate_predictions(y_test_jax, luno_la_pred.mean, luno_la_pred.std)

    return results


def print_results_table(results: Dict[str, Dict]):
    """Print results in a table format matching the paper's tables."""
    print("\n" + "="*70)
    print(f"{'Method':<25} {'RMSE':>12} {'chi2':>10} {'NLL':>12}")
    print("="*70)

    for method, metrics in results.items():
        rmse = metrics.get('rmse', float('nan'))
        chi2 = metrics.get('chi2', float('nan'))
        nll = metrics.get('nll', float('nan'))
        print(f"{method:<25} {rmse:>12.4e} {chi2:>10.3f} {nll:>12.4f}")

    print("="*70)
