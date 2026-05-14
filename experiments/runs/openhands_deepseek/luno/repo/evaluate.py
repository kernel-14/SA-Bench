"""Evaluation module for UQ methods on FNO predictions.

Implements the evaluation pipeline from Section 5 and Appendix D:
- Marginal metrics: RMSE, chi-squared, NLL
- Autoregressive rollout evaluation
- Calibration grid search
"""

from typing import Dict, Optional, List
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from uq import (
    compute_marginal_metrics,
    luno_predict_last_layer,
    luno_predict_isotropic,
    sample_predict,
    input_perturbation_predict,
    ensemble_predict,
    calibrate_sigma2,
    get_laplace_posterior,
    compute_low_rank_ggn,
)


def evaluate_method(
    model,
    X_test: jnp.ndarray,
    y_test: jnp.ndarray,
    method: str = "luno_la",
    sigma2: float = 1.0,
    n_samples: int = 200,
    n_modes=None,
    weight_cov: Optional[jnp.ndarray] = None,
    weight_mean_flat: Optional[jnp.ndarray] = None,
    ensemble_models: Optional[List] = None,
    key: jax.Array = None,
) -> Dict[str, float]:
    """Evaluate a single UQ method on test data.

    Args:
        model: Trained FNO model (or None for ensemble)
        X_test: Test inputs
        y_test: Test targets
        method: One of 'luno_iso', 'luno_la', 'sample_iso', 'sample_la',
                'ensemble', 'input_perturbations', 'deterministic'
        sigma2: Isotropic variance parameter
        n_samples: Number of samples for sampling methods
        n_modes: Fourier modes for LUNO last-layer
        weight_cov: Weight-space covariance (for Laplace methods)
        weight_mean_flat: Flattened weight mean
        ensemble_models: List of models for ensemble
        key: PRNG key

    Returns:
        Dict with 'rmse', 'chi2', 'nll' averaged over test set
    """
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
            pred = luno_predict_last_layer(
                model, x_i, model.last_layer_params(), weight_cov, n_modes
            )
        elif method == "sample_iso":
            pred = sample_predict(
                model, x_i, n_samples=n_samples,
                weight_distribution="isotropic", sigma2=sigma2,
                weight_mean_flat=weight_mean_flat, key=key,
            )
        elif method == "sample_la":
            pred = sample_predict(
                model, x_i, n_samples=n_samples,
                weight_distribution="laplace", weight_cov=weight_cov,
                weight_mean_flat=weight_mean_flat, key=key,
            )
        elif method == "ensemble":
            pred = ensemble_predict(ensemble_models, x_i)
        elif method == "input_perturbations":
            pred = input_perturbation_predict(
                model, x_i, n_samples=n_samples, noise_sigma=jnp.sqrt(sigma2), key=key,
            )
        elif method == "deterministic":
            pred_mean = model(x_i)
            pred = {"mean": pred_mean, "variance": jnp.ones_like(pred_mean) * sigma2}
        else:
            raise ValueError(f"Unknown method: {method}")

        metrics = compute_marginal_metrics(pred["mean"], pred["variance"], y_i)
        total_rmse += metrics["rmse"]
        total_chi2 += metrics["chi2"]
        total_nll += metrics["nll"]

    return {
        "rmse": total_rmse / n_test,
        "chi2": total_chi2 / n_test,
        "nll": total_nll / n_test,
    }


def evaluate_all_methods(
    model,
    X_test: jnp.ndarray,
    y_test: jnp.ndarray,
    methods: List[str],
    sigma2_values: Dict[str, float],
    n_modes=None,
    weight_cov: Optional[jnp.ndarray] = None,
    weight_mean_flat: Optional[jnp.ndarray] = None,
    ensemble_models: Optional[List] = None,
    key: jax.Array = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate multiple UQ methods on the same test set.

    Reproduces the tables in Section 5 of the paper.

    Returns:
        Nested dict: method -> metric -> value
    """
    results = {}
    for method in methods:
        sigma2 = sigma2_values.get(method, 1.0)
        results[method] = evaluate_method(
            model=model,
            X_test=X_test,
            y_test=y_test,
            method=method,
            sigma2=sigma2,
            n_samples=200,
            n_modes=n_modes,
            weight_cov=weight_cov,
            weight_mean_flat=weight_mean_flat,
            ensemble_models=ensemble_models,
            key=key,
        )
    return results


def autoregressive_rollout(
    model,
    initial_input: jnp.ndarray,
    n_rollout_steps: int,
    n_input_steps: int = 10,
    uq_method: Optional[str] = None,
    sigma2: float = 1.0,
    n_modes=None,
    weight_cov: Optional[jnp.ndarray] = None,
    weight_mean_flat: Optional[jnp.ndarray] = None,
    ensemble_models: Optional[List] = None,
    key: jax.Array = None,
) -> Dict[str, jnp.ndarray]:
    """Autoregressive rollout for time-dependent PDE prediction.

    The paper evaluates rollouts over 50 trajectories of the Pos-Neg-Flip dataset.
    Each step: predict next time step, append to input for next prediction.

    Args:
        model: Trained FNO
        initial_input: Initial input of shape (n_input_steps, N, channels)
        n_rollout_steps: Number of autoregressive steps
        n_input_steps: Number of input time steps to use
        uq_method: UQ method for uncertainty during rollout
        ... additional UQ parameters ...

    Returns:
        Dict with 'means' (n_steps, N, out_dim) and 'variances'
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    current_input = initial_input  # (n_input_steps, spatial..., channels)
    means = []
    variances = []

    for step in range(n_rollout_steps):
        # Predict
        if uq_method == "luno_iso":
            pred = luno_predict_isotropic(model, current_input[None], sigma2=sigma2, n_modes=n_modes)
        elif uq_method == "ensemble" and ensemble_models is not None:
            pred = ensemble_predict(ensemble_models, current_input[None])
        else:
            pred_mean = model(current_input[None])
            pred = {"mean": pred_mean[0], "variance": jnp.ones_like(pred_mean[0]) * sigma2}

        means.append(pred["mean"])
        variances.append(pred["variance"])

        # Shift input: remove oldest, append prediction
        new_frame = pred["mean"][0]  # first (and only) batch element
        if new_frame.ndim == 1:
            new_frame = new_frame[:, None]  # (N, 1)
        elif new_frame.ndim == 2:
            new_frame = new_frame[:, :, None]  # (H, W, 1)

        # Current input has shape (n_steps, spatial..., channels)
        # Shift and append
        current_input = jnp.concatenate([current_input[1:], new_frame], axis=0)

    means = jnp.stack(means, axis=0)  # (n_steps, spatial..., out_dim)
    variances = jnp.stack(variances, axis=0)

    return {"means": means, "variances": variances}


def rollout_metrics(
    rollout_means: jnp.ndarray,
    rollout_variances: jnp.ndarray,
    ground_truth: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """Compute per-step metrics for autoregressive rollout.

    Args:
        rollout_means: (n_steps, spatial...) predicted means
        rollout_variances: (n_steps, spatial...) predicted variances
        ground_truth: (n_steps, spatial...) true values

    Returns:
        Dict with 'rmse', 'chi2', 'nll' per step
    """
    n_steps = rollout_means.shape[0]
    rmse_list = []
    chi2_list = []
    nll_list = []

    for t in range(n_steps):
        metrics = compute_marginal_metrics(
            rollout_means[t], rollout_variances[t], ground_truth[t]
        )
        rmse_list.append(metrics["rmse"])
        chi2_list.append(metrics["chi2"])
        nll_list.append(metrics["nll"])

    return {
        "rmse": jnp.array(rmse_list),
        "chi2": jnp.array(chi2_list),
        "nll": jnp.array(nll_list),
    }
