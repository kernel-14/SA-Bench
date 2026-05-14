"""
Calibration of uncertainty hyperparameters.

All UQ methods have a scalar hyperparameter σ² that controls the scale of
the predictive uncertainty. This is calibrated on the validation set by
minimizing the marginal NLL.

Calibration procedure (Appendix D.5):
  - Grid search over 500 log-spaced values of σ²
  - Range: [σ²_min, σ²_max] (default: [1e-6, 1e2])
  - Metric: expected marginal NLL on validation set
  - Each method's hyperparameters are calibrated separately

Methods and their σ² parameters:
  - Input Perturbations: σ = perturbation std
  - LUNO-Iso / Sample-Iso: σ² = weight-space variance
  - LUNO-LA / Sample-LA: σ²_prior = prior variance in Laplace approximation
"""

from typing import Callable, Dict, Optional, Tuple

import jax.numpy as jnp
import numpy as np

from evaluate import compute_marginal_nll


def calibrate_sigma2(
    predict_fn_factory: Callable,
    val_loader,
    sigma2_min: float = 1e-6,
    sigma2_max: float = 1e2,
    n_grid: int = 500,
    n_val: int = 250,
    verbose: bool = True,
) -> Tuple[float, float]:
    """
    Calibrate σ² by minimizing marginal NLL on validation set.

    Args:
        predict_fn_factory: callable(sigma2) -> predict_fn
            where predict_fn(a) -> (mean, std)
        val_loader: validation data loader
        sigma2_min: minimum σ² value
        sigma2_max: maximum σ² value
        n_grid: number of grid points
        n_val: number of validation pairs to use
        verbose: whether to print progress
    Returns:
        best_sigma2: optimal σ²
        best_nll: NLL at optimal σ²
    """
    sigma2_grid = np.logspace(
        np.log10(sigma2_min),
        np.log10(sigma2_max),
        n_grid,
    )

    best_sigma2 = sigma2_grid[0]
    best_nll = float("inf")

    # Collect validation data once
    val_inputs = []
    val_targets = []
    n_collected = 0

    for a_batch, u_batch in val_loader:
        for i in range(a_batch.shape[0]):
            if n_collected >= n_val:
                break
            val_inputs.append(np.array(a_batch[i:i + 1]))
            val_targets.append(np.array(u_batch[i:i + 1]))
            n_collected += 1
        if n_collected >= n_val:
            break

    if verbose:
        print(f"Calibrating over {n_grid} σ² values on {n_collected} validation pairs...")

    for j, sigma2 in enumerate(sigma2_grid):
        predict_fn = predict_fn_factory(sigma2)

        all_means = []
        all_stds = []

        for a_i, u_i in zip(val_inputs, val_targets):
            mean_i, std_i = predict_fn(jnp.array(a_i))
            all_means.append(np.array(mean_i))
            all_stds.append(np.array(std_i))

        mean_pred = np.concatenate(all_means, axis=0)
        std_pred = np.concatenate(all_stds, axis=0)
        target = np.concatenate(val_targets, axis=0)

        nll = compute_marginal_nll(mean_pred, std_pred, target)

        if nll < best_nll:
            best_nll = nll
            best_sigma2 = sigma2

        if verbose and (j + 1) % 50 == 0:
            print(f"  [{j + 1}/{n_grid}] σ²={sigma2:.2e}, NLL={nll:.4f} (best: {best_sigma2:.2e}, {best_nll:.4f})")

    if verbose:
        print(f"Best σ² = {best_sigma2:.4e}, NLL = {best_nll:.4f}")

    return best_sigma2, best_nll


def calibrate_input_perturbation(
    model,
    val_loader,
    n_samples: int = 200,
    sigma2_min: float = 1e-8,
    sigma2_max: float = 1.0,
    n_grid: int = 500,
    n_val: int = 250,
    key=None,
    verbose: bool = True,
) -> float:
    """
    Calibrate input perturbation σ.

    Args:
        model: trained FNO
        val_loader: validation data loader
        n_samples: number of perturbed inputs per prediction
        sigma2_min, sigma2_max: search range for σ²
        n_grid: grid size
        n_val: number of validation pairs
        key: JAX random key
        verbose: print progress
    Returns:
        best_sigma: optimal perturbation std
    """
    import jax
    from baselines.input_perturbations import input_perturbation_mean_std

    if key is None:
        key = jax.random.PRNGKey(42)

    def predict_fn_factory(sigma2):
        sigma = float(np.sqrt(sigma2))

        def predict_fn(a):
            nonlocal key
            key, subkey = jax.random.split(key)
            return input_perturbation_mean_std(model, a, sigma, n_samples, subkey)

        return predict_fn

    best_sigma2, _ = calibrate_sigma2(
        predict_fn_factory, val_loader,
        sigma2_min=sigma2_min, sigma2_max=sigma2_max,
        n_grid=n_grid, n_val=n_val, verbose=verbose,
    )
    return float(np.sqrt(best_sigma2))


def calibrate_luno_iso(
    model,
    val_loader,
    sigma2_min: float = 1e-6,
    sigma2_max: float = 1e2,
    n_grid: int = 500,
    n_val: int = 250,
    verbose: bool = True,
) -> float:
    """
    Calibrate σ² for LUNO-Iso (isotropic Gaussian weight uncertainty).

    Args:
        model: trained FNO
        val_loader: validation data loader
        sigma2_min, sigma2_max: search range
        n_grid: grid size
        n_val: number of validation pairs
        verbose: print progress
    Returns:
        best_sigma2: optimal weight-space variance
    """
    from luno.linearization import luno_predictive_std
    from luno.weight_uncertainty import IsotropicGaussian, get_flat_params

    flat_params = get_flat_params(model)

    def predict_fn_factory(sigma2):
        wu = IsotropicGaussian(mean=flat_params, sigma2=float(sigma2))

        def predict_fn(a):
            mean = model(a)
            std = luno_predictive_std(model, wu, a)
            return mean, std

        return predict_fn

    best_sigma2, _ = calibrate_sigma2(
        predict_fn_factory, val_loader,
        sigma2_min=sigma2_min, sigma2_max=sigma2_max,
        n_grid=n_grid, n_val=n_val, verbose=verbose,
    )
    return best_sigma2


def calibrate_luno_la(
    model,
    V: np.ndarray,
    n_data: int,
    val_loader,
    sigma2_min: float = 1e-6,
    sigma2_max: float = 1e2,
    n_grid: int = 500,
    n_val: int = 250,
    verbose: bool = True,
) -> float:
    """
    Calibrate σ²_prior for LUNO-LA (low-rank Laplace approximation).

    The posterior covariance is Σ = (n * VV^T + σ^{-2} I)^{-1}.
    We calibrate σ² (the prior variance) on the validation set.

    Args:
        model: trained FNO
        V: (p, rank) low-rank GGN factor
        n_data: number of data points used for GGN
        val_loader: validation data loader
        sigma2_min, sigma2_max: search range for prior variance
        n_grid: grid size
        n_val: number of validation pairs
        verbose: print progress
    Returns:
        best_sigma2_prior: optimal prior variance
    """
    import jax
    from luno.linearization import luno_predictive_std
    from luno.weight_uncertainty import LowRankLaplace, get_flat_params

    flat_params = get_flat_params(model)
    V_jax = jnp.array(V)

    def predict_fn_factory(sigma2):
        wu = LowRankLaplace(
            mean=flat_params,
            V=V_jax,
            n_data=n_data,
            sigma2_prior=float(sigma2),
        )

        def predict_fn(a):
            mean = model(a)
            std = luno_predictive_std(model, wu, a)
            return mean, std

        return predict_fn

    best_sigma2, _ = calibrate_sigma2(
        predict_fn_factory, val_loader,
        sigma2_min=sigma2_min, sigma2_max=sigma2_max,
        n_grid=n_grid, n_val=n_val, verbose=verbose,
    )
    return best_sigma2


def calibrate_sample_iso(
    model,
    val_loader,
    n_samples: int = 200,
    sigma2_min: float = 1e-6,
    sigma2_max: float = 1e2,
    n_grid: int = 500,
    n_val: int = 250,
    verbose: bool = True,
) -> float:
    """
    Calibrate σ² for Sample-Iso.

    Args:
        model: trained FNO
        val_loader: validation data loader
        n_samples: number of weight samples
        sigma2_min, sigma2_max: search range
        n_grid: grid size
        n_val: number of validation pairs
        verbose: print progress
    Returns:
        best_sigma2
    """
    import jax
    from luno.sampling import sample_pushforward_mean_std
    from luno.weight_uncertainty import IsotropicGaussian, get_flat_params

    flat_params = get_flat_params(model)
    key = jax.random.PRNGKey(42)

    def predict_fn_factory(sigma2):
        wu = IsotropicGaussian(mean=flat_params, sigma2=float(sigma2))

        def predict_fn(a):
            nonlocal key
            key, subkey = jax.random.split(key)
            return sample_pushforward_mean_std(model, wu, a, n_samples, subkey)

        return predict_fn

    best_sigma2, _ = calibrate_sigma2(
        predict_fn_factory, val_loader,
        sigma2_min=sigma2_min, sigma2_max=sigma2_max,
        n_grid=n_grid, n_val=n_val, verbose=verbose,
    )
    return best_sigma2


def calibrate_sample_la(
    model,
    V: np.ndarray,
    n_data: int,
    val_loader,
    n_samples: int = 200,
    sigma2_min: float = 1e-6,
    sigma2_max: float = 1e2,
    n_grid: int = 500,
    n_val: int = 250,
    verbose: bool = True,
) -> float:
    """
    Calibrate σ²_prior for Sample-LA.

    Args:
        model: trained FNO
        V: (p, rank) low-rank GGN factor
        n_data: number of data points used for GGN
        val_loader: validation data loader
        n_samples: number of weight samples
        sigma2_min, sigma2_max: search range
        n_grid: grid size
        n_val: number of validation pairs
        verbose: print progress
    Returns:
        best_sigma2_prior
    """
    import jax
    from luno.sampling import sample_pushforward_mean_std
    from luno.weight_uncertainty import LowRankLaplace, get_flat_params

    flat_params = get_flat_params(model)
    V_jax = jnp.array(V)
    key = jax.random.PRNGKey(42)

    def predict_fn_factory(sigma2):
        wu = LowRankLaplace(
            mean=flat_params,
            V=V_jax,
            n_data=n_data,
            sigma2_prior=float(sigma2),
        )

        def predict_fn(a):
            nonlocal key
            key, subkey = jax.random.split(key)
            return sample_pushforward_mean_std(model, wu, a, n_samples, subkey)

        return predict_fn

    best_sigma2, _ = calibrate_sigma2(
        predict_fn_factory, val_loader,
        sigma2_min=sigma2_min, sigma2_max=sigma2_max,
        n_grid=n_grid, n_val=n_val, verbose=verbose,
    )
    return best_sigma2
