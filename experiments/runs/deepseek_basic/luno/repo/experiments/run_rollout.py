"""Autoregressive rollout evaluation (Figures 4, 7).

Evaluates UQ methods on iterative autoregressive rollouts where
predictions are recursively fed back as inputs.

This is a critical test since prediction errors accumulate and are 
treated as ground truth for subsequent steps, causing distribution shift.

Key findings from paper:
- Deep ensembles improve RMSE but uncertainty doesn't adapt to increasing error
- LUNO-LA maintains better calibrated uncertainty throughout rollouts
"""

import jax
import jax.numpy as jnp
from typing import Dict, List, Callable, Any

from luno.evaluation import compute_rmse, compute_marginal_nll, compute_chi2_statistic
from experiments.uncertainty_methods import (
    UQResults, input_perturbations, deep_ensemble,
    sample_based_iso, sample_based_la, luno_iso, luno_la,
)


def run_rollout(
    model_fn: Callable,
    params: Any,
    uq_method_fn: Callable,
    initial_input: jnp.ndarray,
    n_steps: int,
    ground_truth: jnp.ndarray = None,
) -> Dict[str, jnp.ndarray]:
    """Run autoregressive rollout with uncertainty quantification.
    
    At each step:
    1. Compute predictive distribution using UQ method
    2. Use mean prediction as input for next step
    3. Track mean, variance, and ground truth
    
    Args:
        model_fn: Model function (predicts next step from last 10)
        params: Model parameters
        uq_method_fn: Function mapping input -> UQResults
        initial_input: Initial condition (10 time steps)
        n_steps: Number of rollout steps
        ground_truth: True trajectory for comparison (optional)
    
    Returns:
        Dict with 'means', 'variances', 'predictions' arrays
    """
    means = []
    variances = []
    predictions = []
    
    current_input = initial_input
    
    for step in range(n_steps):
        uq = uq_method_fn(current_input)
        
        # Store results
        means.append(uq.mean)
        variances.append(uq.variance)
        
        # Use mean as prediction for next step
        next_input = jnp.roll(current_input, shift=-1, axis=0)
        next_input = next_input.at[-1].set(uq.mean)
        current_input = next_input
    
    result = {
        'means': jnp.stack(means),
        'variances': jnp.stack(variances),
    }
    
    if ground_truth is not None:
        result['targets'] = ground_truth[:n_steps]
        result['rmse_per_step'] = jnp.array([
            compute_rmse(means[t], ground_truth[t])
            for t in range(n_steps)
        ])
        result['nll_per_step'] = jnp.array([
            compute_marginal_nll(means[t], ground_truth[t], variances[t])
            for t in range(n_steps)
        ])
    
    return result


def run_rollout_comparison(
    model_fn: Callable,
    params: Any,
    ensemble_models: List,
    ensemble_params: List,
    laplace_belief: Any,
    initial_inputs: jnp.ndarray,  # (n_trajs, 10, n_x, ...)
    ground_truths: jnp.ndarray,   # (n_trajs, n_steps, n_x, ...)
    n_steps: int,
) -> Dict[str, Dict[str, jnp.ndarray]]:
    """Run rollout comparison across all UQ methods.
    
    Args:
        model_fn: FNO model
        params: Trained parameters
        ensemble_models: 10 ensemble members
        ensemble_params: 10 parameter sets
        laplace_belief: LowRankLaplace belief
        initial_inputs: Initial conditions for each trajectory
        ground_truths: True trajectories
        n_steps: Rollout length
    
    Returns:
        Dict: method -> {rmse_curve, nll_curve, chi2_curve}
    """
    ip_sigma = 0.01
    iso_sigma = 1.0
    
    methods = {
        'Input Perturbations': lambda x: input_perturbations(
            model_fn, params, x, ip_sigma, n_perturbations=200),
        'Ensemble': lambda x: deep_ensemble(
            ensemble_models, ensemble_params, x),
        'Sample-Iso': lambda x: sample_based_iso(
            model_fn, params, x, iso_sigma, n_samples=200),
        'LUNO-Iso': lambda x: luno_iso(
            model_fn, params, x, iso_sigma),
        'Sample-LA': lambda x: sample_based_la(
            model_fn, params, x, laplace_belief, n_samples=200),
        'LUNO-LA': lambda x: luno_la(
            model_fn, params, x, laplace_belief),
    }
    
    n_trajs = initial_inputs.shape[0]
    results = {}
    
    for method_name, method_fn in methods.items():
        print(f"Rollout: {method_name}...")
        
        all_rmse = []
        all_nll = []
        
        for traj in range(n_trajs):
            rollout = run_rollout(
                model_fn, params, method_fn,
                initial_inputs[traj], n_steps,
                ground_truths[traj],
            )
            all_rmse.append(rollout['rmse_per_step'])
            all_nll.append(rollout['nll_per_step'])
        
        results[method_name] = {
            'rmse_curve': jnp.mean(jnp.stack(all_rmse), axis=0),
            'nll_curve': jnp.mean(jnp.stack(all_nll), axis=0),
        }
    
    return results
