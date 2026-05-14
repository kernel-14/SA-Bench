"""Low-data regime experiments.

Reproduces Tables 1, 4, 5 from the paper:
- Train FNO for 100 epochs on 25 trajectories
- Evaluate UQ methods on 250 unseen test pairs
- Compare: Input Perturbations, Ensemble, Sample-Iso, LUNO-Iso, 
  Sample-LA, LUNO-LA

Metrics: RMSE, χ², NLL
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple

from luno.evaluation import (
    compute_rmse, compute_marginal_nll, compute_chi2_statistic,
    evaluate_all_metrics, calibrate_hyperparameter, compute_log_spaced_grid,
)
from experiments.uncertainty_methods import (
    UQResults, input_perturbations, deep_ensemble,
    sample_based_iso, sample_based_la, luno_iso, luno_la,
)


def run_low_data_experiment(
    model_fn,
    params,
    ensemble_models,
    ensemble_params,
    laplace_belief,
    test_inputs: jnp.ndarray,
    test_targets: jnp.ndarray,
    val_inputs: jnp.ndarray,
    val_targets: jnp.ndarray,
    n_test: int = 250,
) -> Dict[str, Dict[str, float]]:
    """Run the low-data regime experiment.
    
    Args:
        model_fn: Trained FNO model function
        params: Trained FNO parameters (MAP)
        ensemble_models: List of 10 ensemble model functions
        ensemble_params: List of 10 ensemble parameter sets
        laplace_belief: LowRankLaplace belief
        test_inputs: Test inputs
        test_targets: Test targets
        val_inputs: Validation inputs (for calibration)
        val_targets: Validation targets
        n_test: Number of test pairs (250 in paper)
    
    Returns:
        Dictionary mapping method name -> {rmse, chi2, nll}
    """
    results = {}
    
    # Calibrate hyperparameters on validation set
    print("Calibrating hyperparameters...")
    
    # Calibrate input perturbations σ
    best_ip_sigma, _ = calibrate_hyperparameter(
        lambda s: _eval_input_perturbations(model_fn, params, val_inputs, s),
        val_targets,
        compute_log_spaced_grid(0.01, 500, 4.0),
    )
    
    # Calibrate isotropic σ²
    best_iso_sigma, _ = calibrate_hyperparameter(
        lambda s: _eval_isotropic(model_fn, params, val_inputs, s),
        val_targets,
        compute_log_spaced_grid(1.0, 500, 4.0),
    )
    
    print(f"  Input perturbations σ = {best_ip_sigma:.6f}")
    print(f"  Isotropic σ² = {best_iso_sigma:.6f}")
    
    # Evaluate all methods on test set
    methods = {
        'Input Perturbations': lambda x: input_perturbations(
            model_fn, params, x, best_ip_sigma, n_perturbations=200),
        'Ensemble': lambda x: deep_ensemble(
            ensemble_models, ensemble_params, x),
        'Sample-Iso': lambda x: sample_based_iso(
            model_fn, params, x, best_iso_sigma, n_samples=200),
        'LUNO-Iso': lambda x: luno_iso(
            model_fn, params, x, best_iso_sigma),
        'Sample-LA': lambda x: sample_based_la(
            model_fn, params, x, laplace_belief, n_samples=200),
        'LUNO-LA': lambda x: luno_la(
            model_fn, params, x, laplace_belief),
    }
    
    for name, method_fn in methods.items():
        print(f"Evaluating {name}...")
        preds_list = []
        vars_list = []
        
        for i in range(min(n_test, test_inputs.shape[0])):
            uq_result = method_fn(test_inputs[i])
            
            # Handle different output shapes
            if uq_result.mean.ndim > 1:
                preds_list.append(uq_result.mean.flatten())
                vars_list.append(uq_result.variance.flatten())
            else:
                preds_list.append(uq_result.mean)
                vars_list.append(uq_result.variance)
        
        preds = jnp.stack(preds_list)
        varis = jnp.stack(vars_list)
        
        # Reshape targets to match
        if test_targets[i].ndim > 1:
            tgt = test_targets[:min(n_test, test_inputs.shape[0])].reshape(preds.shape)
        else:
            tgt = test_targets[:min(n_test, test_inputs.shape[0])]
        
        rmse = compute_rmse(preds, tgt)
        chi2 = compute_chi2_statistic(preds, tgt, varis)
        nll = compute_marginal_nll(preds, tgt, varis)
        
        results[name] = {
            'rmse': float(rmse),
            'chi2': float(chi2),
            'nll': float(nll),
        }
        
        print(f"  RMSE={rmse:.6f}, χ²={chi2:.3f}, NLL={nll:.4f}")
    
    return results


def _eval_input_perturbations(model_fn, params, inputs, sigma):
    """Helper for calibration: evaluate input perturbations."""
    all_preds = []
    all_vars = []
    for i in range(inputs.shape[0]):
        uq = input_perturbations(model_fn, params, inputs[i], sigma)
        all_preds.append(uq.mean.flatten())
        all_vars.append(uq.variance.flatten())
    return jnp.stack(all_preds), jnp.stack(all_vars)


def _eval_isotropic(model_fn, params, inputs, sigma_squared):
    """Helper for calibration: evaluate LUNO-Iso."""
    all_preds = []
    all_vars = []
    for i in range(inputs.shape[0]):
        uq = luno_iso(model_fn, params, inputs[i], sigma_squared)
        all_preds.append(uq.mean.flatten())
        all_vars.append(uq.variance.flatten())
    return jnp.stack(all_preds), jnp.stack(all_vars)


def print_results_table(results: Dict[str, Dict[str, float]], title: str = ""):
    """Print results in table format matching Tables 1, 4, 5."""
    if title:
        print(f"\n{title}")
    print(f"{'Method':<25} {'RMSE (↓)':<12} {'χ²':<10} {'NLL (↓)':<12}")
    print("-" * 60)
    for method, metrics in results.items():
        print(f"{method:<25} {metrics['rmse']:.4e}   {metrics['chi2']:.3f}   {metrics['nll']:.4f}")
