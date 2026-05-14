"""Out-of-distribution experiments.

Reproduces Tables 2, 6, 7, 8, 9, 10, 11 from the paper:
- Train FNO on 1000 trajectories of Base dataset
- Evaluate on OOD variants: Base, Flip, Pos, Pos-Neg, Pos-Neg-Flip
- Compare: Input Perturbations, Ensemble, Sample-Iso, LUNO-Iso, 
  Sample-LA, LUNO-LA
"""

import jax
import jax.numpy as jnp
from typing import Dict, List

from luno.evaluation import (
    compute_rmse, compute_marginal_nll, compute_chi2_statistic,
)
from experiments.uncertainty_methods import (
    UQResults, input_perturbations, deep_ensemble,
    sample_based_iso, sample_based_la, luno_iso, luno_la,
)


def run_ood_experiment(
    model_fn,
    params,
    ensemble_models,
    ensemble_params,
    laplace_belief,
    ood_datasets: Dict[str, Dict[str, jnp.ndarray]],
    n_test: int = 250,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Run the OOD experiment.
    
    Args:
        model_fn: Trained FNO (on Base dataset)
        params: Trained parameters
        ensemble_models: 10 ensemble members trained on Base
        ensemble_params: 10 ensemble parameter sets
        laplace_belief: LowRankLaplace belief
        ood_datasets: Dict mapping dataset name -> {'inputs': X, 'targets': Y}
        n_test: Number of test pairs (250 in paper)
    
    Returns:
        Nested dict: method -> dataset -> {rmse, chi2, nll}
    """
    # Fixed hyperparameters (calibrated on validation set from Base)
    # In practice, these would come from calibration on validation data
    ip_sigma = 0.01
    iso_sigma = 1.0
    
    results = {}
    
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
    
    for dataset_name, data in ood_datasets.items():
        print(f"\nEvaluating on {dataset_name}...")
        inputs = data['inputs']
        targets = data['targets']
        
        if dataset_name not in results:
            results[dataset_name] = {}
        
        for method_name, method_fn in methods.items():
            print(f"  {method_name}...")
            preds_list = []
            vars_list = []
            
            for i in range(min(n_test, inputs.shape[0])):
                uq = method_fn(inputs[i])
                if uq.mean.ndim > 1:
                    preds_list.append(uq.mean.flatten())
                    vars_list.append(uq.variance.flatten())
                else:
                    preds_list.append(uq.mean)
                    vars_list.append(uq.variance)
            
            preds = jnp.stack(preds_list)
            varis = jnp.stack(vars_list)
            
            if targets.ndim > preds.ndim:
                tgt = targets[:min(n_test, inputs.shape[0])].reshape(preds.shape)
            else:
                tgt = targets[:min(n_test, inputs.shape[0])]
            
            rmse = float(compute_rmse(preds, tgt))
            chi2 = float(compute_chi2_statistic(preds, tgt, varis))
            nll = float(compute_marginal_nll(preds, tgt, varis))
            
            results[dataset_name][method_name] = {
                'rmse': rmse, 'chi2': chi2, 'nll': nll
            }
            
            print(f"    RMSE={rmse:.4e}, χ²={chi2:.3f}, NLL={nll:.4f}")
    
    return results


def print_ood_results_table(results: Dict, metric: str = 'nll'):
    """Print OOD results in table format matching Table 2/6.
    
    Args:
        results: Nested dict from run_ood_experiment
        metric: Which metric to display ('nll', 'rmse', or 'chi2')
    """
    datasets = list(results.keys())
    methods = list(results[datasets[0]].keys())
    
    # Header
    header = f"{'Method':<25}"
    for ds in datasets:
        ds_short = ds.replace('_', '-').title()
        header += f" {ds_short:<12}"
    print(header)
    print("-" * (25 + 13 * len(datasets)))
    
    # Rows
    for method in methods:
        row = f"{method:<25}"
        for ds in datasets:
            val = results[ds][method][metric]
            row += f" {val:<12.4f}"
        print(row)
