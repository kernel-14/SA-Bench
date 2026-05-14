"""
Parameter inversion experiments using trained SC-FNO models.
Implements Section 3.1 of the paper.

The inversion procedure:
1. Train FNO/SC-FNO surrogate models
2. Given observed solution paths, optimize parameters to minimize
   discrepancy between surrogate predictions and observations
3. Compare inversion accuracy between FNO and SC-FNO
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.sc_fno_experiment import SCFNO, relative_l2_loss, r2_score_torch


def invert_parameters(
    model,
    x_base_observed,
    u_observed,
    param_ranges,
    n_iter=1000,
    lr=0.01,
    device="cpu",
    n_restarts=3
):
    """
    Invert parameters from observed solution paths using the surrogate model.
    
    Uses gradient-based optimization (Adam) to minimize:
    L = ||u_pred(x_base, p) - u_observed||^2
    
    Args:
        model: Trained SCFNO model
        x_base_observed: Base input (batch, ..., d) - initial conditions + coords
        u_observed: Observed solution (batch, ..., 1)
        param_ranges: Dict of {param_name: [min, max]}
        n_iter: Number of optimization iterations
        lr: Learning rate
        device: Computation device
        n_restarts: Number of random restarts
    
    Returns:
        Optimized parameters (batch, n_params)
    """
    model.eval()
    
    x_base = torch.FloatTensor(x_base_observed).to(device)
    u_obs = torch.FloatTensor(u_observed).to(device)
    
    batch_size = x_base.shape[0]
    n_params = len(param_ranges)
    param_names = list(param_ranges.keys())
    
    best_params = None
    best_loss = float("inf")
    
    for restart in range(n_restarts):
        # Initialize parameters randomly within ranges
        params_init = np.zeros((batch_size, n_params))
        for j, (name, (lo, hi)) in enumerate(param_ranges.items()):
            params_init[:, j] = np.random.uniform(lo, hi, batch_size)
        
        params = torch.FloatTensor(params_init).to(device).requires_grad_(True)
        optimizer = optim.Adam([params], lr=lr)
        
        for i in range(n_iter):
            optimizer.zero_grad()
            
            u_pred = model.forward_with_params(x_base, params)
            loss = relative_l2_loss(u_pred, u_obs)
            
            loss.backward()
            optimizer.step()
            
            # Clamp parameters to valid ranges
            with torch.no_grad():
                for j, (name, (lo, hi)) in enumerate(param_ranges.items()):
                    params[:, j].clamp_(lo, hi)
        
        final_loss = loss.item()
        if final_loss < best_loss:
            best_loss = final_loss
            best_params = params.detach().cpu().numpy()
    
    return best_params


def run_inversion_experiment_pde1(
    fno_model,
    sc_fno_model,
    test_data,
    device="cpu",
    save_dir="results/inversion"
):
    """
    Run single-parameter inversion experiment for PDE1.
    Inverts alpha parameter while treating others as known.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    x_base = test_data["x_base"]
    true_params = test_data["params"]
    u_observed = test_data["targets"]
    
    # Single parameter inversion: invert alpha only
    # Use true values for other parameters
    param_ranges_single = {"alpha": [0.0, 0.1]}
    
    print("Running single-parameter inversion (alpha)...")
    
    results = {}
    for name, model in [("FNO", fno_model), ("SC-FNO", sc_fno_model)]:
        # For single param inversion, we need to fix other params
        # Create modified x_base that includes known params
        # This is a simplified version - in practice, known params are embedded in x_base
        
        inverted = invert_parameters(
            model, x_base, u_observed,
            param_ranges_single,
            n_iter=500, lr=0.01, device=device
        )
        
        true_alpha = true_params[:, 1]  # alpha is index 1 in PDE1
        pred_alpha = inverted[:, 0]
        
        r2 = r2_score_torch(
            torch.FloatTensor(pred_alpha),
            torch.FloatTensor(true_alpha)
        )
        l2 = relative_l2_loss(
            torch.FloatTensor(pred_alpha),
            torch.FloatTensor(true_alpha)
        ).item()
        
        results[name] = {
            "r2": r2,
            "relative_l2": l2,
            "true_alpha": true_alpha.tolist(),
            "pred_alpha": pred_alpha.tolist()
        }
        
        print(f"  {name}: R2={r2:.4f}, Relative L2={l2:.4f}")
    
    # Save results
    with open(os.path.join(save_dir, "single_param_inversion.json"), "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, list)} 
                   for k, v in results.items()}, f, indent=2)
    
    # Create scatter plot (Figure 1a equivalent)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax, (name, res) in zip(axes, results.items()):
        ax.scatter(res["true_alpha"], res["pred_alpha"], alpha=0.5, s=10)
        lo, hi = min(res["true_alpha"]), max(res["true_alpha"])
        ax.plot([lo, hi], [lo, hi], "r--", label="Perfect")
        ax.set_xlabel("True alpha")
        ax.set_ylabel("Predicted alpha")
        ax.set_title(f"{name} - Single Param Inversion\nR2={res['r2']:.3f}")
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "single_param_inversion.png"), dpi=150)
    plt.close()
    
    return results


def run_multi_param_inversion_pde1(
    fno_model,
    sc_fno_model,
    test_data,
    device="cpu",
    save_dir="results/inversion"
):
    """
    Run multi-parameter inversion experiment for PDE1.
    Simultaneously inverts all 5 parameters.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    x_base = test_data["x_base"]
    true_params = test_data["params"]
    u_observed = test_data["targets"]
    
    param_ranges = {
        "c": [0.0, 0.25],
        "alpha": [0.0, 0.1],
        "beta": [0.0, 0.25],
        "gamma": [0.0, 0.25],
        "omega": [0.0, 0.25],
    }
    
    print("Running multi-parameter inversion (all 5 params)...")
    
    results = {}
    for name, model in [("FNO", fno_model), ("SC-FNO", sc_fno_model)]:
        inverted = invert_parameters(
            model, x_base, u_observed,
            param_ranges,
            n_iter=1000, lr=0.01, device=device
        )
        
        param_results = {}
        for j, pname in enumerate(param_ranges.keys()):
            true_p = true_params[:, j]
            pred_p = inverted[:, j]
            
            r2 = r2_score_torch(
                torch.FloatTensor(pred_p),
                torch.FloatTensor(true_p)
            )
            l2 = relative_l2_loss(
                torch.FloatTensor(pred_p),
                torch.FloatTensor(true_p)
            ).item()
            
            param_results[pname] = {"r2": r2, "relative_l2": l2}
        
        results[name] = param_results
        
        avg_r2 = np.mean([v["r2"] for v in param_results.values()])
        avg_l2 = np.mean([v["relative_l2"] for v in param_results.values()])
        print(f"  {name}: Avg R2={avg_r2:.4f}, Avg Relative L2={avg_l2:.4f}")
        for pname, pres in param_results.items():
            print(f"    {pname}: R2={pres['r2']:.4f}, L2={pres['relative_l2']:.4f}")
    
    with open(os.path.join(save_dir, "multi_param_inversion_pde1.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def run_multi_param_inversion_pde2(
    fno_model,
    sc_fno_model,
    test_data,
    device="cpu",
    save_dir="results/inversion"
):
    """
    Run multi-parameter inversion experiment for PDE2.
    Simultaneously inverts all 4 parameters.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    x_base = test_data["x_base"]
    true_params = test_data["params"]
    u_observed = test_data["targets"]
    
    param_ranges = {
        "alpha": [0.1, 1.0],
        "gamma": [0.025, 0.25],
        "delta": [0.1, 0.5],
        "omega": [0.01, 0.1],
    }
    
    print("Running multi-parameter inversion for PDE2 (4 params)...")
    
    results = {}
    for name, model in [("FNO", fno_model), ("SC-FNO", sc_fno_model)]:
        inverted = invert_parameters(
            model, x_base, u_observed,
            param_ranges,
            n_iter=1000, lr=0.01, device=device
        )
        
        param_results = {}
        for j, pname in enumerate(param_ranges.keys()):
            true_p = true_params[:, j]
            pred_p = inverted[:, j]
            
            r2 = r2_score_torch(
                torch.FloatTensor(pred_p),
                torch.FloatTensor(true_p)
            )
            l2 = relative_l2_loss(
                torch.FloatTensor(pred_p),
                torch.FloatTensor(true_p)
            ).item()
            
            param_results[pname] = {"r2": r2, "relative_l2": l2}
        
        results[name] = param_results
        
        avg_r2 = np.mean([v["r2"] for v in param_results.values()])
        avg_l2 = np.mean([v["relative_l2"] for v in param_results.values()])
        print(f"  {name}: Avg R2={avg_r2:.4f}, Avg Relative L2={avg_l2:.4f}")
    
    with open(os.path.join(save_dir, "multi_param_inversion_pde2.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def plot_inversion_scatter(results, param_name, save_path):
    """Create scatter plot comparing true vs predicted parameters."""
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]
    
    for ax, (model_name, res) in zip(axes, results.items()):
        if param_name in res:
            true_vals = res[param_name].get("true_values", [])
            pred_vals = res[param_name].get("pred_values", [])
            r2 = res[param_name]["r2"]
            
            if true_vals and pred_vals:
                ax.scatter(true_vals, pred_vals, alpha=0.5, s=10)
                lo = min(min(true_vals), min(pred_vals))
                hi = max(max(true_vals), max(pred_vals))
                ax.plot([lo, hi], [lo, hi], "r--")
            
            ax.set_xlabel(f"True {param_name}")
            ax.set_ylabel(f"Predicted {param_name}")
            ax.set_title(f"{model_name}\nR2={r2:.3f}")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
