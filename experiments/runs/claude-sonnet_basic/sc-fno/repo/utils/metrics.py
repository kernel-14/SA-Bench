"""
Utility functions for SC-FNO experiments.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import json


def compute_metrics(pred, target):
    """
    Compute R2 and relative L2 metrics.
    
    Args:
        pred: Predicted values (numpy array or torch tensor)
        target: True values (numpy array or torch tensor)
    
    Returns:
        Dict with r2 and relative_l2
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    ss_res = np.sum((pred_flat - target_flat) ** 2)
    ss_tot = np.sum((target_flat - target_flat.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    
    rel_l2 = np.linalg.norm(pred_flat - target_flat) / (np.linalg.norm(target_flat) + 1e-8)
    
    return {"r2": float(r2), "relative_l2": float(rel_l2)}


def plot_solution_comparison(u_true, u_pred, t, x=None, title="Solution Comparison", save_path=None):
    """
    Plot comparison between true and predicted solutions.
    
    For 1D: u(t) comparison
    For 2D: u(x, t) heatmap comparison
    """
    if x is None:
        # 1D case: u(t)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].plot(t, u_true, label="True", color="blue")
        axes[0].plot(t, u_pred, label="Predicted", color="red", linestyle="--")
        axes[0].set_xlabel("t")
        axes[0].set_ylabel("u(t)")
        axes[0].set_title("Solution")
        axes[0].legend()
        
        axes[1].plot(t, u_true - u_pred, color="green")
        axes[1].set_xlabel("t")
        axes[1].set_ylabel("Error")
        axes[1].set_title("Prediction Error")
        
        axes[2].text(0.5, 0.5, title, ha="center", va="center", transform=axes[2].transAxes)
        axes[2].axis("off")
    else:
        # 2D case: u(x, t) heatmap
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        vmin = min(u_true.min(), u_pred.min())
        vmax = max(u_true.max(), u_pred.max())
        
        im0 = axes[0].imshow(u_true, aspect="auto", origin="lower",
                              extent=[t[0], t[-1], x[0], x[-1]],
                              vmin=vmin, vmax=vmax, cmap="RdBu_r")
        axes[0].set_title("True u(x,t)")
        axes[0].set_xlabel("t")
        axes[0].set_ylabel("x")
        plt.colorbar(im0, ax=axes[0])
        
        im1 = axes[1].imshow(u_pred, aspect="auto", origin="lower",
                              extent=[t[0], t[-1], x[0], x[-1]],
                              vmin=vmin, vmax=vmax, cmap="RdBu_r")
        axes[1].set_title("Predicted u(x,t)")
        axes[1].set_xlabel("t")
        axes[1].set_ylabel("x")
        plt.colorbar(im1, ax=axes[1])
        
        error = u_true - u_pred
        im2 = axes[2].imshow(error, aspect="auto", origin="lower",
                              extent=[t[0], t[-1], x[0], x[-1]],
                              cmap="RdBu_r")
        axes[2].set_title("Error")
        axes[2].set_xlabel("t")
        axes[2].set_ylabel("x")
        plt.colorbar(im2, ax=axes[2])
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        return fig


def plot_jacobian_comparison(jac_true, jac_pred, param_name, t, x=None, save_path=None):
    """
    Plot comparison between true and predicted Jacobians.
    """
    if x is None:
        # 1D case
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(t, jac_true, label="True", color="blue")
        axes[0].plot(t, jac_pred, label="Predicted", color="red", linestyle="--")
        axes[0].set_xlabel("t")
        axes[0].set_ylabel(f"du/d{param_name}")
        axes[0].set_title(f"Jacobian w.r.t. {param_name}")
        axes[0].legend()
        
        axes[1].plot(t, jac_true - jac_pred, color="green")
        axes[1].set_xlabel("t")
        axes[1].set_ylabel("Error")
        axes[1].set_title("Jacobian Error")
    else:
        # 2D case
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        vmin = min(jac_true.min(), jac_pred.min())
        vmax = max(jac_true.max(), jac_pred.max())
        
        im0 = axes[0].imshow(jac_true, aspect="auto", origin="lower",
                              extent=[t[0], t[-1], x[0], x[-1]],
                              vmin=vmin, vmax=vmax, cmap="RdBu_r")
        axes[0].set_title(f"True du/d{param_name}")
        plt.colorbar(im0, ax=axes[0])
        
        im1 = axes[1].imshow(jac_pred, aspect="auto", origin="lower",
                              extent=[t[0], t[-1], x[0], x[-1]],
                              vmin=vmin, vmax=vmax, cmap="RdBu_r")
        axes[1].set_title(f"Predicted du/d{param_name}")
        plt.colorbar(im1, ax=axes[1])
        
        error = jac_true - jac_pred
        im2 = axes[2].imshow(error, aspect="auto", origin="lower",
                              extent=[t[0], t[-1], x[0], x[-1]],
                              cmap="RdBu_r")
        axes[2].set_title("Error")
        plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        return fig


def plot_training_curves(train_losses, val_losses, save_path=None):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    epochs = range(1, len(train_losses) + 1)
    ax.semilogy(epochs, train_losses, label="Train Loss", color="blue")
    ax.semilogy(epochs, val_losses, label="Val Loss", color="red")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        return fig


def plot_perturbation_robustness(lambdas, fno_r2, sc_fno_r2, save_path=None):
    """
    Plot model performance vs perturbation ratio (Figure 5 equivalent).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(lambdas, fno_r2, "b-o", label="FNO", linewidth=2)
    ax.plot(lambdas, sc_fno_r2, "r-s", label="SC-FNO", linewidth=2)
    ax.set_xlabel("Perturbation ratio λ")
    ax.set_ylabel("R²")
    ax.set_title("Model Robustness to Parameter Perturbations")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        return fig


def plot_sample_size_performance(sample_sizes, fno_r2, sc_fno_r2, metric="R²", save_path=None):
    """
    Plot model performance vs training sample size (Figure 4 equivalent).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.semilogx(sample_sizes, fno_r2, "b-o", label="FNO", linewidth=2)
    ax.semilogx(sample_sizes, sc_fno_r2, "r-s", label="SC-FNO", linewidth=2)
    ax.set_xlabel("Number of training samples")
    ax.set_ylabel(metric)
    ax.set_title(f"Model Performance vs Training Data Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        return fig


def print_results_table(results_dict, metrics=("u_r2", "u_relative_l2")):
    """Print a formatted results table."""
    models = list(results_dict.keys())
    
    print("\n" + "="*80)
    header = f"{'Metric':<30}"
    for m in models:
        header += f" {m:>12}"
    print(header)
    print("-"*80)
    
    for metric in metrics:
        row = f"{metric:<30}"
        for m in models:
            val = results_dict[m].get(metric, float("nan"))
            row += f" {val:>12.4f}"
        print(row)
    
    print("="*80)


def save_results(results, filepath):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filepath}")


def load_results(filepath):
    """Load results from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)
