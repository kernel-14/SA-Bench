"""
Evaluation metrics and parameter inversion for SC-FNO.

Metrics:
  - R² (coefficient of determination)
  - Relative L² error

Parameter inversion (Section 3.1):
  Given observed solution u_obs, find p* = argmin ||û(p) - u_obs||²
  using gradient-based optimization (backpropagation through the surrogate).
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from utils import rebuild_input_with_params


def r2_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Compute R² (coefficient of determination).

    R² = 1 - SS_res / SS_tot
    where SS_res = ||pred - target||² and SS_tot = ||target - mean(target)||²
    """
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    ss_res = torch.sum((pred_flat - target_flat) ** 2)
    ss_tot = torch.sum((target_flat - target_flat.mean()) ** 2)
    return (1.0 - ss_res / (ss_tot + eps)).item()


def relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Compute relative L² error.

    rel_L2 = ||pred - target||_F / ||target||_F
    """
    num = torch.norm(pred.reshape(-1) - target.reshape(-1))
    den = torch.norm(target.reshape(-1))
    return (num / (den + eps)).item()


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Compute both R² and relative L² metrics."""
    return {
        "R2": r2_score(pred, target),
        "relative_L2": relative_l2(pred, target),
    }


def evaluate_model_full(
    model: nn.Module,
    loader,
    device: torch.device,
    equation_type: str,
    compute_jacobian: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model on a data loader, computing metrics for u and Jacobians.

    Args:
        model: trained FNO model
        loader: DataLoader
        device: computation device
        equation_type: "ode", "pde1d", or "pde2d"
        compute_jacobian: whether to compute predicted Jacobians via AD

    Returns:
        metrics: dict with keys "u" and optionally "jacobian"
    """
    model.eval()
    all_u_pred = []
    all_u_true = []
    all_jac_pred = []
    all_jac_true = []

    for batch in loader:
        fno_input = batch["fno_input"].to(device)
        u_out = batch["u_out"].to(device)
        params = batch["params"].to(device)

        if compute_jacobian:
            params_rg = params.detach().requires_grad_(True)
            fno_input_rg = rebuild_input_with_params(fno_input, params_rg, equation_type)
            u_pred = model(fno_input_rg)
        else:
            with torch.no_grad():
                u_pred = model(fno_input)

        all_u_pred.append(u_pred.detach().cpu())
        all_u_true.append(u_out.cpu())

        if compute_jacobian and "jac_out" in batch:
            jac_true = batch["jac_out"].to(device)
            jac_pred = _compute_jacobian_from_pred(u_pred, params_rg, equation_type)
            all_jac_pred.append(jac_pred.detach().cpu())
            all_jac_true.append(jac_true.cpu())

    u_pred_all = torch.cat(all_u_pred, dim=0)
    u_true_all = torch.cat(all_u_true, dim=0)

    metrics = {"u": compute_metrics(u_pred_all, u_true_all)}

    if compute_jacobian and all_jac_pred:
        jac_pred_all = torch.cat(all_jac_pred, dim=0)
        jac_true_all = torch.cat(all_jac_true, dim=0)
        n_params = jac_pred_all.shape[-1]

        metrics["jacobian_mean"] = compute_metrics(jac_pred_all, jac_true_all)
        for p_idx in range(n_params):
            metrics[f"jacobian_p{p_idx}"] = compute_metrics(
                jac_pred_all[..., p_idx], jac_true_all[..., p_idx]
            )

    return metrics


def _compute_jacobian_from_pred(
    u_pred: torch.Tensor,
    params_rg: torch.Tensor,
    equation_type: str,
) -> torch.Tensor:
    """
    Compute Jacobian of u_pred w.r.t. params_rg via AD.

    Returns:
        jac: same shape as jac_true in the batch
    """
    batch = params_rg.shape[0]
    n_params = params_rg.shape[1]

    if equation_type == "ode":
        # u_pred: (batch, T_out, 1)
        T_out = u_pred.shape[1]
        jac = torch.zeros(batch, T_out, n_params, device=u_pred.device)
        for t_idx in range(T_out):
            for b in range(batch):
                grad = torch.autograd.grad(
                    u_pred[b, t_idx, 0], params_rg, retain_graph=True, create_graph=False
                )[0]
                jac[b, t_idx, :] = grad[b].detach()
        return jac

    elif equation_type == "pde1d":
        # u_pred: (batch, Sx, T_out, 1)
        Sx, T_out = u_pred.shape[1], u_pred.shape[2]
        jac = torch.zeros(batch, Sx, T_out, n_params, device=u_pred.device)
        for x_idx in range(Sx):
            for t_idx in range(T_out):
                for b in range(batch):
                    grad = torch.autograd.grad(
                        u_pred[b, x_idx, t_idx, 0], params_rg, retain_graph=True, create_graph=False
                    )[0]
                    jac[b, x_idx, t_idx, :] = grad[b].detach()
        return jac

    elif equation_type == "pde2d":
        # u_pred: (batch, Sx, Sy, 1)
        Sx, Sy = u_pred.shape[1], u_pred.shape[2]
        jac = torch.zeros(batch, Sx, Sy, n_params, device=u_pred.device)
        for x_idx in range(Sx):
            for y_idx in range(Sy):
                for b in range(batch):
                    grad = torch.autograd.grad(
                        u_pred[b, x_idx, y_idx, 0], params_rg, retain_graph=True, create_graph=False
                    )[0]
                    jac[b, x_idx, y_idx, :] = grad[b].detach()
        return jac

    else:
        raise ValueError(f"Unknown equation_type: {equation_type}")


class ParameterInverter:
    """
    Parameter inversion using a trained surrogate model.

    Given observed solution u_obs, find p* = argmin ||û(p) - u_obs||²
    via gradient-based optimization (backpropagation through the surrogate).

    This implements the inversion experiments in Section 3.1.
    """

    def __init__(
        self,
        model: nn.Module,
        equation_type: str,
        param_ranges: Dict[str, Tuple[float, float]],
        device: torch.device = torch.device("cpu"),
        n_iter: int = 1000,
        lr: float = 1e-2,
    ):
        """
        Args:
            model: trained FNO surrogate model
            equation_type: "ode", "pde1d", or "pde2d"
            param_ranges: dict of {name: (lo, hi)} for clamping
            device: computation device
            n_iter: number of optimization iterations
            lr: learning rate for parameter optimization
        """
        self.model = model.to(device)
        self.model.eval()
        self.equation_type = equation_type
        self.param_ranges = param_ranges
        self.device = device
        self.n_iter = n_iter
        self.lr = lr

    def invert(
        self,
        u_obs: torch.Tensor,
        fno_input_template: torch.Tensor,
        p_init: Optional[torch.Tensor] = None,
        fixed_params: Optional[Dict[int, float]] = None,
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Invert parameters from observed solution.

        Args:
            u_obs: observed solution (same shape as model output)
            fno_input_template: template input tensor (non-param channels)
            p_init: initial parameter guess (random if None)
            fixed_params: dict of {param_idx: value} for known parameters

        Returns:
            p_opt: optimized parameters (batch, n_params)
            loss_history: list of loss values during optimization
        """
        batch = u_obs.shape[0]
        n_params = len(self.param_ranges)
        param_names = list(self.param_ranges.keys())

        if p_init is None:
            p_opt = torch.zeros(batch, n_params, device=self.device)
            for i, name in enumerate(param_names):
                lo, hi = self.param_ranges[name]
                p_opt[:, i] = (lo + hi) / 2.0
        else:
            p_opt = p_init.clone().to(self.device)

        p_opt = p_opt.requires_grad_(True)
        optimizer = optim.Adam([p_opt], lr=self.lr)

        loss_history = []
        u_obs = u_obs.to(self.device)

        for iteration in range(self.n_iter):
            optimizer.zero_grad()

            # Clamp parameters to valid ranges
            with torch.no_grad():
                for i, name in enumerate(param_names):
                    lo, hi = self.param_ranges[name]
                    p_opt.data[:, i].clamp_(lo * 0.5, hi * 2.0)

                if fixed_params:
                    for idx, val in fixed_params.items():
                        p_opt.data[:, idx] = val

            # Forward pass through surrogate
            fno_input_rg = rebuild_input_with_params(
                fno_input_template, p_opt, self.equation_type
            )
            u_pred = self.model(fno_input_rg)

            # Inversion loss: ||û(p) - u_obs||²
            loss = torch.mean((u_pred - u_obs) ** 2)
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

        return p_opt.detach(), loss_history

    def invert_batch(
        self,
        u_obs_batch: torch.Tensor,
        fno_input_templates: torch.Tensor,
        p_true: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Invert parameters for a batch of observations.

        Args:
            u_obs_batch: (n_samples, ...) observed solutions
            fno_input_templates: (n_samples, ...) input templates
            p_true: (n_samples, n_params) true parameters for evaluation

        Returns:
            results: dict with p_pred, loss_histories, and optionally metrics
        """
        n_samples = u_obs_batch.shape[0]
        n_params = len(self.param_ranges)
        p_pred = torch.zeros(n_samples, n_params)
        loss_histories = []

        for i in range(n_samples):
            u_obs_i = u_obs_batch[i:i+1]
            template_i = fno_input_templates[i:i+1]
            p_opt, history = self.invert(u_obs_i, template_i)
            p_pred[i] = p_opt[0]
            loss_histories.append(history)

        results = {"p_pred": p_pred, "loss_histories": loss_histories}

        if p_true is not None:
            results["metrics"] = {}
            for p_idx, name in enumerate(self.param_ranges.keys()):
                results["metrics"][name] = compute_metrics(
                    p_pred[:, p_idx], p_true[:, p_idx]
                )

        return results


def evaluate_robustness(
    model: nn.Module,
    base_dataset,
    param_ranges: Dict[str, Tuple[float, float]],
    perturbation_ratios: List[float],
    equation_type: str,
    device: torch.device,
    batch_size: int = 4,
) -> Dict[float, Dict[str, float]]:
    """
    Evaluate model robustness to parameter perturbations beyond training range.

    For each perturbation ratio λ, parameters are sampled from [b, (1+λ)b]
    where b is the upper bound of the training range (Section 3.2).

    Args:
        model: trained FNO model
        base_dataset: base dataset to get input templates
        param_ranges: training parameter ranges
        perturbation_ratios: list of λ values to test
        equation_type: "ode", "pde1d", or "pde2d"
        device: computation device
        batch_size: evaluation batch size

    Returns:
        results: dict mapping λ → metrics
    """
    from data import perturb_params
    from torch.utils.data import DataLoader

    results = {}
    model.eval()

    for lam in perturbation_ratios:
        all_u_pred = []
        all_u_true = []

        loader = DataLoader(base_dataset, batch_size=batch_size, shuffle=False)
        for batch in loader:
            fno_input = batch["fno_input"].to(device)
            u_out = batch["u_out"].to(device)
            params = batch["params"].to(device)

            if lam > 0:
                params_perturbed = perturb_params(params, param_ranges, lam)
                fno_input = rebuild_input_with_params(fno_input, params_perturbed, equation_type)

            with torch.no_grad():
                u_pred = model(fno_input)

            all_u_pred.append(u_pred.cpu())
            all_u_true.append(u_out.cpu())

        u_pred_all = torch.cat(all_u_pred, dim=0)
        u_true_all = torch.cat(all_u_true, dim=0)
        results[lam] = compute_metrics(u_pred_all, u_true_all)

    return results
