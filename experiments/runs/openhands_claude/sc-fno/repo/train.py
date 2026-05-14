"""
Training loop for FNO, FNO-PINN, SC-FNO, and SC-FNO-PINN.

Four model variants (Section 3):
  1. FNO:          L_total = c1 * L_u
  2. FNO-PINN:     L_total = c1 * L_u + c3 * L_eq
  3. SC-FNO:       L_total = c1 * L_u + c2 * L_s
  4. SC-FNO-PINN:  L_total = c1 * L_u + c2 * L_s + c3 * L_eq

Sensitivity loss (Eq. 6):
  L_s = (1/M) * sum_j ||∂û(x_j,t_j;p)/∂p - ∂u(x_j,t_j;p)/∂p||²

Efficiency: randomly sample n < N spatial and t < T temporal points per epoch
(Section 2.4), varying between epochs to cover the full solution space.
"""

import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import rebuild_input_with_params


def relative_l2_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Relative L² loss: ||pred - target||² / ||target||²."""
    return torch.mean(
        torch.norm(pred.reshape(pred.shape[0], -1) - target.reshape(target.shape[0], -1), dim=1)
        / (torch.norm(target.reshape(target.shape[0], -1), dim=1) + eps)
    )


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error loss."""
    return torch.mean((pred - target) ** 2)


def compute_sensitivity_loss(
    model: nn.Module,
    fno_input: torch.Tensor,
    params: torch.Tensor,
    jac_true: torch.Tensor,
    n_spatial_samples: int,
    n_time_samples: int,
    equation_type: str,
) -> torch.Tensor:
    """
    Compute sensitivity loss L_s by comparing predicted and true Jacobians
    at randomly sampled spatial-temporal points (Section 2.4).

    The predicted Jacobian ∂û/∂p is computed via AD applied to the FNO.
    We require params to have requires_grad=True for this computation.

    Args:
        model: FNO model
        fno_input: model input tensor (requires_grad may be set on params channel)
        params: (batch, n_params) physical parameters with requires_grad=True
        jac_true: true Jacobian tensor
          - ODE: (batch, T_out, n_params)
          - PDE1D: (batch, Sx, T_out, n_params)
          - PDE2D: (batch, Sx, Sy, 1, n_params)
        n_spatial_samples: number of spatial points to sample
        n_time_samples: number of time points to sample
        equation_type: "ode", "pde1d", or "pde2d"

    Returns:
        L_s: scalar sensitivity loss
    """
    batch = params.shape[0]
    n_params = params.shape[1]

    # Build input with params embedded, requiring grad w.r.t. params
    # We need to differentiate the model output w.r.t. params
    # params is already embedded in fno_input; we need a fresh forward pass
    # where params has requires_grad=True

    params_rg = params.detach().requires_grad_(True)

    # Rebuild fno_input with params_rg embedded
    fno_input_rg = rebuild_input_with_params(fno_input, params_rg, equation_type)

    u_pred = model(fno_input_rg)  # forward pass

    if equation_type == "ode":
        # u_pred: (batch, T_out, 1)
        T_out = u_pred.shape[1]
        t_indices = _sample_indices(T_out, n_time_samples, u_pred.device)

        loss = torch.tensor(0.0, device=u_pred.device)
        count = 0
        for t_idx in t_indices:
            # Compute ∂û[:,t_idx,:]/∂params via AD
            u_t = u_pred[:, t_idx, 0]  # (batch,)
            jac_pred = torch.zeros(batch, n_params, device=u_pred.device)
            for b in range(batch):
                grad = torch.autograd.grad(
                    u_t[b], params_rg, retain_graph=True, create_graph=True
                )[0]
                jac_pred[b] = grad[b]

            jac_t_true = jac_true[:, t_idx, :]  # (batch, n_params)
            loss = loss + mse_loss(jac_pred, jac_t_true)
            count += 1

        return loss / max(count, 1)

    elif equation_type == "pde1d":
        # u_pred: (batch, Sx, T_out, 1)
        Sx = u_pred.shape[1]
        T_out = u_pred.shape[2]
        x_indices = _sample_indices(Sx, n_spatial_samples, u_pred.device)
        t_indices = _sample_indices(T_out, n_time_samples, u_pred.device)

        loss = torch.tensor(0.0, device=u_pred.device)
        count = 0
        for x_idx in x_indices:
            for t_idx in t_indices:
                u_xt = u_pred[:, x_idx, t_idx, 0]  # (batch,)
                jac_pred = torch.zeros(batch, n_params, device=u_pred.device)
                for b in range(batch):
                    grad = torch.autograd.grad(
                        u_xt[b], params_rg, retain_graph=True, create_graph=True
                    )[0]
                    jac_pred[b] = grad[b]

                jac_xt_true = jac_true[:, x_idx, t_idx, :]  # (batch, n_params)
                loss = loss + mse_loss(jac_pred, jac_xt_true)
                count += 1

        return loss / max(count, 1)

    elif equation_type == "pde2d":
        # u_pred: (batch, Sx, Sy, 1)
        Sx = u_pred.shape[1]
        Sy = u_pred.shape[2]
        x_indices = _sample_indices(Sx, n_spatial_samples, u_pred.device)
        y_indices = _sample_indices(Sy, n_spatial_samples, u_pred.device)

        loss = torch.tensor(0.0, device=u_pred.device)
        count = 0
        for x_idx in x_indices:
            for y_idx in y_indices:
                u_xy = u_pred[:, x_idx, y_idx, 0]  # (batch,)
                jac_pred = torch.zeros(batch, n_params, device=u_pred.device)
                for b in range(batch):
                    grad = torch.autograd.grad(
                        u_xy[b], params_rg, retain_graph=True, create_graph=True
                    )[0]
                    jac_pred[b] = grad[b]

                jac_xy_true = jac_true[:, x_idx, y_idx, :]  # (batch, n_params)
                loss = loss + mse_loss(jac_pred, jac_xy_true)
                count += 1

        return loss / max(count, 1)

    else:
        raise ValueError(f"Unknown equation_type: {equation_type}")


def _sample_indices(size: int, n_samples: int, device: torch.device) -> torch.Tensor:
    """Randomly sample n_samples indices from [0, size)."""
    n = min(n_samples, size)
    return torch.randperm(size, device=device)[:n]


class Trainer:
    """
    Trainer for FNO, FNO-PINN, SC-FNO, and SC-FNO-PINN.

    Implements Algorithms 1, 2, 3 from the paper.
    """

    def __init__(
        self,
        model: nn.Module,
        variant: str,
        equation_type: str,
        c1: float = 1.0,
        c2: float = 1.0,
        c3: float = 1.0,
        learning_rate: float = 1e-3,
        n_spatial_samples: int = 10,
        n_time_samples: int = 10,
        pinn_residual_fn: Optional[Callable] = None,
        device: torch.device = torch.device("cpu"),
        checkpoint_dir: str = "checkpoints",
    ):
        """
        Args:
            model: FNO model
            variant: one of "FNO", "FNO-PINN", "SC-FNO", "SC-FNO-PINN"
            equation_type: "ode", "pde1d", or "pde2d"
            c1: weight for L_u
            c2: weight for L_s (used in SC-FNO variants)
            c3: weight for L_eq (used in PINN variants)
            learning_rate: optimizer learning rate
            n_spatial_samples: spatial points sampled per epoch for L_s
            n_time_samples: time points sampled per epoch for L_s
            pinn_residual_fn: function(u_pred, params) → residual tensor
            device: computation device
            checkpoint_dir: directory to save model checkpoints
        """
        assert variant in ["FNO", "FNO-PINN", "SC-FNO", "SC-FNO-PINN"], \
            f"Unknown variant: {variant}"

        self.model = model.to(device)
        self.variant = variant
        self.equation_type = equation_type
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.n_spatial_samples = n_spatial_samples
        self.n_time_samples = n_time_samples
        self.pinn_residual_fn = pinn_residual_fn
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        self.use_sensitivity = variant in ["SC-FNO", "SC-FNO-PINN"]
        self.use_pinn = variant in ["FNO-PINN", "SC-FNO-PINN"]

        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5)

        self.train_losses: List[Dict[str, float]] = []
        self.val_losses: List[Dict[str, float]] = []

        os.makedirs(checkpoint_dir, exist_ok=True)

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        lu_total = 0.0
        ls_total = 0.0
        leq_total = 0.0
        n_batches = 0

        for batch in loader:
            fno_input = batch["fno_input"].to(self.device)
            u_out = batch["u_out"].to(self.device)
            params = batch["params"].to(self.device)
            jac_out = batch["jac_out"].to(self.device) if self.use_sensitivity else None

            self.optimizer.zero_grad()

            # Forward pass (Algorithm 1, line 5)
            u_pred = self.model(fno_input)

            # Data loss L_u (Algorithm 1, line 6)
            L_u = relative_l2_loss(u_pred, u_out)
            loss = self.c1 * L_u
            lu_total += L_u.item()

            # Sensitivity loss L_s (Algorithm 2, lines 7-8)
            if self.use_sensitivity:
                L_s = compute_sensitivity_loss(
                    self.model,
                    fno_input,
                    params,
                    jac_out,
                    self.n_spatial_samples,
                    self.n_time_samples,
                    self.equation_type,
                )
                loss = loss + self.c2 * L_s
                ls_total += L_s.item()

            # PINN equation loss L_eq (Algorithm 3, line 9)
            if self.use_pinn and self.pinn_residual_fn is not None:
                # Squeeze the output channel dimension before passing to residual fn
                u_pred_sq = u_pred.squeeze(-1)
                residual = self.pinn_residual_fn(u_pred_sq, params)
                L_eq = torch.mean(residual ** 2)
                loss = loss + self.c3 * L_eq
                leq_total += L_eq.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        self.scheduler.step()

        return {
            "total": total_loss / n_batches,
            "L_u": lu_total / n_batches,
            "L_s": ls_total / n_batches if self.use_sensitivity else 0.0,
            "L_eq": leq_total / n_batches if self.use_pinn else 0.0,
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on a data loader."""
        self.model.eval()
        total_loss = 0.0
        lu_total = 0.0
        n_batches = 0

        for batch in loader:
            fno_input = batch["fno_input"].to(self.device)
            u_out = batch["u_out"].to(self.device)

            u_pred = self.model(fno_input)
            L_u = relative_l2_loss(u_pred, u_out)

            lu_total += L_u.item()
            total_loss += L_u.item()
            n_batches += 1

        return {
            "total": total_loss / n_batches,
            "L_u": lu_total / n_batches,
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int,
        save_best: bool = True,
        experiment_name: str = "experiment",
    ) -> Dict[str, List]:
        """
        Full training loop (Algorithms 1-3).

        Args:
            train_loader: training data loader
            val_loader: validation data loader
            n_epochs: number of training epochs
            save_best: whether to save the best model checkpoint
            experiment_name: name for checkpoint files

        Returns:
            history: dict with train/val loss histories
        """
        best_val_loss = float("inf")
        history = {"train": [], "val": [], "epoch_times": []}

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            epoch_time = time.time() - t0

            self.train_losses.append(train_metrics)
            self.val_losses.append(val_metrics)
            history["train"].append(train_metrics)
            history["val"].append(val_metrics)
            history["epoch_times"].append(epoch_time)

            if epoch % 50 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:4d}/{n_epochs} | "
                    f"Train: {train_metrics['total']:.4f} "
                    f"(L_u={train_metrics['L_u']:.4f}, "
                    f"L_s={train_metrics['L_s']:.4f}, "
                    f"L_eq={train_metrics['L_eq']:.4f}) | "
                    f"Val: {val_metrics['total']:.4f} | "
                    f"Time: {epoch_time:.2f}s"
                )

            if save_best and val_metrics["total"] < best_val_loss:
                best_val_loss = val_metrics["total"]
                self.save_checkpoint(
                    os.path.join(self.checkpoint_dir, f"{experiment_name}_best.pt"),
                    epoch,
                    val_metrics,
                )

        return history

    def save_checkpoint(self, path: str, epoch: int, metrics: Dict) -> None:
        """Save model checkpoint."""
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
                "variant": self.variant,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> Dict:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint
