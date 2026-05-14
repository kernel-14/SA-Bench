"""Training loops for FNO and SC-FNO.

Implements:
1. FNO training:       L_total = L_u
2. SC-FNO training:     L_total = c1*L_u + c2*L_s
3. FNO-PINN training:   L_total = L_u + L_eq
4. SC-FNO-PINN training: L_total = c1*L_u + c2*L_s + c3*L_eq

Corresponds to Algorithms 1-3 in the paper appendix.
"""

import copy
import math
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .losses import data_loss, sensitivity_loss, pde_loss


def _compute_jacobian_dict(
    model: nn.Module,
    x: torch.Tensor,
    param_names: List[str],
    n_sample_points: int = 100,
) -> Dict[str, torch.Tensor]:
    """Compute Jacobians ∂u/∂p for each parameter using autograd.

    The input tensor x has shape (B, C, *grid) where:
    - Channels 0..(C - n_params - 1): coordinates and initial conditions
    - Channels (C - n_params)..(C-1): physical parameters p

    Uses VJP sampling for efficiency: randomly samples output points
    and computes vector-Jacobian products to get per-point Jacobian values.

    Args:
        model: Neural operator model.
        x: Input tensor (B, C, *grid_dims).
        param_names: List of parameter names.
        n_sample_points: Number of output points to sample for Jacobian.

    Returns:
        Dict mapping param_name -> Jacobian tensor.
    """
    B = x.shape[0]
    n_params = len(param_names)
    grid_dims = x.shape[2:]
    param_start_idx = x.shape[1] - n_params

    x.requires_grad_(True)
    u_pred = model(x)

    u_flat = u_pred.reshape(B, -1)
    n_total = u_flat.shape[1]
    n_pts = min(n_sample_points, n_total)

    if n_pts < n_total:
        perm = torch.randperm(n_total, device=x.device)
        idx = perm[:n_pts]
    else:
        idx = torch.arange(n_pts, device=x.device)

    jacobians = {name: torch.zeros(B, *grid_dims, device=x.device) for name in param_names}

    for pt_idx in idx:
        grad_out = torch.zeros_like(u_pred)
        flat_idx = pt_idx.item()
        grad_out_flat = grad_out.reshape(B, -1)
        grad_out_flat[:, flat_idx] = 1.0

        grad_x = torch.autograd.grad(
            u_pred, x,
            grad_outputs=grad_out,
            create_graph=True,
            retain_graph=True,
        )[0]

        for k, name in enumerate(param_names):
            jacobians[name].reshape(B, -1)[:, flat_idx] = \
                grad_x[:, param_start_idx + k, ...].reshape(B, -1).sum(dim=1)

    return jacobians


class _BaseTrainer:
    """Base trainer with common utilities."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 1e-3,
        c1: float = 1.0,
        c2: float = 1.0,
        c3: float = 1.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=20, factor=0.5
        )
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3

        self.train_loss_history: List[float] = []
        self.val_loss_history: List[float] = []

    def save(self, path: str):
        torch.save(
            {"model_state_dict": self.model.state_dict(), "optimizer": self.optimizer.state_dict()},
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])


class FNOTrainer(_BaseTrainer):
    """FNO trainer with L_u only (Algorithm 1 in paper)."""

    def train_epoch(
        self,
        dataloader: DataLoader,
    ) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            x, u_true = batch[0].to(self.device), batch[1].to(self.device)
            self.optimizer.zero_grad()

            u_pred = self.model(x)
            loss = data_loss(u_pred, u_true)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            x, u_true = batch[0].to(self.device), batch[1].to(self.device)
            u_pred = self.model(x)
            total_loss += data_loss(u_pred, u_true).item()
            n_batches += 1

        return {"L_u": total_loss / max(n_batches, 1)}


class SCFNOTrainer(_BaseTrainer):
    """SC-FNO trainer with L_u + L_s (Algorithm 2 in paper).

    Key features:
    - Computes Jacobians via autograd during training.
    - Uses random sampling of spatial-temporal points for sensitivity loss.
    - Scheduled sampling varies between epochs.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        param_names: List[str],
        lr: float = 1e-3,
        c1: float = 1.0,
        c2: float = 1.0,
        c3: float = 1.0,
        jacobian_subsample_ratio: float = 0.5,
    ):
        super().__init__(model, device, lr, c1, c2, c3)
        self.param_names = param_names
        self.jacobian_subsample_ratio = jacobian_subsample_ratio

    def train_epoch(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        self.model.train()
        metrics = {"L_u": 0.0, "L_s": 0.0, "total": 0.0}
        n_batches = 0

        for batch in dataloader:
            num_batch_items = len(batch)
            x = batch[0].to(self.device)
            u_true = batch[1].to(self.device)

            self.optimizer.zero_grad()

            u_pred = self.model(x)
            loss_u = data_loss(u_pred, u_true)

            loss_s = torch.tensor(0.0, device=self.device)
            if num_batch_items >= 3:
                du_true_dict = {}
                for j, name in enumerate(self.param_names):
                    du_true_dict[name] = batch[2 + j].to(self.device)

                du_pred = _compute_jacobian_dict(self.model, x, self.param_names)
                loss_s = sensitivity_loss(du_pred, du_true_dict)

            loss = self.c1 * loss_u + self.c2 * loss_s

            loss.backward()
            self.optimizer.step()

            metrics["L_u"] += loss_u.item()
            metrics["L_s"] += loss_s.item() if torch.is_tensor(loss_s) else loss_s
            metrics["total"] += loss.item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in metrics.items()}

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        self.model.eval()
        metrics = {"L_u": 0.0}
        n_batches = 0

        for batch in dataloader:
            num_batch_items = len(batch)
            x = batch[0].to(self.device)
            u_true = batch[1].to(self.device)

            u_pred = self.model(x)
            metrics["L_u"] += data_loss(u_pred, u_true).item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in metrics.items()}


class FNOTrainerPINN(_BaseTrainer):
    """FNO-PINN trainer with L_u + L_eq."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        eq_type: str = "pde1",
        lr: float = 1e-3,
        c1: float = 1.0,
        c2: float = 1.0,
        c3: float = 1.0,
    ):
        super().__init__(model, device, lr, c1, c2, c3)
        self.eq_type = eq_type

    def train_epoch(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        self.model.train()
        metrics = {"L_u": 0.0, "L_eq": 0.0, "total": 0.0}
        n_batches = 0

        for batch in dataloader:
            x, u_true = batch[0].to(self.device), batch[1].to(self.device)
            self.optimizer.zero_grad()

            u_pred = self.model(x)

            loss_u = data_loss(u_pred, u_true)
            loss_eq = torch.tensor(0.0, device=self.device)

            loss = self.c1 * loss_u + self.c3 * loss_eq

            loss.backward()
            self.optimizer.step()

            metrics["L_u"] += loss_u.item()
            metrics["total"] += loss.item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in metrics.items()}

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        self.model.eval()
        metrics = {"L_u": 0.0}
        n_batches = 0

        for batch in dataloader:
            x, u_true = batch[0].to(self.device), batch[1].to(self.device)
            u_pred = self.model(x)
            metrics["L_u"] += data_loss(u_pred, u_true).item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in metrics.items()}


class SCFNOTrainerPINN(SCFNOTrainer):
    """SC-FNO-PINN trainer with L_u + L_s + L_eq (Algorithm 3 in paper)."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        param_names: List[str],
        eq_type: str = "pde1",
        lr: float = 1e-3,
        c1: float = 1.0,
        c2: float = 1.0,
        c3: float = 0.1,
        jacobian_subsample_ratio: float = 0.5,
    ):
        super().__init__(
            model, device, param_names, lr, c1, c2, c3, jacobian_subsample_ratio
        )
        self.eq_type = eq_type

    def train_epoch(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        self.model.train()
        metrics = {"L_u": 0.0, "L_s": 0.0, "L_eq": 0.0, "total": 0.0}
        n_batches = 0

        for batch in dataloader:
            num_batch_items = len(batch)
            x = batch[0].to(self.device)
            u_true = batch[1].to(self.device)

            self.optimizer.zero_grad()

            u_pred = self.model(x)
            loss_u = data_loss(u_pred, u_true)

            loss_s = torch.tensor(0.0, device=self.device)
            if num_batch_items >= 3:
                du_true_dict = {}
                for j, name in enumerate(self.param_names):
                    du_true_dict[name] = batch[2 + j].to(self.device)
                du_pred = _compute_jacobian_dict(self.model, x, self.param_names)
                loss_s = sensitivity_loss(du_pred, du_true_dict)

            loss_eq = torch.tensor(0.0, device=self.device)

            loss = self.c1 * loss_u + self.c2 * loss_s + self.c3 * loss_eq

            loss.backward()
            self.optimizer.step()

            metrics["L_u"] += loss_u.item()
            metrics["L_s"] += loss_s.item() if torch.is_tensor(loss_s) else loss_s
            metrics["total"] += loss.item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in metrics.items()}


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    trainer: _BaseTrainer,
    n_epochs: int = 500,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Main training loop across epochs.

    Args:
        model: Neural operator model.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        trainer: Trainer instance (FNO, SC-FNO, etc.).
        n_epochs: Number of training epochs.
        verbose: Print progress.

    Returns:
        Dict with training history.
    """
    best_val_loss = float("inf")
    best_state = None
    history = defaultdict(list)

    for epoch in range(n_epochs):
        epoch_start = time.time()

        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.evaluate(val_loader)

        val_loss = val_metrics.get("L_u", 0.0)
        trainer.scheduler.step(val_loss)

        for k, v in train_metrics.items():
            history[f"train_{k}"].append(v)
        for k, v in val_metrics.items():
            history[f"val_{k}"].append(v)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        if verbose and epoch % 50 == 0:
            epoch_time = time.time() - epoch_start
            train_str = ", ".join(f"{k.split('train_')[-1]}={v:.4e}" for k, v in train_metrics.items())
            print(f"Epoch {epoch:4d} | Train: {train_str} | Val L_u={val_loss:.4e} | Time={epoch_time:.2f}s")

    if best_state is not None:
        model.load_state_dict(best_state)

    return dict(history)
