## Code: training/trainer.py

```python
## training/trainer.py
"""Unified training loop for all four SC-FNO model variants.

Implements the Trainer class that orchestrates training for:
  - FNO:          data loss L_u only (Algorithm 1 in paper)
  - SC-FNO:       L_u + sensitivity loss L_s (Algorithm 2)
  - FNO-PINN:     L_u + equation loss L_Eq
  - SC-FNO-PINN:  L_u + L_s + L_Eq (Algorithm 3)

The FNO architecture is identical across all variants — only the loss
configuration differs, determined by model.variant.

Hyperparameters from Table C.7/C.8 of the SC-FNO paper:
  - lr = 0.001 (all cases)
  - n_epochs = 500 (all cases)
  - batch_size: 16 (ODEs), 4 (PDE1/2/3), 1 (PDE2 Zoned, PDE4)
  - scheduler: cosine annealing over 500 epochs [DEFAULT]

References:
    - SC-FNO paper Section 2.4: Implementation Details
    - SC-FNO paper Appendix A: Pseudocodes for training loops
    - SC-FNO paper Table C.7: Hyperparameters
    - SC-FNO paper Table C.8: Training times per epoch
    - config.yaml: training.*, pinn.*, training.loss_weights.*
"""

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from losses.data_loss import DataLoss
from losses.pinn_loss import PINNLoss
from losses.sensitivity_loss import SensitivityLoss
from utils.logger import Logger


class Trainer:
    """Unified training loop for all four SC-FNO model variants.

    Reads model.variant to determine which loss terms to activate:
      - 'fno':          L_u only
      - 'sc_fno':       L_u + L_s
      - 'fno_pinn':     L_u + L_Eq
      - 'sc_fno_pinn':  L_u + L_s + L_Eq

    The FNO architecture is identical across all variants. Only the loss
    configuration differs.

    Attributes:
        model: The FNO model instance with a .variant string attribute.
        train_loader: DataLoader yielding batches with keys 'params', 'u0',
                      'u_true', 'jacobians', 'coords'.
        val_loader: DataLoader with the same structure (jacobians optional).
        cfg: The full master configuration dictionary from ConfigLoader.
        optimizer: Adam optimizer initialized with lr from config.
        scheduler: CosineAnnealingLR scheduler over n_epochs.
        data_loss: DataLoss instance (always active).
        sensitivity_loss: SensitivityLoss instance (active if use_sensitivity).
        pinn_loss: PINNLoss instance (active if use_pinn).
        loss_weights: Dict with keys 'c1', 'c2', 'c3' from config.
        use_sensitivity: True if model.variant contains 'sc'.
        use_pinn: True if model.variant contains 'pinn'.
        device: torch.device for tensor operations.
        logger: Logger instance for TensorBoard and file logging.
        best_val_loss: Best validation loss seen so far (for checkpointing).
        checkpoint_dir: Directory for saving checkpoints.
        current_epoch: Current epoch counter (updated during training).
        training_history: Dict tracking per-epoch losses and learning rates.

    Example:
        >>> from models.sc_fno import build_model
        >>> from data.dataset import SCFNODataset
        >>> from torch.utils.data import DataLoader
        >>> from utils.config_loader import ConfigLoader
        >>>
        >>> cfg_loader = ConfigLoader('config.yaml')
        >>> cfg = cfg_loader.cfg
        >>> # Merge equation sub-config with global defaults
        >>> eq_cfg = {**cfg['pde1'], 'variant': 'sc_fno', 'equation': 'pde1',
        ...           'model': {**cfg['model'], **cfg['pde1'].get('model', {})}}
        >>> model = build_model(eq_cfg)
        >>> train_ds = SCFNODataset('data/datasets/pde1.pt', 'train', use_jacobian=True)
        >>> val_ds   = SCFNODataset('data/datasets/pde1.pt', 'val',   use_jacobian=False)
        >>> train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
        >>> val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False)
        >>> trainer = Trainer(model, train_loader, val_loader, cfg)
        >>> history = trainer.train(n_epochs=500)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict,
        run_name: str = "default_run",
    ) -> None:
        """Initializes the Trainer.

        Reads all hyperparameters from cfg — no hardcoded values. Builds the
        optimizer, scheduler, and loss components based on model.variant and
        the equation-specific configuration.

        Args:
            model: The FNO model instance. Must have a .variant attribute
                   (set by build_model in sc_fno.py) and a .equation attribute.
                   Must already be on the target device.
            train_loader: DataLoader for the training split. Batch dicts must
                          contain 'params', 'u0', 'u_true', 'coords', and
                          optionally 'jacobians' (required for SC variants).
            val_loader: DataLoader for the validation split. 'jacobians' key
                        is not required (validation uses L_u only).
            cfg: The full master configuration dictionary loaded from
                 config.yaml. Must contain top-level keys:
                   - 'training': dict with 'lr', 'n_epochs', 'optimizer',
                     'scheduler', 'loss_weights', 'sensitivity_sample_fraction',
                     'sensitivity_sample_max'
                   - 'pinn': dict with 'n_colloc', 'alpha_weight'
                   - 'device': str
                   - 'log_dir': str
                   - 'checkpoint_dir': str
                 Also reads equation-specific sub-configs for batch_size and
                 discretization parameters.
            run_name: Identifier for this training run, used for logging and
                      checkpoint naming. Typically assembled by main.py as
                      "{equation}_{variant}". Default 'default_run'.
        """
        self.model: nn.Module = model
        self.train_loader: DataLoader = train_loader
        self.val_loader: DataLoader = val_loader
        self.cfg: dict = cfg
        self.run_name: str = run_name

        # ------------------------------------------------------------------
        # Device setup
        # ------------------------------------------------------------------
        device_str: str = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.device: torch.device = torch.device(device_str)

        # Move model to device if not already there.
        self.model = self.model.to(self.device)

        # ------------------------------------------------------------------
        # Read global training hyperparameters from config.yaml.
        # ------------------------------------------------------------------
        training_cfg: dict = cfg.get("training", {})

        self.n_epochs: int = int(training_cfg.get("n_epochs", 500))
        lr: float = float(training_cfg.get("lr", 0.001))
        optimizer_name: str = str(training_cfg.get("optimizer", "adam")).lower()
        scheduler_name: str = str(training_cfg.get("scheduler", "cosine")).lower()

        # Loss weights (c1, c2, c3) — not specified in paper, default 1.0.
        loss_weights_cfg: dict = training_cfg.get("loss_weights", {})
        self.loss_weights: Dict[str, float] = {
            "c1": float(loss_weights_cfg.get("c1", 1.0)),
            "c2": float(loss_weights_cfg.get("c2", 1.0)),
            "c3": float(loss_weights_cfg.get("c3", 1.0)),
        }

        # Sensitivity sampling parameters.
        self.sensitivity_sample_fraction: float = float(
            training_cfg.get("sensitivity_sample_fraction", 0.10)
        )
        self.sensitivity_sample_max: int = int(
            training_cfg.get("sensitivity_sample_max", 256)
        )

        # ------------------------------------------------------------------
        # Determine variant flags from model.variant.
        # ------------------------------------------------------------------
        variant: str = str(getattr(model, "variant", "fno")).lower()
        self.use_sensitivity: bool = "sc" in variant
        self.use_pinn: bool = "pinn" in variant

        # ------------------------------------------------------------------
        # Optimizer initialization.
        # ------------------------------------------------------------------
        if optimizer_name == "adam":
            self.optimizer: torch.optim.Optimizer = torch.optim.Adam(
                self.model.parameters(), lr=lr
            )
        elif optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=lr
            )
        elif optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=lr, momentum=0.9
            )
        else:
            # Default to Adam for any unrecognized optimizer name.
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=lr
            )

        # ------------------------------------------------------------------
        # Learning rate scheduler initialization.
        # ------------------------------------------------------------------
        if scheduler_name == "cosine":
            self.scheduler: torch.optim.lr_scheduler._LRScheduler = (
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.n_epochs,
                    eta_min=lr * 1e-3,  # Minimum LR = 0.1% of initial LR.
                )
            )
        elif scheduler_name == "step":
            # Step decay at 60% and 80% of total epochs.
            milestones: List[int] = [
                int(0.6 * self.n_epochs),
                int(0.8 * self.n_epochs),
            ]
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=milestones,
                gamma=0.1,
            )
        elif scheduler_name == "none":
            # No-op scheduler — constant learning rate.
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda epoch: 1.0,
            )
        else:
            # Default to cosine annealing.
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.n_epochs,
                eta_min=lr * 1e-3,
            )

        # ------------------------------------------------------------------
        # Loss component initialization.
        # ------------------------------------------------------------------

        # Data loss L_u — always active.
        self.data_loss: DataLoss = DataLoss()

        # Sensitivity loss L_s — active for sc_fno and sc_fno_pinn.
        self.sensitivity_loss: Optional[SensitivityLoss] = None
        if self.use_sensitivity:
            n_sample_points: int = self._compute_n_sample_points()
            self.sensitivity_loss = SensitivityLoss(
                n_sample_points=n_sample_points
            )

        # PINN equation loss L_Eq — active for fno_pinn and sc_fno_pinn.
        self.pinn_loss: Optional[PINNLoss] = None
        if self.use_pinn:
            pinn_cfg: dict = cfg.get("pinn", {})
            equation_type: str = str(cfg.get("equation", "pde1")).lower()
            n_colloc: int = int(pinn_cfg.get("n_colloc", 256))
            alpha_weight: float = float(pinn_cfg.get("alpha_weight", 1.0))
            self.pinn_loss = PINNLoss(
                equation_type=equation_type,
                n_colloc=n_colloc,
                alpha_weight=alpha_weight,
            )

        # ------------------------------------------------------------------
        # Checkpoint and logging setup.
        # ------------------------------------------------------------------
        self.checkpoint_dir: str = str(cfg.get("checkpoint_dir", "outputs/checkpoints"))
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        log_dir: str = str(cfg.get("log_dir", "outputs/logs"))
        self.logger: Logger = Logger(log_dir=log_dir, run_name=run_name)

        # ------------------------------------------------------------------
        # Training state.
        # ------------------------------------------------------------------
        self.best_val_loss: float = float("inf")
        self.current_epoch: int = 0

        # Training history — granular breakdown per loss component.
        self.training_history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_L_u": [],
            "train_L_s": [],
            "train_L_eq": [],
            "lr": [],
        }

        # ------------------------------------------------------------------
        # Print initialization summary.
        # ------------------------------------------------------------------
        print(
            f"[Trainer] Initialized for variant='{variant}' | "
            f"equation='{cfg.get('equation', 'unknown')}' | "
            f"device={self.device} | "
            f"lr={lr} | "
            f"n_epochs={self.n_epochs} | "
            f"use_sensitivity={self.use_sensitivity} | "
            f"use_pinn={self.use_pinn} | "
            f"loss_weights={self.loss_weights}"
        )
        if self.use_sensitivity and self.sensitivity_loss is not None:
            print(
                f"[Trainer]   SensitivityLoss: "
                f"n_sample_points={self.sensitivity_loss.n_sample_points}"
            )
        if self.use_pinn and self.pinn_loss is not None:
            print(
                f"[Trainer]   PINNLoss: "
                f"equation='{self.pinn_loss.equation_type}' | "
                f"n_colloc={self.pinn_loss.n_colloc} | "
                f"alpha_weight={self.pinn_loss.alpha_weight}"
            )

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, n_epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """Runs the full training loop for n_epochs epochs.

        Implements the training loops described in Algorithms 1, 2, and 3
        of the SC-FNO paper. Saves the best checkpoint (lowest validation
        L_u) and periodic checkpoints every 50 epochs.

        Args:
            n_epochs: Number of epochs to train. If None, uses
                      cfg['training']['n_epochs'] (500 from Table C.7).
                      Passing a value here overrides the config value,
                      which is useful for quick tests and data scaling
                      experiments.

        Returns:
            self.training_history: Dict with keys:
              - 'train_loss': list of mean total training loss per epoch
              - 'val_loss': list of mean validation L_u per epoch
              - 'train_L_u': list of mean L_u component per epoch
              - 'train_L_s': list of mean L_s component per epoch (0.0 if inactive)
              - 'train_L_eq': list of mean L_eq component per epoch (0.0 if inactive)
              - 'lr': list of learning rate per epoch

        Example:
            >>> history = trainer.train(n_epochs=500)
            >>> len(history['train_loss'])  # 500
            >>> min(history['val_loss'])    # best validation loss
        """
        if n_epochs is None:
            n_epochs = self.n_epochs

        print(
            f"[Trainer] Starting training: {n_epochs} epochs | "
            f"variant='{getattr(self.model, 'variant', 'fno')}'"
        )

        # Progress bar over epochs.
        epoch_bar = tqdm(
            range(1, n_epochs + 1),
            desc=f"Training [{getattr(self.model, 'variant', 'fno')}]",
            unit="epoch",
        )

        for epoch in epoch_bar:
            self.current_epoch = epoch

            # ------------------------------------------------------------------
            # Training phase.
            # ------------------------------------------------------------------
            train_metrics: Dict[str, float] = self._train_epoch()
            train_loss: float = train_metrics["total"]
            train_L_u: float = train_metrics["L_u"]
            train_L_s: float = train_metrics["L_s"]
            train_L_eq: float = train_metrics["L_eq"]

            # ------------------------------------------------------------------
            # Validation phase.
            # ------------------------------------------------------------------
            val_loss: float = self._validate()

            # ------------------------------------------------------------------
            # Scheduler step (after validation, before logging).
            # CosineAnnealingLR expects one step per epoch.
            # ------------------------------------------------------------------
            self.scheduler.step()

            # Current learning rate (first param group).
            current_lr: float = float(
                self.optimizer.param_groups[0]["lr"]
            )

            # ------------------------------------------------------------------
            # Logging.
            # ------------------------------------------------------------------
            self.logger.log_scalar("train/loss_total", train_loss, epoch)
            self.logger.log_scalar("train/loss_u", train_L_u, epoch)
            self.logger.log_scalar("val/loss_u", val_loss, epoch)
            self.logger.log_scalar("train/lr", current_lr, epoch)

            if self.use_sensitivity:
                self.logger.log_scalar("train/loss_s", train_L_s, epoch)
            if self.use_pinn:
                self.logger.log_scalar("train/loss_eq", train_L_eq, epoch)

            # ------------------------------------------------------------------
            # Update training history.
            # ------------------------------------------------------------------
            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["train_L_u"].append(train_L_u)
            self.training_history["train_L_s"].append(train_L_s)
            self.training_history["train_L_eq"].append(train_L_eq)
            self.training_history["lr"].append(current_lr)

            # ------------------------------------------------------------------
            # Checkpoint saving.
            # ------------------------------------------------------------------
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(best=True)

            if epoch % 50 == 0:
                self.save_checkpoint(best=False)

            # ------------------------------------------------------------------
            # Update progress bar postfix.
            # ------------------------------------------------------------------
            postfix: Dict[str, str] = {
                "train": f"{train_loss:.4f}",
                "val": f"{val_loss:.4f}",
                "lr": f"{current_lr:.2e}",
            }
            if self.use_sensitivity:
                postfix["L_s"] = f"{train_L_s:.4f}"
            epoch_bar.set_postfix(postfix)

        # ------------------------------------------------------------------
        # Save final checkpoint and results.
        # ------------------------------------------------------------------
        self.save_checkpoint(best=False)
        self.logger.save_results(self.training_history, "training_history.json")

        print(
            f"[Trainer] Training complete. "
            f"Best val loss: {self.best_val_loss:.6f} | "
            f"Final train loss: {self.training_history['train_loss'][-1]:.6f}"
        )

        return self.training_history

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def _train_epoch(self) -> Dict[str, float]:
        """Runs one training epoch over all batches in train_loader.

        Sets model to train mode, iterates over batches, computes the total
        loss (L_u + optional L_s + optional L_eq), backpropagates, and
        updates model weights.

        Returns:
            Dict with keys 'total', 'L_u', 'L_s', 'L_eq' containing the
            mean loss values over all batches in this epoch. All values are
            Python floats.
        """
        self.model.train()

        total_loss_sum: float = 0.0
        L_u_sum: float = 0.0
        L_s_sum: float = 0.0
        L_eq_sum: float = 0.0
        n_batches: int = 0

        for batch in self.train_loader:
            # ------------------------------------------------------------------
            # Move batch tensors to device.
            # Handle None values (e.g., jacobians=None for FNO/FNO-PINN).
            # ------------------------------------------------------------------
            batch_dev: Dict[str, Any] = self._move_batch_to_device(batch)

            # ------------------------------------------------------------------
            # Zero gradients before computing loss.
            # ------------------------------------------------------------------
            self.optimizer.zero_grad()

            # ------------------------------------------------------------------
            # Compute total loss and component breakdown.
            # ------------------------------------------------------------------
            total_loss, loss_components = self._compute_total_loss(batch_dev)

            # ------------------------------------------------------------------
            # Check for NaN/Inf loss — log warning and skip batch.
            # ------------------------------------------------------------------
            if not torch.isfinite(total_loss):
                print(
                    f"\n[Trainer] WARNING: Non-finite loss ({total_loss.item():.4f}) "
                    f"at epoch {self.current_epoch}. Skipping batch."
                )
                # Zero gradients to avoid corrupting model weights.
                self.optimizer.zero_grad()
                continue

            # ------------------------------------------------------------------
            # Backpropagation.
            # ------------------------------------------------------------------
            total_loss.backward()

            # Optional gradient clipping for numerical stability.
            # Not specified in the paper — using a conservative max_norm=1.0.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # ------------------------------------------------------------------
            # Optimizer step.
            # ------------------------------------------------------------------
            self.optimizer.step()

            # ------------------------------------------------------------------
            # Accumulate loss values.
            # ------------------------------------------------------------------
            total_loss_sum += float(total_loss.item())
            L_u_sum += float(loss_components["L_u"])
            L_s_sum += float(loss_components["L_s"])
            L_eq_sum += float(loss_components["L_eq"])
            n_batches += 1

        # Guard against empty loader.
        if n_batches == 0:
            return {"total": 0.0, "L_u": 0.0, "L_s": 0.0, "L_eq": 0.0}

        return {
            "total": total_loss_sum / n_batches,
            "L_u": L_u_sum / n_batches,
            "L_s": L_s_sum / n_batches,
            "L_eq": L_eq_sum / n_batches,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> float:
        """Runs validation over all batches in val_loader.

        Uses only L_u (relative L²) for validation regardless of variant.
        This provides a fair comparison across variants on the primary task
        of solution path prediction.

        Sets model to eval mode and uses torch.no_grad() to avoid building
        computation graphs during validation.

        Returns:
            Mean validation L_u (relative L²) over all batches as a Python
            float. Returns 0.0 if val_loader is empty.
        """
        self.model.eval()

        total_val_loss: float = 0.0
        n_batches: int = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch_dev: Dict[str, Any] = self._move_batch_to_device(batch)

                # Forward pass — no gradient tracking needed.
                u_pred: torch.Tensor = self.model(
                    batch_dev["params"],
                    batch_dev["u0"],
                    batch_dev["coords"],
                )

                # Validation loss: L_u only.
                L_u: torch.Tensor = self.data_loss(u_pred, batch_dev["u_true"])

                if torch.isfinite(L_u):
                    total_val_loss += float(L_u.item())
                    n_batches += 1

        # Restore model to train mode.
        self.model.train()

        if n_batches == 0:
            return 0.0

        return total_val_loss / n_batches

    # ------------------------------------------------------------------
    # Total loss computation
    # ------------------------------------------------------------------

    def _compute_total_loss(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes the total loss for one mini-batch.

        Assembles L_total = c1*L_u + c2*L_s + c3*L_eq based on model.variant.
        Returns both the differentiable total loss tensor and a dict of
        component values (as Python floats) for logging.

        The data loss L_u uses a forward pass without gradient tracking on
        params (standard FNO training). The sensitivity loss L_s uses a
        separate forward pass with params.requires_grad=True (handled
        internally by SensitivityLoss). The PINN loss L_eq uses a forward
        pass with coords.requires_grad=True (handled internally by PINNLoss).

        Args:
            batch: Dict with keys 'params', 'u0', 'u_true', 'coords', and
                   optionally 'jacobians'. All tensors must already be on
                   self.device.

        Returns:
            Tuple of:
              - total_loss: Scalar tensor, differentiable w.r.t. model weights.
              - loss_components: Dict with keys 'L_u', 'L_s', 'L_eq' as
                                 Python floats for logging.

        Raises:
            ValueError: If use_sensitivity=True but batch['jacobians'] is None.
        """
        params: torch.Tensor = batch["params"]
        u0: torch.Tensor = batch["u0"]
        u_true: torch.Tensor = batch["u_true"]
        coords: torch.Tensor = batch["coords"]
        jacobians: Optional[torch.Tensor] = batch.get("jacobians", None)

        # ------------------------------------------------------------------
        # Step 1: Data loss L_u (all variants).
        # Standard forward pass — no gradient tracking on params needed here.
        # ------------------------------------------------------------------
        u_pred: torch.Tensor = self.model(params, u0, coords)
        L_u: torch.Tensor = self.data_loss(u_pred, u_true)

        total_loss: torch.Tensor = self.loss_weights["c1"] * L_u

        # Track component values for logging.
        L_u_val: float = float(L_u.item())
        L_s_val: float = 0.0
        L_eq_val: float = 0.0

        # ------------------------------------------------------------------
        # Step 2: Sensitivity loss L_s (sc_fno, sc_fno_pinn).
        # SensitivityLoss handles its own forward pass with params.requires_grad.
        # ------------------------------------------------------------------
        if self.use_sensitivity and self.sensitivity_loss is not None:
            if jacobians is None:
                raise ValueError(
                    "Trainer._compute_total_loss: batch['jacobians'] is None "
                    "but use_sensitivity=True. Load the dataset with "
                    "use_jacobian=True for SC-FNO variants."
                )

            L_s: torch.Tensor = self.sensitivity_loss(
                model=self.model,
                params=params,
                u0=u0,
                coords=coords,
                j_true=jacobians,
            )

            total_loss = total_loss + self.loss_weights["c2"] * L_s
            L_s_val = float(L_s.item())

        # ------------------------------------------------------------------
        # Step 3: PINN equation loss L_eq (fno_pinn, sc_fno_pinn).
        # PINNLoss handles its own forward pass with coords.requires_grad.
        # ------------------------------------------------------------------
        if self.use_pinn and self.pinn_loss is not None:
            L_eq: torch.Tensor = self.pinn_loss(
                model=self.model,
                params=params,
                u0=u0,
                coords=coords,
            )

            total_loss = total_loss + self.loss_weights["c3"] * L_eq
            L_eq_val = float(L_eq.item())

        loss_components: Dict[str, float] = {
            "L_u": L_u_val,
            "L_s": L_s_val,
            "L_eq": L_eq_val,
        }

        return total_loss, loss_components

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def save_checkpoint(self, best: bool = False) -> None:
        """Saves a training checkpoint to self.checkpoint_dir.

        Saves model state, optimizer state, scheduler state, training history,
        and configuration so that training can be resumed or the model can be
        loaded for evaluation without the original config file.

        Args:
            best: If True, saves as 'best_model.pt' (overwrites previous best).
                  If False, saves as 'checkpoint_epoch_{current_epoch}.pt'.
                  Default False.
        """
        checkpoint: Dict[str, Any] = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "training_history": self.training_history,
            "cfg": self.cfg,
            "variant": getattr(self.model, "variant", "fno"),
            "run_name": self.run_name,
        }

        if best:
            filename: str = "best_model.pt"
        else:
            filename = f"checkpoint_epoch_{self.current_epoch:04d}.pt"

        save_path: str = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, save_path)

        if best:
            print(
                f"[Trainer] Best checkpoint saved: {save_path} "
                f"(val_loss={self.best_val_loss:.6f})"
            )

    def load_checkpoint(self, path: str) -> None:
        """Loads a training checkpoint from disk.

        Restores model weights, optimizer state, scheduler state, and
        training history. Handles loading a GPU-trained checkpoint on CPU
        and vice versa via map_location.

        Args:
            path: Full path to the .pt checkpoint file. Must exist.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            RuntimeError: If the checkpoint is incompatible with the current
                          model architecture.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Checkpoint file not found: '{path}'. "
                f"Ensure the file exists before calling load