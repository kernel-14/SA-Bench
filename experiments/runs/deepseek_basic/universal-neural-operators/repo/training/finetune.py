"""Fine-tuning framework for pretrained neural operators.

As described in Section 3 ("Pre-training and fine-tuning"):
In the fine-tuning stage we fix the parameters θ_F (core operator) to both
highlight the generalizing properties of the operator and to reduce training
costs: only the new adapter parameters (θ_P_ft, θ_L_ft) are trained.

This significantly reduces the number of trainable parameters and speeds up
training, as reflected in the paper's results (Tables 1 & 2).
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable
import time
import copy

from .metrics import compute_metrics


class FineTuner:
    """Handles fine-tuning of pretrained neural operators on new physics problems.

    During fine-tuning, the core operator parameters are frozen, and only the
    lifting and projection adapter parameters are trained. This matches the
    adapter-based approach described in the paper, analogous to LoRA [19].
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        freeze_core: bool = True,
    ):
        """
        Args:
            model: Pretrained neural operator model
            device: Device to train on
            learning_rate: Learning rate for fine-tuning
            weight_decay: Weight decay for optimizer
            freeze_core: Whether to freeze core operator parameters
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.freeze_core = freeze_core

        if freeze_core:
            self._freeze_core_params()

        # Collect only trainable adapter parameters
        trainable_params = self._get_adapter_params()
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=100,
            gamma=0.5,
        )

    def _freeze_core_params(self):
        """Freeze core operator parameters, keep adapter parameters trainable."""
        for name, param in self.model.named_parameters():
            is_adapter = any(x in name for x in ['lifting', 'projection', 'proj', 'lift'])
            if not is_adapter:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def _get_adapter_params(self):
        """Get only adapter (lifting + projection) parameters."""
        params = []
        if hasattr(self.model, 'get_lifting_params'):
            params.extend(self.model.get_lifting_params())
        if hasattr(self.model, 'get_projection_params'):
            params.extend(self.model.get_projection_params())

        # Fallback: filter by name
        if not params:
            for name, param in self.model.named_parameters():
                if any(x in name for x in ['lifting', 'projection', 'proj', 'lift']):
                    params.append(param)
        return params

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Run one fine-tuning epoch.

        Args:
            train_loader: DataLoader for the target physics problem

        Returns:
            Dict with training loss
        """
        self.model.train()
        total_loss = 0.0
        total_batches = 0

        for batch in train_loader:
            if isinstance(batch, tuple) and len(batch) == 2:
                x, y = batch
                grid = None
            elif isinstance(batch, tuple) and len(batch) == 3:
                x, y, grid = batch
            else:
                x, y = batch['input'], batch['target']
                grid = batch.get('grid', None)

            x = x.to(self.device).float()
            y = y.to(self.device).float()
            if grid is not None:
                grid = grid.to(self.device).float()

            self.optimizer.zero_grad()

            pred = self.model(x, grid=grid)
            loss = nn.functional.mse_loss(pred, y)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        self.scheduler.step()
        avg_loss = total_loss / max(total_batches, 1)
        return {'finetune_loss': avg_loss}

    def evaluate(self, loader: DataLoader) -> Dict:
        """Evaluate the fine-tuned model.

        Returns:
            Dict with mse, nmae, nmae_pct metrics
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, tuple) and len(batch) == 2:
                    x, y = batch
                    grid = None
                elif isinstance(batch, tuple) and len(batch) == 3:
                    x, y, grid = batch
                else:
                    x, y = batch['input'], batch['target']
                    grid = batch.get('grid', None)

                x = x.to(self.device).float()
                y = y.to(self.device).float()
                if grid is not None:
                    grid = grid.to(self.device).float()

                pred = self.model(x, grid=grid)
                all_preds.append(pred)
                all_targets.append(y)

        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        return compute_metrics(preds, targets)

    def save_checkpoint(self, path: str):
        """Save fine-tuning state."""
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        torch.save(state, path)

    def load_checkpoint(self, path: str):
        """Load fine-tuning state."""
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])


def train_from_scratch(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 500,
    lr: float = 1e-3,
    device: str = 'cuda',
) -> Dict:
    """Train a model from scratch (all parameters trainable) for comparison.

    This implements the "from scratch" baseline experiments from the paper.
    """
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    best_metrics = {'mse': float('inf'), 'nmae_pct': float('inf')}
    times_per_epoch = []

    for epoch in range(epochs):
        t0 = time.time()

        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            if isinstance(batch, tuple) and len(batch) == 2:
                x, y = batch
                grid = None
            elif isinstance(batch, tuple) and len(batch) == 3:
                x, y, grid = batch
            else:
                x, y = batch['input'], batch['target']
                grid = batch.get('grid', None)

            x = x.to(device).float()
            y = y.to(device).float()
            if grid is not None:
                grid = grid.to(device).float()

            optimizer.zero_grad()
            pred = model(x, grid=grid)
            loss = nn.functional.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        epoch_time = time.time() - t0
        times_per_epoch.append(epoch_time)

        # Validation
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, tuple) and len(batch) == 2:
                    x, y = batch
                    grid = None
                elif isinstance(batch, tuple) and len(batch) == 3:
                    x, y, grid = batch
                else:
                    x, y = batch['input'], batch['target']
                    grid = batch.get('grid', None)
                x = x.to(device).float()
                y = y.to(device).float()
                if grid is not None:
                    grid = grid.to(device).float()
                pred = model(x, grid=grid)
                all_preds.append(pred)
                all_targets.append(y)

        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        metrics = compute_metrics(preds, targets)

        if metrics['mse'] < best_metrics['mse']:
            best_metrics = metrics

    avg_epoch_time = sum(times_per_epoch) / len(times_per_epoch)
    best_metrics['avg_epoch_time'] = avg_epoch_time

    return best_metrics
