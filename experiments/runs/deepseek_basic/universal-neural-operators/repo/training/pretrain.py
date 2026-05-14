"""Multi-physics pretraining framework.

As described in Section 3 ("Pre-training and fine-tuning"):
In the pre-training phase the entire parameters set is subject to optimization.
Problems 1 to N represent separate physical processes, demanding different
(but probably overlapping) sets of input functions.

The adapter-based approach allows simultaneous training on PDE-based problems
with different sets of input functions by using separate lifting/projection
adapters for each physics problem, all sharing the same core FNO/operator.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import time
import copy

from .metrics import compute_metrics, NMAE


class MultiPhysicsPretrainer:
    """Handles pretraining of neural operators on multiple physics datasets.

    The pretraining uses separate lifting and projection adapters for each
    physics problem, while sharing the core FNO/operator blocks. All parameters
    (adapters + core) are trained jointly during pretraining.
    """

    def __init__(
        self,
        model_factory,
        core_model: nn.Module = None,
        device: str = 'cuda',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_step: int = 100,
        scheduler_gamma: float = 0.5,
    ):
        """
        Args:
            model_factory: Function that creates a model given (input_channels, output_channels)
            core_model: Optional shared core model (if None, created by factory)
            device: Device to train on
            learning_rate: Initial learning rate
            weight_decay: Weight decay for AdamW
            scheduler_step: Step size for StepLR scheduler
            scheduler_gamma: Multiplicative factor for LR decay
        """
        self.model_factory = model_factory
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_step = scheduler_step
        self.scheduler_gamma = scheduler_gamma

        self.models: Dict[str, nn.Module] = {}
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        self.schedulers: Dict[str, torch.optim.lr_scheduler._LRScheduler] = {}
        self.core_model = core_model

    def add_physics(
        self,
        name: str,
        input_channels: int,
        output_channels: int,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        model_kwargs: Optional[dict] = None,
    ):
        """Register a physics problem for pretraining.

        Args:
            name: Unique name for this physics problem
            input_channels: Number of input functions
            output_channels: Number of output functions
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation
            model_kwargs: Additional kwargs for model_factory
        """
        if model_kwargs is None:
            model_kwargs = {}

        # Create model with problem-specific input/output channels
        model = self.model_factory(
            input_channels=input_channels,
            output_channels=output_channels,
            **model_kwargs,
        )

        # Share core model if provided
        if self.core_model is not None:
            self._share_core_weights(model, self.core_model)

        model = model.to(self.device)
        self.models[name] = model

        # Create optimizer for this physics
        # In pretraining, ALL parameters are trainable
        self.optimizers[name] = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.schedulers[name] = torch.optim.lr_scheduler.StepLR(
            self.optimizers[name],
            step_size=self.scheduler_step,
            gamma=self.scheduler_gamma,
        )

        # Store data loaders
        if not hasattr(self, 'train_loaders'):
            self.train_loaders = {}
            self.val_loaders = {}
        self.train_loaders[name] = train_loader
        self.val_loaders[name] = val_loader

    def _share_core_weights(self, model: nn.Module, core_model: nn.Module):
        """Share core FNO/operator weights between models."""
        if hasattr(core_model, 'fno_blocks') and hasattr(model, 'fno_blocks'):
            model.fno_blocks = core_model.fno_blocks
        if hasattr(core_model, 'perceiver_blocks') and hasattr(model, 'perceiver_blocks'):
            model.perceiver_blocks = core_model.perceiver_blocks
        if hasattr(core_model, 'coda_blocks') and hasattr(model, 'coda_blocks'):
            model.coda_blocks = core_model.coda_blocks
        if hasattr(core_model, 'mamba') and hasattr(model, 'mamba'):
            model.mamba = core_model.mamba
        if hasattr(core_model, 'stages') and hasattr(model, 'stages'):
            model.stages = core_model.stages

    def pretrain_epoch(self, physics_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Run one pretraining epoch over all physics problems.

        Args:
            physics_names: Subset of physics to train on (default: all)

        Returns:
            Dict with training metrics averaged across physics
        """
        if physics_names is None:
            physics_names = list(self.models.keys())

        total_loss = 0.0
        total_batches = 0

        for name in physics_names:
            model = self.models[name]
            model.train()
            optimizer = self.optimizers[name]
            loader = self.train_loaders[name]

            for batch in loader:
                # Unpack batch - format depends on dataset
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

                optimizer.zero_grad()

                pred = model(x, grid=grid)
                loss = nn.functional.mse_loss(pred, y)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_batches += 1

            self.schedulers[name].step()

        avg_loss = total_loss / max(total_batches, 1)
        return {'pretrain_loss': avg_loss}

    def evaluate(self, physics_names: Optional[List[str]] = None) -> Dict[str, Dict]:
        """Evaluate all models on their validation sets."""
        if physics_names is None:
            physics_names = list(self.models.keys())

        results = {}
        for name in physics_names:
            model = self.models[name]
            model.eval()
            loader = self.val_loaders.get(name)

            if loader is None:
                continue

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

                    pred = model(x, grid=grid)
                    all_preds.append(pred)
                    all_targets.append(y)

            preds = torch.cat(all_preds, dim=0)
            targets = torch.cat(all_targets, dim=0)
            results[name] = compute_metrics(preds, targets)

        return results

    def get_core_model(self) -> nn.Module:
        """Extract the shared core model for fine-tuning."""
        # Return a reference to the core blocks from any model
        if self.models:
            first_model = next(iter(self.models.values()))
            return first_model
        return None

    def save_checkpoint(self, path: str):
        """Save pretraining state."""
        state = {
            'models': {name: model.state_dict() for name, model in self.models.items()},
            'optimizers': {name: opt.state_dict() for name, opt in self.optimizers.items()},
            'schedulers': {name: sch.state_dict() for name, sch in self.schedulers.items()},
        }
        torch.save(state, path)

    def load_checkpoint(self, path: str):
        """Load pretraining state."""
        state = torch.load(path, map_location=self.device)
        for name, model in self.models.items():
            model.load_state_dict(state['models'][name])
        for name, opt in self.optimizers.items():
            opt.load_state_dict(state['optimizers'][name])
        for name, sch in self.schedulers.items():
            sch.load_state_dict(state['schedulers'][name])
