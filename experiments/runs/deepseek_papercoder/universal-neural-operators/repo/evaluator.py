"""
evaluator.py

Evaluator module for the Universal Neural Operators reproduction pipeline.

Computes per‑problem test metrics (MSE and NMAE) using a trained ModelBase,
and provides a method to obtain the average training epoch time from a
previously run Trainer. All hyperparameters are read from the global Config
object (see config.yaml).

The NMAE is computed according to equation (3) of the paper:
    NMAE(θ) = 1/|D_test| Σ_{(a,u)∈D_test} ||G_θ(a) - u||_{1,G} / (max_G u - min_G u + ε)
where ε is a small constant.

Usage:
    evaluator = Evaluator(model, test_datasets, config)
    metrics = evaluator.evaluate("burgers_nu0.1")
    epoch_time = evaluator.time_epoch(trainer)
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

# Local imports (ensure these modules are accessible)
from config import Config
from data_utils import Dataset
from models import ModelBase
from trainer import Trainer  # used only for type hinting and epoch time retrieval


class Evaluator:
    """
    Evaluates a trained neural operator on a set of test datasets.

    Attributes:
        model: The trained ModelBase (contains body + adapters).
        test_datasets: Dictionary mapping problem names to test Dataset instances.
        config: Global configuration.
        device: Torch device used for evaluation.
    """

    def __init__(
        self,
        model: ModelBase,
        test_datasets: Dict[str, Dataset],
        config: Config,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize the Evaluator.

        Args:
            model: A trained ModelBase instance.
            test_datasets: Mapping from problem name (str) to test Dataset.
            config: The global configuration object.
            device: Optional torch device; if None, uses CUDA if available else CPU.
        """
        self.model = model
        self.test_datasets = test_datasets
        self.config = config

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Put model into evaluation mode and move to device.
        self.model.to(self.device)
        self.model.eval()

        # Extract constants from config.
        self.epsilon_nmae = config.eval_params.get("epsilon_nmae", 1e-9)

        # Determine evaluation batch size. We use the training batch size from the
        # 'pretrain' phase as a default; if not specified, fall back to 16.
        train_cfg = config.training_params.get("pretrain", {})
        self.batch_size = train_cfg.get("batch_size", 16)

    @torch.no_grad()
    def evaluate(self, problem_name: str) -> Dict[str, float]:
        """
        Compute MSE and NMAE on the test set for a given problem.

        Args:
            problem_name: Key identifying the test dataset.

        Returns:
            Dictionary with keys 'mse' and 'nmae' (NMAE as a fraction, not percent).

        Raises:
            KeyError: If problem_name is not found in self.test_datasets.
            ValueError: If the corresponding dataset is empty.
        """
        if problem_name not in self.test_datasets:
            raise KeyError(
                f"Problem '{problem_name}' not found in test_datasets. "
                f"Available: {list(self.test_datasets.keys())}"
            )

        ds = self.test_datasets[problem_name]
        if len(ds) == 0:
            raise ValueError(f"Test dataset for problem '{problem_name}' is empty.")

        loader = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        total_mse = 0.0
        total_nmae = 0.0
        count = 0

        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            # Forward through the model (selection of adapter handled inside).
            pred = self.model.forward(problem_name, x)

            # Compute per‑sample MSE (averaged over all spatial/channel dims).
            mse_per_sample = ((pred - y) ** 2).view(y.size(0), -1).mean(dim=1)

            # Compute per‑sample range of the ground truth.
            y_flat = y.view(y.size(0), -1)
            range_y = y_flat.max(dim=1).values - y_flat.min(dim=1).values

            # Compute per‑sample NMAE using the static method.
            nmae_per_sample = self.compute_nmae(pred, y, range_y, self.epsilon_nmae)

            # Accumulate
            total_mse += mse_per_sample.sum().item()
            total_nmae += nmae_per_sample.sum().item()
            count += y.size(0)

        avg_mse = total_mse / count
        avg_nmae = total_nmae / count

        return {"mse": avg_mse, "nmae": avg_nmae}

    @staticmethod
    def compute_nmae(
        pred: torch.Tensor,
        target: torch.Tensor,
        range_val: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        """
        Compute the range‑normalized mean absolute error for a batch.

        Args:
            pred: Predicted tensor, shape (B, ...).
            target: Ground‑truth tensor, same shape.
            range_val: Per‑sample range (max - min) of the target, shape (B,).
            epsilon: Small constant to avoid division by zero.

        Returns:
            Per‑sample NMAE values, shape (B,).
        """
        # Mean absolute error over all non‑batch dimensions.
        abs_error = (pred - target).abs().view(target.size(0), -1).mean(dim=1)

        # Apply denominator with epsilon.
        denom = range_val + epsilon
        return abs_error / denom

    def time_epoch(self, trainer: Trainer) -> float:
        """
        Retrieve the average training epoch time from a previously executed Trainer.

        The Trainer is expected to have computed and stored the average epoch duration
        in its `avg_epoch_time` attribute (see Trainer.__init__ and Trainer.train()).

        Args:
            trainer: A Trainer instance that has completed training.

        Returns:
            Average wall‑clock time per epoch (seconds).

        Raises:
            ValueError: If the Trainer has not recorded any epoch timings.
        """
        if not hasattr(trainer, "avg_epoch_time") or trainer.avg_epoch_time <= 0.0:
            raise ValueError(
                "The provided Trainer does not contain valid epoch timings. "
                "Ensure that training has been completed."
            )
        return trainer.avg_epoch_time

