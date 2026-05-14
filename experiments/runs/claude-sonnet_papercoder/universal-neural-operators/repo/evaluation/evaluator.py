## Code: evaluation/evaluator.py

```python
## evaluation/evaluator.py
"""
Evaluator for the multi-physics neural operator pretraining framework
described in:
  "Towards Universal Neural Operators through Multiphysics Pretraining"

Implements the Evaluator class that runs full test-set inference, computes
the metrics reported in Tables 1 and 2 of the paper (MSE, NMAE, Avg. epoch
time, parameter count), and formats results for comparison across models.

Design contract (Data structures and interfaces):
  Evaluator:
    __init__(model: AdapterFramework, test_dataset: Dataset,
             physics_id: str, config: EvalConfig, device: torch.device)
    evaluate() -> Dict[str, float]
    benchmark_epoch_time(n_epochs: int = 5) -> float
    count_parameters() -> Dict[str, int]
    _run_inference(batch_input: Tensor) -> Tensor
    compare_models(models: Dict[str, AdapterFramework],
                   physics_id: str) -> pd.DataFrame

Config alignment (config.yaml):
  evaluation.metrics: ["nmae", "mse"]   -> metrics to compute and report
  evaluation.nmae_eps: 1.0e-8           -> passed to Metrics.nmae()
  evaluation.report_epoch_time: true    -> whether to benchmark epoch time
  evaluation.epoch_time_warmup: 1       -> warmup epochs excluded from timing
  evaluation.epoch_time_n_runs: 5       -> number of timed epochs to average
  evaluation.output_dir: "results"      -> where to write CSV/JSON outputs

Tensor layout convention (Shared Knowledge #1):
  Channel-first: [B, C, L] for 1D, [B, C, H, W] for 2D.

Normalization (Shared Knowledge #3):
  Datasets normalize inputs/targets to zero mean / unit variance.
  Evaluator denormalizes predictions and targets before computing NMAE
  (NMAE uses the raw value range max-min of the target field).
  Denormalization is applied if the dataset exposes _mean/_std attributes.

Epoch time measurement (Shared Knowledge #6):
  Wall-clock time per epoch measured with time.perf_counter().
  Timing covers forward+backward+optimizer step over the full DataLoader.
  First epoch_time_warmup epochs are excluded from the average.

Dependencies:
  torch, torch.utils.data.DataLoader
  time, logging, os, typing
  pandas
  models/adapter_framework.py -> AdapterFramework
  evaluation/metrics.py       -> Metrics
  utils/config.py             -> EvalConfig
  utils/logging_utils.py      -> get_logger, ResultsTable
  training/losses.py          -> get_loss_fn (for benchmark_epoch_time)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from evaluation.metrics import Metrics
from models.adapter_framework import AdapterFramework
from training.losses import get_loss_fn
from utils.config import EvalConfig
from utils.logging_utils import ResultsTable, get_logger

# ---------------------------------------------------------------------------
# Module-level logger (fallback; Evaluator creates its own instance logger)
# ---------------------------------------------------------------------------

_logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default batch size for inference DataLoader.
# EvalConfig does not have a batch_size field; use this fixed default.
_DEFAULT_EVAL_BATCH_SIZE: int = 16

# Default number of DataLoader worker processes.
# 0 = load data in the main process (safe default; avoids multiprocessing
# issues with HDF5 files and CUDA tensors).
_DEFAULT_NUM_WORKERS: int = 0

# Default number of timed epochs for benchmark_epoch_time.
# From config.yaml: evaluation.epoch_time_n_runs: 5.
_DEFAULT_N_TIMED_EPOCHS: int = 5

# Default number of warmup epochs excluded from timing average.
# From config.yaml: evaluation.epoch_time_warmup: 1.
_DEFAULT_WARMUP_EPOCHS: int = 1

# Default NMAE epsilon for numerical stability.
# From config.yaml: evaluation.nmae_eps: 1.0e-8.
_DEFAULT_NMAE_EPS: float = 1.0e-8

# Small epsilon for normalization denominator (avoids division by zero
# when std is zero for a constant channel).
_NORM_EPS: float = 1.0e-8

# Temporary optimizer learning rate for benchmark_epoch_time.
# Matches config.yaml training.pretrain.lr: 1.0e-3.
_BENCHMARK_LR: float = 1.0e-3


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class Evaluator:
    """Comprehensive model evaluator for neural operator experiments.

    Runs full test-set inference, computes MSE and NMAE metrics (as reported
    in Tables 1 and 2 of the paper), benchmarks epoch wall-clock time, and
    counts model parameters. Supports multi-model comparison via
    compare_models().

    The evaluator handles denormalization transparently: if the test dataset
    was normalized during loading (normalize=True), predictions and targets
    are denormalized before computing NMAE, since NMAE requires the raw
    value range (max - min) of the target field.

    Attributes:
        model: AdapterFramework with the target adapter registered.
        test_dataset: Single-physics test dataset.
        physics_id: Physics identifier string for routing through the model.
        config: EvalConfig with metrics list, output directory, and timing
            parameters.
        _device: Target device (CPU or CUDA).
        _metrics: Metrics instance for computing NMAE, MSE, relative L2.
        _logger: Instance-specific logger.

    Example::

        from models.fno_backbone import FNOBackbone
        from models.adapter_framework import AdapterFramework
        from utils.config import EvalConfig

        backbone = FNOBackbone(hidden_dim=64, n_modes=16, n_layers=4)
        model = AdapterFramework(backbone=backbone, hidden_dim=64)
        model.load_checkpoint('checkpoints/finetune/finetune_best.pt')

        config = EvalConfig(
            metrics=['nmae', 'mse'],
            output_dir='results/exp1',
        )

        evaluator = Evaluator(
            model=model,
            test_dataset=test_ds,
            physics_id='burgers_nu0p001',
            config=config,
            device=torch.device('cuda'),
        )

        metrics = evaluator.evaluate()
        # metrics = {'nmae': 0.0120, 'mse': 1.009e-7}

        avg_epoch_s = evaluator.benchmark_epoch_time(n_epochs=5)
        # avg_epoch_s = 21.91

        param_counts = evaluator.count_parameters()
        # param_counts = {'backbone': 9_800_000, 'adapters': 200_000, 'total': 10_000_000}
    """

    def __init__(
        self,
        model: AdapterFramework,
        test_dataset: Dataset,
        physics_id: str,
        config: EvalConfig,
        device: torch.device,
    ) -> None:
        """Initialise Evaluator.

        Stores all references and sets up the logger. No computation is
        performed in __init__ — all evaluation is deferred to evaluate(),
        benchmark_epoch_time(), and count_parameters().

        Args:
            model: AdapterFramework with the target adapter already registered
                via register_adapter(physics_id, n_in, n_out). The model is
                moved to ``device`` if not already there.
            test_dataset: Single-physics test dataset. Must implement
                __len__ and __getitem__ returning (input_tensor, target_tensor).
                Optionally exposes normalization statistics as _mean, _std,
                _target_mean, _target_std attributes for denormalization.
            physics_id: Physics identifier string for routing. Must be
                registered in the model. Must follow Shared Knowledge #2
                convention (no dots, use 'p' for decimal point).
                Examples: 'burgers_nu0p001', 'heat_conv_alpha0p01'.
            config: EvalConfig populated from config.yaml. Contains:
                - metrics: List[str] of metric names to compute.
                - output_dir: str path for saving results.
                - nmae_eps: float epsilon for NMAE denominator (optional attr).
                - epoch_time_warmup: int warmup epochs (optional attr).
                - epoch_time_n_runs: int timed epochs (optional attr).
            device: Target device. The model and all batches are moved to
                this device. Falls back to CPU with a warning if CUDA is
                requested but unavailable.

        Raises:
            TypeError: If model is not an AdapterFramework instance.
            ValueError: If physics_id is not registered in the model.
        """
        # ── Type validation ───────────────────────────────────────────────
        if not isinstance(model, AdapterFramework):
            raise TypeError(
                f"model must be an AdapterFramework instance, "
                f"got {type(model).__name__}."
            )

        # ── Device resolution with CUDA fallback ──────────────────────────
        if device.type == "cuda" and not torch.cuda.is_available():
            _logger.warning(
                "CUDA requested but not available. Falling back to CPU."
            )
            device = torch.device("cpu")

        # ── Validate physics_id is registered ────────────────────────────
        if physics_id not in model._adapter_registry:
            registered: List[str] = sorted(model._adapter_registry.keys())
            raise ValueError(
                f"physics_id='{physics_id}' is not registered in the "
                f"AdapterFramework. "
                f"Call model.register_adapter('{physics_id}', n_in, n_out) "
                f"before constructing Evaluator. "
                f"Currently registered physics IDs: {registered}."
            )

        # ── Store references ──────────────────────────────────────────────
        self.model: AdapterFramework = model.to(device)
        self.test_dataset: Dataset = test_dataset
        self.physics_id: str = physics_id
        self.config: EvalConfig = config
        self._device: torch.device = device

        # ── Metrics helper ────────────────────────────────────────────────
        # Metrics is a stateless class with static methods; instantiation
        # is a no-op but follows the design contract.
        self._metrics: Metrics = Metrics()

        # ── Logger ────────────────────────────────────────────────────────
        # Create output directory for log file.
        os.makedirs(config.output_dir, exist_ok=True)
        log_file: str = os.path.join(
            config.output_dir,
            f"evaluator_{physics_id}.log",
        )
        self._logger: logging.Logger = get_logger(
            f"Evaluator[{physics_id}]",
            log_file=log_file,
        )

        self._logger.info(
            "Evaluator initialized: physics_id='%s', device='%s', "
            "test_dataset_size=%d, metrics=%s, output_dir='%s'.",
            physics_id,
            str(device),
            len(test_dataset),  # type: ignore[arg-type]
            config.metrics,
            config.output_dir,
        )

    # -----------------------------------------------------------------------
    # Private: single-batch inference
    # -----------------------------------------------------------------------

    def _run_inference(self, batch_input: Tensor) -> Tensor:
        """Run a single forward pass with no gradient tracking.

        Moves the input to the target device, calls
        AdapterFramework.forward(batch_input, physics_id) inside a
        torch.no_grad() context, and returns the prediction tensor.

        The model must be in eval() mode before calling this method.
        The evaluate() method is responsible for setting eval() mode.

        Args:
            batch_input: Input tensor of shape [B, C_in, *spatial].
                Will be moved to self._device if not already there.

        Returns:
            Prediction tensor of shape [B, n_out, *spatial], on self._device.
            No gradients are tracked.

        Raises:
            KeyError: If self.physics_id is not registered in the model
                (propagated from AdapterFramework.forward).
        """
        batch_input = batch_input.to(self._device, non_blocking=True)

        with torch.no_grad():
            pred: Tensor = self.model.forward(batch_input, self.physics_id)

        return pred

    # -----------------------------------------------------------------------
    # Private: denormalization
    # -----------------------------------------------------------------------

    def _denormalize_predictions(
        self,
        preds: Tensor,
        targets: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Denormalize predictions and targets if the dataset was normalized.

        Checks if the test dataset exposes normalization statistics
        (_target_mean, _target_std). If so, applies the inverse transform:
            x_raw = x_normalized * std + mean

        This is required for NMAE computation, which uses the raw value
        range (max - min) of the target field. Computing NMAE on normalized
        values would give incorrect results since the normalization changes
        the value range.

        MSE is also computed on denormalized values to match the paper's
        reported scale (e.g., 1.009×10⁻⁷ in Table 1).

        The method checks for multiple attribute naming conventions used
        by different dataset classes:
          - PDEBenchDataset: _target_mean, _target_std
          - GrayScottDataset: _mean_out, _std_out
          - HeatConvectionDataset: _target_mean, _target_std

        Args:
            preds: Predicted tensor of shape [N, n_out, *spatial], on CPU.
            targets: Ground-truth tensor of shape [N, n_out, *spatial], on CPU.

        Returns:
            Tuple of (denormalized_preds, denormalized_targets), both on CPU.
            If no normalization statistics are found, returns the inputs
            unchanged.
        """
        dataset = self.test_dataset

        # ── Try to find target normalization statistics ───────────────────
        # Different dataset classes use different attribute names.
        target_mean: Optional[Tensor] = None
        target_std: Optional[Tensor] = None

        # PDEBenchDataset and HeatConvectionDataset convention
        if hasattr(dataset, "_target_mean") and hasattr(dataset, "_target_std"):
            target_mean = getattr(dataset, "_target_mean")
            target_std = getattr(dataset, "_target_std")

        # GrayScottDataset convention
        elif hasattr(dataset, "_mean_out") and hasattr(dataset, "_std_out"):
            target_mean = getattr(dataset, "_mean_out")
            target_std = getattr(dataset, "_std_out")

        # No normalization statistics found — return unchanged
        if target_mean is None or target_std is None:
            self._logger.debug(
                "No normalization statistics found on test_dataset. "
                "Metrics will be computed on normalized values."
            )
            return preds, targets

        # ── Check if normalization was actually applied ───────────────────
        # If normalize=False was used, the dataset stores identity stats
        # (mean=0, std=1). In that case, denormalization is a no-op.
        normalize_flag: bool = getattr(dataset, "normalize", True)
        if not normalize_flag:
            self._logger.debug(
                "Dataset normalize=False: skipping denormalization."
            )
            return preds, targets

        # ── Apply inverse normalization: x_raw = x_norm * std + mean ──────
        # Move stats to CPU for computation (preds and targets are on CPU).
        mean_cpu: Tensor = target_mean.cpu()
        std_cpu: Tensor = target_std.cpu()

        # Broadcast: stats shape [1, n_out, 1, ...] broadcasts over [N, n_out, *spatial]
        preds_denorm: Tensor = preds * (std_cpu + _NORM_EPS) + mean_cpu
        targets_denorm: Tensor = targets * (std_cpu + _NORM_EPS) + mean_cpu

        self._logger.debug(
            "Denormalized predictions and targets. "
            "target_mean=%s, target_std=%s.",
            mean_cpu.flatten().tolist(),
            std_cpu.flatten().tolist(),
        )

        return preds_denorm, targets_denorm

    # -----------------------------------------------------------------------
    # Public: full test-set evaluation
    # -----------------------------------------------------------------------

    def evaluate(self) -> Dict[str, float]:
        """Run full test-set inference and compute evaluation metrics.

        Iterates over the test dataset in batches, accumulates all predictions
        and targets in memory, denormalizes them (if the dataset was
        normalized), and computes the metrics specified in config.metrics.

        The method sets the model to eval() mode before inference and
        restores train() mode afterward.

        Returns:
            Dict mapping metric names to float values. Keys are a subset of
            {'nmae', 'mse', 'relative_l2'} as specified in config.metrics.
            NMAE is in percentage units (e.g., 0.0120 means 0.0120 %).
            MSE is in raw squared units (e.g., 1.009e-7).

        Raises:
            RuntimeError: If the test dataset is empty.

        Example::

            results = evaluator.evaluate()
            # results = {'nmae': 0.0120, 'mse': 1.009e-7}
        """
        if len(self.test_dataset) == 0:  # type: ignore[arg-type]
            raise RuntimeError(
                f"Test dataset for physics_id='{self.physics_id}' is empty. "
                f"Cannot evaluate on an empty dataset."
            )

        # ── Set model to eval mode ────────────────────────────────────────
        self.model.eval()

        # ── Create inference DataLoader ───────────────────────────────────
        # No custom collate_fn needed — single-physics dataset returns
        # (input_tensor, target_tensor) 2-tuples.
        pin_memory: bool = (self._device.type == "cuda")

        eval_loader: DataLoader = DataLoader(
            self.test_dataset,
            batch_size=_DEFAULT_EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=_DEFAULT_NUM_WORKERS,
            pin_memory=pin_memory,
            drop_last=False,
        )

        # ── Accumulate predictions and targets ────────────────────────────
        all_preds: List[Tensor] = []
        all_targets: List[Tensor] = []

        self._logger.info(
            "Starting evaluation: %d test samples, %d batches.",
            len(self.test_dataset),  # type: ignore[arg-type]
            len(eval_loader),
        )

        for batch in eval_loader:
            # Single-physics dataset returns (input_tensor, target_tensor).
            # Unpack the 2-tuple.
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                batch_input: Tensor = batch[0]
                batch_target: Tensor = batch[1]
            elif isinstance(batch, (list, tuple)) and len(batch) == 3:
                # MultiPhysicsDataset returns (input, target, physics_id).
                # Handle gracefully in case this evaluator is used with one.
                batch_input = batch[0]
                batch_target = batch[1]
            else:
                raise ValueError(
                    f"Unexpected batch format from DataLoader: "
                    f"expected 2-tuple (input, target) or 3-tuple "
                    f"(input, target, physics_id), got {type(batch)} "
                    f"with length {len(batch)}."
                )

            # ── Run inference ─────────────────────────────────────────────
            pred: Tensor = self._run_inference(batch_input)

            # ── Collect on CPU to avoid GPU memory accumulation ───────────
            all_preds.append(pred.cpu())
            all_targets.append(batch_target.cpu())

        # ── Concatenate all batches ───────────────────────────────────────
        # Shape: [N_test, n_out, *spatial]
        preds_cat: Tensor = torch.cat(all_preds, dim=0)
        targets_cat: Tensor = torch.cat(all_targets, dim=0)

        # ── Align channel counts ──────────────────────────────────────────
        # Predictions have n_out channels (from ProjectionAdapter).
        # Targets may have more channels if the dataset was constructed with
        # extra channels (e.g., padded by collate_fn). Slice to match.
        n_out: int = preds_cat.shape[1]
        targets_cat = targets_cat[:, :n_out]

        # ── Denormalize before computing metrics ──────────────────────────
        # NMAE requires the raw value range (max - min) of the target field.
        # MSE is also computed on denormalized values to match paper scale.
        preds_denorm: Tensor
        targets_denorm: Tensor
        preds_denorm, targets_denorm = self._denormalize_predictions(
            preds_cat, targets_cat
        )

        # ── Compute all metrics ───────────────────────────────────────────
        # Get NMAE epsilon from config (optional attribute, default 1e-8).
        nmae_eps: float = getattr(self.config, "nmae_eps", _DEFAULT_NMAE_EPS)

        all_metric_results: Dict[str, float] = self._metrics.compute_all(
            preds_denorm, targets_denorm, eps=nmae_eps
        )

        # ── Filter to requested metrics ───────────────────────────────────
        # config.metrics specifies which metrics to report (e.g., ['nmae', 'mse']).
        results: Dict[str, float] = {
            metric_name: all_metric_results[metric_name]
            for metric_name in self.config.metrics
            if metric_name in all_metric_results
        }

        # ── Log results ───────────────────────────────────────────────────
        self._logger.info(
            "Evaluation complete for physics_id='%s': %s",
            self.physics_id,
            {k: f"{v:.6e}" if k != "nmae" else f"{v:.4f}%" for k, v in results.items()},
        )

        # ── Restore model to train mode ───────────────────────────────────
        self.model.train()

        return results

    # -----------------------------------------------------------------------
    # Public: epoch time benchmarking
    # -----------------------------------------------------------------------

    def benchmark_epoch_time(self, n_epochs: int = _DEFAULT_N_TIMED_EPOCHS) -> float:
        """Measure average wall-clock time per training epoch.

        Simulates the training loop (forward + loss + backward + optimizer
        step) over the full DataLoader for multiple epochs and returns the
        average epoch time in seconds. This reproduces the "Avg. epoch (s)"
        column in Tables 1 and 2 of the paper.

        The first ``epoch_time_warmup`` epochs are excluded from the average
        to account for CUDA kernel compilation and GPU warm-up overhead.

        Timing methodology (Shared Knowledge #6):
          - Wall-clock time measured with time.perf_counter()
          - Timing wraps the full inner loop (all batches in one epoch)
          - DataLoader iteration time IS included (unavoidable with full-epoch
            timing, but consistent across models)
          - A temporary Adam optimizer is created and discarded after timing

        Args:
            n_epochs: Number of timed epochs to average over. Default 5
                (from config.yaml evaluation.epoch_time_n_runs: 5).
                The actual number of epochs run is
                epoch_time_warmup + n_epochs.

        Returns:
            Average wall-clock time per epoch in seconds (float).
            Returns 0.0 if timing fails (e.g., empty dataset).

        Note:
            The model is set to train() mode during benchmarking and
            restored to eval() mode afterward. The temporary optimizer
            is created over all model parameters (matching the pretraining
            setup) to ensure gradient computation is representative.
        """
        if len(self.test_dataset) == 0:  # type: ignore[arg-type]
            self._logger.warning(
                "benchmark_epoch_time: test dataset is empty. Returning 0.0."
            )
            return 0.0

        # ── Read warmup epochs from config ────────────────────────────────
        warmup_epochs: int = getattr(
            self.config, "epoch_time_warmup", _DEFAULT_WARMUP_EPOCHS
        )

        # ── Create DataLoader for benchmarking ────────────────────────────
        # Use the same batch size as evaluation for consistency.
        pin_memory: bool = (self._device.type == "cuda")

        bench_loader: DataLoader = DataLoader(
            self.test_dataset,
            batch_size=_DEFAULT_EVAL_BATCH_SIZE,
            shuffle=True,  # Shuffle to simulate real training
            num_workers=_DEFAULT_NUM_WORKERS,
            pin_memory=pin_memory,
            drop_last=True,  # Drop last incomplete batch (matches training)
        )

        if len(bench_loader) == 0:
            self._logger.warning(
                "benchmark_epoch_time: DataLoader has 0 batches "
                "(dataset too small for batch_size=%d with drop_last=True). "
                "Returning 0.0.",
                _DEFAULT_EVAL_BATCH_SIZE,
            )
            return 0.0

        # ── Set model to train mode ───────────────────────────────────────
        self.model.train()

        # ── Create temporary optimizer ────────────────────────────────────
        # Use Adam over all parameters (matching pretraining setup).
        # This ensures gradient computation through backbone + adapters.
        temp_optimizer: Adam = Adam(
            self.model.parameters(),
            lr=_BENCHMARK_LR,
        )

        # ── Loss function for benchmarking ────────────────────────────────
        bench_loss_fn: nn.Module = get_loss_fn("mse")

        # ── Warmup phase ──────────────────────────────────────────────────
        # Run warmup_epochs epochs without timing to warm up CUDA kernels
        # and ensure the first timed epoch is representative.
        self._logger.info(
            "benchmark_epoch_time: running %d warmup epoch(s) + %d timed epoch(s).",
            warmup_epochs,
            n_epochs,
        )

        for warmup_epoch in range(warmup_epochs):
            self._run_benchmark_epoch(
                bench_loader, temp_optimizer, bench_loss_fn
            )

        # ── Timed phase ───────────────────────────────────────────────────
        elapsed_times: List[float] = []

        for timed_epoch in range(n_epochs):
            epoch_time: float = self._run_benchmark_epoch(
                bench_loader, temp_optimizer, bench_loss_fn, timed=True
            )
            elapsed_times.append(epoch_time)

            self._logger.debug(
                "benchmark_epoch_time: timed epoch %d/%d = %.4fs.",
                timed_epoch + 1,
                n_epochs,
                epoch_time,
            )

        # ── Compute average ───────────────────────────────────────────────
        if not elapsed_times:
            avg_time: float = 0.0
        else:
            avg_time = sum(elapsed_times) / len(elapsed_times)

        self._logger.info(
            "benchmark_epoch_time: avg=%.4fs over %d timed epochs "
            "(warmup=%d, physics_id='%s').",
            avg_time,
            n_epochs,
            warmup_epochs,
            self.physics_id,
        )

        # ── Cleanup ───────────────────────────────────────────────────────
        # Delete temporary optimizer to free memory.
        del temp_optimizer
        del bench_loss_fn

        # Restore model to eval mode.
        self.model.eval()

        return avg_time

    def _run_benchmark_epoch(
        self,
        loader: DataLoader,
        optimizer: Adam,
        loss_fn: nn.Module,
        timed: bool = False,
    ) -> float:
        """Run one benchmark epoch over the DataLoader.

        Iterates over all batches in the loader, performing forward pass,
        loss computation, backward pass, and optimizer step. Optionally
        measures and returns the total wall-clock time.

        Args:
            loader: DataLoader over the benchmark dataset.
            optimizer: Temporary Adam optimizer for gradient computation.
            loss_fn: MSELoss instance for computing training loss.
            timed: If True, measure and return wall-clock time for this epoch.
                If False, run without timing (warmup mode).

        Returns:
            Wall-clock time for this epoch in seconds if timed=True.
            0.0 if timed=False.
        """
        start_time: float = time.perf_counter() if timed else 0.0

        for batch in loader:
            # Unpack batch (2-tuple or 3-tuple)
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                batch_input: Tensor = batch[0].to(self._device, non_blocking=True)
                batch_target: Tensor = batch[1].to(self._device, non_blocking=True)
            else:
                continue

            # ── Forward pass ──────────────────────────────────────────────
            optimizer.zero_grad(set_to_none=True)
            pred: Tensor = self.model.forward(batch_input, self.physics_id)

            # ── Loss computation ──────────────────────────────────────────
            n_out: int = pred.shape[1]
            target_sliced: Tensor = batch_target[:, :n_out]
            loss: Tensor = loss_fn(pred, target_sliced)

            # ── Backward pass ─────────────────────────────────────────────
            loss.backward()
            optimizer.step()

        if timed:
            end_time: float = time.perf_counter()
            return end_time - start_time

        return 0.0

    # -----------------------------------------------------------------------
    # Public: parameter counting
    # -----------------------------------------------------------------------

    def count_parameters(self) -> Dict[str, int]:
        """Return parameter counts for backbone, adapters, and total.

        Delegates to AdapterFramework.get_param_count() which returns a
        dict with keys 'backbone', 'adapters', and 'total'.

        Returns:
            Dict with keys:
              - 'backbone': int — number of backbone parameters (θ_F)
              - 'adapters': int — total adapter parameters across all
                registered physics (Σ θ_{L_i} + Σ θ_{P_i})
              - 'total': int — backbone + adapters

        Example::

            counts = evaluator.count_parameters()
            # counts = {'backbone': 9_800_000, 'adapters': 200_000,
            #           'total': 10_000_000}
        """
        param_counts: Dict[str, int] = self.model.get_param_count()

        self._logger.info(
            "Parameter counts for physics_id='%s': "
            "backbone=%d, adapters=%d, total=%d.",
            self.physics_id,
            param_counts.get("backbone", 0),
            param_counts.get("adapters", 0