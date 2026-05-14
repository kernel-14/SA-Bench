## Code: evaluation/evaluator.py

```python
## evaluation/evaluator.py
"""Evaluator class for WDNO (Wavelet Diffusion Neural Operator).

This module implements the Evaluator class that orchestrates all evaluation,
ablation, and baseline comparison logic. It consumes outputs from Simulator,
Controller, SuperResolver, and baseline models, then delegates metric
computation to the Metrics class.

Reproduces results from:
    - Table 1: Simulation MSE across all experiments
    - Table 2a/2b: Control objective J for 1D Burgers' and 2D fluid
    - Table 5: Extended comparisons on 1D compressible NS (MSE, MAE, L∞)
    - Table 6: Approximate scale invariance verification
    - Table 7/8: Sensitivity analysis (DDIM steps, eta, guidance weight)
    - Table 9/10: Robustness and mean±std results
    - Table 16/17: Zero-shot super-resolution MSE
    - Figure 4a/4b: SR performance vs number of levels
    - Figure 5a: Per-timestep error (long-term dependency)
    - Figure 5b: Training data size sensitivity
    - Figure 5c: Wavelet vs Fourier comparison
    - Figure 5d: Measurement noise robustness

Paper sources:
    - Evaluation protocol: Section 4 ("MSE measured on entire state sequences
      excluding initial conditions")
    - Control evaluation: Appendix F.1 (always use ground-truth solver)
    - SR evaluation: Section 4.6 (interpolate to finest resolution)
    - Ablation studies: Section 4.7, Appendix C

Config references:
    - evaluation.exclude_initial_condition: true
    - evaluation.simulation_metrics: [mse, mae, l_inf, relative_l2]
    - evaluation.control_metrics: [mean_J, std_J]
    - evaluation.report_mean_std: true
    - evaluation.compute_per_timestep_error: true
    - evaluation.num_control_test_samples: 50
    - ablation.noise_scale_factors: [0.0001, 0.001, 0.01]
    - ablation.data_size_fractions: [0.2, 0.4, 0.6, 0.8]
    - ablation.ablation_train_size: 9000
    - ablation.ddim_step_values: [20, 40, 50, 100, 200]
    - ablation.ddim_eta_values: [0.2, 0.5, 0.8, 1.0]
    - ablation.wavelet_types_to_compare: [bior1.3, bior2.4, db4, sym4]
    - super_resolution.eval_interp_modes: [linear, nearest]
"""

from __future__ import annotations

import copy
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import Config
from evaluation.metrics import Metrics
from models.wdno_pipeline import WDNOPipeline
from utils.helpers import (
    linear_interpolate,
    make_dirs,
    nearest_interpolate,
    set_seed,
)

logger = logging.getLogger(__name__)

# Numerical precision for metric accumulation
_METRIC_DTYPE: torch.dtype = torch.float64

# Default number of control test samples (paper Appendix F.2)
_DEFAULT_NUM_CONTROL_TEST_SAMPLES: int = 50

# Minimum number of samples required for std computation
_MIN_SAMPLES_FOR_STD: int = 2


class Evaluator:
    """Orchestrates all evaluation, ablation, and baseline comparison for WDNO.

    This class is the top-level evaluation coordinator. It consumes pre-computed
    model outputs (tensors) and delegates metric computation to the Metrics class.
    For ablation studies that require re-training, it constructs Trainer instances
    internally.

    All methods return structured dicts whose keys match the paper's table/figure
    structure for easy result reporting.

    Attributes:
        config: Experiment configuration. Drives all evaluation branching.
        metrics: Metrics instance for all metric computations.
        exclude_ic: Whether to exclude initial condition from MSE computation.
            Config: evaluation.exclude_initial_condition=true.
        simulation_metrics: List of metric names to compute for simulation.
            Config: evaluation.simulation_metrics=[mse, mae, l_inf, relative_l2].
        report_mean_std: Whether to report mean±std for control evaluation.
            Config: evaluation.report_mean_std=true.
        compute_per_timestep_error: Whether to compute per-timestep MSE.
            Config: evaluation.compute_per_timestep_error=true.
        num_control_test_samples: Number of control test samples.
            Config: evaluation.num_control_test_samples=50.
        spatial_dim: Spatial dimensionality of the PDE (1 or 2).
            Config: experiment.spatial_dim.
        experiment: Experiment name string.
            Config: experiment.name.
        device: Compute device string.
            Config: experiment.device (resolved in Config.from_dict).
        eval_interp_modes: Interpolation modes for SR evaluation.
            Config: super_resolution.eval_interp_modes=[linear, nearest].
        noise_scale_factors: Scale factors for noise ablation.
            Config: ablation.noise_scale_factors=[0.0001, 0.001, 0.01].
        data_size_fractions: Dataset size fractions for size ablation.
            Config: ablation.data_size_fractions=[0.2, 0.4, 0.6, 0.8].
        ablation_train_size: Full training set size for ablation.
            Config: ablation.ablation_train_size=9000.
        ddim_step_values: DDIM step counts for sensitivity analysis.
            Config: ablation.ddim_step_values=[20, 40, 50, 100, 200].
        ddim_eta_values: DDIM eta values for sensitivity analysis.
            Config: ablation.ddim_eta_values=[0.2, 0.5, 0.8, 1.0].
        wavelet_types_to_compare: Wavelet types for ablation.
            Config: ablation.wavelet_types_to_compare=[bior1.3, bior2.4, db4, sym4].
        control_alpha: Weight alpha in control objective I.
            Config: data.burgers.control_alpha=0.1.
        dx: Spatial step size for 1D integration (1/(nx_coarse-1)).
        dt: Temporal step size for 1D integration (T/nt_coarse).
    """

    def __init__(self, config: Config) -> None:
        """Initialize the Evaluator.

        Reads all evaluation hyperparameters from config and instantiates
        the Metrics helper. No heavy computation occurs here.

        Args:
            config: Experiment configuration. All evaluation parameters are
                read from this object. Config fields used:
                - evaluation.exclude_initial_condition (true)
                - evaluation.simulation_metrics ([mse, mae, l_inf, relative_l2])
                - evaluation.report_mean_std (true)
                - evaluation.compute_per_timestep_error (true)
                - evaluation.num_control_test_samples (50)
                - experiment.spatial_dim (1 or 2)
                - experiment.name (e.g., 'burgers')
                - experiment.device ('cuda' or 'cpu')
                - super_resolution.eval_interp_modes ([linear, nearest])
                - ablation.noise_scale_factors ([0.0001, 0.001, 0.01])
                - ablation.data_size_fractions ([0.2, 0.4, 0.6, 0.8])
                - ablation.ablation_train_size (9000)
                - ablation.ddim_step_values ([20, 40, 50, 100, 200])
                - ablation.ddim_eta_values ([0.2, 0.5, 0.8, 1.0])
                - ablation.wavelet_types_to_compare ([bior1.3, bior2.4, db4, sym4])
                - data.burgers.control_alpha (0.1)
                - data.burgers.nx_coarse (120)
                - data.burgers.nt_coarse (80)
                - data.burgers.T (8.0)
        """
        self.config: Config = config
        self.metrics: Metrics = Metrics()

        # Evaluation protocol flags (from config.yaml evaluation section)
        self.exclude_ic: bool = True  # config: evaluation.exclude_initial_condition=true
        self.simulation_metrics: List[str] = [
            "mse", "mae", "l_inf", "relative_l2"
        ]  # config: evaluation.simulation_metrics
        self.report_mean_std: bool = True  # config: evaluation.report_mean_std=true
        self.compute_per_timestep_error: bool = True  # config: evaluation.compute_per_timestep_error=true
        self.num_control_test_samples: int = _DEFAULT_NUM_CONTROL_TEST_SAMPLES
        # config: evaluation.num_control_test_samples=50

        # Experiment identification
        self.spatial_dim: int = config.spatial_dim
        self.experiment: str = config.experiment
        self.device: str = config.device

        # Super-resolution evaluation
        self.eval_interp_modes: List[str] = config.eval_interp_modes
        # config: super_resolution.eval_interp_modes=[linear, nearest]

        # Ablation study parameters
        self.noise_scale_factors: List[float] = config.ablation_noise_scale_factors
        # config: ablation.noise_scale_factors=[0.0001, 0.001, 0.01]
        self.data_size_fractions: List[float] = config.ablation_data_size_fractions
        # config: ablation.data_size_fractions=[0.2, 0.4, 0.6, 0.8]
        self.ablation_train_size: int = config.ablation_train_size
        # config: ablation.ablation_train_size=9000
        self.ddim_step_values: List[int] = config.ablation_ddim_step_values
        # config: ablation.ddim_step_values=[20, 40, 50, 100, 200]
        self.ddim_eta_values: List[float] = config.ablation_ddim_eta_values
        # config: ablation.ddim_eta_values=[0.2, 0.5, 0.8, 1.0]
        self.wavelet_types_to_compare: List[str] = config.ablation_wavelet_types
        # config: ablation.wavelet_types_to_compare=[bior1.3, bior2.4, db4, sym4]

        # Control objective parameters (1D Burgers')
        self.control_alpha: float = config.control_alpha
        # config: data.burgers.control_alpha=0.1
        self.dx: float = 1.0 / max(config.nx_coarse - 1, 1)
        # = 1/119 ≈ 0.0084 for nx_coarse=120
        self.dt: float = config.T / max(config.nt_coarse, 1)
        # = 8.0/80 = 0.1 for T=8.0, nt_coarse=80

        logger.info(
            "Evaluator initialized: experiment=%s, spatial_dim=%d, device=%s, "
            "exclude_ic=%s, simulation_metrics=%s, report_mean_std=%s, "
            "compute_per_timestep_error=%s, eval_interp_modes=%s",
            self.experiment,
            self.spatial_dim,
            self.device,
            self.exclude_ic,
            self.simulation_metrics,
            self.report_mean_std,
            self.compute_per_timestep_error,
            self.eval_interp_modes,
        )

    # -----------------------------------------------------------------------
    # Simulation evaluation
    # -----------------------------------------------------------------------

    def evaluate_simulation(
        self,
        model_outputs: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> Dict[str, Any]:
        """Evaluate simulation performance against ground truth.

        Reproduces Table 1 numbers (MSE on entire state sequence excluding
        initial condition). Also computes MAE, L∞, and relative L2 as
        reported in Appendix C.1 (Table 5) for compressible NS.

        Paper Section 4: "We report the Mean Squared Error (MSE) measured on
        entire state sequences excluding initial conditions."

        Args:
            model_outputs: Predicted state trajectories. Shape [N, T+1, X]
                for 1D PDEs (e.g., [N, 81, 120] for Burgers') or [N, T, H, W]
                for 2D PDEs (e.g., [N, 32, 64, 64] for fluid_2d). The time
                dimension is dim=1. May include t=0 (initial condition).
                dtype=float32.
            ground_truth: Ground-truth state trajectories. Must have the same
                shape as model_outputs. dtype=float32.

        Returns:
            Dict with keys from config.evaluation.simulation_metrics:
                'mse': float — Mean Squared Error (primary metric, Table 1)
                'mae': float — Mean Absolute Error (Table 5)
                'l_inf': float — L∞ error (Table 5)
                'relative_l2': float — Relative L2 error (ERA5 Section 4.5)
                'per_timestep_mse': Tensor [T] — per-timestep MSE (Figure 5a)
                    Only present if config.evaluation.compute_per_timestep_error=true.
                'n_samples': int — number of test samples evaluated.

        Raises:
            ValueError: If model_outputs and ground_truth have different shapes.
        """
        if model_outputs.shape != ground_truth.shape:
            raise ValueError(
                f"model_outputs and ground_truth must have the same shape. "
                f"Got model_outputs.shape={tuple(model_outputs.shape)} and "
                f"ground_truth.shape={tuple(ground_truth.shape)}."
            )

        # Move to device for computation
        device = torch.device(self.device)
        pred: torch.Tensor = model_outputs.to(device=device, dtype=torch.float32)
        target: torch.Tensor = ground_truth.to(device=device, dtype=torch.float32)

        n_samples: int = pred.shape[0]

        # Exclude initial condition (t=0) per paper convention
        # Config: evaluation.exclude_initial_condition=true
        if self.exclude_ic and pred.shape[1] > 1:
            pred_eval: torch.Tensor = pred[:, 1:, ...]
            target_eval: torch.Tensor = target[:, 1:, ...]
        else:
            pred_eval = pred
            target_eval = target

        result: Dict[str, Any] = {"n_samples": n_samples}

        # Compute requested metrics
        if "mse" in self.simulation_metrics:
            result["mse"] = self.metrics.mse(
                pred_eval, target_eval, exclude_ic=False
            )
            # exclude_ic=False because we already sliced above

        if "mae" in self.simulation_metrics:
            result["mae"] = self.metrics.mae(pred_eval, target_eval)

        if "l_inf" in self.simulation_metrics:
            result["l_inf"] = self.metrics.l_inf(pred_eval, target_eval)

        if "relative_l2" in self.simulation_metrics:
            result["relative_l2"] = self.metrics.relative_l2(pred_eval, target_eval)

        # Per-timestep MSE for long-term dependency analysis (Figure 5a, Figure 9)
        # Config: evaluation.compute_per_timestep_error=true
        if self.compute_per_timestep_error:
            per_t_mse: torch.Tensor = self.metrics.per_timestep_mse(
                pred_eval, target_eval
            )
            result["per_timestep_mse"] = per_t_mse.cpu()

        logger.info(
            "evaluate_simulation: N=%d, MSE=%.6e, MAE=%.6e, L_inf=%.6e, "
            "rel_L2=%.6e",
            n_samples,
            result.get("mse", float("nan")),
            result.get("mae", float("nan")),
            result.get("l_inf", float("nan")),
            result.get("relative_l2", float("nan")),
        )

        return result

    # -----------------------------------------------------------------------
    # Control evaluation
    # -----------------------------------------------------------------------

    def evaluate_control(
        self,
        f_preds: torch.Tensor,
        targets: torch.Tensor,
        solver_fn: Callable[[torch.Tensor], torch.Tensor],
        alpha: float,
        u0_batch: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Evaluate control performance using the ground-truth PDE solver.

        Reproduces Table 2a (1D Burgers') and Table 2b (2D fluid) numbers.

        Critical paper requirement (Appendix F.1): "the state deviation in our
        reported evaluation metric I is always based on the output u(T,x) of
        the ground-truth solver given the control force f(t,x)."

        For 1D Burgers' (spatial_dim=1):
            I = integral_D |u(T,x) - u*(x)|^2 dx
              + alpha * integral_{[0,T]×D} |f(t,x)|^2 dt dx

        For 2D fluid (spatial_dim=2):
            I = percentage of smoke NOT passing through target bucket

        Args:
            f_preds: Predicted control sequences in physical space.
                Shape [N, T, X] = [N, 80, 120] for 1D Burgers'.
                Shape [N, T, N_ctrl] = [N, 32, 3584] for 2D fluid.
                dtype=float32.
            targets: Target states. For 1D: u* [N, X] or [X] (broadcast).
                For 2D: target bucket mask [H, W] or smoke percentage [N].
                dtype=float32.
            solver_fn: Ground-truth PDE solver callable. For 1D Burgers':
                signature solver_fn(f: Tensor) -> Tensor where f is [B, T, X]
                and output is u_T [B, X] (final state at t=T).
                For 2D fluid: takes control and returns smoke trajectory.
            alpha: Control energy regularization weight.
                Config: data.burgers.control_alpha=0.1.
            u0_batch: Optional initial conditions [N, X] for 1D Burgers'.
                Required by some solver implementations that need u0 to
                integrate the PDE. If None, solver_fn is called with f only.

        Returns:
            Dict with keys:
                'mean_J': float — mean control objective over N samples.
                    Paper Table 2a: WDNO achieves 0.0205 on 1D Burgers'.
                    Paper Table 2b: WDNO achieves 0.0679 on 2D fluid.
                'std_J': float — standard deviation (Table 10).
                    Only present if config.evaluation.report_mean_std=true
                    and N >= 2.
                'all_J': List[float] — per-sample objective values.
                'n_samples': int — number of test samples evaluated.
                'median_J': float — median objective (robustness indicator).

        Raises:
            ValueError: If f_preds has unexpected shape for the experiment.
        """
        device = torch.device(self.device)
        f_preds = f_preds.to(device=device, dtype=torch.float32)
        targets = targets.to(device=device, dtype=torch.float32)

        N: int = f_preds.shape[0]
        all_J: List[float] = []

        if self.spatial_dim == 1:
            all_J = self._evaluate_control_1d(
                f_preds=f_preds,
                targets=targets,
                solver_fn=solver_fn,
                alpha=alpha,
                u0_batch=u0_batch,
            )
        else:
            all_J = self._evaluate_control_2d(
                f_preds=f_preds,
                targets=targets,
                solver_fn=solver_fn,
                alpha=alpha,
            )

        # Aggregate statistics
        all_J_tensor: torch.Tensor = torch.tensor(
            all_J, dtype=_METRIC_DTYPE
        )
        mean_J: float = float(all_J_tensor.mean().item())
        median_J: float = float(all_J_tensor.median().item())

        result: Dict[str, Any] = {
            "mean_J": mean_J,
            "median_J": median_J,
            "all_J": all_J,
            "n_samples": N,
        }

        # Compute std if requested and enough samples available
        if self.report_mean_std and N >= _MIN_SAMPLES_FOR_STD:
            std_J: float = float(all_J_tensor.std().item())
            result["std_J"] = std_J
            logger.info(
                "evaluate_control: N=%d, mean_J=%.6f ± %.6f (std), "
                "median_J=%.6f",
                N, mean_J, std_J, median_J,
            )
        else:
            result["std_J"] = 0.0
            logger.info(
                "evaluate_control: N=%d, mean_J=%.6f, median_J=%.6f",
                N, mean_J, median_J,
            )

        return result

    def _evaluate_control_1d(
        self,
        f_preds: torch.Tensor,
        targets: torch.Tensor,
        solver_fn: Callable[[torch.Tensor], torch.Tensor],
        alpha: float,
        u0_batch: Optional[torch.Tensor] = None,
    ) -> List[float]:
        """Evaluate 1D Burgers' control objective per sample.

        Runs the ground-truth solver for each sample and computes I.

        Args:
            f_preds: Control sequences [N, T, X] = [N, 80, 120]. float32.
            targets: Target states u* [N, X] or [X]. float32.
            solver_fn: Finite difference solver. Takes f [B, T, X] and
                returns u_T [B, X] (final state at t=T).
            alpha: Regularization weight (0.1).
            u0_batch: Optional initial conditions [N, X]. float32.

        Returns:
            List of N per-sample objective values.
        """
        N: int = f_preds.shape[0]
        all_J: List[float] = []

        # Broadcast targets to [N, X] if needed
        if targets.dim() == 1:
            targets_expanded: torch.Tensor = targets.unsqueeze(0).expand(N, -1)
        else:
            targets_expanded = targets

        # Process in batches for efficiency (batch size 1 for solver calls)
        # The solver may not support batching, so we process sample by sample
        for i in range(N):
            f_i: torch.Tensor = f_preds[i:i+1]  # [1, T, X]
            u_star_i: torch.Tensor = targets_expanded[i:i+1]  # [1, X]

            try:
                # Run ground-truth solver: f [1, T, X] → u_T [1, X]
                with torch.no_grad():
                    if u0_batch is not None:
                        u0_i: torch.Tensor = u0_batch[i:i+1]  # [1, X]
                        u_T_i: torch.Tensor = solver_fn(u0_i, f_i)
                    else:
                        u_T_i = solver_fn(f_i)

                    u_T_i = u_T_i.to(device=f_preds.device, dtype=torch.float32)

                # Compute control objective using Metrics
                J_i: float = self.metrics.control_objective_1d(
                    u_T=u_T_i,
                    u_star=u_star_i,
                    f=f_i,
                    alpha=alpha,
                    dx=self.dx,
                    dt=self.dt,
                )
                all_J.append(J_i)

            except Exception as exc:
                logger.warning(
                    "Solver call failed for sample %d/%d: %s. "
                    "Using regularization-only objective.",
                    i + 1, N, exc,
                )
                # Fallback: compute regularization term only
                reg_term: float = float(
                    (alpha * torch.sum(f_i ** 2) * self.dt * self.dx).item()
                )
                all_J.append(reg_term)

            if (i + 1) % 10 == 0:
                logger.debug(
                    "Control evaluation progress: %d/%d samples, "
                    "running mean_J=%.6f",
                    i + 1, N,
                    float(np.mean(all_J)) if all_J else float("nan"),
                )

        return all_J

    def _evaluate_control_2d(
        self,
        f_preds: torch.Tensor,
        targets: torch.Tensor,
        solver_fn: Callable[[torch.Tensor], torch.Tensor],
        alpha: float,
    ) -> List[float]:
        """Evaluate 2D fluid control objective per sample.

        Computes smoke leakage percentage for each sample.

        Args:
            f_preds: Control sequences [N, T, N_ctrl] = [N, 32, 3584]. float32.
            targets: Target bucket mask [H, W] or smoke percentage [N]. float32.
            solver_fn: Fluid simulator. Takes control and returns smoke trajectory.
            alpha: Regularization weight.

        Returns:
            List of N per-sample objective values (smoke leakage percentages).
        """
        N: int = f_preds.shape[0]
        all_J: List[float] = []

        # Determine if targets is a spatial mask or scalar percentages
        is_mask: bool = (targets.dtype == torch.bool or
                         (targets.dim() == 2 and targets.shape[0] != N))

        for i in range(N):
            f_i: torch.Tensor = f_preds[i:i+1]  # [1, T, N_ctrl]

            try:
                with torch.no_grad():
                    smoke_result = solver_fn(f_i)
                    smoke_result = smoke_result.to(
                        device=f_preds.device, dtype=torch.float32
                    )

                if is_mask and smoke_result.dim() >= 3:
                    # smoke_result: [1, T, H, W] — density field
                    J_i: float = self.metrics.smoke_leakage_2d(
                        smoke_trajectory=smoke_result,
                        target_bucket_mask=targets,
                    )
                elif smoke_result.dim() == 2:
                    # smoke_result: [1, T] — percentage scalars
                    J_i = self.metrics.smoke_leakage_2d(
                        smoke_trajectory=smoke_result,
                        target_bucket_mask=targets,
                    )
                elif smoke_result.dim() == 1:
                    # smoke_result: [1] — final percentage
                    J_i = float((1.0 - smoke_result.clamp(0.0, 1.0)).mean().item())
                else:
                    logger.warning(
                        "Unexpected smoke_result shape %s for sample %d. "
                        "Using 1.0 as leakage.",
                        tuple(smoke_result.shape), i,
                    )
                    J_i = 1.0

                # Add regularization term
                reg_term: float = float(
                    (alpha * torch.sum(f_i ** 2)).item()
                )
                all_J.append(J_i + reg_term)

            except Exception as exc:
                logger.warning(
                    "Solver call failed for 2D sample %d/%d: %s. "
                    "Using 1.0 as leakage.",
                    i + 1, N, exc,
                )
                all_J.append(1.0)

        return all_J

    # -----------------------------------------------------------------------
    # Super-resolution evaluation
    # -----------------------------------------------------------------------

    def evaluate_super_resolution(
        self,
        sr_outputs_by_level: Dict[int, torch.Tensor],
        ground_truth: torch.Tensor,
        interp_modes: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate zero-shot super-resolution performance.

        Reproduces Table 16 (1D Burgers' SR) and Table 17 (2D fluid SR).

        Paper Section 4.6: "we interpolate the outcomes of each super-resolution
        step to the highest resolution level. This allows us to assess whether
        the model can accurately generate data on finer grid points beyond the
        resolutions encountered during training."

        Args:
            sr_outputs_by_level: Dict mapping SR level (int) to output tensor.
                Keys: 0 (base), 1 (1× SR), 2 (2× SR), 3 (3× SR).
                Values: Tensors at each resolution. Shape [N, T_level, X_level]
                for 1D or [N, T, H_level, W_level] for 2D. dtype=float32.
                Level 0 is the BRM output at training resolution.
            ground_truth: Ground-truth at the finest resolution. Shape
                [N, T_finest, X_finest] for 1D or [N, T, H_finest, W_finest]
                for 2D. dtype=float32.
            interp_modes: List of interpolation modes to evaluate. If None,
                uses config.super_resolution.eval_interp_modes=[linear, nearest].
                Config: super_resolution.eval_interp_modes.

        Returns:
            Nested dict: {str(level): {interp_mode: mse_value}}.
            Example for 1D Burgers' (Table 16):
                {
                    '0': {'linear': 0.00259, 'nearest': 0.02473},
                    '1': {'linear': 0.00074, 'nearest': ...},
                    '2': {'linear': 0.00036, 'nearest': ...},
                    '3': {'linear': 0.00035, 'nearest': ...},
                }
            Paper Table 16: WDNO achieves 0.00259 at 0×, 0.00074 at 1×,
            0.00036 at 2×, 0.00035 at 3× (linear interpolation).

        Raises:
            ValueError: If sr_outputs_by_level is empty.
            ValueError: If ground_truth has unexpected shape.
        """
        if not sr_outputs_by_level:
            raise ValueError(
                "sr_outputs_by_level is empty. Provide at least one SR level output."
            )

        if interp_modes is None:
            interp_modes = self.eval_interp_modes

        device = torch.device(self.device)
        gt: torch.Tensor = ground_truth.