## Code: evaluation/benchmark.py
```python
## evaluation/benchmark.py
"""Benchmark orchestration for the Robotic World Model (RWM) project.

This module provides the ``Benchmark`` class, which coordinates all evaluation
experiments described in the paper:

  - Section 4.1 / Fig. 3a: Autoregressive trajectory prediction accuracy
  - Section 4.2 / Fig. 3b: Noise robustness (RWM vs MLP baseline)
  - Section 4.3 / Fig. 4:  Generality across robotic environments
  - Section 4.4 / Fig. 5:  Policy learning comparison (MBPO-PPO vs baselines)
  - Section A.4.1 / Fig. S8: Horizon ablation heatmaps

The class is stateless between experiments — each ``run_*`` method is
self-contained and can be called independently. Models are registered via
``register_model`` before running benchmarks.

Usage:
    benchmark = Benchmark(config=cfg, logger=logger)
    benchmark.register_model("RWM-AR", rwm_model)
    benchmark.register_model("MLP", mlp_model)
    benchmark.register_model("RSSM", rssm_model)
    benchmark.register_model("Transformer", transformer_model)
    benchmark.register_model("RWM-TF", rwm_tf_model)

    # Run individual experiments:
    results = benchmark.run_prediction_benchmark(test_dataset)
    noise_results = benchmark.run_noise_robustness(rwm, mlp, test_data, noise_levels)
    ablation_results = benchmark.run_horizon_ablation(replay_buffer, m_values, n_values)
    policy_results = benchmark.run_policy_comparison(env, methods)
"""

import copy
import os
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from data.replay_buffer import ReplayBuffer
from data.trajectory_dataset import TrajectoryDataset
from envs.base_env import BaseEnv
from evaluation.metrics import Metrics
from evaluation.visualizer import Visualizer
from models.rwm import GRUWorldModel
from utils.common import get_device, set_seed
from utils.logger import Logger

# ---------------------------------------------------------------------------
# Default evaluation forecast horizon for long-horizon robustness testing.
# The paper's Fig. 3a and Fig. 4 show rollouts beyond the training horizon
# N=8 to demonstrate long-horizon stability. We use 100 steps to match
# the MBPO-PPO imagination horizon (Table S11: imagination_steps=100).
# ---------------------------------------------------------------------------
_DEFAULT_EVAL_FORECAST_STEPS: int = 100

# Default batch size for evaluation DataLoaders.
# Smaller than training batch_size=1024 to avoid OOM during long-horizon
# autoregressive rollouts (100 steps × 4096 envs would be too large).
_DEFAULT_EVAL_BATCH_SIZE: int = 256

# Fraction of replay buffer trajectories held out for evaluation in ablation.
_ABLATION_EVAL_FRACTION: float = 0.1

# Minimum number of trajectories required for a valid evaluation set.
_MIN_EVAL_TRAJECTORIES: int = 1

# Default number of policy evaluation episodes for run_policy_comparison.
_DEFAULT_EVAL_EPISODES: int = 10

# Minimum number of trajectories required to run the ablation study.
_MIN_ABLATION_TRAJECTORIES: int = 2


class Benchmark:
    """Orchestrates all evaluation experiments for the RWM paper.

    Coordinates models, metrics, and visualization to reproduce the paper's
    key figures. Each ``run_*`` method is self-contained and can be called
    independently after registering the relevant models via ``register_model``.

    The class stores a model registry (``self.models``) that maps string names
    to model instances. All registered models must implement the
    ``autoregressive_rollout(obs_history, action_history, future_actions,
    n_steps)`` interface, which is satisfied by ``GRUWorldModel``,
    ``MLPModel``, ``RSSMModel``, and ``TransformerModel``.

    Attributes:
        config: Full experiment configuration from ``config.yaml``.
        logger: Shared logger for metrics and checkpoint tracking.
        metrics: ``Metrics`` instance for prediction error computation.
        visualizer: ``Visualizer`` instance for figure generation.
        models: Dict mapping model name strings to model instances.
            Populated via ``register_model``. Keys: ``"RWM-AR"``,
            ``"RWM-TF"``, ``"MLP"``, ``"RSSM"``, ``"Transformer"``.
        device: PyTorch device for all tensor operations. Resolved from
            ``config.device`` (``"cuda"`` or ``"cpu"``).
        obs_dim: World model observation dimension. 45 for ANYmal D
            (``config.anymal_d.obs_dim``), 96 for Unitree G1
            (``config.unitree_g1.obs_dim``).
        action_dim: Action space dimension. 12 for ANYmal D, 29 for G1.
        priv_dim: Privileged information dimension. 8 for ANYmal D, 30 for G1.
        forecast_horizon: Training forecast horizon N=8 (``config.rwm.forecast_horizon``).
        history_horizon: History horizon M=32 (``config.rwm.history_horizon``).
        batch_size: Training batch size 1024 (``config.rwm_training.batch_size``).
        max_iterations: Max training iterations 2500 (``config.rwm_training.max_iterations``).
        noise_levels: Noise levels for robustness eval (``config.noise_robustness.noise_levels``).
        m_values: M values for ablation (``config.ablation.m_values``).
        n_values: N values for ablation (``config.ablation.n_values``).
    """

    def __init__(
        self,
        config: Any,
        logger: Logger,
    ) -> None:
        """Initialize the Benchmark from the experiment configuration.

        Resolves all hyperparameters from the config, initializes the
        ``Metrics`` and ``Visualizer`` instances, and prepares the model
        registry.

        Args:
            config: Hydra ``DictConfig`` or plain dict containing the full
                experiment configuration from ``config.yaml``. Must contain:
                - ``config.robot``: "anymal_d" or "unitree_g1"
                - ``config.anymal_d`` or ``config.unitree_g1``: robot sub-config
                  with ``obs_dim``, ``action_dim``, ``priv_dim``
                - ``config.rwm``: world model config with ``history_horizon=32``,
                  ``forecast_horizon=8``
                - ``config.rwm_training``: training config with
                  ``batch_size=1024``, ``max_iterations=2500``
                - ``config.noise_robustness.noise_levels``: [0.01, 0.05, 0.1, 0.2]
                - ``config.ablation.m_values``: [8, 16, 32, 64]
                - ``config.ablation.n_values``: [1, 2, 4, 8, 16]
                - ``config.device``: "cuda" or "cpu"
                - ``config.log_dir``: log directory path
            logger: Shared logger instance from ``utils/logger.py``.
                Used for metric logging throughout all benchmark runs.

        Raises:
            ValueError: If ``config.robot`` is not "anymal_d" or "unitree_g1".
            KeyError: If required config fields are missing.
        """
        # ----------------------------------------------------------------
        # 1. Store references
        # ----------------------------------------------------------------
        self.config: Any = config
        self.logger: Logger = logger

        # ----------------------------------------------------------------
        # 2. Resolve device
        # ----------------------------------------------------------------
        self.device: torch.device = get_device(str(config.device))

        # ----------------------------------------------------------------
        # 3. Resolve robot type and extract robot-specific dimensions
        # ----------------------------------------------------------------
        robot_type: str = str(config.robot)
        _supported_robots: Tuple[str, ...] = ("anymal_d", "unitree_g1")
        if robot_type not in _supported_robots:
            raise ValueError(
                f"Unsupported robot type '{robot_type}' in config.robot. "
                f"Expected one of: {_supported_robots}. "
                "Check the 'robot' field in config.yaml."
            )
        self.robot_type: str = robot_type

        # Access robot-specific sub-config (e.g., config.anymal_d)
        robot_cfg = config[robot_type]

        # Dimensions from Tables S2-S4
        self.obs_dim: int = int(robot_cfg.obs_dim)
        self.action_dim: int = int(robot_cfg.action_dim)
        self.priv_dim: int = int(robot_cfg.priv_dim)

        # ----------------------------------------------------------------
        # 4. Extract RWM architecture parameters from config.rwm (Table S10)
        # ----------------------------------------------------------------
        rwm_cfg = config.rwm

        # History horizon M = 32 (Table S10)
        self.history_horizon: int = int(rwm_cfg.history_horizon)

        # Forecast horizon N = 8 (Table S10)
        self.forecast_horizon: int = int(rwm_cfg.forecast_horizon)

        # ----------------------------------------------------------------
        # 5. Extract training parameters from config.rwm_training (Table S10)
        # ----------------------------------------------------------------
        rwm_training_cfg = config.rwm_training

        # Batch size: 1024 (Table S10)
        self.batch_size: int = int(rwm_training_cfg.batch_size)

        # Max iterations: 2500 (Table S10)
        self.max_iterations: int = int(rwm_training_cfg.max_iterations)

        # ----------------------------------------------------------------
        # 6. Extract noise robustness configuration
        # ----------------------------------------------------------------
        # noise_levels: [0.01, 0.05, 0.1, 0.2] (config.yaml)
        try:
            self.noise_levels: List[float] = [
                float(v) for v in config.noise_robustness.noise_levels
            ]
        except (AttributeError, KeyError):
            self.noise_levels = [0.01, 0.05, 0.1, 0.2]
            warnings.warn(
                "config.noise_robustness.noise_levels not found. "
                "Using default: [0.01, 0.05, 0.1, 0.2].",
                UserWarning,
                stacklevel=2,
            )

        # ----------------------------------------------------------------
        # 7. Extract ablation study configuration
        # ----------------------------------------------------------------
        # m_values: [8, 16, 32, 64] (config.yaml)
        # n_values: [1, 2, 4, 8, 16] (config.yaml)
        try:
            self.m_values: List[int] = [int(v) for v in config.ablation.m_values]
            self.n_values: List[int] = [int(v) for v in config.ablation.n_values]
        except (AttributeError, KeyError):
            self.m_values = [8, 16, 32, 64]
            self.n_values = [1, 2, 4, 8, 16]
            warnings.warn(
                "config.ablation.m_values or config.ablation.n_values not found. "
                "Using defaults: m_values=[8,16,32,64], n_values=[1,2,4,8,16].",
                UserWarning,
                stacklevel=2,
            )

        # ----------------------------------------------------------------
        # 8. Extract MBPO-PPO max iterations for policy comparison
        # ----------------------------------------------------------------
        try:
            self.ppo_max_iterations: int = int(config.mbpo_ppo.max_iterations)
        except (AttributeError, KeyError):
            self.ppo_max_iterations = 2500
            warnings.warn(
                "config.mbpo_ppo.max_iterations not found. Using default: 2500.",
                UserWarning,
                stacklevel=2,
            )

        # ----------------------------------------------------------------
        # 9. Initialize model registry
        # ----------------------------------------------------------------
        # Empty dict populated via register_model() before running benchmarks.
        # Keys: "RWM-AR", "RWM-TF", "MLP", "RSSM", "Transformer"
        self.models: Dict[str, Any] = {}

        # ----------------------------------------------------------------
        # 10. Initialize Metrics instance
        # ----------------------------------------------------------------
        self.metrics: Metrics = Metrics()

        # ----------------------------------------------------------------
        # 11. Initialize Visualizer with figures subdirectory
        # ----------------------------------------------------------------
        # Save figures to {config.log_dir}/figures/ for organization.
        try:
            log_dir: str = str(config.log_dir)
        except (AttributeError, KeyError):
            log_dir = "logs"
            warnings.warn(
                "config.log_dir not found. Using default: 'logs'.",
                UserWarning,
                stacklevel=2,
            )

        figures_dir: str = os.path.join(log_dir, "figures")
        self.visualizer: Visualizer = Visualizer(save_dir=figures_dir)

        print(
            f"[Benchmark] Initialized for robot '{self.robot_type}'. "
            f"obs_dim={self.obs_dim}, action_dim={self.action_dim}, "
            f"priv_dim={self.priv_dim}, device={self.device}. "
            f"Figures will be saved to: {figures_dir}"
        )

    # ----------------------------------------------------------------
    # Public interface methods
    # ----------------------------------------------------------------

    def register_model(self, name: str, model: Any) -> None:
        """Register a world model for use in benchmark experiments.

        Stores the model in the internal registry under the given name.
        The model must implement the ``autoregressive_rollout`` interface:
            ``model.autoregressive_rollout(obs_history, action_history,
            future_actions, n_steps) -> Tuple[Tensor, Tensor, Tensor, Tensor]``

        This interface is satisfied by ``GRUWorldModel``, ``MLPModel``,
        ``RSSMModel``, and ``TransformerModel``.

        Models should be registered before calling any ``run_*`` method.
        Registering a model with an existing name overwrites the previous
        entry (useful for updating a model after fine-tuning).

        Args:
            name: String identifier for the model. Recommended names matching
                the paper's terminology: ``"RWM-AR"``, ``"RWM-TF"``,
                ``"MLP"``, ``"RSSM"``, ``"Transformer"``. Used as keys in
                result dicts and as labels in figures.
            model: Model instance with ``autoregressive_rollout`` method.
                Must already be on the target device (call ``.to(device)``
                before registering). The model's training/eval mode is
                managed by the benchmark methods.
        """
        if not hasattr(model, "autoregressive_rollout"):
            warnings.warn(
                f"[Benchmark.register_model] Model '{name}' does not have an "
                "'autoregressive_rollout' method. Benchmark methods that call "
                "autoregressive_rollout will fail for this model. "
                "Ensure the model implements the required interface.",
                UserWarning,
                stacklevel=2,
            )
        self.models[name] = model

    def run_prediction_benchmark(
        self,
        test_dataset: TrajectoryDataset,
        eval_forecast_steps: int = _DEFAULT_EVAL_FORECAST_STEPS,
    ) -> Dict[str, Tensor]:
        """Benchmark autoregressive prediction accuracy across all registered models.

        Reproduces Figure 4 from the paper: relative autoregressive prediction
        error curves for all registered models across the evaluation dataset.
        Each model is evaluated with ``eval_forecast_steps`` autoregressive
        steps (default: 100, matching the MBPO-PPO imagination horizon).

        The evaluation uses the full test dataset without shuffling for
        deterministic, reproducible results. All models are evaluated in
        ``torch.no_grad()`` mode for efficiency.

        **Long-horizon evaluation:** The ``eval_forecast_steps`` parameter
        allows evaluating beyond the training forecast horizon N=8 to test
        long-horizon robustness. The model continues rolling out autoregressively
        beyond its training horizon — this is the key test of the dual-
        autoregressive mechanism's stability (Section 4.1, Fig. 3a).

        Args:
            test_dataset: ``TrajectoryDataset`` instance providing test windows.
                Must have been constructed with at least ``forecast_horizon``
                steps available per window. Typically a held-out portion of
                the replay buffer not used during training.
            eval_forecast_steps: Number of autoregressive forecast steps for
                evaluation. Default: 100 (matching MBPO-PPO imagination horizon
                from ``config.yaml mbpo_ppo.imagination_steps``). Set to
                ``config.rwm.forecast_horizon`` (8) for training-horizon
                evaluation only.

        Returns:
            A dict mapping model name (str) to relative prediction error curve
            (``Tensor[eval_forecast_steps]``). Each element ``result[k]`` is
            the normalized mean L2 error at forecast step k, averaged over
            the test dataset. Example:
                {
                    "RWM-AR": Tensor([0.02, 0.03, 0.04, ...]),  # shape [100]
                    "RWM-TF": Tensor([0.05, 0.08, 0.12, ...]),
                    "MLP":    Tensor([0.08, 0.15, 0.30, ...]),
                    "RSSM":   Tensor([0.04, 0.06, 0.09, ...]),
                }

        Raises:
            ValueError: If no models have been registered via ``register_model``.
            ValueError: If ``eval_forecast_steps`` <= 0.
        """
        # ----------------------------------------------------------------
        # 1. Validate inputs
        # ----------------------------------------------------------------
        if len(self.models) == 0:
            raise ValueError(
                "No models registered. Call register_model() before "
                "run_prediction_benchmark(). "
                "Example: benchmark.register_model('RWM-AR', rwm_model)"
            )
        if eval_forecast_steps <= 0:
            raise ValueError(
                f"eval_forecast_steps must be positive, got {eval_forecast_steps}."
            )

        # ----------------------------------------------------------------
        # 2. Build evaluation DataLoader
        # ----------------------------------------------------------------
        # shuffle=False for deterministic evaluation ordering.
        # num_workers=0 to avoid multiprocessing overhead during evaluation.
        # pin_memory=False since we're not in a training hot path.
        eval_loader: DataLoader = test_dataset.get_dataloader(
            batch_size=_DEFAULT_EVAL_BATCH_SIZE,
            shuffle=False,
        )

        # ----------------------------------------------------------------
        # 3. Evaluate each registered model
        # ----------------------------------------------------------------
        results: Dict[str, Tensor] = {}

        for model_name, model in self.models.items():
            print(
                f"[Benchmark] Evaluating model '{model_name}' with "
                f"{eval_forecast_steps} forecast steps..."
            )

            try:
                error_curve: Tensor = self._evaluate_model_on_loader(
                    model=model,
                    dataloader=eval_loader,
                    n_steps=eval_forecast_steps,
                )
                results[model_name] = error_curve

                # Log mean error for this model
                mean_error: float = error_curve.mean().item()
                self.logger.log(
                    {f"benchmark/{model_name}/mean_error": mean_error},
                    step=0,
                )
                print(
                    f"[Benchmark] '{model_name}': mean relative error = {mean_error:.4f}"
                )

            except Exception as exc:
                warnings.warn(
                    f"[Benchmark.run_prediction_benchmark] Evaluation failed for "
                    f"model '{model_name}': {exc}. Skipping this model.",
                    UserWarning,
                    stacklevel=2,
                )
                # Store a zero error curve as placeholder so the figure still renders
                results[model_name] = torch.zeros(
                    eval_forecast_steps,
                    dtype=torch.float32,
                )

        # ----------------------------------------------------------------
        # 4. Visualize results
        # ----------------------------------------------------------------
        if results:
            self.visualizer.plot_prediction_error_curves(
                errors=results,
                title=f"Autoregressive Prediction Error ({self.robot_type})",
            )

        return results

    def run_noise_robustness(
        self,
        model: Any,
        baseline: Any,
        test_data: TrajectoryDataset,
        noise_levels: Optional[List[float]] = None,
    ) -> Dict[str, Dict[float, Tensor]]:
        """Evaluate world model robustness under Gaussian noise perturbations.

        Reproduces Figure 3b from the paper: relative prediction error curves
        for RWM (yellow family) and MLP baseline (grey family) at varying
        Gaussian noise levels applied to both observations and actions.

        Both models are evaluated with the same noise perturbations for a
        fair comparison. The noise is applied to the input history (obs and
        actions) and future actions, not to the model's internal predictions.

        The paper states: "we analyze its performance under Gaussian noise
        perturbations applied to both observations and actions." (Section 4.2)

        Args:
            model: Trained RWM (``GRUWorldModel``) instance for the yellow
                curves in Fig. 3b. Must be on the target device and trained
                with autoregressive training (RWM-AR). Must implement
                ``autoregressive_rollout``.
            baseline: Trained MLP baseline (``MLPModel``) instance for the
                grey curves in Fig. 3b. Must be on the target device and
                trained with the same autoregressive objective as RWM for
                a fair comparison (Section 4.2). Must implement
                ``autoregressive_rollout``.
            test_data: ``TrajectoryDataset`` instance providing test windows
                for noise robustness evaluation. Should be a held-out set
                not used during training.
            noise_levels: List of Gaussian noise standard deviations to test.
                If ``None``, uses ``config.noise_robustness.noise_levels``
                = ``[0.01, 0.05, 0.1, 0.2]`` from ``config.yaml``.
                Default: None.

        Returns:
            A nested dict with two keys:
              - ``"rwm"``: Dict mapping noise level (float) to RWM error
                curve (``Tensor[T]``). T = eval forecast steps.
              - ``"mlp"``: Dict mapping noise level (float) to MLP error
                curve (``Tensor[T]``).
            Example:
                {
                    "rwm": {
                        0.01: Tensor([0.03, 0.04, ...]),
                        0.05: Tensor([0.05, 0.07, ...]),
                        ...
                    },
                    "mlp": {
                        0.01: Tensor([0.05, 0.09, ...]),
                        0.05: Tensor([0.10, 0.20, ...]),
                        ...
                    }
                }

        Raises:
            ValueError: If ``noise_levels`` is empty after resolution.
        """
        # ----------------------------------------------------------------
        # 1. Resolve noise levels
        # ----------------------------------------------------------------
        effective_noise_levels: List[float] = (
            noise_levels if noise_levels is not None else self.noise_levels
        )

        if not effective_noise_levels:
            raise ValueError(
                "noise_levels is empty. Provide at least one noise level or "
                "set config.noise_robustness.noise_levels in config.yaml. "
                "The paper uses [0.01, 0.05, 0.1, 0.2] (Section 4.2)."
            )

        # ----------------------------------------------------------------
        # 2. Compute noise robustness for RWM (yellow curves)
        # ----------------------------------------------------------------
        print(
            f"[Benchmark] Computing noise robustness for RWM at "
            f"{len(effective_noise_levels)} noise levels: {effective_noise_levels}"
        )

        rwm_errors: Dict[float, Tensor] = self.metrics.compute_noise_robustness(
            model=model,
            test_dataset=test_data,
            noise_levels=effective_noise_levels,
            device=str(self.device),
            n_steps=_DEFAULT_EVAL_FORECAST_STEPS,
            eval_batch_size=_DEFAULT_EVAL_BATCH_SIZE,
        )

        # ----------------------------------------------------------------
        # 3. Compute noise robustness for MLP baseline (grey curves)
        # ----------------------------------------------------------------
        print(
            f"[Benchmark] Computing noise robustness for MLP baseline at "
            f"{len(effective_noise_levels)} noise levels..."
        )

        mlp_errors: Dict[float, Tensor] = self.metrics.compute_noise_robustness(
            model=baseline,
            test_dataset=test_data,
            noise_levels=effective_noise_levels,
            device=str(self.device),
            n_steps=_DEFAULT_EVAL_FORECAST_STEPS,
            eval_batch_size=_DEFAULT_EVAL_BATCH_SIZE,
        )

        # ----------------------------------------------------------------
        # 4. Log summary statistics
        # ----------------------------------------------------------------
        for noise_level in effective_noise_levels:
            rwm_mean: float = (
                rwm_errors[noise_level].mean().item()
                if noise_level in rwm_errors and len(rwm_errors[noise_level]) > 0
                else float("nan")
            )
            mlp_mean: float = (
                mlp_errors[noise_level].mean().item()
                if noise_level in mlp_errors and len(mlp_errors[noise_level]) > 0
                else float("nan")
            )

            self.logger.log(
                {
                    f"noise_robustness/rwm/noise_{noise_level}": rwm_mean,
                    f"noise_robustness/mlp/noise_{noise_level}": mlp_mean,
                },
                step=0,
            )
            print(
                f"[Benchmark] Noise σ={noise_level}: "
                f"RWM mean error={rwm_mean:.4f}, MLP mean error={mlp_mean:.4f}"
            )

        # ----------------------------------------------------------------
        # 5. Visualize
        # ----------------------------------------------------------------
        self.visualizer.plot_noise_robustness(
            rwm_errors=rwm_errors,
            mlp_errors=mlp_errors,
            noise_levels=effective_noise_levels,
        )

        return {"rwm": rwm_errors, "mlp": mlp_errors}

    def run_horizon_ablation(
        self,
        replay_buffer: ReplayBuffer,
        m_values: Optional[List[int]] = None,
        n_values: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Run the horizon ablation study over (M, N) grid (Fig. S8).

        Reproduces Figure S8 from the paper: 2D heatmaps of relative
        autoregressive prediction error and training time as a function of
        history horizon M and forecast horizon N.

        For each (M, N) combination in the grid:
          1. Constructs a ``TrajectoryDataset`` with the given horizons
          2. Trains a fresh ``GRUWorldModel`` from scratch for ``max_iterations``
          3. Evaluates on a held-out set with a long forecast horizon (100 steps)
          4. Records the mean relative prediction error and wall-clock training time

        The paper's key findings (Section A.4.1):
          - Larger M consistently reduces prediction error (more context)
          - Larger N improves long-horizon accuracy but increases training time
          - N=1 (teacher-forcing) trains fast but performs poorly
          - Optimal trade-off: M=32, N=8 (the paper's default settings)

        **Computational cost:** This method runs ``len(m_values) × len(n_values)``
        full training runs. With the default grid (4×5=20 runs) and ~1 hour
        per run, this is a ~20-hour experiment. Progress is logged after each
        (M, N) pair so the experiment can be monitored.

        Args:
            replay_buffer: ``ReplayBuffer`` containing training trajectories.
                Must have at least ``_MIN_ABLATION_TRAJECTORIES=2`` trajectories.
                10% of trajectories are held out for evaluation.
            m_values: List of history horizon values M to sweep. If ``None``,
                uses ``config.ablation.m_values`` = ``[8, 16, 32, 64]``.
                Default: None.
            n_values: List of forecast horizon values N to sweep. If ``None``,
                uses ``config.ablation.n_values`` = ``[1, 2, 4, 8, 16]``.
                Default: None.

        Returns:
            A dict with the following keys:
              - ``"errors"``: ``np.ndarray[len_M, len_N]`` — mean relative
                prediction error for each (M, N) combination. ``np.nan`` for
                combinations that failed (OOM or other errors).
              - ``"times"``: ``np.ndarray[len_M, len_N]`` — wall-clock training
                time in minutes for each (M, N) combination.
              - ``"m_values"``: ``List[int]`` — the M values used.
              - ``"n_values"``: ``List[int]`` — the N values used.
            Example:
                {
                    "errors": np.array([[0.15, 0.10, 0.07, 0.06, 0.05],
                                        [0.12, 0.08, 0.05, 0.04, 0.04],
                                        ...]),
                    "times":  np.array([[0.5, 1.0, 2.0, 4.0, 8.0],
                                        [0.5, 1.0, 2.0, 4.0, 8.0],
                                        ...]),
                    "m_values": [8, 16, 32, 64],
                    "n_values": [1, 2, 4, 8, 16],
                }

        Raises:
            ValueError: If ``replay_buffer`` has fewer than
                ``_MIN_ABLATION_TRAJECTORIES`` trajectories.
        """
        # ----------------------------------------------------------------
        # 1. Resolve M and N values
        # ----------------------------------------------------------------
        effective_m_values: List[int] = (
            m_values if m_values is not None else self.m_values
        )
        effective_n_values: List[int] = (
            n_values if n_values is not None else self.n_values
        )

        # ----------------------------------------------------------------
        # 2. Validate replay buffer
        # ----------------------------------------------------------------
        n_total_trajs: int = len(replay_buffer)
        if n_total_trajs < _MIN_ABLATION_TRAJECTORIES:
            raise ValueError(
                f"replay_buffer has only {n_total_trajs} trajectories. "
                f"Need at least {_MIN_ABLATION_TRAJECTORIES} for the ablation study "
                "(one for training, one for evaluation). "
                "Collect more data before running the ablation."
            )

        # ----------------------------------------------------------------
        # 3. Split replay buffer into train and eval sets
        # ----------------------------------------------------------------
        all_trajs: List[Dict[str, Any]] = replay_buffer.get_all_trajectories()
        n_eval: int = max(
            _MIN_EVAL_TRAJECTORIES,
            int(len(all_tra