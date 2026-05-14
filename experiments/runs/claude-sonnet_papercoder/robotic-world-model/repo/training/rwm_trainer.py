## Code: training/rwm_trainer.py

```python
## training/rwm_trainer.py
"""RWM world model trainer implementing Algorithm 1 steps 3-4 from the paper.

This module provides ``RWMTrainer``, the training engine for the GRU-based
world model. It implements the self-supervised autoregressive training
objective from Section 3.2 of the paper:

    L = (1/N) * Σ_{k=1}^{N} α^k * [L_o(o'_{t+k}, o_{t+k}) + L_c(c'_{t+k}, c_{t+k})]

Key design decisions:
  - L_o: Gaussian NLL for observations (trains both mean and variance)
  - L_c: BCE with logits for binary contacts + Gaussian NLL for continuous priv
  - Gradient clipping at max_norm=1.0 for GRU stability over N=8 AR steps
  - CosineAnnealingLR scheduler over 2500 epochs
  - num_iterations treated as epochs (full dataset passes), not gradient steps

Training parameters (Table S10):
  - Learning rate: 1e-4
  - Weight decay: 1e-5
  - Batch size: 1024
  - History horizon M: 32
  - Forecast horizon N: 8
  - Forecast decay α: 1.0

Usage:
    trainer = RWMTrainer(model, config, logger)
    trainer.train(replay_buffer, num_iterations=2500)
    # Or for fine-tuning in MBPO-PPO:
    metrics = trainer.train_epoch(dataloader)
"""

import math
import os
import warnings
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.replay_buffer import ReplayBuffer
from data.trajectory_dataset import TrajectoryDataset
from models.rwm import GRUWorldModel
from utils.common import get_device
from utils.logger import Logger

# ---------------------------------------------------------------------------
# Log-std clamping bounds for numerical stability in _gaussian_nll.
# Consistent with GRUWorldModel's clamping range.
# Range [-10, 2] corresponds to std in [exp(-10), exp(2)] ≈ [4.5e-5, 7.4].
# ---------------------------------------------------------------------------
_LOGSTD_MIN: float = -10.0
_LOGSTD_MAX: float = 2.0

# Maximum gradient norm for gradient clipping.
# Essential for GRU training with N=8 autoregressive steps where gradients
# can grow exponentially through the unrolled computation graph.
_MAX_GRAD_NORM: float = 1.0

# Log interval default (iterations between logging)
_DEFAULT_LOG_INTERVAL: int = 10

# Save interval default (iterations between checkpoint saves)
_DEFAULT_SAVE_INTERVAL: int = 500

# Default checkpoint directory
_DEFAULT_CHECKPOINT_DIR: str = "checkpoints"


class RWMTrainer:
    """Training engine for the GRU-based Robotic World Model.

    Implements Algorithm 1 steps 3-4 from the paper: training the world
    model p_φ on replay buffer data using the self-supervised autoregressive
    objective (Section 3.2, Eq. 2).

    The trainer handles:
      - Autoregressive forward pass through GRUWorldModel
      - Multi-step prediction loss with Gaussian NLL (observations) and
        BCE/NLL (privileged information)
      - Adam optimizer with CosineAnnealingLR scheduler
      - Gradient clipping for GRU stability
      - Checkpoint saving and loading
      - Metric logging via the shared Logger

    The same trainer instance is used for both:
      1. Pretraining on 6M simulation transitions (Algorithm 1, step 4 in
         the pretraining phase)
      2. Online fine-tuning during MBPO-PPO (Algorithm 1, step 4 in the
         policy optimization loop)

    Attributes:
        model: The GRU world model being trained.
        config: Full experiment configuration from config.yaml.
        logger: Shared logger for metrics and checkpoints.
        device: Target device (CUDA or CPU) for all tensor operations.
        optimizer: Adam optimizer for the world model parameters.
        scheduler: CosineAnnealingLR scheduler decaying over max_iterations.
        obs_dim: World model observation dimension (45 for ANYmal D, 96 for G1).
        priv_dim: Privileged information dimension (8 for ANYmal D, 30 for G1).
        history_horizon: Number of historical steps M for GRU context (32).
        forecast_horizon: Number of forecast steps N for outer AR (8).
        forecast_decay: Decay factor α for multi-step loss weighting (1.0).
        binary_priv_end: Index where binary priv dims end. 8 for ANYmal D
            (all contacts), 26 for Unitree G1 (body contacts only).
        global_step: Total number of gradient steps taken across all epochs.
        best_loss: Best (lowest) training loss observed, for checkpoint saving.
    """

    def __init__(
        self,
        model: GRUWorldModel,
        config: Any,
        logger: Logger,
    ) -> None:
        """Initialize the RWM trainer from the experiment configuration.

        Resolves all training hyperparameters from the config, initializes
        the Adam optimizer and CosineAnnealingLR scheduler, and determines
        the privileged information type split (binary vs continuous) for
        the correct loss function selection.

        Args:
            model: Instantiated GRU world model from ``models/rwm.py``.
                Must already be on the target device (call ``.to(device)``
                before passing to this constructor).
            config: Hydra ``DictConfig`` or plain dict containing the full
                experiment configuration from ``config.yaml``. Must contain:
                - ``config.robot``: "anymal_d" or "unitree_g1"
                - ``config.anymal_d`` or ``config.unitree_g1``: robot sub-config
                  with ``obs_dim``, ``priv_dim``, ``priv_slices``
                - ``config.rwm``: world model architecture config with
                  ``history_horizon=32``, ``forecast_horizon=8``,
                  ``forecast_decay=1.0``
                - ``config.rwm_training``: training hyperparameters with
                  ``learning_rate=1e-4``, ``weight_decay=1e-5``,
                  ``batch_size=1024``, ``max_iterations=2500``
                - ``config.device``: "cuda" or "cpu"
                - ``config.log_dir``: log directory path
                - ``config.checkpoint_dir``: checkpoint directory path
                - ``config.logging``: logging config with ``log_interval``,
                  ``save_interval``
            logger: Shared logger instance from ``utils/logger.py``.
                Used for metric logging and model summary output.

        Raises:
            ValueError: If ``config.robot`` is not "anymal_d" or "unitree_g1".
            KeyError: If required config fields are missing.
        """
        # ----------------------------------------------------------------
        # 1. Store references
        # ----------------------------------------------------------------
        self.model: GRUWorldModel = model
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
        _supported_robots = ("anymal_d", "unitree_g1")
        if robot_type not in _supported_robots:
            raise ValueError(
                f"Unsupported robot type '{robot_type}' in config.robot. "
                f"Expected one of: {_supported_robots}. "
                "Check the 'robot' field in config.yaml."
            )
        self.robot_type: str = robot_type

        # Access robot-specific sub-config (e.g., config.anymal_d)
        robot_cfg = config[robot_type]

        # Dimensions from Tables S2-S3
        self.obs_dim: int = int(robot_cfg.obs_dim)
        self.priv_dim: int = int(robot_cfg.priv_dim)

        # ----------------------------------------------------------------
        # 4. Determine privileged info type split (binary vs continuous)
        # ----------------------------------------------------------------
        # This drives the loss function selection in _compute_loss:
        #   - Binary dims (contacts): BCE with logits
        #   - Continuous dims (foot height/velocity for G1): Gaussian NLL
        #
        # ANYmal D (priv_dim=8, Table S3):
        #   - knee_contact: [0, 4] — binary
        #   - foot_contact: [4, 8] — binary
        #   → binary_priv_end = 8 (all priv is binary)
        #
        # Unitree G1 (priv_dim=30, Table S3):
        #   - body_contact: [0, 26] — binary
        #   - foot_height: [26, 28] — continuous
        #   - foot_velocity: [28, 30] — continuous
        #   → binary_priv_end = 26 (first 26 dims are binary)
        priv_slices = robot_cfg.priv_slices

        if robot_type == "anymal_d":
            # All 8 priv dims are binary contacts (Table S3)
            self.binary_priv_end: int = self.priv_dim  # 8
        else:
            # Unitree G1: body_contact ends at index 26 (Table S3)
            # body_contact: [0, 26] → binary_priv_end = 26
            body_contact_slice = priv_slices.body_contact
            # body_contact_slice is [start, end] from config.yaml
            self.binary_priv_end = int(body_contact_slice[1])  # 26

        # Continuous priv starts where binary ends
        self.continuous_priv_start: int = self.binary_priv_end

        # ----------------------------------------------------------------
        # 5. Extract training horizons from config.rwm (Table S10)
        # ----------------------------------------------------------------
        rwm_cfg = config.rwm

        # History horizon M = 32 (Table S10)
        self.history_horizon: int = int(rwm_cfg.history_horizon)

        # Forecast horizon N = 8 (Table S10)
        self.forecast_horizon: int = int(rwm_cfg.forecast_horizon)

        # Forecast decay α = 1.0 (Table S10: no decay)
        self.forecast_decay: float = float(rwm_cfg.forecast_decay)

        # ----------------------------------------------------------------
        # 6. Extract training hyperparameters from config.rwm_training (Table S10)
        # ----------------------------------------------------------------
        rwm_training_cfg = config.rwm_training

        # Learning rate: 1e-4 (Table S10)
        learning_rate: float = float(rwm_training_cfg.learning_rate)

        # Weight decay: 1e-5 (Table S10)
        weight_decay: float = float(rwm_training_cfg.weight_decay)

        # Batch size: 1024 (Table S10)
        self.batch_size: int = int(rwm_training_cfg.batch_size)

        # Max iterations (epochs): 2500 (Table S10)
        self.max_iterations: int = int(rwm_training_cfg.max_iterations)

        # ----------------------------------------------------------------
        # 7. Initialize Adam optimizer (Table S10: optimizer="adam")
        # ----------------------------------------------------------------
        self.optimizer: torch.optim.Adam = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # ----------------------------------------------------------------
        # 8. Initialize CosineAnnealingLR scheduler
        # ----------------------------------------------------------------
        # T_max = max_iterations = 2500 epochs
        # The scheduler decays LR from learning_rate to 0 over 2500 epochs.
        # Called once per epoch (after train_epoch), not per gradient step.
        self.scheduler: torch.optim.lr_scheduler.CosineAnnealingLR = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.max_iterations,
            )
        )

        # ----------------------------------------------------------------
        # 9. Extract logging and checkpoint config
        # ----------------------------------------------------------------
        # Log interval: log every N iterations (default: 10)
        try:
            self.log_interval: int = int(config.logging.log_interval)
        except (AttributeError, KeyError):
            self.log_interval = _DEFAULT_LOG_INTERVAL

        # Save interval: save checkpoint every N iterations (default: 500)
        try:
            self.save_interval: int = int(config.logging.save_interval)
        except (AttributeError, KeyError):
            self.save_interval = _DEFAULT_SAVE_INTERVAL

        # Checkpoint directory
        try:
            self.checkpoint_dir: str = str(config.checkpoint_dir)
        except (AttributeError, KeyError):
            self.checkpoint_dir = _DEFAULT_CHECKPOINT_DIR

        # ----------------------------------------------------------------
        # 10. Initialize tracking variables
        # ----------------------------------------------------------------
        # Total gradient steps taken across all epochs (for logging x-axis)
        self.global_step: int = 0

        # Best training loss observed (for best-model checkpoint saving)
        self.best_loss: float = float("inf")

        # Log model summary to logger
        self.logger.log_model_summary(self.model)

    # ----------------------------------------------------------------
    # Private helper methods
    # ----------------------------------------------------------------

    def _gaussian_nll(
        self,
        pred_mean: Tensor,
        pred_logstd: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Compute the mean Gaussian negative log-likelihood loss.

        Implements the NLL of a diagonal Gaussian distribution:
            -log p(x | μ, σ) = 0.5 * (log(2π) + 2*logstd + ((x - μ) / σ)²)

        Used as L_o for observation predictions and L_c for continuous
        privileged information (foot height/velocity for Unitree G1).

        The loss is:
          1. Summed over the feature dimension D (treating each dim as
             independent Gaussian) — standard for multivariate diagonal Gaussian
          2. Meaned over the batch dimension B

        This gives a scalar loss that scales with the dimensionality of the
        prediction, which is appropriate for comparing losses across different
        observation spaces (45-dim ANYmal D vs 96-dim Unitree G1).

        Args:
            pred_mean: Predicted Gaussian mean of shape ``[B, D]``.
                For observations: D = obs_dim (45 or 96).
                For continuous priv: D = priv_dim - binary_priv_end.
            pred_logstd: Predicted log standard deviation of shape ``[B, D]``.
                Will be clamped to ``[_LOGSTD_MIN, _LOGSTD_MAX]`` internally
                to prevent numerical overflow in exp(logstd).
            target: Ground truth values of shape ``[B, D]``.
                For observations: ground truth world model obs from dataset.
                For continuous priv: ground truth foot heights/velocities.

        Returns:
            Scalar tensor representing the mean NLL loss over the batch.
            Retains gradient graph for backpropagation through pred_mean
            and pred_logstd.
        """
        # Clamp logstd to prevent exp(logstd) from overflowing or underflowing.
        # Without clamping, early training can produce logstd = ±100, causing
        # exp(logstd) = inf or 0, which leads to NaN losses.
        pred_logstd_clamped: Tensor = torch.clamp(
            pred_logstd,
            _LOGSTD_MIN,
            _LOGSTD_MAX,
        )

        # Compute sigma = exp(logstd) — always positive
        sigma: Tensor = torch.exp(pred_logstd_clamped)

        # Compute per-element NLL:
        # -log p(x | μ, σ) = 0.5 * (log(2π) + 2*logstd + ((x - μ) / σ)²)
        # Using the log(2π) constant for correctness (though it doesn't affect
        # gradients, it makes the loss value interpretable as actual NLL).
        log_2pi: float = math.log(2.0 * math.pi)

        # Squared normalized residual: ((target - mean) / sigma)^2
        normalized_residual_sq: Tensor = ((target - pred_mean) / sigma) ** 2

        # Per-element NLL: [B, D]
        nll_per_element: Tensor = 0.5 * (
            log_2pi
            + 2.0 * pred_logstd_clamped
            + normalized_residual_sq
        )

        # Sum over feature dimension D (independent Gaussian per dimension)
        # then mean over batch dimension B → scalar
        # Shape: [B, D] → [B] → scalar
        nll_loss: Tensor = nll_per_element.sum(dim=-1).mean()

        return nll_loss

    def _compute_loss(
        self,
        pred_obs_means: Tensor,
        pred_obs_logstds: Tensor,
        true_obs_seq: Tensor,
        pred_priv_means: Tensor,
        pred_priv_logstds: Tensor,
        true_priv_seq: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute the multi-step autoregressive prediction loss (Eq. 2).

        Implements the full loss from Section 3.2 of the paper:
            L = (1/N) * Σ_{k=1}^{N} α^k * [L_o(o'_{t+k}, o_{t+k}) + L_c(c'_{t+k}, c_{t+k})]

        where:
          - L_o: Gaussian NLL for observations (always)
          - L_c: BCE with logits for binary contacts + Gaussian NLL for
            continuous priv (foot height/velocity for Unitree G1 only)
          - α = 1.0 (no decay, Table S10), but implemented generally

        The loss is computed per forecast step k and accumulated with the
        decay weight α^k. With α=1.0, all steps contribute equally.

        **Privileged info loss split:**
          - ANYmal D (priv_dim=8): all 8 dims are binary contacts → BCE only
          - Unitree G1 (priv_dim=30): dims 0:26 binary (BCE), dims 26:30
            continuous (Gaussian NLL)

        Args:
            pred_obs_means: Predicted observation means, shape
                ``[B, N, obs_dim]``. From ``model.autoregressive_rollout``.
            pred_obs_logstds: Predicted observation log-stds, shape
                ``[B, N, obs_dim]``. From ``model.autoregressive_rollout``.
            true_obs_seq: Ground truth observations, shape
                ``[B, N, obs_dim]``. From ``batch['target_obs']``.
            pred_priv_means: Predicted privileged info means, shape
                ``[B, N, priv_dim]``. From ``model.autoregressive_rollout``.
            pred_priv_logstds: Predicted privileged info log-stds, shape
                ``[B, N, priv_dim]``. From ``model.autoregressive_rollout``.
            true_priv_seq: Ground truth privileged info, shape
                ``[B, N, priv_dim]``. From ``batch['target_priv']``.

        Returns:
            A tuple ``(total_loss, obs_loss_mean, priv_loss_mean)`` where:
              - ``total_loss``: Scalar tensor, the full weighted multi-step
                loss. Retains gradient graph for backpropagation.
              - ``obs_loss_mean``: Detached float, mean observation loss
                across all N steps. For logging only.
              - ``priv_loss_mean``: Detached float, mean privileged info
                loss across all N steps. For logging only.
        """
        # Accumulators for the weighted sum over N forecast steps
        total_loss: Tensor = torch.zeros(1, device=self.device)

        # Detached accumulators for logging (no gradient needed)
        obs_loss_accum: float = 0.0
        priv_loss_accum: float = 0.0

        for k in range(self.forecast_horizon):
            # ----------------------------------------------------------------
            # Extract per-step predictions and targets (step k, 0-indexed)
            # ----------------------------------------------------------------
            # Observation predictions and targets
            obs_mean_k: Tensor = pred_obs_means[:, k, :]    # [B, obs_dim]
            obs_logstd_k: Tensor = pred_obs_logstds[:, k, :]  # [B, obs_dim]
            obs_true_k: Tensor = true_obs_seq[:, k, :]      # [B, obs_dim]

            # Privileged info predictions and targets
            priv_mean_k: Tensor = pred_priv_means[:, k, :]    # [B, priv_dim]
            priv_logstd_k: Tensor = pred_priv_logstds[:, k, :]  # [B, priv_dim]
            priv_true_k: Tensor = true_priv_seq[:, k, :]      # [B, priv_dim]

            # ----------------------------------------------------------------
            # Observation loss L_o: Gaussian NLL (always)
            # ----------------------------------------------------------------
            obs_loss_k: Tensor = self._gaussian_nll(
                pred_mean=obs_mean_k,
                pred_logstd=obs_logstd_k,
                target=obs_true_k,
            )
            # obs_loss_k: scalar

            # ----------------------------------------------------------------
            # Privileged info loss L_c: BCE + optional Gaussian NLL
            # ----------------------------------------------------------------
            priv_loss_k: Tensor = self._compute_priv_loss(
                priv_mean_k=priv_mean_k,
                priv_logstd_k=priv_logstd_k,
                priv_true_k=priv_true_k,
            )
            # priv_loss_k: scalar

            # ----------------------------------------------------------------
            # Apply decay weight: α^(k+1) (paper uses 1-indexed k)
            # ----------------------------------------------------------------
            # With α=1.0 (Table S10), weight_k = 1.0 for all k.
            # Implemented generally for ablation studies with different α.
            weight_k: float = self.forecast_decay ** (k + 1)

            # Accumulate weighted step loss
            total_loss = total_loss + weight_k * (obs_loss_k + priv_loss_k)

            # Accumulate detached sub-losses for logging
            obs_loss_accum += obs_loss_k.detach().item()
            priv_loss_accum += priv_loss_k.detach().item()

        # ----------------------------------------------------------------
        # Normalize by N (forecast horizon)
        # ----------------------------------------------------------------
        total_loss = total_loss / self.forecast_horizon
        obs_loss_mean: float = obs_loss_accum / self.forecast_horizon
        priv_loss_mean: float = priv_loss_accum / self.forecast_horizon

        return total_loss, obs_loss_mean, priv_loss_mean

    def _compute_priv_loss(
        self,
        priv_mean_k: Tensor,
        priv_logstd_k: Tensor,
        priv_true_k: Tensor,
    ) -> Tensor:
        """Compute the privileged information loss for one forecast step.

        Handles the heterogeneous privileged information vector:
          - Binary portion (contacts): BCE with logits
          - Continuous portion (foot height/velocity, G1 only): Gaussian NLL

        The split point is determined by ``self.binary_priv_end``:
          - ANYmal D: binary_priv_end = 8 (all priv is binary)
          - Unitree G1: binary_priv_end = 26 (dims 0:26 binary, 26:30 continuous)

        **BCE with logits for binary contacts:**
        The priv head outputs raw logits (not sigmoid-activated). Using
        ``F.binary_cross_entropy_with_logits`` is numerically more stable
        than applying sigmoid then BCE. The ground truth contact values are
        binary (0.0 or 1.0) from the simulator.

        Args:
            priv_mean_k: Predicted privileged info mean (logits for binary
                dims) of shape ``[B, priv_dim]``.
            priv_logstd_k: Predicted privileged info log-std of shape
                ``[B, priv_dim]``. Only used for continuous dims (G1).
            priv_true_k: Ground truth privileged info of shape
                ``[B, priv_dim]``. Binary values (0.0/1.0) for contact dims,
                continuous values for foot height/velocity dims.

        Returns:
            Scalar tensor representing the total privileged info loss for
            this forecast step. Retains gradient graph.
        """
        # ----------------------------------------------------------------
        # Binary portion: BCE with logits for contact flags
        # ----------------------------------------------------------------
        # Slice binary dims: priv[:, :binary_priv_end]
        # ANYmal D: [:, :8] — all 8 dims
        # Unitree G1: [:, :26] — first 26 dims (body contacts)
        binary_logits: Tensor = priv_mean_k[:, : self.binary_priv_end]
        # shape: [B, binary_priv_end]

        binary_targets: Tensor = priv_true_k[:, : self.binary_priv_end].float()
        # shape: [B, binary_priv_end]

        # BCE with logits: numerically stable, handles logits directly
        # reduction='mean' averages over both batch and binary dims
        binary_loss: Tensor = F.binary_cross_entropy_with_logits(
            binary_logits,
            binary_targets,
            reduction="mean",
        )
        # binary_loss: scalar

        # ----------------------------------------------------------------
        # Continuous portion: Gaussian NLL for foot height/velocity (G1 only)
        # ----------------------------------------------------------------
        # For ANYmal D: continuous_priv_start = binary_priv_end = 8 = priv_dim
        # → no continuous dims → continuous_loss = 0
        # For Unitree G1: continuous_priv_start = 26, priv_dim = 30
        # → continuous dims: [:, 26:30] (foot height + foot velocity)
        continuous_loss: Tensor = torch.zeros(1, device=self.device)

        if self.continuous_priv_start < self.priv_dim:
            # Slice continuous dims: priv[:, binary_priv_end:]
            continuous_mean: Tensor = priv_mean_k[:, self.continuous_priv_start :]
            # shape: [B, priv_dim - binary_priv_end]

            continuous_logstd: Tensor = priv_logstd_k[:, self.continuous_priv_start :]
            # shape: [B, priv_dim - binary_priv_end]

            continuous_targets: Tensor = priv_true_k[:, self.continuous_priv_start :]
            # shape: [B, priv_dim - binary_priv_end]

            continuous_loss = self._gaussian_nll(
                pred_mean=continuous_mean,
                pred_logstd=continuous_logstd,
                target=continuous_targets,
            )
            # continuous_loss: scalar

        # Total privileged info loss for this step
        priv_loss_k: Tensor = binary_loss + continuous_loss

        return priv_loss_k

    def _autoregressive_forward(
        self,
        batch: Dict[str, Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run the autoregressive forward pass and return all prediction tensors.

        Extracts tensors from the batch dict (as produced by
        ``TrajectoryDataset.__getitem__``), moves them to the target device,
        and calls ``model.autoregressive_rollout`` to get predictions for
        all N forecast steps.

        The reparameterization trick is applied **inside**
        ``model.autoregressive_rollout`` during the outer autoregression loop
        (sampling ``next_obs = mean + eps * exp(logstd)`` for feeding back
        into the GRU). This ensures gradients flow through the stochastic
        sampling back to the model parameters. This method does not need to
        apply reparameterization itself.

        Args:
            batch: Dict from ``TrajectoryDataset.__getitem__`` with keys:
                - ``'obs_history'``: ``Tensor[B, M, obs_dim]`` on CPU
                - ``'action_history'``: ``Tensor[B, M, action_dim]`` on CPU
                - ``'target_obs'``: ``Tensor[B, N, obs_dim]`` on CPU
                - ``'target_priv'``: ``Tensor[B, N, priv_dim]`` on CPU
                - ``'future_actions'``: ``Tensor[B, N, action_dim]`` on CPU

        Returns:
            A tuple of four tensors, all on ``self.device``:
              - ``pred_obs_means``: ``[B, N, obs_dim]`` — predicted obs means
              - ``pred_obs_logstds``: ``[B, N, obs_dim]`` — predicted obs log-stds
              - ``pred_priv_means``: ``[B, N, priv_dim]`` — predicted priv means
              - ``pred_priv_logstds``: ``[B, N, priv_dim]`` — predicted priv log-stds
        """
        # ----------------------------------------------------------------
        # 1. Move batch tensors to target device
        # ----------------------------------------------------------------
        # DataLoader produces CPU tensors (pin_memory=True in TrajectoryDataset
        # speeds up the CPU→GPU transfer but doesn't move to GPU automatically).
        obs_history: Tensor = batch["obs_history"].to(self.device)
        # shape: [B, M, obs_dim]

        action_history: Tensor = batch["action_history"].to(self.device)
        # shape: [B, M, action_dim]

        future_actions: Tensor = batch["future_actions"].to(self.device)
        # shape: [B, N, action_dim]

        # ----------------------------------------------------------------
        # 2. Call model.autoregressive_rollout
        # ----------------------------------------------------------------
        # The model implements the dual-autoregressive mechanism:
        #   - Inner AR: GRU processes M=32 history steps → hidden state h
        #   - Outer AR: N=8 forecast steps, each sampling next_obs via
        #     reparameterization and feeding back into the GRU
        #
        # Returns four tensors: (means, logstds) for both obs and priv.
        # This is the extended return signature needed for the NLL loss.
        pred_obs_means: Tensor
        pred_obs_logstds: Tensor
        pred_priv_means: Tensor
        pred_priv_logstds: Tensor

        pred_obs_means, pred_obs_logstds, pred_priv_means, pred_priv_logstds = (
            self.model.autoregressive_rollout(
                obs_history=obs_history,
                action_history=action_history,
                future_actions=future_actions,
                n_steps=self.forecast_