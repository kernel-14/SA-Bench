## Code: evaluation/metrics.py
```python
## evaluation/metrics.py
"""Evaluation metrics for the Robotic World Model (RWM) project.

This module provides the ``Metrics`` class with static methods for quantifying
world model prediction accuracy and policy performance. It is consumed by:
  - ``evaluation/benchmark.py``: benchmark loops for prediction accuracy and
    noise robustness experiments (Sections 4.1, 4.2, 4.3, Fig. 3, 4, S8)
  - ``training/rwm_trainer.py``: validation loss reporting during world model
    training
  - ``training/mbpo_ppo_trainer.py``: policy evaluation and training curve
    smoothing (Fig. 5)

All methods are ``@staticmethod`` — the class is a pure namespace with no
instance state. Methods operate on PyTorch tensors and return either tensors
(for further GPU computation) or numpy arrays (for matplotlib plotting).

The relative prediction error metric ``e_k`` is the primary evaluation metric
used throughout the paper:
    e_k = mean_batch(||pred[:,k,:] - target[:,k,:]||_2) / (mean_batch(||target[:,k,:]||_2) + eps)

This normalization makes the metric comparable across different observation
spaces (45-dim ANYmal D vs 96-dim Unitree G1) and different forecast steps.
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Small epsilon for numerical stability in relative error normalization.
# Added to the denominator only to prevent division by zero when the target
# norm is near zero (e.g., stationary robot with near-zero velocities).
_EPS: float = 1e-8

# Moving average window size for smoothing training curves (Fig. 5 left plot).
# Specified in the design document as window=10.
_SMOOTHING_WINDOW: int = 10

# Default batch size for noise robustness evaluation DataLoader.
# Smaller than training batch_size=1024 to avoid OOM during long-horizon
# autoregressive rollouts with noise perturbations.
_EVAL_BATCH_SIZE: int = 256


class Metrics:
    """Static metric computation methods for world model and policy evaluation.

    All methods are ``@staticmethod`` — instantiate this class only if you
    need to group metric calls under a single object reference. The class
    holds no state.

    The primary metric is ``relative_prediction_error``, which computes the
    normalized L2 prediction error per forecast step. This is the ``e_k``
    metric used in all figures and tables of the paper.

    Usage:
        # World model evaluation:
        error_curve = Metrics.relative_prediction_error(pred_obs, target_obs)
        # shape: [T] — one scalar per forecast step

        # Policy evaluation:
        mean_r, std_r = Metrics.mean_tracking_reward(episode_rewards)

        # Noise robustness experiment (Section 4.2):
        results = Metrics.compute_noise_robustness(
            model=rwm, test_dataset=dataset,
            noise_levels=[0.01, 0.05, 0.1, 0.2], device="cuda"
        )
    """

    def __init__(self) -> None:
        """Initialize Metrics (no-op — all methods are static).

        This constructor exists only for interface compliance. The class
        holds no instance state. All methods can be called directly on the
        class without instantiation: ``Metrics.relative_prediction_error(...)``.
        """
        pass

    @staticmethod
    def relative_prediction_error(
        pred: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Compute the relative autoregressive prediction error per forecast step.

        This is the primary evaluation metric ``e_k`` used throughout the paper
        (Sections 4.1, 4.2, 4.3, Fig. 3, 4, S8). It measures how much the
        model's autoregressive predictions deviate from ground truth, normalized
        by the magnitude of the ground truth to make the metric comparable
        across different observation spaces and forecast horizons.

        Formula:
            e_k = mean_B(||pred[:,k,:] - target[:,k,:]||_2)
                  / (mean_B(||target[:,k,:]||_2) + eps)

        where:
          - B is the batch dimension (number of test trajectories)
          - k is the forecast step index (0 to T-1)
          - D is the observation feature dimension (obs_dim)
          - eps = 1e-8 for numerical stability

        The normalization is applied *after* averaging over the batch (not
        per-sample) to avoid instability when individual target norms are
        near zero. This matches the formula in the design specification.

        **Gradient flow:** This method does NOT call ``.detach()``. If called
        with ``requires_grad=True`` tensors during training, gradients will
        flow. Callers in training loops should use ``torch.no_grad()`` context
        or call ``.detach()`` before passing tensors to this method.

        Args:
            pred: Predicted observation sequences of shape ``[B, T, D]``.
                B = batch size (number of test trajectories or environments),
                T = number of forecast steps (e.g., N=8 during training,
                100+ during long-horizon evaluation),
                D = observation feature dimension (45 for ANYmal D, 96 for G1).
                May be on any device (CPU or CUDA).
            target: Ground truth observation sequences of shape ``[B, T, D]``.
                Must have the same shape as ``pred`` and be on the same device.

        Returns:
            Relative prediction error tensor of shape ``[T]``. Each element
            ``result[k]`` is the normalized mean L2 error at forecast step k,
            averaged over the batch. Values are non-negative floats.
            The tensor is on the same device as the input tensors.

        Raises:
            ValueError: If ``pred`` and ``target`` have different shapes.
            ValueError: If ``pred`` is not 3-dimensional (must be [B, T, D]).

        Example:
            >>> pred = torch.randn(32, 8, 45)   # 32 trajectories, 8 steps, 45-dim
            >>> target = torch.randn(32, 8, 45)
            >>> error = Metrics.relative_prediction_error(pred, target)
            >>> error.shape
            torch.Size([8])
        """
        # ----------------------------------------------------------------
        # 1. Validate input shapes
        # ----------------------------------------------------------------
        if pred.ndim != 3:
            raise ValueError(
                f"pred must be 3-dimensional [B, T, D], got shape {tuple(pred.shape)}. "
                "Ensure the prediction tensor has batch, time, and feature dimensions."
            )
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must have the same shape. "
                f"Got pred.shape={tuple(pred.shape)}, target.shape={tuple(target.shape)}."
            )

        # ----------------------------------------------------------------
        # 2. Compute per-sample L2 norm of the prediction error at each step k
        # ----------------------------------------------------------------
        # pred - target: [B, T, D] — element-wise difference
        # torch.norm(..., dim=-1): [B, T] — L2 norm over the D dimension
        # Each element [b, k] is ||pred[b,k,:] - target[b,k,:]||_2
        error_norms: Tensor = torch.norm(pred - target, dim=-1)
        # shape: [B, T]

        # ----------------------------------------------------------------
        # 3. Compute per-sample L2 norm of the target at each step k
        # ----------------------------------------------------------------
        # torch.norm(target, dim=-1): [B, T] — L2 norm over the D dimension
        # Each element [b, k] is ||target[b,k,:]||_2
        target_norms: Tensor = torch.norm(target, dim=-1)
        # shape: [B, T]

        # ----------------------------------------------------------------
        # 4. Average both over the batch dimension (dim=0)
        # ----------------------------------------------------------------
        # mean_error[k] = (1/B) * Σ_b ||pred[b,k,:] - target[b,k,:]||_2
        mean_error: Tensor = error_norms.mean(dim=0)
        # shape: [T]

        # mean_target[k] = (1/B) * Σ_b ||target[b,k,:]||_2
        mean_target: Tensor = target_norms.mean(dim=0)
        # shape: [T]

        # ----------------------------------------------------------------
        # 5. Compute relative error with numerical stability
        # ----------------------------------------------------------------
        # e_k = mean_error[k] / (mean_target[k] + eps)
        # eps = 1e-8 prevents division by zero when target norms are near zero
        # (e.g., stationary robot with near-zero velocities at all joints)
        result: Tensor = mean_error / (mean_target + _EPS)
        # shape: [T]

        return result

    @staticmethod
    def per_step_error(
        pred_sequence: Tensor,
        target_sequence: Tensor,
    ) -> Tensor:
        """Compute the relative prediction error per forecast step (alias).

        This is a semantic alias for ``relative_prediction_error``. It exists
        for readability at call sites in ``evaluation/benchmark.py`` where the
        "per step" nature of the output is the primary concern, rather than
        the normalization semantics.

        See ``relative_prediction_error`` for full documentation.

        Args:
            pred_sequence: Predicted observation sequences of shape
                ``[B, T, D]``. Same as ``pred`` in ``relative_prediction_error``.
            target_sequence: Ground truth observation sequences of shape
                ``[B, T, D]``. Same as ``target`` in ``relative_prediction_error``.

        Returns:
            Relative prediction error tensor of shape ``[T]``. Identical to
            the output of ``relative_prediction_error(pred_sequence, target_sequence)``.

        Example:
            >>> errors = Metrics.per_step_error(pred_obs, target_obs)
            >>> errors.shape
            torch.Size([8])  # T=8 forecast steps
        """
        return Metrics.relative_prediction_error(pred_sequence, target_sequence)

    @staticmethod
    def mean_tracking_reward(
        rewards: Tensor,
    ) -> Tuple[float, float]:
        """Compute mean and standard deviation of tracking reward across episodes.

        Summarizes policy performance for reporting in Section 4.4 (Fig. 5
        right plot) and Table 1 (real tracking reward: ``0.90 ± 0.04``).

        The reward is averaged over time steps within each episode (not summed)
        to produce a scale-invariant per-episode mean reward. Then the mean
        and standard deviation are computed over episodes.

        **Why mean over T (not sum):** The paper reports "mean reward" not
        "return". Using mean over T makes the metric comparable across episodes
        of different lengths and is consistent with the reward formulation in
        Section A.1.2 where individual reward terms are bounded (e.g., linear
        velocity tracking reward is in [0, 1.0]).

        Args:
            rewards: Reward tensor of shape ``[N_episodes, T]``.
                N_episodes = number of evaluation episodes,
                T = number of timesteps per episode.
                Values are per-step rewards from the environment or world model.
                May be on any device (CPU or CUDA).

        Returns:
            A tuple ``(mean_reward, std_reward)`` where:
              - ``mean_reward``: Mean of per-episode mean rewards, as a Python
                float. This is the primary performance metric.
              - ``std_reward``: Standard deviation of per-episode mean rewards,
                as a Python float. Returns ``0.0`` if ``N_episodes < 2``
                (Bessel's correction with N-1=0 would give NaN).

        Raises:
            ValueError: If ``rewards`` is not 2-dimensional (must be
                [N_episodes, T]).

        Example:
            >>> rewards = torch.rand(10, 100)  # 10 episodes, 100 steps each
            >>> mean_r, std_r = Metrics.mean_tracking_reward(rewards)
            >>> print(f"Reward: {mean_r:.3f} ± {std_r:.3f}")
        """
        # ----------------------------------------------------------------
        # 1. Validate input shape
        # ----------------------------------------------------------------
        if rewards.ndim != 2:
            raise ValueError(
                f"rewards must be 2-dimensional [N_episodes, T], "
                f"got shape {tuple(rewards.shape)}. "
                "Ensure the rewards tensor has episode and timestep dimensions."
            )

        n_episodes: int = rewards.shape[0]

        # ----------------------------------------------------------------
        # 2. Compute per-episode mean reward (average over time steps T)
        # ----------------------------------------------------------------
        # episode_returns[i] = (1/T) * Σ_t rewards[i, t]
        # Using mean (not sum) for scale-invariance across episode lengths.
        episode_returns: Tensor = rewards.mean(dim=1)
        # shape: [N_episodes]

        # ----------------------------------------------------------------
        # 3. Compute mean over episodes
        # ----------------------------------------------------------------
        mean_reward: float = episode_returns.mean().item()

        # ----------------------------------------------------------------
        # 4. Compute standard deviation over episodes
        # ----------------------------------------------------------------
        # Handle edge case: std() with N_episodes < 2 gives NaN due to
        # Bessel's correction (divides by N-1 = 0).
        if n_episodes < 2:
            std_reward: float = 0.0
        else:
            # torch.std uses Bessel's correction (unbiased=True) by default,
            # which is the standard convention for reporting ± values.
            std_reward = episode_returns.std().item()

            # Guard against NaN from degenerate inputs (all rewards identical)
            if not np.isfinite(std_reward):
                std_reward = 0.0

        return mean_reward, std_reward

    @staticmethod
    def model_error_over_training(
        errors: List[float],
    ) -> np.ndarray:
        """Smooth the per-iteration model error curve for visualization.

        Applies a moving average with window=10 to the raw per-iteration
        model error values, producing a smooth curve suitable for Fig. 5
        (left plot: "Model error and policy mean reward for ANYmal D and
        Unitree G1 velocity tracking task with MBPO-PPO").

        The raw error is noisy due to mini-batch variance in the world model
        training loss. Smoothing makes the trend visible without obscuring
        the overall convergence behavior.

        **Moving average implementation:** Uses ``np.convolve`` with
        ``mode='same'`` to preserve the array length for aligned plotting
        against the iteration axis. The ``mode='same'`` setting pads with
        zeros at the edges, which slightly underestimates the smoothed value
        at the beginning and end of the curve. This is acceptable for
        visualization purposes.

        Args:
            errors: List of model error values, one per training iteration.
                Typically the relative prediction error ``e`` computed on a
                held-out validation set or the training loss. Length equals
                the number of training iterations (up to 2500 per Table S10).

        Returns:
            Smoothed error curve as a numpy array of the same length as
            ``errors``. Each element is the moving average of the surrounding
            ``_SMOOTHING_WINDOW=10`` values. Returns an empty array if
            ``errors`` is empty.

        Example:
            >>> raw_errors = [0.5, 0.4, 0.45, 0.3, 0.35, 0.28, 0.25, 0.22]
            >>> smoothed = Metrics.model_error_over_training(raw_errors)
            >>> len(smoothed) == len(raw_errors)
            True
        """
        # ----------------------------------------------------------------
        # 1. Handle empty input
        # ----------------------------------------------------------------
        if not errors:
            return np.array([], dtype=np.float64)

        # ----------------------------------------------------------------
        # 2. Convert to numpy array
        # ----------------------------------------------------------------
        arr: np.ndarray = np.array(errors, dtype=np.float64)

        # ----------------------------------------------------------------
        # 3. Apply moving average with window=10
        # ----------------------------------------------------------------
        # np.convolve with mode='same' preserves the array length.
        # The kernel is a uniform average: [1/W, 1/W, ..., 1/W] of length W.
        # mode='same' pads with zeros at the edges, which slightly
        # underestimates the smoothed value at the start and end.
        # This is acceptable for visualization — the trend is preserved.
        kernel: np.ndarray = np.ones(_SMOOTHING_WINDOW, dtype=np.float64) / _SMOOTHING_WINDOW
        smoothed: np.ndarray = np.convolve(arr, kernel, mode="same")

        # ----------------------------------------------------------------
        # 4. Correct edge effects from zero-padding
        # ----------------------------------------------------------------
        # np.convolve with mode='same' divides by the full window size even
        # at the edges where fewer than W values are available. This causes
        # the smoothed values at the edges to be artificially small.
        # Correct by dividing by the actual number of contributing values.
        #
        # For a window of size W and array of length N:
        #   - At position i (0-indexed), the number of contributing values is:
        #     min(i + W//2 + 1, N) - max(0, i - W//2)
        # This correction ensures the moving average is unbiased at the edges.
        n: int = len(arr)
        half_w: int = _SMOOTHING_WINDOW // 2

        for i in range(n):
            # Number of actual values contributing to position i
            start_idx: int = max(0, i - half_w)
            end_idx: int = min(n, i + half_w + 1)
            actual_count: int = end_idx - start_idx

            if actual_count < _SMOOTHING_WINDOW and actual_count > 0:
                # Re-compute the average with the correct denominator
                smoothed[i] = arr[start_idx:end_idx].mean()

        return smoothed

    @staticmethod
    def compute_noise_robustness(
        model: object,
        test_dataset: object,
        noise_levels: List[float],
        device: str = "cuda",
        n_steps: Optional[int] = None,
        eval_batch_size: int = _EVAL_BATCH_SIZE,
    ) -> Dict[float, Tensor]:
        """Evaluate world model robustness under Gaussian noise perturbations.

        Implements the noise robustness experiment from Section 4.2 of the
        paper (Fig. 3b). Evaluates how the relative prediction error grows
        under Gaussian noise perturbations of varying magnitude applied to
        both observations and actions.

        The paper states: "we analyze its performance under Gaussian noise
        perturbations applied to both observations and actions." The noise is
        applied to the *inputs* (history and future actions), not to the
        model's internal predictions. This tests robustness to distributional
        shift in the model's inputs.

        **Long-horizon evaluation:** The ``n_steps`` parameter allows rolling
        out beyond the training forecast horizon N=8 to test long-horizon
        robustness (as shown in Fig. 3b where the x-axis extends beyond 8
        steps). When ``n_steps > N``, the future actions are extended by
        repeating the last available action.

        **Model interface (duck-typed):** The ``model`` argument must have an
        ``autoregressive_rollout(obs_history, action_history, future_actions,
        n_steps)`` method returning ``(pred_obs_means, pred_obs_logstds,
        pred_priv_means, pred_priv_logstds)``. Both ``GRUWorldModel`` and
        ``MLPModel`` satisfy this interface.

        Args:
            model: World model instance with ``autoregressive_rollout`` method.
                Must be on the target device. Typically ``GRUWorldModel`` or
                ``MLPModel`` from the benchmark comparison.
            test_dataset: ``TrajectoryDataset`` instance providing test windows.
                Must have ``get_dataloader(batch_size, shuffle)`` method and
                ``forecast_horizon`` attribute.
            noise_levels: List of Gaussian noise standard deviations to test.
                From ``config.yaml``: ``noise_robustness.noise_levels:
                [0.01, 0.05, 0.1, 0.2]``. Each value is applied as the std
                of N(0, noise_level²) noise added to all inputs.
            device: Device string for tensor operations. From ``config.device``.
                Default: "cuda".
            n_steps: Number of autoregressive forecast steps for evaluation.
                If ``None``, uses the dataset's ``forecast_horizon`` (N=8).
                Set to a larger value (e.g., 50 or 100) for long-horizon
                robustness evaluation matching Fig. 3b. Default: None.
            eval_batch_size: Batch size for the evaluation DataLoader.
                Smaller than training batch_size to avoid OOM during long-
                horizon rollouts. Default: 256.

        Returns:
            A dict mapping each noise level (float) to a relative prediction
            error curve (``Tensor[n_steps]``). The error curves are on CPU
            for compatibility with matplotlib plotting. Example:
                {
                    0.01: Tensor([0.05, 0.06, 0.07, ...]),  # shape [n_steps]
                    0.05: Tensor([0.08, 0.12, 0.18, ...]),
                    0.10: Tensor([0.15, 0.25, 0.40, ...]),
                    0.20: Tensor([0.30, 0.55, 0.90, ...]),
                }

        Raises:
            ValueError: If ``noise_levels`` is empty.
            AttributeError: If ``model`` does not have ``autoregressive_rollout``
                method or ``test_dataset`` does not have ``get_dataloader`` method.

        Example:
            >>> results = Metrics.compute_noise_robustness(
            ...     model=rwm, test_dataset=test_ds,
            ...     noise_levels=[0.01, 0.05, 0.1, 0.2],
            ...     device="cuda", n_steps=50
            ... )
            >>> for noise_level, error_curve in results.items():
            ...     print(f"Noise {noise_level}: final error = {error_curve[-1]:.3f}")
        """
        # ----------------------------------------------------------------
        # 1. Validate inputs
        # ----------------------------------------------------------------
        if not noise_levels:
            raise ValueError(
                "noise_levels must be non-empty. "
                "Check config.noise_robustness.noise_levels in config.yaml. "
                "The paper uses [0.01, 0.05, 0.1, 0.2] (Section 4.2)."
            )

        if not hasattr(model, "autoregressive_rollout"):
            raise AttributeError(
                "model must have an 'autoregressive_rollout' method. "
                "Ensure the model is a GRUWorldModel or MLPModel instance."
            )

        if not hasattr(test_dataset, "get_dataloader"):
            raise AttributeError(
                "test_dataset must have a 'get_dataloader' method. "
                "Ensure test_dataset is a TrajectoryDataset instance."
            )

        # ----------------------------------------------------------------
        # 2. Resolve evaluation device
        # ----------------------------------------------------------------
        eval_device: torch.device
        if torch.cuda.is_available() and "cuda" in device:
            eval_device = torch.device(device)
        else:
            eval_device = torch.device("cpu")

        # ----------------------------------------------------------------
        # 3. Determine number of forecast steps
        # ----------------------------------------------------------------
        # Use dataset's forecast_horizon if n_steps not specified
        dataset_n: int = int(getattr(test_dataset, "forecast_horizon", 8))
        eval_n_steps: int = n_steps if n_steps is not None else dataset_n

        # ----------------------------------------------------------------
        # 4. Build evaluation DataLoader (shuffle=False for determinism)
        # ----------------------------------------------------------------
        # Use the dataset's get_dataloader method with evaluation settings:
        # - shuffle=False: deterministic evaluation order
        # - num_workers=0: avoid multiprocessing overhead for evaluation
        # - pin_memory=False: not needed for evaluation
        dataloader: DataLoader = test_dataset.get_dataloader(  # type: ignore[union-attr]
            batch_size=eval_batch_size,
            shuffle=False,
        )

        # ----------------------------------------------------------------
        # 5. Set model to evaluation mode
        # ----------------------------------------------------------------
        # Disables dropout (if any) and batch normalization training mode.
        # The model is restored to its original mode after evaluation.
        model_nn = model  # type: ignore[assignment]
        was_training: bool = False
        if hasattr(model_nn, "training"):
            was_training = bool(model_nn.training)
            model_nn.eval()

        # ----------------------------------------------------------------
        # 6. Evaluate at each noise level
        # ----------------------------------------------------------------
        results: Dict[float, Tensor] = {}

        for noise_level in noise_levels:
            # Accumulators for predictions and targets across all batches
            all_preds: List[Tensor] = []
            all_targets: List[Tensor] = []

            with torch.no_grad():
                for batch in dataloader:
                    # ----------------------------------------------------------------
                    # 6a. Extract and move batch tensors to device
                    # ----------------------------------------------------------------
                    obs_history: Tensor = batch["obs_history"].to(eval_device)
                    # shape: [B, M, obs_dim]

                    action_history: Tensor = batch["action_history"].to(eval_device)
                    # shape: [B, M, action_dim]

                    future_actions: Tensor = batch["future_actions"].to(eval_device)
                    # shape: [B, N, action_dim]

                    target_obs: Tensor = batch["target_obs"].to(eval_device)
                    # shape: [B, N, obs_dim]

                    batch_size_b: int = obs_history.shape[0]
                    n_dataset: int = future_actions.shape[1]  # N from dataset

                    # ----------------------------------------------------------------
                    # 6b. Extend future_actions if eval_n_steps > dataset N
                    # ----------------------------------------------------------------
                    # When evaluating long-horizon robustness (n_steps > N),
                    # extend future_actions by repeating the last action column.
                    # This is a reasonable approximation — the policy would
                    # continue to output similar actions in a stable state.
                    if eval_n_steps > n_dataset:
                        # Last action in the dataset window: [B, 1, action_dim]
                        last_action: Tensor = future_actions[:, -1:, :]
                        # Number of additional steps needed
                        extra_steps: int = eval_n_steps - n_dataset
                        # Repeat the last action for the extra steps
                        extra_actions: Tensor = last_action.expand(
                            batch_size_b, extra_steps, -1
                        )
                        # Concatenate: [B, eval_n_steps, action_dim]
                        future_actions_extended: Tensor = torch.cat(
                            [future_actions, extra_actions], dim=1
                        )
                    else:
                        # Use only the first eval_n_steps actions
                        future_actions_extended = future_actions[:, :eval_n_steps, :]

                    # ----------------------------------------------------------------
                    # 6c. Apply Gaussian noise to all inputs
                    # ----------------------------------------------------------------
                    # Noise is applied to observations AND actions as described
                    # in Section 4.2: "Gaussian noise perturbations applied to
                    # both observations and actions."
                    # noise ~ N(0, noise_level²) for each element independently.

                    # Noisy observation history: [B, M, obs_dim]
                    noisy_obs_history: Tensor = (
                        obs_history
                        + torch.randn_like(obs_history) * noise_level
                    )

                    # Noisy action history: [B, M, action_dim]
                    noisy_action_history: Tensor = (
                        action_history
                        + torch.randn_like(action_history) * noise_level
                    )

                    # Noisy future actions: [B, eval_n_steps, action_dim]
                    noisy_future_actions: Tensor = (
                        future_actions_extended
                        + torch.randn_like(future_actions_extended) * noise_level
                    )

                    # ----------------------------------------------------------------
                    # 6d. Run autoregressive rollout with noisy inputs
                    # ----------------------------------------------------------------
                    # The model's autoregressive_rollout returns:
                    # (pred_obs_means, pred_obs_logstds, pred_priv_means, pred_priv_logstds)
                    # We only need pred_obs_means for the prediction error metric.
                    try:
                        rollout_output = model_nn.autoregressive_rollout(  # type: ignore[union-attr]
                            obs_history=noisy_obs_history,
                            action_history=noisy_action_history,
                            future_actions=noisy_future_actions,
                            n_steps=eval_n_steps,
                        )
                    except Exception as exc:
                        warnings.warn(
                            f"[Metrics.compute_noise_robustness] autoregressive_rollout "
                            f"failed at noise_level={noise_level}: {exc}. "
                            "Skipping this batch.",
                            UserWarning,
                            stacklevel=2,
                        )
                        continue

                    # Extract predicted observation means: [B, eval_n_steps, obs_dim]
                    # The rollout returns a tuple; first element is obs means.
                    if isinstance(rollout_output, (tuple, list)):
                        pred_obs_means: Tensor = rollout_output[0]
                    else:
                        # Fallback: assume the output is directly the obs means
                        pred_obs_means = rollout_output

                    # ----------------------------------------------------------------
                    # 6e. Align target_obs with eval_n_steps
                    # ----------------------------------------------------------------
                    # target_obs has shape [B, N, obs_dim] from the dataset.
                    # If eval_n_steps > N, we can only compare the first N steps
                    # against ground truth. For steps beyond N, we have no ground
                    # truth — skip those steps in the error computation.
                    #
                    # If eval_n_steps <= N, use only the first eval_n_steps targets.
                    compare_steps: int = min(eval_n_steps, target_obs.shape[1])

                    pred_to_compare: Tensor = pred_obs_means[:, :compare_steps, :]
                    # shape: [B, compare_steps, obs_dim]

                    target_to_compare: Tensor = target_obs[:, :compare_steps, :]
                    # shape: [B, compare_steps, obs_dim]

                    # Move to CPU before appending to avoid accumulating GPU tensors
                    all_preds.append(pred_to_compare.cpu())
                    all_targets.append(target_to_compare.cpu())

            # ----------------------------------------------------------------
            # 6f. Compute error curve for this noise level
            # ----------------------------------------------------------------
            if not all_preds:
                warnings.warn(
                    f"[Metrics.compute_noise_robustness] No valid predictions "
                    f"collected for noise_level={noise_level}. "
                    "Storing zero error curve.",
                    UserWarning,
                    stacklevel=2,
                )
                # Store a zero error curve of the expected length
                results[noise_level] = torch.zeros(
                    min(eval_n_steps, dataset_n),
                    dtype=torch.float32,