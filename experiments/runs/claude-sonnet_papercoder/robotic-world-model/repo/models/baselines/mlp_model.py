## models/baselines/mlp_model.py
"""MLP baseline world model for the Robotic World Model (RWM) project.

This module implements the MLP baseline described in Table S8 of the paper.
It serves as the primary comparison point for two experiments:

1. **Noise robustness** (Section 4.2, Fig. 3b): Trained autoregressively with
   the same history and forecast horizons as RWM. The paper demonstrates that
   RWM's GRU-based architecture significantly outperforms this MLP baseline
   under Gaussian noise perturbations, attributing the gap to the dual-
   autoregressive mechanism's ability to maintain stable hidden state
   representations.

2. **Architecture benchmark** (Section 4.3, Fig. 4): Compared against RWM-AR,
   RWM-TF, RSSM, and Transformer across diverse robotic environments. The MLP
   consistently shows higher prediction errors, especially in complex locomotion
   tasks where temporal dependencies are critical.

Architecture (Table S8):
  - Backbone: 2-layer MLP, hidden size [256, 256], ReLU activation
  - Input: flattened M-step history of [obs, action] pairs
  - Heads: same 2-layer MLP heads as RWM (256 → 128 → output_dim, ReLU)
  - Output: Gaussian distribution (mean + log-std) over next observation

Key difference from RWM: no recurrent state. The entire M-step history is
flattened into a single vector, losing the sequential inductive bias that
the GRU exploits. This is why the MLP accumulates errors faster in
autoregressive rollouts — it cannot maintain a compact, dynamically updated
representation of temporal context.

Usage:
    model = MLPModel(config)
    # Teacher-forcing (N=1):
    obs_mean, obs_logstd, priv_mean, priv_logstd = model.forward(
        obs_history, action_history
    )
    # Autoregressive rollout (N steps):
    pred_obs_means, pred_obs_logstds, pred_priv_means, pred_priv_logstds = (
        model.autoregressive_rollout(obs_history, action_history, future_actions, n_steps=8)
    )
"""

from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from utils.common import sample_gaussian


# ---------------------------------------------------------------------------
# Log-std clamping bounds for numerical stability.
# Consistent with GRUWorldModel's clamping range.
# Range [-10, 2] corresponds to std in [exp(-10), exp(2)] ≈ [4.5e-5, 7.4],
# which covers all physically meaningful prediction uncertainties while
# preventing NaN losses from extreme variance values in early training.
# ---------------------------------------------------------------------------
_LOGSTD_MIN: float = -10.0
_LOGSTD_MAX: float = 2.0


def _build_mlp_head(
    input_size: int,
    hidden_size: int,
    output_size: int,
) -> nn.Sequential:
    """Build a 2-layer MLP prediction head with ReLU activation.

    Implements the "heads, MLP, 128, ReLU" specification from Table S7,
    shared between RWM and all baseline models to isolate the backbone
    as the comparison variable.

    Architecture: Linear(input → hidden) → ReLU → Linear(hidden → output)

    Args:
        input_size: Input feature dimension. Equals the backbone output
            size (256 for both RWM and MLP baseline).
        hidden_size: Hidden layer dimension. 128 from Table S7
            (``config.rwm.mlp_head_hidden``).
        output_size: Output feature dimension. Either ``obs_dim`` or
            ``priv_dim`` depending on which head is being built.

    Returns:
        A 3-module ``nn.Sequential``: Linear → ReLU → Linear.
        No activation on the final layer — callers apply clamping for
        logstd heads or use raw output for mean heads.
    """
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
    )


class MLPModel(nn.Module):
    """MLP baseline world model with autoregressive rollout capability.

    Implements the MLP baseline from Table S8 of the paper. The model
    processes the entire M-step observation-action history by flattening
    it into a single vector and passing it through a 2-layer MLP backbone.

    Unlike ``GRUWorldModel``, this model has no recurrent state. The
    "memory" is entirely encoded in the flattened M-step context window.
    During autoregressive rollouts, the window slides forward by one step
    at each prediction, replacing the oldest real observation with the
    model's own prediction.

    This design makes the MLP a natural ablation of the GRU's sequential
    processing: both models see the same M-step context, but the GRU
    processes it step-by-step (building a compressed hidden state), while
    the MLP treats all M steps as a flat bag of features.

    The model exposes the same interface as ``GRUWorldModel`` so that
    ``Benchmark.run_prediction_benchmark`` and
    ``Benchmark.run_noise_robustness`` can call both interchangeably
    without type-checking.

    Attributes:
        obs_dim: World model observation dimension. 45 for ANYmal D
            (Table S2), 96 for Unitree G1 (Table S2).
        action_dim: Action space dimension. 12 for ANYmal D (Table S4),
            29 for Unitree G1 (Table S4).
        priv_dim: Privileged information dimension. 8 for ANYmal D
            (Table S3), 30 for Unitree G1 (Table S3).
        history_horizon: Number of historical steps M for context.
            Corresponds to ``config.rwm.history_horizon: 32``.
        input_dim: Flattened input dimension = (obs_dim + action_dim) * M.
            For ANYmal D: (45 + 12) * 32 = 1824.
            For Unitree G1: (96 + 29) * 32 = 4000.
        net: 2-layer MLP backbone. Architecture: Linear(input_dim → 256)
            → ReLU → Linear(256 → 256) → ReLU. Matches Table S8.
        obs_head_mean: MLP head predicting observation mean. Output: obs_dim.
        obs_head_logstd: MLP head predicting observation log-std. Output: obs_dim.
        priv_head_mean: MLP head predicting privileged info mean. Output: priv_dim.
        priv_head_logstd: MLP head predicting privileged info log-std. Output: priv_dim.
    """

    def __init__(self, config: object) -> None:
        """Initialize the MLP baseline from the experiment configuration.

        Extracts robot-specific dimensions from the robot sub-config and
        RWM architecture parameters from the ``rwm`` sub-config. Builds
        the 2-layer MLP backbone and four prediction heads.

        The backbone hidden size (256) and head hidden size (128) are fixed
        by Table S8 and Table S7 respectively. The input dimension is
        computed as ``(obs_dim + action_dim) * history_horizon``.

        Args:
            config: Hydra ``DictConfig`` or plain dict containing the full
                experiment configuration from ``config.yaml``. Must contain:
                - ``config.robot``: "anymal_d" or "unitree_g1"
                - ``config[robot_type].obs_dim``: observation dimension
                - ``config[robot_type].action_dim``: action dimension
                - ``config[robot_type].priv_dim``: privileged info dimension
                - ``config.rwm.history_horizon``: M = 32 (Table S10)
                - ``config.rwm.mlp_head_hidden``: 128 (Table S7)
                - ``config.device``: device string ("cuda" or "cpu")

        Raises:
            ValueError: If ``config.robot`` is not "anymal_d" or "unitree_g1".
            KeyError: If required config fields are missing.
        """
        super().__init__()

        # ----------------------------------------------------------------
        # 1. Resolve robot type and extract robot-specific dimensions
        # ----------------------------------------------------------------
        robot_type: str = str(config.robot)  # type: ignore[union-attr]
        _supported_robots: Tuple[str, ...] = ("anymal_d", "unitree_g1")
        if robot_type not in _supported_robots:
            raise ValueError(
                f"Unsupported robot type '{robot_type}' in config.robot. "
                f"Expected one of: {_supported_robots}. "
                "Check the 'robot' field in config.yaml."
            )

        # Access robot-specific sub-config (e.g., config.anymal_d)
        robot_cfg = config[robot_type]  # type: ignore[index]

        # Dimensions from Tables S2-S4
        self.obs_dim: int = int(robot_cfg.obs_dim)
        self.action_dim: int = int(robot_cfg.action_dim)
        self.priv_dim: int = int(robot_cfg.priv_dim)

        # ----------------------------------------------------------------
        # 2. Extract architecture parameters from config
        # ----------------------------------------------------------------
        rwm_cfg = config.rwm  # type: ignore[union-attr]

        # History horizon M = 32 (Table S10, config.rwm.history_horizon)
        self.history_horizon: int = int(rwm_cfg.history_horizon)

        # MLP head hidden size = 128 (Table S7, config.rwm.mlp_head_hidden)
        self.mlp_head_hidden: int = int(rwm_cfg.mlp_head_hidden)

        # Backbone hidden size = 256 (Table S8: "hidden shape 256, 256")
        # This is fixed by the paper's Table S8 specification.
        self._backbone_hidden: int = 256

        # Device for model placement
        self.device: str = str(config.device)  # type: ignore[union-attr]

        # ----------------------------------------------------------------
        # 3. Compute flattened input dimension
        # ----------------------------------------------------------------
        # Each history step contributes (obs_dim + action_dim) features.
        # All M steps are concatenated into a single flat vector.
        # ANYmal D: (45 + 12) * 32 = 1824
        # Unitree G1: (96 + 29) * 32 = 4000
        self.input_dim: int = (self.obs_dim + self.action_dim) * self.history_horizon

        # ----------------------------------------------------------------
        # 4. Build 2-layer MLP backbone (Table S8)
        # ----------------------------------------------------------------
        # Architecture: Linear(input_dim → 256) → ReLU → Linear(256 → 256) → ReLU
        # Matches Table S8: "hidden shape 256, 256, activation ReLU"
        self.net: nn.Sequential = nn.Sequential(
            nn.Linear(self.input_dim, self._backbone_hidden),
            nn.ReLU(),
            nn.Linear(self._backbone_hidden, self._backbone_hidden),
            nn.ReLU(),
        )

        # ----------------------------------------------------------------
        # 5. Build four MLP prediction heads (same as RWM, Table S7)
        # ----------------------------------------------------------------
        # Using the same head architecture as GRUWorldModel to isolate
        # the backbone as the comparison variable between MLP and RWM.

        # Observation prediction heads: output obs_dim
        self.obs_head_mean: nn.Sequential = _build_mlp_head(
            input_size=self._backbone_hidden,
            hidden_size=self.mlp_head_hidden,
            output_size=self.obs_dim,
        )
        self.obs_head_logstd: nn.Sequential = _build_mlp_head(
            input_size=self._backbone_hidden,
            hidden_size=self.mlp_head_hidden,
            output_size=self.obs_dim,
        )

        # Privileged information prediction heads: output priv_dim
        # For ANYmal D: priv_dim=8 (binary contacts → BCE loss in trainer)
        # For Unitree G1: priv_dim=30 (mixed binary + continuous)
        self.priv_head_mean: nn.Sequential = _build_mlp_head(
            input_size=self._backbone_hidden,
            hidden_size=self.mlp_head_hidden,
            output_size=self.priv_dim,
        )
        self.priv_head_logstd: nn.Sequential = _build_mlp_head(
            input_size=self._backbone_hidden,
            hidden_size=self.mlp_head_hidden,
            output_size=self.priv_dim,
        )

    def forward(
        self,
        obs_history: Tensor,
        action_history: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Predict the next observation distribution (teacher-forcing mode).

        Implements the N=1 special case: processes the M-step history by
        flattening it into a single vector, passing through the MLP backbone,
        and applying the four prediction heads to produce the Gaussian
        distribution parameters for the next observation and privileged info.

        This method is used for:
          - Teacher-forcing training (N=1 case of the autoregressive objective)
          - Single-step evaluation during benchmarking
          - Building block for ``autoregressive_rollout``

        For autoregressive training with N > 1, use ``autoregressive_rollout``
        which calls this method in a loop with a sliding context window.

        Args:
            obs_history: Historical observations of shape
                ``[B, M, obs_dim]``. Contains the M most recent world model
                observations (Table S2). For ANYmal D: ``[B, 32, 45]``.
                For Unitree G1: ``[B, 32, 96]``.
            action_history: Historical actions of shape
                ``[B, M, action_dim]``. Contains the M most recent joint
                position targets (Table S4). For ANYmal D: ``[B, 32, 12]``.
                For Unitree G1: ``[B, 32, 29]``.

        Returns:
            A tuple ``(obs_mean, obs_logstd, priv_mean, priv_logstd)``
            where each tensor has shape ``[B, D]``:
              - ``obs_mean``: Predicted observation mean, shape
                ``[B, obs_dim]``.
              - ``obs_logstd``: Predicted observation log-std, shape
                ``[B, obs_dim]``. Clamped to ``[_LOGSTD_MIN, _LOGSTD_MAX]``
                for numerical stability.
              - ``priv_mean``: Predicted privileged info mean, shape
                ``[B, priv_dim]``.
              - ``priv_logstd``: Predicted privileged info log-std, shape
                ``[B, priv_dim]``. Clamped to ``[_LOGSTD_MIN, _LOGSTD_MAX]``.
        """
        # ----------------------------------------------------------------
        # 1. Concatenate obs and action along the feature dimension.
        # ----------------------------------------------------------------
        # obs_history:    [B, M, obs_dim]
        # action_history: [B, M, action_dim]
        # x:              [B, M, obs_dim + action_dim]
        x: Tensor = torch.cat([obs_history, action_history], dim=-1)

        # ----------------------------------------------------------------
        # 2. Flatten the time and feature dimensions into a single vector.
        # ----------------------------------------------------------------
        # [B, M, obs_dim + action_dim] → [B, M * (obs_dim + action_dim)]
        # Use reshape (not view) to handle non-contiguous tensors safely.
        # Non-contiguous tensors arise from sliding window concatenation
        # in autoregressive_rollout (torch.cat creates contiguous tensors,
        # but slicing [:, 1:, :] may not be contiguous in all PyTorch versions).
        batch_size: int = x.shape[0]
        x_flat: Tensor = x.reshape(batch_size, -1)
        # shape: [B, M * (obs_dim + action_dim)] = [B, input_dim]

        # ----------------------------------------------------------------
        # 3. Pass through the 2-layer MLP backbone.
        # ----------------------------------------------------------------
        # net: Linear(input_dim → 256) → ReLU → Linear(256 → 256) → ReLU
        features: Tensor = self.net(x_flat)
        # shape: [B, 256]

        # ----------------------------------------------------------------
        # 4. Apply four MLP prediction heads.
        # ----------------------------------------------------------------
        obs_mean: Tensor = self.obs_head_mean(features)    # [B, obs_dim]

        # Clamp logstd to prevent numerical overflow in exp(logstd).
        # Without clamping, early training can produce extreme values
        # (e.g., logstd = ±100) that cause NaN losses.
        obs_logstd: Tensor = torch.clamp(
            self.obs_head_logstd(features),
            _LOGSTD_MIN,
            _LOGSTD_MAX,
        )  # [B, obs_dim]

        priv_mean: Tensor = self.priv_head_mean(features)   # [B, priv_dim]

        priv_logstd: Tensor = torch.clamp(
            self.priv_head_logstd(features),
            _LOGSTD_MIN,
            _LOGSTD_MAX,
        )  # [B, priv_dim]

        return obs_mean, obs_logstd, priv_mean, priv_logstd

    def autoregressive_rollout(
        self,
        obs_history: Tensor,
        action_history: Tensor,
        future_actions: Tensor,
        n_steps: int = 8,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Perform N-step autoregressive prediction with sliding context window.

        Implements the outer autoregression loop for the MLP baseline.
        Since the MLP has no recurrent state, the M-step context window
        slides forward at each prediction step: the oldest real observation
        is dropped and the model's own prediction is appended.

        This is the MLP analog of ``GRUWorldModel.autoregressive_rollout``,
        exposing the same interface so ``Benchmark`` can call both models
        interchangeably. The key difference is that the GRU maintains a
        compressed hidden state updated at each step, while the MLP must
        carry the full M-step window explicitly.

        The reparameterization trick (``sample_gaussian``) is applied at
        each step to enable gradient flow through the stochastic sampling,
        consistent with the autoregressive training objective (Eq. 2 in
        the paper). This is what allows the multi-step prediction loss to
        propagate gradients back through all N steps.

        **Gradient flow path for step k:**
        ```
        loss_k → obs_mean[k] → features[k] → net(x_flat[k])
               → x_flat[k] → next_obs[k-1] (via reparameterization)
               → obs_mean[k-1] → ... (chain back to step 0)
        ```

        **Why the MLP accumulates errors faster than RWM:**
        At each step, the MLP sees a window where the oldest M-k real
        observations have been replaced by predictions. The GRU, by
        contrast, maintains a hidden state that was initialized from real
        observations and is updated incrementally — it can "remember"
        patterns from the real history even as the window fills with
        predictions. This is the core advantage of the dual-autoregressive
        mechanism demonstrated in Fig. 3b of the paper.

        Args:
            obs_history: Historical observations of shape
                ``[B, M, obs_dim]``. M=32 during training (Table S10).
                For ANYmal D: ``[B, 32, 45]``.
            action_history: Historical actions of shape
                ``[B, M, action_dim]``. M=32 during training.
                For ANYmal D: ``[B, 32, 12]``.
            future_actions: Actions during the forecast horizon, shape
                ``[B, N, action_dim]``. N=8 during training (Table S10).
                ``future_actions[:, k, :]`` is the action at forecast step k,
                appended to the sliding window when predicting step k+1.
            n_steps: Number of forecast steps. Must be <= ``future_actions.shape[1]``
                during training (N=8). Can be larger during evaluation for
                long-horizon rollout assessment (e.g., 100+ steps for Fig. 3a).
                Default: 8.

        Returns:
            A tuple of four tensors, all of shape ``[B, n_steps, D]``:
              - ``pred_obs_means``: Predicted observation means,
                shape ``[B, n_steps, obs_dim]``.
              - ``pred_obs_logstds``: Predicted observation log-stds,
                shape ``[B, n_steps, obs_dim]``. Clamped to
                ``[_LOGSTD_MIN, _LOGSTD_MAX]``.
              - ``pred_priv_means``: Predicted privileged info means,
                shape ``[B, n_steps, priv_dim]``.
              - ``pred_priv_logstds``: Predicted privileged info log-stds,
                shape ``[B, n_steps, priv_dim]``.

        Raises:
            ValueError: If ``n_steps`` > ``future_actions.shape[1]``, which
                would require actions beyond what was provided.
        """
        if n_steps > future_actions.shape[1]:
            raise ValueError(
                f"n_steps={n_steps} exceeds future_actions sequence length "
                f"{future_actions.shape[1]}. Provide at least n_steps future "
                "actions. During training, n_steps=N=8 and future_actions has "
                "shape [B, N, action_dim] (config.rwm.forecast_horizon=8)."
            )

        # ----------------------------------------------------------------
        # 1. Initialize sliding context window buffers.
        # ----------------------------------------------------------------
        # Clone to avoid modifying the input tensors in-place.
        # These buffers will be updated at each autoregressive step by
        # dropping the oldest step and appending the new prediction.
        current_obs_window: Tensor = obs_history.clone()
        # shape: [B, M, obs_dim]

        current_action_window: Tensor = action_history.clone()
        # shape: [B, M, action_dim]

        # ----------------------------------------------------------------
        # 2. Initialize output collectors.
        # ----------------------------------------------------------------
        # Collect predictions as lists and stack at the end.
        # This is more memory-efficient than pre-allocating [B, N, D] tensors
        # and filling in-place (avoids in-place ops that can break autograd).
        pred_obs_means_list: List[Tensor] = []
        pred_obs_logstds_list: List[Tensor] = []
        pred_priv_means_list: List[Tensor] = []
        pred_priv_logstds_list: List[Tensor] = []

        # ----------------------------------------------------------------
        # 3. Autoregressive loop: N forecast steps.
        # ----------------------------------------------------------------
        for k in range(n_steps):
            # ----------------------------------------------------------------
            # Step 3a: Predict distribution from current context window.
            # ----------------------------------------------------------------
            # forward() flattens the window and passes through the MLP backbone.
            obs_mean_k: Tensor
            obs_logstd_k: Tensor
            priv_mean_k: Tensor
            priv_logstd_k: Tensor
            obs_mean_k, obs_logstd_k, priv_mean_k, priv_logstd_k = self.forward(
                current_obs_window,
                current_action_window,
            )
            # obs_mean_k:   [B, obs_dim]
            # obs_logstd_k: [B, obs_dim]
            # priv_mean_k:  [B, priv_dim]
            # priv_logstd_k:[B, priv_dim]

            # ----------------------------------------------------------------
            # Step 3b: Collect mean predictions for loss computation.
            # ----------------------------------------------------------------
            # We store MEANS (not samples) for the loss computation in
            # RWMTrainer._compute_loss, which uses (mean, logstd, target)
            # to compute the Gaussian NLL: -log p(target | mean, exp(logstd)).
            # The sample is only used to feed back into the next step.
            pred_obs_means_list.append(obs_mean_k)
            pred_obs_logstds_list.append(obs_logstd_k)
            pred_priv_means_list.append(priv_mean_k)
            pred_priv_logstds_list.append(priv_logstd_k)

            # ----------------------------------------------------------------
            # Step 3c: Sample next observation via reparameterization trick.
            # ----------------------------------------------------------------
            # sample_gaussian(mean, logstd) = mean + randn_like(mean) * exp(logstd)
            # The randomness is in eps = randn_like(mean), which is NOT part of
            # the computation graph. Gradients flow through mean and logstd,
            # enabling end-to-end optimization of the multi-step prediction loss.
            #
            # This is the same reparameterization used in GRUWorldModel and
            # MBPOPPOTrainer — using the shared utility from utils/common.py
            # for consistency across the codebase.
            next_obs: Tensor = sample_gaussian(obs_mean_k, obs_logstd_k)
            # shape: [B, obs_dim] — differentiable w.r.t. obs_mean_k, obs_logstd_k

            # ----------------------------------------------------------------
            # Step 3d: Slide the context window forward by one step.
            # ----------------------------------------------------------------
            # Drop the oldest step (index 0) and append the new prediction.
            # This maintains the invariant that current_obs_window always has
            # shape [B, M, obs_dim] with the M most recent observations.
            #
            # After sliding:
            #   current_obs_window[:, 0:M-1, :] = old current_obs_window[:, 1:M, :]
            #   current_obs_window[:, M-1:M, :] = next_obs (predicted)
            #
            # torch.cat creates a new contiguous tensor, so reshape in forward()
            # will always work correctly on the updated window.
            current_obs_window = torch.cat(
                [
                    current_obs_window[:, 1:, :],   # [B, M-1, obs_dim]
                    next_obs.unsqueeze(1),           # [B, 1,   obs_dim]
                ],
                dim=1,
            )
            # shape: [B, M, obs_dim] — window slid forward by one step

            # Similarly slide the action window, appending the future action
            # at forecast step k. This action was taken at the same timestep
            # as the predicted observation next_obs.
            # future_actions[:, k:k+1, :]: shape [B, 1, action_dim]
            current_action_window = torch.cat(
                [
                    current_action_window[:, 1:, :],    # [B, M-1, action_dim]
                    future_actions[:, k:k+1, :],        # [B, 1,   action_dim]
                ],
                dim=1,
            )
            # shape: [B, M, action_dim] — window slid forward by one step

        # ----------------------------------------------------------------
        # 4. Stack predictions along the time dimension.
        # ----------------------------------------------------------------
        # torch.stack(list_of_[B, D], dim=1) → [B, N, D]
        pred_obs_means: Tensor = torch.stack(pred_obs_means_list, dim=1)
        # shape: [B, n_steps, obs_dim]

        pred_obs_logstds: Tensor = torch.stack(pred_obs_logstds_list, dim=1)
        # shape: [B, n_steps, obs_dim]

        pred_priv_means: Tensor = torch.stack(pred_priv_means_list, dim=1)
        # shape: [B, n_steps, priv_dim]

        pred_priv_logstds: Tensor = torch.stack(pred_priv_logstds_list, dim=1)
        # shape: [B, n_steps, priv_dim]

        return pred_obs_means, pred_obs_logstds, pred_priv_means, pred_priv_logstds
