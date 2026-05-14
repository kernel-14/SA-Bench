## Code: models/rwm.py

```python
## models/rwm.py
"""Robotic World Model (RWM) — GRU-based neural network world model.

This module implements the core RWM architecture described in Section 3.2
and Table S7 of the paper. The model learns to predict future observations
autoregressively using a dual-autoregressive mechanism:

  (i)  Inner autoregression: GRU hidden states are updated sequentially
       over the M-step history window (context processing).
  (ii) Outer autoregression: predicted observations are fed back as inputs
       over the N-step forecast horizon (training-time exposure to own errors).

The network outputs a Gaussian distribution (mean + log-std) over the next
observation and privileged information, enabling:
  - Reparameterization-based gradient flow through stochastic sampling
  - Uncertainty-aware predictions for downstream planning
  - Separate loss terms for observations (L_o) and privileged info (L_c)

Architecture (Table S7):
  - Base: 2-layer GRU, hidden size 256 (input: obs_dim + action_dim)
  - Heads: 2-layer MLP (256 → 128 → output_dim), ReLU activation
  - Four heads: obs_mean, obs_logstd, priv_mean, priv_logstd

Training parameters (Table S10):
  - History horizon M = 32
  - Forecast horizon N = 8
  - Forecast decay α = 1.0 (no decay)

Usage:
    model = GRUWorldModel(config)
    # Training (autoregressive):
    pred_obs, pred_obs_logstd, pred_priv, pred_priv_logstd = (
        model.autoregressive_rollout(obs_hist, act_hist, future_acts, n_steps=8)
    )
    # Inference (single step):
    obs_mean, obs_logstd, priv_mean, priv_logstd, new_hidden = (
        model.step(obs, action, hidden)
    )
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from utils.common import sample_gaussian


# ---------------------------------------------------------------------------
# Log-std clamping bounds for numerical stability.
# Without clamping, early training can produce extreme variance values
# (e.g., logstd = ±100) that cause NaN losses via exp(logstd).
# Range [-5, 2] corresponds to std in [exp(-5), exp(2)] ≈ [0.007, 7.4],
# which covers all physically meaningful prediction uncertainties.
# ---------------------------------------------------------------------------
_LOGSTD_MIN: float = -5.0
_LOGSTD_MAX: float = 2.0


def _build_mlp_head(
    input_size: int,
    hidden_size: int,
    output_size: int,
) -> nn.Sequential:
    """Build a 2-layer MLP head with ReLU activation.

    Implements the "heads, MLP, 128, ReLU" specification from Table S7.
    The architecture is: Linear(input → hidden) → ReLU → Linear(hidden → output).

    Args:
        input_size: Input feature dimension. Equals ``gru_hidden_size`` (256)
            for all heads in RWM.
        hidden_size: Hidden layer dimension. Equals ``mlp_head_hidden`` (128)
            from ``config.rwm.mlp_head_hidden``.
        output_size: Output feature dimension. Either ``obs_dim`` or
            ``priv_dim`` depending on which head is being built.

    Returns:
        A 3-layer ``nn.Sequential``: Linear → ReLU → Linear.
        The final Linear has no activation — the caller applies clamping
        (for logstd heads) or uses the raw output (for mean heads).
    """
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
    )


class GRUWorldModel(nn.Module):
    """GRU-based world model with dual-autoregressive prediction mechanism.

    Implements the Robotic World Model (RWM) architecture from Section 3.2
    and Table S7 of the paper. The model predicts the distribution of the
    next observation and privileged information given a history of
    observation-action pairs.

    The dual-autoregressive mechanism consists of:
      - **Inner autoregression** (``encode_history``): GRU processes M
        historical steps sequentially, building a rich hidden state that
        captures unobservable dynamics and partial observability.
      - **Outer autoregression** (``autoregressive_rollout``): For each of
        N forecast steps, the model predicts the next observation, samples
        from the predicted distribution via reparameterization, and feeds
        the sample back as input for the next step. This exposes the model
        to its own prediction errors during training, mitigating the
        train-test distribution mismatch that causes error accumulation.

    The reparameterization trick (``sample_observation``) enables gradient
    flow through the stochastic sampling in the outer autoregression loop,
    allowing end-to-end optimization of the multi-step prediction loss
    (Eq. 2 in the paper).

    Attributes:
        obs_dim: World model observation dimension. 45 for ANYmal D
            (Table S2), 96 for Unitree G1 (Table S2).
        action_dim: Action space dimension. 12 for ANYmal D (Table S4),
            29 for Unitree G1 (Table S4).
        priv_dim: Privileged information dimension. 8 for ANYmal D
            (Table S3), 30 for Unitree G1 (Table S3).
        hidden_size: GRU hidden state size per layer. 256 (Table S7).
        num_layers: Number of GRU layers. 2 (Table S7: "hidden shape 256, 256").
        mlp_head_hidden: Hidden size of MLP prediction heads. 128 (Table S7).
        gru: 2-layer GRU with input size ``obs_dim + action_dim`` and
            hidden size 256. ``batch_first=True`` for [B, T, D] convention.
        obs_head_mean: MLP head predicting observation mean. Output: obs_dim.
        obs_head_logstd: MLP head predicting observation log-std. Output: obs_dim.
        priv_head_mean: MLP head predicting privileged info mean. Output: priv_dim.
        priv_head_logstd: MLP head predicting privileged info log-std. Output: priv_dim.
    """

    def __init__(self, config: object) -> None:
        """Initialize the GRU world model from the experiment configuration.

        Extracts robot-specific dimensions from the robot sub-config and
        RWM architecture parameters from the ``rwm`` sub-config. Builds
        the GRU base and four MLP prediction heads.

        The config object must provide access to:
          - ``config[robot_type].obs_dim``: observation dimension
          - ``config[robot_type].action_dim``: action dimension
          - ``config[robot_type].priv_dim``: privileged info dimension
          - ``config.rwm.gru_hidden_size``: 256 (Table S7)
          - ``config.rwm.gru_num_layers``: 2 (Table S7)
          - ``config.rwm.mlp_head_hidden``: 128 (Table S7)
          - ``config.robot``: "anymal_d" or "unitree_g1"

        Args:
            config: Hydra ``DictConfig`` or plain dict containing the full
                experiment configuration from ``config.yaml``. Must contain
                the ``rwm`` sub-config and the robot-specific sub-config
                (``anymal_d`` or ``unitree_g1``).

        Raises:
            ValueError: If ``config.robot`` is not "anymal_d" or "unitree_g1".
            KeyError: If required config fields are missing.
        """
        super().__init__()

        # ----------------------------------------------------------------
        # 1. Resolve robot type and extract robot-specific dimensions
        # ----------------------------------------------------------------
        robot_type: str = str(config.robot)  # type: ignore[union-attr]
        _supported_robots = ("anymal_d", "unitree_g1")
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
        # 2. Extract RWM architecture parameters from config.rwm (Table S7)
        # ----------------------------------------------------------------
        rwm_cfg = config.rwm  # type: ignore[union-attr]

        # GRU hidden size: 256 per layer (Table S7: "hidden shape 256, 256")
        self.hidden_size: int = int(rwm_cfg.gru_hidden_size)

        # Number of GRU layers: 2 (Table S7: two 256-dim layers)
        self.num_layers: int = int(rwm_cfg.gru_num_layers)

        # MLP head hidden size: 128 (Table S7: "heads, MLP, 128, ReLU")
        self.mlp_head_hidden: int = int(rwm_cfg.mlp_head_hidden)

        # ----------------------------------------------------------------
        # 3. Build GRU base (Table S7: "base, GRU, 256 256")
        # ----------------------------------------------------------------
        # Input at each step: concatenated [obs, action] vector.
        # batch_first=True: input/output shape is [B, T, D] (not [T, B, D]).
        # Note: GRU hidden state shape is ALWAYS [num_layers, B, hidden_size]
        # regardless of batch_first — this is PyTorch's convention.
        self.gru: nn.GRU = nn.GRU(
            input_size=self.obs_dim + self.action_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        # ----------------------------------------------------------------
        # 4. Build four MLP prediction heads (Table S7: "heads, MLP, 128, ReLU")
        # ----------------------------------------------------------------
        # Observation prediction heads: output obs_dim
        self.obs_head_mean: nn.Sequential = _build_mlp_head(
            input_size=self.hidden_size,
            hidden_size=self.mlp_head_hidden,
            output_size=self.obs_dim,
        )
        self.obs_head_logstd: nn.Sequential = _build_mlp_head(
            input_size=self.hidden_size,
            hidden_size=self.mlp_head_hidden,
            output_size=self.obs_dim,
        )

        # Privileged information prediction heads: output priv_dim
        # Used for the additional learning objective L_c (Eq. 2 in paper).
        # For ANYmal D: priv_dim=8 (binary contacts → BCE loss in trainer)
        # For Unitree G1: priv_dim=30 (mixed binary + continuous)
        self.priv_head_mean: nn.Sequential = _build_mlp_head(
            input_size=self.hidden_size,
            hidden_size=self.mlp_head_hidden,
            output_size=self.priv_dim,
        )
        self.priv_head_logstd: nn.Sequential = _build_mlp_head(
            input_size=self.hidden_size,
            hidden_size=self.mlp_head_hidden,
            output_size=self.priv_dim,
        )

    # ----------------------------------------------------------------
    # Core methods
    # ----------------------------------------------------------------

    def encode_history(
        self,
        obs_history: Tensor,
        action_history: Tensor,
    ) -> Tensor:
        """Process M historical steps to build the GRU hidden state.

        Implements the **inner autoregression** of the dual-autoregressive
        mechanism (Section 3.2). The GRU processes all M historical
        observation-action pairs sequentially, building a rich hidden state
        that encodes unobservable dynamics and partial observability.

        The sequential nature of the GRU means each step's hidden state
        depends on all previous steps — this IS the inner autoregression,
        even though PyTorch executes it in a single vectorized forward pass.

        The final hidden state encodes the entire M-step history and serves
        as the initialization for the outer autoregression in
        ``autoregressive_rollout`` and for the imagination loop in
        ``MBPOPPOTrainer.imagine_trajectories``.

        Args:
            obs_history: Historical observations of shape
                ``[B, M, obs_dim]``. Contains the M most recent world model
                observations (Table S2). For ANYmal D: ``[B, 32, 45]``.
            action_history: Historical actions of shape
                ``[B, M, action_dim]``. Contains the M most recent joint
                position targets (Table S4). For ANYmal D: ``[B, 32, 12]``.

        Returns:
            GRU hidden state tensor of shape ``[num_layers, B, hidden_size]``
            = ``[2, B, 256]``. This is the hidden state after processing all
            M historical steps. Note: shape does NOT follow batch_first
            convention — this is PyTorch's standard for GRU hidden states.

        Raises:
            RuntimeError: If ``obs_history`` and ``action_history`` have
                inconsistent batch sizes or sequence lengths.
        """
        # Concatenate obs and action along the feature dimension.
        # obs_history:    [B, M, obs_dim]
        # action_history: [B, M, action_dim]
        # x:              [B, M, obs_dim + action_dim]
        x: Tensor = torch.cat([obs_history, action_history], dim=-1)

        # Process all M steps through the GRU in a single forward pass.
        # The GRU hidden state is initialized to zeros (PyTorch default)
        # when h_0 is not provided — consistent with the zero-initialization
        # design choice noted in the project design document.
        #
        # gru_output: [B, M, hidden_size] — all intermediate hidden states
        # hidden:     [num_layers, B, hidden_size] — final hidden state
        # We only need the final hidden state for downstream use.
        _, hidden = self.gru(x)

        return hidden  # shape: [num_layers, B, hidden_size] = [2, B, 256]

    def forward(
        self,
        obs_history: Tensor,
        action_history: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Predict the next observation distribution (teacher-forcing mode).

        Implements the N=1 special case of autoregressive training, also
        known as teacher-forcing (Section 3.2, Fig. 2b). Processes the M
        historical steps and predicts the distribution of the NEXT step.

        This method is used for:
          - Evaluation in teacher-forcing mode (RWM-TF baseline)
          - Quick single-step prediction during benchmarking
          - Building block for ``autoregressive_rollout``

        For training RWM-AR (the proposed method), use
        ``autoregressive_rollout`` instead, which implements the full
        N-step outer autoregression.

        Args:
            obs_history: Historical observations of shape
                ``[B, M, obs_dim]``.
            action_history: Historical actions of shape
                ``[B, M, action_dim]``.

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
                ``[B, priv_dim]``.
        """
        # Inner autoregression: process M historical steps.
        # hidden: [num_layers, B, hidden_size]
        hidden: Tensor = self.encode_history(obs_history, action_history)

        # Extract the last GRU layer's hidden state for the MLP heads.
        # hidden[-1]: [B, hidden_size] = [B, 256]
        # The last layer produces the most abstract representation.
        h_last: Tensor = hidden[-1]  # shape: [B, hidden_size]

        # Apply four MLP heads to predict the Gaussian distribution parameters.
        obs_mean: Tensor = self.obs_head_mean(h_last)    # [B, obs_dim]
        obs_logstd: Tensor = torch.clamp(
            self.obs_head_logstd(h_last),
            _LOGSTD_MIN,
            _LOGSTD_MAX,
        )  # [B, obs_dim]

        priv_mean: Tensor = self.priv_head_mean(h_last)   # [B, priv_dim]
        priv_logstd: Tensor = self.priv_head_logstd(h_last)  # [B, priv_dim]

        return obs_mean, obs_logstd, priv_mean, priv_logstd

    def autoregressive_rollout(
        self,
        obs_history: Tensor,
        action_history: Tensor,
        future_actions: Tensor,
        n_steps: int = 8,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Perform N-step autoregressive prediction (outer autoregression).

        Implements the **outer autoregression** of the dual-autoregressive
        mechanism (Section 3.2, Fig. 2a). For each of N forecast steps:
          1. Predict the next observation distribution from the current
             GRU hidden state.
          2. Sample the next observation via the reparameterization trick
             (enabling gradient flow through the stochastic sampling).
          3. Feed the sampled observation back as input to the GRU for the
             next step.

        This training scheme exposes the model to its own prediction errors,
        reducing the mismatch between training and inference distributions
        that causes error accumulation in teacher-forcing models (Section 3.2).

        The reparameterization trick (``eps = randn_like(mean); sample = mean
        + eps * exp(logstd)``) ensures gradients flow from step k+1's
        prediction error back through step k's sampling operation, enabling
        end-to-end optimization of the multi-step prediction loss (Eq. 2).

        **Returns means AND log-stds** (not just means as stated in the
        design spec) because ``RWMTrainer._compute_loss`` needs both to
        compute the Gaussian NLL loss. The design spec description is
        incomplete on this point.

        Args:
            obs_history: Historical observations of shape
                ``[B, M, obs_dim]``. M=32 during training (Table S10).
            action_history: Historical actions of shape
                ``[B, M, action_dim]``. M=32 during training.
            future_actions: Actions during the forecast horizon, shape
                ``[B, N, action_dim]``. N=8 during training (Table S10).
                ``future_actions[:, k, :]`` is the action at forecast step k,
                used as GRU input when predicting the observation at step k+1.
            n_steps: Number of forecast steps. Must equal
                ``future_actions.shape[1]`` during training (N=8). Can be
                larger during evaluation for long-horizon rollout assessment
                (e.g., 100+ steps for Fig. 3a). Default: 8.

        Returns:
            A tuple of four tensors, all of shape ``[B, n_steps, D]``:
              - ``pred_obs_means``: Predicted observation means,
                shape ``[B, N, obs_dim]``.
              - ``pred_obs_logstds``: Predicted observation log-stds,
                shape ``[B, N, obs_dim]``. Clamped to
                ``[_LOGSTD_MIN, _LOGSTD_MAX]``.
              - ``pred_priv_means``: Predicted privileged info means,
                shape ``[B, N, priv_dim]``.
              - ``pred_priv_logstds``: Predicted privileged info log-stds,
                shape ``[B, N, priv_dim]``.

        Raises:
            ValueError: If ``n_steps`` > ``future_actions.shape[1]``, which
                would require actions beyond what was provided.
        """
        if n_steps > future_actions.shape[1]:
            raise ValueError(
                f"n_steps={n_steps} exceeds future_actions sequence length "
                f"{future_actions.shape[1]}. Provide at least n_steps future "
                "actions. During training, n_steps=N=8 and future_actions has "
                "shape [B, N, action_dim]."
            )

        # ----------------------------------------------------------------
        # Phase 1: Inner autoregression — process M historical steps.
        # ----------------------------------------------------------------
        # hidden: [num_layers, B, hidden_size] = [2, B, 256]
        hidden: Tensor = self.encode_history(obs_history, action_history)

        # Extract last layer's hidden state for the MLP heads.
        # h_last: [B, hidden_size]
        h_last: Tensor = hidden[-1]

        # ----------------------------------------------------------------
        # Phase 2: Outer autoregression — N forecast steps.
        # ----------------------------------------------------------------
        # Storage lists for stacking at the end.
        # Using lists + torch.stack is more memory-efficient than pre-allocating
        # a [B, N, D] tensor and filling it in-place (avoids in-place ops
        # that can break autograd).
        pred_obs_means_list = []
        pred_obs_logstds_list = []
        pred_priv_means_list = []
        pred_priv_logstds_list = []

        for k in range(n_steps):
            # ----------------------------------------------------------------
            # Step 2a: Predict distribution from current hidden state.
            # ----------------------------------------------------------------
            obs_mean_k: Tensor = self.obs_head_mean(h_last)    # [B, obs_dim]
            obs_logstd_k: Tensor = torch.clamp(
                self.obs_head_logstd(h_last),
                _LOGSTD_MIN,
                _LOGSTD_MAX,
            )  # [B, obs_dim]

            priv_mean_k: Tensor = self.priv_head_mean(h_last)   # [B, priv_dim]
            priv_logstd_k: Tensor = self.priv_head_logstd(h_last)  # [B, priv_dim]

            # ----------------------------------------------------------------
            # Step 2b: Store mean and logstd predictions for loss computation.
            # ----------------------------------------------------------------
            # We store MEANS (not samples) for the loss — the NLL loss in
            # RWMTrainer._compute_loss uses (mean, logstd, target) to compute
            # -log p(target | mean, exp(logstd)).
            pred_obs_means_list.append(obs_mean_k)
            pred_obs_logstds_list.append(obs_logstd_k)
            pred_priv_means_list.append(priv_mean_k)
            pred_priv_logstds_list.append(priv_logstd_k)

            # ----------------------------------------------------------------
            # Step 2c: Sample next observation via reparameterization trick.
            # ----------------------------------------------------------------
            # This is the key step for gradient flow through outer autoregression.
            # sample_gaussian(mean, logstd) = mean + randn_like(mean) * exp(logstd)
            # The randomness is in eps = randn_like(mean), which is NOT part of
            # the computation graph. Gradients flow through mean and logstd.
            #
            # Using the shared utility from utils/common.py for consistency
            # with MBPOPPOTrainer.imagine_trajectories.
            next_obs: Tensor = sample_gaussian(obs_mean_k, obs_logstd_k)
            # next_obs: [B, obs_dim] — differentiable w.r.t. obs_mean_k, obs_logstd_k

            # ----------------------------------------------------------------
            # Step 2d: Prepare GRU input for the next step.
            # ----------------------------------------------------------------
            # GRU input = [predicted_obs, future_action_k]
            # future_actions[:, k, :]: action at forecast step k, shape [B, action_dim]
            action_k: Tensor = future_actions[:, k, :]  # [B, action_dim]

            # Concatenate predicted obs and action: [B, obs_dim + action_dim]
            gru_input: Tensor = torch.cat([next_obs, action_k], dim=-1)

            # Add sequence dimension for batch_first GRU: [B, 1, obs_dim + action_dim]
            gru_input = gru_input.unsqueeze(1)

            # ----------------------------------------------------------------
            # Step 2e: Update GRU hidden state.
            # ----------------------------------------------------------------
            # Pass current hidden state as h_0 to continue from where we left off.
            # This is critical — do NOT re-initialize hidden to zeros here.
            # gru_out: [B, 1, hidden_size]
            # hidden:  [num_layers, B, hidden_size]
            _, hidden = self.gru(gru_input, hidden)

            # Update h_last for the next iteration's head predictions.
            h_last = hidden[-1]  # [B, hidden_size]

        # ----------------------------------------------------------------
        # Phase 3: Stack predictions along the time dimension.
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

    def step(
        self,
        obs: Tensor,
        action: Tensor,
        hidden: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Perform a single-step world model prediction for imagination rollouts.

        Efficient single-step inference interface used by
        ``MBPOPPOTrainer.imagine_trajectories`` during the T=100 step
        imagination loop. Processes one (obs, action) pair through the GRU
        and applies the prediction heads to produce the next observation
        distribution.

        This method is called in a loop (T=100 times per imagination rollout
        for 4096 parallel environments), so it is optimized for minimal
        overhead: no list allocations, no stacking, single GRU step.

        The typical usage pattern in ``MBPOPPOTrainer.imagine_trajectories``:
          1. Call ``encode_history`` once to initialize ``hidden`` from the
             replay buffer context.
          2. Call ``step`` in a loop for T=100 steps, passing ``new_hidden``
             back as ``hidden`` at each iteration.
          3. Call ``sample_observation(obs_mean, obs_logstd)`` to get the
             next observation for the policy and reward computation.

        Args:
            obs: Current world model observation of shape ``[B, obs_dim]``.
                For ANYmal D: ``[B, 45]``. For Unitree G1: ``[B, 96]``.
                This is the observation at the current imagination step,
                either from the replay buffer (step 0) or from the previous
                ``sample_observation`` call (steps 1-99).
            action: Current action (joint position targets) of shape
                ``[B, action_dim]``. For ANYmal D: ``[B, 12]``. For G1:
                ``[B, 29]``. Produced by ``PolicyNetwork.get_action(policy_obs)``.
            hidden: Current GRU hidden state of shape
                ``[num_layers, B, hidden_size]`` = ``[2, B, 256]``.
                Initialized by ``encode_history`` at the start of each
                imagination rollout; updated by this method at each step.

        Returns:
            A tuple of five tensors:
              - ``obs_mean``: Predicted next observation mean, shape
                ``[B, obs_dim]``.
              - ``obs_logstd``: Predicted next observation log-std, shape
                ``[B, obs_dim]``. Clamped to ``[_LOGSTD_MIN, _LOGSTD_MAX]``.
              - ``priv_mean``: Predicted next privileged info mean, shape
                ``[B, priv_dim]``.
              - ``priv_logstd``: Predicted next privileged info log-std,
                shape ``[B, priv_dim]``.
              - ``new_hidden``: Updated GRU hidden state, shape
                ``[num_layers, B, hidden_size]``. Pass this as ``hidden``
                to the next ``step`` call.
        """
        # ----------------------------------------------------------------
        # 1. Prepare GRU input: concatenate obs and action.
        # ----------------------------------------------------------------
        # obs:    [B, obs_dim]
        # action: [B, action_dim]
        # x:      [B, obs_dim + action_dim]
        x: Tensor = torch.cat([obs, action], dim=-1)

        # Add sequence dimension for batch_first GRU.
        # GRU expects [B, T, D] with batch_first=True; T=1 for single step.
        # x_seq: [B, 1, obs_dim + action_dim]
        x_seq: Tensor = x.unsqueeze(1)

        # ----------------------------------------------------------------
        # 2. Single GRU step with the provided hidden state.
        # ----------------------------------------------------------------
        # gru_out:    [B, 1, hidden_size] — output for the single step
        # new_hidden: [num_layers, B, hidden_size] — updated hidden state
        gru_out: Tensor
        new_hidden: Tensor
        gru_out, new_hidden = self.gru(x_seq, hidden)

        # Extract the output for the single step: [B, hidden_size]
        # squeeze(1) removes the sequence dimension (T=1).
        h_last: Tensor = gru_out.squeeze(1)  # [B, hidden_size]

        