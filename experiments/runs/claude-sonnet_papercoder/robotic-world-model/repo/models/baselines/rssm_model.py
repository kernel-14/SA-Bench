## Code: models/baselines/rssm_model.py

```python
## models/baselines/rssm_model.py
"""Recurrent State-Space Model (RSSM) baseline for the RWM project.

This module implements the RSSM baseline described in Table S8 of the paper.
RSSM is the core architecture used in Dreamer/PlaNet and serves as a key
comparison point against RWM in Section 4.3 (Generality across Robotic
Environments).

The RSSM maintains a compound latent state (h_t, z_t):
  - h_t: deterministic GRU hidden state, shape [B, hidden_size=256]
  - z_t: stochastic categorical latent, shape [B, latent_dim * num_categories]
         = [B, 64 * 32] = [B, 2048]

The categorical distribution uses latent_dim=64 independent categorical
variables, each with num_categories=32 classes (Table S8). Gradients flow
through the discrete sampling via the straight-through estimator.

Key distinction from RWM: RSSM operates through a latent space — it
compresses observations into a stochastic categorical latent z combined
with a deterministic GRU hidden state h, rather than predicting directly
in observation space. The prior network predicts z from h alone (enabling
imagination without observations), while the posterior refines z using
the actual observation (used during training).

Architecture (Table S8):
  - Type: GRU
  - Hidden size: 256
  - Layers: 2 (GRU cell + encoder/prior/posterior MLPs)
  - Latent dimension: 64
  - Prior type: categorical
  - Categories: 32
  - Default training: teacher-forcing (use_autoregressive_training: false)

The paper notes: "when trained with autoregressive training, RSSM achieves
a performance comparable to the proposed GRU-based architecture. Nevertheless,
we opt for the GRU-based model due to its simplicity and computational
efficiency." (Section 4.3)

Usage:
    model = RSSMModel(config)
    # Teacher-forcing (N=1):
    obs_mean, obs_logstd, priv_mean, priv_logstd = model.forward(
        obs_history, action_history
    )
    # Autoregressive rollout (N steps):
    pred_obs_means, pred_priv_means = model.autoregressive_rollout(
        obs_history, action_history, future_actions, n_steps=8
    )
    # Single-step inference for imagination:
    obs_mean, obs_logstd, priv_mean, priv_logstd, new_hidden = model.step(
        obs, action, hidden=(h, z)
    )
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils.common import sample_gaussian


# ---------------------------------------------------------------------------
# Log-std clamping bounds for numerical stability.
# Consistent with GRUWorldModel and MLPModel clamping ranges.
# Range [-10, 2] corresponds to std in [exp(-10), exp(2)] ≈ [4.5e-5, 7.4].
# ---------------------------------------------------------------------------
_LOGSTD_MIN: float = -10.0
_LOGSTD_MAX: float = 2.0

# Encoder output size — fixed at 256 to match the GRU hidden size.
# This allows direct concatenation of (h, obs_embed) for the posterior.
_ENCODER_OUTPUT_SIZE: int = 256

# Backbone hidden size for prior, posterior, and decoder MLPs.
# Fixed at 256 to match Table S8: "hidden size 256".
_MLP_HIDDEN_SIZE: int = 256


def _build_mlp(
    input_size: int,
    hidden_size: int,
    output_size: int,
    activation: nn.Module = None,
) -> nn.Sequential:
    """Build a 2-layer MLP with ReLU activation.

    Used for the encoder, prior, posterior, and decoder components of RSSM.
    All internal MLPs use the same 2-layer structure for consistency.

    Args:
        input_size: Input feature dimension.
        hidden_size: Hidden layer dimension. 256 for all RSSM components.
        output_size: Output feature dimension.
        activation: Activation module for the hidden layer. Defaults to
            nn.ReLU() if None. Passed as a module instance (not class) to
            allow sharing activation instances if needed.

    Returns:
        A 3-module ``nn.Sequential``: Linear → Activation → Linear.
        No activation on the final layer — callers apply clamping for
        logstd outputs or use raw output for logit/mean outputs.
    """
    if activation is None:
        activation = nn.ReLU()
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        activation,
        nn.Linear(hidden_size, output_size),
    )


class RSSMModel(nn.Module):
    """Recurrent State-Space Model (RSSM) baseline world model.

    Implements the RSSM architecture from Table S8 of the paper. The model
    maintains a compound latent state (h, z) where h is a deterministic GRU
    hidden state and z is a stochastic categorical latent variable.

    The model supports two training modes:
      - Teacher-forcing (default, ``use_autoregressive_training=False``):
        Uses ground truth observations at every step. Corresponds to N=1
        in the autoregressive training framework (Section 3.2, Fig. 2b).
      - Autoregressive training (``use_autoregressive_training=True``):
        Uses the model's own predictions for future steps. Enables the
        RSSM to match RWM performance when trained this way (Section 4.3).

    The model exposes the same public interface as ``GRUWorldModel`` so that
    ``RWMTrainer``, ``Benchmark``, and ``MBPOPPOTrainer`` can use it
    interchangeably. The key difference is that ``encode_history`` returns
    a ``Tuple[Tensor, Tensor]`` (h, z) instead of a single ``Tensor``, and
    ``step`` accepts and returns a ``Tuple[Tensor, Tensor]`` as hidden state.

    Attributes:
        obs_dim: World model observation dimension. 45 for ANYmal D
            (Table S2), 96 for Unitree G1 (Table S2).
        action_dim: Action space dimension. 12 for ANYmal D (Table S4),
            29 for Unitree G1 (Table S4).
        priv_dim: Privileged information dimension. 8 for ANYmal D
            (Table S3), 30 for Unitree G1 (Table S3).
        hidden_size: GRU hidden state size. 256 (Table S8).
        latent_dim: Number of independent categorical variables. 64 (Table S8).
        num_categories: Number of classes per categorical variable. 32 (Table S8).
        latent_flat_dim: Flattened latent dimension = latent_dim * num_categories
            = 64 * 32 = 2048.
        history_horizon: Number of historical steps M for context. 32 (Table S10).
        forecast_horizon: Number of forecast steps N. 8 (Table S10).
        use_autoregressive_training: Whether to use autoregressive training.
            False by default (teacher-forcing, Table S8).
        encoder: MLP encoding observations to feature vectors. Input: obs_dim,
            output: _ENCODER_OUTPUT_SIZE=256.
        gru_cell: Single-step GRU cell. Input: obs_dim + action_dim + latent_flat_dim,
            hidden: hidden_size=256.
        prior_net: MLP predicting prior latent logits from h. Input: hidden_size,
            output: latent_dim * num_categories.
        posterior_net: MLP predicting posterior latent logits from (h, obs_embed).
            Input: hidden_size + _ENCODER_OUTPUT_SIZE, output: latent_dim * num_categories.
        obs_head: MLP decoding (h, z) to observation distribution parameters.
            Input: hidden_size + latent_flat_dim, output: obs_dim * 2 (mean + logstd).
        priv_head: MLP decoding (h, z) to privileged info distribution parameters.
            Input: hidden_size + latent_flat_dim, output: priv_dim * 2 (mean + logstd).
    """

    def __init__(self, config: object) -> None:
        """Initialize the RSSM model from the experiment configuration.

        Extracts robot-specific dimensions from the robot sub-config and
        RSSM architecture parameters from the ``baselines.rssm`` sub-config.
        Builds the encoder, GRU cell, prior, posterior, and decoder components.

        Args:
            config: Hydra ``DictConfig`` or plain dict containing the full
                experiment configuration from ``config.yaml``. Must contain:
                - ``config.robot``: "anymal_d" or "unitree_g1"
                - ``config[robot_type].obs_dim``: observation dimension
                - ``config[robot_type].action_dim``: action dimension
                - ``config[robot_type].priv_dim``: privileged info dimension
                - ``config.baselines.rssm.hidden_size``: 256 (Table S8)
                - ``config.baselines.rssm.latent_dim``: 64 (Table S8)
                - ``config.baselines.rssm.num_categories``: 32 (Table S8)
                - ``config.baselines.rssm.use_autoregressive_training``: false
                - ``config.rwm.history_horizon``: 32 (Table S10)
                - ``config.rwm.forecast_horizon``: 8 (Table S10)

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
        # 2. Extract RSSM architecture parameters from config.baselines.rssm
        # ----------------------------------------------------------------
        rssm_cfg = config.baselines.rssm  # type: ignore[union-attr]

        # GRU hidden state size: 256 (Table S8: "hidden size 256")
        self.hidden_size: int = int(rssm_cfg.hidden_size)

        # Number of independent categorical variables: 64 (Table S8: "latent dimension 64")
        self.latent_dim: int = int(rssm_cfg.latent_dim)

        # Number of classes per categorical variable: 32 (Table S8: "categories 32")
        self.num_categories: int = int(rssm_cfg.num_categories)

        # Flattened latent dimension: 64 * 32 = 2048
        # This is the size of z when reshaped from [B, latent_dim, num_categories]
        # to [B, latent_dim * num_categories] for concatenation with h.
        self.latent_flat_dim: int = self.latent_dim * self.num_categories

        # Training mode: teacher-forcing (default) or autoregressive
        # Table S8: "use_autoregressive_training: false" (default)
        self.use_autoregressive_training: bool = bool(
            rssm_cfg.use_autoregressive_training
        )

        # ----------------------------------------------------------------
        # 3. Extract horizon parameters from config.rwm (shared with RWM)
        # ----------------------------------------------------------------
        # All baselines use the same context during training and evaluation
        # per Section 4.3: "All models are given the same context during
        # training and evaluation."
        rwm_cfg = config.rwm  # type: ignore[union-attr]

        # History horizon M = 32 (Table S10)
        self.history_horizon: int = int(rwm_cfg.history_horizon)

        # Forecast horizon N = 8 (Table S10)
        self.forecast_horizon: int = int(rwm_cfg.forecast_horizon)

        # ----------------------------------------------------------------
        # 4. Build encoder: obs_dim -> _ENCODER_OUTPUT_SIZE=256
        # ----------------------------------------------------------------
        # Encodes the current observation into a feature vector for the
        # posterior network. The output size matches hidden_size to allow
        # direct concatenation of (h, obs_embed) for the posterior input.
        self.encoder: nn.Sequential = _build_mlp(
            input_size=self.obs_dim,
            hidden_size=_MLP_HIDDEN_SIZE,
            output_size=_ENCODER_OUTPUT_SIZE,
        )

        # ----------------------------------------------------------------
        # 5. Build GRU cell (single-step, no sequence dimension)
        # ----------------------------------------------------------------
        # Input at each step: concat(obs_t, action_t, z_{t-1})
        # Input size: obs_dim + action_dim + latent_flat_dim
        # For ANYmal D: 45 + 12 + 2048 = 2105
        # For Unitree G1: 96 + 29 + 2048 = 2173
        #
        # Using nn.GRUCell (not nn.GRU) because we need to interleave GRU
        # updates with latent sampling at each step. nn.GRUCell processes
        # one step at a time, which is required for the RSSM's sequential
        # posterior/prior computation.
        self.gru_cell: nn.GRUCell = nn.GRUCell(
            input_size=self.obs_dim + self.action_dim + self.latent_flat_dim,
            hidden_size=self.hidden_size,
        )

        # ----------------------------------------------------------------
        # 6. Build prior network: h -> latent logits
        # ----------------------------------------------------------------
        # Predicts the distribution of z_t given only h_t (no observation).
        # Used during autoregressive rollout (imagination) when ground truth
        # observations are unavailable.
        # Input: h_t, shape [B, hidden_size=256]
        # Output: prior logits, shape [B, latent_dim * num_categories=2048]
        self.prior_net: nn.Sequential = _build_mlp(
            input_size=self.hidden_size,
            hidden_size=_MLP_HIDDEN_SIZE,
            output_size=self.latent_flat_dim,
        )

        # ----------------------------------------------------------------
        # 7. Build posterior network: (h, obs_embed) -> latent logits
        # ----------------------------------------------------------------
        # Refines the latent estimate using the actual observation.
        # Used during training (teacher-forcing) and history encoding.
        # Input: concat(h_t, obs_embed_t), shape [B, hidden_size + encoder_output]
        #      = [B, 256 + 256] = [B, 512]
        # Output: posterior logits, shape [B, latent_dim * num_categories=2048]
        self.posterior_net: nn.Sequential = _build_mlp(
            input_size=self.hidden_size + _ENCODER_OUTPUT_SIZE,
            hidden_size=_MLP_HIDDEN_SIZE,
            output_size=self.latent_flat_dim,
        )

        # ----------------------------------------------------------------
        # 8. Build observation decoder head: (h, z) -> obs distribution
        # ----------------------------------------------------------------
        # Decodes the compound state (h, z) into a Gaussian distribution
        # over the next observation. Output is mean + logstd concatenated.
        # Input: concat(h, z), shape [B, hidden_size + latent_flat_dim]
        #      = [B, 256 + 2048] = [B, 2304]
        # Output: [B, obs_dim * 2] — split into mean [B, obs_dim] and logstd [B, obs_dim]
        self.obs_head: nn.Sequential = _build_mlp(
            input_size=self.hidden_size + self.latent_flat_dim,
            hidden_size=_MLP_HIDDEN_SIZE,
            output_size=self.obs_dim * 2,
        )

        # ----------------------------------------------------------------
        # 9. Build privileged info decoder head: (h, z) -> priv distribution
        # ----------------------------------------------------------------
        # Decodes the compound state (h, z) into a distribution over
        # privileged information. Same structure as obs_head.
        # Input: concat(h, z), shape [B, hidden_size + latent_flat_dim]
        # Output: [B, priv_dim * 2] — split into mean and logstd
        self.priv_head: nn.Sequential = _build_mlp(
            input_size=self.hidden_size + self.latent_flat_dim,
            hidden_size=_MLP_HIDDEN_SIZE,
            output_size=self.priv_dim * 2,
        )

    # ----------------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------------

    def _straight_through_sample(self, logits: Tensor) -> Tensor:
        """Sample from categorical distribution using straight-through estimator.

        Implements the straight-through gradient estimator for discrete
        categorical samples. This enables backpropagation through the
        discrete sampling operation, which is otherwise non-differentiable.

        The straight-through trick:
            z = hard + probs - probs.detach()

        Forward pass: z.data == hard.data (discrete one-hot)
        Backward pass: dz/d_logits == d_probs/d_logits (continuous gradient)

        This is the standard approach used in DreamerV2/V3 for categorical
        latent variables. The key identity ensures:
          - The forward computation uses the hard (argmax) sample
          - The backward computation uses the soft (softmax) probabilities
          - No gradient flows through the detached probs term

        Args:
            logits: Categorical logits of shape ``[B, latent_dim, num_categories]``
                = ``[B, 64, 32]``. These are the raw (unnormalized) log-probabilities
                for each of the latent_dim categorical variables.

        Returns:
            Straight-through sample of shape ``[B, latent_dim * num_categories]``
            = ``[B, 2048]``. In the forward pass, this is a flattened one-hot
            encoding of the argmax. In the backward pass, gradients flow through
            the softmax probabilities.
        """
        batch_size: int = logits.shape[0]

        # ----------------------------------------------------------------
        # 1. Compute soft probabilities via softmax over the categories dim.
        # ----------------------------------------------------------------
        # probs: [B, latent_dim, num_categories]
        # Each row along the last dimension sums to 1.0.
        probs: Tensor = F.softmax(logits, dim=-1)

        # ----------------------------------------------------------------
        # 2. Compute hard one-hot encoding via argmax.
        # ----------------------------------------------------------------
        # indices: [B, latent_dim] — index of the most probable category
        indices: Tensor = logits.argmax(dim=-1)

        # hard: [B, latent_dim, num_categories] — one-hot encoding
        # F.one_hot returns a LongTensor; convert to float for arithmetic.
        hard: Tensor = F.one_hot(indices, num_classes=self.num_categories).float()

        # ----------------------------------------------------------------
        # 3. Apply straight-through estimator.
        # ----------------------------------------------------------------
        # z = hard + probs - probs.detach()
        # Forward: z.data == hard.data (discrete)
        # Backward: dz/d_logits == d_probs/d_logits (continuous)
        #
        # The .detach() on probs removes it from the computation graph,
        # so the gradient of z w.r.t. logits is exactly d_probs/d_logits
        # (the Jacobian of softmax), not zero.
        z: Tensor = hard + probs - probs.detach()
        # z: [B, latent_dim, num_categories]

        # ----------------------------------------------------------------
        # 4. Flatten to [B, latent_dim * num_categories].
        # ----------------------------------------------------------------
        # reshape is safe here since z is contiguous after the arithmetic ops.
        z_flat: Tensor = z.reshape(batch_size, self.latent_flat_dim)
        # z_flat: [B, latent_flat_dim] = [B, 2048]

        return z_flat

    def _decode_state(
        self,
        h: Tensor,
        z: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Decode the compound state (h, z) into observation and priv distributions.

        Applies the obs_head and priv_head decoders to the concatenated
        compound state, splits the output into mean and logstd, and clamps
        the logstd for numerical stability.

        Args:
            h: Deterministic GRU hidden state of shape ``[B, hidden_size]``
                = ``[B, 256]``.
            z: Flattened stochastic latent of shape ``[B, latent_flat_dim]``
                = ``[B, 2048]``.

        Returns:
            A tuple ``(obs_mean, obs_logstd, priv_mean, priv_logstd)`` where:
              - ``obs_mean``: shape ``[B, obs_dim]``
              - ``obs_logstd``: shape ``[B, obs_dim]``, clamped to
                ``[_LOGSTD_MIN, _LOGSTD_MAX]``
              - ``priv_mean``: shape ``[B, priv_dim]``
              - ``priv_logstd``: shape ``[B, priv_dim]``, clamped to
                ``[_LOGSTD_MIN, _LOGSTD_MAX]``
        """
        # Concatenate deterministic and stochastic states.
        # state: [B, hidden_size + latent_flat_dim] = [B, 2304]
        state: Tensor = torch.cat([h, z], dim=-1)

        # ----------------------------------------------------------------
        # Decode observation distribution.
        # ----------------------------------------------------------------
        # obs_out: [B, obs_dim * 2]
        obs_out: Tensor = self.obs_head(state)

        # Split into mean and logstd along the last dimension.
        # chunk(2, dim=-1) splits [B, obs_dim*2] into two [B, obs_dim] tensors.
        obs_mean: Tensor
        obs_logstd_raw: Tensor
        obs_mean, obs_logstd_raw = obs_out.chunk(2, dim=-1)
        # obs_mean:     [B, obs_dim]
        # obs_logstd_raw: [B, obs_dim]

        # Clamp logstd to prevent numerical overflow in exp(logstd).
        obs_logstd: Tensor = torch.clamp(obs_logstd_raw, _LOGSTD_MIN, _LOGSTD_MAX)

        # ----------------------------------------------------------------
        # Decode privileged information distribution.
        # ----------------------------------------------------------------
        # priv_out: [B, priv_dim * 2]
        priv_out: Tensor = self.priv_head(state)

        priv_mean: Tensor
        priv_logstd_raw: Tensor
        priv_mean, priv_logstd_raw = priv_out.chunk(2, dim=-1)
        # priv_mean:     [B, priv_dim]
        # priv_logstd_raw: [B, priv_dim]

        priv_logstd: Tensor = torch.clamp(priv_logstd_raw, _LOGSTD_MIN, _LOGSTD_MAX)

        return obs_mean, obs_logstd, priv_mean, priv_logstd

    def _init_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor]:
        """Initialize the compound latent state (h, z) to zeros.

        Creates zero-initialized tensors for both the deterministic GRU
        hidden state h and the stochastic categorical latent z. Zero
        initialization for z corresponds to a near-uniform categorical
        distribution (softmax of zeros is uniform), which is a reasonable
        prior at the start of a trajectory.

        Args:
            batch_size: Number of parallel environments B.
            device: Target device for the tensors.

        Returns:
            A tuple ``(h, z)`` where:
              - ``h``: Zero tensor of shape ``[B, hidden_size]`` = ``[B, 256]``
              - ``z``: Zero tensor of shape ``[B, latent_flat_dim]`` = ``[B, 2048]``
        """
        h: Tensor = torch.zeros(
            batch_size,
            self.hidden_size,
            dtype=torch.float32,
            device=device,
        )
        z: Tensor = torch.zeros(
            batch_size,
            self.latent_flat_dim,
            dtype=torch.float32,
            device=device,
        )
        return h, z

    def _process_history_step(
        self,
        obs_t: Tensor,
        action_t: Tensor,
        h: Tensor,
        z: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Process one history step using the posterior (ground truth obs available).

        Implements one step of the inner autoregression during history encoding.
        Uses the posterior network to refine the latent estimate using the
        actual observation.

        Step logic:
          1. Encode observation: obs_embed = encoder(obs_t)
          2. Update GRU: h_new = gru_cell(concat(obs_t, action_t, z), h)
          3. Compute posterior logits: post_logits = posterior_net(concat(h_new, obs_embed))
          4. Sample z_new via straight-through: z_new = _straight_through_sample(post_logits)

        Args:
            obs_t: Current observation of shape ``[B, obs_dim]``.
            action_t: Current action of shape ``[B, action_dim]``.
            h: Current GRU hidden state of shape ``[B, hidden_size]``.
            z: Current flattened latent of shape ``[B, latent_flat_dim]``.

        Returns:
            A tuple ``(new_h, new_z)`` where:
              - ``new_h``: Updated GRU hidden state, shape ``[B, hidden_size]``
              - ``new_z``: Updated latent sample, shape ``[B, latent_flat_dim]``
        """
        # ----------------------------------------------------------------
        # 1. Encode observation into feature vector.
        # ----------------------------------------------------------------
        # obs_embed: [B, _ENCODER_OUTPUT_SIZE=256]
        obs_embed: Tensor = self.encoder(obs_t)

        # ----------------------------------------------------------------
        # 2. Update deterministic GRU hidden state.
        # ----------------------------------------------------------------
        # GRU input: concat(obs_t, action_t, z)
        # shape: [B, obs_dim + action_dim + latent_flat_dim]
        gru_input: Tensor = torch.cat([obs_t, action_t, z], dim=-1)

        # GRUCell: (input [B, input_size], hidden [B, hidden_size]) -> [B, hidden_size]
        new_h: Tensor = self.gru_cell(gru_input, h)
        # new_h: [B, hidden_size=256]

        # ----------------------------------------------------------------
        # 3. Compute posterior logits using (h_new, obs_embed).
        # ----------------------------------------------------------------
        # posterior_input: concat(new_h, obs_embed), shape [B, 256 + 256] = [B, 512]
        posterior_input: Tensor = torch.cat([new_h, obs_embed], dim=-1)

        # post_logits_flat: [B, latent_flat_dim=2048]
        post_logits_flat: Tensor = self.posterior_net(posterior_input)

        # Reshape to [B, latent_dim, num_categories] for straight-through sampling.
        batch_size: int = obs_t.shape[0]
        post_logits: Tensor = post_logits_flat.reshape(
            batch_size, self.latent_dim, self.num_categories
        )

        # ----------------------------------------------------------------
        # 4. Sample new latent via straight-through estimator.
        # ----------------------------------------------------------------
        # new_z: [B, latent_flat_dim=2048]
        new_z: Tensor = self._straight_through_sample(post_logits)

        return new_h, new_z

    def _process_forecast_step(
        self,
        obs_t: Tensor,
        action_t: Tensor,
        h: Tensor,
        z: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Process one forecast step using the prior (no ground truth obs).

        Implements one step of the outer autoregression during imagination.
        Uses the prior network to predict the latent from h alone (no
        observation available during imagination).

        Step logic:
          1. Update GRU: h_new = gru_cell(concat(obs_t, action_t, z), h)
             where obs_t is the model's own predicted observation (sampled)
          2. Compute prior logits: prior_logits = prior_net(h_new)
          3. Sample z_new via straight-through: z_new = _straight_through_sample(prior_logits)

        Args:
            obs_t: Predicted (sampled) observation of shape ``[B, obs_dim]``.
                This is the model's own prediction from the previous step,
                NOT a ground truth observation.
            action_t: Current action of shape ``[B, action_dim]``.
            h: Current GRU hidden state of shape ``[B, hidden_size]``.
            z: Current flattened latent of shape ``[B, latent_flat_dim]``.

        Returns:
            A tuple ``(new_h, new_z)`` where:
              - ``new_h``: Updated GRU hidden state, shape ``[B, hidden_size]``
              - ``new_z``: Updated latent sample from prior, shape ``[B, latent_flat_dim]``
        """
        # ----------------------------------------------------------------
        # 1. Update deterministic GRU hidden state.
        # ----------------------------------------------------------------
        # GRU input: concat(obs_t, action_t, z)
        # obs_t here is the model's own predicted observation (not ground truth)
        gru_input: Tensor = torch.cat([obs_t, action_t, z], dim=-1)
        # shape: [B, obs_dim + action_dim + latent_flat_dim]

        new_h: Tensor = self.gru_cell(gru_input, h)
        # new_h: [B, hidden_size=256]

        # ----------------------------------------------------------------
        # 2. Compute prior logits from h_