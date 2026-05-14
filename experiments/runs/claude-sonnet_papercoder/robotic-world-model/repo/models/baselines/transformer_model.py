## Code: models/baselines/transformer_model.py

```python
## models/baselines/transformer_model.py
"""Transformer baseline world model for the Robotic World Model (RWM) project.

This module implements the decoder-only Transformer baseline described in
Table S8 of the paper. It serves as one of three baselines (MLP, RSSM,
Transformer) compared against RWM-AR in Section 4.3 (Generality across
Robotic Environments).

Architecture (Table S8):
  - Type: decoder-only transformer
  - Dimension (d_model): 64
  - Attention heads (nhead): 8
  - Layers: 2
  - Context length: 32 (matches history_horizon M=32 from Table S10)
  - Positional encoding: sinusoidal
  - Feedforward dimension: 256 (4x d_model, standard)

The model is trained with **teacher-forcing by default**
(``use_autoregressive_training: false`` in ``configs/baselines.yaml``),
meaning it uses ground truth observations as inputs at every step during
training — equivalent to N=1 in the RWM autoregressive framework.

**GPU Memory Limitation (Section 4.3):**
Autoregressive training (AR) with gradient propagation through N forecast
steps is NOT supported for this architecture due to GPU memory constraints.
With N=8 forecast steps, batch_size=1024, and d_model=64, the computation
graph depth causes OOM on RTX 4090. This model is trained with teacher-
forcing (N=1) by default. Autoregressive rollout is available for evaluation
only — use within a ``torch.no_grad()`` context to avoid OOM.

The paper states: "training transformer architectures with autoregressive
training does not scale effectively, as the multi-step gradient propagation
in autoregressive forecasting leads to GPU memory constraints, limiting their
practicality for this approach." (Section 4.3)

The decoder-only design is implemented using ``nn.TransformerDecoder`` with
``memory=tgt`` (same sequence passed as both target and memory). A causal
mask prevents attending to future tokens, enforcing left-to-right attention.
The last token's output is used to predict the distribution of the next
observation and privileged information.

Usage:
    model = TransformerModel(config)
    # Teacher-forcing (N=1, default training mode):
    obs_mean, obs_logstd, priv_mean, priv_logstd = model.forward(
        obs_history, action_history
    )
    # Autoregressive rollout (evaluation only — use torch.no_grad()):
    with torch.no_grad():
        pred_obs_means, pred_obs_logstds, pred_priv_means, pred_priv_logstds = (
            model.autoregressive_rollout(
                obs_history, action_history, future_actions, n_steps=8
            )
        )
"""

import math
import warnings
from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from utils.common import sample_gaussian


# ---------------------------------------------------------------------------
# Log-std clamping bounds for numerical stability.
# Consistent with GRUWorldModel and MLPModel clamping ranges.
# Range [-5, 2] corresponds to std in [exp(-5), exp(2)] ≈ [0.007, 7.4],
# which covers all physically meaningful prediction uncertainties while
# preventing NaN losses from extreme variance values in early training.
# ---------------------------------------------------------------------------
_LOGSTD_MIN: float = -5.0
_LOGSTD_MAX: float = 2.0


def _sinusoidal_encoding(length: int, d_model: int) -> Tensor:
    """Compute sinusoidal positional encoding for a sequence of given length.

    Implements the standard sinusoidal positional encoding from
    "Attention is All You Need" (Vaswani et al., 2017):

        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    This encoding is deterministic and position-dependent, allowing the
    transformer to distinguish tokens at different positions in the sequence.
    It is registered as a buffer (not a parameter) in ``TransformerModel``
    so it moves with ``.to(device)`` but is not updated by the optimizer.

    The encoding is pre-computed once in ``__init__`` for the full
    ``context_length=32`` and sliced to the actual sequence length M in
    ``forward``. This handles the edge case where M < context_length.

    Args:
        length: Sequence length to compute encodings for. Corresponds to
            ``context_length=32`` from ``configs/baselines.yaml``, which
            matches ``history_horizon=32`` from ``config.rwm``.
        d_model: Model dimension. 64 from ``configs/baselines.yaml``
            (``baselines.transformer.d_model``). Must be even for the
            sin/cos alternation to work correctly.

    Returns:
        Positional encoding tensor of shape ``[length, d_model]``.
        Each row ``pe[pos]`` contains the encoding for position ``pos``.
        The tensor is on CPU (moved to the target device when registered
        as a buffer via ``register_buffer``).

    Raises:
        ValueError: If ``d_model`` is odd (sin/cos alternation requires
            even d_model for clean pairing).
    """
    if d_model % 2 != 0:
        raise ValueError(
            f"d_model must be even for sinusoidal positional encoding, "
            f"got d_model={d_model}. "
            "Check baselines.transformer.d_model in configs/baselines.yaml."
        )

    # ----------------------------------------------------------------
    # 1. Create position indices: [0, 1, ..., length-1] as column vector.
    # ----------------------------------------------------------------
    # position: [length, 1] — broadcast over d_model dimension
    position: Tensor = torch.arange(
        length, dtype=torch.float32
    ).unsqueeze(1)  # shape: [length, 1]

    # ----------------------------------------------------------------
    # 2. Compute division terms for the frequency scaling.
    # ----------------------------------------------------------------
    # div_term[i] = 1 / 10000^(2i / d_model) = exp(-log(10000) * 2i / d_model)
    # for i in [0, 1, ..., d_model/2 - 1]
    # div_term: [d_model/2]
    div_term: Tensor = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )  # shape: [d_model // 2]

    # ----------------------------------------------------------------
    # 3. Compute sinusoidal encodings.
    # ----------------------------------------------------------------
    # pe: [length, d_model], initialized to zeros
    pe: Tensor = torch.zeros(length, d_model, dtype=torch.float32)

    # Even indices (0, 2, 4, ...): sin(position * div_term)
    # position * div_term broadcasts: [length, 1] * [d_model/2] -> [length, d_model/2]
    pe[:, 0::2] = torch.sin(position * div_term)

    # Odd indices (1, 3, 5, ...): cos(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe  # shape: [length, d_model]


class TransformerModel(nn.Module):
    """Decoder-only Transformer baseline world model.

    Implements the Transformer baseline from Table S8 of the paper. The model
    processes the M-step observation-action history as a sequence of tokens,
    applies causal self-attention via a decoder-only architecture, and predicts
    the distribution of the next observation and privileged information from
    the last token's output.

    The decoder-only design is implemented using ``nn.TransformerDecoder``
    with ``memory=tgt`` (same sequence as both target and memory). A causal
    mask on ``tgt_mask`` prevents attending to future tokens. The cross-
    attention in each decoder layer effectively becomes a second self-attention
    over the same sequence, which is the standard decoder-only trick in PyTorch.

    **Key difference from RWM:** The Transformer processes all M tokens in
    parallel (O(M²) attention), while the GRU processes them sequentially
    (O(M) time). The Transformer's global attention can capture long-range
    dependencies but lacks the GRU's inductive bias for sequential dynamics.
    In practice, the GRU outperforms the Transformer for robotic locomotion
    tasks (Section 4.3), likely because the sequential inductive bias is
    well-matched to the temporal structure of robot dynamics.

    **Training mode:** Teacher-forcing by default (``use_autoregressive_training:
    false``). The ``autoregressive_rollout`` method is available for evaluation
    only — do NOT call it with gradients enabled (OOM risk).

    Attributes:
        obs_dim: World model observation dimension. 45 for ANYmal D
            (Table S2), 96 for Unitree G1 (Table S2).
        action_dim: Action space dimension. 12 for ANYmal D (Table S4),
            29 for Unitree G1 (Table S4).
        priv_dim: Privileged information dimension. 8 for ANYmal D
            (Table S3), 30 for Unitree G1 (Table S3).
        d_model: Transformer model dimension. 64 (Table S8,
            ``baselines.transformer.d_model``).
        nhead: Number of attention heads. 8 (Table S8,
            ``baselines.transformer.nhead``).
        num_layers: Number of transformer decoder layers. 2 (Table S8,
            ``baselines.transformer.num_layers``).
        context_length: Maximum sequence length for positional encoding.
            32 (Table S8, ``baselines.transformer.context_length``).
            Matches ``history_horizon=32`` from ``config.rwm``.
        history_horizon: Number of historical steps M for context.
            32 (Table S10, ``config.rwm.history_horizon``).
        forecast_horizon: Number of forecast steps N. 8 (Table S10,
            ``config.rwm.forecast_horizon``).
        use_autoregressive_training: Whether AR training is enabled.
            False by default (Table S8). AR training causes OOM for
            this architecture (Section 4.3).
        input_proj: Linear projection from ``obs_dim + action_dim`` to
            ``d_model``. Maps each timestep's ``[o_t, a_t]`` concatenation
            into the transformer's token space.
        transformer: ``nn.TransformerDecoder`` with 2 decoder layers,
            d_model=64, nhead=8, dim_feedforward=256, batch_first=True.
        pos_encoding: Sinusoidal positional encoding buffer of shape
            ``[context_length, d_model]`` = ``[32, 64]``. Registered as
            a buffer (not a parameter) — moves with ``.to(device)`` but
            is not updated by the optimizer.
        obs_head_mean: Linear head predicting observation mean.
            Input: d_model=64, output: obs_dim.
        obs_head_logstd: Linear head predicting observation log-std.
            Input: d_model=64, output: obs_dim.
        priv_head_mean: Linear head predicting privileged info mean.
            Input: d_model=64, output: priv_dim.
        priv_head_logstd: Linear head predicting privileged info log-std.
            Input: d_model=64, output: priv_dim.
    """

    def __init__(self, config: object) -> None:
        """Initialize the Transformer baseline from the experiment configuration.

        Extracts robot-specific dimensions from the robot sub-config and
        Transformer architecture parameters from the ``baselines.transformer``
        sub-config. Builds the input projection, positional encoding buffer,
        transformer decoder, and four linear prediction heads.

        The positional encoding is pre-computed once and registered as a
        buffer so it is automatically moved to the correct device when
        ``.to(device)`` is called on the model.

        Args:
            config: Hydra ``DictConfig`` or plain dict containing the full
                experiment configuration from ``config.yaml``. Must contain:
                - ``config.robot``: "anymal_d" or "unitree_g1"
                - ``config[robot_type].obs_dim``: observation dimension
                - ``config[robot_type].action_dim``: action dimension
                - ``config[robot_type].priv_dim``: privileged info dimension
                - ``config.baselines.transformer.d_model``: 64 (Table S8)
                - ``config.baselines.transformer.nhead``: 8 (Table S8)
                - ``config.baselines.transformer.num_layers``: 2 (Table S8)
                - ``config.baselines.transformer.context_length``: 32 (Table S8)
                - ``config.baselines.transformer.dim_feedforward``: 256
                - ``config.baselines.transformer.use_autoregressive_training``:
                  false (Table S8)
                - ``config.rwm.history_horizon``: 32 (Table S10)
                - ``config.rwm.forecast_horizon``: 8 (Table S10)

        Raises:
            ValueError: If ``config.robot`` is not "anymal_d" or "unitree_g1".
            ValueError: If ``d_model`` is not divisible by ``nhead``
                (required for multi-head attention).
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
        # 2. Extract Transformer architecture parameters from
        #    config.baselines.transformer (Table S8)
        # ----------------------------------------------------------------
        transformer_cfg = config.baselines.transformer  # type: ignore[union-attr]

        # Model dimension: 64 (Table S8: "dimension 64")
        self.d_model: int = int(transformer_cfg.d_model)

        # Number of attention heads: 8 (Table S8: "heads 8")
        self.nhead: int = int(transformer_cfg.nhead)

        # Number of decoder layers: 2 (Table S8: "layers 2")
        self.num_layers: int = int(transformer_cfg.num_layers)

        # Context length for positional encoding: 32 (Table S8: "context length 32")
        # Matches history_horizon=32 from config.rwm (Table S10).
        self.context_length: int = int(transformer_cfg.context_length)

        # Feedforward dimension: 256 (config.yaml: "dim_feedforward: 256")
        # Standard 4x d_model ratio: 4 * 64 = 256.
        self.dim_feedforward: int = int(transformer_cfg.dim_feedforward)

        # Training mode: teacher-forcing (default) or autoregressive.
        # Table S8: "use_autoregressive_training: false"
        # AR training causes OOM for this architecture (Section 4.3).
        self.use_autoregressive_training: bool = bool(
            transformer_cfg.use_autoregressive_training
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
        # 4. Validate d_model divisibility by nhead
        # ----------------------------------------------------------------
        # Multi-head attention requires d_model to be divisible by nhead
        # so that each head gets d_model // nhead = 64 // 8 = 8 dimensions.
        if self.d_model % self.nhead != 0:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by nhead={self.nhead} "
                "for multi-head attention. "
                "Check baselines.transformer.d_model and baselines.transformer.nhead "
                "in configs/baselines.yaml."
            )

        # ----------------------------------------------------------------
        # 5. Warn if AR training is enabled (OOM risk)
        # ----------------------------------------------------------------
        if self.use_autoregressive_training:
            warnings.warn(
                "TransformerModel: use_autoregressive_training=True is set. "
                "This may cause GPU OOM errors during training due to multi-step "
                "gradient propagation through the transformer's computation graph. "
                "The paper (Section 4.3) notes this limitation: 'training transformer "
                "architectures with autoregressive training does not scale effectively, "
                "as the multi-step gradient propagation in autoregressive forecasting "
                "leads to GPU memory constraints.' "
                "Consider setting use_autoregressive_training=false in "
                "configs/baselines.yaml.",
                UserWarning,
                stacklevel=2,
            )

        # ----------------------------------------------------------------
        # 6. Build input projection: (obs_dim + action_dim) -> d_model
        # ----------------------------------------------------------------
        # Projects each timestep's concatenated [o_t, a_t] vector from the
        # raw feature space into the transformer's d_model=64 token space.
        # For ANYmal D: (45 + 12) = 57 -> 64
        # For Unitree G1: (96 + 29) = 125 -> 64
        self.input_proj: nn.Linear = nn.Linear(
            self.obs_dim + self.action_dim,
            self.d_model,
        )

        # ----------------------------------------------------------------
        # 7. Build transformer decoder (decoder-only architecture)
        # ----------------------------------------------------------------
        # Using nn.TransformerDecoder with memory=tgt (same sequence) to
        # implement a decoder-only transformer. The causal mask on tgt_mask
        # enforces left-to-right attention.
        #
        # batch_first=True: input/output shape is [B, T, D] (not [T, B, D]).
        # This is consistent with the rest of the codebase which uses
        # batch-first convention throughout.
        decoder_layer: nn.TransformerDecoderLayer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=0.0,          # No dropout — deterministic evaluation needed
            activation="relu",    # Standard ReLU activation
            batch_first=True,     # [B, T, D] convention
            norm_first=False,     # Post-norm (standard transformer)
        )
        self.transformer: nn.TransformerDecoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=self.num_layers,
        )

        # ----------------------------------------------------------------
        # 8. Register sinusoidal positional encoding as a buffer
        # ----------------------------------------------------------------
        # Pre-compute the full context_length=32 positional encoding once.
        # Registered as a buffer (not a parameter) so it:
        #   - Moves with .to(device) automatically
        #   - Is NOT updated by the optimizer
        #   - Is included in state_dict for checkpoint saving/loading
        #
        # Shape: [context_length, d_model] = [32, 64]
        pe: Tensor = _sinusoidal_encoding(self.context_length, self.d_model)
        self.register_buffer("pos_encoding", pe)
        # self.pos_encoding: Tensor[context_length, d_model] = Tensor[32, 64]

        # ----------------------------------------------------------------
        # 9. Build four linear prediction heads
        # ----------------------------------------------------------------
        # The design specifies single Linear layers (not MLP heads) for the
        # Transformer baseline, unlike RWM and MLP which use 2-layer MLP heads.
        # This is consistent with the "decoder" architecture where the
        # transformer layers already provide sufficient non-linearity.
        # Input: d_model=64 (last token's output from transformer)

        # Observation prediction heads: output obs_dim
        self.obs_head_mean: nn.Linear = nn.Linear(self.d_model, self.obs_dim)
        self.obs_head_logstd: nn.Linear = nn.Linear(self.d_model, self.obs_dim)

        # Privileged information prediction heads: output priv_dim
        # For ANYmal D: priv_dim=8 (binary contacts → BCE loss in trainer)
        # For Unitree G1: priv_dim=30 (mixed binary + continuous)
        self.priv_head_mean: nn.Linear = nn.Linear(self.d_model, self.priv_dim)
        self.priv_head_logstd: nn.Linear = nn.Linear(self.d_model, self.priv_dim)

    # ----------------------------------------------------------------
    # Core methods
    # ----------------------------------------------------------------

    def forward(
        self,
        obs_history: Tensor,
        action_history: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Predict the next observation distribution (teacher-forcing mode).

        Implements the N=1 special case: processes the M-step history as a
        sequence of tokens through the causal transformer, and predicts the
        distribution of the NEXT step from the last token's output.

        This is the primary training method for the Transformer baseline
        (teacher-forcing, ``use_autoregressive_training=False``). It is also
        called iteratively by ``autoregressive_rollout`` for evaluation.

        **Decoder-only implementation detail:**
        ``nn.TransformerDecoder`` is used with ``memory=tgt`` (same sequence
        passed as both target and memory). The causal mask on ``tgt_mask``
        prevents attending to future tokens. The ``memory_mask`` is also set
        to the causal mask to maintain full causality in the cross-attention
        layers. This implements the decoder-only architecture described in
        Table S8 ("type: decoder").

        **Last token output:**
        Only the last token's output (``out[:, -1, :]``) is used for
        prediction. This token has attended to all previous tokens (via
        causal attention) and encodes the prediction for the next timestep.
        This is consistent with GPT-style autoregressive language models.

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
        # 2. Project each token to d_model dimensions.
        # ----------------------------------------------------------------
        # input_proj: Linear(obs_dim + action_dim -> d_model)
        # x: [B, M, d_model=64]
        x = self.input_proj(x)

        # ----------------------------------------------------------------
        # 3. Add sinusoidal positional encoding.
        # ----------------------------------------------------------------
        # Sequence length M may be <= context_length=32.
        # Slice pos_encoding to match the actual sequence length.
        # pos_encoding: [context_length, d_model] = [32, 64]
        # pos_encoding[:M, :]: [M, d_model]
        # Broadcasting: [B, M, d_model] + [M, d_model] -> [B, M, d_model]
        seq_len: int = x.shape[1]  # M
        x = x + self.pos_encoding[:seq_len, :]  # type: ignore[index]
        # x: [B, M, d_model=64]

        # ----------------------------------------------------------------
        # 4. Generate causal (upper-triangular) attention mask.
        # ----------------------------------------------------------------
        # The causal mask prevents token t from attending to tokens t+1, ..., M-1.
        # This enforces left-to-right (autoregressive) attention, which is
        # critical for the decoder-only architecture.
        #
        # generate_square_subsequent_mask returns a float mask with:
        #   - 0.0 on and below the diagonal (allowed attention)
        #   - -inf above the diagonal (masked attention)
        # Shape: [M, M]
        #
        # The mask must be on the same device as x for the transformer to work.
        causal_mask: Tensor = nn.Transformer.generate_square_subsequent_mask(
            seq_len,
            device=x.device,
            dtype=x.dtype,
        )
        # causal_mask: [M, M], float tensor with 0.0 and -inf values

        # ----------------------------------------------------------------
        # 5. Pass through transformer decoder (decoder-only trick).
        # ----------------------------------------------------------------
        # Decoder-only implementation: pass x as BOTH tgt and memory.
        # - tgt: the sequence to decode (with causal mask)
        # - memory: the "encoder output" (same sequence, no mask needed
        #   since we apply causal mask to tgt_mask)
        #
        # Setting memory_mask=causal_mask ensures the cross-attention also
        # respects causality (token t cannot attend to future memory tokens).
        # This is important for maintaining full causal consistency.
        #
        # out: [B, M, d_model=64]
        out: Tensor = self.transformer(
            tgt=x,
            memory=x,
            tgt_mask=causal_mask,
            memory_mask=causal_mask,
            tgt_is_causal=True,
            memory_is_causal=True,
        )
        # out: [B, M, d_model=64]

        # ----------------------------------------------------------------
        # 6. Extract the last token's output for next-step prediction.
        # ----------------------------------------------------------------
        # The last token (index M-1) has attended to all previous tokens
        # via causal attention and encodes the prediction for the next
        # timestep (step M, i.e., the first forecast step).
        # last_out: [B, d_model=64]
        last_out: Tensor = out[:, -1, :]

        # ----------------------------------------------------------------
        # 7. Apply four linear prediction heads.
        # ----------------------------------------------------------------
        # Observation distribution parameters
        obs_mean: Tensor = self.obs_head_mean(last_out)    # [B, obs_dim]

        # Clamp logstd to prevent numerical overflow in exp(logstd).
        # Without clamping, early training can produce extreme values
        # (e.g., logstd = ±100) that cause NaN losses.
        obs_logstd: Tensor = torch.clamp(
            self.obs_head_logstd(last_out),
            _LOGSTD_MIN,
            _LOGSTD_MAX,
        )  # [B, obs_dim]

        # Privileged information distribution parameters
        priv_mean: Tensor = self.priv_head_mean(last_out)   # [B, priv_dim]

        priv_logstd: Tensor = torch.clamp(
            self.priv_head_logstd(last_out),
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

        Implements the outer autoregression loop for the Transformer baseline.
        For each of ``n_steps`` forecast steps:
          1. Run ``forward`` on the current M-step context window.
          2. Sample the next observation via reparameterization.
          3. Slide the context window forward by one step (drop oldest,
             append new prediction).

        **IMPORTANT — GPU Memory Warning:**
        This method should only be called within a ``torch.no_grad()`` context
        during evaluation. Calling it with gradients enabled (during training)
        will cause GPU OOM errors because each ``forward`` call creates a new
        transformer computation graph, and N=8 sequential calls create a graph
        8x deeper than a single forward pass. This is the documented limitation
        from Section 4.3 of the paper.

        Example safe usage:
            with torch.no_grad():
                pred_obs, pred_obs_logstds, pred_priv, pred_priv_logstds = (
                    model.autoregressive_rollout(obs_hist, act_hist, fut_acts, n_steps=8)
                )

        **Why the Transformer accumulates errors differently from RWM:**
        At each step, the Transformer re-processes the entire M-step window
        from scratch (O(M²) attention). The GRU, by contrast, maintains a
        hidden state that is updated incrementally (O(M) per step). When the
        window fills with predictions (rather than real observations), the
        Transformer's global attention can amplify errors