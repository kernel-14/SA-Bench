```python
## models/positional_encoding.py
"""Positional encoding schemes for Pyramidal Flow Matching.

Implements all position encoding required by the paper (Section 3.4, Fig. 3b):

1. Standard 2D sinusoidal encoding for spatial tokens.
2. Spatial pyramid extrapolation: scales position indices by the downsampling
   factor so that lower-resolution stages see full-resolution position values,
   enabling fine-grained detail at higher stages (Yang et al., 2024).
3. Temporal pyramid interpolation: compresses temporal positions for history
   frames so that history tokens are spatially aligned with current tokens.
4. 1D Rotary Position Embedding (RoPE) for the temporal dimension, supporting
   flexible video durations (Su et al., 2024).
5. RoPE application to query/key tensors in attention.

All encodings are computed analytically (no learnable parameters).

Usage:
    from models.positional_encoding import PositionalEncoding

    pos_enc = PositionalEncoding(config)

    # Spatial encoding for current generation tokens at stage k
    spatial_enc = pos_enc.extrapolate_spatial(h, w, stage_id=1, K=3)

    # Temporal RoPE for attention
    rope = pos_enc.temporal_rope(seq_len=16, dim=72)
    q_rotated = pos_enc.apply_rope(q, rope)

    # History temporal positions
    hist_enc = pos_enc.interpolate_history_positions([0, 8, 16], stage_id=1)
"""

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from utils.logging import get_logger

## ---------------------------------------------------------------------------
## Module-level logger
## ---------------------------------------------------------------------------
logger = get_logger(__name__)


class PositionalEncoding(nn.Module):
    """All positional encoding schemes for the Pyramidal Flow Matching model.

    Handles spatial sinusoidal encoding with pyramid extrapolation, temporal
    1D RoPE with pyramid interpolation, and their application to attention
    query/key tensors.

    No learnable parameters — all encodings are deterministic functions of
    position indices and configuration values.

    Attributes:
        hidden_dim: Full embedding dimension (1152 from config).
        num_heads: Number of attention heads (16 from config).
        head_dim: Per-head dimension (72 from config).
        num_stages: Number of pyramid stages K (3 from config).
        downsample_factors: Spatial downsampling factor per stage_id.
            downsample_factors[k] = 2^k, so [1, 2, 4] for K=3.
        max_height: Maximum spatial height from resolution buckets (768).
        max_width: Maximum spatial width from resolution buckets (768).
        max_seq_len: Maximum temporal sequence length (241 from config).
        spatial_mode: "extrapolate" for spatial pyramid (from config).
        temporal_mode: "interpolate" for temporal history (from config).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initializes PositionalEncoding from the project config.

        Reads all required values from configs/default.yaml via the
        omegaconf DictConfig (or plain dict) passed as ``config``.

        Args:
            config: Project configuration dictionary. Expected keys:
                - config.model.hidden_dim (int): 1152
                - config.model.num_heads (int): 16
                - config.model.head_dim (int): 72
                - config.pyramid.num_stages (int): 3
                - config.pyramid.downsample_factors (list[int]): [1, 2, 4]
                - config.model.position_encoding.spatial_pyramid_mode (str)
                - config.model.position_encoding.temporal_history_mode (str)
                - config.data.resolution.buckets (list[list[int]])
                - config.data.frames.stage3_frames (int): 241
        """
        super().__init__()

        # ----------------------------------------------------------------
        # Model architecture parameters
        # ----------------------------------------------------------------
        model_cfg: Dict[str, Any] = config.get("model", {})
        self.hidden_dim: int = int(model_cfg.get("hidden_dim", 1152))
        self.num_heads: int = int(model_cfg.get("num_heads", 16))
        self.head_dim: int = int(model_cfg.get("head_dim", 72))

        # ----------------------------------------------------------------
        # Pyramid parameters
        # ----------------------------------------------------------------
        pyramid_cfg: Dict[str, Any] = config.get("pyramid", {})
        self.num_stages: int = int(pyramid_cfg.get("num_stages", 3))
        self.downsample_factors: List[int] = list(
            pyramid_cfg.get("downsample_factors", [1, 2, 4])
        )

        # Validate that downsample_factors has one entry per stage
        if len(self.downsample_factors) != self.num_stages:
            raise ValueError(
                f"len(downsample_factors)={len(self.downsample_factors)} "
                f"must equal num_stages={self.num_stages}. "
                f"Got downsample_factors={self.downsample_factors}."
            )

        # ----------------------------------------------------------------
        # Position encoding mode flags
        # ----------------------------------------------------------------
        pe_cfg: Dict[str, Any] = model_cfg.get("position_encoding", {})
        self.spatial_mode: str = str(
            pe_cfg.get("spatial_pyramid_mode", "extrapolate")
        )
        self.temporal_mode: str = str(
            pe_cfg.get("temporal_history_mode", "interpolate")
        )

        # ----------------------------------------------------------------
        # Resolution and sequence length limits
        # ----------------------------------------------------------------
        data_cfg: Dict[str, Any] = config.get("data", {})
        resolution_cfg: Dict[str, Any] = data_cfg.get("resolution", {})
        buckets: List[List[int]] = list(
            resolution_cfg.get(
                "buckets",
                [[256, 256], [384, 384], [512, 512], [768, 768]],
            )
        )

        # Derive max_height and max_width from the largest bucket
        if buckets:
            self.max_height: int = max(int(b[0]) for b in buckets)
            self.max_width: int = max(int(b[1]) for b in buckets)
        else:
            self.max_height = 768
            self.max_width = 768

        frames_cfg: Dict[str, Any] = data_cfg.get("frames", {})
        self.max_seq_len: int = int(frames_cfg.get("stage3_frames", 241))

        logger.info(
            "PositionalEncoding initialized: hidden_dim=%d, num_heads=%d, "
            "head_dim=%d, num_stages=%d, downsample_factors=%s, "
            "max_height=%d, max_width=%d, max_seq_len=%d, "
            "spatial_mode=%s, temporal_mode=%s",
            self.hidden_dim,
            self.num_heads,
            self.head_dim,
            self.num_stages,
            self.downsample_factors,
            self.max_height,
            self.max_width,
            self.max_seq_len,
            self.spatial_mode,
            self.temporal_mode,
        )

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _sinusoidal_1d(
        self,
        positions: Tensor,
        dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Computes 1D sinusoidal encoding for a sequence of positions.

        Uses the standard formula from Vaswani et al. (2017):
            PE(pos, 2i)   = sin(pos / 10000^(2i / dim))
            PE(pos, 2i+1) = cos(pos / 10000^(2i / dim))

        Args:
            positions: 1D tensor of position indices, shape [N]. May be
                non-integer (e.g., interpolated positions).
            dim: Embedding dimension for this encoding. Must be even.
            device: Target device for the output tensor.
            dtype: Output dtype. Defaults to float32 for precision.

        Returns:
            Tensor of shape [N, dim] containing sinusoidal encodings.

        Raises:
            ValueError: If ``dim`` is odd (sinusoidal encoding requires
                an even dimension for sin/cos pairing).
        """
        if dim % 2 != 0:
            raise ValueError(
                f"Sinusoidal encoding requires an even dimension, got dim={dim}."
            )

        half_dim: int = dim // 2

        # Inverse frequencies: theta_i = 1 / 10000^(2i / dim)
        # Shape: [half_dim]
        inv_freq: Tensor = 1.0 / (
            10000.0
            ** (
                torch.arange(0, half_dim, dtype=torch.float32, device=device)
                / half_dim
            )
        )

        # Outer product: positions [N] x inv_freq [half_dim] -> [N, half_dim]
        positions_float: Tensor = positions.to(dtype=torch.float32, device=device)
        freqs: Tensor = torch.outer(positions_float, inv_freq)  # [N, half_dim]

        # Interleave sin and cos: [N, dim]
        encoding: Tensor = torch.cat(
            [torch.sin(freqs), torch.cos(freqs)], dim=-1
        )  # [N, dim]

        return encoding.to(dtype=dtype)

    def _make_2d_encoding(
        self,
        row_positions: Tensor,
        col_positions: Tensor,
        h: int,
        w: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Constructs 2D sinusoidal encoding from row and column positions.

        Encodes rows and columns independently using half the hidden_dim
        each, then concatenates to form the full [h*w, hidden_dim] encoding.

        Args:
            row_positions: 1D tensor of row position values, shape [h].
            col_positions: 1D tensor of column position values, shape [w].
            h: Number of rows (height of the spatial grid).
            w: Number of columns (width of the spatial grid).
            device: Target device.
            dtype: Output dtype.

        Returns:
            Tensor of shape [h*w, hidden_dim] containing 2D sinusoidal
            position encodings. Rows vary along the first axis, columns
            along the second.
        """
        half_dim: int = self.hidden_dim // 2

        # Row encoding: [h, half_dim]
        row_enc: Tensor = self._sinusoidal_1d(
            row_positions, dim=half_dim, device=device, dtype=dtype
        )

        # Column encoding: [w, half_dim]
        col_enc: Tensor = self._sinusoidal_1d(
            col_positions, dim=half_dim, device=device, dtype=dtype
        )

        # Broadcast to [h, w, half_dim] each, then flatten to [h*w, hidden_dim]
        # row_enc: [h, 1, half_dim] -> [h, w, half_dim]
        # col_enc: [1, w, half_dim] -> [h, w, half_dim]
        row_enc_2d: Tensor = row_enc.unsqueeze(1).expand(h, w, half_dim)
        col_enc_2d: Tensor = col_enc.unsqueeze(0).expand(h, w, half_dim)

        # Concatenate along feature dim: [h, w, hidden_dim]
        encoding_2d: Tensor = torch.cat(
            [row_enc_2d, col_enc_2d], dim=-1
        )  # [h, w, hidden_dim]

        # Flatten spatial dims: [h*w, hidden_dim]
        return encoding_2d.reshape(h * w, self.hidden_dim)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def sinusoidal_2d(
        self,
        h: int,
        w: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Computes standard 2D sinusoidal position encoding.

        Encodes a spatial grid of height ``h`` and width ``w`` using the
        standard sinusoidal formula from Vaswani et al. (2017). Row and
        column indices are encoded independently using half the hidden_dim
        each, then concatenated.

        This is the base encoding used at full resolution (stage_id=0).
        For pyramid stages, use ``extrapolate_spatial`` instead.

        Args:
            h: Height of the spatial grid (number of rows).
            w: Width of the spatial grid (number of columns).
            device: Target device for the output tensor. Defaults to CPU.
            dtype: Output dtype. Defaults to float32.

        Returns:
            Tensor of shape [h*w, hidden_dim] containing 2D sinusoidal
            position encodings. The flattening order is row-major
            (row 0 col 0, row 0 col 1, ..., row h-1 col w-1).

        Example:
            >>> pos_enc = PositionalEncoding(config)
            >>> enc = pos_enc.sinusoidal_2d(h=96, w=96)
            >>> enc.shape
            torch.Size([9216, 1152])
        """
        if device is None:
            device = torch.device("cpu")

        # Standard integer position indices
        row_positions: Tensor = torch.arange(
            0, h, dtype=torch.float32, device=device
        )
        col_positions: Tensor = torch.arange(
            0, w, dtype=torch.float32, device=device
        )

        return self._make_2d_encoding(
            row_positions, col_positions, h, w, device=device, dtype=dtype
        )

    def extrapolate_spatial(
        self,
        h: int,
        w: int,
        stage_id: int,
        K: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Computes spatially extrapolated 2D sinusoidal encoding for a pyramid stage.

        Implements the spatial pyramid extrapolation described in Section 3.4
        and Fig. 3b of the paper: "we extrapolate position encoding in the
        spatial pyramid for better fine-grained detail (Yang et al., 2024)."

        At pyramid stage ``stage_id``, the latent has spatial resolution
        ``H / 2^stage_id × W / 2^stage_id``. Instead of encoding positions
        [0, 1, ..., h-1], we encode positions
        [0, factor, 2*factor, ..., (h-1)*factor] where
        ``factor = downsample_factors[stage_id] = 2^stage_id``.

        This ensures the model sees position values in the same range as
        full-resolution training, even at compressed resolutions — analogous
        to NTK-aware extrapolation in RoPE.

        At stage_id=0 (full resolution), factor=1, so this is identical to
        ``sinusoidal_2d``.

        Args:
            h: Height of the spatial grid at the current pyramid stage.
                This is the compressed height (e.g., H // 2^stage_id).
            w: Width of the spatial grid at the current pyramid stage.
            stage_id: Current pyramid stage index. 0 = full resolution
                (final stage), K-1 = lowest resolution (first stage).
                Must be in [0, K-1].
            K: Total number of pyramid stages (3 from config).
            device: Target device. Defaults to CPU.
            dtype: Output dtype. Defaults to float32.

        Returns:
            Tensor of shape [h*w, hidden_dim] containing extrapolated 2D
            sinusoidal position encodings.

        Raises:
            ValueError: If ``stage_id`` is outside [0, K-1].

        Example:
            >>> pos_enc = PositionalEncoding(config)
            >>> # At stage 2 (1/4 resolution), h=24, w=24, factor=4
            >>> enc = pos_enc.extrapolate_spatial(h=24, w=24, stage_id=2, K=3)
            >>> enc.shape
            torch.Size([576, 1152])
            >>> # Positions are [0, 4, 8, ..., 92] — same range as full-res
        """
        if stage_id < 0 or stage_id >= K:
            raise ValueError(
                f"stage_id={stage_id} is out of range [0, {K-1}]. "
                f"Must be in [0, K-1] where K={K}."
            )

        if device is None:
            device = torch.device("cpu")

        # Downsampling factor at this stage: 2^stage_id
        # downsample_factors[stage_id] gives the factor directly from config
        if stage_id < len(self.downsample_factors):
            factor: int = self.downsample_factors[stage_id]
        else:
            # Fallback: compute from stage_id if config list is shorter
            factor = 2 ** stage_id
            logger.warning(
                "stage_id=%d exceeds len(downsample_factors)=%d. "
                "Using computed factor=2^%d=%d.",
                stage_id,
                len(self.downsample_factors),
                stage_id,
                factor,
            )

        # Extrapolated position indices: stride through full-resolution grid
        # Row positions: [0, factor, 2*factor, ..., (h-1)*factor]
        row_positions: Tensor = torch.arange(
            0, h, dtype=torch.float32, device=device
        ) * float(factor)

        # Column positions: [0, factor, 2*factor, ..., (w-1)*factor]
        col_positions: Tensor = torch.arange(
            0, w, dtype=torch.float32, device=device
        ) * float(factor)

        return self._make_2d_encoding(
            row_positions, col_positions, h, w, device=device, dtype=dtype
        )

    def interpolate_history_positions(
        self,
        history_frames: List[int],
        stage_id: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Computes interpolated temporal position encodings for history frames.

        Implements the temporal pyramid interpolation described in Section 3.3
        and Fig. 3b: "interpolating it in the temporal pyramid input to
        spatially align the history conditions."

        History frames at pyramid stage ``stage_id`` are compressed at
        different factors depending on their distance from the current frame:
        - Most distant frames: compressed at factor ``2^(stage_id+1)``
        - Most recent frame: compressed at factor ``2^stage_id``

        The temporal position of each history frame is interpolated
        (divided by its compression factor) so that the history tokens
        are temporally aligned with the current generation tokens.

        Args:
            history_frames: List of original temporal frame indices for the
                history condition. Each entry is the original frame index
                in the full video (e.g., [0, 8, 16] for frames at 8-frame
                intervals). The last entry is the most recent history frame.
            stage_id: Current pyramid stage index. 0 = full resolution,
                K-1 = lowest resolution. Used to determine compression
                factors for history frames.
            device: Target device. Defaults to CPU.
            dtype: Output dtype. Defaults to float32.

        Returns:
            Tensor of shape [len(history_frames), hidden_dim] containing
            interpolated temporal position encodings for each history frame.
            Each row corresponds to one history frame's temporal encoding.

        Example:
            >>> pos_enc = PositionalEncoding(config)
            >>> # History frames at original indices [0, 8, 16]
            >>> # At stage_id=1: most distant at factor 4, most recent at factor 2
            >>> enc = pos_enc.interpolate_history_positions([0, 8, 16], stage_id=1)
            >>> enc.shape
            torch.Size([3, 1152])
        """
        if device is None:
            device = torch.device("cpu")

        num_history: int = len(history_frames)
        if num_history == 0:
            return torch.zeros(
                0, self.hidden_dim, dtype=dtype, device=device
            )

        # Determine compression factors for each history frame.
        # Per Section 3.3: the history condition at stage k uses:
        #   - Down(x, 2^(k+1)) for older frames (all but the most recent)
        #   - Down(x, 2^k) for the most recent frame
        # The temporal compression factor matches the spatial compression.
        #
        # Most recent frame (last in list): factor = 2^stage_id
        # All other frames: factor = 2^(stage_id + 1)
        recent_factor: float = float(2 ** stage_id)
        distant_factor: float = float(2 ** (stage_id + 1))

        # Build interpolated position values for each history frame
        interpolated_positions: List[float] = []
        for i, frame_idx in enumerate(history_frames):
            is_most_recent: bool = (i == num_history - 1)
            compression_factor: float = (
                recent_factor if is_most_recent else distant_factor
            )
            # Interpolated position: original index / compression factor
            # This compresses the temporal position to align with current tokens
            interpolated_pos: float = float(frame_idx) / compression_factor
            interpolated_positions.append(interpolated_pos)

        # Convert to tensor for batch encoding
        positions_tensor: Tensor = torch.tensor(
            interpolated_positions, dtype=torch.float32, device=device
        )  # [num_history]

        # Encode using 1D sinusoidal encoding over the full hidden_dim
        # (temporal position encoding for history uses the full embedding dim)
        encoding: Tensor = self._sinusoidal_1d(
            positions_tensor,
            dim=self.hidden_dim,
            device=device,
            dtype=dtype,
        )  # [num_history, hidden_dim]

        return encoding

    def temporal_rope(
        self,
        seq_len: int,
        dim: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Computes 1D Rotary Position Embedding (RoPE) for temporal positions.

        Implements the 1D RoPE from Su et al. (2024) used for the temporal
        dimension of MM-DiT (Section 4.1: "1D Rotary Position Embedding
        (RoPE) is added to support flexible training with different video
        durations").

        The output contains concatenated cosine and sine values for each
        position, which are used by ``apply_rope`` to rotate query/key
        tensors in attention.

        RoPE construction:
            theta_i = 1 / (10000^(2i / dim))  for i = 0, ..., dim//2 - 1
            freqs[t, i] = t * theta_i
            rope[t] = cat([cos(freqs[t]), sin(freqs[t])])

        Args:
            seq_len: Number of temporal positions to encode. Typically the
                number of latent frames in the current sequence.
            dim: Dimension of the RoPE encoding. Should equal ``head_dim``
                (72 from config) since RoPE is applied per attention head.
                Must be even.
            device: Target device. Defaults to CPU.
            dtype: Output dtype. Defaults to float32 for precision in
                frequency computation.

        Returns:
            Tensor of shape [seq_len, dim] where the first ``dim//2``
            values are cosines and the last ``dim//2`` values are sines.
            This is consumed by ``apply_rope``.

        Raises:
            ValueError: If ``dim`` is odd.

        Example:
            >>> pos_enc = PositionalEncoding(config)
            >>> rope = pos_enc.temporal_rope(seq_len=16, dim=72)
            >>> rope.shape
            torch.Size([16, 72])
        """
        if dim % 2 != 0:
            raise ValueError(
                f"RoPE requires an even dimension, got dim={dim}. "
                "The dimension must be even for sin/cos pairing."
            )

        if device is None:
            device = torch.device("cpu")

        half_dim: int = dim // 2

        # Inverse frequencies: theta_i = 1 / 10000^(2i / dim)
        # Shape: [half_dim]
        inv_freq: Tensor = 1.0 / (
            10000.0
            ** (
                torch.arange(0, half_dim, dtype=torch.float32, device=device)
                / half_dim
            )
        )

        # Position indices: [0, 1, ..., seq_len-1]
        # Shape: [seq_len]
        positions: Tensor = torch.arange(
            0, seq_len, dtype=torch.float32, device=device
        )

        # Outer product: [seq_len, half_dim]
        freqs: Tensor = torch.outer(positions, inv_freq)

        # Concatenate cos and sin: [seq_len, dim]
        rope: Tensor = torch.cat(
            [torch.cos(freqs), torch.sin(freqs)], dim=-1
        )

        return rope.to(dtype=dtype)

    def apply_rope(
        self,
        x: Tensor,
        rope: Tensor,
    ) -> Tensor:
        """Applies Rotary Position Embedding to query or key tensors.

        Implements the RoPE rotation formula from Su et al. (2024):
            x_rotated = cat([
                x1 * cos - x2 * sin,
                x2 * cos + x1 * sin
            ], dim=-1)
        where x1 = x[..., :dim//2] and x2 = x[..., dim//2:].

        The ``rope`` tensor (from ``temporal_rope``) is broadcast over the
        batch and head dimensions automatically.

        This method is applied to the temporal component of Q/K vectors
        inside each MM-DiT attention block. The spatial component uses
        additive sinusoidal encoding (handled in ``extrapolate_spatial``).

        Args:
            x: Query or key tensor of shape [B, num_heads, seq_len, head_dim].
                The last dimension must match the dimension of ``rope``.
            rope: RoPE tensor of shape [seq_len, head_dim] from
                ``temporal_rope``. The seq_len must match x.shape[2].

        Returns:
            Tensor of the same shape as ``x`` with rotary position
            embedding applied. The dtype matches the input ``x``.

        Raises:
            ValueError: If the last dimension of ``x`` does not match the
                last dimension of ``rope``, or if the sequence lengths
                do not match.

        Example:
            >>> pos_enc = PositionalEncoding(config)
            >>> B, H, L, D = 2, 16, 16, 72
            >>> q = torch.randn(B, H, L, D)
            >>> rope = pos_enc.temporal_rope(seq_len=L, dim=D)
            >>> q_rotated = pos_enc.apply_rope(q, rope)
            >>> q_rotated.shape
            torch.Size([2, 16, 16, 72])
        """
        # Validate shapes
        if x.shape[-1] != rope.shape[-1]:
            raise ValueError(
                f"Last dimension of x ({x.shape[-1]}) must match "
                f"last dimension of rope ({rope.shape[-1]}). "
                f"x.shape={tuple(x.shape)}, rope.shape={tuple(rope.shape)}."
            )

        seq_len_x: int = x.shape[2]
        seq_len_rope: int = rope.shape[0]
        if seq_len_x != seq_len_rope:
            raise ValueError(
                f"Sequence length of x ({seq_len_x}) must match "
                f"sequence length of rope ({seq_len_rope}). "
                f"x.shape={tuple(x.shape)}, rope.shape={tuple(rope.shape)}."
            )

        dim: int = x.shape[-1]
        half_dim: int = dim // 2

        # Split x into two halves along the last dimension
        x1: Tensor = x[..., :half_dim]   # [B, num_heads, seq_len, half_dim]
        x2: Tensor = x[..., half_dim:]   # [B, num_heads, seq_len, half_dim]

        # Split rope into cos and sin components
        # rope: [seq_len, dim] -> cos: [seq_len, half_dim], sin: [seq_len, half_dim]
        cos_vals: Tensor = rope[:, :half_dim]   # [seq_len, half_dim]
        sin_vals: Tensor = rope[:, half_dim:]   # [seq_len, half_dim]

        # Broadcast rope over batch and head dimensions:
        # [seq_len, half_dim] -> [1, 1, seq_len, half_dim]
        cos_vals = cos_vals.unsqueeze(0).unsqueeze(0)
        sin_vals = sin_vals.unsqueeze(0).unsqueeze(0)

        # Cast rope to match x dtype (x may be bfloat16 during training)
        cos_vals = cos_vals.to(dtype=x.dtype, device=x.device)
        sin_vals = sin_vals.to(dtype=x.dtype, device=x.device)

        # Apply rotation:
        # x_rotated = cat([x1*cos - x2*sin, x2*cos + x1*sin], dim=-1)
        x_rotated: Tensor = torch.cat(
            [
                x1 * cos_vals - x2 * sin_vals,
                x2 * cos_vals + x1 * sin_vals,
            ],
            dim=-1,
        )  # [B, num_heads, seq_len, dim]

        return x_rotated

    def forward(
        self,
        h: int,
        w: int,
        stage_id: int,
        seq_len: int,
        history_frames: Optional[List[int]] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> Dict[str, Tensor]:
        """Convenience method computing all encodings for a single forward pass.

        Computes spatial encoding for current tokens, temporal RoPE for
        attention, and optionally history temporal encodings. This is the
        primary entry point called by MMDiT.forward().

        Args:
            h: Spatial height of the current latent at this pyramid stage.
            w: Spatial width of the current latent at this pyramid stage.
            stage_id: Current pyramid stage index (0=full res, K-1=lowest).
            seq_len: Temporal sequence length (number of latent frames).
            history_frames: Optional list of original frame indices for
                history conditioning. If None, history encoding is skipped.
            device: Target device. Defaults to CPU.
            dtype: Output dtype for all encodings.

        Returns:
            Dictionary with keys:
                - ``"spatial"``: Tensor [h*w, hidden_dim] — spatial encoding
                  for current tokens (extrapolated for pyramid stage).