## utils.py
"""Shared utility functions for Ca2-VDM.

This module provides foundational utilities used across the entire Ca2-VDM
codebase: seeding, logging, video I/O, model introspection, FLOPs counting,
metric tracking, LR scheduling, and timing.

This file has zero internal project dependencies and must be imported before
any other project module.

Paper: Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal
Generation and Cache Sharing.
"""

import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def get_logger(name: str) -> logging.Logger:
    """Return a consistently formatted logger for the given module name.

    Idempotent: calling this multiple times with the same name returns the
    same logger without adding duplicate handlers.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A configured :class:`logging.Logger` instance writing to stdout at
        INFO level with timestamp, name, and level prefix.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls (e.g., in notebooks).
    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    # Prevent propagation to the root logger to avoid duplicate output when
    # the root logger also has handlers configured.
    logger.propagate = False

    return logger


# Module-level logger for utils itself.
_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for full reproducibility.

    Covers Python's ``random``, NumPy, PyTorch CPU, and all CUDA devices.
    Also disables cuDNN non-determinism.

    Args:
        seed: Integer seed value. Defaults to 42 (matches ``config.yaml``
              ``inference.seed``).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all is a no-op when CUDA is unavailable.
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    _logger.info("Global seed set to %d.", seed)


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------


def save_video(
    frames: torch.Tensor,
    path: str,
    fps: int = 8,
) -> None:
    """Save a video tensor to disk as an MP4 file using imageio-ffmpeg.

    Args:
        frames: Float tensor of shape ``(T, C, H, W)`` with values in
                ``[-1, 1]``. ``C`` must be 3 (RGB).
        path: Output file path. Parent directories are created automatically.
              Should end with ``.mp4``.
        fps: Frames per second for the output video. Defaults to 8, which is
             suitable for 256×256 preview videos.

    Raises:
        ValueError: If ``frames`` does not have shape ``(T, 3, H, W)``.
        RuntimeError: If the imageio ffmpeg plugin is unavailable.
    """
    if frames.ndim != 4:
        raise ValueError(
            f"frames must be a 4-D tensor (T, C, H, W), got shape {frames.shape}."
        )
    if frames.shape[1] != 3:
        raise ValueError(
            f"frames must have C=3 (RGB), got C={frames.shape[1]}."
        )

    # Ensure we work on CPU with no gradient tracking.
    frames_cpu: torch.Tensor = frames.detach().cpu()

    # Convert [-1, 1] float → [0, 255] uint8.
    frames_uint8: np.ndarray = (
        ((frames_cpu + 1.0) / 2.0 * 255.0)
        .clamp(0.0, 255.0)
        .byte()
        .permute(0, 2, 3, 1)  # (T, C, H, W) → (T, H, W, C)
        .numpy()
    )

    # Create parent directory if needed.
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    try:
        import imageio  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "imageio is required for save_video. "
            "Install it with: pip install imageio imageio-ffmpeg"
        ) from exc

    # Try to use the ffmpeg plugin; fall back to saving individual PNG frames
    # if ffmpeg is not available.
    try:
        writer = imageio.get_writer(
            path,
            fps=fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
            macro_block_size=1,  # Allows arbitrary resolutions (e.g., 256×256).
        )
        try:
            for frame in frames_uint8:
                writer.append_data(frame)
        finally:
            writer.close()
        _logger.info("Saved video (%d frames, %d fps) to '%s'.", len(frames_uint8), fps, path)
    except Exception as ffmpeg_exc:  # noqa: BLE001
        _logger.warning(
            "ffmpeg writer failed (%s). Falling back to saving individual PNG frames.",
            ffmpeg_exc,
        )
        png_dir = Path(path).with_suffix("")
        png_dir.mkdir(parents=True, exist_ok=True)
        for idx, frame in enumerate(frames_uint8):
            frame_path = str(png_dir / f"frame_{idx:05d}.png")
            imageio.imwrite(frame_path, frame)
        _logger.info(
            "Saved %d PNG frames to '%s'.", len(frames_uint8), str(png_dir)
        )


def load_video(
    path: str,
    num_frames: int,
    resolution: int,
) -> torch.Tensor:
    """Load a fixed number of frames from a video file using decord.

    Frames are sampled uniformly across the full video duration. If the video
    has fewer frames than requested, the last frame is repeated to pad.

    Args:
        path: Path to the video file (any format supported by decord).
        num_frames: Number of frames to return (``T``).
        resolution: Not applied here — resizing is handled by
                    :class:`data.video_transforms.VideoTransforms`. Kept in
                    the signature for API consistency and future use.

    Returns:
        Uint8 tensor of shape ``(T, H, W, C)`` where ``T == num_frames``,
        ``H`` and ``W`` are the native video dimensions, and ``C == 3``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        RuntimeError: If decord cannot open the video file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video file not found: '{path}'.")

    try:
        import decord  # type: ignore[import]
        decord.bridge.set_bridge("torch")
    except ImportError as exc:
        raise RuntimeError(
            "decord is required for load_video. "
            "Install it with: pip install decord"
        ) from exc

    try:
        # Use CPU context for DataLoader worker safety (GPU context has issues
        # with forked processes in multi-worker DataLoaders).
        vr = decord.VideoReader(path, ctx=decord.cpu(0))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"decord could not open video '{path}': {exc}") from exc

    total_frames: int = len(vr)

    if total_frames >= num_frames:
        # Uniformly spaced indices across the full video.
        indices: np.ndarray = np.linspace(
            0, total_frames - 1, num_frames, dtype=int
        )
    else:
        # Pad by repeating the last frame.
        available_indices = np.arange(total_frames, dtype=int)
        pad_count = num_frames - total_frames
        padding = np.full(pad_count, total_frames - 1, dtype=int)
        indices = np.concatenate([available_indices, padding], axis=0)

    # get_batch returns a torch.Tensor of shape (T, H, W, C) when bridge='torch'.
    frames: torch.Tensor = vr.get_batch(indices.tolist())

    # Ensure uint8 output.
    if frames.dtype != torch.uint8:
        frames = frames.byte()

    return frames  # (T, H, W, C)


# ---------------------------------------------------------------------------
# Model introspection
# ---------------------------------------------------------------------------


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.

    Args:
        model: Any :class:`torch.nn.Module`.

    Returns:
        Total number of parameters with ``requires_grad=True``.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_flops(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    **kwargs: Any,
) -> Dict[str, Any]:
    """Count multiply-add operations (MACs) for a model forward pass.

    Uses fvcore's :class:`~fvcore.nn.FlopCountAnalysis`. Returns MACs
    (multiply-accumulate operations); by convention 1 MAC ≈ 2 FLOPs, but
    the paper reports MACs directly as "FLOPs" (Figure 8).

    Matches the paper's efficiency evaluation setup:
    - 56 frames (7 AR steps) with 1 denoising step
      (``config.yaml`` ``evaluation.efficiency.flop_count_frames: 56``).

    Args:
        model: The model to profile.
        input_shape: Shape of the primary input tensor (used to construct a
                     dummy input via ``torch.zeros``).
        **kwargs: Additional keyword arguments passed to the model's
                  ``forward`` method alongside the dummy input. Use this to
                  supply ``t_vec``, ``text_emb``, etc.

    Returns:
        A dict with keys:
        - ``"total"`` (int): Total MACs, or ``-1`` if counting failed.
        - ``"by_module"`` (dict): Per-module MAC breakdown as returned by
          :meth:`~fvcore.nn.FlopCountAnalysis.by_module`.
        - ``"by_operator"`` (dict): Per-operator MAC breakdown.
    """
    try:
        from fvcore.nn import FlopCountAnalysis  # type: ignore[import]
        from fvcore.nn.jit_handles import generic_activation_jit  # noqa: F401
    except ImportError:
        _logger.warning(
            "fvcore is not installed. compute_flops returning -1. "
            "Install with: pip install fvcore"
        )
        return {"total": -1, "by_module": {}, "by_operator": {}}

    dummy_input = torch.zeros(input_shape)

    try:
        with torch.no_grad():
            flop_counter = FlopCountAnalysis(model, (dummy_input,))
            # Suppress per-operator warnings for unsupported ops (e.g., custom
            # attention implementations with non-standard control flow).
            flop_counter.unsupported_ops_warnings(False)
            flop_counter.uncalled_modules_warnings(False)

            total_macs: int = flop_counter.total()
            by_module: Dict[str, int] = dict(flop_counter.by_module())
            by_operator: Dict[str, int] = dict(flop_counter.by_operator())

        _logger.info(
            "FLOPs (MACs) for input shape %s: %s (%.2f G).",
            input_shape,
            total_macs,
            total_macs / 1e9,
        )
        return {
            "total": total_macs,
            "by_module": by_module,
            "by_operator": by_operator,
        }
    except Exception as exc:  # noqa: BLE001
        _logger.warning(
            "FlopCountAnalysis failed: %s. Returning -1.", exc
        )
        return {"total": -1, "by_module": {}, "by_operator": {}}


# ---------------------------------------------------------------------------
# Metric tracking
# ---------------------------------------------------------------------------


class AverageMeter:
    """Track the running average of a scalar metric.

    Supports weighted updates so that batch-averaged losses can be correctly
    accumulated by passing ``n=batch_size``.

    Attributes:
        name: Human-readable name for display in log messages.
        val: Most recently recorded value.
        avg: Running weighted average.
        sum: Cumulative weighted sum.
        count: Total weight accumulated (sum of all ``n`` values).

    Example::

        meter = AverageMeter("loss")
        for batch in dataloader:
            loss = compute_loss(batch)
            meter.update(loss.item(), n=len(batch))
        print(meter)  # "loss: 0.1234 (avg: 0.1456)"
    """

    def __init__(self, name: str = "") -> None:
        """Initialise the meter in a zeroed state.

        Args:
            name: Optional display name used in :meth:`__str__`.
        """
        self.name: str = name
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def reset(self) -> None:
        """Reset all accumulators to zero."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """Record a new observation.

        Args:
            val: The scalar value to record. If ``val`` is already a
                 batch-averaged quantity, pass ``n=batch_size`` so the
                 running average is weighted correctly.
            n: Weight for this observation (typically the batch size).
               Defaults to 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __str__(self) -> str:
        """Return a human-readable summary string.

        Returns:
            A string of the form ``"<name>: <val:.4f> (avg: <avg:.4f>)"``.
        """
        prefix = f"{self.name}: " if self.name else ""
        return f"{prefix}{self.val:.4f} (avg: {self.avg:.4f})"

    def __repr__(self) -> str:
        return (
            f"AverageMeter(name={self.name!r}, val={self.val:.4f}, "
            f"avg={self.avg:.4f}, count={self.count})"
        )


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


class Timer:
    """Context manager for measuring wall-clock elapsed time.

    Used by :class:`evaluation.evaluator.Evaluator` for efficiency benchmarks
    matching Table 5 and Figure 6 of the paper.

    Attributes:
        elapsed: Elapsed time in seconds. Set after ``__exit__`` is called.

    Example::

        with Timer() as t:
            generate_video(...)
        print(f"Generation took {t.elapsed:.2f}s")
    """

    def __init__(self) -> None:
        """Initialise the timer with elapsed time set to 0."""
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        """Record the start time and return self for use in ``with`` blocks."""
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """Compute elapsed time on exit, regardless of whether an exception occurred."""
        self.elapsed = time.perf_counter() - self._start

    def __str__(self) -> str:
        return f"Timer(elapsed={self.elapsed:.4f}s)"


# ---------------------------------------------------------------------------
# Learning rate scheduling
# ---------------------------------------------------------------------------


def cosine_lr_schedule(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
) -> float:
    """Apply a cosine learning rate schedule with linear warmup.

    The schedule has two phases:
    1. **Linear warmup** (steps 0 … warmup_steps − 1): LR increases linearly
       from 0 to ``base_lr``.
    2. **Cosine decay** (steps warmup_steps … total_steps): LR decays from
       ``base_lr`` to 0 following a cosine curve.

    The paper uses a fixed LR of 2e-5 (``config.yaml`` ``learning_rate``).
    This function is provided as an optional enhancement; pass
    ``warmup_steps=0`` to disable warmup and use pure cosine decay, or call
    it with ``total_steps=num_steps`` and ``warmup_steps=0`` to approximate
    a constant LR (cosine decay from 2e-5 to ~0 over training).

    Args:
        optimizer: The AdamW optimizer whose param-group LRs will be updated
                   in-place.
        step: Current training step (0-indexed).
        total_steps: Total number of training steps.
        warmup_steps: Number of linear warmup steps. Set to 0 to skip warmup.
        base_lr: Peak learning rate reached after warmup (e.g., 2e-5).

    Returns:
        The learning rate value applied at this step (for logging).
    """
    if warmup_steps > 0 and step < warmup_steps:
        # Linear warmup: lr = base_lr * (step / warmup_steps)
        lr: float = base_lr * float(step) / float(max(1, warmup_steps))
    else:
        # Cosine decay from base_lr to 0.
        decay_steps = max(1, total_steps - warmup_steps)
        progress = float(step - warmup_steps) / float(decay_steps)
        # Clamp progress to [0, 1] so LR doesn't go negative after total_steps.
        progress = min(progress, 1.0)
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr
