## utils/normalizer.py
"""Normalizer utility for Prioritized Generative Replay (PGR).

Provides per-feature standardization of transition tuples (s, a, s', r)
for stable diffusion model training. Fit once on the first inner loop call,
then frozen for the remainder of training.
"""

from typing import Optional

import numpy as np
import torch


class Normalizer:
    """Per-feature standardization for transition tuples used by ConditionalDiffusion.

    Computes mean and standard deviation over a fitting dataset and applies
    z-score normalization. Designed to be fit exactly once — subsequent calls
    to ``fit()`` are silently ignored to preserve consistency between the
    diffusion model's training and inference spaces.

    The normalizer is a plain Python class (not ``nn.Module``) because it
    holds no learnable parameters. Mean and std tensors are stored directly
    on ``self.device`` to avoid device mismatches during forward passes.

    Typical usage inside ``ConditionalDiffusion``::

        # First inner loop call only:
        normalizer.fit(concatenated_transitions)   # shape (N, input_dim)

        # Every train_step:
        x0_norm = normalizer.normalize(x0)         # shape (B, input_dim)

        # After reverse diffusion:
        x0_hat = normalizer.denormalize(x0_norm_hat)

    Attributes:
        device: Target device string (e.g. ``"cuda"`` or ``"cpu"``).
        mean: Per-feature mean tensor of shape ``(1, input_dim)`` after fitting.
        std: Per-feature standard deviation tensor of shape ``(1, input_dim)``
            after fitting, clamped to a minimum of ``1e-8``.
        fitted: Whether ``fit()`` has been called successfully.
    """

    def __init__(self, device: str = "cuda") -> None:
        """Initialises the normalizer without computing any statistics.

        Args:
            device: PyTorch device string on which mean and std tensors will
                be stored.  Must match the device used by ``ConditionalDiffusion``.
        """
        self.device: str = device
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None
        self.fitted: bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, data: torch.Tensor) -> None:
        """Computes and stores per-feature mean and std from a dataset.

        This method is idempotent after the first successful call — if
        ``self.fitted`` is already ``True``, the method returns immediately
        without modifying the stored statistics.  This enforces the
        "fit once, then freeze" semantics required for stable diffusion
        training across multiple inner loop calls.

        Args:
            data: Float tensor of shape ``(N, input_dim)`` containing
                concatenated transition components ``(s, a, s', r)``.
                Must have at least one row.  NaN/Inf values should be
                cleaned by the caller before fitting.

        Raises:
            ValueError: If ``data`` has fewer than 2 dimensions or is empty.
        """
        # Guard: silently skip refitting after the first successful call.
        if self.fitted:
            return

        if data.dim() < 2:
            raise ValueError(
                f"Expected data with at least 2 dimensions, got shape {data.shape}."
            )
        if data.shape[0] == 0:
            raise ValueError("Cannot fit normalizer on an empty dataset.")

        # Ensure float32 on the target device.
        data_f: torch.Tensor = data.to(dtype=torch.float32, device=self.device)

        # Per-feature statistics — keepdim=True gives shape (1, input_dim)
        # for broadcasting against (B, input_dim) inputs in normalize/denormalize.
        mean: torch.Tensor = data_f.mean(dim=0, keepdim=True)

        # Use unbiased=False (population std) for consistency; with large N
        # the difference is negligible, but it avoids NaN for N=1.
        std: torch.Tensor = data_f.std(dim=0, keepdim=True, unbiased=False)

        # Clamp std to prevent division by zero for constant features
        # (e.g. reward = 0 throughout early training, or bounded action dims).
        std = std.clamp(min=1e-8)

        self.mean = mean
        self.std = std
        self.fitted = True

    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        """Applies z-score normalization using the fitted statistics.

        Computes ``(data - mean) / (std + 1e-8)`` with per-feature
        broadcasting.  Returns a new tensor; the input is not modified.

        Args:
            data: Float tensor of shape ``(B, input_dim)`` to normalize.
                Must be on the same device as the normalizer.

        Returns:
            Normalized tensor of the same shape and device as ``data``.

        Raises:
            RuntimeError: If ``fit()`` has not been called yet.
        """
        self._assert_fitted("normalize")

        data_f: torch.Tensor = data.to(dtype=torch.float32)

        # self.mean and self.std are (1, input_dim) — broadcast over batch dim.
        return (data_f - self.mean) / (self.std + 1e-8)  # type: ignore[operator]

    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        """Inverts z-score normalization to recover the original scale.

        Computes ``data * (std + 1e-8) + mean``, the exact inverse of
        ``normalize()``.  Returns a new tensor; the input is not modified.

        Args:
            data: Normalized float tensor of shape ``(B, input_dim)``.
                Must be on the same device as the normalizer.

        Returns:
            Denormalized tensor of the same shape and device as ``data``.

        Raises:
            RuntimeError: If ``fit()`` has not been called yet.
        """
        self._assert_fitted("denormalize")

        data_f: torch.Tensor = data.to(dtype=torch.float32)

        # Exact inverse of normalize: x_orig = x_norm * (std + eps) + mean.
        return data_f * (self.std + 1e-8) + self.mean  # type: ignore[operator]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _assert_fitted(self, method_name: str) -> None:
        """Raises RuntimeError if the normalizer has not been fitted yet.

        Args:
            method_name: Name of the calling method, used in the error message.

        Raises:
            RuntimeError: If ``self.fitted`` is ``False``.
        """
        if not self.fitted:
            raise RuntimeError(
                f"Normalizer.{method_name}() called before Normalizer.fit(). "
                "Call fit() with a representative dataset before normalizing data."
            )

    # ── Serialization helpers (used by ConditionalDiffusion.save/load) ────────

    def state_dict(self) -> dict:
        """Returns a serializable state dictionary for checkpointing.

        Returns:
            Dictionary containing ``mean``, ``std``, ``fitted``, and
            ``device`` fields suitable for ``torch.save``.
        """
        return {
            "mean": self.mean.cpu() if self.mean is not None else None,
            "std": self.std.cpu() if self.std is not None else None,
            "fitted": self.fitted,
            "device": self.device,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restores normalizer state from a checkpoint dictionary.

        Args:
            state: Dictionary previously returned by ``state_dict()``.
        """
        self.device = state.get("device", self.device)
        self.fitted = bool(state.get("fitted", False))

        mean_cpu: Optional[torch.Tensor] = state.get("mean", None)
        std_cpu: Optional[torch.Tensor] = state.get("std", None)

        if mean_cpu is not None:
            self.mean = mean_cpu.to(device=self.device, dtype=torch.float32)
        if std_cpu is not None:
            self.std = std_cpu.to(device=self.device, dtype=torch.float32)

    def __repr__(self) -> str:
        """Returns a concise string representation of the normalizer state."""
        if self.fitted and self.mean is not None:
            return (
                f"Normalizer(fitted=True, input_dim={self.mean.shape[-1]}, "
                f"device='{self.device}')"
            )
        return f"Normalizer(fitted=False, device='{self.device}')"
