## utils/ema.py
"""Exponential Moving Average (EMA) for consistency model weights.

This module implements the EMA weight tracking required by Algorithm 1 of the
paper "Improving Consistency Models with Generator-Augmented Flows". The EMA
model is used exclusively for stop-gradient endpoint prediction:

    x_hat = sg(f_ema(x_ti, sigma_ti))

The paper's ablation (Table 3) confirms EMA is critical: removing it raises
FID from 5.95 to 6.73 on CIFAR-10.

Typical usage in the training loop::

    ema = EMA(model, decay=0.9999)

    # Inside training step, for endpoint prediction:
    ema.apply_shadow()
    with torch.no_grad():
        x_hat = model(x_ti, sigma_i)
    ema.restore()

    # After optimizer step:
    optimizer.step()
    ema.update(model)
"""

from typing import Dict, Optional

import torch
import torch.nn as nn


class EMA:
    """Exponential Moving Average of model parameters and buffers.

    Maintains a smoothed copy of all named parameters and buffers from a
    ``torch.nn.Module``. The shadow weights are updated after each optimizer
    step using:

        shadow[name] = decay * shadow[name] + (1 - decay) * param.data

    Buffers (e.g. BatchNorm running statistics) are copied directly rather
    than EMA-averaged, since they already represent running statistics
    maintained by PyTorch and a lagged average would be incorrect.

    The class does **not** hold a persistent reference to the model after
    ``__init__``. The model is only accessed transiently in ``update()``,
    ``apply_shadow()``, and ``restore()``.

    Attributes:
        decay: Smoothing coefficient in ``[0, 1)``. Higher values produce
            slower-moving averages. Default 0.9999 follows iCT convention.
        shadow: Dict mapping parameter/buffer names to their EMA-smoothed
            tensors. These are the "EMA model weights".
        backup: Temporary storage for online model weights during the
            ``apply_shadow`` / ``restore`` cycle. Empty outside that cycle.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        """Initialise EMA by snapshotting the model's current state.

        Takes a ``clone().detach()`` copy of all named parameters and named
        buffers. The clone ensures we own independent memory; the detach
        ensures no gradient graph is attached to the shadow weights.

        ``None`` buffers (placeholder registrations) are skipped silently.

        Args:
            model: The consistency model whose weights will be tracked.
                Typically a ``ConsistencyModel`` instance wrapping a
                ``SongUNet``.
            decay: EMA decay coefficient. Must satisfy ``0 <= decay < 1``.
                Default 0.9999 matches the iCT (Song & Dhariwal, 2024)
                convention and provides very smooth averaging over ~10k steps.

        Raises:
            ValueError: If ``decay`` is outside ``[0, 1)``.
        """
        if not (0.0 <= decay < 1.0):
            raise ValueError(
                f"EMA decay must be in [0, 1), got {decay}."
            )

        self.decay: float = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}

        # Snapshot named parameters
        for name, param in model.named_parameters():
            self.shadow[name] = param.data.clone().detach()

        # Snapshot named buffers (skip None placeholders)
        for name, buffer in model.named_buffers():
            if buffer is not None:
                self.shadow[name] = buffer.data.clone().detach()

    def update(self, model: nn.Module) -> None:
        """Update shadow weights with the latest online model weights.

        Must be called **after** ``optimizer.step()`` so the shadow tracks
        the freshly updated parameters.

        Parameters are updated with the EMA rule:
            shadow[name] = decay * shadow[name] + (1 - decay) * param.data

        Buffers are copied directly (no EMA) since they represent running
        statistics that should reflect the current data distribution, not a
        lagged average.

        Unknown keys (names not in ``self.shadow``) are skipped to guard
        against unexpected architecture changes.

        Args:
            model: The online (training) model after an optimizer step.
        """
        with torch.no_grad():
            # EMA update for learnable parameters
            for name, param in model.named_parameters():
                if name not in self.shadow:
                    # New parameter added after init — initialise shadow entry
                    self.shadow[name] = param.data.clone().detach()
                    continue
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

            # Direct copy for buffers (e.g. BatchNorm running mean/variance)
            for name, buffer in model.named_buffers():
                if buffer is None:
                    continue
                if name not in self.shadow:
                    self.shadow[name] = buffer.data.clone().detach()
                    continue
                self.shadow[name].copy_(buffer.data)

    def apply_shadow(self, model: nn.Module) -> None:
        """Temporarily replace online model weights with EMA shadow weights.

        Saves the current online weights to ``self.backup``, then loads the
        EMA shadow weights into the model. After this call the model behaves
        as the EMA model and can be used for endpoint prediction.

        Must always be followed by a call to ``restore()`` before the next
        optimizer step, otherwise the online weights are lost.

        If ``self.backup`` is already non-empty (indicating a nested call),
        a warning is printed and the existing backup is preserved to avoid
        data loss.

        Args:
            model: The consistency model to temporarily modify in-place.
        """
        if self.backup:
            print(
                "[EMA] WARNING: apply_shadow() called while backup is "
                "non-empty. A previous apply_shadow() may not have been "
                "followed by restore(). Skipping backup overwrite to "
                "preserve existing backup."
            )
        else:
            # Back up current online parameters
            for name, param in model.named_parameters():
                self.backup[name] = param.data.clone()

            # Back up current buffers
            for name, buffer in model.named_buffers():
                if buffer is not None:
                    self.backup[name] = buffer.data.clone()

        # Load EMA shadow weights into model parameters
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

        # Load EMA shadow state into model buffers
        for name, buffer in model.named_buffers():
            if buffer is not None and name in self.shadow:
                buffer.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        """Restore the online model weights from backup.

        Reverses the effect of ``apply_shadow()``. After this call the model
        is back to its online (training) state and gradient computation can
        proceed normally.

        ``self.backup`` is cleared after restoration to free memory and to
        make accidental double-restore detectable (it would be a no-op on an
        empty backup).

        Args:
            model: The consistency model to restore in-place. Must be the
                same model instance that was passed to ``apply_shadow()``.
        """
        if not self.backup:
            # Nothing to restore — apply_shadow was not called or backup was
            # already cleared. This is a no-op rather than an error to keep
            # the training loop robust.
            return

        # Restore online parameters from backup
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])

        # Restore buffers from backup
        for name, buffer in model.named_buffers():
            if buffer is not None and name in self.backup:
                buffer.data.copy_(self.backup[name])

        # Clear backup to free memory and signal that restore is complete
        self.backup.clear()

    def state_dict(self) -> dict:
        """Serialise EMA state for checkpoint saving.

        Returns shadow weights as CPU tensors for device-portable checkpoints.
        The decay value is also stored so it can be restored exactly.

        Returns:
            Dictionary with keys:
                - ``'shadow'``: dict mapping names to CPU float tensors.
                - ``'decay'``: float scalar.

        Example::

            torch.save(
                {'model': model.state_dict(), 'ema': ema.state_dict()},
                'checkpoint.pt'
            )
        """
        cpu_shadow: Dict[str, torch.Tensor] = {
            name: tensor.cpu().clone()
            for name, tensor in self.shadow.items()
        }
        return {
            "shadow": cpu_shadow,
            "decay": self.decay,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore EMA state from a checkpoint dictionary.

        Moves shadow tensors to the same device as the current shadow weights
        (inferred from the first entry in ``self.shadow``). This handles the
        common case of loading a CPU-saved checkpoint onto a GPU.

        If ``self.shadow`` is empty (e.g. loading before any model is
        attached), tensors are kept on their saved device.

        Args:
            state: Dictionary as returned by ``state_dict()``. Must contain
                keys ``'shadow'`` (dict of tensors) and ``'decay'`` (float).

        Raises:
            KeyError: If ``state`` is missing required keys.
        """
        if "shadow" not in state or "decay" not in state:
            raise KeyError(
                "EMA state_dict must contain keys 'shadow' and 'decay'. "
                f"Got keys: {list(state.keys())}"
            )

        self.decay = float(state["decay"])

        # Infer target device from current shadow (if populated)
        target_device: Optional[torch.device] = None
        if self.shadow:
            first_tensor = next(iter(self.shadow.values()))
            target_device = first_tensor.device

        loaded_shadow: Dict[str, torch.Tensor] = state["shadow"]
        self.shadow = {}

        for name, tensor in loaded_shadow.items():
            if target_device is not None:
                self.shadow[name] = tensor.to(device=target_device).clone()
            else:
                self.shadow[name] = tensor.clone()

    def __repr__(self) -> str:
        """Return a human-readable summary of the EMA state."""
        num_params = sum(
            t.numel() for t in self.shadow.values()
        )
        return (
            f"EMA(decay={self.decay}, "
            f"num_tracked_tensors={len(self.shadow)}, "
            f"total_elements={num_params:,}, "
            f"backup_active={bool(self.backup)})"
        )
