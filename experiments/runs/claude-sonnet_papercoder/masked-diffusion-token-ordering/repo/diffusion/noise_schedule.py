## diffusion/noise_schedule.py
"""Noise schedule for Masked Diffusion Models (MDMs).

Implements the forward masking process schedule α_t used throughout the
masked diffusion framework described in "Train for the Worst, Plan for the
Best: Understanding Token Ordering in Masked Diffusions".

The schedule controls how tokens are masked during training and how many
tokens are unmasked at each reverse inference step.  Two schedule types are
supported:

- **Linear**: α_t = 1 - t  (default, per config.yaml)
- **Cosine**: α_t = cos(πt/2)²

Both satisfy the boundary conditions α_0 = 1 (no masking) and α_1 = 0
(fully masked), as required by the MDM framework.

This module has no dependencies on other project files and is safe to import
first in any dependency chain.

Typical usage::

    schedule = NoiseSchedule(schedule_type='linear', T=50)

    # During training (MDMLoss):
    t = schedule.sample_t(batch_size=128, device='cuda')   # [128]
    alpha_t = schedule.alpha(t)                             # [128]
    mask_prob = schedule.get_mask_prob(t)                   # [128]
    weight = schedule.alpha_prime(t) / (1.0 - alpha_t + 1e-8)

    # During inference (VanillaSampler / AdaptiveSampler):
    timesteps = schedule.get_timesteps(n_steps=50)          # [51]
    for i in range(len(timesteps) - 1):
        t_val, s_val = timesteps[i], timesteps[i + 1]
        alpha_t_val = schedule.alpha(t_val.unsqueeze(0))
        alpha_s_val = schedule.alpha(s_val.unsqueeze(0))
        unmask_prob = (alpha_s_val - alpha_t_val) / (1.0 - alpha_t_val + 1e-8)
"""

import logging
import math
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Supported schedule type identifiers.
_VALID_SCHEDULE_TYPES: tuple = ("linear", "cosine")

# Small epsilon used to guard against division by zero in derived quantities
# such as the loss weight α_t' / (1 - α_t) and the unmasking probability
# (α_s - α_t) / (1 - α_t).  Consumers (MDMLoss, samplers) are expected to
# apply their own clamping; this constant is exposed for documentation.
_EPSILON: float = 1e-8


class NoiseSchedule:
    """Masking noise schedule for the MDM forward and reverse processes.

    Encapsulates the schedule function α_t and its derivative, providing
    all quantities needed by the training loss and inference samplers.

    The schedule must satisfy:
        - α_0 = 1  (no masking at t = 0, clean data)
        - α_1 = 0  (fully masked at t = 1)
        - α_t is monotonically decreasing on [0, 1]

    Attributes:
        schedule_type: One of ``'linear'`` or ``'cosine'``.
        T: Default number of discrete inference steps.  Used as the default
            argument to :meth:`get_timesteps` when ``n_steps`` is not
            provided explicitly.
    """

    def __init__(self, schedule_type: str = "linear", T: int = 50) -> None:
        """Initializes the NoiseSchedule.

        Args:
            schedule_type: Type of noise schedule.  Must be one of
                ``'linear'`` or ``'cosine'``.  Defaults to ``'linear'``
                (``α_t = 1 - t``), which is the schedule specified in
                ``config.yaml`` under ``noise_schedule.type``.
            T: Default number of discrete reverse-process steps used by
                :meth:`get_timesteps` when called without an explicit
                ``n_steps`` argument.  Corresponds to
                ``inference.n_steps`` in the experiment configs (e.g.
                ``nae_sat.inference.n_steps: 50``).

        Raises:
            ValueError: If ``schedule_type`` is not one of the supported
                values or if ``T`` is not a positive integer.
        """
        if schedule_type not in _VALID_SCHEDULE_TYPES:
            raise ValueError(
                f"Unsupported schedule_type '{schedule_type}'.  "
                f"Expected one of {_VALID_SCHEDULE_TYPES}."
            )
        if not isinstance(T, int) or T <= 0:
            raise ValueError(
                f"T must be a positive integer, got T={T!r}."
            )

        self.schedule_type: str = schedule_type
        self.T: int = T

        logger.info(
            "NoiseSchedule initialized: schedule_type='%s', T=%d.",
            self.schedule_type,
            self.T,
        )

    # ------------------------------------------------------------------
    # Core schedule functions
    # ------------------------------------------------------------------

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Computes α_t for the given noise levels.

        Returns the probability that a token remains *unmasked* at noise
        level ``t``.  Equivalently, ``1 - alpha(t)`` is the masking
        probability.

        Boundary values:
            - ``alpha(0) = 1.0``  (no masking, clean data)
            - ``alpha(1) = 0.0``  (fully masked)

        Args:
            t: Float tensor of noise levels in ``[0, 1]``.  Accepts any
                shape; the output has the same shape.

        Returns:
            Float tensor of the same shape as ``t``, with values clamped
            to ``[0, 1]`` for numerical stability.
        """
        t_clamped: torch.Tensor = torch.clamp(t.float(), min=0.0, max=1.0)

        if self.schedule_type == "linear":
            # α_t = 1 - t
            result: torch.Tensor = 1.0 - t_clamped
        else:
            # α_t = cos(π·t/2)²
            result = torch.cos(math.pi * t_clamped / 2.0) ** 2

        # Clamp to [0, 1] to guard against floating-point drift at boundaries.
        return torch.clamp(result, min=0.0, max=1.0)

    def alpha_prime(self, t: torch.Tensor) -> torch.Tensor:
        """Computes dα_t/dt, the time-derivative of the schedule.

        Used to compute the continuous-time ELBO loss weight
        ``α_t' / (1 - α_t)`` in :class:`diffusion.mdm_loss.MDMLoss`.

        The derivative is always ≤ 0 on ``[0, 1]`` since α_t is
        monotonically decreasing.

        Args:
            t: Float tensor of noise levels in ``[0, 1]``.  Accepts any
                shape; the output has the same shape.

        Returns:
            Float tensor of the same shape as ``t`` containing dα_t/dt.

        Note:
            The loss weight ``α_t' / (1 - α_t)`` diverges as ``t → 0``
            for the linear schedule (``-1 / t``).  Callers should clamp
            ``t`` away from zero (e.g. ``t = torch.clamp(t, min=1e-5)``)
            before computing the weight.
        """
        t_clamped: torch.Tensor = torch.clamp(t.float(), min=0.0, max=1.0)

        if self.schedule_type == "linear":
            # dα_t/dt = -1  (constant)
            result: torch.Tensor = torch.full_like(t_clamped, fill_value=-1.0)
        else:
            # α_t = cos(πt/2)²
            # dα_t/dt = 2·cos(πt/2)·(-sin(πt/2))·(π/2)
            #         = -π/2 · sin(πt)
            result = -(math.pi / 2.0) * torch.sin(math.pi * t_clamped)

        return result

    def get_mask_prob(self, t: torch.Tensor) -> torch.Tensor:
        """Returns the masking probability at noise level ``t``.

        This is simply ``1 - α_t``: the probability that any given token
        is replaced by the ``[MASK]`` token at noise level ``t``.

        Args:
            t: Float tensor of noise levels in ``[0, 1]``.  Accepts any
                shape; the output has the same shape.

        Returns:
            Float tensor of the same shape as ``t``, with values in
            ``[0, 1]``.
        """
        return 1.0 - self.alpha(t)

    # ------------------------------------------------------------------
    # Sampling utilities
    # ------------------------------------------------------------------

    def sample_t(
        self,
        batch_size: int,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Samples noise levels ``t ~ Uniform(0, 1)`` for a training batch.

        Each sequence in the batch receives an independently sampled noise
        level, as required by the continuous-time ELBO training objective
        (Equation 1 in the paper).

        Args:
            batch_size: Number of noise levels to sample.  Corresponds to
                the training batch size (e.g. ``128`` from
                ``nae_sat.data.batch_size`` in ``config.yaml``).
            device: Target device string (e.g. ``'cpu'``, ``'cuda'``).

        Returns:
            Float tensor of shape ``[batch_size]`` with values uniformly
            sampled from ``[0, 1]``.
        """
        if batch_size <= 0:
            raise ValueError(
                f"batch_size must be a positive integer, got {batch_size}."
            )
        return torch.rand(batch_size, device=device, dtype=torch.float32)

    def get_timesteps(self, n_steps: Optional[int] = None) -> torch.Tensor:
        """Returns evenly spaced timesteps for the reverse inference process.

        Generates a sequence of ``n_steps + 1`` timesteps linearly spaced
        from ``1.0`` (fully masked) down to ``0.0`` (fully unmasked).
        Inference samplers iterate over consecutive pairs
        ``(timesteps[i], timesteps[i+1])`` as ``(t, s)`` where ``s < t``.

        Example for ``n_steps=4``::

            [1.0, 0.75, 0.5, 0.25, 0.0]

        Example for ``n_steps=50`` (used in all puzzle experiments)::

            [1.0, 0.98, 0.96, ..., 0.02, 0.0]

        Args:
            n_steps: Number of reverse diffusion steps.  Defaults to
                ``self.T`` (set at construction time).  Corresponds to
                ``inference.n_steps`` in the experiment configs (e.g.
                ``nae_sat.inference.n_steps: 50``).

        Returns:
            Float tensor of shape ``[n_steps + 1]`` with values linearly
            spaced from ``1.0`` to ``0.0`` inclusive.

        Raises:
            ValueError: If ``n_steps`` is not a positive integer.
        """
        resolved_steps: int = n_steps if n_steps is not None else self.T

        if not isinstance(resolved_steps, int) or resolved_steps <= 0:
            raise ValueError(
                f"n_steps must be a positive integer, got {resolved_steps!r}."
            )

        # torch.linspace(1.0, 0.0, n_steps + 1) gives exactly n_steps + 1
        # evenly spaced values from 1.0 down to 0.0 inclusive.
        timesteps: torch.Tensor = torch.linspace(
            1.0, 0.0, resolved_steps + 1, dtype=torch.float32
        )
        return timesteps
