## training/scheduler.py
"""Learning rate scheduler for NaViL's staged training pipeline.

This module implements ``LRScheduler``, a standalone learning rate scheduler
that supports the two distinct schedules used across NaViL's three training
stages (from configs/navil_2b.yaml and configs/navil_9b.yaml):

- ``"constant_warmup"``: linear warmup from 0 to ``peak_lr`` over
  ``warmup_steps``, then constant at ``peak_lr`` for the remainder.
  Used in S1.1 (70k steps, lr=5e-5) and S1.2 (40k steps, lr=5e-5).

- ``"cosine"``: linear warmup from 0 to ``peak_lr`` over ``warmup_steps``,
  then cosine decay from ``peak_lr`` to ``min_lr`` over the remaining steps.
  Used in S2 (30k steps, lr=2e-5).

Config alignment (configs/navil_2b.yaml):
    training.s1_1.lr_schedule:   "constant_warmup"
    training.s1_1.peak_lr:       5.0e-5
    training.s1_1.warmup_steps:  200
    training.s1_1.steps:         70000

    training.s1_2.lr_schedule:   "constant_warmup"
    training.s1_2.peak_lr:       5.0e-5
    training.s1_2.warmup_steps:  200
    training.s1_2.steps:         40000

    training.s2.lr_schedule:     "cosine"
    training.s2.peak_lr:         2.0e-5
    training.s2.warmup_steps:    200
    training.s2.steps:           30000

Design constraints:
- Standalone class (not a subclass of torch.optim.lr_scheduler._LRScheduler).
  This avoids last_epoch bookkeeping and keeps the LR trajectory transparent.
- A new instance is created per training stage in NaViLTrainer.setup_stage().
  current_step resets to 0 at the start of each stage.
- All optimizer param groups receive the same absolute LR value (consistent
  with NaViLTrainer.setup_optimizer() which creates weight-decay vs
  non-weight-decay groups that share the same LR).
- No internal project dependencies (leaf utility module).

LR schedule formulas:

    Warmup phase (step < warmup_steps):
        lr = peak_lr * (step / warmup_steps)

    constant_warmup post-warmup (step >= warmup_steps):
        lr = peak_lr

    cosine post-warmup (step >= warmup_steps):
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + cos(pi * progress))

    At progress=0: lr = peak_lr  (seamless transition from warmup)
    At progress=1: lr = min_lr   (end of training)
"""

import math
import logging
from typing import Optional

import torch.optim

logger: logging.Logger = logging.getLogger(__name__)


class LRScheduler:
    """Custom learning rate scheduler for NaViL's staged training pipeline.

    Supports two schedule types:
    - ``"constant_warmup"``: linear warmup then constant plateau.
    - ``"cosine"``: linear warmup then cosine decay to ``min_lr``.

    A new instance should be created for each training stage. The internal
    ``current_step`` counter starts at 0 and is incremented by each call
    to ``step()``.

    Args:
        optimizer:      The AdamW optimizer whose ``param_groups[*]['lr']``
                        will be updated on each ``step()`` call.
        schedule_type:  LR schedule type. One of ``"constant_warmup"`` or
                        ``"cosine"``. Defaults to ``"constant_warmup"``.
        peak_lr:        Maximum learning rate reached at the end of warmup
                        and maintained (constant_warmup) or decayed from
                        (cosine). Defaults to ``5e-5`` (S1.1/S1.2 setting).
        warmup_steps:   Number of steps for linear warmup from 0 to
                        ``peak_lr``. Defaults to ``200`` (from config).
        total_steps:    Total number of training steps for this stage.
                        Used by cosine decay to compute progress.
                        Defaults to ``70000`` (S1.1 setting).
        min_lr:         Minimum learning rate for cosine decay. The LR
                        decays to this value at ``total_steps``. Has no
                        effect for ``"constant_warmup"`` schedule.
                        Defaults to ``0.0``.

    Attributes:
        optimizer:      Stored optimizer reference.
        schedule_type:  Stored schedule type string.
        peak_lr:        Stored peak learning rate.
        warmup_steps:   Stored warmup step count.
        total_steps:    Stored total step count.
        min_lr:         Stored minimum learning rate.
        current_step:   Internal step counter. Starts at 0. Incremented
                        by each ``step()`` call. Can be set directly to
                        restore state from a checkpoint.

    Raises:
        ValueError: If ``schedule_type`` is not ``"constant_warmup"`` or
                    ``"cosine"``.
        ValueError: If ``peak_lr`` is not positive.
        ValueError: If ``warmup_steps`` is negative.
        ValueError: If ``total_steps`` is not positive.
        ValueError: If ``min_lr`` is negative.

    Example::

        # Stage S1.1: constant warmup, 70k steps
        scheduler = LRScheduler(
            optimizer=optimizer,
            schedule_type="constant_warmup",
            peak_lr=5e-5,
            warmup_steps=200,
            total_steps=70000,
            min_lr=0.0,
        )
        for step in range(70000):
            scheduler.step()  # updates optimizer LR and increments counter

        # Stage S2: cosine decay, 30k steps
        scheduler_s2 = LRScheduler(
            optimizer=optimizer,
            schedule_type="cosine",
            peak_lr=2e-5,
            warmup_steps=200,
            total_steps=30000,
            min_lr=0.0,
        )
    """

    # Valid schedule type strings
    _VALID_SCHEDULE_TYPES: tuple = ("constant_warmup", "cosine")

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        schedule_type: str = "constant_warmup",
        peak_lr: float = 5e-5,
        warmup_steps: int = 200,
        total_steps: int = 70000,
        min_lr: float = 0.0,
    ) -> None:
        """Initialise LRScheduler and validate all parameters.

        Args:
            optimizer:      AdamW optimizer to update.
            schedule_type:  ``"constant_warmup"`` or ``"cosine"``.
            peak_lr:        Maximum learning rate. Must be positive.
            warmup_steps:   Linear warmup duration in steps. Must be >= 0.
            total_steps:    Total stage duration in steps. Must be positive.
            min_lr:         Cosine decay floor. Must be >= 0.

        Raises:
            ValueError: If any parameter is out of valid range.
        """
        # ------------------------------------------------------------------ #
        # Validate schedule_type                                               #
        # ------------------------------------------------------------------ #
        if schedule_type not in self._VALID_SCHEDULE_TYPES:
            raise ValueError(
                f"schedule_type must be one of {self._VALID_SCHEDULE_TYPES}, "
                f"got '{schedule_type}'."
            )

        # ------------------------------------------------------------------ #
        # Validate numeric parameters                                          #
        # ------------------------------------------------------------------ #
        if peak_lr <= 0.0:
            raise ValueError(
                f"peak_lr must be positive, got {peak_lr}. "
                "Typical values: 5e-5 (S1.1/S1.2), 2e-5 (S2)."
            )

        if warmup_steps < 0:
            raise ValueError(
                f"warmup_steps must be >= 0, got {warmup_steps}. "
                "Set to 0 to disable warmup (not recommended)."
            )

        if total_steps <= 0:
            raise ValueError(
                f"total_steps must be positive, got {total_steps}. "
                "Typical values: 70000 (S1.1), 40000 (S1.2), 30000 (S2)."
            )

        if min_lr < 0.0:
            raise ValueError(
                f"min_lr must be >= 0, got {min_lr}. "
                "Set to 0.0 to decay to zero (default)."
            )

        # ------------------------------------------------------------------ #
        # Store all parameters                                                 #
        # ------------------------------------------------------------------ #
        self.optimizer: torch.optim.Optimizer = optimizer
        self.schedule_type: str = schedule_type
        self.peak_lr: float = peak_lr
        self.warmup_steps: int = warmup_steps
        self.total_steps: int = total_steps
        self.min_lr: float = min_lr

        # Internal step counter — starts at 0, incremented by step()
        self.current_step: int = 0

        logger.info(
            "LRScheduler initialised: schedule_type=%s, peak_lr=%.2e, "
            "warmup_steps=%d, total_steps=%d, min_lr=%.2e",
            schedule_type,
            peak_lr,
            warmup_steps,
            total_steps,
            min_lr,
        )

    def get_lr(self, step: int) -> float:
        """Compute the learning rate for a given step index.

        This is a pure function with no side effects — it does not modify
        ``current_step`` or the optimizer. It can be called freely for
        inspection or logging without advancing the scheduler state.

        Schedule logic:

        **Warmup phase** (``step < warmup_steps``):
            ``lr = peak_lr * (step / warmup_steps)``
            At step=0: lr=0. At step=warmup_steps: lr=peak_lr.

        **constant_warmup post-warmup** (``step >= warmup_steps``):
            ``lr = peak_lr``
            Constant plateau for the remainder of the stage.

        **cosine post-warmup** (``step >= warmup_steps``):
            ``progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)``
            ``lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + cos(pi * progress))``
            At progress=0: lr=peak_lr. At progress=1: lr=min_lr.
            For step > total_steps: progress > 1, cos(pi*progress) < -1 is
            clamped to min_lr via max(min_lr, lr).

        Args:
            step: The step index (0-indexed). Typically ``current_step``
                  from the internal counter, but any non-negative integer
                  is valid for inspection purposes.

        Returns:
            The learning rate as a float for the given step.

        Example::

            scheduler = LRScheduler(
                optimizer, "cosine", peak_lr=2e-5,
                warmup_steps=200, total_steps=30000
            )
            # Check LR at key points
            print(scheduler.get_lr(0))       # ≈ 0.0 (start of warmup)
            print(scheduler.get_lr(100))     # ≈ 1e-5 (mid warmup)
            print(scheduler.get_lr(200))     # = 2e-5 (end of warmup)
            print(scheduler.get_lr(15100))   # ≈ 1e-5 (mid cosine)
            print(scheduler.get_lr(30000))   # = 0.0 (end of training)
        """
        # ------------------------------------------------------------------ #
        # Warmup phase: linear ramp from 0 to peak_lr                        #
        # ------------------------------------------------------------------ #
        # Guard warmup_steps=0 to avoid division by zero.
        # When warmup_steps=0, the warmup phase is skipped entirely and we
        # jump directly to the post-warmup schedule at step 0.
        if self.warmup_steps > 0 and step < self.warmup_steps:
            # Linear interpolation: lr = peak_lr * (step / warmup_steps)
            # Using min(1.0, ...) as a safety clamp for floating-point edge cases
            warmup_progress: float = min(1.0, step / self.warmup_steps)
            lr: float = self.peak_lr * warmup_progress
            return lr

        # ------------------------------------------------------------------ #
        # Post-warmup phase: branch on schedule_type                          #
        # ------------------------------------------------------------------ #
        if self.schedule_type == "constant_warmup":
            # ---------------------------------------------------------------- #
            # Constant plateau at peak_lr                                      #
            # ---------------------------------------------------------------- #
            return self.peak_lr

        elif self.schedule_type == "cosine":
            # ---------------------------------------------------------------- #
            # Cosine decay from peak_lr to min_lr                             #
            # ---------------------------------------------------------------- #
            # Compute progress in [0, 1] over the post-warmup decay phase.
            # Guard denominator: if total_steps == warmup_steps, the decay
            # phase has zero length; return min_lr immediately.
            decay_steps: int = self.total_steps - self.warmup_steps
            if decay_steps <= 0:
                return self.min_lr

            # Clamp step to [warmup_steps, total_steps] before computing progress
            # to handle the edge case where step > total_steps (e.g., if the
            # training loop overshoots by one step).
            clamped_step: int = min(step, self.total_steps)
            progress: float = (clamped_step - self.warmup_steps) / decay_steps
            # progress is in [0, 1] after clamping

            # Standard cosine annealing formula:
            #   lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + cos(pi * progress))
            # At progress=0: cos(0) = 1  → lr = min_lr + (peak_lr - min_lr) = peak_lr
            # At progress=1: cos(pi) = -1 → lr = min_lr + 0 = min_lr
            cosine_factor: float = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = self.min_lr + (self.peak_lr - self.min_lr) * cosine_factor

            # Safety clamp: ensure lr never goes below min_lr due to floating-point
            lr = max(self.min_lr, lr)
            return lr

        else:
            # This branch should never be reached due to __init__ validation,
            # but included for defensive completeness.
            logger.warning(
                "Unknown schedule_type '%s' in get_lr(). Returning peak_lr.",
                self.schedule_type,
            )
            return self.peak_lr

    def step(self) -> None:
        """Apply the current step's LR to the optimizer and advance the counter.

        Two operations in order:
        1. Compute ``lr = get_lr(current_step)`` for the current step.
        2. Write ``lr`` into every param group of the optimizer.
        3. Increment ``current_step`` by 1.

        The LR update happens **before** the increment so that:
        - Step 0 applies the warmup-start LR (≈ 0), which is correct for
          the very first optimizer step.
        - Step ``warmup_steps`` applies ``peak_lr``, the first post-warmup LR.

        This method should be called once per training step, after
        ``optimizer.step()`` and before the next forward pass:

            loss.backward()
            optimizer.step()
            scheduler.step()   ← updates LR for the NEXT optimizer step

        Alternatively, some implementations call ``scheduler.step()`` before
        ``optimizer.step()`` to set the LR for the current step. Either
        convention is consistent as long as it is applied uniformly throughout
        training. NaViLTrainer calls ``scheduler.step()`` after
        ``optimizer.step()`` (post-step update convention).

        Returns:
            None. This is a pure side-effect method.

        Example::

            for global_step in range(total_steps):
                loss = model(batch)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                # After step(): scheduler.current_step == global_step + 1
        """
        # ------------------------------------------------------------------ #
        # Step 1: Compute LR for the current step                             #
        # ------------------------------------------------------------------ #
        current_lr: float = self.get_lr(self.current_step)

        # ------------------------------------------------------------------ #
        # Step 2: Apply LR to all optimizer param groups                      #
        # ------------------------------------------------------------------ #
        # All param groups receive the same absolute LR. This is consistent
        # with NaViLTrainer.setup_optimizer() which creates weight-decay and
        # non-weight-decay groups that share the same LR (only weight_decay
        # differs between groups).
        param_group: dict
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr

        # ------------------------------------------------------------------ #
        # Step 3: Increment internal step counter                             #
        # ------------------------------------------------------------------ #
        self.current_step += 1

        # Log LR at warmup boundary and every 1000 steps for debugging
        if (
            self.current_step == self.warmup_steps
            or self.current_step % 1000 == 0
        ):
            logger.debug(
                "LRScheduler step=%d: lr=%.6e (schedule=%s)",
                self.current_step,
                current_lr,
                self.schedule_type,
            )
