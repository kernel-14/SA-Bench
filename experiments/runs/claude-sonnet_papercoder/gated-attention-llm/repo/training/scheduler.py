## training/scheduler.py
"""Warmup + cosine decay learning rate scheduler for gated attention experiments.

This module implements WarmupCosineScheduler, a manual learning rate scheduler
that applies linear warmup followed by cosine decay. It is used by the Trainer
for all experiments in the paper.

Schedule phases:
    Phase 1 — Linear warmup: lr = max_lr * step / warmup_steps
        From step=0 (lr=0) to step=warmup_steps (lr=max_lr).
        Paper: "warms up to a maximum LR of 2e-3 in 1k steps" (MoE, Sec 3.2.1).

    Phase 2 — Cosine decay: lr = min_lr + 0.5*(max_lr-min_lr)*(1+cos(pi*progress))
        From step=warmup_steps (lr=max_lr) to step=total_steps (lr=min_lr).
        Paper: "decays using cosine to 3e-5" (MoE, Sec 3.2.1).

Config values used (from config.yaml):
    training.max_lr: Peak learning rate after warmup.
        MoE: 2e-3, Dense 28L 400B: 4e-3, Dense 28L 3.5T: 4.5e-3,
        Dense 48L high-LR: 8e-3, Dense 48L 1T: 5.3e-3.
    training.min_lr: Final learning rate after cosine decay.
        MoE: 3e-5, Dense: 1e-5.
    training.warmup_steps: Number of linear warmup steps.
        All configs: 1000 (paper: "warms up ... in 1k steps").
    training.total_steps: Total optimization steps.
        MoE: 100000 (explicit in config moe_15a2b_400b.total_steps: 100000).
        Dense: computed by Trainer as train_tokens // (batch_size * max_seq_len).

Design notes:
    - This is NOT a subclass of torch.optim.lr_scheduler._LRScheduler.
      It is a standalone stateful object that directly sets optimizer param group LRs.
    - get_lr(step) is a pure function (no side effects) for logging and testing.
    - step() advances the internal counter and applies the LR to the optimizer.
    - All optimizer param groups receive the same LR (weight decay differs, not LR).
"""

import math
from typing import List

from torch.optim import Optimizer


class WarmupCosineScheduler:
    """Manual learning rate scheduler with linear warmup and cosine decay.

    Implements the two-phase LR schedule used across all paper experiments:
        1. Linear warmup from 0 to max_lr over warmup_steps steps.
        2. Cosine decay from max_lr to min_lr over (total_steps - warmup_steps) steps.

    This scheduler directly modifies optimizer param group learning rates,
    making it compatible with any PyTorch optimizer without subclassing
    torch.optim.lr_scheduler._LRScheduler.

    Numerical guarantees:
        - get_lr(0) == 0.0 (warmup starts from zero)
        - get_lr(warmup_steps) == max_lr (exact, no discontinuity at transition)
        - get_lr(total_steps) == min_lr (exact, cosine reaches minimum)
        - get_lr(step > total_steps) == min_lr (clamped, never goes below min_lr)
        - Monotonically non-increasing after warmup (cosine on [0, pi])

    Attributes:
        optimizer: The AdamW optimizer instance whose param group LRs are updated.
        warmup_steps: Number of linear warmup steps. From training.warmup_steps.
        max_lr: Peak learning rate reached at end of warmup. From training.max_lr.
        min_lr: Final learning rate at end of cosine decay. From training.min_lr.
        total_steps: Total optimization steps. From training.total_steps or
            computed by Trainer as train_tokens // (batch_size * max_seq_len).
        current_step: Mutable counter tracking the current optimization step.
            Initialized to 0. Can be set externally by Trainer.load_checkpoint()
            to resume training from a checkpoint at the correct LR.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 1000,
        max_lr: float = 2e-3,
        min_lr: float = 3e-5,
        total_steps: int = 100000,
    ) -> None:
        """Initialize the warmup cosine scheduler.

        Sets the initial learning rate on all optimizer param groups to 0.0
        (the starting point of the linear warmup phase). This ensures the
        optimizer begins training with lr=0 regardless of what lr was set
        during optimizer construction.

        Args:
            optimizer: PyTorch optimizer instance (AdamW in all paper experiments).
                All param groups will receive the same LR from this scheduler.
                The Trainer creates two param groups (decay/no-decay) that both
                receive the same LR — weight decay differs between groups, not LR.
            warmup_steps: Number of linear warmup steps. From training.warmup_steps.
                Default 1000 (paper: "warms up to a maximum LR ... in 1k steps",
                Sec 3.2.1). All model configs use warmup_steps=1000.
            max_lr: Peak learning rate reached at the end of warmup.
                From training.max_lr. Default 2e-3 (MoE config).
                Dense configs use 4e-3, 4.5e-3, 5.3e-3, or 8e-3.
            min_lr: Final learning rate at the end of cosine decay.
                From training.min_lr. Default 3e-5 (MoE config: "decays using
                cosine to 3e-5", Sec 3.2.1). Dense configs use 1e-5.
            total_steps: Total number of optimization steps.
                From training.total_steps (explicit for MoE: 100000).
                For dense models, computed by Trainer as:
                    total_steps = train_tokens // (batch_size * max_seq_len)
                Default 100000 (MoE config: "comprising 100k optimization steps",
                Sec 3.2.1).

        Raises:
            ValueError: If warmup_steps < 0, total_steps < 0, max_lr <= 0,
                min_lr < 0, or min_lr > max_lr.
        """
        # Input validation
        if warmup_steps < 0:
            raise ValueError(
                f"warmup_steps must be non-negative, got {warmup_steps}."
            )
        if total_steps < 0:
            raise ValueError(
                f"total_steps must be non-negative, got {total_steps}."
            )
        if max_lr <= 0.0:
            raise ValueError(
                f"max_lr must be positive, got {max_lr}."
            )
        if min_lr < 0.0:
            raise ValueError(
                f"min_lr must be non-negative, got {min_lr}."
            )
        if min_lr > max_lr:
            raise ValueError(
                f"min_lr ({min_lr}) must be <= max_lr ({max_lr})."
            )

        self.optimizer: Optimizer = optimizer
        self.warmup_steps: int = warmup_steps
        self.max_lr: float = max_lr
        self.min_lr: float = min_lr
        self.total_steps: int = total_steps

        # Mutable step counter — can be set externally by Trainer.load_checkpoint()
        # to resume training from a checkpoint at the correct LR position.
        self.current_step: int = 0

        # Initialize all optimizer param groups to lr=0.0 (start of warmup).
        # This overrides whatever lr was set during optimizer construction,
        # ensuring training always starts from lr=0 regardless of config.
        self._set_lr(0.0)

    def _set_lr(self, lr: float) -> None:
        """Set the learning rate on all optimizer param groups.

        Updates every param group in the optimizer to the specified lr.
        The Trainer creates two param groups (weight-decay and no-weight-decay),
        both of which receive the same LR from this scheduler.

        Args:
            lr: Learning rate value to set on all param groups.
        """
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self, step: int) -> float:
        """Compute the learning rate for a given step without side effects.

        This is a pure function — it does not modify any state. It can be
        called at any step for logging, testing, or LR curve visualization
        without advancing the scheduler's internal counter.

        The schedule has two phases:

        Phase 1 — Linear warmup (step < warmup_steps):
            lr = max_lr * step / warmup_steps
            At step=0: lr=0.0 (training starts from zero LR).
            At step=warmup_steps-1: lr approaches max_lr but hasn't reached it.

        Phase 2 — Cosine decay (step >= warmup_steps):
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))
            At step=warmup_steps: progress=0, cos(0)=1, lr=max_lr (continuous).
            At step=total_steps: progress=1, cos(pi)=-1, lr=min_lr.
            At step>total_steps: progress clamped to 1.0, lr=min_lr.

        Continuity at transition (step=warmup_steps):
            Phase 1 at step=warmup_steps-1: lr ≈ max_lr * (warmup_steps-1)/warmup_steps
            Phase 2 at step=warmup_steps: progress=0, lr=max_lr
            The transition is smooth — Phase 2 starts exactly at max_lr.

        Args:
            step: The optimization step index (0-based). Corresponds to
                self.current_step when called from step().

        Returns:
            Learning rate as a float for the given step.
            Always in the range [min_lr, max_lr] for step >= 0.
            Returns 0.0 for step=0 (start of warmup).
            Returns max_lr for step=warmup_steps (end of warmup).
            Returns min_lr for step >= total_steps (end of cosine decay).

        Example (MoE config: max_lr=2e-3, min_lr=3e-5, warmup=1000, total=100000):
            >>> scheduler.get_lr(0)
            0.0
            >>> scheduler.get_lr(500)
            0.001  # halfway through warmup
            >>> scheduler.get_lr(1000)
            0.002  # peak LR at end of warmup
            >>> scheduler.get_lr(100000)
            3e-05  # min LR at end of cosine decay
        """
        # -----------------------------------------------------------------------
        # Phase 1: Linear warmup
        # lr = max_lr * step / warmup_steps
        # -----------------------------------------------------------------------
        if step < self.warmup_steps:
            # Guard against warmup_steps=0 (degenerate case: no warmup)
            if self.warmup_steps == 0:
                return self.max_lr
            return self.max_lr * float(step) / float(self.warmup_steps)

        # -----------------------------------------------------------------------
        # Phase 2: Cosine decay
        # progress = (step - warmup_steps) / (total_steps - warmup_steps)
        # lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))
        # -----------------------------------------------------------------------

        # Guard against degenerate case: total_steps <= warmup_steps
        # (no cosine phase — return max_lr for all post-warmup steps)
        cosine_steps: int = self.total_steps - self.warmup_steps
        if cosine_steps <= 0:
            return self.max_lr

        # Compute progress in [0, 1], clamped to avoid going below min_lr
        # after total_steps is reached.
        progress: float = float(step - self.warmup_steps) / float(cosine_steps)

        # Clamp progress to [0, 1] to handle step > total_steps gracefully.
        # This ensures lr never drops below min_lr.
        progress = min(progress, 1.0)
        progress = max(progress, 0.0)

        # Cosine annealing formula:
        # lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))
        # At progress=0: cos(0)=1  → lr = min_lr + (max_lr - min_lr) = max_lr
        # At progress=1: cos(pi)=-1 → lr = min_lr + 0 = min_lr
        lr: float = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
            1.0 + math.cos(math.pi * progress)
        )

        return lr

    def step(self) -> None:
        """Advance the scheduler by one step and update optimizer learning rates.

        Called once per optimization step by the Trainer, after optimizer.step().
        Computes the LR for the current step, applies it to all optimizer param
        groups, then increments the internal step counter.

        The order of operations:
            1. Compute lr = get_lr(self.current_step)
            2. Set lr on all optimizer param groups
            3. Increment self.current_step

        This means:
            - Before the first call to step(): optimizer has lr=0.0 (set in __init__)
            - After the first call to step(): optimizer has lr=get_lr(0)=0.0,
              current_step=1
            - After the second call: optimizer has lr=get_lr(1), current_step=2
            - ...
            - After the warmup_steps-th call: optimizer has lr≈max_lr, current_step=warmup_steps

        Note:
            The Trainer also calls get_lr(step) separately for logging purposes
            (via ExperimentLogger.log_step). This is safe because get_lr() has
            no side effects.

        Note on checkpoint resumption:
            Trainer.load_checkpoint() sets self.current_step = global_step after
            loading a checkpoint. The next call to step() will then compute the
            correct LR for the resumed step, ensuring seamless LR continuation.
        """
        # Step 1: Compute LR for the current step (pure function, no side effects)
        lr: float = self.get_lr(self.current_step)

        # Step 2: Apply LR to all optimizer param groups
        # The Trainer creates two param groups (weight-decay and no-weight-decay).
        # Both receive the same LR — only weight_decay differs between groups.
        self._set_lr(lr)

        # Step 3: Advance the step counter
        self.current_step += 1

    def get_last_lr(self) -> List[float]:
        """Return the last computed learning rate for all param groups.

        Provides compatibility with code that expects a PyTorch-style scheduler
        interface (e.g., logging utilities that call scheduler.get_last_lr()).

        Returns the LR that was most recently applied to the optimizer, which
        corresponds to get_lr(current_step - 1) if step() has been called at
        least once, or 0.0 if step() has never been called.

        Returns:
            List of floats, one per optimizer param group. All values are
            identical since this scheduler applies the same LR to all groups.
        """
        # The last applied LR corresponds to the step before current_step
        last_step: int = max(0, self.current_step - 1)
        last_lr: float = self.get_lr(last_step)
        return [last_lr] * len(self.optimizer.param_groups)

    def state_dict(self) -> dict:
        """Return the scheduler state for checkpointing.

        Captures all mutable state needed to resume training from a checkpoint.
        The optimizer state is saved separately by CheckpointManager.

        Returns:
            Dict containing:
                - 'current_step': Current optimization step counter (int).
                - 'warmup_steps': Warmup steps configuration (int).
                - 'max_lr': Peak learning rate (float).
                - 'min_lr': Minimum learning rate (float).
                - 'total_steps': Total optimization steps (int).

        Usage in CheckpointManager.save():
            checkpoint['scheduler_state'] = scheduler.state_dict()
        """
        return {
            "current_step": self.current_step,
            "warmup_steps": self.warmup_steps,
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
            "total_steps": self.total_steps,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore the scheduler state from a checkpoint.

        Restores all mutable state from a previously saved state_dict.
        After calling this method, the scheduler will resume from the
        correct step with the correct LR configuration.

        Args:
            state_dict: Dict previously returned by state_dict(). Must contain
                'current_step', 'warmup_steps', 'max_lr', 'min_lr', 'total_steps'.

        Usage in CheckpointManager.load():
            scheduler.load_state_dict(checkpoint['scheduler_state'])

        Note:
            After loading state, the optimizer's param group LRs are updated
            to match the restored step's LR. This ensures the optimizer is
            in a consistent state immediately after checkpoint restoration,
            even before the next call to step().
        """
        self.current_step = int(state_dict["current_step"])
        self.warmup_steps = int(state_dict["warmup_steps"])
        self.max_lr = float(state_dict["max_lr"])
        self.min_lr = float(state_dict["min_lr"])
        self.total_steps = int(state_dict["total_steps"])

        # Restore the optimizer LR to match the current step.
        # This ensures the optimizer is in a consistent state immediately
        # after checkpoint restoration, before the next call to step().
        restored_lr: float = self.get_lr(self.current_step)
        self._set_lr(restored_lr)

    def __repr__(self) -> str:
        """Return a human-readable string representation of the scheduler.

        Returns:
            String summarizing the scheduler configuration and current state.
        """
        current_lr: float = self.get_lr(self.current_step)
        return (
            f"WarmupCosineScheduler("
            f"max_lr={self.max_lr:.2e}, "
            f"min_lr={self.min_lr:.2e}, "
            f"warmup_steps={self.warmup_steps}, "
            f"total_steps={self.total_steps}, "
            f"current_step={self.current_step}, "
            f"current_lr={current_lr:.6e}"
            f")"
        )
