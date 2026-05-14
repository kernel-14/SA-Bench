## training/lr_scheduler.py
"""Learning rate schedulers for OLMoE pretraining and adaptation.

Implements three-phase LR scheduling for pretraining (warmup → cosine → linear
annealing) and constant LR scheduling for SFT/DPO/KTO adaptation.

Training loss schedule (from Table 10 and Appendix B of the paper):

  Phase 1 — Linear Warmup [0, warmup_steps):
      lr = peak_lr * (step / warmup_steps)
      warmup_steps = 2500, peak_lr = 4e-4

  Phase 2 — Cosine Decay [warmup_steps, cosine_end_step):
      progress = (step - warmup_steps) / (cosine_end_step - warmup_steps)
      lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + cos(π * progress))
      min_lr = 4e-5

  Phase 3 — Linear Annealing [cosine_end_step, max_steps]:
      progress = (step - cosine_end_step) / annealing_steps
      lr = min_lr * (1 - progress)
      annealing_min_lr = 0.0 (decays to zero)

Adaptation schedules (Appendix B):
  SFT:  constant lr = 2e-5
  DPO:  constant lr = 5e-7
  KTO:  constant lr = 5e-7

Configuration values used (from config.yaml):
  pretraining.learning_rate: 4.0e-04       # peak LR (Table 10, v2 corrected)
  pretraining.min_lr: 4.0e-05              # cosine decay floor
  pretraining.annealing_min_lr: 0.0        # annealing decays to zero
  pretraining.warmup_steps: 2500           # linear warmup duration
  pretraining.total_tokens: 5_133_000_000_000
  pretraining.annealing_tokens: 100_000_000_000
  pretraining.batch_size_tokens: 4_194_304  # 1024 * 4096
  sft.learning_rate: 2.0e-05
  dpo.learning_rate: 5.0e-07
  kto.learning_rate: 5.0e-07
"""

import logging
import math
from typing import Union

from torch.optim import Optimizer

from config import TrainingConfig

logger = logging.getLogger(__name__)


class LRScheduler:
    """Three-phase learning rate scheduler for OLMoE pretraining.

    Implements the exact LR schedule from Table 10 and Appendix B:
      1. Linear warmup from 0 to peak_lr over warmup_steps steps
      2. Cosine decay from peak_lr to min_lr over the main training phase
      3. Linear annealing from min_lr to 0 over the final annealing_steps steps

    This is a manual step-based scheduler (not a torch.optim.lr_scheduler
    subclass) because the three-phase logic with token-based annealing
    detection is cleaner to implement directly. The caller passes the global
    training step to step() on every iteration.

    The scheduler is stateless with respect to the step count — it computes
    LR purely from the passed current_step argument. This makes checkpoint
    resume trivial: after loading a checkpoint at step N, call step(N) to
    restore the correct LR before continuing training.

    Phase boundaries (derived from config.yaml pretraining section):
        warmup_steps      = 2500
        max_steps         = total_tokens // batch_size_tokens ≈ 1,223,958
        annealing_steps   = annealing_tokens // batch_size_tokens ≈ 23,842
        cosine_end_step   = max_steps - annealing_steps ≈ 1,200,116

    LR values at key steps:
        step 0:              lr = 0.0
        step 1250:           lr = 2e-4  (warmup midpoint)
        step 2500:           lr = 4e-4  (peak, cosine start)
        step ~613,000:       lr ≈ 2.2e-4 (cosine midpoint)
        step ~1,200,116:     lr = 4e-5  (cosine end / annealing start)
        step ~1,211,987:     lr ≈ 2e-5  (annealing midpoint)
        step ~1,223,958:     lr = 0.0   (training end)

    Attributes:
        optimizer: The optimizer whose param_group LRs are updated.
        warmup_steps: Number of linear warmup steps (2500).
        peak_lr: Maximum learning rate reached after warmup (4e-4).
        min_lr: Minimum LR at end of cosine phase / start of annealing (4e-5).
        annealing_min_lr: LR at end of annealing phase (0.0).
        max_steps: Total training steps derived from total_tokens / batch_size_tokens.
        annealing_steps: Steps in the annealing phase.
        cosine_end_step: Step at which cosine phase ends and annealing begins.

    Example:
        >>> config = TrainingConfig()
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        >>> scheduler = LRScheduler(optimizer, config)
        >>> for step in range(config.max_steps):
        ...     scheduler.step(step)
        ...     loss.backward()
        ...     optimizer.step()
    """

    def __init__(self, optimizer: Optimizer, config: TrainingConfig) -> None:
        """Initialize LRScheduler from TrainingConfig.

        Derives all step counts from token counts and batch size to ensure
        consistency with the training loop. All values are read from config
        at construction time — no hardcoded constants.

        Args:
            optimizer: The AdamW optimizer whose param_group LRs will be
                       updated on each call to step(). Must already have
                       param groups configured (e.g., via create_optimizer()).
            config: TrainingConfig instance. Key fields used:
                    - learning_rate (4e-4): peak LR after warmup
                    - min_lr (4e-5): floor for cosine decay
                    - annealing_min_lr (0.0): floor for linear annealing
                    - warmup_steps (2500): linear warmup duration
                    - total_tokens (5.133T): total pretraining tokens
                    - annealing_tokens (100B): tokens in annealing phase
                    - batch_size_tokens (~4M): tokens per training step

        Raises:
            ValueError: If warmup_steps >= max_steps, or if annealing_steps
                        >= (max_steps - warmup_steps), indicating degenerate
                        schedule configuration.
        """
        self.optimizer: Optimizer = optimizer

        # -----------------------------------------------------------------------
        # LR values from config.yaml (pretraining section)
        # -----------------------------------------------------------------------
        self.peak_lr: float = config.learning_rate
        """Peak learning rate = 4e-4 (config.yaml: pretraining.learning_rate)."""

        self.min_lr: float = config.min_lr
        """Cosine decay floor = 4e-5 (config.yaml: pretraining.min_lr)."""

        self.annealing_min_lr: float = config.annealing_min_lr
        """Annealing floor = 0.0 (config.yaml: pretraining.annealing_min_lr)."""

        self.warmup_steps: int = config.warmup_steps
        """Linear warmup steps = 2500 (config.yaml: pretraining.warmup_steps)."""

        # -----------------------------------------------------------------------
        # Derive step counts from token counts.
        #
        # Use integer division (//) to get exact step counts. The slight
        # rounding (e.g., 1,223,958 vs exact) is acceptable and consistent
        # with how the training loop counts steps.
        #
        # config.max_steps and config.annealing_steps are pre-computed in
        # TrainingConfig.__post_init__, so we use them directly.
        # -----------------------------------------------------------------------
        self.max_steps: int = config.max_steps
        """Total training steps ≈ 1,223,958 (total_tokens // batch_size_tokens)."""

        self.annealing_steps: int = config.annealing_steps
        """Annealing phase steps ≈ 23,842 (annealing_tokens // batch_size_tokens)."""

        # -----------------------------------------------------------------------
        # Compute phase boundary: step at which cosine ends and annealing begins.
        # -----------------------------------------------------------------------
        self.cosine_end_step: int = self.max_steps - self.annealing_steps
        """Step at which cosine decay ends and linear annealing begins."""

        # -----------------------------------------------------------------------
        # Validate schedule configuration.
        # -----------------------------------------------------------------------
        if self.warmup_steps < 0:
            raise ValueError(
                f"warmup_steps must be >= 0, got {self.warmup_steps}."
            )
        if self.max_steps <= 0:
            raise ValueError(
                f"max_steps must be > 0, got {self.max_steps}. "
                f"Check total_tokens and batch_size_tokens in config."
            )
        if self.annealing_steps < 0:
            raise ValueError(
                f"annealing_steps must be >= 0, got {self.annealing_steps}."
            )
        if self.cosine_end_step <= self.warmup_steps and self.cosine_end_step > 0:
            raise ValueError(
                f"cosine_end_step ({self.cosine_end_step}) must be > warmup_steps "
                f"({self.warmup_steps}). The cosine phase has zero or negative "
                f"duration. Check total_tokens, annealing_tokens, and warmup_steps."
            )
        if self.peak_lr <= 0:
            raise ValueError(
                f"peak_lr must be > 0, got {self.peak_lr}."
            )
        if self.min_lr < 0:
            raise ValueError(
                f"min_lr must be >= 0, got {self.min_lr}."
            )
        if self.min_lr > self.peak_lr:
            raise ValueError(
                f"min_lr ({self.min_lr}) must be <= peak_lr ({self.peak_lr})."
            )
        if self.annealing_min_lr < 0:
            raise ValueError(
                f"annealing_min_lr must be >= 0, got {self.annealing_min_lr}."
            )
        if self.annealing_min_lr > self.min_lr:
            raise ValueError(
                f"annealing_min_lr ({self.annealing_min_lr}) must be <= "
                f"min_lr ({self.min_lr})."
            )

        logger.info(
            f"LRScheduler initialized: "
            f"warmup_steps={self.warmup_steps}, "
            f"peak_lr={self.peak_lr:.2e}, "
            f"min_lr={self.min_lr:.2e}, "
            f"annealing_min_lr={self.annealing_min_lr:.2e}, "
            f"max_steps={self.max_steps:,}, "
            f"annealing_steps={self.annealing_steps:,}, "
            f"cosine_end_step={self.cosine_end_step:,}"
        )

    def get_lr(self, current_step: int) -> float:
        """Compute the learning rate for the given training step.

        Implements the three-phase schedule from Table 10 and Appendix B:

        Phase 1 — Linear Warmup [0, warmup_steps):
            lr = peak_lr * (current_step / warmup_steps)
            Linearly ramps from 0 at step 0 to peak_lr at step warmup_steps.
            If warmup_steps == 0, this phase is skipped (returns peak_lr).

        Phase 2 — Cosine Decay [warmup_steps, cosine_end_step):
            progress = (current_step - warmup_steps) / (cosine_end_step - warmup_steps)
            lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + cos(π * progress))
            Smoothly decays from peak_lr to min_lr using a cosine curve.
            At progress=0: lr = peak_lr. At progress=1: lr = min_lr.

        Phase 3 — Linear Annealing [cosine_end_step, max_steps]:
            progress = (current_step - cosine_end_step) / annealing_steps
            lr = min_lr + (annealing_min_lr - min_lr) * progress
            Linearly decays from min_lr to annealing_min_lr=0.
            At progress=0: lr = min_lr. At progress=1: lr = 0.

        After max_steps: returns 0.0 (training is complete).

        Args:
            current_step: The current global training step (0-indexed).
                          Step 0 is the first training step before any
                          parameter update.

        Returns:
            Learning rate as a float for the given step. Always in the range
            [annealing_min_lr, peak_lr] = [0.0, 4e-4] for OLMoE-1B-7B.

        Examples:
            >>> scheduler.get_lr(0)
            0.0
            >>> scheduler.get_lr(1250)  # warmup midpoint
            0.0002
            >>> scheduler.get_lr(2500)  # peak
            0.0004
            >>> scheduler.get_lr(scheduler.max_steps)
            0.0
        """
        # -----------------------------------------------------------------------
        # After training is complete: return 0.0.
        # This handles the edge case where the training loop runs one extra step.
        # -----------------------------------------------------------------------
        if current_step >= self.max_steps:
            return 0.0

        # -----------------------------------------------------------------------
        # Phase 1: Linear Warmup [0, warmup_steps)
        #
        # lr = peak_lr * (current_step / warmup_steps)
        #
        # At step 0: lr = 0.0 (training starts with zero LR)
        # At step warmup_steps - 1: lr ≈ peak_lr (just below peak)
        # At step warmup_steps: transitions to cosine phase at peak_lr
        #
        # Edge case: warmup_steps == 0 means no warmup phase.
        # In this case, we skip directly to cosine decay starting at peak_lr.
        # -----------------------------------------------------------------------
        if self.warmup_steps > 0 and current_step < self.warmup_steps:
            # Linear interpolation from 0 to peak_lr.
            lr: float = self.peak_lr * (current_step / self.warmup_steps)
            return lr

        # -----------------------------------------------------------------------
        # Phase 2: Cosine Decay [warmup_steps, cosine_end_step)
        #
        # Standard cosine annealing formula:
        #   progress = (step - warmup_steps) / (cosine_end_step - warmup_steps)
        #   lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + cos(π * progress))
        #
        # At progress=0 (step=warmup_steps):
        #   lr = min_lr + 0.5*(peak_lr-min_lr)*(1+cos(0)) = min_lr + (peak_lr-min_lr) = peak_lr ✓
        # At progress=1 (step=cosine_end_step):
        #   lr = min_lr + 0.5*(peak_lr-min_lr)*(1+cos(π)) = min_lr + 0 = min_lr ✓
        #
        # Guard: if cosine_end_step == warmup_steps (degenerate, no cosine phase),
        # skip directly to annealing. This won't happen with the paper's config
        # but is defensive practice.
        # -----------------------------------------------------------------------
        if current_step < self.cosine_end_step:
            cosine_duration: int = self.cosine_end_step - self.warmup_steps

            if cosine_duration <= 0:
                # Degenerate case: no cosine phase, jump to min_lr.
                return self.min_lr

            # Compute progress in [0, 1] through the cosine phase.
            # Clamp to [0, 1] to handle any floating-point boundary issues.
            progress: float = (current_step - self.warmup_steps) / cosine_duration
            progress = max(0.0, min(1.0, progress))

            # Cosine annealing: smooth decay from peak_lr to min_lr.
            lr = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (
                1.0 + math.cos(math.pi * progress)
            )
            return lr

        # -----------------------------------------------------------------------
        # Phase 3: Linear Annealing [cosine_end_step, max_steps]
        #
        # Linearly decays from min_lr to annealing_min_lr=0 over annealing_steps.
        #
        # progress = (step - cosine_end_step) / annealing_steps
        # lr = min_lr + (annealing_min_lr - min_lr) * progress
        #    = min_lr * (1 - progress)  [when annealing_min_lr = 0]
        #
        # At progress=0 (step=cosine_end_step): lr = min_lr = 4e-5 ✓
        # At progress=1 (step=max_steps):       lr = 0.0 ✓
        #
        # This matches Appendix B: "linearly decay the learning rate to 0"
        # during the final 100B tokens.
        #
        # Guard: if annealing_steps == 0 (degenerate), return annealing_min_lr.
        # -----------------------------------------------------------------------
        if self.annealing_steps <= 0:
            return self.annealing_min_lr

        # Compute progress in [0, 1] through the annealing phase.
        # Clamp to [0, 1] to handle boundary steps cleanly.
        annealing_progress: float = (
            (current_step - self.cosine_end_step) / self.annealing_steps
        )
        annealing_progress = max(0.0, min(1.0, annealing_progress))

        # Linear interpolation from min_lr to annealing_min_lr.
        lr = self.min_lr + (self.annealing_min_lr - self.min_lr) * annealing_progress
        return lr

    def step(self, current_step: int) -> None:
        """Update optimizer learning rate for the given training step.

        Computes the LR for current_step via get_lr() and sets it on all
        parameter groups in the optimizer. This must be called BEFORE the
        optimizer.step() call in the training loop to ensure the correct LR
        is used for the parameter update.

        The scheduler is stateless — it does not maintain an internal step
        counter. The caller (Trainer.train_step) passes the global step
        explicitly. This makes checkpoint resume trivial: after loading a
        checkpoint at step N, call scheduler.step(N) to restore the correct
        LR before continuing training.

        In distributed training (FSDP), the optimizer is already wrapped by
        FSDP. Updating param_group['lr'] propagates correctly through FSDP
        without any special handling.

        Args:
            current_step: The current global training step (0-indexed).
                          Must be in [0, max_steps] for meaningful LR values.
                          Steps beyond max_steps return LR=0.0.
        """
        lr: float = self.get_lr(current_step)

        # Update all parameter groups in the optimizer.
        # In standard AdamW, there is typically one parameter group (all params
        # with weight_decay=0.1). We update all groups for robustness.
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def __repr__(self) -> str:
        """Return string representation of the scheduler configuration.

        Returns:
            Human-readable string showing all schedule parameters.
        """
        return (
            f"LRScheduler("
            f"peak_lr={self.peak_lr:.2e}, "
            f"min_lr={self.min_lr:.2e}, "
            f"annealing_min_lr={self.annealing_min_lr:.2e}, "
            f"warmup_steps={self.warmup_steps:,}, "
            f"cosine_end_step={self.cosine_end_step:,}, "
            f"max_steps={self.max_steps:,}, "
            f"annealing_steps={self.annealing_steps:,}"
            f")"
        )


class ConstantLRScheduler:
    """Constant learning rate scheduler for SFT, DPO, and KTO adaptation.

    Used during instruction tuning (SFT) and preference tuning (DPO/KTO)
    where the learning rate is held constant throughout training, as specified
    in Appendix B of the paper:
        SFT:  constant lr = 2e-5  (config.yaml: sft.learning_rate)
        DPO:  constant lr = 5e-7  (config.yaml: dpo.learning_rate)
        KTO:  constant lr = 5e-7  (config.yaml: kto.learning_rate)

    Provides the same interface as LRScheduler (step() and get_lr()) so that
    all trainer classes can use schedulers interchangeably without type checks.

    The initial LR is set on all optimizer param groups immediately in __init__,
    ensuring the optimizer starts with the correct LR even if step() is not
    called before the first optimizer.step().

    Attributes:
        optimizer: The optimizer whose param_group LRs are updated.
        lr: The constant learning rate value.

    Example:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        >>> scheduler = ConstantLRScheduler(optimizer, lr=2e-5)
        >>> for step in range(num_steps):
        ...     scheduler.step(step)  # No-op, LR stays constant
        ...     loss.backward()
        ...     optimizer.step()
    """

    def __init__(self, optimizer: Optimizer, lr: float) -> None:
        """Initialize ConstantLRScheduler.

        Sets the constant LR on all optimizer param groups immediately.

        Args:
            optimizer: The optimizer whose param_group LRs will be maintained
                       at the constant value lr.
            lr: The constant learning rate to use throughout training.
                For SFT: 2e-5 (config.yaml: sft.learning_rate)
                For DPO: 5e-7 (config.yaml: dpo.learning_rate)
                For KTO: 5e-7 (config.yaml: kto.learning_rate)

        Raises:
            ValueError: If lr is not positive.
        """
        if lr <= 0:
            raise ValueError(
                f"lr must be > 0 for ConstantLRScheduler, got {lr}. "
                f"Check sft.learning_rate, dpo.learning_rate, or kto.learning_rate "
                f"in config.yaml."
            )

        self.optimizer: Optimizer = optimizer
        self.lr: float = lr

        # Set the initial LR on all param groups immediately.
        # This ensures the optimizer starts with the correct LR even if
        # step() is not called before the first optimizer.step().
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr

        logger.info(
            f"ConstantLRScheduler initialized: lr={self.lr:.2e}"
        )

    def get_lr(self, current_step: int) -> float:
        """Return the constant learning rate (independent of step).

        Args:
            current_step: The current training step. Ignored — LR is constant.

        Returns:
            The constant learning rate self.lr.
        """
        return self.lr

    def step(self, current_step: int) -> None:
        """Update optimizer LR (no-op for constant schedule, but re-sets for safety).

        Re-sets the LR on all param groups to self.lr. This is technically a
        no-op since the LR never changes, but re-setting ensures correctness
        if the optimizer's param groups were modified externally (e.g., by
        gradient checkpointing or FSDP internals).

        Args:
            current_step: The current training step. Ignored — LR is constant.
        """
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr

    def __repr__(self) -> str:
        """Return string representation of the scheduler.

        Returns:
            Human-readable string showing the constant LR value.
        """
        return f"ConstantLRScheduler(lr={self.lr:.2e})"


def create_scheduler(
    optimizer: Optimizer,
    config: TrainingConfig,
    mode: str = "pretrain",
    sft_lr: float = 2.0e-05,
    dpo_lr: float = 5.0e-07,
    kto_lr: float = 5.0e-07,
) -> Union[LRScheduler, ConstantLRScheduler]:
    """Factory function to create the appropriate LR scheduler for a training mode.

    Provides a uniform interface for all trainer classes — they call
    scheduler.step(global_step) regardless of the training mode.

    Supported modes and their schedulers:
        'pretrain': Three-phase schedule (warmup → cosine → annealing)
                    Uses all hyperparameters from TrainingConfig.
                    LRScheduler(optimizer, config)

        'sft':      Constant LR = 2e-5 throughout SFT training.
                    ConstantLRScheduler(optimizer, lr=sft_lr)
                    (config.yaml: sft.learning_rate = 2.0e-05)

        'dpo':      Constant LR = 5e-7 throughout DPO training.
                    ConstantLRScheduler(optimizer, lr=dpo_lr)
                    (config.yaml: dpo.learning_rate = 5.0e-07)

        'kto':      Constant LR = 5e-7 throughout KTO training.
                    ConstantLRScheduler(optimizer, lr=kto_lr)
                    (config.yaml: kto.learning_rate = 5.0e-07)

    Args:
        optimizer: The optimizer to attach the scheduler to. Must already
                   have param groups configured.
        config: TrainingConfig instance. Used for pretraining schedule
                parameters. For adaptation modes, only the lr arguments
                are used (sft_lr, dpo_lr, kto_lr).
        mode: Training mode string. One of: 'pretrain', 'sft', 'dpo', 'kto'.
              Case-insensitive. Defaults to 'pretrain'.
        sft_lr: Constant LR for SFT mode.
                Default: 2e-5 (config.yaml: sft.learning_rate).
        dpo_lr: Constant LR for DPO mode.
                Default: 5e-7 (config.yaml: dpo.learning_rate).
        kto_lr: Constant LR for KTO mode.
                Default: 5e-7 (config.yaml: kto.learning_rate).

    Returns:
        LRScheduler for 'pretrain' mode, or ConstantLRScheduler for
        'sft', 'dpo', and 'kto' modes.

    Raises:
        ValueError: If mode is not one of the supported values.

    Example:
        >>> # Pretraining
        >>> scheduler = create_scheduler(optimizer, config, mode='pretrain')
        >>> isinstance(scheduler, LRScheduler)
        True

        >>> # SFT
        >>> scheduler = create_scheduler(optimizer, config, mode='sft', sft_lr=2e-5)
        >>> isinstance(scheduler, ConstantLRScheduler)
        True
        >>> scheduler.get_lr(0)
        2e-05

        >>> # DPO
        >>> scheduler = create_scheduler(optimizer, config, mode='dpo', dpo_lr=5e-7)
        >>> scheduler.get_lr(1000)
        5e-07
    """
    mode_lower: str = mode.lower()

    if mode_lower == "pretrain":
        logger.info(
            f"Creating LRScheduler for pretraining: "
            f"peak_lr={config.learning_rate:.2e}, "
            f"min_lr={config.min_lr:.2e}, "
            f"warmup_steps={config.warmup_steps}"
        )
        return LRScheduler(optimizer=optimizer, config=config)

    elif mode_lower == "sft":
        logger.info(
            f"Creating ConstantLRScheduler for SFT: lr={sft_lr:.2e}"
        )
        return ConstantLRScheduler(optimizer=optimizer, lr=sft_lr)

    elif mode_lower == "dpo":
        logger.info(
            f"Creating ConstantLRScheduler for DPO: lr={dpo_lr:.2e}"
        )
        return ConstantLRScheduler(optimizer=optimizer, lr=dpo_lr)

    elif mode_lower == "kto":
        logger.info(
            f"Creating ConstantLRScheduler for KTO: lr={kto_lr:.2e}"
        )
        return ConstantLRScheduler(optimizer=optimizer, lr=kto_lr)

    else:
        raise ValueError(
            f"Unsupported scheduler mode: '{mode}'. "
            f"Must be one of: 'pretrain', 'sft', 'dpo', 'kto'. "
            f"Got: '{mode}'."
        )
