## utils/lr_scheduler.py
"""Learning rate scheduling utilities for P2VAE and FMT training.

Implements a cosine decay schedule with linear warmup as specified in the paper
(Section 4.1): 10% linear warmup over total training steps, followed by cosine
decay to zero. Also provides a linear LR scaling function for adjusting the
base learning rate (1e-4 at batch size 256) to different batch sizes.

Both P2VAETrainer and FMTTrainer use CosineWarmupScheduler with identical
schedule shapes but different AdamW betas:
  - P2VAE: betas=(0.9, 0.995), weight_decay=1e-4
  - FMT:   betas=(0.9, 0.95),  weight_decay=0.01

The scheduler itself is agnostic to these differences.
"""

import math
from typing import List

import torch
from torch.optim.lr_scheduler import _LRScheduler


def get_lr_scale(
    base_lr: float,
    batch_size: int,
    base_batch_size: int = 256,
) -> float:
    """Compute linearly scaled learning rate for a given batch size.

    Implements the linear scaling rule: LR ∝ batch_size, calibrated to the
    paper's reference of base_lr=1e-4 at batch_size=256 (config.yaml:
    p2vae.training.base_lr and p2vae.training.base_batch_size).

    This function is called before constructing the AdamW optimizer so the
    optimizer is initialized with the already-scaled LR. The CosineWarmupScheduler
    then modulates this scaled LR via warmup and cosine factors.

    Examples:
        >>> get_lr_scale(1e-4, 256, 256)
        0.0001   # no change at reference batch size
        >>> get_lr_scale(1e-4, 512, 256)
        0.0002   # double batch → double LR
        >>> get_lr_scale(1e-4, 128, 256)
        5e-05    # half batch → half LR

    Args:
        base_lr: Reference learning rate calibrated to base_batch_size.
            From config.yaml: p2vae.training.base_lr = 1e-4.
        batch_size: Actual batch size used in training.
        base_batch_size: Reference batch size for the base_lr calibration.
            From config.yaml: p2vae.training.base_batch_size = 256.

    Returns:
        Scaled learning rate: base_lr * (batch_size / base_batch_size).
    """
    return base_lr * (float(batch_size) / float(base_batch_size))


class CosineWarmupScheduler(_LRScheduler):
    """Cosine LR decay schedule with linear warmup.

    Implements the two-phase schedule described in the paper (Section 4.1):

    Phase 1 — Linear warmup (steps 0 to warmup_steps - 1):
        lr = base_lr * (current_step / warmup_steps)
        Ramps linearly from 0 to base_lr over warmup_steps steps.

    Phase 2 — Cosine decay (steps warmup_steps to total_steps - 1):
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        cosine_factor = 0.5 * (1 + cos(π * progress))
        lr = base_lr * (min_lr_ratio + (1 - min_lr_ratio) * cosine_factor)
        Decays smoothly from base_lr to base_lr * min_lr_ratio.

    With min_lr_ratio=0.0 (from config.yaml: p2vae.training.min_lr_ratio),
    the LR decays all the way to zero at the end of training.

    The scheduler is called once per training step (not per epoch). The parent
    class _LRScheduler uses last_epoch as the step counter; passing last_epoch=-1
    to __init__ triggers an initial step() call that sets the LR to the warmup
    start value (near zero) before the first training step.

    Checkpoint compatibility: The parent's state_dict() / load_state_dict()
    save and restore last_epoch and _last_lr, enabling seamless training
    resumption. Trainers call scheduler.load_state_dict(ckpt['scheduler'])
    to restore the exact LR trajectory.

    Attributes:
        total_steps: Total number of training steps (100k from config).
        warmup_steps: Number of linear warmup steps (10% of total_steps).
        min_lr_ratio: Floor for LR as a fraction of base_lr (0.0 from config).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_ratio: float = 0.1,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """Initialize the cosine warmup scheduler.

        Args:
            optimizer: The AdamW optimizer instance to schedule. The optimizer
                must already be initialized with the scaled LR (via get_lr_scale)
                before being passed here.
            total_steps: Total number of training steps. From config.yaml:
                p2vae.training.total_steps = 100000 and
                fmt.training.total_steps = 100000.
            warmup_ratio: Fraction of total_steps used for linear warmup.
                From config.yaml: p2vae.training.warmup_ratio = 0.1 (10%).
                warmup_steps = int(total_steps * warmup_ratio) = 10000.
            min_lr_ratio: Minimum LR as a fraction of base_lr. The LR floor
                is base_lr * min_lr_ratio. From config.yaml:
                p2vae.training.min_lr_ratio = 0.0 (decays to zero).
            last_epoch: The index of the last completed step. Pass -1 (default)
                for fresh training; pass (resumed_step - 1) when resuming from
                a checkpoint so the parent's first step() call advances to the
                correct step. The parent class increments last_epoch by 1 on
                each step() call, so last_epoch=-1 → first get_lr() call sees
                last_epoch=0.

        Raises:
            ValueError: If total_steps <= 0, warmup_ratio is not in [0, 1],
                or min_lr_ratio is not in [0, 1].
        """
        if total_steps <= 0:
            raise ValueError(
                f"total_steps must be positive, got {total_steps}."
            )
        if not (0.0 <= warmup_ratio <= 1.0):
            raise ValueError(
                f"warmup_ratio must be in [0, 1], got {warmup_ratio}."
            )
        if not (0.0 <= min_lr_ratio <= 1.0):
            raise ValueError(
                f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}."
            )

        self.total_steps: int = total_steps
        self.warmup_steps: int = int(total_steps * warmup_ratio)
        self.min_lr_ratio: float = min_lr_ratio

        # Call parent __init__ last: it immediately calls step() → get_lr(),
        # so all instance attributes must be set before this line.
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> List[float]:
        """Compute the learning rate for each optimizer param group.

        Called automatically by the parent's step() method. Uses self.last_epoch
        as the current step counter (0-indexed). Returns one LR per param group,
        scaling each group's base_lr by the warmup or cosine factor.

        The computation uses math.cos and math.pi (CPU scalars) rather than
        torch operations since this runs on the CPU scheduler thread.

        Returns:
            List of learning rates, one per optimizer param group. Each LR is
            the corresponding base_lr (from self.base_lrs) scaled by the
            warmup or cosine factor appropriate for the current step.
        """
        current_step: int = self.last_epoch

        # Clamp steps beyond the schedule to the minimum LR floor.
        if current_step >= self.total_steps:
            return [base_lr * self.min_lr_ratio for base_lr in self.base_lrs]

        lrs: List[float] = []
        for base_lr in self.base_lrs:
            if self.warmup_steps > 0 and current_step < self.warmup_steps:
                # Phase 1: Linear warmup from 0 to base_lr.
                # At step 0: lr = 0. At step warmup_steps: lr = base_lr.
                lr: float = base_lr * (float(current_step) / float(self.warmup_steps))
            else:
                # Phase 2: Cosine decay from base_lr to base_lr * min_lr_ratio.
                # Guard against degenerate case where warmup covers all steps.
                decay_steps: int = max(self.total_steps - self.warmup_steps, 1)
                decay_elapsed: int = current_step - self.warmup_steps

                # progress ∈ [0, 1]: 0 at start of decay, 1 at end.
                progress: float = float(decay_elapsed) / float(decay_steps)

                # Cosine factor: 1.0 at progress=0, 0.0 at progress=1.
                cosine_factor: float = 0.5 * (1.0 + math.cos(math.pi * progress))

                # Scale between min_lr_ratio and 1.0.
                lr = base_lr * (
                    self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_factor
                )

            # Clamp to the floor to prevent numerical underflow below the minimum.
            lr = max(lr, base_lr * self.min_lr_ratio)
            lrs.append(lr)

        return lrs
