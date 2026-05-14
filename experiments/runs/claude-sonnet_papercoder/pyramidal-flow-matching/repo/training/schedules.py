## training/schedules.py
"""Learning rate scheduling and optimizer construction for Pyramidal Flow Matching.

Provides three public utilities consumed exclusively by Trainer:

1. get_constant_schedule_with_warmup: Linear warmup then constant LR.
   Used in ALL THREE training stages per the paper (Table 4, Appendix B).

2. get_cosine_schedule_with_warmup: Linear warmup then cosine decay.
   Included as a utility for experimentation; not the paper's choice.

3. build_optimizer: Constructs AdamW with stage-specific hyperparameters
   and proper parameter grouping (weight decay excluded for biases/norms).

Paper reference (Table 4, Appendix B):
    All three stages use AdamW with constant LR + linear warmup.
    Stage 1: lr=1e-4, beta2=0.999, warmup=1k, total=50k
    Stage 2: lr=1e-4, beta2=0.95,  warmup=1k, total=200k
    Stage 3: lr=5e-5, beta2=0.95,  warmup=1k, total=50k

Config reference (configs/default.yaml):
    training.stage{1,2,3}.learning_rate
    training.stage{1,2,3}.beta1
    training.stage{1,2,3}.beta2
    training.stage{1,2,3}.eps
    training.stage{1,2,3}.weight_decay
    training.stage{1,2,3}.warmup_steps
    training.stage{1,2,3}.total_steps

Usage:
    from training.schedules import (
        get_constant_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
        build_optimizer,
    )

    # Build optimizer from stage sub-config
    optimizer = build_optimizer(model, stage_config)

    # Build constant-LR-with-warmup scheduler (paper's choice)
    scheduler = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=1000
    )

    # Training loop
    for step, batch in enumerate(dataloader):
        loss = compute_loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
"""

import math
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR

from utils.logging import get_logger

## ---------------------------------------------------------------------------
## Module-level logger
## ---------------------------------------------------------------------------
logger = get_logger(__name__)


## ---------------------------------------------------------------------------
## Learning rate schedule factories
## ---------------------------------------------------------------------------


def get_constant_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int = 1000,
) -> LambdaLR:
    """Creates a constant LR schedule with linear warmup.

    This is the learning rate schedule used in ALL THREE training stages
    of the paper (Table 4, Appendix B):
        "Learning rate schedule: Constant with warmup"

    Schedule behavior:
        - Steps [0, num_warmup_steps): LR scales linearly from 0 to base_lr.
          Multiplier = current_step / max(1, num_warmup_steps)
        - Steps [num_warmup_steps, ∞): LR stays constant at base_lr.
          Multiplier = 1.0

    The base LR is set in the optimizer (via build_optimizer) and this
    scheduler multiplies it by the returned scalar. So the actual LR at
    step t is: actual_lr = base_lr * lr_lambda(t).

    Args:
        optimizer: The AdamW optimizer to attach the schedule to.
            Constructed by build_optimizer() with the stage-specific base LR.
        num_warmup_steps: Number of linear warmup steps. From config:
            - Stage 1: config.training.stage1.warmup_steps = 1000
            - Stage 2: config.training.stage2.warmup_steps = 1000
            - Stage 3: config.training.stage3.warmup_steps = 1000
            Defaults to 1000 (paper's value for all stages).
            If 0, the schedule is constant at base_lr from step 0 (no warmup).

    Returns:
        A LambdaLR scheduler that applies the warmup-then-constant multiplier
        to all parameter groups in the optimizer.

    Example:
        >>> optimizer = build_optimizer(model, stage1_config)
        >>> scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=1000)
        >>> for step in range(50000):
        ...     loss.backward()
        ...     optimizer.step()
        ...     scheduler.step()
        ...     optimizer.zero_grad()
    """
    # Validate inputs
    if num_warmup_steps < 0:
        raise ValueError(
            f"num_warmup_steps must be >= 0, got num_warmup_steps={num_warmup_steps}. "
            f"Set to 0 to disable warmup."
        )

    def lr_lambda(current_step: int) -> float:
        """Computes the LR multiplier for the given training step.

        Args:
            current_step: Current global training step (0-indexed).
                LambdaLR calls this with the internal step counter.

        Returns:
            Float multiplier in [0.0, 1.0] applied to the base LR.
        """
        # Edge case: no warmup requested → always return 1.0
        if num_warmup_steps == 0:
            return 1.0

        # Warmup phase: linear ramp from 0 to 1
        if current_step < num_warmup_steps:
            # Use max(1, num_warmup_steps) to prevent division by zero
            # when num_warmup_steps=1 and current_step=0
            return float(current_step) / float(max(1, num_warmup_steps))

        # Post-warmup phase: constant at 1.0
        return 1.0

    scheduler: LambdaLR = LambdaLR(optimizer, lr_lambda=lr_lambda)

    logger.info(
        "Constant-with-warmup LR schedule created: "
        "num_warmup_steps=%d",
        num_warmup_steps,
    )

    return scheduler


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int = 1000,
    num_training_steps: int = 50000,
) -> LambdaLR:
    """Creates a cosine decay LR schedule with linear warmup.

    NOT used by the paper (which uses constant LR with warmup for all stages),
    but included as a utility for experimentation and ablation studies.

    Schedule behavior:
        - Steps [0, num_warmup_steps): Linear ramp from 0 to base_lr.
          Multiplier = current_step / max(1, num_warmup_steps)
        - Steps [num_warmup_steps, num_training_steps): Cosine decay from
          base_lr to 0.
          Multiplier = 0.5 * (1 + cos(π * progress))
          where progress = (step - warmup) / max(1, total - warmup)
        - Steps [num_training_steps, ∞): Multiplier = 0.0 (LR reaches zero)

    Edge cases:
        - If num_warmup_steps >= num_training_steps: post-warmup steps use
          constant multiplier 1.0 (no cosine decay possible).
        - If current_step > num_training_steps: multiplier clamped to 0.0.

    Args:
        optimizer: The AdamW optimizer to attach the schedule to.
        num_warmup_steps: Number of linear warmup steps. Defaults to 1000.
            Must be >= 0.
        num_training_steps: Total number of training steps (warmup + decay).
            Defaults to 50000 (Stage 1 and Stage 3 total steps from config).
            Must be > 0.

    Returns:
        A LambdaLR scheduler applying warmup-then-cosine-decay multiplier
        to all parameter groups in the optimizer.

    Raises:
        ValueError: If num_warmup_steps < 0 or num_training_steps <= 0.

    Example:
        >>> optimizer = build_optimizer(model, stage1_config)
        >>> scheduler = get_cosine_schedule_with_warmup(
        ...     optimizer,
        ...     num_warmup_steps=1000,
        ...     num_training_steps=50000,
        ... )
    """
    # Validate inputs
    if num_warmup_steps < 0:
        raise ValueError(
            f"num_warmup_steps must be >= 0, got num_warmup_steps={num_warmup_steps}."
        )
    if num_training_steps <= 0:
        raise ValueError(
            f"num_training_steps must be > 0, got num_training_steps={num_training_steps}."
        )

    def lr_lambda(current_step: int) -> float:
        """Computes the LR multiplier for the given training step.

        Args:
            current_step: Current global training step (0-indexed).

        Returns:
            Float multiplier in [0.0, 1.0] applied to the base LR.
        """
        # Warmup phase: linear ramp from 0 to 1
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Edge case: warmup covers all training steps → stay at 1.0
        if num_warmup_steps >= num_training_steps:
            return 1.0

        # Post-training: LR has decayed to zero
        if current_step >= num_training_steps:
            return 0.0

        # Cosine decay phase
        # progress ∈ [0, 1]: how far through the decay phase we are
        decay_steps: int = num_training_steps - num_warmup_steps
        steps_since_warmup: int = current_step - num_warmup_steps
        progress: float = float(steps_since_warmup) / float(max(1, decay_steps))

        # Cosine annealing: 0.5 * (1 + cos(π * progress))
        # At progress=0: multiplier = 1.0 (full LR)
        # At progress=1: multiplier = 0.0 (zero LR)
        multiplier: float = 0.5 * (1.0 + math.cos(math.pi * progress))

        return multiplier

    scheduler: LambdaLR = LambdaLR(optimizer, lr_lambda=lr_lambda)

    logger.info(
        "Cosine-with-warmup LR schedule created: "
        "num_warmup_steps=%d, num_training_steps=%d",
        num_warmup_steps,
        num_training_steps,
    )

    return scheduler


## ---------------------------------------------------------------------------
## Optimizer construction
## ---------------------------------------------------------------------------


def _get_parameter_groups(
    model: nn.Module,
    weight_decay: float,
) -> List[Dict[str, Any]]:
    """Separates model parameters into weight-decay and no-weight-decay groups.

    Standard AdamW best practice: exclude 1D parameters (biases, LayerNorm
    weights/biases, GroupNorm weights/biases, embedding weights) from weight
    decay. Only apply weight decay to 2D+ weight matrices.

    This is not explicitly stated in the paper but is implied by the AdamW
    optimizer choice and is standard practice for transformer training.

    The separation logic:
        - param.requires_grad == False: skip entirely (frozen text encoders)
        - param.ndim < 2: no weight decay (biases, 1D norm params)
        - param.ndim >= 2: apply weight decay (weight matrices)

    Args:
        model: The nn.Module whose parameters to group. May contain frozen
            sub-modules (e.g., TextEncoders with freeze_encoders=True).
        weight_decay: Weight decay coefficient for the decay group.
            From config: training.stage{N}.weight_decay = 1e-4.

    Returns:
        List of two parameter group dicts:
            [
                {"params": [weight matrices], "weight_decay": weight_decay},
                {"params": [biases, norms], "weight_decay": 0.0},
            ]
        If all parameters are frozen, returns groups with empty param lists.
        PyTorch's AdamW handles empty param groups gracefully.
    """
    # Collect trainable parameters, separated by decay eligibility
    decay_params: List[torch.nn.Parameter] = []
    no_decay_params: List[torch.nn.Parameter] = []

    # Track parameter names for logging (debug only)
    decay_names: List[str] = []
    no_decay_names: List[str] = []

    for name, param in model.named_parameters():
        # Skip frozen parameters (e.g., T5 and CLIP text encoders)
        if not param.requires_grad:
            continue

        if param.ndim < 2:
            # 1D parameters: biases, LayerNorm/GroupNorm weight and bias,
            # embedding weights (1D after flattening), scalar parameters
            no_decay_params.append(param)
            no_decay_names.append(name)
        else:
            # 2D+ parameters: weight matrices in Linear, Conv layers, etc.
            decay_params.append(param)
            decay_names.append(name)

    # Log parameter group sizes for debugging
    total_trainable: int = len(decay_params) + len(no_decay_params)
    total_decay_numel: int = sum(p.numel() for p in decay_params)
    total_no_decay_numel: int = sum(p.numel() for p in no_decay_params)

    logger.info(
        "Parameter groups: "
        "trainable=%d params (%.1f M), "
        "with_decay=%d params (%.1f M), "
        "no_decay=%d params (%.1f M)",
        total_trainable,
        (total_decay_numel + total_no_decay_numel) / 1e6,
        len(decay_params),
        total_decay_numel / 1e6,
        len(no_decay_params),
        total_no_decay_numel / 1e6,
    )

    logger.debug(
        "Weight decay params (first 10): %s",
        decay_names[:10],
    )
    logger.debug(
        "No weight decay params (first 10): %s",
        no_decay_names[:10],
    )

    # Build parameter group dicts for AdamW
    param_groups: List[Dict[str, Any]] = [
        {
            "params": decay_params,
            "weight_decay": float(weight_decay),
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        },
    ]

    return param_groups


def build_optimizer(
    model: nn.Module,
    config: Dict[str, Any],
) -> AdamW:
    """Constructs an AdamW optimizer with stage-specific hyperparameters.

    Reads hyperparameters from the stage sub-config (e.g., config.training.stage1)
    and constructs an AdamW optimizer with proper parameter grouping:
    - Weight matrices (ndim >= 2): subject to weight decay
    - Biases and norm parameters (ndim < 2): no weight decay
    - Frozen parameters (requires_grad=False): excluded entirely

    Paper hyperparameters (Table 4, Appendix B):
        Stage 1: lr=1e-4, beta1=0.9, beta2=0.999, eps=1e-6, wd=1e-4
        Stage 2: lr=1e-4, beta1=0.9, beta2=0.95,  eps=1e-6, wd=1e-4
        Stage 3: lr=5e-5, beta1=0.9, beta2=0.95,  eps=1e-6, wd=1e-4

    Note: gradient clipping (grad_clip=1.0) is NOT applied here. It belongs
    in Trainer.train_step() via torch.nn.utils.clip_grad_norm_().

    Args:
        model: The nn.Module to optimize. Frozen sub-modules (e.g., text
            encoders with freeze_encoders=True) are automatically excluded
            from the optimizer's parameter groups.
        config: Stage-level sub-config dictionary. Expected keys:
            - learning_rate (float): Base learning rate. Default: 1e-4.
              Stage 1: 1e-4, Stage 2: 1e-4, Stage 3: 5e-5.
            - beta1 (float): AdamW beta1. Default: 0.9 (all stages).
            - beta2 (float): AdamW beta2. Default: 0.999.
              Stage 1: 0.999, Stages 2 & 3: 0.95.
            - eps (float): AdamW epsilon. Default: 1e-6 (all stages).
            - weight_decay (float): Weight decay coefficient. Default: 1e-4.
            These keys correspond to configs/default.yaml:
                training.stage1.learning_rate, training.stage1.beta1, etc.

    Returns:
        A configured torch.optim.AdamW optimizer with two parameter groups:
        one with weight decay (weight matrices) and one without (biases/norms).

    Raises:
        ValueError: If learning_rate <= 0, or if beta1/beta2 are outside (0, 1),
            or if eps <= 0, or if weight_decay < 0.

    Example:
        >>> # Build optimizer for Stage 1
        >>> stage1_cfg = dict(config.training.stage1)
        >>> optimizer = build_optimizer(model, stage1_cfg)
        >>> print(optimizer.defaults['lr'])
        0.0001

        >>> # Build optimizer for Stage 3 (reduced LR)
        >>> stage3_cfg = dict(config.training.stage3)
        >>> optimizer = build_optimizer(model, stage3_cfg)
        >>> print(optimizer.defaults['lr'])
        5e-05
    """
    # ----------------------------------------------------------------
    # Parse hyperparameters from stage sub-config with paper defaults
    # ----------------------------------------------------------------
    learning_rate: float = float(config.get("learning_rate", 1.0e-4))
    beta1: float = float(config.get("beta1", 0.9))
    beta2: float = float(config.get("beta2", 0.999))
    eps: float = float(config.get("eps", 1.0e-6))
    weight_decay: float = float(config.get("weight_decay", 1.0e-4))

    # ----------------------------------------------------------------
    # Validate hyperparameters
    # ----------------------------------------------------------------
    if learning_rate <= 0.0:
        raise ValueError(
            f"learning_rate must be > 0, got learning_rate={learning_rate}. "
            f"Check configs/default.yaml training.stage{{N}}.learning_rate."
        )
    if not (0.0 < beta1 < 1.0):
        raise ValueError(
            f"beta1 must be in (0, 1), got beta1={beta1}. "
            f"Check configs/default.yaml training.stage{{N}}.beta1."
        )
    if not (0.0 < beta2 < 1.0):
        raise ValueError(
            f"beta2 must be in (0, 1), got beta2={beta2}. "
            f"Check configs/default.yaml training.stage{{N}}.beta2."
        )
    if eps <= 0.0:
        raise ValueError(
            f"eps must be > 0, got eps={eps}. "
            f"Check configs/default.yaml training.stage{{N}}.eps."
        )
    if weight_decay < 0.0:
        raise ValueError(
            f"weight_decay must be >= 0, got weight_decay={weight_decay}. "
            f"Check configs/default.yaml training.stage{{N}}.weight_decay."
        )

    # ----------------------------------------------------------------
    # Build parameter groups with weight decay separation
    # ----------------------------------------------------------------
    param_groups: List[Dict[str, Any]] = _get_parameter_groups(
        model=model,
        weight_decay=weight_decay,
    )

    # ----------------------------------------------------------------
    # Construct AdamW optimizer
    # ----------------------------------------------------------------
    # Note: weight_decay in AdamW defaults is overridden per-group above.
    # We set it to 0.0 in defaults to avoid double-applying decay.
    # The per-group weight_decay values take precedence over the default.
    optimizer: AdamW = AdamW(
        params=param_groups,
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=0.0,  # Per-group weight_decay overrides this default
    )

    # Log optimizer configuration for reproducibility
    logger.info(
        "AdamW optimizer built: lr=%.2e, betas=(%.3f, %.3f), "
        "eps=%.2e, weight_decay=%.2e",
        learning_rate,
        beta1,
        beta2,
        eps,
        weight_decay,
    )

    return optimizer


## ---------------------------------------------------------------------------
## Convenience factory: build optimizer + scheduler together
## ---------------------------------------------------------------------------


def build_optimizer_and_scheduler(
    model: nn.Module,
    stage_config: Dict[str, Any],
    schedule_type: str = "constant_with_warmup",
) -> Tuple[AdamW, LambdaLR]:
    """Convenience factory that builds both optimizer and scheduler together.

    Constructs the AdamW optimizer and the appropriate LR scheduler for a
    given training stage in a single call. This is the primary entry point
    used by Trainer.__init__() to set up optimization for each stage.

    Args:
        model: The nn.Module to optimize. Frozen sub-modules are excluded.
        stage_config: Stage-level sub-config dictionary. Expected keys:
            - learning_rate (float): Base LR. Default: 1e-4.
            - beta1 (float): AdamW beta1. Default: 0.9.
            - beta2 (float): AdamW beta2. Default: 0.999.
            - eps (float): AdamW epsilon. Default: 1e-6.
            - weight_decay (float): Weight decay. Default: 1e-4.
            - warmup_steps (int): Warmup steps. Default: 1000.
            - total_steps (int): Total training steps. Default: 50000.
              Only used when schedule_type="cosine_with_warmup".
        schedule_type: Type of LR schedule to use. One of:
            - "constant_with_warmup" (default): Paper's choice for all stages.
              Uses get_constant_schedule_with_warmup().
            - "cosine_with_warmup": Cosine decay after warmup.
              Uses get_cosine_schedule_with_warmup().

    Returns:
        Tuple (optimizer, scheduler) where:
            - optimizer: Configured AdamW with parameter groups.
            - scheduler: LambdaLR with the requested schedule.

    Raises:
        ValueError: If schedule_type is not one of the supported values.

    Example:
        >>> # Stage 1 setup (paper's configuration)
        >>> stage1_cfg = {
        ...     'learning_rate': 1e-4,
        ...     'beta1': 0.9,
        ...     'beta2': 0.999,
        ...     'eps': 1e-6,
        ...     'weight_decay': 1e-4,
        ...     'warmup_steps': 1000,
        ...     'total_steps': 50000,
        ... }
        >>> optimizer, scheduler = build_optimizer_and_scheduler(
        ...     model, stage1_cfg, schedule_type="constant_with_warmup"
        ... )

        >>> # Stage 3 setup (reduced LR, same schedule type)
        >>> stage3_cfg = {
        ...     'learning_rate': 5e-5,
        ...     'beta1': 0.9,
        ...     'beta2': 0.95,
        ...     'eps': 1e-6,
        ...     'weight_decay': 1e-4,
        ...     'warmup_steps': 1000,
        ...     'total_steps': 50000,
        ... }
        >>> optimizer, scheduler = build_optimizer_and_scheduler(
        ...     model, stage3_cfg, schedule_type="constant_with_warmup"
        ... )
    """
    # Validate schedule_type
    supported_schedules: Tuple[str, ...] = (
        "constant_with_warmup",
        "cosine_with_warmup",
    )
    if schedule_type not in supported_schedules:
        raise ValueError(
            f"Unsupported schedule_type: {schedule_type!r}. "
            f"Must be one of: {supported_schedules}. "
            f"The paper uses 'constant_with_warmup' for all three stages "
            f"(Table 4, Appendix B)."
        )

    # Parse schedule-specific hyperparameters
    num_warmup_steps: int = int(stage_config.get("warmup_steps", 1000))
    num_training_steps: int = int(stage_config.get("total_steps", 50000))

    # Build optimizer
    optimizer: AdamW = build_optimizer(model=model, config=stage_config)

    # Build scheduler
    scheduler: LambdaLR
    if schedule_type == "constant_with_warmup":
        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
        )
    else:
        # schedule_type == "cosine_with_warmup"
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    logger.info(
        "Optimizer and scheduler built: schedule_type=%s, "
        "warmup_steps=%d, total_steps=%d",
        schedule_type,
        num_warmup_steps,
        num_training_steps,
    )

    return optimizer, scheduler
