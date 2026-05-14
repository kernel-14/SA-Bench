```python
## training/optimizer.py
"""Optimizer and learning rate scheduler factory functions for SAM 2 training.

This module provides two factory functions:
    - build_optimizer(): AdamW with layer-wise LR decay on the Hiera image encoder
    - build_scheduler(): Reciprocal sqrt LR schedule with linear warmup and cooldown

Both functions implement the exact hyperparameters from Table 12 and Appendix D.2
of the SAM 2 paper.

Config references (config.yaml):
    pretrain.optimizer.learning_rate: 4.0e-4
    pretrain.optimizer.weight_decay: 0.1
    pretrain.optimizer.beta1: 0.9
    pretrain.optimizer.beta2: 0.999
    pretrain.optimizer.gradient_clip_type: "l2"
    pretrain.optimizer.gradient_clip_max: 0.1
    pretrain.layer_wise_decay.{encoder_type}: 0.8 / 0.9 / 0.925
    pretrain.scheduler.timescale: 1000
    pretrain.scheduler.warmup_iters: 1000
    pretrain.scheduler.cooldown_iters: 5000
    pretrain.steps: 90000
    training.num_iterations: 200000
    finetuning.num_iterations: 50000
    finetuning.learning_rate_multiplier: 0.5
    model.image_encoder_type: "hiera_b_plus"

Paper references:
    Table 12: "optimizer: AdamW, β1=0.9, β2=0.999, weight_decay=0.1,
        lr=4e-4, lr_schedule: reciprocal_sqrt timescale=1000,
        warmup: linear 1k iters, cooldown: linear 5k iters,
        layer-wise decay: 0.8 (T,S), 0.9 (B+), 0.925 (L)"
    Appendix D.2.1: "apply layer decay (Clark et al., 2020) on the image
        encoder and follow a reciprocal square-root schedule (Zhai et al., 2022)"
"""

import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default hyperparameters from config.yaml (Table 12)
# ---------------------------------------------------------------------------

# Default base learning rate (config: pretrain.optimizer.learning_rate)
_DEFAULT_BASE_LR: float = 4.0e-4

# Default weight decay (config: pretrain.optimizer.weight_decay)
_DEFAULT_WEIGHT_DECAY: float = 0.1

# Default AdamW betas (config: pretrain.optimizer.beta1/beta2)
_DEFAULT_BETA1: float = 0.9
_DEFAULT_BETA2: float = 0.999

# Default gradient clipping (config: pretrain.optimizer.gradient_clip_*)
_DEFAULT_GRAD_CLIP_TYPE: str = "l2"
_DEFAULT_GRAD_CLIP_MAX: float = 0.1

# Default scheduler parameters (config: pretrain.scheduler.*)
_DEFAULT_TIMESCALE: int = 1000
_DEFAULT_WARMUP_ITERS: int = 1000
_DEFAULT_COOLDOWN_ITERS: int = 5000

# Layer-wise decay rates per encoder size (config: pretrain.layer_wise_decay.*)
_LAYER_DECAY_RATES: Dict[str, float] = {
    "hiera_t":      0.8,
    "hiera_s":      0.8,
    "hiera_b_plus": 0.9,
    "hiera_l":      0.925,
}

# Approximate total Hiera block counts per encoder size.
# Used as fallback when the model is not available for runtime inspection.
# Derived from global attention block indices in config.yaml:
#   hiera_t: global at [5,7,9] → ~10 blocks
#   hiera_s: global at [7,10,13] → ~14 blocks
#   hiera_b_plus: global at [12,16,20] → ~21 blocks
#   hiera_l: global at [23,33,43] → ~44 blocks
_HIERA_APPROX_BLOCK_COUNTS: Dict[str, int] = {
    "hiera_t":      10,
    "hiera_s":      14,
    "hiera_b_plus": 21,
    "hiera_l":      44,
}

# Parameter name patterns that should NOT receive weight decay
# (bias terms and normalization layer parameters)
_NO_WEIGHT_DECAY_PATTERNS: Tuple[str, ...] = (
    "bias",
    "norm.weight",
    "norm.bias",
    "ln.weight",
    "ln.bias",
    "layer_norm.weight",
    "layer_norm.bias",
    "layernorm.weight",
    "layernorm.bias",
    "bn.weight",
    "bn.bias",
    "batch_norm.weight",
    "batch_norm.bias",
    "group_norm.weight",
    "group_norm.bias",
    # Positional embeddings and learned embeddings should not be decayed
    "pos_embed",
    "positional_encoding",
    "temporal_pe",
    "occlusion_embedding",
    "no_mask_embed",
    "not_a_point_embed",
    "point_embeddings",
    "iou_token",
    "mask_tokens",
    "occlusion_token",
)


# ---------------------------------------------------------------------------
# Private helper: layer ID assignment for Hiera backbone
# ---------------------------------------------------------------------------


def _get_layer_id_for_hiera(param_name: str, num_layers: int) -> int:
    """Map a Hiera backbone parameter name to its layer depth index.

    Used to assign each parameter to the correct LR decay group. The depth
    index determines the LR multiplier: `decay_rate ^ (num_layers - depth)`.

    Naming conventions in Hiera (timm implementation):
        - `patch_embed.*`  → depth 0 (deepest decay, smallest LR)
        - `blocks.{i}.*`   → depth i + 1
        - `norm.*`         → depth num_layers (no decay, full LR)
        - `head.*`         → depth num_layers (no decay, full LR)
        - `fpn.*`          → depth num_layers (no decay, full LR)
        - anything else    → depth num_layers (no decay, full LR)

    Args:
        param_name: Dot-separated parameter name relative to the backbone
            module (e.g., "blocks.3.attn.qkv.weight").
        num_layers: Total number of transformer blocks in the backbone.
            Parameters at this depth receive no decay (multiplier = 1.0).

    Returns:
        Integer depth index in [0, num_layers]. Depth 0 receives the most
        decay; depth num_layers receives no decay.
    """
    # Patch embedding: deepest decay
    if "patch_embed" in param_name:
        return 0

    # Transformer blocks: depth = block_index + 1
    if "blocks." in param_name:
        # Extract block index from name like "blocks.3.attn.qkv.weight"
        parts: List[str] = param_name.split("blocks.")
        if len(parts) >= 2:
            block_part: str = parts[1]
            block_idx_str: str = block_part.split(".")[0]
            try:
                block_idx: int = int(block_idx_str)
                # Clamp to valid range [1, num_layers]
                return min(block_idx + 1, num_layers)
            except ValueError:
                pass

    # All other parameters (norm, head, fpn, positional embeddings, etc.)
    # receive no layer-wise decay
    return num_layers


def _should_skip_weight_decay(param_name: str) -> bool:
    """Determine whether a parameter should be excluded from weight decay.

    Excludes bias terms, normalization layer parameters, and learned
    embedding parameters from weight decay. This is standard practice for
    AdamW with vision transformers and is consistent with SAM's training.

    Args:
        param_name: Full dot-separated parameter name.

    Returns:
        True if the parameter should NOT receive weight decay (weight_decay=0).
        False if the parameter should receive the configured weight decay.
    """
    param_name_lower: str = param_name.lower()
    for pattern in _NO_WEIGHT_DECAY_PATTERNS:
        if pattern.lower() in param_name_lower:
            return True
    return False


# ---------------------------------------------------------------------------
# Private helper: parameter group construction
# ---------------------------------------------------------------------------


def _get_param_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    encoder_type: str,
    layer_decay_rates: Dict[str, float],
    lr_multiplier: float = 1.0,
) -> List[Dict[str, Any]]:
    """Construct optimizer parameter groups with per-layer LR for the image encoder.

    Builds separate parameter groups for:
        1. Each layer depth of the image encoder backbone (with LR decay)
        2. All other model parameters (full base LR, no layer decay)

    Within each group, parameters are further split into:
        - Parameters with weight decay (weight_decay > 0)
        - Parameters without weight decay (bias, norm, embeddings)

    Args:
        model: The SAM2Model instance (or any nn.Module with an
            image_encoder.backbone submodule).
        base_lr: Base learning rate before any decay or multiplier.
        weight_decay: Weight decay coefficient for eligible parameters.
        encoder_type: Hiera encoder variant name (e.g., "hiera_b_plus").
            Used to look up the layer decay rate.
        layer_decay_rates: Dict mapping encoder type to decay rate.
            From config: pretrain.layer_wise_decay.
        lr_multiplier: Global LR multiplier applied to all groups.
            Use 0.5 for fine-tuning (config: finetuning.learning_rate_multiplier).

    Returns:
        List of parameter group dicts, each containing:
            - "params": List of parameter tensors
            - "lr": Effective learning rate for this group
            - "weight_decay": Weight decay for this group
            - "name": Human-readable group name (for logging/debugging)

    Note:
        Parameters with requires_grad=False are automatically excluded.
        This handles the frozen image encoder during fine-tuning.
    """
    # Get the layer decay rate for this encoder type
    decay_rate: float = layer_decay_rates.get(encoder_type, 0.9)

    # Determine the number of Hiera backbone layers at runtime
    # Fall back to approximate counts if backbone is not accessible
    num_backbone_layers: int = _HIERA_APPROX_BLOCK_COUNTS.get(encoder_type, 21)

    # Try to get the actual block count from the model
    try:
        backbone = model.image_encoder.backbone  # type: ignore[attr-defined]
        if hasattr(backbone, "blocks"):
            num_backbone_layers = len(backbone.blocks)
            logger.debug(
                "_get_param_groups: Detected %d Hiera backbone blocks for %s.",
                num_backbone_layers,
                encoder_type,
            )
    except AttributeError:
        logger.debug(
            "_get_param_groups: Could not access image_encoder.backbone.blocks. "
            "Using approximate block count %d for %s.",
            num_backbone_layers,
            encoder_type,
        )

    # Total depth levels: 0 (patch_embed) through num_backbone_layers (head/norm)
    # That's num_backbone_layers + 1 distinct depth levels
    num_depth_levels: int = num_backbone_layers + 1

    # ------------------------------------------------------------------
    # Collect image encoder backbone parameters by depth level
    # ------------------------------------------------------------------
    # Structure: depth_level -> {"decay": [params], "no_decay": [params]}
    encoder_param_groups: Dict[int, Dict[str, List[nn.Parameter]]] = {
        depth: {"decay": [], "no_decay": []}
        for depth in range(num_depth_levels)
    }

    # Track which parameters have been assigned to encoder groups
    encoder_param_ids: Set[int] = set()

    # Check if the model has an image encoder with a backbone
    has_backbone: bool = False
    try:
        backbone_module = model.image_encoder.backbone  # type: ignore[attr-defined]
        has_backbone = True
    except AttributeError:
        backbone_module = None
        logger.debug(
            "_get_param_groups: model.image_encoder.backbone not found. "
            "All parameters will use base LR."
        )

    if has_backbone and backbone_module is not None:
        for param_name, param in backbone_module.named_parameters():
            if not param.requires_grad:
                continue

            # Assign to depth level
            depth: int = _get_layer_id_for_hiera(param_name, num_backbone_layers)
            depth = min(depth, num_backbone_layers)  # clamp to valid range

            # Determine weight decay eligibility
            if _should_skip_weight_decay(param_name):
                encoder_param_groups[depth]["no_decay"].append(param)
            else:
                encoder_param_groups[depth]["decay"].append(param)

            encoder_param_ids.add(id(param))

    # Also handle FPN and other image_encoder submodules (not backbone)
    # These get the full base LR (no layer-wise decay)
    encoder_non_backbone_decay: List[nn.Parameter] = []
    encoder_non_backbone_no_decay: List[nn.Parameter] = []

    try:
        image_encoder_module = model.image_encoder  # type: ignore[attr-defined]
        for param_name, param in image_encoder_module.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) in encoder_param_ids:
                continue  # Already assigned to backbone groups

            if _should_skip_weight_decay(param_name):
                encoder_non_backbone_no_decay.append(param)
            else:
                encoder_non_backbone_decay.append(param)

            encoder_param_ids.add(id(param))
    except AttributeError:
        pass

    # ------------------------------------------------------------------
    # Collect all other model parameters (non-encoder)
    # ------------------------------------------------------------------
    other_decay: List[nn.Parameter] = []
    other_no_decay: List[nn.Parameter] = []

    for param_name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in encoder_param_ids:
            continue  # Already assigned to encoder groups

        if _should_skip_weight_decay(param_name):
            other_no_decay.append(param)
        else:
            other_decay.append(param)

    # ------------------------------------------------------------------
    # Build parameter group list
    # ------------------------------------------------------------------
    param_groups: List[Dict[str, Any]] = []

    # Effective base LR with global multiplier
    effective_base_lr: float = base_lr * lr_multiplier

    # --- Image encoder backbone groups (with layer-wise decay) ---
    for depth in range(num_depth_levels):
        # LR multiplier for this depth:
        # depth 0 (patch_embed): decay_rate ^ num_backbone_layers (most decay)
        # depth num_backbone_layers (head/norm): decay_rate ^ 0 = 1.0 (no decay)
        exponent: int = num_backbone_layers - depth
        lr_scale: float = decay_rate ** exponent
        group_lr: float = effective_base_lr * lr_scale

        decay_params: List[nn.Parameter] = encoder_param_groups[depth]["decay"]
        no_decay_params: List[nn.Parameter] = encoder_param_groups[depth]["no_decay"]

        if decay_params:
            param_groups.append({
                "params": decay_params,
                "lr": group_lr,
                "weight_decay": weight_decay,
                "name": f"encoder_backbone_depth{depth}_decay",
            })

        if no_decay_params:
            param_groups.append({
                "params": no_decay_params,
                "lr": group_lr,
                "weight_decay": 0.0,
                "name": f"encoder_backbone_depth{depth}_no_decay",
            })

    # --- Image encoder non-backbone (FPN, etc.) at full base LR ---
    if encoder_non_backbone_decay:
        param_groups.append({
            "params": encoder_non_backbone_decay,
            "lr": effective_base_lr,
            "weight_decay": weight_decay,
            "name": "encoder_non_backbone_decay",
        })

    if encoder_non_backbone_no_decay:
        param_groups.append({
            "params": encoder_non_backbone_no_decay,
            "lr": effective_base_lr,
            "weight_decay": 0.0,
            "name": "encoder_non_backbone_no_decay",
        })

    # --- All other model parameters at full base LR ---
    if other_decay:
        param_groups.append({
            "params": other_decay,
            "lr": effective_base_lr,
            "weight_decay": weight_decay,
            "name": "other_decay",
        })

    if other_no_decay:
        param_groups.append({
            "params": other_no_decay,
            "lr": effective_base_lr,
            "weight_decay": 0.0,
            "name": "other_no_decay",
        })

    # Log summary
    total_params: int = sum(
        sum(p.numel() for p in group["params"])
        for group in param_groups
    )
    logger.info(
        "_get_param_groups: Built %d parameter groups, "
        "%d total trainable parameters, "
        "encoder_type=%s, decay_rate=%.3f, "
        "effective_base_lr=%.2e",
        len(param_groups),
        total_params,
        encoder_type,
        decay_rate,
        effective_base_lr,
    )

    # Log per-group summary at debug level
    for group in param_groups:
        group_param_count: int = sum(p.numel() for p in group["params"])
        logger.debug(
            "  Group '%s': %d params, lr=%.2e, wd=%.4f",
            group["name"],
            group_param_count,
            group["lr"],
            group["weight_decay"],
        )

    return param_groups


# ---------------------------------------------------------------------------
# Public API: build_optimizer
# ---------------------------------------------------------------------------


def build_optimizer(
    model: nn.Module,
    base_lr: float = _DEFAULT_BASE_LR,
    weight_decay: float = _DEFAULT_WEIGHT_DECAY,
    beta1: float = _DEFAULT_BETA1,
    beta2: float = _DEFAULT_BETA2,
    encoder_type: str = "hiera_b_plus",
    layer_decay_rates: Optional[Dict[str, float]] = None,
    lr_multiplier: float = 1.0,
) -> optim.AdamW:
    """Build AdamW optimizer with layer-wise LR decay on the Hiera image encoder.

    Constructs separate parameter groups for each layer depth of the Hiera
    backbone, applying exponentially decaying learning rates from the output
    layers (full LR) to the input layers (most decayed LR). All other model
    components (memory attention, prompt encoder, mask decoder, memory encoder,
    memory bank) use the full base learning rate.

    Gradient clipping is NOT applied here — use get_grad_clip_params() and
    apply clip_grad_norm_() in the training loop (utils/misc.py clip_gradients).

    Config references (config.yaml):
        pretrain.optimizer.learning_rate: 4.0e-4
        pretrain.optimizer.weight_decay: 0.1
        pretrain.optimizer.beta1: 0.9
        pretrain.optimizer.beta2: 0.999
        pretrain.layer_wise_decay.hiera_b_plus: 0.9

    Paper reference:
        Table 12: "optimizer: AdamW, β1=0.9, β2=0.999, weight_decay=0.1"
        Appendix D.2.1: "apply layer decay (Clark et al., 2020) on the image encoder"

    Args:
        model: The model whose parameters will be optimized. Expected to have
            an image_encoder.backbone submodule for layer-wise decay.
            Parameters with requires_grad=False are automatically excluded
            (handles frozen image encoder during fine-tuning).
        base_lr: Base learning rate before layer-wise decay.
            Defaults to 4.0e-4 (config: pretrain.optimizer.learning_rate).
        weight_decay: L2 weight decay coefficient for eligible parameters.
            Bias, norm, and embedding parameters are excluded from decay.
            Defaults to 0.1 (config: pretrain.optimizer.weight_decay).
        beta1: AdamW first moment decay coefficient.
            Defaults to 0.9 (config: pretrain.optimizer.beta1).
        beta2: AdamW second moment decay coefficient.
            Defaults to 0.999 (config: pretrain.optimizer.beta2).
        encoder_type: Hiera encoder variant name. Used to look up the
            layer-wise decay rate from layer_decay_rates.
            Defaults to "hiera_b_plus" (config: model.image_encoder_type).
        layer_decay_rates: Dict mapping encoder type to decay rate.
            Defaults to the values from config.yaml pretrain.layer_wise_decay:
            {"hiera_t": 0.8, "hiera_s": 0.8, "hiera_b_plus": 0.9, "hiera_l": 0.925}.
        lr_multiplier: Global LR multiplier applied to all parameter groups.
            Use 1.0 for pre-training and full training.
            Use 0.5 for fine-tuning (config: finetuning.learning_rate_multiplier).
            Defaults to 1.0.

    Returns:
        Configured torch.optim.AdamW optimizer with per-layer parameter groups.

    Example:
        # Pre-training
        optimizer = build_optimizer(
            model=sam2_model,
            base_lr=4e-4,
            weight_decay=0.1,
            encoder_type="hiera_b_plus",
        )

        # Fine-tuning (half LR, image encoder frozen externally)
        optimizer = build_optimizer(
            model=sam2_model,
            base_lr=4e-4,
            weight_decay=0.1,
            encoder_type="hiera_b_plus",
            lr_multiplier=0.5,
        )
    """
    if layer_decay_rates is None:
        layer_decay_rates = _LAYER_DECAY_RATES.copy()

    # Validate lr_multiplier
    if lr_multiplier <= 0.0:
        raise ValueError(
            f"lr_multiplier must be positive, got {lr_multiplier}."
        )

    # Build parameter groups with layer-wise LR decay
    param_groups: List[Dict[str, Any]] = _get_param_groups(
        model=model,
        base_lr=base_lr,
        weight_decay=weight_decay,
        encoder_type=encoder_type,
        layer_decay_rates=layer_decay_rates,
        lr_multiplier=lr_multiplier,
    )

    if not param_groups:
        raise ValueError(
            "No trainable parameters found in the model. "
            "Check that model parameters have requires_grad=True."
        )

    # Construct AdamW optimizer
    # Note: weight_decay is set per-group (0.0 for no-decay groups),
    # so we pass 0.0 as the global default to avoid double-applying decay.
    optimizer: optim.AdamW = optim.AdamW(
        param_groups,
        lr=base_lr * lr_multiplier,  # default lr (overridden per group)
        betas=(beta1, beta2),
        weight_decay=0.0,  # per-group weight_decay takes precedence
        eps=1e-8,
    )

    logger.info(
        "build_optimizer: AdamW created with %d parameter groups, "
        "base_lr=%.2e, weight_decay=%.4f, betas=(%.3f, %.3f), "
        "encoder_type=%s, lr_multiplier=%.2f",
        len(param_groups),
        base_lr,
        weight_decay,
        beta1,
        beta2,
        encoder_type,
        lr_multiplier,
    )

    return optimizer


# ---------------------------------------------------------------------------
# Public API: build_scheduler
# ---------------------------------------------------------------------------


def build_scheduler(
    optimizer: optim.Optimizer,
    total_steps: int,
    warmup_iters: int = _DEFAULT_WARMUP_ITERS,
    cooldown_iters: int = _DEFAULT_COOLDOWN_ITERS,
    timescale: int = _DEFAULT_TIMESCALE,
    last_epoch: int = -1,
) -> LambdaLR:
    """Build reciprocal sqrt LR schedule with linear warmup and cooldown.

    Implements the three-phase schedule from Table 12 of the SAM 2 paper:

    Phase 1 — Linear warmup (steps 0 to warmup_iters):
        multiplier(step) = step / warmup_iters

    Phase 2 — Reciprocal sqrt decay (steps warmup_iters to cooldown_start):
        multiplier(step) = timescale / sqrt(max(step, timescale))

    Phase 3 — Linear cooldown (steps cooldown_start to total_steps):
        cooldown_start = total_steps - cooldown_iters
        rsqrt_at_start = timescale / sqrt(max(cooldown_start, timescale))
        multiplier(step) = rsqrt_at_start * (total_steps - step) / cooldown_iters

    The multiplier is applied on top of each parameter group's initial `lr`
    (which already encodes layer-wise decay). This means the scheduler scales
    all groups proportionally, preserving the relative LR ratios between groups.

    Config references (config.yaml):
        pretrain.scheduler.timescale: 1000
        pretrain.scheduler.warmup_iters: 1000
        pretrain.scheduler.cooldown_iters: 5000
        pretrain.steps: 90000

    Paper reference:
        Table 12: "lr_schedule: reciprocal_sqrt, timescale=1000,
            warmup: linear 1k iters, cooldown: linear 5k iters"
        Zhai et al. (2022): "Scaling vision transformers" — source of the
            reciprocal sqrt schedule.

    Args:
        optimizer: The optimizer whose LR will be scheduled. Typically the
            AdamW returned by build_optimizer().
        total_steps: Total number of training iterations.
            Pre-training: 90000 (config: pretrain.steps).
            Full training: 200000 (config: training.num_iterations).
            Fine-tuning: 50000 (config: finetuning.num_iterations).
        warmup_iters: Number of linear warmup steps.
            Defaults to 1000 (config: pretrain.scheduler.warmup_iters).
        cooldown_iters: Number of linear cooldown steps at the end of training.
            Defaults to 5000 (config: pretrain.scheduler.cooldown_iters).
        timescale: Reciprocal sqrt timescale parameter. The LR peaks at
            step=timescale and decays as 1/sqrt(step/timescale) thereafter.
            Defaults to 1000 (config: pretrain.scheduler.timescale).
        last_epoch: The index of the last completed epoch/step. Pass the
            current step when resuming from a checkpoint to correctly
            restore the schedule position. Defaults to -1 (start from step 0).

    Returns:
        torch.optim.lr_scheduler.LambdaLR scheduler. Call scheduler.step()
        once per optimizer step (not per epoch).

    Raises:
        ValueError: If warmup_iters + cooldown_iters >= total_steps (degenerate
            schedule where warmup and cooldown overlap).
        ValueError: If any of the step counts are negative.

    Example:
        scheduler = build_scheduler(
            optimizer=optimizer,
            total_steps=90000,
            warmup_iters=1000,
            cooldown_iters=5000,
            timescale=1000,
        )
        # In training loop:
        optimizer.step()
        scheduler.step()
    """
    # Validate inputs
    if total_steps <= 0:
        raise ValueError(
            f"total_steps must be positive, got {total_steps}."
        )
    if warmup_iters < 0:
        raise ValueError(
            f"warmup_iters must be non-negative, got {warmup_iters}."
        )
    if cooldown_iters < 0:
        raise ValueError(
            f"cooldown_iters must be non-negative, got {cooldown_iters}."
        )
    if timescale <= 0:
        raise ValueError(
            f"timescale must be positive, got {timescale}."
        )

    # Check for degenerate schedule (warmup and cooldown overlap)
    cooldown_start: int = total_steps - cooldown_iters
    if cooldown_start <= warmup_iters and cooldown_iters > 0 and warmup_iters > 0:
        logger.warning(
            "build_scheduler: warmup_iters (%d) + cooldown_iters (%d) >= "
            "total_steps (%d). The warmup and cooldown phases overlap. "
            "Consider reducing warmup_iters or cooldown_iters.",
            warmup_iters,
            cooldown_iters,
            total_steps,
        )

    # Precompute the reciprocal sqrt value at the cooldown start step.
    # This is the LR multiplier at the beginning of the cooldown phase.
    # We compute it once here (captured in the closure) for efficiency.
    rsqrt_at_cooldown_start: float = (
        float(timescale) / math.sqrt(max(cooldown_start, timescale))
        if cooldown_start > 0
        else 1.0
    )

    # ------------------------------------------------------------------
    # LR multiplier lambda function
    # ------------------------------------------------------------------
    # LambdaLR calls this function with the current step (0-indexed).
    # The returned value is multiplied by each parameter group's initial lr.
    # ------------------------------------------------------------------

    def lr_lambda(step: int) -> float:
        """Compute the LR multiplier for the given training step.

        Args:
            step: Current training step (0-indexed). LambdaLR passes
                `last_epoch + 1` on the first call after initialization.

        Returns:
            Float LR multiplier in [0.0, ~1.0]. The actual LR for each
            parameter group is: group_initial_lr * lr_lambda(step).
        """
        # ------------------------------------------------------------------
        # Phase 1: Linear warmup (steps 0 to warmup_iters - 1)
        # LR increases linearly from 0 to base_lr over warmup_iters steps.
        # At step 0: multiplier = 0.0 (LR starts at 0)
        # At step warmup_iters: multiplier = 1.0 (full LR)
        # ------------------------------------------------------------------
        if warmup_iters > 0 and step < warmup_iters:
            return float(step) / float(warmup_iters)

        # ------------------------------------------------------------------
        # Phase 3: Linear cooldown (steps cooldown_start to total_steps)
        # LR decreases linearly from rsqrt_at_cooldown_start to 0.
        # ------------------------------------------------------------------
        if cooldown_iters > 