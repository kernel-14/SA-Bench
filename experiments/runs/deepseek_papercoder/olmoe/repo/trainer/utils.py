# trainer/utils.py
"""
FSDP setup, optimizer/scheduler creation, and checkpoint I/O for OLMoE.

Implements the exact training infrastructure required by the OLMoE‑1B‑7B
reproduction, aligning with the paper's Appendix B and the config.yaml
specification.

Functions:
    setup_fsdp_model      – wrap a MoETransformer with FullyShardedDataParallel
    get_optimizer_and_scheduler – create AdamW + two‑phase LR schedule
    save_checkpoint       – persist full training state (model, optim, sched)
    load_checkpoint       – restore training state from a saved checkpoint
"""

from __future__ import annotations

import functools
import logging
import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    ModuleWrapPolicy,
    transformer_auto_wrap_policy,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.optim.optimizer import Optimizer

# Import the custom model classes to use in auto‑wrap policy
from model.moe_transformer import MoETransformer, TransformerBlock

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. FSDP Model Wrapping
# ---------------------------------------------------------------------------
def setup_fsdp_model(
    model: MoETransformer,
    config: Dict[str, Any],
) -> MoETransformer:
    """
    Wraps the pre‑initialised MoETransformer with FSDP and returns the wrapped
    model.

    Configured from the ``fsdp`` and ``pretraining`` sections of the YAML config:
        - sharding strategy: SHARD_GRAD_OP (ZeRO‑2 style)
        - mixed precision:   bfloat16 for params, gradients, buffers
        - auto‑wrap policy:  wraps each ``TransformerBlock`` individually
        - forward/backward prefetch: enabled

    The paper uses 256 H100 GPUs with NVLink/InfiniBand; this function
    assumes the process group has already been initialised.

    Args:
        model:  An *unwrapped* instance of MoETransformer.
        config: The full project configuration (dictionary loaded from config.yaml).

    Returns:
        FSDP‑wrapped model, ready for training.
    """
    # Determine mixed precision policy from pretraining section
    precision_str = config["pretraining"].get("precision", "bf16").lower()
    if precision_str == "bf16":
        mp_dtype = torch.bfloat16
    elif precision_str == "fp16":
        mp_dtype = torch.float16
    else:
        raise ValueError(f"Unsupported precision: {precision_str}")

    mixed_precision_policy = MixedPrecision(
        param_dtype=mp_dtype,
        reduce_dtype=mp_dtype,
        buffer_dtype=mp_dtype,
    )

    # Define auto‑wrap policy: wrap each TransformerBlock, but not the
    # top‑level MoETransformer’s embedding or output projection.
    # Use ModuleWrapPolicy which directly wraps instances of given module types.
    auto_wrap_policy = ModuleWrapPolicy(
        (TransformerBlock,)          # wrap every TransformerBlock individually
    )

    # FSDP arguments from config.fsdp
    fsdp_cfg = config.get("fsdp", {})
    backward_prefetch = BackwardPrefetch.BACKWARD_PRE
    forward_prefetch = fsdp_cfg.get("forward_prefetch", True)

    wrapped_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        mixed_precision=mixed_precision_policy,
        backward_prefetch=backward_prefetch,
        forward_prefetch=forward_prefetch,
        use_orig_params=True,       # required for optimizer state dict compatibility
        device_id=torch.cuda.current_device(),
    )

    logger.info(
        "FSDP wrapping complete. Sharding=%s, precision=%s, "
        "auto_wrap=TransformerBlock",
        ShardingStrategy.SHARD_GRAD_OP,
        precision_str,
    )
    return wrapped_model


# ---------------------------------------------------------------------------
# 2. Optimizer and Scheduler Creation
# ---------------------------------------------------------------------------
def get_optimizer_and_scheduler(
    model: nn.Module,
    config: Dict[str, Any],
) -> Tuple[Optimizer, LRScheduler]:
    """
    Create the AdamW optimizer and a two‑phase learning rate scheduler.

    Hyperparameters are read strictly from the ``pretraining`` section:
        - optimizer: adamw (fixed)
        - peak_learning_rate, minimum_learning_rate
        - warmup_steps, total_tokens, seq_length, global_batch_size_samples
        - adam_beta1, adam_beta2, weight_decay, adam_epsilon
        - annealing_tokens, annealing_schedule (linear)

    Schedule phases (as described in §2 and §3):
        Phase 1: warmup (linear) + cosine decay to min_lr
        Phase 2: linear annealing from min_lr to 0

    Args:
        model:  The FSDP‑wrapped model (or raw model). AdamW groups all
                parameters together (no per‑group exclusions).
        config: Full configuration dict.

    Returns:
        (optimizer, scheduler) tuple.
    """
    pretrain_cfg = config["pretraining"]
    model_cfg = config["model"]

    # ---------- Optimizer ----------
    adam_betas = (pretrain_cfg["adam_beta1"], pretrain_cfg["adam_beta2"])
    optimizer = AdamW(
        model.parameters(),
        lr=pretrain_cfg["peak_learning_rate"],
        betas=adam_betas,
        weight_decay=pretrain_cfg["weight_decay"],          # 0.1, applied to all parameters
        eps=pretrain_cfg["adam_epsilon"],                   # 1e-8
    )

    # ---------- Compute step counts ----------
    batch_tokens = pretrain_cfg["global_batch_size_samples"] * model_cfg.get(
        "max_sequence_length", pretrain_cfg["seq_length"]
    )
    total_steps = int(pretrain_cfg["total_tokens"] // batch_tokens)
    annealing_tokens = pretrain_cfg["annealing_tokens"]
    annealing_steps = int(annealing_tokens // batch_tokens)
    main_steps = total_steps - annealing_steps
    warmup_steps = pretrain_cfg["warmup_steps"]

    # ---------- Learning rate multipliers ----------
    peak_lr = pretrain_cfg["peak_learning_rate"]
    min_lr = pretrain_cfg["minimum_learning_rate"]
    min_ratio = min_lr / peak_lr  # 0.1

    def lr_lambda(step: int) -> float:
        """Compute multiplier for the current training step."""
        if step < warmup_steps:
            # Linear warmup from 0 → 1
            return max(0.0, step / max(1, warmup_steps))

        if step < main_steps:
            # Cosine decay from 1 → min_ratio
            progress = (step - warmup_steps) / max(1, main_steps - warmup_steps - 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine

        # Phase 2: linear annealing from min_ratio → 0
        step_in_anneal = step - main_steps
        total_anneal = max(1, annealing_steps - 1)
        progress = step_in_anneal / total_anneal
        return min_ratio * (1.0 - progress)

    scheduler = LambdaLR(optimizer, lr_lambda)

    logger.info(
        "Optimizer: AdamW (β1=%.3f, β2=%.3f, ε=%.0e, wd=%.3f)",
        adam_betas[0], adam_betas[1], pretrain_cfg["adam_epsilon"],
        pretrain_cfg["weight_decay"],
    )
    logger.info(
        "LR schedule: warmup=%d, main=%d, anneal=%d, peak=%.2e, min=%.2e",
        warmup_steps, main_steps, annealing_steps, peak_lr, min_lr,
    )
    logger.info("Total steps=%d, batch tokens=%.1fM", total_steps, batch_tokens / 1e6)

    return optimizer, scheduler


# ---------------------------------------------------------------------------
# 3. Checkpoint Saving
# ---------------------------------------------------------------------------
def save_checkpoint(
    fsdp_model: FSDP,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    step: int,
    path: str,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a full training checkpoint: model weights, optimizer state, scheduler
    state, and the current step number.

    The function gathers the full state dictionary from all ranks onto rank 0,
    offloads it to CPU, and writes a single file. Only rank 0 performs the
    actual disk write.

    Args:
        fsdp_model: FSDP‑wrapped model.
        optimizer:  The AdamW optimizer.
        scheduler:  The LambdaLR scheduler.
        step:       Current global training step (int).
        path:       Destination file path.
        config:     Optional config dict to store alongside the checkpoint.

    Raises:
        RuntimeError: If the checkpoint cannot be saved.
    """
    # Synchronise before saving to avoid stale reads
    dist.barrier()

    if dist.get_rank() != 0:
        return

    try:
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        # – model state –
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
            fsdp_model, StateDictType.FULL_STATE_DICT, save_policy
        ):
            model_state: Dict[str, torch.Tensor] = fsdp_model.state_dict()

        # – optimizer state –
        # use_orig_params=True ensures compatible state dict
        optim_state = FSDP.optim_state_dict(fsdp_model, optimizer)

        # – scheduler state –
        sched_state = scheduler.state_dict()

        checkpoint: Dict[str, Any] = {
            "model": model_state,
            "optimizer": optim_state,
            "scheduler": sched_state,
            "step": step,
        }
        if config is not None:
            checkpoint["config"] = config

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(checkpoint, path)
        logger.info("Checkpoint saved (step %d) → %s", step, path)

    except Exception as e:
        logger.error("Failed to save checkpoint at step %d: %s", step, str(e))
        raise RuntimeError(f"Checkpoint save failed: {e}") from e


# ---------------------------------------------------------------------------
# 4. Checkpoint Loading
# ---------------------------------------------------------------------------
def load_checkpoint(
    fsdp_model: FSDP,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    path: str,
    config: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Restore a training checkpoint from disk.

    Loads the full model state, optimizer state, and scheduler state,
    reshards the data across all workers, and returns the saved step number.

    All ranks must call this function; it reads the checkpoint file from a
    shared filesystem.

    Args:
        fsdp_model: FSDP‑wrapped model (pre‑initialised with the correct architecture).
        optimizer:  Optimizer instance (must match the saved state structure).
        scheduler:  Scheduler instance (must match the saved state structure).
        path:       Path to the checkpoint file.
        config:     Optional config to verify against the saved config (not implemented
                    but can be added for strict reproducibility).

    Returns:
        The global step number (int) at which the checkpoint was saved.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        RuntimeError:      If loading fails due to architecture or state mismatch.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    logger.info("Loading checkpoint from %s", path)

    # Load the full checkpoint on every rank (shared NFS)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    # – model state dict restore –
    model_state = checkpoint["model"]
    with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT):
        fsdp_model.load_state_dict(model_state)
    logger.debug("Model state loaded.")

    # – optimizer state dict restore –
    optim_state = checkpoint["optimizer"]
    # Convert to a version compatible with current sharding
    optim_state_to_load = FSDP.optim_state_dict_to_load(
        fsdp_model, optimizer, optim_state
    )
    optimizer.load_state_dict(optim_state_to_load)
    logger.debug("Optimizer state loaded.")

    # – scheduler state dict restore –
    scheduler.load_state_dict(checkpoint["scheduler"])
    logger.debug("Scheduler state loaded.")

    step = checkpoint["step"]
    logger.info("Resuming from step %d", step)

    # (Optional) config validation could go here
    return step

