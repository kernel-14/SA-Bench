## utils/misc.py
"""General utility functions for SAM 2 reproduction.

This module provides foundational utilities used across the entire project:
checkpoint save/load, distributed training helpers, seed setting, logging
setup, and config merging via OmegaConf.

This file has zero internal imports to avoid circular dependencies.
"""

import contextlib
import copy
import json
import logging
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter


# ---------------------------------------------------------------------------
# Config loading and merging
# ---------------------------------------------------------------------------


def load_config(
    config_path: str,
    overrides: Optional[List[str]] = None,
) -> DictConfig:
    """Load and optionally merge a YAML config file.

    Args:
        config_path: Path to the base YAML config file.
        overrides: Optional list of dotlist override strings, e.g.
            ["model.input_resolution=512", "pretrain.batch_size=128"].

    Returns:
        Merged DictConfig object.
    """
    cfg = OmegaConf.load(config_path)
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)
    return cfg


def get_model_config(cfg: DictConfig) -> dict:
    """Extract the model sub-config as a plain dict.

    Args:
        cfg: Full DictConfig loaded from config.yaml.

    Returns:
        Plain dict of model configuration values.
    """
    return OmegaConf.to_container(cfg.model, resolve=True)


def get_pretrain_config(cfg: DictConfig) -> dict:
    """Extract the pre-training sub-config as a plain dict.

    Args:
        cfg: Full DictConfig loaded from config.yaml.

    Returns:
        Plain dict of pre-training configuration values.
    """
    return OmegaConf.to_container(cfg.pretrain, resolve=True)


def get_train_config(cfg: DictConfig) -> dict:
    """Extract the full training and fine-tuning sub-configs as a plain dict.

    Args:
        cfg: Full DictConfig loaded from config.yaml.

    Returns:
        Plain dict with keys 'training' and 'finetuning'.
    """
    return {
        "training": OmegaConf.to_container(cfg.training, resolve=True),
        "finetuning": OmegaConf.to_container(cfg.finetuning, resolve=True),
    }


def get_eval_config(cfg: DictConfig) -> dict:
    """Extract the evaluation sub-config as a plain dict.

    Args:
        cfg: Full DictConfig loaded from config.yaml.

    Returns:
        Plain dict of evaluation configuration values.
    """
    return OmegaConf.to_container(cfg.evaluation, resolve=True)


# ---------------------------------------------------------------------------
# Seed setting
# ---------------------------------------------------------------------------


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value.
        deterministic: If True, enable deterministic CUDA operations at the
            cost of performance. Defaults to False for training speed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def set_seed_for_rank(
    seed: int,
    rank: int,
    deterministic: bool = False,
) -> None:
    """Set rank-offset seeds for distributed training.

    Each rank receives seed + rank to ensure different data sampling per GPU
    while maintaining reproducibility.

    Args:
        seed: Base integer seed value.
        rank: Current process rank.
        deterministic: If True, enable deterministic CUDA operations.
    """
    set_seed(seed + rank, deterministic=deterministic)


# ---------------------------------------------------------------------------
# Distributed training helpers
# ---------------------------------------------------------------------------


def init_distributed(backend: str = "nccl") -> Tuple[int, int, int]:
    """Initialize distributed training if environment variables are set.

    Reads RANK, WORLD_SIZE, and LOCAL_RANK from environment (set by torchrun
    or submitit). Falls back to single-GPU mode if not set.

    Args:
        backend: Distributed backend. Defaults to "nccl" for GPU training.

    Returns:
        Tuple of (rank, world_size, local_rank).
    """
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        dist.init_process_group(backend=backend)
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def is_distributed() -> bool:
    """Check whether distributed training is active.

    Returns:
        True if torch.distributed is available and initialized.
    """
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Get the current process rank.

    Returns:
        Rank integer; 0 for single-GPU mode.
    """
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get the total number of processes.

    Returns:
        World size integer; 1 for single-GPU mode.
    """
    if is_distributed():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check whether the current process is the main (rank 0) process.

    Returns:
        True if rank == 0.
    """
    return get_rank() == 0


def barrier() -> None:
    """Synchronize all distributed processes.

    No-op in single-GPU mode. Must be called after checkpoint saves to
    prevent rank 0 from advancing while other ranks are still reading.
    """
    if is_distributed():
        dist.barrier()


def reduce_dict(
    input_dict: Dict[str, torch.Tensor],
    average: bool = True,
) -> Dict[str, torch.Tensor]:
    """All-reduce a dictionary of scalar tensors across all ranks.

    Args:
        input_dict: Dictionary mapping metric names to scalar tensors.
        average: If True, divide by world_size after reduction.

    Returns:
        Reduced dictionary on all ranks.
    """
    if not is_distributed():
        return input_dict

    world_size = get_world_size()
    keys = sorted(input_dict.keys())
    values = torch.stack([input_dict[k].float() for k in keys])

    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    if average:
        values /= world_size

    return {k: v for k, v in zip(keys, values)}


def wrap_model_ddp(
    model: nn.Module,
    device_ids: List[int],
    find_unused_parameters: bool = False,
) -> nn.Module:
    """Wrap a model with DistributedDataParallel.

    Args:
        model: The model to wrap.
        device_ids: List of GPU device IDs for this process.
        find_unused_parameters: Set True when some parameters do not receive
            gradients (e.g., frozen image encoder during fine-tuning).

    Returns:
        DDP-wrapped model.
    """
    return nn.parallel.DistributedDataParallel(
        model,
        device_ids=device_ids,
        find_unused_parameters=find_unused_parameters,
    )


def unwrap_model(model: nn.Module) -> nn.Module:
    """Unwrap a DDP-wrapped model to access the underlying module.

    Args:
        model: Potentially DDP-wrapped model.

    Returns:
        Underlying nn.Module without DDP wrapper.
    """
    if isinstance(model, nn.parallel.DistributedDataParallel):
        return model.module
    return model


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------


def _strip_ddp_prefix(state_dict: dict) -> dict:
    """Strip 'module.' prefix from DDP-saved state dict keys.

    Args:
        state_dict: State dict potentially containing 'module.' prefixes.

    Returns:
        State dict with prefixes removed.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k[len("module."):] if k.startswith("module.") else k
        new_state_dict[new_key] = v
    return new_state_dict


def _get_rng_state() -> dict:
    """Capture current RNG states for exact training resume.

    Returns:
        Dict containing Python, NumPy, and PyTorch RNG states.
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def _restore_rng_state(rng_state: dict) -> None:
    """Restore RNG states from a saved checkpoint.

    Args:
        rng_state: Dict of RNG states as returned by _get_rng_state().
    """
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch_cpu"])
    torch.cuda.set_rng_state_all(rng_state["torch_cuda"])


def save_checkpoint(
    state: dict,
    checkpoint_dir: str,
    filename: str,
    is_best: bool = False,
) -> None:
    """Save a training checkpoint to disk (main process only).

    The state dict should contain:
        - step: current training step
        - epoch: current epoch
        - model_state_dict: unwrapped model state
        - optimizer_state_dict: optimizer state
        - scheduler_state_dict: LR scheduler state
        - config: serialized config dict
        - best_metric: best metric seen so far
        - rng_state: RNG states for exact resume

    Args:
        state: Dictionary containing all checkpoint data.
        checkpoint_dir: Directory to save the checkpoint.
        filename: Checkpoint filename (e.g., "checkpoint_step_1000.pth").
        is_best: If True, also save as "best_model.pth".
    """
    if not is_main_process():
        barrier()
        return

    ensure_dir(checkpoint_dir)
    save_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, save_path)

    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        shutil.copyfile(save_path, best_path)

    barrier()


def save_checkpoint_periodic(
    state: dict,
    checkpoint_dir: str,
    step: int,
    save_every: int,
    keep_last_n: int = 3,
) -> None:
    """Save checkpoint periodically and always save as 'latest.pth'.

    Keeps only the last `keep_last_n` periodic checkpoints to manage disk
    usage. Always overwrites 'latest.pth' for easy resume.

    Args:
        state: Dictionary containing all checkpoint data.
        checkpoint_dir: Directory to save checkpoints.
        step: Current training step.
        save_every: Save frequency in steps.
        keep_last_n: Number of periodic checkpoints to retain.
    """
    if not is_main_process():
        barrier()
        return

    ensure_dir(checkpoint_dir)

    # Always save latest
    latest_path = os.path.join(checkpoint_dir, "latest.pth")
    torch.save(state, latest_path)

    # Save periodic checkpoint
    if step % save_every == 0:
        periodic_filename = f"checkpoint_step_{step:08d}.pth"
        periodic_path = os.path.join(checkpoint_dir, periodic_filename)
        torch.save(state, periodic_path)

        # Clean up old periodic checkpoints
        _cleanup_old_checkpoints(checkpoint_dir, keep_last_n=keep_last_n)

    barrier()


def _cleanup_old_checkpoints(checkpoint_dir: str, keep_last_n: int = 3) -> None:
    """Remove old periodic checkpoints, keeping only the most recent N.

    Args:
        checkpoint_dir: Directory containing checkpoints.
        keep_last_n: Number of most recent checkpoints to keep.
    """
    pattern = "checkpoint_step_"
    checkpoints = sorted(
        [
            f
            for f in os.listdir(checkpoint_dir)
            if f.startswith(pattern) and f.endswith(".pth")
        ]
    )
    # Remove oldest checkpoints beyond keep_last_n
    for old_ckpt in checkpoints[:-keep_last_n]:
        old_path = os.path.join(checkpoint_dir, old_ckpt)
        os.remove(old_path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    device: str = "cuda",
    restore_rng: bool = False,
) -> dict:
    """Load a checkpoint and restore model, optimizer, and scheduler states.

    Handles checkpoints saved from DDP-wrapped models by stripping the
    'module.' prefix from state dict keys.

    Args:
        checkpoint_path: Path to the checkpoint file.
        model: Model to load weights into (may be DDP-wrapped).
        optimizer: Optional optimizer to restore state.
        scheduler: Optional LR scheduler to restore state.
        device: Device to map tensors to.
        restore_rng: If True, restore RNG states for exact resume.

    Returns:
        Full checkpoint dict so caller can extract step, best_metric, etc.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Strip DDP prefix if present
    model_state = _strip_ddp_prefix(checkpoint["model_state_dict"])
    unwrap_model(model).load_state_dict(model_state, strict=True)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if restore_rng and "rng_state" in checkpoint:
        _restore_rng_state(checkpoint["rng_state"])

    return checkpoint


def load_pretrained_weights(
    model: nn.Module,
    pretrained_path: str,
    strict: bool = False,
    device: str = "cuda",
) -> None:
    """Load only model weights from a checkpoint, ignoring optimizer/scheduler.

    Used to initialize from a pre-training checkpoint before full training.
    Logs missing and unexpected keys for debugging.

    Args:
        model: Model to load weights into.
        pretrained_path: Path to the pre-training checkpoint.
        strict: If False, allow partial loading (missing/unexpected keys).
        device: Device to map tensors to.
    """
    logger = logging.getLogger(__name__)
    checkpoint = torch.load(pretrained_path, map_location=device)

    if "model_state_dict" in checkpoint:
        state_dict = _strip_ddp_prefix(checkpoint["model_state_dict"])
    else:
        # Assume the file is a raw state dict
        state_dict = _strip_ddp_prefix(checkpoint)

    missing_keys, unexpected_keys = unwrap_model(model).load_state_dict(
        state_dict, strict=strict
    )

    if missing_keys:
        logger.info("Missing keys when loading pretrained weights (%d):", len(missing_keys))
        for k in missing_keys[:20]:  # Log first 20 to avoid spam
            logger.info("  MISSING: %s", k)
        if len(missing_keys) > 20:
            logger.info("  ... and %d more", len(missing_keys) - 20)

    if unexpected_keys:
        logger.info(
            "Unexpected keys when loading pretrained weights (%d):",
            len(unexpected_keys),
        )
        for k in unexpected_keys[:20]:
            logger.info("  UNEXPECTED: %s", k)
        if len(unexpected_keys) > 20:
            logger.info("  ... and %d more", len(unexpected_keys) - 20)

    logger.info(
        "Loaded pretrained weights from %s (strict=%s)", pretrained_path, strict
    )


def load_hiera_mae_weights(
    model: nn.Module,
    hiera_checkpoint_path: str,
    device: str = "cuda",
) -> None:
    """Load MAE pre-trained Hiera weights into SAM2Model's image encoder.

    Maps Hiera checkpoint keys to SAM2Model's image_encoder.backbone.*
    namespace. Handles key mismatches between Hiera's standalone format
    and SAM2's nested format.

    Args:
        model: SAM2Model instance whose image_encoder.backbone will be loaded.
        hiera_checkpoint_path: Path to the MAE pre-trained Hiera checkpoint.
        device: Device to map tensors to.
    """
    logger = logging.getLogger(__name__)
    checkpoint = torch.load(hiera_checkpoint_path, map_location=device)

    # Hiera MAE checkpoints may store weights under 'model' or 'state_dict'
    if "model" in checkpoint:
        hiera_state = checkpoint["model"]
    elif "state_dict" in checkpoint:
        hiera_state = checkpoint["state_dict"]
    else:
        hiera_state = checkpoint

    hiera_state = _strip_ddp_prefix(hiera_state)

    # Map Hiera keys to SAM2Model's image_encoder.backbone namespace
    mapped_state = {}
    prefix = "image_encoder.backbone."
    for k, v in hiera_state.items():
        mapped_key = prefix + k
        mapped_state[mapped_key] = v

    missing_keys, unexpected_keys = unwrap_model(model).load_state_dict(
        mapped_state, strict=False
    )

    # Count how many backbone keys were successfully loaded
    loaded_count = sum(
        1 for k in mapped_state if k not in missing_keys
    )
    logger.info(
        "Loaded %d/%d Hiera MAE weights into image_encoder.backbone",
        loaded_count,
        len(mapped_state),
    )
    if missing_keys:
        backbone_missing = [k for k in missing_keys if "image_encoder.backbone" in k]
        if backbone_missing:
            logger.warning(
                "%d backbone keys not found in model: %s ...",
                len(backbone_missing),
                backbone_missing[:5],
            )


def build_checkpoint_state(
    model: nn.Module,
    optimizer: Any,
    scheduler: Any,
    step: int,
    epoch: int,
    best_metric: float,
    cfg: Optional[DictConfig] = None,
) -> dict:
    """Build a complete checkpoint state dictionary.

    Args:
        model: Model (may be DDP-wrapped; will be unwrapped automatically).
        optimizer: Optimizer instance.
        scheduler: LR scheduler instance.
        step: Current training step.
        epoch: Current epoch.
        best_metric: Best validation metric seen so far.
        cfg: Optional OmegaConf config to serialize into checkpoint.

    Returns:
        Checkpoint state dict ready for torch.save().
    """
    config_dict = {}
    if cfg is not None:
        config_dict = OmegaConf.to_container(cfg, resolve=True)

    return {
        "step": step,
        "epoch": epoch,
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else {},
        "config": config_dict,
        "best_metric": best_metric,
        "rng_state": _get_rng_state(),
    }


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def setup_logger(
    name: str,
    log_dir: Optional[str] = None,
    rank: int = 0,
) -> logging.Logger:
    """Create and configure a named logger.

    Rank 0 logs at INFO level; all other ranks log at WARNING to suppress
    noise from 255 worker processes during 256-GPU training.

    Args:
        name: Logger name (typically __name__ of the calling module).
        log_dir: Optional directory to write a log file. Only rank 0 writes.
        rank: Current process rank.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    level = logging.INFO if rank == 0 else logging.WARNING
    logger.setLevel(level)

    formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (rank 0 only)
    if log_dir is not None and rank == 0:
        ensure_dir(log_dir)
        log_file = os.path.join(log_dir, "train.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    return logger


def setup_tensorboard(
    log_dir: str,
    rank: int = 0,
) -> Optional[SummaryWriter]:
    """Create a TensorBoard SummaryWriter on the main process only.

    Args:
        log_dir: Directory for TensorBoard event files.
        rank: Current process rank.

    Returns:
        SummaryWriter on rank 0, None on all other ranks.
    """
    if rank == 0:
        ensure_dir(log_dir)
        return SummaryWriter(log_dir=log_dir)
    return None


def log_metrics(
    writer: Optional[SummaryWriter],
    metrics: Dict[str, float],
    step: int,
    prefix: str = "",
) -> None:
    """Log scalar metrics to TensorBoard.

    Args:
        writer: SummaryWriter instance, or None (no-op).
        metrics: Dictionary of metric name to scalar value.
        step: Current training step for the x-axis.
        prefix: Optional prefix prepended to each metric tag.
    """
    if writer is None:
        return
    for key, value in metrics.items():
        tag = f"{prefix}/{key}" if prefix else key
        writer.add_scalar(tag, value, step)


def log_images(
    writer: Optional[SummaryWriter],
    tag: str,
    images: torch.Tensor,
    step: int,
    max_images: int = 4,
) -> None:
    """Log a batch of images to TensorBoard for visual debugging.

    Args:
        writer: SummaryWriter instance, or None (no-op).
        tag: Tag name for the image group.
        images: Image tensor of shape (N, C, H, W) in [0, 1] range.
        step: Current training step.
        max_images: Maximum number of images to log.
    """
    if writer is None:
        return
    images = images[:max_images]
    writer.add_images(tag, images, step)


def format_metrics(metrics: Dict[str, float]) -> str:
    """Format a metrics dictionary as a human-readable string.

    Args:
        metrics: Dictionary of metric name to scalar value.

    Returns:
        Formatted string, e.g. "loss=1.234 focal=0.987 dice=0.123".
    """
    parts = [f"{k}={v:.4f}" for k, v in sorted(metrics.items())]
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Device management
# ---------------------------------------------------------------------------


def get_device(local_rank: int = 0) -> torch.device:
    """Get the appropriate torch device for the current process.

    Args:
        local_rank: Local GPU rank within the current node.

    Returns:
        torch.device for CUDA if available, else CPU.
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def move_to_device(data: Any, device: torch.device) -> Any:
    """Recursively move tensors or collections of tensors to a device.

    Handles tensors, lists, tuples, and dicts. Non-tensor values are
    returned unchanged.

    Args:
        data: Tensor, list, tuple, or dict potentially containing tensors.
        device: Target torch.device.

    Returns:
        Data with all tensors moved to the specified device.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        moved = [move_to_device(item, device) for item in data]
        return type(data)(moved)
    return data


# ---------------------------------------------------------------------------
# Precision helpers
# ---------------------------------------------------------------------------


def get_autocast_context(precision: str = "bfloat16"):
    """Return the appropriate autocast context manager for the given precision.

    The paper uses bfloat16 throughout (Table 12, Appendix D.3).

    Args:
        precision: One of "bfloat16", "float16", or "float32".

    Returns:
        Context manager for automatic mixed precision.
    """
    if precision == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    elif precision == "float16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        return contextlib.nullcontext()


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------


def ensure_dir(path: str) -> str:
    """Create a directory and all parents if they do not exist.

    Args:
        path: Directory path to create.

    Returns:
        The input path, for chaining.
    """
    os.makedirs(path, exist_ok=True)
    return path


def get_timestamp() -> str:
    """Return the current datetime as a compact string for naming experiments.

    Returns:
        String in format "YYYYMMDD_HHMMSS".
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Gradient utilities
# ---------------------------------------------------------------------------


def clip_gradients(
    model: nn.Module,
    clip_type: str = "l2",
    clip_max: float = 0.1,
) -> float:
    """Clip model gradients and return the pre-clip gradient norm.

    From config: pretrain.optimizer.gradient_clip_type = "l2",
                 pretrain.optimizer.gradient_clip_max = 0.1

    Args:
        model: Model whose parameters' gradients will be clipped.
        clip_type: "l2" for norm clipping, "value" for value clipping.
        clip_max: Maximum gradient norm (l2) or value (value).

    Returns:
        Gradient norm before clipping (for logging). Returns 0.0 if no
        gradients are available.
    """
    params = [p for p in model.parameters() if p.grad is not None]
    if not params:
        return 0.0

    if clip_type == "l2":
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=clip_max
        )
        return float(grad_norm)
    elif clip_type == "value":
        # Compute norm manually before clipping
        total_norm = 0.0
        for p in params:
            total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_max)
        return total_norm
    else:
        raise ValueError(f"Unknown clip_type: {clip_type}. Use 'l2' or 'value'.")


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count total, trainable, and frozen parameters in a model.

    Args:
        model: The model to inspect.

    Returns:
        Dict with keys "total", "trainable", "frozen".
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {"total": total, "trainable": trainable, "frozen": frozen}


# ---------------------------------------------------------------------------
# JSON result saving
# ---------------------------------------------------------------------------


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """Save evaluation results to a JSON file (main process only).

    Args:
        results: Nested dictionary of evaluation results.
        output_path: Full path to the output JSON file.
    """
    if not is_main_process():
        return

    ensure_dir(os.path.dirname(output_path) or ".")

    # Convert any non-serializable values (e.g., numpy floats) to Python floats
    def _convert(obj: Any) -> Any:
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [_convert(item) for item in obj]
        return obj

    serializable_results = _convert(results)

    with open(output_path, "w") as f:
        json.dump(serializable_results, f, indent=2)
