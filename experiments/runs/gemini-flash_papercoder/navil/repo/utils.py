import os
import sys
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR, CosineAnnealingLR, SequentialLR

from loguru import logger

# Assuming Config class is available from config.py
# To avoid circular import or direct file dependency, we'll assume `Config` is passed
# or imported globally by `main.py` if needed by helper functions directly.
# For now, `Config` type hint will be used.
# If `Config` class is needed for `get_numerical_precision` or other simple mapping,
# and cannot be passed directly, it would typically be imported:
# from config import Config
# For this file, functions requiring Config will receive it as an argument.

class Config:
    """Placeholder for the Config class defined in config.py.
    This is to satisfy type hinting without creating a circular dependency.
    Actual Config object will be passed from main.py.
    """
    def get(self, key: str, default: Any = None) -> Any:
        """Dummy get method for type hinting."""
        raise NotImplementedError("This is a placeholder. Use the actual Config object.")

    # Add attributes that are expected to be accessed
    common: Any = None
    model_architecture: Any = None
    training_stages: Any = None
    special_tokens: Any = None
    evaluation: Any = None
    llm_name_or_path: str = "" # Example of an attribute

def setup_logging(rank: int = 0) -> None:
    """
    Configures the logging system to provide consistent and informative output.
    Only the primary (rank 0) process logs extensively; other processes log
    warnings or errors to avoid cluttered output in distributed training.

    Args:
        rank: The rank of the current process in a distributed training setup.
              Defaults to 0 (main process).
    """
    logger.remove()  # Remove default handlers

    # Configure stdout handler for all processes
    # Only rank 0 gets INFO and higher, others get WARNING and higher
    log_level = "INFO" if rank == 0 else "WARNING"
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        colorize=True,
        filter=lambda record: record["level"].no >= logger.level(log_level).no
    )

    if rank == 0:
        logger.info("Logging setup complete for rank 0 (main process).")
    else:
        logger.info(f"Logging setup complete for rank {rank} (worker process), logging WARNING and above.")

def setup_distributed_training() -> Tuple[int, int, str]:
    """
    Initializes PyTorch's distributed backend for multi-GPU or multi-node training.

    Returns:
        A tuple containing:
            - rank (int): The global rank of the current process.
            - world_size (int): The total number of participating processes.
            - device (str): The device string (e.g., "cuda:0" or "cpu").
    """
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
        logger.info(f"Distributed training already initialized: rank={rank}, world_size={world_size}, device={device}")
        return rank, world_size, device

    # Check for standard environment variables for distributed training
    if (os.environ.get("RANK") is not None and
        os.environ.get("WORLD_SIZE") is not None and
        os.environ.get("MASTER_ADDR") is not None and
        os.environ.get("MASTER_PORT") is not None):

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        master_addr = os.environ["MASTER_ADDR"]
        master_port = os.environ["MASTER_PORT"]

        if world_size > 1 and torch.cuda.is_available():
            logger.info(f"Initializing distributed training with rank {rank}/{world_size} on {master_addr}:{master_port}")
            dist.init_process_group(backend="nccl", rank=rank, world_size=world_size,
                                    init_method=f"env://{master_addr}:{master_port}")
            device = f"cuda:{rank}"
            torch.cuda.set_device(device)
            logger.info(f"Distributed training initialized on device {device}")
            return rank, world_size, device
        elif world_size > 1 and not torch.cuda.is_available():
            logger.warning("Distributed training requested but CUDA is not available. Falling back to CPU.")
            dist.init_process_group(backend="gloo", rank=rank, world_size=world_size) # gloo for CPU distributed
            device = "cpu"
            return rank, world_size, device
        else:
            logger.info("WORLD_SIZE is 1. Running in single-process mode.")
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            if torch.cuda.is_available():
                torch.cuda.set_device(device)
            return 0, 1, device
    else:
        logger.info("Distributed environment variables not found. Running in single-process mode.")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
        return 0, 1, device

def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    step: int,
    stage_name: str,
    config: Config,
    is_best: bool = False,
    is_latest: bool = False,
    rank: int = 0,
) -> None:
    """
    Saves the current state of the model, optimizer, learning rate scheduler,
    and training progress. Only executed by the rank 0 process.

    Args:
        model: The model to save. Can be DDP-wrapped or not.
        optimizer: The optimizer state to save.
        scheduler: The learning rate scheduler state to save.
        step: The current training step.
        stage_name: The name of the current training stage (e.g., "stage_1_1").
        config: The configuration object, containing `checkpoint_dir`.
        is_best: If True, also save a copy as 'best.pt'.
        is_latest: If True, also save a copy as 'latest.pt'.
        rank: The global rank of the current process. Checkpoint is saved only by rank 0.
    """
    if rank != 0:
        return

    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Unwrap DDP model if necessary
    model_to_save = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model

    checkpoint_dict = {
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step,
        "stage_name": stage_name,
        "config_snapshot": config.__dict__, # Save the full config snapshot for reproducibility
    }

    model_variant_name = config.get("model_variant_name", "navil_model") # Assuming this attribute exists in Config
    checkpoint_filename = f"{model_variant_name}_{stage_name}_step_{step}.pt"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

    torch.save(checkpoint_dict, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")

    if is_latest:
        latest_path = os.path.join(checkpoint_dir, "latest.pt")
        torch.save(checkpoint_dict, latest_path) # Overwrite latest
        logger.info(f"Latest checkpoint updated to {latest_path}")

    if is_best:
        best_path = os.path.join(checkpoint_dir, "best.pt")
        torch.save(checkpoint_dict, best_path) # Overwrite best
        logger.info(f"Best checkpoint updated to {best_path}")

def load_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    checkpoint_path: str,
    device: str,
    rank: int = 0,
) -> Tuple[int, str]:
    """
    Loads a previously saved checkpoint into the model, optimizer, and scheduler.

    Args:
        model: The model to load state into. Can be DDP-wrapped or not.
        optimizer: The optimizer to load state into.
        scheduler: The learning rate scheduler to load state into.
        checkpoint_path: The path to the checkpoint file.
        device: The device to map the loaded tensors to (e.g., "cuda:0" or "cpu").
        rank: The global rank of the current process.

    Returns:
        A tuple containing:
            - loaded_step (int): The training step from the loaded checkpoint.
            - loaded_stage_name (str): The training stage from the loaded checkpoint.
    """
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint file not found at {checkpoint_path}. Starting from scratch.")
        return 0, "initial"

    try:
        # Load checkpoint data
        checkpoint_dict = torch.load(checkpoint_path, map_location=device)

        # Unwrap DDP model if necessary
        model_to_load = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model

        model_to_load.load_state_dict(checkpoint_dict["model_state_dict"])
        optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint_dict["scheduler_state_dict"])

        loaded_step = checkpoint_dict["step"]
        loaded_stage_name = checkpoint_dict["stage_name"]

        logger.info(f"Checkpoint loaded successfully from {checkpoint_path}. "
                    f"Resuming from step {loaded_step} in stage {loaded_stage_name}.")
        return loaded_step, loaded_stage_name

    except Exception as e:
        logger.error(f"Error loading checkpoint from {checkpoint_path}: {e}. Starting from scratch.")
        return 0, "initial"

def get_learning_rate_scheduler(
    optimizer: Optimizer,
    config: Config,
    current_stage_config: Dict[str, Any],
    total_steps: int,
    current_step_in_stage: int = 0,
) -> _LRScheduler:
    """
    Creates and returns a learning rate scheduler based on the configuration
    of the current training stage.

    Args:
        optimizer: The optimizer for which to create the scheduler.
        config: The global configuration object.
        current_stage_config: A dictionary containing the training parameters
                              for the current stage.
        total_steps: The total number of steps for the current training stage.
        current_step_in_stage: The current step count within the stage (for resuming).

    Returns:
        An instance of `torch.optim.lr_scheduler._LRScheduler`.
    """
    lr_schedule_type = current_stage_config.get("lr_schedule", "constant_with_warmup")
    peak_learning_rate = current_stage_config.get("peak_learning_rate")
    warmup_steps = config.get("common.warmup_steps", 200)

    if peak_learning_rate is None:
        raise ValueError(f"Peak learning rate must be specified for stage {lr_schedule_type}.")

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0 # Default for constant after warmup

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    if lr_schedule_type == "constant_with_warmup":
        # After warmup, LR remains constant at peak_learning_rate
        # For LambdaLR, if lr_lambda returns 1.0, it means LR = initial_lr * 1.0
        # So the optimizer's initial_lr should be peak_learning_rate.
        # However, LambdaLR applies a factor. We need to ensure the optimizer's LR
        # is set to peak_learning_rate AFTER warmup.
        # For simplicity, we can set base LR for optimizer to peak_learning_rate,
        # and warmup_scheduler multiplies it by a factor.
        # Alternatively, use SequentialLR
        
        # Adjust base LR for optimizer if it's not already peak_learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = peak_learning_rate

        def constant_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0

        scheduler = LambdaLR(optimizer, lr_lambda=constant_lambda)

    elif lr_schedule_type == "cosine_decay":
        # After warmup, cosine decay to a minimum LR (e.g., 0)
        # min_lr_ratio = config.get("common.min_lr_ratio", 0.0) # Assume min_lr_ratio is 0 if not specified
        # min_lr = peak_learning_rate * min_lr_ratio
        
        # The paper typically implies decay to 0 or a very small value.
        # Let's assume decay to 0 for `CosineAnnealingLR` for now, `eta_min=0`.
        
        # CosineAnnealingLR expects T_max (number of iterations for the cosine cycle)
        # after warmup.
        if total_steps <= warmup_steps:
            logger.warning("Total steps are not greater than warmup steps for cosine decay. LR will be constant after warmup.")
            return warmup_scheduler

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=(total_steps - warmup_steps),
            eta_min=0.0 # Decay to 0
        )
        
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        
    else:
        raise ValueError(f"Unknown LR schedule type: {lr_schedule_type}")

    # Set current step for scheduler if resuming
    if current_step_in_stage > 0:
        # Step the scheduler to the correct position if resuming
        # For SequentialLR, this needs careful handling.
        # A common practice is to create a dummy optimizer, step it, and copy state.
        # Or, just step the scheduler `current_step_in_stage` times.
        # For simplicity, let's step it. This might not be perfectly accurate for SequentialLR
        # if the milestones are relative to the scheduler itself.
        # A more robust solution might involve directly setting the LR.
        for _ in range(current_step_in_stage):
            scheduler.step()
        logger.info(f"LR scheduler stepped {current_step_in_stage} times to match resumed state.")

    return scheduler

def get_optimizer(
    trainable_params: List[nn.Parameter],
    config: Config,
    current_stage_config: Dict[str, Any],
) -> Optimizer:
    """
    Creates and returns an AdamW optimizer instance for the model's trainable parameters.

    Args:
        trainable_params: A list of model parameters that require gradients.
        config: The global configuration object.
        current_stage_config: A dictionary containing the training parameters
                              for the current stage.

    Returns:
        An instance of `torch.optim.Optimizer`.
    """
    optimizer_name = config.get("common.optimizer.name", "AdamW")
    beta1 = config.get("common.optimizer.beta1", 0.9)
    beta2 = config.get("common.optimizer.beta2", 0.95)
    eps = config.get("common.optimizer.eps", 1.0e-8)
    weight_decay = current_stage_config.get("weight_decay", 0.01)
    peak_learning_rate = current_stage_config.get("peak_learning_rate", 1e-5) # Initial LR for optimizer

    if optimizer_name.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=peak_learning_rate, # This LR will be managed by the scheduler
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_name}")

    logger.info(f"Optimizer {optimizer_name} initialized with initial LR: {peak_learning_rate}, "
                f"betas: ({beta1}, {beta2}), eps: {eps}, weight_decay: {weight_decay}.")
    return optimizer


# RoPE helper functions
# These are commonly implemented in attention modules, but provided here as utilities.

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates the input tensor by half its dimension along the last axis."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def _apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies rotary positional embeddings to the input tensor."""
    # cos and sin are (seq_len, head_dim) or (H*W, head_dim)
    # x is (..., seq_len, head_dim) or (..., H*W, head_dim)
    return (x * cos) + (_rotate_half(x) * sin)

def apply_rope_1d(query: torch.Tensor, key: torch.Tensor, seq_len: int, head_dim: int, rope_theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies 1D Rotary Positional Embeddings (RoPE) to query and key tensors.

    Args:
        query: Query tensor (batch_size, num_heads, seq_len, head_dim).
        key: Key tensor (batch_size, num_heads, seq_len, head_dim).
        seq_len: The sequence length.
        head_dim: The dimension of a single attention head.
        rope_theta: Hyperparameter for RoPE frequency calculation.

    Returns:
        A tuple of (query_rotated, key_rotated) tensors.
    """
    device, dtype = query.device, query.dtype

    # Generate inverse frequencies
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))

    # Generate positions (t)
    t = torch.arange(seq_len, device=device, dtype=torch.float32)

    # Compute frequencies and embeddings
    freqs = torch.outer(t, inv_freq) # (seq_len, head_dim // 2)
    emb = torch.cat((freqs, freqs), dim=-1).to(dtype) # (seq_len, head_dim)

    cos_emb = emb.cos()
    sin_emb = emb.sin()

    # Reshape cos/sin for broadcasting: (1, 1, seq_len, head_dim)
    cos_emb = cos_emb.view(1, 1, seq_len, head_dim)
    sin_emb = sin_emb.view(1, 1, seq_len, head_dim)

    # Apply RoPE
    query_rotated = _apply_rotary_pos_emb(query, cos_emb, sin_emb)
    key_rotated = _apply_rotary_pos_emb(key, cos_emb, sin_emb)

    return query_rotated, key_rotated

def apply_rope_2d(
    query: torch.Tensor,
    key: torch.Tensor,
    height: int,
    width: int,
    head_dim: int,
    rope_theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies 2D Rotary Positional Embeddings (RoPE) to query and key tensors
    for a 2D grid of tokens (e.g., from a visual encoder).

    This implementation generates separate 1D RoPEs for height and width dimensions
    and combines them to form 2D positional embeddings, which are then applied.
    The tokens are assumed to be flattened from a (H, W) grid to a (H*W) sequence.

    Args:
        query: Query tensor (batch_size, num_heads, H*W, head_dim).
        key: Key tensor (batch_size, num_heads, H*W, head_dim).
        height: Original height of the 2D grid.
        width: Original width of the 2D grid.
        head_dim: The dimension of a single attention head.
        rope_theta: Hyperparameter for RoPE frequency calculation.

    Returns:
        A tuple of (query_rotated, key_rotated) tensors.
    """
    device, dtype = query.device, query.dtype

    # Generate inverse frequencies
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))

    # Positions for height (y-axis)
    t_y = torch.arange(height, device=device, dtype=torch.float32)
    freqs_y = torch.outer(t_y, inv_freq) # (height, head_dim // 2)
    emb_y = torch.cat((freqs_y, freqs_y), dim=-1).to(dtype) # (height, head_dim)

    # Positions for width (x-axis)
    t_x = torch.arange(width, device=device, dtype=torch.float32)
    freqs_x = torch.outer(t_x, inv_freq) # (width, head_dim // 2)
    emb_x = torch.cat((freqs_x, freqs_x), dim=-1).to(dtype) # (width, head_dim)

    # Combine 1D embeddings to form 2D embeddings
    # Expand to (H, W, head_dim) for broadcasting later
    cos_emb_y = emb_y.cos().unsqueeze(1) # (height, 1, head_dim)
    sin_emb_y = emb_y.sin().unsqueeze(1) # (height, 1, head_dim)

    cos_emb_x = emb_x.cos().unsqueeze(0) # (1, width, head_dim)
    sin_emb_x = emb_x.sin().unsqueeze(0) # (1, width, head_dim)

    # For 2D RoPE, we combine rotations by element-wise multiplication of complex numbers
    # which translates to:
    # cos(a+b) = cos(a)cos(b) - sin(a)sin(b)
    # sin(a+b) = sin(a)cos(b) + cos(a)sin(b)

    cos_2d = cos_emb_y * cos_emb_x - sin_emb_y * sin_emb_x # (height, width, head_dim)
    sin_2d = sin_emb_y * cos_emb_x + cos_emb_y * sin_emb_x # (height, width, head_dim)

    # Flatten the 2D embeddings to match token sequence (H*W, head_dim)
    cos_2d = cos_2d.view(height * width, head_dim).view(1, 1, height * width, head_dim) # (1, 1, H*W, head_dim)
    sin_2d = sin_2d.view(height * width, head_dim).view(1, 1, height * width, head_dim) # (1, 1, H*W, head_dim)

    # Apply RoPE
    query_rotated = _apply_rotary_pos_emb(query, cos_2d, sin_2d)
    key_rotated = _apply_rotary_pos_emb(key, cos_2d, sin_2d)

    return query_rotated, key_rotated

def get_model_size_in_params(model: nn.Module) -> int:
    """
    Calculates the total number of trainable parameters in a PyTorch model.

    Args:
        model: The PyTorch model.

    Returns:
        The total number of trainable parameters.
    """
    # Sum only parameters that require gradients
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_numerical_precision(precision_str: str) -> torch.dtype:
    """
    Converts a string representation of numerical precision to a torch.dtype.

    Args:
        precision_str: A string representing the desired precision (e.g., "bfloat16", "float16", "float32").

    Returns:
        The corresponding torch.dtype.

    Raises:
        ValueError: If an unsupported precision string is provided.
    """
    if precision_str == "bfloat16":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            logger.warning("bfloat16 requested but not supported by hardware/CUDA. Falling back to float32.")
            return torch.float32
        return torch.bfloat16
    elif precision_str == "float16":
        return torch.float16
    elif precision_str == "float32":
        return torch.float32
    else:
        raise ValueError(f"Unsupported numerical precision: {precision_str}. "
                         "Supported options are 'bfloat16', 'float16', 'float32'.")

