## utils/distributed.py
"""Distributed training utilities for Pyramidal Flow Matching.

Provides a clean, consistent interface over PyTorch's ``torch.distributed``
primitives for multi-GPU training on 128 NVIDIA A100 GPUs with NCCL backend,
as specified in configs/default.yaml (training.distributed.backend: "nccl").

All public functions degrade gracefully to single-process behavior when
distributed training is not initialized, enabling the same code to run in
both distributed and non-distributed contexts (unit tests, single-GPU
inference, etc.).

Usage:
    from utils.distributed import (
        init_distributed,
        get_rank,
        get_local_rank,
        get_world_size,
        is_main_process,
        reduce_tensor,
        barrier,
    )

    # In main.py, called first before any other component:
    init_distributed()
"""

import os
from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor

from utils.logging import get_logger

## ---------------------------------------------------------------------------
## Module-level logger
## ---------------------------------------------------------------------------
logger = get_logger(__name__)

## ---------------------------------------------------------------------------
## Module-level state
## ---------------------------------------------------------------------------
_initialized: bool = False


## ---------------------------------------------------------------------------
## Public API
## ---------------------------------------------------------------------------


def init_distributed() -> None:
    """Initializes the PyTorch distributed process group for multi-GPU training.

    Uses the NCCL backend as specified in configs/default.yaml
    (training.distributed.backend: "nccl"). Reads ``RANK``, ``LOCAL_RANK``,
    and ``WORLD_SIZE`` environment variables set by the launcher (torchrun,
    deepspeed launcher, or accelerate launch).

    Safe to call multiple times — subsequent calls are no-ops (idempotent).
    Safe to call in non-distributed contexts (single-GPU or CPU) — skips
    initialization when ``WORLD_SIZE`` is not set or equals 1.

    Raises:
        RuntimeError: If NCCL process group initialization fails (e.g., due
            to missing ``MASTER_ADDR`` or ``MASTER_PORT`` environment
            variables, or NCCL communication errors).
    """
    global _initialized

    # Idempotency guard: skip if already initialized
    if _initialized or dist.is_initialized():
        logger.warning(
            "init_distributed() called but distributed is already initialized. "
            "Skipping re-initialization."
        )
        _initialized = True
        return

    # Read launcher-set environment variables
    world_size_str: str = os.environ.get("WORLD_SIZE", "1")
    rank_str: str = os.environ.get("RANK", "0")
    local_rank_str: str = os.environ.get("LOCAL_RANK", "0")

    try:
        world_size: int = int(world_size_str)
        rank: int = int(rank_str)
        local_rank: int = int(local_rank_str)
    except ValueError as exc:
        raise RuntimeError(
            f"Failed to parse distributed environment variables: "
            f"WORLD_SIZE={world_size_str!r}, RANK={rank_str!r}, "
            f"LOCAL_RANK={local_rank_str!r}. "
            f"Ensure the training launcher sets these correctly."
        ) from exc

    # Single-process fallback: skip distributed init
    if world_size <= 1:
        logger.info(
            "WORLD_SIZE=%d — running in single-process mode. "
            "Distributed initialization skipped.",
            world_size,
        )
        _initialized = True
        return

    # Validate CUDA availability before NCCL init
    if not torch.cuda.is_available():
        raise RuntimeError(
            "NCCL backend requires CUDA, but torch.cuda.is_available() "
            "returned False. Cannot initialize distributed training."
        )

    # Validate required environment variables for rendezvous
    master_addr: str = os.environ.get("MASTER_ADDR", "")
    master_port: str = os.environ.get("MASTER_PORT", "")
    if not master_addr or not master_port:
        raise RuntimeError(
            "MASTER_ADDR and MASTER_PORT environment variables must be set "
            "for distributed training. "
            f"Got MASTER_ADDR={master_addr!r}, MASTER_PORT={master_port!r}. "
            "Ensure the training launcher (torchrun / deepspeed) sets these."
        )

    # Set the CUDA device for this process before NCCL init
    # This must happen before init_process_group to avoid NCCL device mismatches
    torch.cuda.set_device(local_rank)

    # Initialize NCCL process group
    # [Paper: configs/default.yaml training.distributed.backend: "nccl"]
    try:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize NCCL process group. "
            f"rank={rank}, world_size={world_size}, "
            f"MASTER_ADDR={master_addr}, MASTER_PORT={master_port}. "
            f"Original error: {exc}"
        ) from exc

    _initialized = True

    logger.info(
        "Distributed training initialized: rank=%d / world_size=%d, "
        "local_rank=%d, backend=nccl, device=cuda:%d",
        rank,
        world_size,
        local_rank,
        local_rank,
    )


def get_rank() -> int:
    """Returns the global rank of the current process.

    Safe to call before ``init_distributed()`` — returns 0 in that case,
    which is the correct behavior for single-process contexts.

    Returns:
        Integer global rank in [0, world_size). Returns 0 if distributed
        training is not initialized.

    Example:
        >>> rank = get_rank()
        >>> if rank == 0:
        ...     print("I am the main process")
    """
    if dist.is_initialized():
        return dist.get_rank()
    # Fallback: read from environment (set by launcher even before init)
    try:
        return int(os.environ.get("RANK", "0"))
    except ValueError:
        return 0


def get_local_rank() -> int:
    """Returns the local rank (GPU index on the current node).

    The local rank is used for ``torch.cuda.set_device()`` and for the
    VAE's long-video scatter logic across GPUs
    (config.vae.scatter_long_videos: true in configs/default.yaml).

    Returns:
        Integer local rank in [0, num_gpus_per_node). Returns 0 if the
        ``LOCAL_RANK`` environment variable is not set.

    Example:
        >>> device = torch.device(f"cuda:{get_local_rank()}")
    """
    try:
        return int(os.environ.get("LOCAL_RANK", "0"))
    except ValueError:
        return 0


def get_world_size() -> int:
    """Returns the total number of processes across all nodes.

    Safe to call before ``init_distributed()`` — returns 1 in that case.

    Returns:
        Integer total number of processes. Returns 1 if distributed
        training is not initialized.

    Example:
        >>> sampler = DistributedSampler(
        ...     dataset,
        ...     num_replicas=get_world_size(),
        ...     rank=get_rank(),
        ... )
    """
    if dist.is_initialized():
        return dist.get_world_size()
    # Fallback: read from environment
    try:
        return int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        return 1


def is_main_process() -> bool:
    """Returns True if the current process is the main (rank 0) process.

    Used to gate operations that should only happen once across all ranks:
    - Saving checkpoints (utils/checkpointing.py)
    - Writing to TensorBoard / W&B (utils/logging.py)
    - Writing evaluation outputs (evaluation/*)
    - Printing progress bars

    Returns:
        True if rank == 0, False otherwise.

    Example:
        >>> if is_main_process():
        ...     torch.save(state_dict, "checkpoint.pt")
    """
    return get_rank() == 0


def reduce_tensor(tensor: Tensor, op: str = "mean") -> Tensor:
    """All-reduces a tensor across all ranks and returns the result.

    Used by ``Trainer.train_step()`` to aggregate loss values for logging
    so that the logged loss reflects the average across all 128 GPUs.

    The input tensor must be on a CUDA device (NCCL requirement).

    Args:
        tensor: A CUDA tensor to reduce. Scalar tensors (0-dim) and
            multi-dimensional tensors are both supported.
        op: Reduction operation, one of:
            - ``"mean"``: All-reduce sum, then divide by world_size.
            - ``"sum"``: All-reduce sum only.
            Defaults to ``"mean"`` for loss aggregation.

    Returns:
        A new tensor containing the reduced value. The original tensor
        is not modified (a clone is used internally).

    Raises:
        ValueError: If ``op`` is not one of ``"mean"`` or ``"sum"``.
        RuntimeError: If the tensor is not on a CUDA device and distributed
            training is active (NCCL requires CUDA tensors).

    Example:
        >>> loss = compute_loss(batch)
        >>> loss_reduced = reduce_tensor(loss, op="mean")
        >>> if is_main_process():
        ...     log_metrics({"train/loss": loss_reduced.item()}, step=step)
    """
    if op not in ("mean", "sum"):
        raise ValueError(
            f"Unsupported reduction op: {op!r}. Must be 'mean' or 'sum'."
        )

    # Single-process fast path: no communication needed
    if not dist.is_initialized() or get_world_size() == 1:
        return tensor

    # NCCL requires CUDA tensors
    if not tensor.is_cuda:
        local_rank: int = get_local_rank()
        if torch.cuda.is_available():
            logger.warning(
                "reduce_tensor received a CPU tensor. Moving to cuda:%d "
                "for NCCL all-reduce. Consider moving the tensor to CUDA "
                "before calling reduce_tensor.",
                local_rank,
            )
            tensor = tensor.cuda(local_rank)
        else:
            raise RuntimeError(
                "reduce_tensor requires a CUDA tensor when distributed "
                "training is active (NCCL backend), but the input tensor "
                "is on CPU and CUDA is not available."
            )

    # Clone to avoid in-place modification of the caller's tensor
    reduced: Tensor = tensor.clone()

    # All-reduce: sum across all ranks
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)

    # Normalize to mean if requested
    if op == "mean":
        reduced = reduced / get_world_size()

    return reduced


def barrier() -> None:
    """Synchronizes all processes at a barrier point.

    Blocks until all ranks in the process group have called this function.
    Used to ensure all ranks are in sync before and after checkpoint saves,
    preventing race conditions where rank 0 saves a checkpoint while other
    ranks have already moved on to the next training step.

    No-op when distributed training is not initialized (single-process mode).

    Example:
        >>> if is_main_process():
        ...     save_checkpoint(model, optimizer, step, path)
        >>> barrier()  # All ranks wait here until rank 0 finishes saving
    """
    if dist.is_initialized():
        dist.barrier()


def cleanup_distributed() -> None:
    """Destroys the distributed process group and releases NCCL resources.

    Should be called at the end of training to cleanly shut down the
    distributed backend. Safe to call when distributed is not initialized
    (no-op in that case).

    Example:
        >>> try:
        ...     trainer.train()
        ... finally:
        ...     cleanup_distributed()
    """
    global _initialized

    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info(
            "Distributed process group destroyed (rank %d).", get_rank()
        )

    _initialized = False


def get_device() -> torch.device:
    """Returns the appropriate torch.device for the current process.

    Uses the local rank to select the correct CUDA device in multi-GPU
    training. Falls back to CPU if CUDA is not available.

    Returns:
        A ``torch.device`` pointing to ``cuda:<local_rank>`` if CUDA is
        available, or ``cpu`` otherwise.

    Example:
        >>> device = get_device()
        >>> model = model.to(device)
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda:{get_local_rank()}")
    return torch.device("cpu")
