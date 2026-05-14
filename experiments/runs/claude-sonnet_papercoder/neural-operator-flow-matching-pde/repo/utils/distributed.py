## utils/distributed.py
"""Distributed training utilities for DDP setup and inter-process communication.

This module provides clean wrappers around torch.distributed primitives,
supporting the 4x H-100 GPU DDP training setup described in the paper.
All functions degrade gracefully to no-ops or safe defaults when called
in non-distributed (single-GPU or CPU) contexts.
"""

import os
from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor


def setup_ddp(
    rank: int,
    world_size: int,
    backend: str = "nccl",
    master_addr: str = "localhost",
    master_port: str = "12355",
) -> None:
    """Initialize the distributed process group for DDP training.

    Sets MASTER_ADDR and MASTER_PORT environment variables if not already
    present (preserving cluster-injected values from SLURM/PBS schedulers),
    then initializes the NCCL process group and binds the current process
    to its corresponding GPU.

    Args:
        rank: Rank of the current process (0 to world_size - 1).
        world_size: Total number of processes in the distributed group.
        backend: Distributed backend. Defaults to 'nccl' for NVIDIA H-100 GPUs
            as specified in config.yaml (p2vae.hardware.gpu_type: H100).
        master_addr: Default master address if MASTER_ADDR env var is not set.
        master_port: Default master port if MASTER_PORT env var is not set.

    Raises:
        RuntimeError: If the process group is already initialized (double init).
    """
    if dist.is_initialized():
        return

    # Set environment variables only if not already provided by the scheduler.
    # Cluster environments (SLURM, PBS) inject these; local runs need defaults.
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = master_addr

    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = master_port

    # Initialize the process group. For NVIDIA H-100 GPUs (config.yaml:
    # p2vae.hardware.gpu_type: H100), NCCL is the optimal backend.
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )

    # Bind this process to its corresponding GPU device.
    # Without this, all processes default to GPU 0 causing OOM errors.
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)


def cleanup_ddp() -> None:
    """Tear down the distributed process group cleanly.

    Guards against double-cleanup by checking initialization state first.
    Safe to call in non-distributed contexts (evaluation, preprocessing).
    Without cleanup, spawned processes may hang or leave zombie processes.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    """Return the rank of the current process.

    Returns:
        Rank of the current process (0 to world_size - 1), or 0 if the
        distributed group is not initialized (single-GPU / CPU context).
    """
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Return the total number of processes in the distributed group.

    Returns:
        World size if distributed is initialized, otherwise 1. The fallback
        of 1 ensures the same code path works for single-GPU debugging without
        conditional branching in the trainers.
    """
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Return True only for the main process (rank 0).

    Used to gate logging, checkpoint saving, and wandb initialization so
    only one process performs these operations. All three trainers use this
    pattern: ``if is_main_process(): wandb.log(...)``.

    Returns:
        True if the current process is rank 0 or if distributed is not
        initialized (single-process context is always the main process).
    """
    return get_rank() == 0


def reduce_tensor(tensor: Tensor, world_size: Optional[int] = None) -> Tensor:
    """Average a tensor metric across all GPU processes via all-reduce.

    Used to compute globally-averaged validation metrics (L2RE, VRMSE) that
    are computed locally on each GPU's shard of the validation set. Uses
    all_reduce (not reduce to rank 0) so every process has the averaged
    metric; is_main_process() then gates the actual logging call.

    Args:
        tensor: Metric tensor to average. Must be on a CUDA device when
            distributed is initialized. Scalar tensors (shape []) are
            supported for loss averaging.
        world_size: Number of processes to average over. If None, uses
            get_world_size() automatically.

    Returns:
        A new tensor containing the mean value across all processes.
        Returns the input tensor unchanged if distributed is not initialized
        (no-op for single-GPU contexts).
    """
    if not dist.is_initialized():
        return tensor

    effective_world_size: int = world_size if world_size is not None else get_world_size()

    # Clone to avoid in-place modification of the caller's tensor.
    reduced: Tensor = tensor.clone()

    # Sum across all processes, then divide to get the mean.
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced = reduced / effective_world_size

    return reduced


def barrier() -> None:
    """Synchronize all processes at a barrier point.

    Used to ensure all processes finish a validation pass or data loading
    step before rank 0 saves a checkpoint. Safe to call in non-distributed
    contexts (no-op when distributed is not initialized).
    """
    if dist.is_initialized():
        dist.barrier()
