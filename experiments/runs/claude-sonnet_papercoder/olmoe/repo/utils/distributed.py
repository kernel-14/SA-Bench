## utils/distributed.py
"""Distributed training utilities for OLMoE using PyTorch FSDP.

Provides the distributed training infrastructure for OLMoE-1B-7B pretraining
and adaptation. Wraps PyTorch's FullyShardedDataParallel (FSDP) with ZeRO-3
sharding, NCCL backend initialization, and utility helpers for rank-aware
logging and metric aggregation.

Hardware configuration (from config.yaml):
    pretraining.num_gpus: 256          # H100 GPUs
    pretraining.gpu_type: "H100"
    pretraining.interconnect_intra: "NVLink"
    pretraining.interconnect_inter: "InfiniBand"
    pretraining.fsdp: true
    pretraining.zero_stage: 3
    pretraining.bf16: true
    pretraining.fp32_reduce: true      # gradient_reduce_dtype: fp32 (Table 10)

FSDP configuration derived from config.yaml:
    - ShardingStrategy.FULL_SHARD (ZeRO-3): shards params, grads, optimizer states
    - MixedPrecision: param_dtype=bfloat16, reduce_dtype=float32, buffer_dtype=bfloat16
    - Auto-wrap policy: wrap at OLMoEBlock level (one block = one FSDP unit)
    - BackwardPrefetch.BACKWARD_PRE: overlaps communication with computation
    - sync_module_states=True: broadcast rank 0 weights to all ranks at init

Usage pattern in main.py:
    DistributedUtils.init_process_group()
    model = OLMoEModel(config)
    model = DistributedUtils.setup_fsdp(model, training_config)
    # ... training loop ...
    DistributedUtils.barrier()

Usage pattern in trainer.py:
    if DistributedUtils.is_main_process():
        wandb.log(metrics)
    reduced = DistributedUtils.all_reduce_dict(metrics)
"""

import logging
import os
from functools import partial
from typing import Dict, Optional, Set, Type

import torch
import torch.distributed
import torch.nn as nn
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    FullyShardedDataParallel,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from config import TrainingConfig

logger = logging.getLogger(__name__)


class DistributedUtils:
    """Static utility class for distributed training with PyTorch FSDP.

    All methods are @staticmethod — no instance state is maintained.
    This class groups related distributed utilities under a single namespace
    and provides single-GPU fallbacks for all operations so the same code
    runs on 1 or 256 GPUs without modification.

    Typical usage sequence:
        1. DistributedUtils.init_process_group()   # at program start
        2. model = DistributedUtils.setup_fsdp(model, config)  # after model init
        3. DistributedUtils.is_main_process()      # guard logging/saving
        4. DistributedUtils.all_reduce_dict(metrics)  # aggregate metrics
        5. DistributedUtils.barrier()              # synchronize ranks

    All methods degrade gracefully to no-ops or identity operations when
    running on a single GPU (world_size=1) or when the process group has
    not been initialized.
    """

    @staticmethod
    def init_process_group(backend: str = "nccl") -> None:
        """Initialize the distributed process group for multi-GPU training.

        Reads rank, world size, and master address from environment variables
        set by the launcher (torchrun / torch.distributed.launch). Binds the
        current process to its local GPU via torch.cuda.set_device().

        Environment variables read (set by torchrun):
            RANK:        Global rank of this process (0 to world_size-1)
            LOCAL_RANK:  Local rank on this node (0 to num_gpus_per_node-1)
            WORLD_SIZE:  Total number of processes across all nodes
            MASTER_ADDR: IP address of the rank-0 node
            MASTER_PORT: Port for the rendezvous server on rank-0

        Safe to call multiple times — returns immediately if already initialized.
        Safe to call on single-GPU setups — initializes a trivial group with
        world_size=1.

        Args:
            backend: Distributed backend to use. Default "nccl" is required
                     for GPU-to-GPU communication on H100s with NVLink/InfiniBand
                     (config.yaml: pretraining.interconnect_intra: "NVLink",
                     pretraining.interconnect_inter: "InfiniBand").
                     Use "gloo" for CPU-only debugging.

        Raises:
            RuntimeError: If CUDA is not available and backend is "nccl".
        """
        # Guard: skip if already initialized (safe for repeated calls).
        if torch.distributed.is_initialized():
            logger.debug(
                "Process group already initialized. "
                f"rank={torch.distributed.get_rank()}, "
                f"world_size={torch.distributed.get_world_size()}"
            )
            return

        # Validate CUDA availability for NCCL backend.
        if backend == "nccl" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available but backend='nccl' was requested. "
                "NCCL requires NVIDIA GPUs. "
                "Use backend='gloo' for CPU-only debugging."
            )

        # Read distributed configuration from environment variables.
        # torchrun sets these automatically; provide defaults for single-GPU runs.
        local_rank: int = int(os.environ.get("LOCAL_RANK", "0"))
        rank: int = int(os.environ.get("RANK", "0"))
        world_size: int = int(os.environ.get("WORLD_SIZE", "1"))

        # Set the CUDA device for this process BEFORE init_process_group.
        # This is critical for NCCL: each process must be bound to its local GPU
        # so that NCCL can use the correct NVLink/PCIe topology.
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            logger.debug(
                f"Set CUDA device to local_rank={local_rank} "
                f"(GPU {torch.cuda.current_device()}: "
                f"{torch.cuda.get_device_name(local_rank)})"
            )

        # Initialize the process group.
        # init_method="env://" reads MASTER_ADDR and MASTER_PORT from environment.
        # For single-GPU runs (world_size=1), this creates a trivial group.
        torch.distributed.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )

        logger.info(
            f"Distributed process group initialized: "
            f"backend={backend}, "
            f"rank={rank}, "
            f"local_rank={local_rank}, "
            f"world_size={world_size}, "
            f"master_addr={os.environ.get('MASTER_ADDR', 'localhost')}, "
            f"master_port={os.environ.get('MASTER_PORT', '29500')}"
        )

    @staticmethod
    def get_rank() -> int:
        """Return the global rank of the current process.

        The global rank uniquely identifies this process across all nodes.
        Rank 0 is the "main process" used for logging, checkpoint saving,
        and wandb initialization.

        Returns:
            Global rank in [0, world_size). Returns 0 if the process group
            has not been initialized (single-GPU fallback).

        Example:
            >>> DistributedUtils.get_rank()
            0  # on rank-0 process
            >>> DistributedUtils.get_rank()
            3  # on rank-3 process (e.g., 4th GPU on first node)
        """
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        return 0

    @staticmethod
    def get_world_size() -> int:
        """Return the total number of processes in the distributed group.

        The world size equals the total number of GPUs used for training.
        For OLMoE-1B-7B pretraining: 256 (config.yaml: pretraining.num_gpus).

        Returns:
            Total number of processes. Returns 1 if the process group has
            not been initialized (single-GPU fallback).

        Example:
            >>> DistributedUtils.get_world_size()
            256  # full OLMoE-1B-7B pretraining setup
            >>> DistributedUtils.get_world_size()
            1    # single-GPU debugging
        """
        if torch.distributed.is_initialized():
            return torch.distributed.get_world_size()
        return 1

    @staticmethod
    def is_main_process() -> bool:
        """Return True if this is the main (rank-0) process.

        Used throughout the codebase to guard operations that should only
        happen once across all ranks:
            - wandb.init() in utils/logging_utils.py
            - Checkpoint file writing in utils/checkpoint.py
            - Console logging in training/trainer.py
            - Evaluation result printing

        Returns:
            True if rank == 0, False otherwise. Always True for single-GPU
            runs (rank defaults to 0).

        Example:
            >>> if DistributedUtils.is_main_process():
            ...     wandb.log({"loss": 1.23})
            ...     torch.save(checkpoint, "checkpoint.pt")
        """
        return DistributedUtils.get_rank() == 0

    @staticmethod
    def barrier() -> None:
        """Synchronize all processes at a barrier point.

        Blocks until all processes in the distributed group have reached
        this call. Used to ensure all ranks complete an operation before
        any rank proceeds.

        Common usage patterns:
            1. After checkpoint saving: rank 0 writes files, all ranks wait
               before proceeding to ensure the checkpoint is fully written
               before any rank might try to read it.
            2. After model initialization: ensure all ranks have completed
               setup before training begins.
            3. After evaluation: ensure all ranks finish eval before the
               next training step begins.

        No-op if the process group has not been initialized (single-GPU).

        Example:
            >>> if DistributedUtils.is_main_process():
            ...     save_checkpoint(model, "checkpoint.pt")
            >>> DistributedUtils.barrier()  # all ranks wait here
            >>> # safe to proceed — checkpoint is fully written
        """
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    @staticmethod
    def all_reduce_dict(metrics: Dict[str, float]) -> Dict[str, float]:
        """Average metric values across all ranks via all-reduce.

        Performs a SUM all-reduce on all metric values and divides by
        world_size to compute the mean. This gives rank 0 (and all ranks)
        the true global average of each metric across all GPUs.

        Must be called on ALL ranks simultaneously with the same set of keys.
        This is a collective operation — if any rank skips it, all other
        ranks will hang waiting for the missing rank.

        Implementation uses a single batched all-reduce for efficiency:
        all metric values are stacked into one tensor, reduced in one
        communication round, then unstacked. This minimizes communication
        overhead compared to one all-reduce per metric.

        Args:
            metrics: Dictionary mapping metric names to float values.
                     Each rank should pass the same keys (e.g., 'ce_loss',
                     'lb_loss', 'rz_loss', 'grad_norm'). Values may differ
                     across ranks (e.g., different mini-batch losses).
                     Pass 0.0 for metrics not computed on a given rank.

        Returns:
            Dictionary with the same keys as input, but values replaced
            by the mean across all ranks. Returns the input dict unchanged
            if world_size == 1 (no communication needed).

        Example:
            >>> # On rank 0: loss = 2.1, On rank 1: loss = 1.9
            >>> metrics = {"ce_loss": 2.1, "grad_norm": 0.5}
            >>> reduced = DistributedUtils.all_reduce_dict(metrics)
            >>> reduced["ce_loss"]  # mean of 2.1 and 1.9
            2.0

        Note:
            The returned dict always contains float values (via .item()),
            not tensors. This is safe for logging to wandb and console.
        """
        world_size: int = DistributedUtils.get_world_size()

        # Fast path: single-GPU or uninitialized — no communication needed.
        if world_size == 1 or not torch.distributed.is_initialized():
            return metrics

        # Fast path: empty metrics dict.
        if not metrics:
            return metrics

        # -----------------------------------------------------------------------
        # Batched all-reduce: stack all values into one tensor for efficiency.
        # One communication round instead of len(metrics) rounds.
        # -----------------------------------------------------------------------
        keys: list = list(metrics.keys())

        # Build a float32 tensor on the current CUDA device.
        # float32 is used for precision — metric values may be small (e.g., 1e-4)
        # and BF16 would lose precision in the reduction.
        device: torch.device = torch.device(
            f"cuda:{torch.cuda.current_device()}"
            if torch.cuda.is_available()
            else "cpu"
        )
        values_tensor: torch.Tensor = torch.tensor(
            [metrics[k] for k in keys],
            dtype=torch.float32,
            device=device,
        )

        # All-reduce: sum across all ranks.
        # After this call, all ranks have the same tensor containing the sum.
        torch.distributed.all_reduce(
            values_tensor,
            op=torch.distributed.ReduceOp.SUM,
        )

        # Divide by world_size to get the mean.
        values_tensor = values_tensor / world_size

        # Unstack back into a dict with Python float values.
        reduced: Dict[str, float] = {
            k: values_tensor[i].item()
            for i, k in enumerate(keys)
        }

        return reduced

    @staticmethod
    def setup_fsdp(
        model: nn.Module,
        config: TrainingConfig,
        transformer_layer_cls: Optional[Set[Type]] = None,
    ) -> nn.Module:
        """Wrap the OLMoE model with FSDP for ZeRO-3 distributed training.

        Applies PyTorch FullyShardedDataParallel (FSDP) with the configuration
        specified in config.yaml (pretraining section):
            - ZeRO-3 sharding: shards parameters, gradients, and optimizer states
              across all GPUs (config.yaml: pretraining.zero_stage: 3)
            - BF16 mixed precision: parameters in BF16, gradient reduction in FP32
              (config.yaml: pretraining.bf16: true, pretraining.fp32_reduce: true)
            - Auto-wrap at OLMoEBlock level: each transformer block is an
              independent FSDP unit for optimal memory distribution
            - Backward prefetch: overlaps parameter gathering with backward pass
            - sync_module_states=True: broadcasts rank-0 weights to all ranks

        If config.fsdp is False or world_size == 1, skips FSDP wrapping and
        moves the model to the current CUDA device. This allows single-GPU
        debugging without code changes.

        FSDP wrapping granularity:
            Wrapping at OLMoEBlock level is optimal for OLMoE because:
            - Each block contains 64 experts (~402M params), providing good
              memory distribution across 256 GPUs
            - Per-expert wrapping would increase FSDP communication overhead
            - Whole-model wrapping would reduce parallelism benefits

        Mixed precision configuration (from config.yaml and Table 10):
            param_dtype=bfloat16:   Model parameters stored in BF16 during forward
            reduce_dtype=float32:   Gradient all-reduce in FP32 for numerical stability
                                    (config.yaml: pretraining.gradient_reduce_dtype: fp32)
            buffer_dtype=bfloat16:  Buffers (e.g., RoPE cos/sin caches) in BF16

        Args:
            model: The OLMoEModel to wrap. Must be on CPU or the current CUDA
                   device. Should NOT already be FSDP-wrapped.
                   _init_weights() must have been called before wrapping.
            config: TrainingConfig instance. Key fields used:
                    - fsdp (True): whether to apply FSDP wrapping
                    - zero_stage (3): ZeRO stage (maps to FULL_SHARD)
                    - bf16 (True): whether to use BF16 mixed precision
                    - fp32_reduce (True): whether to reduce gradients in FP32
            transformer_layer_cls: Set of nn.Module classes to use as FSDP
                    wrapping boundaries. If None, defaults to {OLMoEBlock}.
                    Pass a custom set for ablation experiments with different
                    architectures. Using a set allows wrapping multiple classes
                    (e.g., {OLMoEBlock, SomeOtherBlock}).

        Returns:
            The model wrapped with FSDP (if config.fsdp=True and world_size>1),
            or the original model moved to the current CUDA device (otherwise).
            The returned model has the same interface as the input model for
            forward() calls.

        Raises:
            RuntimeError: If CUDA is not available and config.fsdp is True.
            ValueError: If config.zero_stage is not 3 (only ZeRO-3 is supported).

        Example:
            >>> config = TrainingConfig()
            >>> model = OLMoEModel(OLMoEConfig())
            >>> model = DistributedUtils.setup_fsdp(model, config)
            >>> isinstance(model, FullyShardedDataParallel)
            True  # when fsdp=True and world_size > 1
        """
        # -----------------------------------------------------------------------
        # Validate prerequisites.
        # -----------------------------------------------------------------------
        if config.fsdp and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available but config.fsdp=True. "
                "FSDP with NCCL backend requires NVIDIA GPUs. "
                "Set config.fsdp=False for CPU-only debugging."
            )

        if config.fsdp and config.zero_stage != 3:
            raise ValueError(
                f"Only ZeRO-3 (zero_stage=3) is supported, "
                f"got zero_stage={config.zero_stage}. "
                f"ZeRO-3 maps to ShardingStrategy.FULL_SHARD in FSDP."
            )

        # -----------------------------------------------------------------------
        # Single-GPU or FSDP disabled: move to device and return.
        # -----------------------------------------------------------------------
        world_size: int = DistributedUtils.get_world_size()

        if not config.fsdp or world_size == 1:
            if torch.cuda.is_available():
                device: torch.device = torch.device(
                    f"cuda:{torch.cuda.current_device()}"
                )
                model = model.to(device)
                logger.info(
                    f"FSDP disabled (config.fsdp={config.fsdp}, "
                    f"world_size={world_size}). "
                    f"Model moved to {device}."
                )
            else:
                logger.info(
                    "FSDP disabled and CUDA unavailable. "
                    "Model remains on CPU (debugging mode)."
                )
            return model

        # -----------------------------------------------------------------------
        # Determine the auto-wrap policy.
        #
        # Import OLMoEBlock locally to avoid circular imports at module load time.
        # utils/distributed.py is imported before model/olmoe_model.py in some
        # initialization sequences (e.g., when setting up logging before model).
        # -----------------------------------------------------------------------
        if transformer_layer_cls is None:
            # Local import to avoid circular dependency at module level.
            # model/olmoe_model.py imports from model/* but not from utils/,
            # so this local import is safe.
            from model.olmoe_model import OLMoEBlock  # noqa: PLC0415
            wrap_cls: Set[Type] = {OLMoEBlock}
        else:
            wrap_cls = transformer_layer_cls

        # Build the auto-wrap policy using transformer_auto_wrap_policy.
        # This wraps each instance of the specified classes as an independent
        # FSDP unit. For OLMoE-1B-7B: wraps each of the 16 OLMoEBlocks.
        auto_wrap: partial = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=wrap_cls,
        )

        logger.info(
            f"FSDP auto-wrap policy: wrapping classes = "
            f"{[cls.__name__ for cls in wrap_cls]}"
        )

        # -----------------------------------------------------------------------
        # Configure mixed precision.
        #
        # From config.yaml (pretraining section) and Table 10:
        #   bf16: true                -> param_dtype = bfloat16
        #   fp32_reduce: true         -> reduce_dtype = float32
        #   gradient_reduce_dtype: fp32 -> reduce_dtype = float32
        #   optimizer_state_dtype: fp32 -> handled by FSDP default (FP32 states)
        #
        # BF16 parameters: reduces memory by ~2x vs FP32 during forward/backward
        # FP32 gradient reduction: prevents precision loss in all-reduce across
        #   256 GPUs where small gradient values could underflow in BF16
        # BF16 buffers: RoPE cos/sin caches, running stats — don't need FP32
        # -----------------------------------------------------------------------
        if config.bf16:
            mixed_precision_config: MixedPrecision = MixedPrecision(
                param_dtype=torch.bfloat16,   # Parameters in BF16 (config.yaml: bf16: true)
                reduce_dtype=torch.float32,    # Gradient reduction in FP32 (Table 10)
                buffer_dtype=torch.bfloat16,   # Buffers (RoPE caches, etc.) in BF16
            )
            logger.info(
                "FSDP MixedPrecision: "
                "param_dtype=bfloat16, "
                "reduce_dtype=float32, "
                "buffer_dtype=bfloat16"
            )
        else:
            # Full FP32 training (for debugging or ablations without BF16).
            mixed_precision_config = MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
            )
            logger.info(
                "FSDP MixedPrecision: all float32 (bf16=False)"
            )

        # -----------------------------------------------------------------------
        # Configure sharding strategy.
        #
        # ZeRO-3 (config.yaml: zero_stage: 3) maps to FULL_SHARD:
        #   - Parameters: sharded across all GPUs (gathered on-demand)
        #   - Gradients: sharded after backward pass
        #   - Optimizer states: sharded across all GPUs
        #
        # This provides maximum memory savings at the cost of more communication.
        # For 256 H100s with NVLink/InfiniBand, the communication overhead is
        # acceptable and the memory savings enable training 6.9B total params.
        # -----------------------------------------------------------------------
        sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
        logger.info(
            f"FSDP ShardingStrategy: FULL_SHARD (ZeRO-3, "
            f"config.zero_stage={config.zero_stage})"
        )

        # -----------------------------------------------------------------------
        # Configure backward prefetch.
        #
        # BACKWARD_PRE: prefetch the next layer's parameters during the current
        # layer's backward pass. This overlaps NCCL all-gather communication
        # with CUDA computation, improving throughput on H100s with InfiniBand.
        # -----------------------------------------------------------------------
        backward_prefetch: BackwardPrefetch = BackwardPrefetch.BACKWARD_PRE

        # -----------------------------------------------------------------------
        # Wrap the model with FSDP.
        #
        # Key parameters:
        #   device_id: Bind to the current CUDA device (set by init_process_group)
        #   sync_module_states=True: Broadcast rank-0 weights to all ranks.
        #     This is CRITICAL for correctness: without it, each rank initializes
        #     its own weights from the truncated normal distribution, leading to
        #     different starting points and divergent training.
        #   cpu_offload=None: No CPU offloading (H100s have sufficient HBM)
        # -----------------------------------------------------------------------
        current_device: int = torch.cuda.current_device()

        logger.info(
            f"Wrapping model with FSDP: "
            f"world_size={world_size}, "
            f"device=cuda:{current_device}, "
            f"sync_module_states=True"
        )

        wrapped_model: FullyShardedDataParallel = FullyShardedDataParallel(
            module=model,
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision_config,
            auto_wrap_policy=auto_wrap,
            backward_prefetch=backward_prefetch,
            device_id=current_device,
            sync_module_states=True,   # Broadcast rank-0 init weights to all ranks
            cpu_offload=None,          # No CPU offloading (H100 has sufficient HBM)
            limit_all_gathers=True,    # Limit concurrent all-gathers for memory efficiency
            use_orig_params=True,      # Preserve original param structure for optimizer
        )

        logger.info(
            f"FSDP wrapping complete: "
            f"rank={DistributedUtils.get_rank()}, "
            f"world_size={world_size}, "
            f"sharding=FULL_SHARD (ZeRO-3), "
            f"bf16={config.bf16}"
        )

        return wrapped_model

    @staticmethod
    def cleanup() -> None:
        """Destroy the distributed process group and release resources.

        Should be called at the end of training to cleanly shut down the
        NCCL communicators and free associated GPU memory. Failure to call
        this can cause hanging processes or resource leaks.

        Safe to call multiple times — no-op if not initialized.

        Example:
            >>> try:
            ...     train(model, data)
            ... finally:
            ...     DistributedUtils.cleanup()
        """
        if torch.distributed.is_initialized():
            rank: int = DistributedUtils.get_rank()
            world_size: int = DistributedUtils.get_world_size()
            torch.distributed.destroy_process_group()
            logger.info(
                f"Distributed process group destroyed: "
                f"rank={rank}, world_size={world_size}"
            )

    @staticmethod
    def get_local_rank() -> int:
        """Return the local rank of the current process on its node.

        The local rank identifies which GPU on the current node this process
        uses. For a node with 8 GPUs, local ranks are 0-7.

        Reads from the LOCAL_RANK environment variable set by torchrun.
        Returns 0 as a fallback for single-GPU or non-distributed runs.

        Returns:
            Local rank in [0, num_gpus_per_node). Returns 0 if LOCAL_RANK
            is not set in the environment.

        Example:
            >>> DistributedUtils.get_local_rank()
            3  # this process uses GPU 3 on its node
        """
        return int(os.environ.get("LOCAL_RANK", "0"))

    @staticmethod
    def get_node_rank() -> int:
        """Return the rank of the current node among all nodes.

        Computed as global_rank // local_world_size. For OLMoE-1B-7B with
        256 GPUs on 32 nodes (8 GPUs each), node ranks are 0-31.

        Returns:
            Node rank in [0, num_nodes). Returns 0 for single-node or
            non-distributed runs.

        Example:
            >>> DistributedUtils.get_node_rank()
            2  # this process is on the 3rd node (0-indexed)
        """
        global_rank: int = DistributedUtils.get_rank()
        local_rank: int = DistributedUtils.get_local_rank()

        # Avoid division by zero for single-GPU runs.
        if local_rank == 0 and global_rank == 0:
            return 0

        # Infer local world size from environment (set by torchrun).
        # LOCAL_WORLD_SIZE is the number of GPUs per node.
        local_world_size: int = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))

        if local_world_size <= 0:
            return 0

        return global_rank // local_world_size

    @staticmethod
    def print_rank0(message: str) -> None:
        """Print a message only on rank 0 to avoid duplicate output.

        Convenience wrapper around is_main_process() for simple console
        output during initialization and debugging. For structured logging,
        use utils/logging_utils.py instead.

        Args:
            message: The message to print. Only printed on rank 0.

        Example:
            >>> DistributedUtils.print_rank0(f"Training on {world_size} GPUs")
            Training on 256 GPUs  # only printed once, not 256 times
        """
        if DistributedUtils.is_main_process():
            print(message, flush=True)

    @staticmethod
    def synchronize_model_weights(model: nn.Module) -> None:
        """Broadcast model weights from rank 0 to all other ranks.

        Used when a model is initialized independently on each rank (e.g.,
        loaded from a checkpoint on rank 0 only) and needs to be synchronized
        before training. For FSDP-wrapped models, use sync_module_states=True
        in setup_fsdp() instead.

        This is a collective operation — must be called on all ranks.

        Args:
            model: The nn.Module whose parameters should be broadcast from
                   rank 0 to all other ranks. Must not be FSDP-wrapped
                   (FSDP handles its own synchronization internally).

        Note:
            For FSDP models, parameter synchronization is handled by
            sync_module_states=True in setup_fsdp(). This method is for
            non-FSDP models only (e.g., reference model in DPO that is
            not wrapped with FSDP).
        """
        if not torch.distributed.is_initialized():
            return

        # Broadcast all parameters from rank 0 to all other ranks.
        # This ensures all ranks start with identical weights.
        for param in model.parameters():
            torch.distributed.broadcast(param.data, src=0)

        # Broadcast all buffers (e.g., RoPE cos/sin caches, BN running stats).
        for buffer in model.buffers():
            torch.distributed.broadcast(buffer.data, src=0)

        logger.debug(
            f"Model weights synchronized from rank 0 to all ranks "
            f"(world_size={DistributedUtils.get_world_size()})"
        )
