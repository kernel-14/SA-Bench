# evaluation/metrics.py
"""Metric computation utilities for the MoE-POT evaluation pipeline.

Provides three standalone functions for measuring prediction quality and
inference performance:

1. l2_relative_error: Computes the L2 Relative Error (L2RE) between
   predicted and ground-truth PDE fields. This is the universal evaluation
   metric reported in all paper tables (Table 1, 2, 3, 12, etc.).

2. autoregressive_l2re: Computes per-timestep L2RE across a rollout
   sequence. Used for the rollout error analysis in Appendix C.3 (Table 12),
   which reports L2RE at frames 50, 70, and 100 for the SWE dataset.

3. compute_inference_time: Measures average single-step inference time in
   milliseconds using CUDA event timing. Used for Table 3 in the paper,
   which compares inference times across DPOT and MoE-POT variants.

Mathematical formulations from the paper:

    L2RE = ||pred - target||_2 / ||target||_2          (Section B.3)

    Rel-ℓ_2 = ||x_pred - x_gt||_2 / ||x_gt||_2        (Section B.3)

From config.yaml (evaluation section):
    metric: "L2RE"                  (l2_relative_error is this metric)
    rollout_steps: 10               (autoregressive_l2re sequence length)
    inference_warmup_runs: 10       (default warmup parameter)
    inference_timed_runs: 100       (default num_runs parameter)

This module has no dependencies on other project modules — only torch,
torch.nn, numpy, typing, and time. This prevents circular imports and
makes the module independently testable.
"""

import time
from typing import List

import numpy as np
import torch
import torch.nn as nn


def l2_relative_error(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """Computes the mean L2 Relative Error (L2RE) over a batch of PDE fields.

    Implements the primary evaluation metric from the paper (Section B.3):
        Rel-ℓ_2 = ||pred - target||_2 / ||target||_2

    The norm is computed over all non-batch dimensions jointly for each
    sample (full Euclidean norm over the flattened field), then averaged
    over the batch dimension. This is the universal metric reported in all
    paper tables (Table 1, 2, 3, 12, etc.).

    Computation pipeline:
        1. Flatten (B, ...) → (B, N) where N = product of all non-batch dims
        2. Numerator per sample: ||pred_b - target_b||_2, shape (B,)
        3. Denominator per sample: ||target_b||_2 + epsilon, shape (B,)
        4. Per-sample L2RE: numerator / denominator, shape (B,)
        5. Batch mean → scalar float

    Design notes:
      - Per-sample then average (not batch-aggregate then divide) matches
        the paper's evaluation protocol and prevents large-batch samples
        from dominating the metric.
      - Epsilon guard (1e-8) on denominator prevents division by zero for
        zero-padded channels or near-zero initial conditions.
      - Returns Python float (detached from computation graph, on CPU) for
        clean integration with logging and metric accumulation code.
      - Works for any shape (B, ...) — handles both (B, C, H, W) single-frame
        predictions and (B, T, C, H, W) multi-frame sequences.

    Args:
        pred: Predicted PDE field tensor of shape (B, ...) where B is the
            batch dimension and ... is any combination of spatial, channel,
            and temporal dimensions. In standard usage:
              - (B, C, H, W) for single-frame predictions from MoEPOT.forward()
              - (B, T, C, H, W) for multi-frame rollout sequences
            Must be on the same device as target.
        target: Ground-truth PDE field tensor of the same shape as pred.
            Must be on the same device as pred.

    Returns:
        Mean L2 relative error over the batch as a Python float. Values are
        in [0, ∞) where 0 indicates perfect prediction. Typical values
        during evaluation range from 0.001 (excellent) to 1.0+ (poor).
        Representative values from the paper (Table 1):
          - MoE-POT-S zero-shot on NS(1e-3): 0.00583
          - MoE-POT-S zero-shot on CNS(0.1,0.01): 0.00959
          - MoE-POT-S zero-shot on SWE: 0.00289

    Raises:
        RuntimeError: If pred and target are on different devices (raised
            by PyTorch during the subtraction operation).
    """
    # Small constant to prevent division by zero.
    # 1e-8 is appropriate for float32 PDE field magnitudes and consistent
    # with the epsilon used in Preprocessor.normalize() and L2RelativeLoss.
    epsilon: float = 1e-8

    batch_size: int = pred.shape[0]

    # ----------------------------------------------------------------
    # Step 1: Flatten all non-batch dimensions
    # ----------------------------------------------------------------
    # Reshape from (B, ...) to (B, N) where N = product of all non-batch dims.
    # This ensures torch.norm computes the full Euclidean norm over all
    # field elements jointly for each sample, matching the paper's ||·||_2
    # notation where the norm spans the entire spatial field.
    # Input:  (B, C, H, W) or (B, T, C, H, W) or any (B, ...)
    # Output: (B, N) where N = C*H*W or T*C*H*W
    pred_flat: torch.Tensor = pred.reshape(batch_size, -1)
    target_flat: torch.Tensor = target.reshape(batch_size, -1)

    # ----------------------------------------------------------------
    # Step 2: Compute per-sample numerator ||pred - target||_2
    # ----------------------------------------------------------------
    # diff_flat shape: (B, N)
    # numerator shape: (B,) — per-sample L2 norm of the prediction error
    # torch.norm with p=2, dim=-1 computes the L2 norm over the last
    # dimension (N) independently for each sample in the batch.
    diff_flat: torch.Tensor = pred_flat - target_flat
    numerator: torch.Tensor = torch.norm(diff_flat, p=2, dim=-1)
    # Shape: (B,)

    # ----------------------------------------------------------------
    # Step 3: Compute per-sample denominator ||target||_2 + epsilon
    # ----------------------------------------------------------------
    # target_norm shape: (B,)
    # Adding epsilon prevents division by zero for:
    #   - Padded channels filled with constant 1.0 (near-zero variation)
    #   - Zero-initialized outputs at the start of training
    #   - PDE fields with very small magnitudes (e.g., near-zero initial
    #     conditions in some SWE or DR configurations)
    target_norm: torch.Tensor = torch.norm(target_flat, p=2, dim=-1)
    # Shape: (B,)
    denominator: torch.Tensor = target_norm + epsilon
    # Shape: (B,)

    # ----------------------------------------------------------------
    # Step 4: Per-sample relative error
    # ----------------------------------------------------------------
    # rel_error shape: (B,) — values in [0, ∞)
    rel_error: torch.Tensor = numerator / denominator
    # Shape: (B,)

    # ----------------------------------------------------------------
    # Step 5: Batch mean → scalar float
    # ----------------------------------------------------------------
    # Average over the batch dimension to get a scalar.
    # .item() detaches from the computation graph and moves to CPU,
    # returning a Python float for clean integration with logging code.
    return float(rel_error.mean().item())


def autoregressive_l2re(
    pred_sequence: torch.Tensor,
    target_sequence: torch.Tensor,
) -> List[float]:
    """Computes per-timestep L2RE across a rollout sequence.

    Used for the rollout error analysis in Appendix C.3 (Table 12), which
    reports L2RE at frames 50, 70, and 100 for the SWE dataset to illustrate
    error accumulation in auto-regressive prediction.

    Iterates over the time dimension and calls l2_relative_error() at each
    timestep, returning a list of T float values. This allows callers to
    inspect the error at any specific frame (e.g., frame 50, 70, 100 as
    in Table 12) or compute statistics over the full rollout.

    From Table 12 in the paper (SWE dataset, 100-step rollout):
        DPOT-S:    Frame 50: 0.0031, Frame 70: 0.0034, Frame 100: 0.0051
        MoE-POT-S: Frame 50: 0.0024, Frame 70: 0.0026, Frame 100: 0.0035

    Args:
        pred_sequence: Predicted rollout sequence of shape (B, T, C, H, W)
            where:
              - B: Batch size
              - T: Number of rollout timesteps (e.g., 100 for SWE analysis,
                10 for standard evaluation per config.yaml rollout_steps: 10)
              - C: Number of channels = 4 (config.yaml architecture.max_channels)
              - H: Spatial height = 128 (config.yaml architecture.target_resolution)
              - W: Spatial width = 128
            Produced by Evaluator.autoregressive_rollout() via sliding-window
            auto-regressive prediction.
        target_sequence: Ground-truth rollout sequence of the same shape
            (B, T, C, H, W) as pred_sequence. Must be on the same device.

    Returns:
        List of T Python floats, where the t-th element is the mean L2RE
        at timestep t across the batch. The list has length T = pred_sequence.shape[1].
        Returns an empty list [] if T = 0 (valid edge case, no error raised).

        Example for T=100 SWE rollout:
            result[49]  → L2RE at frame 50 (0-indexed: frame 49)
            result[69]  → L2RE at frame 70
            result[99]  → L2RE at frame 100

    Raises:
        ValueError: If pred_sequence and target_sequence have different shapes.
        RuntimeError: If pred_sequence and target_sequence are on different
            devices (raised by PyTorch during the subtraction in l2_relative_error).
    """
    # ----------------------------------------------------------------
    # Input validation
    # ----------------------------------------------------------------
    if pred_sequence.shape != target_sequence.shape:
        raise ValueError(
            f"pred_sequence and target_sequence must have the same shape. "
            f"Got pred_sequence.shape={pred_sequence.shape} and "
            f"target_sequence.shape={target_sequence.shape}."
        )

    # Extract the number of timesteps T from dimension 1.
    t_steps: int = pred_sequence.shape[1]

    # Handle edge case: empty sequence.
    if t_steps == 0:
        return []

    # ----------------------------------------------------------------
    # Per-timestep L2RE computation
    # ----------------------------------------------------------------
    # Iterate over T timesteps and compute L2RE at each frame.
    # Reuses l2_relative_error() to avoid code duplication.
    l2re_per_timestep: List[float] = []

    t_idx: int
    for t_idx in range(t_steps):
        # Extract single-frame tensors at timestep t_idx.
        # pred_sequence[:, t_idx, :, :, :] shape: (B, C, H, W)
        # target_sequence[:, t_idx, :, :, :] shape: (B, C, H, W)
        pred_t: torch.Tensor = pred_sequence[:, t_idx, :, :, :]
        target_t: torch.Tensor = target_sequence[:, t_idx, :, :, :]

        # Compute mean L2RE over the batch at this timestep.
        # l2_relative_error returns a Python float.
        l2re_t: float = l2_relative_error(pred_t, target_t)
        l2re_per_timestep.append(l2re_t)

    return l2re_per_timestep


def compute_inference_time(
    model: nn.Module,
    input_tensor: torch.Tensor,
    num_runs: int = 100,
    warmup: int = 10,
    device: str = "cuda",
) -> float:
    """Measures average single-step inference time in milliseconds.

    Used for Table 3 in the paper, which compares inference times across
    DPOT and MoE-POT variants on the NS(1e-5) dataset:
        DPOT-Tiny (7.5M):   5.5 ms
        DPOT-Small (30M):   6.5 ms
        DPOT-Medium (158M): 16.7 ms
        DPOT-Large (493M):  24.3 ms
        MoE-POT-Tiny (17M): 8.8 ms
        MoE-POT-Small (90M): 12.7 ms
        MoE-POT-Medium (288M): 16.6 ms

    Uses CUDA event timing for accurate GPU benchmarking. CUDA operations
    are asynchronous — Python-level timing (time.time()) would be inaccurate
    because the Python call returns before the GPU finishes. CUDA events
    record timestamps on the GPU timeline, giving true kernel execution time.

    Warmup runs are essential to ensure:
      - CUDA kernels are compiled and cached (JIT compilation overhead)
      - GPU is at steady-state temperature
      - Memory is allocated (first-run allocation overhead)
    Without warmup, the first few runs would be artificially slow.

    Falls back to time.perf_counter() based CPU timing when CUDA is not
    available, making the function usable in testing environments without GPUs.

    From config.yaml (evaluation section):
        inference_warmup_runs: 10   (default warmup=10)
        inference_timed_runs: 100   (default num_runs=100)

    Args:
        model: The model to benchmark. Typically a MoEPOT instance.
            Will be set to eval mode (model.eval()) before timing.
            The model's forward() method is expected to return a tuple
            (u_pred, balance_loss) — the return value is discarded since
            we only care about timing.
        input_tensor: A representative input batch of shape (B, T, C, H, W).
            Should already be on the target device for accurate timing.
            Typically a single batch from the NS(1e-5) test DataLoader
            (as used in Table 3 of the paper).
        num_runs: Number of timed forward passes. Default 100, matching
            config.yaml evaluation.inference_timed_runs: 100. More runs
            give more stable timing estimates.
        warmup: Number of warmup forward passes before timing begins.
            Default 10, matching config.yaml evaluation.inference_warmup_runs: 10.
            These runs are not timed and are used to warm up the GPU.
        device: Target device string. Default 'cuda'. If CUDA is not
            available, falls back to CPU timing automatically.

    Returns:
        Average inference time per forward pass in milliseconds as a Python
        float. Returns 0.0 if num_runs <= 0 (guard against invalid input).

    Raises:
        RuntimeError: If the model forward pass fails (e.g., shape mismatch
            between input_tensor and model's expected input shape).
    """
    # Guard against invalid num_runs.
    if num_runs <= 0:
        return 0.0

    # ----------------------------------------------------------------
    # Step 1: Set model to eval mode
    # ----------------------------------------------------------------
    # eval() disables dropout and batch normalization training behavior.
    # This matches the inference setting used in the paper's Table 3.
    model.eval()

    # ----------------------------------------------------------------
    # Step 2: Determine timing method based on device availability
    # ----------------------------------------------------------------
    # Use CUDA event timing when CUDA is available (accurate GPU timing).
    # Fall back to time.perf_counter() for CPU-only environments.
    use_cuda_timing: bool = (
        torch.cuda.is_available()
        and device.startswith("cuda")
    )

    # Move input tensor to the target device if not already there.
    # This ensures the timing reflects actual inference conditions.
    target_device: torch.device = torch.device(device if use_cuda_timing else "cpu")
    input_on_device: torch.Tensor = input_tensor.to(target_device)

    # ----------------------------------------------------------------
    # Step 3: Warmup phase (not timed)
    # ----------------------------------------------------------------
    # Run warmup forward passes to:
    #   - Compile and cache CUDA kernels (JIT compilation)
    #   - Allocate GPU memory for intermediate activations
    #   - Bring GPU to steady-state temperature
    # Without warmup, the first few timed runs would be artificially slow.
    with torch.no_grad():
        warmup_idx: int
        for warmup_idx in range(warmup):
            # Discard the return value — we only care about side effects
            # (kernel compilation, memory allocation).
            _ = model(input_on_device)

    # Synchronize after warmup to ensure all warmup operations complete
    # before timing begins. This prevents warmup overhead from bleeding
    # into the first timed run.
    if use_cuda_timing:
        torch.cuda.synchronize()

    # ----------------------------------------------------------------
    # Step 4: Timed phase
    # ----------------------------------------------------------------
    # Collect per-run timing measurements.
    elapsed_times_ms: List[float] = []

    with torch.no_grad():
        run_idx: int
        for run_idx in range(num_runs):
            if use_cuda_timing:
                # --------------------------------------------------------
                # CUDA event timing (accurate GPU benchmarking)
                # --------------------------------------------------------
                # torch.cuda.Event with enable_timing=True records a
                # timestamp on the GPU timeline when .record() is called.
                # start.elapsed_time(end) returns the time between the two
                # events in milliseconds (float).
                #
                # torch.cuda.synchronize() after end.record() ensures the
                # GPU has finished all operations before we read the elapsed
                # time. Without synchronize(), the GPU might still be running
                # when we call elapsed_time(), giving incorrect results.
                start_event: torch.cuda.Event = torch.cuda.Event(
                    enable_timing=True
                )
                end_event: torch.cuda.Event = torch.cuda.Event(
                    enable_timing=True
                )

                # Record start timestamp on the GPU timeline.
                start_event.record()

                # Run the model forward pass.
                # MoEPOT.forward() returns (u_pred, balance_loss) tuple.
                # We discard the return value — only timing matters here.
                _ = model(input_on_device)

                # Record end timestamp on the GPU timeline.
                end_event.record()

                # Synchronize: wait for all GPU operations to complete.
                # This is CRITICAL for accurate per-run timing. Without
                # synchronize(), CUDA operations are asynchronous and the
                # Python thread would proceed before the GPU finishes,
                # making elapsed_time() return incorrect values.
                torch.cuda.synchronize()

                # Compute elapsed time in milliseconds.
                # start_event.elapsed_time(end_event) returns a float in ms.
                elapsed_ms: float = start_event.elapsed_time(end_event)
                elapsed_times_ms.append(elapsed_ms)

            else:
                # --------------------------------------------------------
                # CPU timing fallback (for testing without GPU)
                # --------------------------------------------------------
                # time.perf_counter() provides the highest-resolution timer
                # available on the platform. For CPU inference, this is
                # accurate since CPU operations are synchronous.
                start_time: float = time.perf_counter()

                # Run the model forward pass.
                _ = model(input_on_device)

                end_time: float = time.perf_counter()

                # Convert seconds to milliseconds.
                elapsed_ms = (end_time - start_time) * 1000.0
                elapsed_times_ms.append(elapsed_ms)

    # ----------------------------------------------------------------
    # Step 5: Compute and return average inference time
    # ----------------------------------------------------------------
    # Use numpy for robust mean computation over the list of floats.
    # np.mean() handles edge cases (e.g., all-zero list) gracefully.
    if not elapsed_times_ms:
        return 0.0

    avg_inference_time_ms: float = float(np.mean(elapsed_times_ms))

    return avg_inference_time_ms
