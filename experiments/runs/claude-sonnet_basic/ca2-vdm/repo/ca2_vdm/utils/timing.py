"""
Timing utilities for Ca2-VDM efficiency analysis.

Measures generation time for autoregressive video generation,
reproducing the timing results from Section 4.3 (Table 5, Figure 6).
"""

import time
from contextlib import contextmanager
from typing import Dict, List, Optional

import torch


class TimingContext:
    """
    Context manager for timing GPU operations.

    Uses CUDA events for accurate GPU timing.
    """

    def __init__(self, use_cuda: bool = True):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.elapsed_ms = 0.0

    def __enter__(self):
        if self.use_cuda:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.use_cuda:
            self.end_event.record()
            torch.cuda.synchronize()
            self.elapsed_ms = self.start_event.elapsed_time(self.end_event)
        else:
            self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000

    @property
    def elapsed_seconds(self) -> float:
        return self.elapsed_ms / 1000.0


def measure_generation_time(
    model,
    first_frame: torch.Tensor,
    num_frames: int,
    num_denoising_steps: int = 100,
    use_kv_cache: bool = True,
    num_warmup: int = 1,
    num_runs: int = 3,
) -> Dict[str, float]:
    """
    Measure autoregressive generation time.

    Reproduces the timing measurements from Table 5 and Figure 6.

    Args:
        model: Ca2-VDM model.
        first_frame: Initial frame of shape (B, C, H, W).
        num_frames: Total frames to generate.
        num_denoising_steps: DDPM denoising steps.
        use_kv_cache: Whether to use KV-cache.
        num_warmup: Number of warmup runs.
        num_runs: Number of timed runs.

    Returns:
        Dict with timing statistics.
    """
    device = first_frame.device
    use_cuda = device.type == "cuda"

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            model.autoregressive_generate(
                first_frame=first_frame,
                num_frames=num_frames,
                num_denoising_steps=num_denoising_steps,
                use_kv_cache=use_kv_cache,
            )

    # Timed runs
    times = []
    for _ in range(num_runs):
        with TimingContext(use_cuda) as timer:
            with torch.no_grad():
                model.autoregressive_generate(
                    first_frame=first_frame,
                    num_frames=num_frames,
                    num_denoising_steps=num_denoising_steps,
                    use_kv_cache=use_kv_cache,
                )
        times.append(timer.elapsed_seconds)

    return {
        "mean_seconds": sum(times) / len(times),
        "min_seconds": min(times),
        "max_seconds": max(times),
        "all_times": times,
    }


def measure_cumulative_time(
    model,
    first_frame: torch.Tensor,
    chunk_size: int,
    max_ar_steps: int,
    num_denoising_steps: int = 100,
    use_kv_cache: bool = True,
) -> List[float]:
    """
    Measure cumulative generation time at each AR step.

    Reproduces Figure 6 from the paper.

    Args:
        model: Ca2-VDM model.
        first_frame: Initial frame of shape (B, C, H, W).
        chunk_size: l, frames per AR step.
        max_ar_steps: Maximum number of AR steps.
        num_denoising_steps: DDPM denoising steps.
        use_kv_cache: Whether to use KV-cache.

    Returns:
        List of cumulative times (seconds) at each AR step.
    """
    device = first_frame.device
    use_cuda = device.type == "cuda"

    cumulative_times = []
    total_time = 0.0

    for step in range(1, max_ar_steps + 1):
        num_frames = 1 + step * chunk_size  # 1 initial frame + step * l generated frames

        with TimingContext(use_cuda) as timer:
            with torch.no_grad():
                model.autoregressive_generate(
                    first_frame=first_frame,
                    num_frames=num_frames,
                    num_denoising_steps=num_denoising_steps,
                    use_kv_cache=use_kv_cache,
                )

        cumulative_times.append(timer.elapsed_seconds)

    return cumulative_times
