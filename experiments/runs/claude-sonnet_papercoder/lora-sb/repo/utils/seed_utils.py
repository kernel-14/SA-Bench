## utils/seed_utils.py
"""Seed and device utilities for LoRA-SB reproduction experiments.

This module provides two module-level functions that support reproducibility
and device management across the entire experiment pipeline. The paper reports
results averaged over 3 random seeds [42, 43, 44] (config: reproducibility.seeds),
so deterministic behavior is essential for fair comparison across methods.

Typical usage:
    from utils.seed_utils import set_seed, get_device

    device = get_device()
    set_seed(42)
"""

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for all libraries to ensure reproducibility.

    Sets the random state for Python's built-in random module, NumPy, PyTorch
    CPU, and PyTorch CUDA. Also configures CuDNN for deterministic behavior,
    consistent with config.yaml's ``reproducibility.deterministic: true``.

    This function must be called at the start of each seed iteration in
    ExperimentRunner._run_single_seed(), before any model loading, data
    loading, or gradient estimation. This ensures:

    - Dataset sampling in load_init_subset() is reproducible per seed.
    - torch.svd_lowrank() (which uses randomized power iteration internally)
      produces consistent SVD decompositions across runs.
    - Training data shuffling in DataLoader is consistent per seed.
    - All stochastic operations in model initialization are deterministic.

    Note:
        Setting ``torch.backends.cudnn.benchmark = False`` may slightly reduce
        throughput on variable-length sequences but is required for full
        determinism. This is acceptable given the paper's focus on
        reproducibility over raw speed.

    Args:
        seed: The integer random seed to use. The paper uses seeds [42, 43, 44]
            as specified in config.yaml under ``reproducibility.seeds``.
            Defaults to 42.
    """
    # 1. Python built-in random — used by dataset sampling utilities
    random.seed(seed)

    # 2. NumPy — used by HuggingFace datasets internally and evaluate metrics
    np.random.seed(seed)

    # 3. PyTorch CPU — covers all CPU tensor operations
    torch.manual_seed(seed)

    # 4. PyTorch CUDA — covers all GPU tensor operations across all devices.
    #    Using manual_seed_all (not manual_seed) is safer for multi-GPU
    #    environments and respects CUDA_VISIBLE_DEVICES.
    torch.cuda.manual_seed_all(seed)

    # 5. CuDNN determinism — required by config.yaml reproducibility.deterministic: true.
    #    deterministic=True forces CuDNN to use deterministic algorithms.
    #    benchmark=False disables the auto-tuner that selects non-deterministic
    #    algorithms based on input shapes.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return the appropriate compute device for the current environment.

    The paper runs all experiments on a single NVIDIA A6000 GPU
    (config.yaml: ``hardware.gpu: a6000``). This function returns a CUDA
    device if available, falling back to CPU for environments without a GPU.

    Using ``torch.device("cuda")`` (without an explicit index) defers to
    PyTorch's default device selection, which correctly respects the
    ``CUDA_VISIBLE_DEVICES`` environment variable. This is preferable to
    hardcoding ``"cuda:0"`` which could fail if the environment remaps devices.

    Returns:
        A ``torch.device`` instance pointing to ``"cuda"`` if a CUDA-capable
        GPU is available, otherwise ``"cpu"``.

    Example:
        >>> device = get_device()
        >>> print(device)
        device(type='cuda')  # on a machine with a GPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
