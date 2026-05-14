## utils.py
"""Shared utility functions for nGPT and GPT experiment reproduction.

This module provides foundational primitives used across all other modules in
the project. It has no dependencies on other project files, sitting at the
bottom of the dependency graph.

Key components:
    - Reproducibility: set_seed()
    - Device management: get_device()
    - Model inspection: count_parameters()
    - Learning rate scheduling: get_cosine_schedule_with_warmup()
    - Checkpoint I/O: save_checkpoint(), load_checkpoint()
    - Logging: setup_logger()
    - Metric tracking: AverageMeter
    - Weight normalization: normalize_matrix()
    - Analysis: compute_condition_number()
    - Visualization: plot_training_curves(), plot_parameter_distributions()
    - Constants: DTYPE_MAP

Typical usage:
    from utils import set_seed, get_device, AverageMeter, normalize_matrix

    set_seed(42)
    device = get_device()
    meter = AverageMeter("train_loss")
    meter.update(loss.item())
"""

import logging
import math
import os
import pathlib
import random
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Maps dtype string names (from config.yaml architecture.dtype) to torch dtypes.
# Used by model.py and trainer.py to resolve the configured dtype.
DTYPE_MAP: Dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    """Create a directory and all parent directories if they do not exist.

    Args:
        path: The directory path to create. Accepts both file paths (creates
            the parent directory) and directory paths.
    """
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set random seeds for full reproducibility across Python, NumPy, and PyTorch.

    Sets seeds for all relevant random number generators and configures cuDNN
    to use deterministic algorithms. This trades a small amount of throughput
    for reproducible results, which is essential for research reproduction.

    Args:
        seed: The integer seed value. Defaults to 42 (from config.yaml
            experiment.seed).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Disable cuDNN auto-tuning to prevent non-deterministic kernel selection.
    # benchmark=False trades throughput for reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Device management
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return the appropriate compute device for the current environment.

    For distributed training (DDP), the caller is responsible for setting the
    specific device index (e.g., cuda:0, cuda:1) using LOCAL_RANK. This
    function returns the base device type.

    Returns:
        torch.device("cuda") if a CUDA-capable GPU is available,
        otherwise torch.device("cpu").
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model inspection
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters in a model.

    Used to verify model size against paper Table 2:
        - GPT-500m: 468.2M parameters
        - nGPT-500m: 468.4M parameters
        - GPT-1B: 1025.7M parameters
        - nGPT-1B: 1026.1M parameters

    Args:
        model: The PyTorch model to inspect.

    Returns:
        Total number of trainable parameters as an integer.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Learning rate scheduling
# ---------------------------------------------------------------------------

def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    warmup_steps: int,
    max_steps: int,
) -> LambdaLR:
    """Create a cosine annealing LR schedule with optional linear warmup.

    Implements the learning rate schedule used by both GPT and nGPT:
        - GPT: linear warmup for warmup_steps=2000, then cosine decay to 0
        - nGPT: no warmup (warmup_steps=0), pure cosine decay from step 0

    The schedule decays the learning rate to exactly 0 at max_steps, matching
    config.yaml training.gpt.final_lr: 0.0 and training.ngpt.final_lr: 0.0.

    Schedule formula:
        Phase 1 (warmup, only if warmup_steps > 0):
            lr_multiplier = current_step / warmup_steps

        Phase 2 (cosine decay):
            progress = (current_step - warmup_steps) / (max_steps - warmup_steps)
            lr_multiplier = 0.5 * (1 + cos(pi * progress))

    Args:
        optimizer: The optimizer whose learning rate will be scheduled.
        warmup_steps: Number of linear warmup steps. Set to 0 for nGPT
            (config.yaml training.ngpt.warmup_steps: 0). Set to 2000 for GPT
            (config.yaml training.gpt.warmup_steps: 2000).
        max_steps: Total number of training steps. The LR reaches 0 at this
            step (config.yaml training.max_steps: 200000).

    Returns:
        A LambdaLR scheduler that applies the computed multiplier to the
        optimizer's base learning rate at each step.
    """
    def lr_lambda(current_step: int) -> float:
        # Phase 1: Linear warmup
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        # Phase 2: Cosine annealing to 0
        # Clamp progress to [0, 1] to handle steps beyond max_steps gracefully
        progress = float(current_step - warmup_steps) / float(
            max(1, max_steps - warmup_steps)
        )
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, path: str) -> None:
    """Save training state to a checkpoint file.

    Creates parent directories as needed. The state dict is expected to
    contain: model_state_dict, optimizer_state_dict, scheduler_state_dict,
    step, val_loss, and config (as a plain dict). The caller (trainer.py)
    constructs this dict.

    Args:
        state: Dictionary containing all training state to persist.
        path: Full file path for the checkpoint (e.g.,
            "outputs/checkpoints/ngpt_500m_step_50000.pt").
    """
    _ensure_dir(str(pathlib.Path(path).parent))
    torch.save(state, path)


def load_checkpoint(path: str) -> dict:
    """Load training state from a checkpoint file.

    Always loads to CPU first to avoid device mismatch errors when a
    checkpoint saved on 8 GPUs is loaded on a different configuration.
    The caller (trainer.py) is responsible for moving tensors to the
    appropriate device and calling model.load_state_dict(), etc.

    Args:
        path: Full file path to the checkpoint file.

    Returns:
        The state dictionary as saved by save_checkpoint().

    Raises:
        FileNotFoundError: If the checkpoint file does not exist at path.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{path}'. "
            "Verify the path or train a model first."
        )
    # map_location="cpu" ensures compatibility across different GPU configurations
    return torch.load(path, map_location="cpu")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(name: str, log_dir: str = "outputs/logs") -> logging.Logger:
    """Create and configure a logger with both console and file output.

    Avoids adding duplicate handlers if called multiple times with the same
    name (e.g., during unit tests or when modules are reloaded).

    Args:
        name: Logger name, used as the identifier in log messages and as the
            stem of the log file name (e.g., "trainer" → "outputs/logs/trainer.log").
        log_dir: Directory for log files. Created if it does not exist.
            Defaults to "outputs/logs" (config.yaml experiment.log_dir).

    Returns:
        A configured logging.Logger instance.
    """
    _ensure_dir(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — writes to stdout
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler — writes to log_dir/name.log
    log_file = os.path.join(log_dir, f"{name}.log")
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------------
# Metric tracking
# ---------------------------------------------------------------------------

class AverageMeter:
    """Track running averages of scalar metrics during training.

    Maintains a cumulative sum and count to compute a running average without
    storing all individual values. Supports weighted updates for cases where
    the input value is already an average over n samples.

    Attributes:
        name: Metric name for display in log messages.
        val: Most recently updated value.
        avg: Running average (sum / count).
        sum: Cumulative weighted sum.
        count: Total number of samples accumulated.

    Example:
        meter = AverageMeter("train_loss")
        for batch in dataloader:
            loss = compute_loss(batch)
            meter.update(loss.item())
        print(f"Average loss: {meter.avg:.4f}")
    """

    def __init__(self, name: str = "metric") -> None:
        """Initialize the meter with zero accumulators.

        Args:
            name: Human-readable name for this metric, used in __str__.
        """
        self.name: str = name
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def reset(self) -> None:
        """Reset all accumulators to zero.

        Call this at the start of each epoch or evaluation phase to clear
        accumulated statistics from the previous phase.
        """
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """Add a new observation to the running average.

        Args:
            val: The value to add. If this is already an average over n
                samples (e.g., a batch mean loss), pass n=batch_size so the
                weighted average is computed correctly.
            n: Number of samples that val represents. Defaults to 1 for
                per-step scalar values.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __str__(self) -> str:
        """Return a concise string representation for logging.

        Returns:
            String in the format "name: avg_value" showing the running average.
        """
        return f"{self.name}: {self.avg:.6f}"

    def __repr__(self) -> str:
        return (
            f"AverageMeter(name={self.name!r}, avg={self.avg:.6f}, "
            f"count={self.count})"
        )


# ---------------------------------------------------------------------------
# Weight normalization
# ---------------------------------------------------------------------------

def normalize_matrix(W: torch.Tensor, dim: int) -> torch.Tensor:
    """Normalize a weight matrix along a specified dimension to unit L2 norm.

    This is the core normalization primitive for nGPT. Each slice along `dim`
    is scaled to have unit L2 norm. Does NOT modify the input tensor in-place;
    returns a new tensor. The caller is responsible for in-place assignment
    (e.g., param.data.copy_(normalize_matrix(param.data, dim=1))).

    Normalization convention (from paper Section 2.3, 2.4, Shared Knowledge):
        - Embedding matrices (V, d_model): normalize along dim=1
          (each token embedding row becomes unit norm)
        - Weight matrices (out_features, in_features) where in_features=d_model
          (e.g., Wq, Wk, Wv, Wu, Wv_mlp): normalize along dim=1
          (each row is a d_model-dim vector that left-multiplies h)
        - Output projection matrices where out_features=d_model
          (e.g., Wo, WoMLP): normalize along dim=0
          (each column is a d_model-dim output vector)

    Args:
        W: The weight tensor to normalize. Shape is typically
            (out_features, in_features) for linear layers or
            (num_embeddings, embedding_dim) for embeddings.
        dim: The dimension along which to normalize. Each slice along this
            dimension will have unit L2 norm after normalization.

    Returns:
        A new tensor of the same shape as W, with each slice along `dim`
        having unit L2 norm. Uses F.normalize which handles zero-norm vectors
        gracefully (returns zero vectors rather than NaN).
    """
    return F.normalize(W, p=2, dim=dim)


# ---------------------------------------------------------------------------
# Analysis utilities
# ---------------------------------------------------------------------------

def compute_condition_number(matrix: torch.Tensor) -> float:
    """Compute the condition number (σ_max / σ_min) of a weight matrix.

    Used to reproduce Figure 5 (condition numbers of attention and MLP
    matrices at different layer depths) and Figure 11 (condition numbers
    with and without post-training normalization).

    The condition number measures how ill-conditioned a matrix is:
        - Condition number ≈ 1: well-conditioned (nGPT target)
        - Large condition number: ill-conditioned (observed in GPT attention)

    Args:
        matrix: The weight matrix to analyze. Can be any 2D tensor. Will be
            detached from the computation graph and cast to float32 for
            numerical stability in SVD.

    Returns:
        The condition number as a Python float. Returns a large value
        (1e10) if the matrix has zero minimum singular value (rank-deficient).
    """
    # Detach and cast to float32 — bfloat16 has insufficient precision for SVD
    W = matrix.detach().float().cpu()

    # Compute singular values only (more efficient than full SVD)
    # torch.linalg.svdvals returns values in descending order
    S = torch.linalg.svdvals(W)

    sigma_max = S[0].item()
    sigma_min = S[-1].item()

    # Guard against division by zero for rank-deficient matrices
    eps = 1e-10
    if sigma_min < eps:
        return sigma_max / eps

    return sigma_max / sigma_min


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_training_curves(
    gpt_losses: List[float],
    ngpt_losses: List[float],
    save_path: str,
    token_counts: Optional[List[float]] = None,
    log_scale: bool = False,
    title: str = "Validation Loss During Training",
) -> None:
    """Plot validation loss curves for GPT and nGPT side by side.

    Reproduces Figure 1 (validation loss vs. training iterations) and
    Figure 2 (final validation loss vs. token budget) style plots from
    the paper.

    Args:
        gpt_losses: List of validation loss values for the GPT baseline,
            one per evaluation checkpoint.
        ngpt_losses: List of validation loss values for nGPT, one per
            evaluation checkpoint. Must have the same length as gpt_losses.
        save_path: Full file path to save the plot (e.g.,
            "outputs/figures/training_curves.png"). Parent directory is
            created if it does not exist.
        token_counts: Optional list of token counts (in billions) corresponding
            to each loss value. If provided, used as the x-axis (reproducing
            Figure 2 style). If None, step indices are used (Figure 1 style).
        log_scale: If True, use logarithmic scale for the x-axis. Useful for
            Figure 2 style plots where token budgets span orders of magnitude.
        title: Plot title string.
    """
    _ensure_dir(str(pathlib.Path(save_path).parent))

    fig, ax = plt.subplots(figsize=(10, 6))

    # Determine x-axis values
    if token_counts is not None:
        x_gpt = token_counts[: len(gpt_losses)]
        x_ngpt = token_counts[: len(ngpt_losses)]
        x_label = "Training Tokens (Billions)"
    else:
        x_gpt = list(range(len(gpt_losses)))
        x_ngpt = list(range(len(ngpt_losses)))
        x_label = "Training Steps"

    # Plot curves
    ax.plot(x_gpt, gpt_losses, linestyle="-", linewidth=2, label="GPT", color="#1f77b4")
    ax.plot(
        x_ngpt,
        ngpt_losses,
        linestyle="--",
        linewidth=2,
        label="nGPT",
        color="#ff7f0e",
    )

    # Axis formatting
    if log_scale and token_counts is not None:
        ax.set_xscale("log")

    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel("Validation Loss", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_parameter_distributions(
    data: Dict[str, Union[List[float], np.ndarray]],
    save_path: str,
    title: str = "Learned Parameter Distributions",
) -> None:
    """Plot histograms of learned nGPT parameter distributions.

    Reproduces Figure 6 (eigen learning rates αA, αM; MLP scaling factors
    su, sv; QK scaling sqk; logit scaling sz) and Figure 15 (distributions
    under different conditions) from the paper.

    Args:
        data: Dictionary mapping parameter names to arrays of values.
            Expected keys (any subset): "alpha_a", "alpha_m", "sqk",
            "su", "sv", "sz". Values are flat arrays of all parameter
            values across all layers.
            Example: {"alpha_a": [0.21, 0.23, ...], "alpha_m": [0.31, ...]}
        save_path: Full file path to save the plot. Parent directory is
            created if it does not exist.
        title: Overall figure title.
    """
    _ensure_dir(str(pathlib.Path(save_path).parent))

    n_params = len(data)
    if n_params == 0:
        return

    # Arrange subplots in a grid: up to 3 columns
    n_cols = min(3, n_params)
    n_rows = math.ceil(n_params / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    # Flatten axes array for uniform indexing
    if n_params == 1:
        axes_flat = [axes]
    elif n_rows == 1:
        axes_flat = list(axes)
    else:
        axes_flat = [ax for row in axes for ax in row]

    # Parameter display names for cleaner labels
    display_names: Dict[str, str] = {
        "alpha_a": r"$\alpha_A$ (Attention Eigen LR)",
        "alpha_m": r"$\alpha_M$ (MLP Eigen LR)",
        "sqk": r"$s_{qk}$ (QK Scaling)",
        "su": r"$s_u$ (MLP u-gate Scaling)",
        "sv": r"$s_v$ (MLP v-gate Scaling)",
        "sz": r"$s_z$ (Logit Scaling)",
    }

    for idx, (param_name, values) in enumerate(data.items()):
        ax = axes_flat[idx]
        values_array = np.asarray(values, dtype=np.float32).flatten()

        # Plot histogram
        sns.histplot(values_array, bins=50, ax=ax, kde=True, color="#2196F3")

        # Annotate with mean value
        mean_val = float(np.mean(values_array))
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.4f}")
        ax.legend(fontsize=9)

        # Labels
        display_name = display_names.get(param_name, param_name)
        ax.set_title(display_name, fontsize=11)
        ax.set_xlabel("Value", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_params, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
