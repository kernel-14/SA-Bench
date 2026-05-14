"""
Shared utilities: logging, checkpointing, learning rate scheduling, and
noise schedule helpers.
"""

import os
import math
import logging
import random
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Optimizer and LR schedule (Section C.1)
# ---------------------------------------------------------------------------

def build_optimizer(
    model: nn.Module,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.95,
    weight_decay: float = 0.1,
) -> AdamW:
    """
    AdamW optimizer with weight decay (Loshchilov & Hutter, 2017).
    Excludes bias and LayerNorm parameters from weight decay.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "ln" in name or "norm" in name or "pos_emb" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return AdamW(param_groups, lr=lr, betas=(beta1, beta2))


def build_cosine_lr_scheduler(
    optimizer: AdamW,
    warmup_steps: int,
    total_steps: int,
    lr_min_ratio: float = 0.1,
) -> LambdaLR:
    """
    Cosine learning rate schedule with linear warmup (Section C.1).

    lr(t) = lr_max * [warmup_factor * cosine_decay]
    where lr_min = lr_max * lr_min_ratio
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_min_ratio + (1.0 - lr_min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    step: int,
    loss: float,
    output_dir: str,
    filename: str = "checkpoint.pt",
):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    torch.save({
        "step": step,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }, path)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[AdamW],
    scheduler: Optional[LambdaLR],
    path: str,
    device: str = "cuda",
) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return {"step": checkpoint["step"], "loss": checkpoint["loss"]}


# ---------------------------------------------------------------------------
# IsoFLOP analysis utilities (Section C.1)
# ---------------------------------------------------------------------------

def compute_flops_per_token(
    n_params: int,
    seq_len: int,
) -> float:
    """
    Approximate FLOPs per token for a transformer.
    Following Hoffmann et al. (2022): FLOPs ≈ 6 * N * D
    where N = non-embedding parameters, D = number of tokens.
    """
    return 6.0 * n_params


def compute_training_tokens(
    total_flops: float,
    n_params: int,
) -> int:
    """
    Compute number of training tokens for a given FLOP budget.
    tokens = total_flops / (6 * n_params)
    """
    return int(total_flops / (6.0 * n_params))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """
    Compute token-level accuracy.

    Args:
        predictions: (B, L) predicted token ids
        targets:     (B, L) ground truth token ids
        mask:        (B, L) boolean mask — if provided, only evaluate masked positions

    Returns:
        accuracy: float
    """
    if mask is not None:
        correct = (predictions[mask] == targets[mask]).float()
        return correct.mean().item()
    return (predictions == targets).float().mean().item()


def compute_puzzle_accuracy(
    predictions: torch.Tensor,
    solutions: torch.Tensor,
    puzzles: Optional[torch.Tensor] = None,
) -> float:
    """
    Compute puzzle-level accuracy (fraction of fully correct solutions).

    Args:
        predictions: (B, L) predicted sequences
        solutions:   (B, L) ground truth solutions
        puzzles:     (B, L) original puzzles (if provided, only check empty positions)

    Returns:
        accuracy: float (fraction of puzzles solved correctly)
    """
    B = predictions.shape[0]

    if puzzles is not None:
        empty_mask = (puzzles == 0)
        correct_per_puzzle = (predictions[empty_mask] == solutions[empty_mask])
        # Reshape to check per-puzzle
        correct_puzzles = 0
        for b in range(B):
            empty_b = empty_mask[b]
            if empty_b.any():
                if (predictions[b][empty_b] == solutions[b][empty_b]).all():
                    correct_puzzles += 1
            else:
                correct_puzzles += 1
        return correct_puzzles / B
    else:
        correct = (predictions == solutions).all(dim=1).float()
        return correct.mean().item()


def compute_entropy(sequences: torch.Tensor, vocab_size: int) -> float:
    """
    Compute average token entropy of generated sequences (Section D.1.2).

    entropy = Σ_i p_i * log(p_i) where p_i = #{x^j = i} / L
    """
    B, L = sequences.shape
    entropies = []
    for b in range(B):
        counts = torch.bincount(sequences[b], minlength=vocab_size).float()
        probs = counts / L
        probs = probs[probs > 0]
        entropy = -(probs * torch.log(probs)).sum().item()
        entropies.append(entropy)
    return float(np.mean(entropies))


# ---------------------------------------------------------------------------
# Sudoku validation
# ---------------------------------------------------------------------------

def is_valid_sudoku(grid: torch.Tensor) -> bool:
    """
    Check if a 9×9 Sudoku solution is valid.

    Args:
        grid: (81,) tensor of token ids (1-9)

    Returns:
        True if valid
    """
    grid = grid.view(9, 9)
    digits = set(range(1, 10))

    # Check rows
    for r in range(9):
        if set(grid[r].tolist()) != digits:
            return False

    # Check columns
    for c in range(9):
        if set(grid[:, c].tolist()) != digits:
            return False

    # Check 3×3 boxes
    for br in range(3):
        for bc in range(3):
            box = grid[br*3:(br+1)*3, bc*3:(bc+1)*3]
            if set(box.flatten().tolist()) != digits:
                return False

    return True


def compute_sudoku_accuracy(
    predictions: torch.Tensor,
    solutions: torch.Tensor,
    puzzles: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute Sudoku accuracy metrics.

    Returns:
        dict with 'puzzle_accuracy' (fraction of fully correct solutions)
        and 'valid_accuracy' (fraction of valid Sudoku grids)
    """
    B = predictions.shape[0]
    correct = 0
    valid = 0

    for b in range(B):
        pred = predictions[b]
        sol = solutions[b]

        if puzzles is not None:
            empty = (puzzles[b] == 0)
            if (pred[empty] == sol[empty]).all():
                correct += 1
        else:
            if (pred == sol).all():
                correct += 1

        if is_valid_sudoku(pred):
            valid += 1

    return {
        "puzzle_accuracy": correct / B,
        "valid_accuracy": valid / B,
    }


# ---------------------------------------------------------------------------
# WandB logging helper
# ---------------------------------------------------------------------------

def log_metrics(
    metrics: Dict[str, float],
    step: int,
    use_wandb: bool = False,
    prefix: str = "",
):
    if prefix:
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

    log_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
    logger = get_logger("metrics")
    logger.info(f"Step {step} | {log_str}")

    if use_wandb:
        try:
            import wandb
            wandb.log(metrics, step=step)
        except ImportError:
            pass
