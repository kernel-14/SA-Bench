"""
Evaluation utilities for PEFT experiments.

Includes:
  - Top-1 accuracy evaluation
  - Prediction similarity analysis (Figure 3a)
  - Confidence-based prediction overlap analysis (Figures 1b, 3b)
  - Per-task and per-group accuracy aggregation for VTAB-1K
  - Ranking frequency analysis (Figure 2)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    VTAB_ALL_TASKS,
    VTAB_NATURAL,
    VTAB_SPECIALIZED,
    VTAB_STRUCTURED,
    VTAB_NUM_CLASSES,
)
from utils import AverageMeter, accuracy


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run inference and return (logits, predictions, labels).
    Returns tensors on CPU.
    """
    model.eval()
    all_logits, all_preds, all_labels = [], [], []

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=-1)

        all_logits.append(logits.cpu())
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    return (
        torch.cat(all_logits, dim=0),
        torch.cat(all_preds, dim=0),
        torch.cat(all_labels, dim=0),
    )


def compute_top1_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute top-1 accuracy as percentage."""
    return (preds == labels).float().mean().item() * 100.0


# ---------------------------------------------------------------------------
# Prediction similarity analysis (Section 4, Figure 3a)
# ---------------------------------------------------------------------------

def compute_prediction_similarity_matrix(
    predictions_dict: Dict[str, torch.Tensor],
) -> np.ndarray:
    """
    Compute pairwise prediction similarity matrix.
    Entry (i, j) = percentage of samples where method i and j predict the same class.

    Args:
        predictions_dict: {method_name: predictions tensor [N]}

    Returns:
        similarity_matrix: [num_methods, num_methods] numpy array
    """
    methods = list(predictions_dict.keys())
    n = len(methods)
    matrix = np.zeros((n, n))

    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            preds_i = predictions_dict[m1]
            preds_j = predictions_dict[m2]
            agreement = (preds_i == preds_j).float().mean().item() * 100.0
            matrix[i, j] = agreement

    return matrix, methods


# ---------------------------------------------------------------------------
# Confidence-based overlap analysis (Figures 1b, 3b)
# ---------------------------------------------------------------------------

def get_top_k_confident_indices(
    logits: torch.Tensor,
    k: int = 5000,
) -> torch.Tensor:
    """Return indices of the k most confident predictions (highest max softmax prob)."""
    probs = F.softmax(logits, dim=-1)
    confidence = probs.max(dim=-1).values
    _, indices = confidence.topk(k)
    return indices


def get_bottom_k_confident_indices(
    logits: torch.Tensor,
    k: int = 5000,
) -> torch.Tensor:
    """Return indices of the k least confident predictions."""
    probs = F.softmax(logits, dim=-1)
    confidence = probs.max(dim=-1).values
    _, indices = confidence.topk(k, largest=False)
    return indices


def compute_correct_prediction_overlap(
    logits_dict: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    k: int = 5000,
) -> Dict[str, set]:
    """
    For each method, find the set of sample indices that are:
      - Among the k most confident predictions
      - Correctly predicted

    Returns dict mapping method name to set of correct high-confidence indices.
    """
    correct_sets = {}
    for method, logits in logits_dict.items():
        top_k_idx = get_top_k_confident_indices(logits, k)
        preds = logits.argmax(dim=-1)
        correct_mask = (preds == labels)
        correct_top_k = set(
            top_k_idx[correct_mask[top_k_idx]].tolist()
        )
        correct_sets[method] = correct_top_k
    return correct_sets


def compute_wrong_prediction_overlap(
    logits_dict: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    k: int = 5000,
) -> Dict[str, set]:
    """
    For each method, find the set of sample indices that are:
      - Among the k least confident predictions
      - Incorrectly predicted
    """
    wrong_sets = {}
    for method, logits in logits_dict.items():
        bottom_k_idx = get_bottom_k_confident_indices(logits, k)
        preds = logits.argmax(dim=-1)
        wrong_mask = (preds != labels)
        wrong_bottom_k = set(
            bottom_k_idx[wrong_mask[bottom_k_idx]].tolist()
        )
        wrong_sets[method] = wrong_bottom_k
    return wrong_sets


def compute_venn_overlaps(sets_dict: Dict[str, set]) -> Dict[str, int]:
    """
    Compute pairwise and triple overlaps for Venn diagram visualization.
    Works for 2 or 3 methods.
    """
    methods = list(sets_dict.keys())
    overlaps = {}

    if len(methods) == 3:
        m1, m2, m3 = methods
        s1, s2, s3 = sets_dict[m1], sets_dict[m2], sets_dict[m3]
        overlaps[f"{m1}_only"] = len(s1 - s2 - s3)
        overlaps[f"{m2}_only"] = len(s2 - s1 - s3)
        overlaps[f"{m3}_only"] = len(s3 - s1 - s2)
        overlaps[f"{m1}_{m2}"] = len(s1 & s2 - s3)
        overlaps[f"{m1}_{m3}"] = len(s1 & s3 - s2)
        overlaps[f"{m2}_{m3}"] = len(s2 & s3 - s1)
        overlaps[f"all"] = len(s1 & s2 & s3)
    elif len(methods) == 2:
        m1, m2 = methods
        s1, s2 = sets_dict[m1], sets_dict[m2]
        overlaps[f"{m1}_only"] = len(s1 - s2)
        overlaps[f"{m2}_only"] = len(s2 - s1)
        overlaps[f"both"] = len(s1 & s2)

    return overlaps


# ---------------------------------------------------------------------------
# VTAB-1K aggregation and ranking (Figure 2)
# ---------------------------------------------------------------------------

def aggregate_vtab_results(
    results: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate VTAB-1K results by group.

    Args:
        results: {method: {task: accuracy}}

    Returns:
        aggregated: {method: {group: mean_accuracy, "overall": mean_accuracy}}
    """
    aggregated = {}
    for method, task_accs in results.items():
        natural_accs = [task_accs[t] for t in VTAB_NATURAL if t in task_accs]
        specialized_accs = [task_accs[t] for t in VTAB_SPECIALIZED if t in task_accs]
        structured_accs = [task_accs[t] for t in VTAB_STRUCTURED if t in task_accs]
        all_accs = natural_accs + specialized_accs + structured_accs

        aggregated[method] = {
            "natural": np.mean(natural_accs) if natural_accs else 0.0,
            "specialized": np.mean(specialized_accs) if specialized_accs else 0.0,
            "structured": np.mean(structured_accs) if structured_accs else 0.0,
            "overall": np.mean(all_accs) if all_accs else 0.0,
        }
    return aggregated


def compute_ranking_frequency(
    results: Dict[str, Dict[str, float]],
    group: str = "natural",
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute ranking frequency matrix for a given VTAB group (Figure 2).

    Element (i, j) = number of times method i ranks j-th in the group.

    Args:
        results: {method: {task: accuracy}}
        group: "natural", "specialized", or "structured"

    Returns:
        freq_matrix: [num_methods, num_methods] array
        methods: list of method names (sorted by mean rank)
    """
    group_tasks = {
        "natural": VTAB_NATURAL,
        "specialized": VTAB_SPECIALIZED,
        "structured": VTAB_STRUCTURED,
    }[group]

    methods = list(results.keys())
    n = len(methods)
    freq_matrix = np.zeros((n, n), dtype=int)
    mean_ranks = np.zeros(n)

    for task in group_tasks:
        task_accs = [(m, results[m].get(task, 0.0)) for m in methods]
        # Sort by accuracy descending to get ranks
        task_accs_sorted = sorted(task_accs, key=lambda x: x[1], reverse=True)
        for rank, (method, _) in enumerate(task_accs_sorted):
            method_idx = methods.index(method)
            freq_matrix[method_idx, rank] += 1
            mean_ranks[method_idx] += rank + 1  # 1-indexed rank

    mean_ranks /= len(group_tasks)

    # Sort methods by mean rank
    sorted_indices = np.argsort(mean_ranks)
    freq_matrix = freq_matrix[sorted_indices]
    methods_sorted = [methods[i] for i in sorted_indices]
    mean_ranks_sorted = mean_ranks[sorted_indices]

    return freq_matrix, methods_sorted, mean_ranks_sorted


def compute_relative_std_dev(
    results: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """
    Compute relative standard deviation (std / mean) across methods for each task.
    This measures how similar different PEFT methods are on each task.
    """
    all_tasks = VTAB_ALL_TASKS
    rel_std = {}

    for task in all_tasks:
        accs = [results[m].get(task, 0.0) for m in results if task in results[m]]
        if len(accs) > 1:
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            rel_std[task] = (std_acc / mean_acc * 100) if mean_acc > 0 else 0.0
        else:
            rel_std[task] = 0.0

    return rel_std


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate_all_methods_vtab(
    models_dict: Dict[str, nn.Module],
    task_name: str,
    test_loader: DataLoader,
    device: torch.device,
    k_confident: int = 5000,
) -> Dict:
    """
    Evaluate all PEFT methods on a single VTAB task.
    Returns accuracy, prediction similarity, and confidence overlap analysis.
    """
    all_logits = {}
    all_preds = {}
    all_labels = None

    for method, model in models_dict.items():
        logits, preds, labels = get_predictions(model, test_loader, device)
        all_logits[method] = logits
        all_preds[method] = preds
        if all_labels is None:
            all_labels = labels

    # Accuracy per method
    accuracies = {
        method: compute_top1_accuracy(preds, all_labels)
        for method, preds in all_preds.items()
    }

    # Prediction similarity matrix
    sim_matrix, methods = compute_prediction_similarity_matrix(all_preds)

    # Confidence-based overlap
    correct_sets = compute_correct_prediction_overlap(all_logits, all_labels, k_confident)
    wrong_sets = compute_wrong_prediction_overlap(all_logits, all_labels, k_confident)

    return {
        "task": task_name,
        "accuracies": accuracies,
        "similarity_matrix": sim_matrix.tolist(),
        "methods": methods,
        "correct_overlap_sets": {k: list(v) for k, v in correct_sets.items()},
        "wrong_overlap_sets": {k: list(v) for k, v in wrong_sets.items()},
    }


def save_evaluation_results(results: Dict, output_path: str) -> None:
    """Save evaluation results to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def load_evaluation_results(results_dir: str) -> Dict[str, Dict[str, float]]:
    """
    Load all evaluation results from a directory.
    Returns {method: {task: accuracy}}.
    """
    results = {}
    results_dir = Path(results_dir)

    for json_file in results_dir.glob("*_results.json"):
        with open(json_file) as f:
            data = json.load(f)
        method = data.get("method", "unknown")
        task = data.get("task", "unknown")
        acc = data.get("test_acc", 0.0)

        if method not in results:
            results[method] = {}
        results[method][task] = acc

    return results
