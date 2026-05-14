"""
Ensemble methods for PEFT models.

Implements majority vote ensemble over multiple PEFT methods (Figure 4).
Also supports logit averaging (soft ensemble).

The paper shows that despite similar accuracy, different PEFT methods make
diverse predictions, enabling consistent gains from ensemble methods.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate import get_predictions, compute_top1_accuracy


def majority_vote_ensemble(
    predictions_list: List[torch.Tensor],
    num_classes: int,
) -> torch.Tensor:
    """
    Majority vote over a list of prediction tensors.

    Args:
        predictions_list: list of [N] tensors with class predictions
        num_classes: number of classes

    Returns:
        ensemble_preds: [N] tensor with majority vote predictions
    """
    N = predictions_list[0].shape[0]
    vote_counts = torch.zeros(N, num_classes, dtype=torch.long)

    for preds in predictions_list:
        for i in range(N):
            vote_counts[i, preds[i]] += 1

    return vote_counts.argmax(dim=-1)


def soft_ensemble(
    logits_list: List[torch.Tensor],
) -> torch.Tensor:
    """
    Soft ensemble: average logits (or probabilities) across methods.

    Args:
        logits_list: list of [N, C] logit tensors

    Returns:
        ensemble_preds: [N] tensor with argmax of averaged logits
    """
    avg_logits = torch.stack(logits_list, dim=0).mean(dim=0)
    return avg_logits.argmax(dim=-1)


def evaluate_ensemble(
    models_dict: Dict[str, nn.Module],
    loader: DataLoader,
    device: torch.device,
    ensemble_type: str = "majority_vote",
    num_classes: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate ensemble of PEFT models on a dataset.

    Args:
        models_dict: {method_name: model}
        loader: DataLoader
        device: torch device
        ensemble_type: "majority_vote" or "soft"
        num_classes: required for majority vote

    Returns:
        dict with per-method accuracy and ensemble accuracy
    """
    all_logits = {}
    all_preds = {}
    all_labels = None

    for method, model in models_dict.items():
        logits, preds, labels = get_predictions(model, loader, device)
        all_logits[method] = logits
        all_preds[method] = preds
        if all_labels is None:
            all_labels = labels

    # Per-method accuracy
    per_method_acc = {
        method: compute_top1_accuracy(preds, all_labels)
        for method, preds in all_preds.items()
    }

    # Ensemble accuracy
    if ensemble_type == "majority_vote":
        if num_classes is None:
            num_classes = all_logits[list(all_logits.keys())[0]].shape[-1]
        ensemble_preds = majority_vote_ensemble(
            list(all_preds.values()), num_classes
        )
    else:
        ensemble_preds = soft_ensemble(list(all_logits.values()))

    ensemble_acc = compute_top1_accuracy(ensemble_preds, all_labels)

    # Worst method accuracy (baseline for Figure 4)
    worst_acc = min(per_method_acc.values())
    best_acc = max(per_method_acc.values())

    return {
        "per_method": per_method_acc,
        "ensemble": ensemble_acc,
        "worst_method": worst_acc,
        "best_method": best_acc,
        "ensemble_gain_over_worst": ensemble_acc - worst_acc,
        "ensemble_gain_over_best": ensemble_acc - best_acc,
    }


def compute_ensemble_gains_vtab(
    results_per_task: Dict[str, Dict],
) -> Dict[str, float]:
    """
    Compute ensemble gain over worst method for each VTAB task (Figure 4).

    Args:
        results_per_task: {task: evaluate_ensemble output}

    Returns:
        {task: gain_over_worst}
    """
    gains = {}
    for task, result in results_per_task.items():
        gains[task] = result.get("ensemble_gain_over_worst", 0.0)
    return gains


def run_ensemble_analysis(
    models_dict: Dict[str, nn.Module],
    task_loaders: Dict[str, DataLoader],
    device: torch.device,
    ensemble_type: str = "majority_vote",
) -> Dict[str, Dict]:
    """
    Run ensemble analysis across all VTAB tasks.

    Returns per-task ensemble results including gain over worst method.
    """
    all_results = {}

    for task, loader in task_loaders.items():
        print(f"Ensemble analysis for task: {task}")
        result = evaluate_ensemble(
            models_dict=models_dict,
            loader=loader,
            device=device,
            ensemble_type=ensemble_type,
        )
        all_results[task] = result
        print(f"  Ensemble acc: {result['ensemble']:.2f}% "
              f"(+{result['ensemble_gain_over_worst']:.2f}% over worst)")

    return all_results
