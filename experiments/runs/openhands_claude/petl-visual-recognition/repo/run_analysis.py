"""
Prediction diversity analysis (Section 4, Figures 3, 12, 13).

Analyzes prediction similarity and diversity across PEFT methods:
  - Pairwise prediction similarity matrices (Figure 3a, 13)
  - Correct prediction overlap for top-5K confident samples (Figure 1b, 12)
  - Wrong prediction overlap for bottom-5K confident samples (Figure 3b)
  - Within-group vs. cross-group prediction similarity
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import VTAB_ALL_TASKS, VTAB_NUM_CLASSES, PEFT_DEFAULT_CONFIGS
from data import get_vtab_dataloaders
from evaluate import (
    get_predictions,
    compute_prediction_similarity_matrix,
    compute_correct_prediction_overlap,
    compute_wrong_prediction_overlap,
    compute_venn_overlaps,
)
from models.vit import build_peft_model
from utils import set_seed


def load_models_for_task(
    task_name: str,
    methods: List[str],
    checkpoint_dir: str,
    device: torch.device,
) -> Dict[str, nn.Module]:
    """Load fine-tuned models for a given task."""
    num_classes = VTAB_NUM_CLASSES[task_name]
    models = {}

    for method in methods:
        ckpt_path = Path(checkpoint_dir) / f"{method}_{task_name}_best.pth"
        if not ckpt_path.exists():
            print(f"  Checkpoint not found: {ckpt_path}")
            continue

        model = build_peft_model(
            peft_method=method,
            num_classes=num_classes,
            peft_config=PEFT_DEFAULT_CONFIGS.get(method, {}),
            pretrained=False,
        )
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        models[method] = model

    return models


def run_prediction_analysis(
    task_name: str,
    methods: List[str],
    checkpoint_dir: str,
    data_dir: str,
    device: torch.device,
    k_confident: int = 5000,
    output_dir: str = "./outputs/analysis",
) -> Dict:
    """
    Run full prediction diversity analysis for a single task.
    """
    print(f"\nAnalyzing predictions for task: {task_name}")

    # Load models
    models = load_models_for_task(task_name, methods, checkpoint_dir, device)
    if not models:
        print(f"  No models found for {task_name}")
        return {}

    # Get test dataloader
    loaders = get_vtab_dataloaders(
        task_name=task_name,
        batch_size=64,
        data_dir=data_dir,
    )
    test_loader = loaders["test"]

    # Get predictions for all methods
    all_logits = {}
    all_preds = {}
    all_labels = None

    for method, model in models.items():
        logits, preds, labels = get_predictions(model, test_loader, device)
        all_logits[method] = logits
        all_preds[method] = preds
        if all_labels is None:
            all_labels = labels

    # Prediction similarity matrix
    sim_matrix, method_names = compute_prediction_similarity_matrix(all_preds)

    # Accuracy per method
    accuracies = {
        m: (all_preds[m] == all_labels).float().mean().item() * 100.0
        for m in all_preds
    }

    # Confidence-based overlap
    correct_sets = compute_correct_prediction_overlap(all_logits, all_labels, k_confident)
    wrong_sets = compute_wrong_prediction_overlap(all_logits, all_labels, k_confident)

    # Venn diagram overlaps for representative methods (LoRA, Adapter, SSF)
    venn_methods = [m for m in ["lora", "houl_adapter", "ssf"] if m in all_logits]
    if len(venn_methods) >= 2:
        venn_correct = compute_venn_overlaps(
            {m: correct_sets[m] for m in venn_methods if m in correct_sets}
        )
        venn_wrong = compute_venn_overlaps(
            {m: wrong_sets[m] for m in venn_methods if m in wrong_sets}
        )
    else:
        venn_correct, venn_wrong = {}, {}

    results = {
        "task": task_name,
        "accuracies": accuracies,
        "similarity_matrix": sim_matrix.tolist(),
        "methods": method_names,
        "venn_correct_overlap": venn_correct,
        "venn_wrong_overlap": venn_wrong,
        "avg_accuracy": sum(accuracies.values()) / len(accuracies) if accuracies else 0.0,
    }

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir) / f"{task_name}_prediction_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Avg accuracy: {results['avg_accuracy']:.2f}%")
    print(f"  Similarity matrix saved to {out_path}")

    return results


def run_all_prediction_analysis(
    tasks: List[str],
    methods: List[str],
    checkpoint_dir: str,
    data_dir: str,
    device: torch.device,
    output_dir: str = "./outputs/analysis",
) -> None:
    """Run prediction analysis for all VTAB tasks (Figure 13)."""
    all_results = {}

    for task in tasks:
        results = run_prediction_analysis(
            task_name=task,
            methods=methods,
            checkpoint_dir=checkpoint_dir,
            data_dir=data_dir,
            device=device,
            output_dir=output_dir,
        )
        if results:
            all_results[task] = results

    # Summary: average pairwise similarity across tasks
    print("\n" + "=" * 60)
    print("Prediction Diversity Summary")
    print("=" * 60)
    for task, res in all_results.items():
        sim_matrix = torch.tensor(res["similarity_matrix"])
        # Off-diagonal mean similarity
        n = sim_matrix.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool)
        avg_sim = sim_matrix[mask].mean().item()
        print(f"  {task:<30}: avg pairwise similarity = {avg_sim:.1f}%")

    summary_path = Path(output_dir) / "prediction_analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run prediction diversity analysis")
    parser.add_argument("--tasks", nargs="+", default=["dtd", "retinopathy", "dmlab"],
                        help="VTAB tasks to analyze (default: DTD, Retinopathy, DMLab)")
    parser.add_argument("--methods", nargs="+",
                        default=["lora", "houl_adapter", "adaptformer", "bitfit",
                                 "vpt_deep", "ssf", "convpass", "fact_tt",
                                 "difffit", "layernorm", "repadapter", "fact_tk",
                                 "pfeif_adapter", "vpt_shallow"],
                        help="PEFT methods to compare")
    parser.add_argument("--checkpoint_dir", type=str, default="./outputs/vtab",
                        help="Directory containing model checkpoints")
    parser.add_argument("--data_dir", type=str, default="./data/vtab")
    parser.add_argument("--output_dir", type=str, default="./outputs/analysis")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--all_tasks", action="store_true",
                        help="Run analysis on all 19 VTAB tasks")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tasks = VTAB_ALL_TASKS if args.all_tasks else args.tasks

    run_all_prediction_analysis(
        tasks=tasks,
        methods=args.methods,
        checkpoint_dir=args.checkpoint_dir,
        data_dir=args.data_dir,
        device=device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
