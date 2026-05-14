"""
Run many-shot experiments (Section 5, Figure 5).

Evaluates PEFT methods on full-size datasets:
  - CIFAR-100 (natural, domain-close to ImageNet)
  - RESISC45 (specialized, remote sensing)
  - Clevr-Distance (structured, synthetic)

Varies the number of trainable parameters to show the accuracy vs. parameter
efficiency tradeoff.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch

from config import ManyShotConfig, MANYSHOT_DATASETS, PEFT_DEFAULT_CONFIGS
from train import run_manyshot_experiment
from utils import set_seed


# Parameter size variants for Figure 5
# Each entry is a (method, config) pair with different bottleneck sizes
PARAM_SIZE_VARIANTS = {
    "lora": [
        {"rank": 1},    # ~0.036M
        {"rank": 8},    # ~0.295M
        {"rank": 16},   # ~0.590M
        {"rank": 32},   # ~1.179M
    ],
    "houl_adapter": [
        {"bottleneck_dim": 4, "scale_factor": 0.1},   # ~0.165M
        {"bottleneck_dim": 8, "scale_factor": 0.1},   # ~0.330M
        {"bottleneck_dim": 16, "scale_factor": 0.1},  # ~0.599M
        {"bottleneck_dim": 32, "scale_factor": 0.1},  # ~1.198M
    ],
    "adaptformer": [
        {"bottleneck_dim": 4, "scale_factor": 0.1},
        {"bottleneck_dim": 16, "scale_factor": 0.1},
        {"bottleneck_dim": 32, "scale_factor": 0.1},
    ],
    "vpt_deep": [
        {"num_prompts": 5},
        {"num_prompts": 10},
        {"num_prompts": 50},
        {"num_prompts": 100},
    ],
    "fact_tt": [
        {"rank": 8, "scale_factor": 1.0},
        {"rank": 16, "scale_factor": 1.0},
        {"rank": 32, "scale_factor": 1.0},
    ],
}


def run_manyshot_param_sweep(
    method: str,
    dataset: str,
    config: ManyShotConfig,
    device: torch.device,
    data_root: str,
    output_dir: str,
) -> List[Dict]:
    """
    Run many-shot experiment with varying parameter sizes.
    Reproduces Figure 5.
    """
    variants = PARAM_SIZE_VARIANTS.get(method, [PEFT_DEFAULT_CONFIGS.get(method, {})])
    results = []

    for peft_config in variants:
        print(f"\nMethod: {method}, Config: {peft_config}, Dataset: {dataset}")
        try:
            result = run_manyshot_experiment(
                peft_method=method,
                dataset_name=dataset,
                config=config,
                device=device,
                data_root=data_root,
                output_dir=output_dir,
            )
            result["peft_config"] = peft_config
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run many-shot PEFT experiments")
    parser.add_argument("--methods", nargs="+",
                        default=["lora", "houl_adapter", "adaptformer", "bitfit",
                                 "vpt_deep", "ssf", "fact_tt", "linear", "full"],
                        help="PEFT methods to evaluate")
    parser.add_argument("--datasets", nargs="+", default=MANYSHOT_DATASETS)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs/manyshot")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--param_sweep", action="store_true",
                        help="Sweep over parameter sizes (Figure 5)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config = ManyShotConfig()

    all_results = {}

    for dataset in args.datasets:
        all_results[dataset] = {}
        data_root = os.path.join(args.data_dir, dataset)

        for method in args.methods:
            if args.param_sweep:
                results = run_manyshot_param_sweep(
                    method=method,
                    dataset=dataset,
                    config=config,
                    device=device,
                    data_root=data_root,
                    output_dir=args.output_dir,
                )
                all_results[dataset][method] = results
            else:
                try:
                    result = run_manyshot_experiment(
                        peft_method=method,
                        dataset_name=dataset,
                        config=config,
                        device=device,
                        data_root=data_root,
                        output_dir=args.output_dir,
                    )
                    all_results[dataset][method] = result["test_acc"]
                    print(f"{method}/{dataset}: {result['test_acc']:.2f}%")
                except Exception as e:
                    print(f"ERROR: {method}/{dataset}: {e}")
                    all_results[dataset][method] = 0.0

    summary_path = Path(args.output_dir) / "manyshot_summary.json"
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nMany-shot Results:")
    for dataset in args.datasets:
        print(f"\n{dataset}:")
        for method, acc in all_results[dataset].items():
            if isinstance(acc, float):
                print(f"  {method:<20}: {acc:.2f}%")


if __name__ == "__main__":
    main()
