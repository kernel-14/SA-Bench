"""
Run robustness experiments with CLIP ViT-B/16 (Section 7, Table 2, Figure 1c).

Fine-tunes CLIP on 100-shot ImageNet and evaluates on:
  - ImageNet-1K (target distribution)
  - ImageNet-V2, ImageNet-R, ImageNet-S, ImageNet-A (distribution shifts)

Also runs WiSE sweep to reproduce Figure 1c and Figure 14.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from config import RobustnessConfig, PEFT_DEFAULT_CONFIGS
from data import get_distribution_shift_dataloaders, get_imagenet_dataloaders
from evaluate import get_predictions, compute_top1_accuracy
from wise import wise_sweep
from utils import set_seed


ROBUSTNESS_METHODS = [
    "full",
    "bitfit",
    "layernorm",
    "houl_adapter",
    "adaptformer",
    "repadapter",
    "convpass",
    "lora",
    "fact_tk",
]


def run_robustness_all_methods(
    methods: List[str],
    config: RobustnessConfig,
    device: torch.device,
    imagenet_root: str,
    shift_data_roots: Dict[str, str],
    output_dir: str,
    run_wise: bool = True,
) -> None:
    """
    Run robustness experiments for all methods.
    Reproduces Table 2 and Figure 1c.
    """
    from train import run_robustness_experiment

    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    # Get distribution shift dataloaders
    shift_loaders = get_distribution_shift_dataloaders(
        data_roots=shift_data_roots,
        batch_size=config.batch_size,
        image_size=config.image_size,
        use_clip_norm=True,
    )

    # Target distribution dataloader (ImageNet val)
    imagenet_loaders = get_imagenet_dataloaders(
        imagenet_root=imagenet_root,
        num_shots=config.num_shots,
        batch_size=config.batch_size,
        image_size=config.image_size,
        use_clip_norm=True,
        use_strong_augmentation=False,
    )
    target_loader = imagenet_loaders["val"]

    for method in methods:
        print(f"\n{'='*60}")
        print(f"Robustness experiment: {method}")
        print(f"{'='*60}")

        try:
            results, finetuned_model = run_robustness_experiment(
                peft_method=method,
                config=config,
                device=device,
                imagenet_root=imagenet_root,
                output_dir=output_dir,
            )

            # Evaluate on distribution shift datasets
            shift_accs = {}
            for ds_name, ds_loader in shift_loaders.items():
                _, preds, labels = get_predictions(finetuned_model, ds_loader, device)
                shift_accs[ds_name] = compute_top1_accuracy(preds, labels)

            avg_shift_acc = sum(shift_accs.values()) / len(shift_accs) if shift_accs else 0.0

            results.update({
                "shift_accs": shift_accs,
                "avg_shift_acc": avg_shift_acc,
            })
            all_results[method] = results

            print(f"ImageNet: {results['imagenet_acc']:.2f}%")
            print(f"Avg shift: {avg_shift_acc:.2f}%")
            for ds, acc in shift_accs.items():
                print(f"  {ds}: {acc:.2f}%")

            # WiSE sweep
            if run_wise:
                print(f"\nRunning WiSE sweep for {method}...")
                # Load pre-trained model for WiSE
                import open_clip
                clip_model, _, _ = open_clip.create_model_and_transforms(
                    "ViT-B-16", pretrained="openai"
                )
                pretrained_model = clip_model.visual.to(device)

                wise_results = wise_sweep(
                    finetuned_model=finetuned_model,
                    pretrained_model=pretrained_model,
                    peft_method=method,
                    target_loader=target_loader,
                    shift_loaders=shift_loaders,
                    device=device,
                    alphas=config.wise_alphas,
                )
                results["wise_sweep"] = wise_results

                wise_path = Path(output_dir) / f"{method}_wise_sweep.json"
                with open(wise_path, "w") as f:
                    json.dump(wise_results, f, indent=2)

        except Exception as e:
            print(f"ERROR for {method}: {e}")
            all_results[method] = {"error": str(e)}

    # Save summary
    summary_path = Path(output_dir) / "robustness_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print Table 2
    print("\n" + "=" * 80)
    print("Table 2: Robustness Results")
    print("=" * 80)
    print(f"{'Method':<20} {'ImageNet':>10} {'Avg Shift':>12} {'V2':>8} {'R':>8} {'S':>8} {'A':>8}")
    print("-" * 80)
    for method, res in all_results.items():
        if "error" in res:
            continue
        in_acc = res.get("imagenet_acc", 0.0)
        avg_shift = res.get("avg_shift_acc", 0.0)
        v2 = res.get("shift_accs", {}).get("imagenet_v2", 0.0)
        r = res.get("shift_accs", {}).get("imagenet_r", 0.0)
        s = res.get("shift_accs", {}).get("imagenet_s", 0.0)
        a = res.get("shift_accs", {}).get("imagenet_a", 0.0)
        print(f"{method:<20} {in_acc:>10.2f} {avg_shift:>12.2f} {v2:>8.2f} {r:>8.2f} {s:>8.2f} {a:>8.2f}")


def main():
    parser = argparse.ArgumentParser(description="Run CLIP robustness experiments")
    parser.add_argument("--methods", nargs="+", default=ROBUSTNESS_METHODS)
    parser.add_argument("--imagenet_root", type=str, required=True,
                        help="Path to ImageNet dataset root")
    parser.add_argument("--imagenet_v2", type=str, default="",
                        help="Path to ImageNet-V2 dataset")
    parser.add_argument("--imagenet_r", type=str, default="",
                        help="Path to ImageNet-R dataset")
    parser.add_argument("--imagenet_s", type=str, default="",
                        help="Path to ImageNet-S dataset")
    parser.add_argument("--imagenet_a", type=str, default="",
                        help="Path to ImageNet-A dataset")
    parser.add_argument("--output_dir", type=str, default="./outputs/robustness")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_wise", action="store_true", help="Skip WiSE sweep")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config = RobustnessConfig()

    shift_data_roots = {}
    if args.imagenet_v2:
        shift_data_roots["imagenet_v2"] = args.imagenet_v2
    if args.imagenet_r:
        shift_data_roots["imagenet_r"] = args.imagenet_r
    if args.imagenet_s:
        shift_data_roots["imagenet_s"] = args.imagenet_s
    if args.imagenet_a:
        shift_data_roots["imagenet_a"] = args.imagenet_a

    run_robustness_all_methods(
        methods=args.methods,
        config=config,
        device=device,
        imagenet_root=args.imagenet_root,
        shift_data_roots=shift_data_roots,
        output_dir=args.output_dir,
        run_wise=not args.no_wise,
    )


if __name__ == "__main__":
    main()
