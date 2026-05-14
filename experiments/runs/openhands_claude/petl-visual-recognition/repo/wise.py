"""
Weight-Space Ensembles (WiSE) for PEFT models.

WiSE linearly interpolates the weights of a fine-tuned model with those of
the original pre-trained backbone using a mixing coefficient α ∈ [0, 1].

For different PEFT method types:
  - Direct selective tuning (BitFit, DiffFit, LayerNorm):
      Merge the fine-tuned parameters with the original model parameters.
  - Adapter-based methods (Houl., Pfeif., AdaptFormer, ConvPass, RepAdapter):
      Scale the adapter output by α (feature-level ensemble).
      α=0: pure pre-trained features, α=1: full adapter contribution.
  - Efficient selective tuning (LoRA, FacT):
      Scale the additive residuals ΔW by α.
      α=0: original weights, α=1: full LoRA/FacT update.
  - Head: always interpolated between fine-tuned and zero-shot weights.

Reference: Wortsman et al., "Robust Fine-Tuning of Zero-Shot Models", CVPR 2022.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluate import get_predictions, compute_top1_accuracy


def wise_merge_selective(
    finetuned_model: nn.Module,
    pretrained_model: nn.Module,
    alpha: float,
) -> nn.Module:
    """
    WiSE for direct selective tuning methods (BitFit, LayerNorm, DiffFit).
    Linearly interpolates all parameters:
      θ_wise = α * θ_finetuned + (1 - α) * θ_pretrained

    Args:
        finetuned_model: fine-tuned model
        pretrained_model: original pre-trained model
        alpha: mixing coefficient (α=1 → fully fine-tuned, α=0 → pre-trained)

    Returns:
        merged model (copy of finetuned_model with interpolated weights)
    """
    merged = copy.deepcopy(finetuned_model)
    pretrained_state = pretrained_model.state_dict()
    finetuned_state = finetuned_model.state_dict()

    merged_state = {}
    for key in finetuned_state:
        if key in pretrained_state:
            merged_state[key] = (
                alpha * finetuned_state[key] + (1 - alpha) * pretrained_state[key]
            )
        else:
            merged_state[key] = finetuned_state[key]

    merged.load_state_dict(merged_state)
    return merged


def wise_scale_adapters(
    model: nn.Module,
    alpha: float,
) -> nn.Module:
    """
    WiSE for adapter-based methods: scale adapter outputs by α.
    This is equivalent to feature-level ensembling:
      h_wise = h_original + α * Adapter(h)

    Modifies the scale factor of all Adapter modules in-place on a copy.
    """
    merged = copy.deepcopy(model)

    for module in merged.modules():
        # Scale the adapter's scale factor
        if hasattr(module, "scale") and hasattr(module, "down_proj"):
            module.scale = module.scale * alpha
        # For RepAdapter
        if hasattr(module, "scale") and hasattr(module, "down_proj") and hasattr(module, "up_projs"):
            module.scale = module.scale * alpha

    return merged


def wise_scale_lora(
    model: nn.Module,
    alpha: float,
) -> nn.Module:
    """
    WiSE for LoRA: scale the low-rank updates by α.
    ΔW_wise = α * W_down @ W_up
    Achieved by scaling W_up by α.
    """
    merged = copy.deepcopy(model)

    for module in merged.modules():
        if hasattr(module, "lora_q_up") and hasattr(module, "lora_v_up"):
            with torch.no_grad():
                module.lora_q_up.weight.mul_(alpha)
                module.lora_v_up.weight.mul_(alpha)

    return merged


def wise_scale_fact(
    model: nn.Module,
    alpha: float,
) -> nn.Module:
    """
    WiSE for FacT: scale the tensor decomposition factors by α.
    ΔW_wise = α * s * Σ ×_2 U^T ×_3 V^T
    Achieved by scaling the scale factor s by α.
    """
    merged = copy.deepcopy(model)

    for module in merged.modules():
        if hasattr(module, "scale") and hasattr(module, "sigma"):
            module.scale = module.scale * alpha
        if hasattr(module, "scale") and hasattr(module, "shared_a"):
            module.scale = module.scale * alpha

    return merged


def wise_interpolate_head(
    finetuned_head: nn.Linear,
    zeroshot_head: nn.Linear,
    alpha: float,
) -> nn.Linear:
    """
    Interpolate classification head weights:
      W_head_wise = α * W_finetuned + (1 - α) * W_zeroshot
    """
    merged_head = copy.deepcopy(finetuned_head)
    with torch.no_grad():
        merged_head.weight.data = (
            alpha * finetuned_head.weight.data
            + (1 - alpha) * zeroshot_head.weight.data
        )
        if finetuned_head.bias is not None and zeroshot_head.bias is not None:
            merged_head.bias.data = (
                alpha * finetuned_head.bias.data
                + (1 - alpha) * zeroshot_head.bias.data
            )
    return merged_head


def apply_wise(
    finetuned_model: nn.Module,
    pretrained_model: Optional[nn.Module],
    peft_method: str,
    alpha: float,
    zeroshot_head: Optional[nn.Linear] = None,
) -> nn.Module:
    """
    Apply WiSE to a PEFT model with mixing coefficient α.

    Args:
        finetuned_model: fine-tuned PEFT model
        pretrained_model: original pre-trained model (needed for selective methods)
        peft_method: PEFT method name
        alpha: mixing coefficient (α=1 → fine-tuned, α=0 → pre-trained)
        zeroshot_head: zero-shot classifier head for head interpolation

    Returns:
        WiSE-merged model
    """
    selective_methods = {"bitfit", "layernorm", "difffit", "linear", "full"}
    adapter_methods = {"houl_adapter", "pfeif_adapter", "adaptformer", "convpass", "repadapter"}
    lora_methods = {"lora"}
    fact_methods = {"fact_tt", "fact_tk"}

    if peft_method in selective_methods:
        if pretrained_model is None:
            raise ValueError("pretrained_model required for selective tuning WiSE")
        merged = wise_merge_selective(finetuned_model, pretrained_model, alpha)
    elif peft_method in adapter_methods:
        merged = wise_scale_adapters(finetuned_model, alpha)
    elif peft_method in lora_methods:
        merged = wise_scale_lora(finetuned_model, alpha)
    elif peft_method in fact_methods:
        merged = wise_scale_fact(finetuned_model, alpha)
    else:
        # Default: treat as selective
        if pretrained_model is not None:
            merged = wise_merge_selective(finetuned_model, pretrained_model, alpha)
        else:
            merged = copy.deepcopy(finetuned_model)

    # Interpolate head if zero-shot head provided
    if zeroshot_head is not None and hasattr(merged, "head"):
        merged.head = wise_interpolate_head(merged.head, zeroshot_head, alpha)

    return merged


def wise_sweep(
    finetuned_model: nn.Module,
    pretrained_model: Optional[nn.Module],
    peft_method: str,
    target_loader: DataLoader,
    shift_loaders: Dict[str, DataLoader],
    device: torch.device,
    alphas: Optional[List[float]] = None,
    zeroshot_head: Optional[nn.Linear] = None,
) -> List[Dict]:
    """
    Sweep over mixing coefficients α and evaluate on target + distribution shift data.
    Reproduces Figure 1c and Figure 14.

    Args:
        finetuned_model: fine-tuned PEFT model
        pretrained_model: original pre-trained model
        peft_method: PEFT method name
        target_loader: DataLoader for target distribution (ImageNet)
        shift_loaders: {dataset_name: DataLoader} for distribution shifts
        device: torch device
        alphas: list of α values to sweep
        zeroshot_head: zero-shot classifier head

    Returns:
        list of dicts with alpha, target_acc, avg_shift_acc, per_shift_acc
    """
    if alphas is None:
        alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    results = []

    for alpha in alphas:
        print(f"  WiSE α={alpha:.1f}")
        merged = apply_wise(
            finetuned_model, pretrained_model, peft_method, alpha, zeroshot_head
        )
        merged = merged.to(device)
        merged.eval()

        # Target distribution accuracy
        _, target_preds, target_labels = get_predictions(merged, target_loader, device)
        target_acc = compute_top1_accuracy(target_preds, target_labels)

        # Distribution shift accuracy
        shift_accs = {}
        for ds_name, ds_loader in shift_loaders.items():
            _, shift_preds, shift_labels = get_predictions(merged, ds_loader, device)
            shift_accs[ds_name] = compute_top1_accuracy(shift_preds, shift_labels)

        avg_shift_acc = sum(shift_accs.values()) / len(shift_accs) if shift_accs else 0.0

        result = {
            "alpha": alpha,
            "target_acc": target_acc,
            "avg_shift_acc": avg_shift_acc,
            "per_shift_acc": shift_accs,
        }
        results.append(result)
        print(f"    Target: {target_acc:.2f}%, Avg shift: {avg_shift_acc:.2f}%")

    return results


def find_best_wise_alpha(
    wise_results: List[Dict],
    metric: str = "avg_shift_acc",
) -> Tuple[float, Dict]:
    """Find the α that maximizes the given metric."""
    best = max(wise_results, key=lambda x: x[metric])
    return best["alpha"], best
