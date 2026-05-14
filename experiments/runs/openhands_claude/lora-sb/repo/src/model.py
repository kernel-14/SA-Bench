"""Model wrapping utilities: apply PEFT methods to pre-trained models."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from .lora_layers import (
    DoRALayer,
    LinearWithLoRA,
    LinearWithLoRASB,
    LinearWithLoRAXS,
    LoRALayer,
    LoRAProLayer,
    LoRASBLayer,
    LoRAXSLayer,
    PiSSALayer,
    rsLoRALayer,
)


# Target module names per task type
CAUSAL_LM_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
SEQ_CLS_TARGET_MODULES = ["query", "key", "value", "dense"]  # RoBERTa self-attention only


def _replace_linear_with_lora(
    model: nn.Module,
    target_module_names: List[str],
    rank: int,
    alpha: float,
    dropout: float,
    method: str,
) -> nn.Module:
    """Replace target Linear layers with LoRA/rsLoRA/PiSSA/DoRA/LoRA-Pro variants."""
    for parent_name, parent_module in list(model.named_modules()):
        for child_name, child_module in list(parent_module.named_children()):
            if not isinstance(child_module, nn.Linear):
                continue
            if child_name not in target_module_names:
                continue

            in_features = child_module.in_features
            out_features = child_module.out_features
            weight = child_module.weight.data

            if method == "lora":
                lora_layer = LoRALayer(in_features, out_features, rank, alpha, dropout)
                new_module = LinearWithLoRA(child_module, lora_layer)
            elif method == "rslora":
                lora_layer = rsLoRALayer(in_features, out_features, rank, alpha, dropout)
                new_module = LinearWithLoRA(child_module, lora_layer)
            elif method == "pissa":
                lora_layer = PiSSALayer(in_features, out_features, rank, alpha, weight, dropout)
                new_module = LinearWithLoRA(child_module, lora_layer)
            elif method == "dora":
                lora_layer = DoRALayer(in_features, out_features, rank, alpha, weight, dropout)
                new_module = LinearWithLoRA(child_module, lora_layer)
            elif method == "lora_pro":
                lora_layer = LoRAProLayer(in_features, out_features, rank, alpha, dropout)
                new_module = LinearWithLoRA(child_module, lora_layer)
            else:
                raise ValueError(f"Unknown method: {method}")

            # Freeze original weight
            child_module.weight.requires_grad_(False)
            if child_module.bias is not None:
                child_module.bias.requires_grad_(False)

            setattr(parent_module, child_name, new_module)

    return model


def _replace_linear_with_lora_xs(
    model: nn.Module,
    target_module_names: List[str],
    rank: int,
    alpha: float,
    dropout: float,
) -> nn.Module:
    """Replace target Linear layers with LoRA-XS layers."""
    for parent_name, parent_module in list(model.named_modules()):
        for child_name, child_module in list(parent_module.named_children()):
            if not isinstance(child_module, nn.Linear):
                continue
            if child_name not in target_module_names:
                continue

            in_features = child_module.in_features
            out_features = child_module.out_features
            weight = child_module.weight.data

            lora_layer = LoRAXSLayer(in_features, out_features, rank, alpha, weight, dropout)
            new_module = LinearWithLoRAXS(child_module, lora_layer)

            child_module.weight.requires_grad_(False)
            if child_module.bias is not None:
                child_module.bias.requires_grad_(False)

            setattr(parent_module, child_name, new_module)

    return model


def _replace_linear_with_lora_sb(
    model: nn.Module,
    target_module_names: List[str],
    init_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    dropout: float,
) -> nn.Module:
    """Replace target Linear layers with LoRA-SB layers using pre-computed initialization."""
    for parent_name, parent_module in list(model.named_modules()):
        for child_name, child_module in list(parent_module.named_children()):
            if not isinstance(child_module, nn.Linear):
                continue
            if child_name not in target_module_names:
                continue

            # Build full name to look up in init_dict
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name

            if full_name not in init_dict:
                continue

            B_init, R_init, A_init = init_dict[full_name]
            rank = B_init.shape[1]

            lora_layer = LoRASBLayer(
                in_features=child_module.in_features,
                out_features=child_module.out_features,
                rank=rank,
                B_init=B_init,
                R_init=R_init,
                A_init=A_init,
                dropout=dropout,
            )
            new_module = LinearWithLoRASB(child_module, lora_layer)

            child_module.weight.requires_grad_(False)
            if child_module.bias is not None:
                child_module.bias.requires_grad_(False)

            setattr(parent_module, child_name, new_module)

    return model


def freeze_base_model(model: nn.Module) -> None:
    """Freeze all parameters in the base model."""
    for param in model.parameters():
        param.requires_grad_(False)


def count_trainable_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: nn.Module) -> int:
    """Count total parameters."""
    return sum(p.numel() for p in model.parameters())


def get_trainable_parameter_names(model: nn.Module) -> List[str]:
    return [name for name, p in model.named_parameters() if p.requires_grad]


def apply_lora(
    model: nn.Module,
    target_module_names: List[str],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    method: str = "lora",
) -> nn.Module:
    """Apply LoRA (or variant) to the model.

    Args:
        model: Pre-trained model.
        target_module_names: Names of Linear layers to adapt.
        rank: LoRA rank.
        alpha: LoRA alpha scaling factor.
        dropout: Dropout rate.
        method: One of 'lora', 'rslora', 'pissa', 'dora', 'lora_pro'.

    Returns:
        Model with LoRA adapters applied.
    """
    freeze_base_model(model)
    model = _replace_linear_with_lora(model, target_module_names, rank, alpha, dropout, method)
    return model


def apply_lora_xs(
    model: nn.Module,
    target_module_names: List[str],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> nn.Module:
    """Apply LoRA-XS to the model.

    Args:
        model: Pre-trained model.
        target_module_names: Names of Linear layers to adapt.
        rank: LoRA rank.
        alpha: LoRA alpha (set to r per LoRA-XS guidelines).
        dropout: Dropout rate.

    Returns:
        Model with LoRA-XS adapters applied.
    """
    freeze_base_model(model)
    model = _replace_linear_with_lora_xs(model, target_module_names, rank, alpha, dropout)
    return model


def apply_lora_sb(
    model: nn.Module,
    target_module_names: List[str],
    init_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    dropout: float = 0.0,
) -> nn.Module:
    """Apply LoRA-SB to the model using pre-computed initialization.

    Args:
        model: Pre-trained model.
        target_module_names: Names of Linear layers to adapt.
        init_dict: Dict mapping full module name to (B_init, R_init, A_init).
        dropout: Dropout rate.

    Returns:
        Model with LoRA-SB adapters applied.
    """
    freeze_base_model(model)
    model = _replace_linear_with_lora_sb(model, target_module_names, init_dict, dropout)
    return model


def load_causal_lm(
    model_name: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
) -> Tuple[nn.Module, object]:
    """Load a causal language model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    return model, tokenizer


def load_seq_cls(
    model_name: str,
    num_labels: int,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[nn.Module, object]:
    """Load a sequence classification model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    return model, tokenizer
