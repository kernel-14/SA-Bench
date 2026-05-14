"""
Transfer learning utilities for neural operators.

Implements the adapter-based approach described in the paper:
"The lift and proj blocks are considered as the adapters, representing the mappings,
associated with the problem-specific part of dynamics: they are introduced to contain
different cardinality input sets, projecting into the fixed number of hidden features
and contain small number of parameters."

"In the fine-tuning stage we fix the parameters theta_F both to highlight the
generalizing properties of the operator and to reduce training costs: only the new
adapter parameters (theta_P_ft, theta_L_ft) are trained."
"""

import torch
import torch.nn as nn
from typing import Optional, List


def freeze_backbone(model: nn.Module) -> None:
    """
    Freeze the backbone (FNO blocks) parameters.
    
    In fine-tuning, only the adapter (lifting + projection) parameters are trained.
    The backbone parameters are frozen to preserve the pretrained representations.
    
    Args:
        model: Neural operator model with get_backbone_params() method
    """
    if hasattr(model, 'get_backbone_params'):
        backbone_params = set(id(p) for p in model.get_backbone_params())
        for param in model.parameters():
            if id(param) in backbone_params:
                param.requires_grad = False
    else:
        # Fallback: freeze all non-adapter parameters
        # Assumes lifting and projection are named accordingly
        for name, param in model.named_parameters():
            if 'lifting' not in name and 'projection' not in name:
                param.requires_grad = False


def unfreeze_backbone(model: nn.Module) -> None:
    """
    Unfreeze all model parameters.
    
    Args:
        model: Neural operator model
    """
    for param in model.parameters():
        param.requires_grad = True


def create_new_adapters(
    model: nn.Module,
    new_n_input: int,
    new_n_output: int,
) -> nn.Module:
    """
    Create new adapter layers for a fine-tuning task with different input/output dimensions.
    
    This implements the key contribution of the paper: the ability to handle
    different input function sets by replacing only the lifting and projection layers.
    
    Args:
        model: Pretrained neural operator model
        new_n_input: Number of input functions for the new task
        new_n_output: Number of output functions for the new task
    
    Returns:
        Model with new adapters (backbone frozen)
    """
    width = model.width
    
    # Replace lifting layer with new one for different input size
    model.lifting = nn.Linear(new_n_input, width)
    
    # Replace projection layer with new one for different output size
    model.projection = nn.Sequential(
        nn.Linear(width, width * 2),
        nn.GELU(),
        nn.Linear(width * 2, new_n_output),
    )
    
    # Initialize new adapters
    nn.init.xavier_uniform_(model.lifting.weight)
    nn.init.zeros_(model.lifting.bias)
    
    for layer in model.projection:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    # Freeze backbone
    freeze_backbone(model)
    
    return model


def get_adapter_param_count(model: nn.Module) -> int:
    """Count the number of trainable adapter parameters."""
    if hasattr(model, 'get_adapter_params'):
        return sum(p.numel() for p in model.get_adapter_params() if p.requires_grad)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_backbone_param_count(model: nn.Module) -> int:
    """Count the number of backbone parameters."""
    if hasattr(model, 'get_backbone_params'):
        return sum(p.numel() for p in model.get_backbone_params())
    return 0


def get_total_param_count(model: nn.Module) -> int:
    """Count total number of parameters."""
    return sum(p.numel() for p in model.parameters())


def print_param_summary(model: nn.Module, model_name: str = "Model") -> None:
    """Print a summary of model parameters."""
    total = get_total_param_count(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    print(f"\n{model_name} Parameter Summary:")
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Frozen parameters: {frozen:,}")
    
    if hasattr(model, 'get_adapter_params'):
        adapter = get_adapter_param_count(model)
        backbone = get_backbone_param_count(model)
        print(f"  Adapter parameters: {adapter:,}")
        print(f"  Backbone parameters: {backbone:,}")


def load_pretrained_backbone(
    model: nn.Module,
    checkpoint_path: str,
    strict: bool = False,
) -> nn.Module:
    """
    Load pretrained backbone weights into a model.
    
    Args:
        model: Target model
        checkpoint_path: Path to pretrained checkpoint
        strict: Whether to strictly enforce that the keys match
    
    Returns:
        Model with loaded backbone weights
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Load only backbone parameters (skip adapter parameters)
    model_state = model.state_dict()
    
    # Filter to only backbone parameters
    backbone_state = {
        k: v for k, v in state_dict.items()
        if k in model_state and 'lifting' not in k and 'projection' not in k
    }
    
    model_state.update(backbone_state)
    model.load_state_dict(model_state, strict=strict)
    
    print(f"Loaded {len(backbone_state)} backbone parameter tensors from {checkpoint_path}")
    
    return model
