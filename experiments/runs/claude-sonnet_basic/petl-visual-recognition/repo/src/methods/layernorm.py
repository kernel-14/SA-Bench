"""
LayerNorm Tuning: Strong Baselines for Parameter-Efficient Few-Shot Fine-Tuning.
Basu et al., 2024.

Only tunes the LayerNorm parameters in each Transformer layer.
"""

import torch
import torch.nn as nn


def apply_layernorm(model, **kwargs):
    """
    Apply LayerNorm tuning to a ViT model.
    Freezes all parameters except LayerNorm parameters.
    
    Args:
        model: ViT model (timm)
        **kwargs: Additional arguments (unused)
    
    Returns:
        model with only LayerNorm parameters trainable
    """
    # First freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze LayerNorm parameters
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            for param in module.parameters():
                param.requires_grad = True
    
    # Always keep head trainable
    for name, param in model.named_parameters():
        if 'head' in name:
            param.requires_grad = True
    
    return model
