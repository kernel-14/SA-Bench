"""
BitFit: Simple Parameter-Efficient Fine-Tuning for Transformer-based Masked Language-Models.
Zaken et al., 2022.

BitFit only tunes the bias terms of the pre-trained model.
For ViT, this includes:
- Bias terms in QKV projections and FC layer in MSA block
- Bias terms in two FC layers in MLP block
- Bias terms in two LN blocks
- Bias in patch embedding projection
"""

import torch
import torch.nn as nn


def apply_bitfit(model, **kwargs):
    """
    Apply BitFit to a ViT model.
    Freezes all parameters except bias terms.
    
    Args:
        model: ViT model (timm)
        **kwargs: Additional arguments (unused)
    
    Returns:
        model with only bias terms trainable
    """
    # First freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze bias terms
    for name, param in model.named_parameters():
        if 'bias' in name:
            param.requires_grad = True
    
    # Always keep head trainable
    for name, param in model.named_parameters():
        if 'head' in name:
            param.requires_grad = True
    
    return model
