"""
Vision Transformer (ViT) backbone with support for PEFT methods.
Based on timm's ViT implementation with modifications for PEFT.
"""

import torch
import torch.nn as nn
from timm.models import create_model


def create_vit_model(model_name='vit_base_patch16_224_in21k', pretrained=True, num_classes=1000, drop_path_rate=0.0):
    """Create a ViT model with optional pretrained weights."""
    model = create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
    )
    return model


def freeze_backbone(model):
    """Freeze all backbone parameters."""
    for name, param in model.named_parameters():
        if 'head' not in name:
            param.requires_grad = False


def unfreeze_all(model):
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True


def count_trainable_params(model):
    """Count trainable parameters in millions."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def count_total_params(model):
    """Count total parameters in millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6
