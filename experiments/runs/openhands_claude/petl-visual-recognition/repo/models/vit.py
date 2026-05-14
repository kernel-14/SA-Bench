"""
ViT backbone wrapper that supports all PEFT methods.

The intermediate features within a Transformer block follow the notation from Figure 9:
  h1  = input Z_{m-1}
  h2  = LN1(h1)
  h3  = [Q, K, V] projections
  h4  = attention output (before FC_attn projection)
  h5  = MSA(h2) + h1  (Z'_m, after residual)
  h6  = LN2(h5)
  h7  = GELU(h6 @ W1 + b1)  (MLP intermediate)
  h8  = h7 @ W2 + b2  (MLP output before residual)
  h9  = h8 + h5  (Z_m, after MLP residual)
  h10 = output of layer
"""

from __future__ import annotations

import math
from functools import partial
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import Block, Attention, Mlp


def build_vit(
    model_name: str = "vit_base_patch16_224",
    pretrained: bool = True,
    num_classes: int = 0,
    drop_path_rate: float = 0.0,
    img_size: int = 224,
) -> nn.Module:
    """Load a timm ViT model with optional pretrained weights."""
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        img_size=img_size,
    )
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all backbone parameters."""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_head(model: nn.Module) -> None:
    """Unfreeze the classification head."""
    if hasattr(model, "head"):
        for param in model.head.parameters():
            param.requires_grad = True


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class PEFTViT(nn.Module):
    """
    Wrapper around a timm ViT that applies a PEFT method.

    The PEFT method is applied by replacing the transformer blocks with
    modified versions that incorporate the PEFT modules.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        peft_method: str = "linear",
        peft_config: Optional[Dict] = None,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.peft_method = peft_method
        self.peft_config = peft_config or {}
        self.num_classes = num_classes

        embed_dim = backbone.embed_dim
        num_layers = len(backbone.blocks)

        # Classification head (randomly initialized)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.zeros_(self.head.bias)
        nn.init.trunc_normal_(self.head.weight, std=0.02)

        # Apply PEFT method
        self._apply_peft(embed_dim, num_layers, drop_path_rate)

    def _apply_peft(self, embed_dim: int, num_layers: int, drop_path_rate: float) -> None:
        """Freeze backbone and configure trainable parameters per PEFT method."""
        method = self.peft_method

        if method == "linear":
            freeze_backbone(self.backbone)

        elif method == "full":
            # All parameters trainable
            pass

        elif method == "bitfit":
            from models.peft.selective import apply_bitfit
            freeze_backbone(self.backbone)
            apply_bitfit(self.backbone)

        elif method == "layernorm":
            from models.peft.selective import apply_layernorm_tuning
            freeze_backbone(self.backbone)
            apply_layernorm_tuning(self.backbone)

        elif method == "difffit":
            from models.peft.selective import apply_difffit
            freeze_backbone(self.backbone)
            apply_difffit(self.backbone)

        elif method == "ssf":
            from models.peft.selective import apply_ssf
            freeze_backbone(self.backbone)
            apply_ssf(self.backbone, embed_dim)

        elif method == "vpt_shallow":
            from models.peft.vpt import apply_vpt_shallow
            freeze_backbone(self.backbone)
            apply_vpt_shallow(self.backbone, **self.peft_config)

        elif method == "vpt_deep":
            from models.peft.vpt import apply_vpt_deep
            freeze_backbone(self.backbone)
            apply_vpt_deep(self.backbone, **self.peft_config)

        elif method == "houl_adapter":
            from models.peft.adapters import apply_houl_adapter
            freeze_backbone(self.backbone)
            apply_houl_adapter(self.backbone, embed_dim, **self.peft_config)

        elif method == "pfeif_adapter":
            from models.peft.adapters import apply_pfeif_adapter
            freeze_backbone(self.backbone)
            apply_pfeif_adapter(self.backbone, embed_dim, **self.peft_config)

        elif method == "adaptformer":
            from models.peft.adapters import apply_adaptformer
            freeze_backbone(self.backbone)
            apply_adaptformer(self.backbone, embed_dim, **self.peft_config)

        elif method == "convpass":
            from models.peft.adapters import apply_convpass
            freeze_backbone(self.backbone)
            apply_convpass(self.backbone, embed_dim, **self.peft_config)

        elif method == "repadapter":
            from models.peft.adapters import apply_repadapter
            freeze_backbone(self.backbone)
            apply_repadapter(self.backbone, embed_dim, **self.peft_config)

        elif method == "lora":
            from models.peft.lora import apply_lora
            freeze_backbone(self.backbone)
            apply_lora(self.backbone, embed_dim, **self.peft_config)

        elif method == "fact_tt":
            from models.peft.fact import apply_fact_tt
            freeze_backbone(self.backbone)
            apply_fact_tt(self.backbone, embed_dim, num_layers, **self.peft_config)

        elif method == "fact_tk":
            from models.peft.fact import apply_fact_tk
            freeze_backbone(self.backbone)
            apply_fact_tk(self.backbone, embed_dim, num_layers, **self.peft_config)

        else:
            raise ValueError(f"Unknown PEFT method: {method}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        # Use CLS token
        if features.dim() == 3:
            cls_token = features[:, 0]
        else:
            cls_token = features
        return self.head(cls_token)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return CLS token features before the head."""
        features = self.backbone.forward_features(x)
        if features.dim() == 3:
            return features[:, 0]
        return features


def build_peft_model(
    peft_method: str,
    num_classes: int,
    peft_config: Optional[Dict] = None,
    drop_path_rate: float = 0.0,
    pretrained: bool = True,
    model_name: str = "vit_base_patch16_224.augreg_in21k",
    img_size: int = 224,
) -> PEFTViT:
    """Build a ViT with the specified PEFT method applied."""
    backbone = build_vit(
        model_name=model_name,
        pretrained=pretrained,
        num_classes=0,  # Remove original head
        drop_path_rate=drop_path_rate,
        img_size=img_size,
    )
    model = PEFTViT(
        backbone=backbone,
        num_classes=num_classes,
        peft_method=peft_method,
        peft_config=peft_config,
        drop_path_rate=drop_path_rate,
    )
    return model
