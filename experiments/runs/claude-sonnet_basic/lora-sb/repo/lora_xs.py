"""
LoRA-XS implementation for comparison baseline.

LoRA-XS inserts a learnable r×r matrix R between fixed B and A matrices:
    W = W0 + s*B*R*A

B and A are initialized using SVD of the pre-trained weight matrix (PiSSA-style),
using the most significant singular vectors.

Reference: Bałazy et al., "LoRA-XS: Low-Rank Adaptation with Extremely Small Number
of Parameters" (arXiv:2405.17604)
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Dict, Optional
import math


class LoRAXSLayer(nn.Module):
    """
    LoRA-XS layer: W = W0 + s*B*R*A
    
    B and A are fixed (initialized from SVD of pre-trained weights).
    Only R (r×r) is trainable.
    s = alpha/r is the scaling factor.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Fixed matrices (not trainable)
        self.register_buffer('B', torch.zeros(out_features, rank))
        self.register_buffer('A', torch.zeros(rank, in_features))
        
        # Trainable r×r matrix, initialized to identity
        self.R = nn.Parameter(torch.eye(rank))
        
        self._initialized = False
    
    def initialize_from_weight(self, weight: Tensor):
        """
        Initialize B and A using SVD of the pre-trained weight matrix.
        Uses the top-r singular vectors (PiSSA-style initialization).
        
        W ≈ U[:, :r] * S[:r] * V[:r, :]
        B = U[:, :r], A = V[:r, :], R = diag(S[:r]) / scaling
        """
        try:
            U, S, Vh = torch.linalg.svd(weight.float(), full_matrices=False)
        except Exception:
            U, S, Vh = torch.svd(weight.float(), some=True)
        
        r = self.rank
        
        # B: (out_features, r), A: (r, in_features)
        B_init = U[:, :r].contiguous()
        A_init = Vh[:r, :].contiguous()
        # R initialized to identity (standard LoRA-XS)
        R_init = torch.eye(r)
        
        self.B.copy_(B_init.to(self.B.dtype))
        self.A.copy_(A_init.to(self.A.dtype))
        self.R.data.copy_(R_init)
        self._initialized = True
    
    def forward(self, x: Tensor) -> Tensor:
        # s * B * R * A * x
        lora_out = x @ self.A.T  # (..., rank)
        lora_out = lora_out @ self.R.T  # (..., rank)
        lora_out = lora_out @ self.B.T  # (..., out_features)
        return self.scaling * lora_out


class LoRAXSLinear(nn.Module):
    """Linear layer with LoRA-XS adaptation."""
    
    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
        
        self.lora_xs = LoRAXSLayer(
            base_layer.in_features,
            base_layer.out_features,
            rank=rank,
            alpha=alpha,
        )
        self.rank = rank
    
    def forward(self, x: Tensor) -> Tensor:
        return self.base_layer(x) + self.lora_xs(x)
    
    @property
    def weight(self):
        return self.base_layer.weight
    
    @property
    def bias(self):
        return self.base_layer.bias


def apply_lora_xs(
    model: nn.Module,
    target_modules: List[str],
    rank: int,
    alpha: float,
) -> nn.Module:
    """Apply LoRA-XS to specified modules."""
    
    def _get_submodule(model, key):
        parent = model
        parts = key.split('.')
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent, parts[-1], getattr(parent, parts[-1])
    
    modules_to_replace = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            for target in target_modules:
                if target in name or name.endswith(target):
                    modules_to_replace[name] = module
                    break
    
    for name, module in modules_to_replace.items():
        parent, child_name, _ = _get_submodule(model, name)
        lora_xs_layer = LoRAXSLinear(module, rank=rank, alpha=alpha)
        setattr(parent, child_name, lora_xs_layer)
    
    return model


def initialize_lora_xs_pissa(model: nn.Module):
    """
    Initialize LoRA-XS layers using SVD of pre-trained weights (PiSSA-style).
    """
    for name, module in model.named_modules():
        if isinstance(module, LoRAXSLinear):
            weight = module.base_layer.weight.data
            module.lora_xs.initialize_from_weight(weight)
