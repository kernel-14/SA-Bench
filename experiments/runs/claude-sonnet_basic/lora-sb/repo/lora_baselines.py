"""
Baseline LoRA implementations for comparison.

Implements:
- LoRA: Standard low-rank adaptation (Hu et al., 2021)
- rsLoRA: Rank-stabilized LoRA with sqrt(r) scaling (Kalajdzievski, 2023)
- PiSSA: Principal Singular Values and Singular Vectors Adaptation (Meng et al., 2024)
- DoRA: Weight-Decomposed Low-Rank Adaptation (Liu et al., 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional
import math


# ============================================================
# Standard LoRA
# ============================================================

class LoRALayer(nn.Module):
    """
    Standard LoRA layer: W = W0 + s*B*A
    
    B: (out_features, rank), initialized to zeros
    A: (rank, in_features), initialized with Kaiming uniform
    s = alpha / rank
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()
        
        # Initialize A with Kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
    
    def forward(self, x: Tensor) -> Tensor:
        x_dropped = self.dropout(x)
        return self.scaling * (x_dropped @ self.lora_A.T @ self.lora_B.T)


class LoRALinear(nn.Module):
    """Linear layer with standard LoRA adaptation."""
    
    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
        
        self.lora = LoRALayer(
            base_layer.in_features,
            base_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        self.rank = rank
    
    def forward(self, x: Tensor) -> Tensor:
        return self.base_layer(x) + self.lora(x)
    
    @property
    def weight(self):
        return self.base_layer.weight
    
    @property
    def bias(self):
        return self.base_layer.bias


def apply_lora(
    model: nn.Module,
    target_modules: List[str],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> nn.Module:
    """Apply standard LoRA to specified modules."""
    
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
        lora_layer = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, child_name, lora_layer)
    
    return model


# ============================================================
# rsLoRA (Rank-Stabilized LoRA)
# ============================================================

class rsLoRALayer(nn.Module):
    """
    rsLoRA layer: W = W0 + (alpha/sqrt(r)) * B * A
    
    Uses sqrt(r) scaling instead of r for better rank stability.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        # rsLoRA uses alpha/sqrt(r) instead of alpha/r
        self.scaling = alpha / math.sqrt(rank)
        
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
    
    def forward(self, x: Tensor) -> Tensor:
        x_dropped = self.dropout(x)
        return self.scaling * (x_dropped @ self.lora_A.T @ self.lora_B.T)


class rsLoRALinear(nn.Module):
    """Linear layer with rsLoRA adaptation."""
    
    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
        
        self.lora = rsLoRALayer(
            base_layer.in_features,
            base_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        self.rank = rank
    
    def forward(self, x: Tensor) -> Tensor:
        return self.base_layer(x) + self.lora(x)
    
    @property
    def weight(self):
        return self.base_layer.weight
    
    @property
    def bias(self):
        return self.base_layer.bias


def apply_rslora(
    model: nn.Module,
    target_modules: List[str],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> nn.Module:
    """Apply rsLoRA to specified modules."""
    
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
        lora_layer = rsLoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, child_name, lora_layer)
    
    return model


# ============================================================
# PiSSA (Principal Singular Values and Singular Vectors Adaptation)
# ============================================================

class PiSSALayer(nn.Module):
    """
    PiSSA layer: Initializes A and B using principal singular vectors of W0.
    
    W0 = U*S*V^T ≈ U[:,:r]*S[:r]*V[:r,:]^T + residual
    A_init = sqrt(S[:r]) * V[:r,:]
    B_init = U[:,:r] * sqrt(S[:r])
    
    The residual is absorbed into the frozen weight.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()
        
        # Default initialization (will be overridden by initialize_from_weight)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
    
    def initialize_from_weight(self, weight: Tensor):
        """Initialize using principal singular vectors of the weight matrix."""
        try:
            U, S, Vh = torch.linalg.svd(weight.float(), full_matrices=False)
        except Exception:
            U, S, Vh = torch.svd(weight.float(), some=True)
        
        r = self.rank
        sqrt_S = torch.sqrt(S[:r])
        
        # A = sqrt(S) * V^T, B = U * sqrt(S)
        A_init = torch.diag(sqrt_S) @ Vh[:r, :]
        B_init = U[:, :r] @ torch.diag(sqrt_S)
        
        self.lora_A.data.copy_(A_init.to(self.lora_A.dtype))
        self.lora_B.data.copy_(B_init.to(self.lora_B.dtype))
    
    def forward(self, x: Tensor) -> Tensor:
        x_dropped = self.dropout(x)
        return self.scaling * (x_dropped @ self.lora_A.T @ self.lora_B.T)


class PiSSALinear(nn.Module):
    """Linear layer with PiSSA adaptation."""
    
    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        # For PiSSA, the base weight is modified to be the residual
        # W_residual = W0 - B_init * A_init (the low-rank part is moved to adapter)
        self.base_layer = base_layer
        
        self.lora = PiSSALayer(
            base_layer.in_features,
            base_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        self.rank = rank
        
        # Initialize from weight and update base weight to residual
        self._initialize_pissa()
        
        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
    
    def _initialize_pissa(self):
        """Initialize PiSSA and update base weight to residual."""
        weight = self.base_layer.weight.data
        self.lora.initialize_from_weight(weight)
        
        # Update base weight to residual: W_residual = W0 - scaling * B * A
        with torch.no_grad():
            lora_weight = self.lora.scaling * (self.lora.lora_B @ self.lora.lora_A)
            self.base_layer.weight.data -= lora_weight.to(weight.dtype)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.base_layer(x) + self.lora(x)
    
    @property
    def weight(self):
        return self.base_layer.weight
    
    @property
    def bias(self):
        return self.base_layer.bias


def apply_pissa(
    model: nn.Module,
    target_modules: List[str],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> nn.Module:
    """Apply PiSSA to specified modules."""
    
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
        lora_layer = PiSSALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, child_name, lora_layer)
    
    return model


# ============================================================
# DoRA (Weight-Decomposed Low-Rank Adaptation)
# ============================================================

class DoRALayer(nn.Module):
    """
    DoRA layer: Decomposes weight into magnitude and direction components.
    
    W = m * (W0 + B*A) / ||W0 + B*A||_col
    where m is a learnable magnitude vector.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        weight: Tensor,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Magnitude vector initialized to column norms of W0
        col_norms = weight.norm(dim=1, keepdim=True).squeeze()
        self.magnitude = nn.Parameter(col_norms.clone())
        
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
    
    def forward(self, x: Tensor, weight: Tensor) -> Tensor:
        # Compute adapted weight direction
        lora_weight = self.scaling * (self.lora_B @ self.lora_A)
        adapted_weight = weight + lora_weight
        
        # Normalize columns
        col_norms = adapted_weight.norm(dim=1, keepdim=True)
        normalized_weight = adapted_weight / col_norms
        
        # Scale by magnitude
        final_weight = self.magnitude.unsqueeze(1) * normalized_weight
        
        x_dropped = self.dropout(x)
        return F.linear(x_dropped, final_weight)


class DoRALinear(nn.Module):
    """Linear layer with DoRA adaptation."""
    
    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
        
        self.dora = DoRALayer(
            base_layer.in_features,
            base_layer.out_features,
            rank=rank,
            alpha=alpha,
            weight=base_layer.weight.data,
            dropout=dropout,
        )
        self.rank = rank
    
    def forward(self, x: Tensor) -> Tensor:
        # DoRA replaces the standard linear computation
        bias = self.base_layer.bias
        out = self.dora(x, self.base_layer.weight)
        if bias is not None:
            out = out + bias
        return out
    
    @property
    def weight(self):
        return self.base_layer.weight
    
    @property
    def bias(self):
        return self.base_layer.bias


def apply_dora(
    model: nn.Module,
    target_modules: List[str],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> nn.Module:
    """Apply DoRA to specified modules."""
    
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
        lora_layer = DoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, child_name, lora_layer)
    
    return model
