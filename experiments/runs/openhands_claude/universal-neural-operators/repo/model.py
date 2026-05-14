from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from modules import (
    CodaNOBackbone,
    FNOBackbone1d,
    FNOBackbone2d,
    LiftingLayer,
    LocalAttnFNOBackbone1d,
    LocalAttnFNOBackbone2d,
    MambaFNOBackbone1d,
    MambaFNOBackbone2d,
    PerceiverIOBackbone,
    ProjectionLayer,
    SwinBackbone,
)

# ---------------------------------------------------------------------------
# Base neural operator: G_θ = P ∘ F ∘ L
# ---------------------------------------------------------------------------

class BaseNeuralOperator(nn.Module):
    """
    Abstract base for all neural operator models.
    Subclasses provide the backbone F; this class wires L → F → P.
    """

    def __init__(self, lifting: LiftingLayer, backbone: nn.Module, projection: ProjectionLayer) -> None:
        super().__init__()
        self.lifting = lifting
        self.backbone = backbone
        self.projection = projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lifting(x)
        x = self.backbone(x)
        x = self.projection(x)
        return x

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters for fine-tuning (only adapters trained)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def adapter_parameters(self) -> List[nn.Parameter]:
        """Return only lifting + projection parameters (for fine-tuning optimizer)."""
        return list(self.lifting.parameters()) + list(self.projection.parameters())

    def backbone_parameters(self) -> List[nn.Parameter]:
        return list(self.backbone.parameters())


# ---------------------------------------------------------------------------
# FNO (baseline)
# ---------------------------------------------------------------------------

class FNO1d(BaseNeuralOperator):
    """
    Fourier Neural Operator for 1-D problems.
    ~10^6 parameters with default settings.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        hidden_dim: int = 64,
        n_layers: int = 4,
        modes: int = 16,
    ) -> None:
        lifting = LiftingLayer(n_in, hidden_dim)
        backbone = FNOBackbone1d(hidden_dim, n_layers, modes)
        projection = ProjectionLayer(hidden_dim, n_out)
        super().__init__(lifting, backbone, projection)


class FNO2d(BaseNeuralOperator):
    """
    Fourier Neural Operator for 2-D problems.
    ~10^6 parameters with default settings.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        hidden_dim: int = 64,
        n_layers: int = 4,
        modes1: int = 12,
        modes2: int = 12,
    ) -> None:
        lifting = LiftingLayer(n_in, hidden_dim)
        backbone = FNOBackbone2d(hidden_dim, n_layers, modes1, modes2)
        projection = ProjectionLayer(hidden_dim, n_out)
        super().__init__(lifting, backbone, projection)


# ---------------------------------------------------------------------------
# MambaFNO (post-lifting Mamba + FNO)
# ~10^7 parameters
# ---------------------------------------------------------------------------

class MambaFNO1d(BaseNeuralOperator):
    """MambaFNO for 1-D problems: Mamba SSM inserted after lifting."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        hidden_dim: int = 128,
        n_layers: int = 4,
        modes: int = 16,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        lifting = LiftingLayer(n_in, hidden_dim)
        backbone = MambaFNOBackbone1d(hidden_dim, n_layers, modes, d_state=d_state, d_conv=d_conv, expand=expand)
        projection = ProjectionLayer(hidden_dim, n_out)
        super().__init__(lifting, backbone, projection)


class MambaFNO2d(BaseNeuralOperator):
    """MambaFNO for 2-D problems: Mamba SSM inserted after lifting."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        hidden_dim: int = 128,
        n_layers: int = 4,
        modes1: int = 12,
        modes2: int = 12,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        lifting = LiftingLayer(n_in, hidden_dim)
        backbone = MambaFNOBackbone2d(hidden_dim, n_layers, modes1, modes2, d_state=d_state, d_conv=d_conv, expand=expand)
        projection = ProjectionLayer(hidden_dim, n_out)
        super().__init__(lifting, backbone, projection)


# ---------------------------------------------------------------------------
# LocalAttnFNO (post-lifting local attention + FNO)
# ---------------------------------------------------------------------------

class LocalAttnFNO1d(BaseNeuralOperator):
    """LocalAttnFNO for 1-D problems."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        hidden_dim: int = 128,
        n_layers: int = 4,
        modes: int = 16,
        num_heads: int = 4,
        window_size: int = 16,
    ) -> None:
        lifting = LiftingLayer(n_in, hidden_dim)
        backbone = LocalAttnFNOBackbone1d(hidden_dim, n_layers, modes, num_heads=num_heads, window_size=window_size)
        projection = ProjectionLayer(hidden_dim, n_out)
        super().__init__(lifting, backbone, projection)


class LocalAttnFNO2d(BaseNeuralOperator):
    """LocalAttnFNO for 2-D problems."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        hidden_dim: int = 128,
        n_layers: int = 4,
        modes1: int = 12,
        modes2: int = 12,
        num_heads: int = 4,
        window_size: int = 8,
    ) -> None:
        lifting = LiftingLayer(n_in, hidden_dim)
        backbone = LocalAttnFNOBackbone2d(hidden_dim, n_layers, modes1, modes2, num_heads=num_heads, window_size=window_size)
        projection = ProjectionLayer(hidden_dim, n_out)
        super().__init__(lifting, backbone, projection)


# ---------------------------------------------------------------------------
# PerceiverNO (Perceiver IO-based neural operator)
# ~10^8 parameters
# ---------------------------------------------------------------------------

class PerceiverNO(BaseNeuralOperator):
    """
    Perceiver IO-based neural operator.
    Encoder: cross-attn(FNO(input) → latent)
    Processor: self-attn(latent)
    Decoder: cross-attn(latent → output positions)
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        hidden_dim: int = 256,
        latent_dim: int = 256,
        n_latents: int = 128,
        n_self_attn_layers: int = 6,
        modes1: int = 12,
        modes2: int = 12,
        num_heads: int = 8,
        spatial_dim: int = 2,
    ) -> None:
        lifting = LiftingLayer(n_in, hidden_dim)
        backbone = PerceiverIOBackbone(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_latents=n_latents,
            n_self_attn_layers=n_self_attn_layers,
            modes1=modes1,
            modes2=modes2,
            num_heads=num_heads,
            spatial_dim=spatial_dim,
        )
        projection = ProjectionLayer(hidden_dim, n_out)
        super().__init__(lifting, backbone, projection)


# ---------------------------------------------------------------------------
# CoDA-NO (codomain attention neural operator)
# ~10^8 parameters
# ---------------------------------------------------------------------------

class CodaNO1d(BaseNeuralOperator):
    """CoDA-NO for 1-D problems."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
        modes1: int = 16,
        num_heads: int = 8,
    ) -> None:
        lifting = LiftingLayer(n_in, hidden_dim)
        backbone = CodaNOBackbone(hidden_dim, n_layers, modes1, modes1, num_heads=num_heads, spatial_dim=1)
        projection = ProjectionLayer(hidden_dim, n_out)
        super().__init__(lifting, backbone, projection)


class CodaNO2d(BaseNeuralOperator):
    """CoDA-NO for 2-D problems."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
        modes1: int = 12,
        modes2: int = 12,
        num_heads: int = 8,
    ) -> None:
        lifting = LiftingLayer(n_in, hidden_dim)
        backbone = CodaNOBackbone(hidden_dim, n_layers, modes1, modes2, num_heads=num_heads, spatial_dim=2)
        projection = ProjectionLayer(hidden_dim, n_out)
        super().__init__(lifting, backbone, projection)


# ---------------------------------------------------------------------------
# SwinNO (Swin-v2 transformer-based neural operator)
# ~10^9 parameters with large hidden_dim
# ---------------------------------------------------------------------------

class SwinNO(nn.Module):
    """
    Swin-v2 transformer-based neural operator for 2-D problems.
    Uses hierarchical shifted-window attention as the backbone.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        hidden_dim: int = 512,
        n_layers: int = 8,
        window_size: int = 8,
        num_heads: int = 16,
    ) -> None:
        super().__init__()
        self.lifting = LiftingLayer(n_in, hidden_dim)
        self.backbone = SwinBackbone(hidden_dim, n_layers, window_size=window_size, num_heads=num_heads)
        self.projection = ProjectionLayer(hidden_dim, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lifting(x)
        x = self.backbone(x)
        x = self.projection(x)
        return x

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def adapter_parameters(self) -> List[nn.Parameter]:
        return list(self.lifting.parameters()) + list(self.projection.parameters())

    def backbone_parameters(self) -> List[nn.Parameter]:
        return list(self.backbone.parameters())


# ---------------------------------------------------------------------------
# Multi-physics model wrapper
# Supports simultaneous training on N physics problems with shared backbone
# and problem-specific adapters (Section 3, pre-training phase)
# ---------------------------------------------------------------------------

class MultiPhysicsModel(nn.Module):
    """
    Wrapper for pre-training on N physics problems simultaneously.

    Each physics problem i has its own lifting L_i and projection P_i (adapters),
    while the backbone F is shared across all problems.

    Pre-training: optimize (θ_{P_1}, ..., θ_{P_N}, θ_F, θ_{L_1}, ..., θ_{L_N})
    Fine-tuning:  freeze θ_F, optimize only new adapter (θ_{P_ft}, θ_{L_ft})
    """

    def __init__(
        self,
        backbone: nn.Module,
        physics_configs: List[Dict],
        hidden_dim: int,
    ) -> None:
        """
        Args:
            backbone: shared FNO/Mamba/Perceiver backbone
            physics_configs: list of dicts with keys 'name', 'n_in', 'n_out'
            hidden_dim: hidden dimension of the backbone
        """
        super().__init__()
        self.backbone = backbone
        self.hidden_dim = hidden_dim

        self.liftings = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        for cfg in physics_configs:
            name = cfg["name"]
            self.liftings[name] = LiftingLayer(cfg["n_in"], hidden_dim)
            self.projections[name] = ProjectionLayer(hidden_dim, cfg["n_out"])

    def forward(self, x: torch.Tensor, physics_name: str) -> torch.Tensor:
        x = self.liftings[physics_name](x)
        x = self.backbone(x)
        x = self.projections[physics_name](x)
        return x

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def add_physics(self, name: str, n_in: int, n_out: int) -> None:
        """Add a new physics problem adapter for fine-tuning."""
        self.liftings[name] = LiftingLayer(n_in, self.hidden_dim)
        self.projections[name] = ProjectionLayer(self.hidden_dim, n_out)

    def adapter_parameters(self, physics_name: Optional[str] = None) -> List[nn.Parameter]:
        """Return adapter parameters for a specific physics or all physics."""
        if physics_name is not None:
            return (
                list(self.liftings[physics_name].parameters())
                + list(self.projections[physics_name].parameters())
            )
        params = []
        for lifting in self.liftings.values():
            params.extend(lifting.parameters())
        for proj in self.projections.values():
            params.extend(proj.parameters())
        return params

    def backbone_parameters(self) -> List[nn.Parameter]:
        return list(self.backbone.parameters())


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def build_model(
    model_type: str,
    n_in: int,
    n_out: int,
    spatial_dim: int = 2,
    **kwargs,
) -> nn.Module:
    """
    Factory for building any model by name.

    Args:
        model_type: one of 'fno', 'mamba_fno', 'local_attn_fno',
                    'perceiver_no', 'coda_no', 'swin_no'
        n_in: number of input functions
        n_out: number of output functions
        spatial_dim: 1 or 2
        **kwargs: model-specific hyperparameters
    """
    if model_type == "fno":
        cls = FNO1d if spatial_dim == 1 else FNO2d
    elif model_type == "mamba_fno":
        cls = MambaFNO1d if spatial_dim == 1 else MambaFNO2d
    elif model_type == "local_attn_fno":
        cls = LocalAttnFNO1d if spatial_dim == 1 else LocalAttnFNO2d
    elif model_type == "perceiver_no":
        return PerceiverNO(n_in=n_in, n_out=n_out, spatial_dim=spatial_dim, **kwargs)
    elif model_type == "coda_no":
        cls = CodaNO1d if spatial_dim == 1 else CodaNO2d
    elif model_type == "swin_no":
        if spatial_dim != 2:
            raise ValueError("SwinNO only supports 2-D spatial domains")
        return SwinNO(n_in=n_in, n_out=n_out, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return cls(n_in=n_in, n_out=n_out, **kwargs)


def build_multiphysics_model(
    backbone_type: str,
    physics_configs: List[Dict],
    hidden_dim: int,
    spatial_dim: int = 2,
    **backbone_kwargs,
) -> MultiPhysicsModel:
    """
    Build a MultiPhysicsModel with a shared backbone.

    Args:
        backbone_type: backbone architecture name
        physics_configs: list of {'name': str, 'n_in': int, 'n_out': int}
        hidden_dim: shared hidden dimension
        spatial_dim: 1 or 2
        **backbone_kwargs: passed to backbone constructor
    """
    _backbone_map = {
        ("fno", 1): FNOBackbone1d,
        ("fno", 2): FNOBackbone2d,
        ("mamba_fno", 1): MambaFNOBackbone1d,
        ("mamba_fno", 2): MambaFNOBackbone2d,
        ("local_attn_fno", 1): LocalAttnFNOBackbone1d,
        ("local_attn_fno", 2): LocalAttnFNOBackbone2d,
        ("perceiver_no", 1): PerceiverIOBackbone,
        ("perceiver_no", 2): PerceiverIOBackbone,
        ("coda_no", 1): CodaNOBackbone,
        ("coda_no", 2): CodaNOBackbone,
        ("swin_no", 2): SwinBackbone,
    }
    key = (backbone_type, spatial_dim)
    if key not in _backbone_map:
        raise ValueError(f"Unknown backbone type/spatial_dim combination: {key}")

    backbone_cls = _backbone_map[key]
    backbone = backbone_cls(hidden_dim=hidden_dim, **backbone_kwargs)
    return MultiPhysicsModel(backbone, physics_configs, hidden_dim)
