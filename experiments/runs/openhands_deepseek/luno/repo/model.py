"""Fourier Neural Operator model (Li et al., 2021) as described in the LUNO paper."""

from typing import Tuple, Callable
import jax
import jax.numpy as jnp
from flax import nnx

from layers import (
    LiftingLayer,
    ProjectionLayer,
    FourierBlock1d,
    FourierBlock2d,
)


class FNO1d(nnx.Module):
    """1D Fourier Neural Operator.

    Architecture: Lifting -> (FourierBlock)^{L} -> Projection
    With 4 Fourier blocks, 12 modes, 18 hidden dims as in paper.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        n_modes: int,
        n_blocks: int,
        rngs: nnx.Rngs,
    ):
        self.lifting = LiftingLayer(input_dim, hidden_dim, rngs)
        self.blocks = [
            FourierBlock1d(hidden_dim, n_modes, rngs)
            for _ in range(n_blocks)
        ]
        self.projection = ProjectionLayer(hidden_dim, output_dim, rngs)
        self.hidden_dim = hidden_dim

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (batch, N, input_dim) or (N, input_dim)
        # Lifting
        v = self.lifting(x)  # (..., N, hidden_dim)
        # Fourier blocks
        for block in self.blocks:
            v = block(v)
        # Projection
        out = self.projection(v)  # (..., N, output_dim)
        return out

    def get_hidden_states(self, x: jnp.ndarray) -> dict:
        """Get intermediate hidden states for LUNO last-layer linearization."""
        states = {}
        v = self.lifting(x)
        for i, block in enumerate(self.blocks):
            states[f"pre_block_{i}"] = v
            v = block(v)
        states["pre_projection"] = v  # v^{(L)}
        states["output"] = self.projection(v)
        return states

    def last_layer_params(self):
        """Extract last Fourier block parameters for last-layer Laplace."""
        last_block = self.blocks[-1]
        return {
            "R_real": last_block.spectral_conv.R_real.value,
            "R_imag": last_block.spectral_conv.R_imag.value,
            "W": last_block.W.value,
        }


class FNO2d(nnx.Module):
    """2D Fourier Neural Operator.

    Architecture: Lifting -> (FourierBlock)^{L} -> Projection
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        n_modes: Tuple[int, int],
        n_blocks: int,
        rngs: nnx.Rngs,
    ):
        self.lifting = LiftingLayer(input_dim, hidden_dim, rngs)
        self.blocks = [
            FourierBlock2d(hidden_dim, n_modes, rngs)
            for _ in range(n_blocks)
        ]
        self.projection = ProjectionLayer(hidden_dim, output_dim, rngs)
        self.hidden_dim = hidden_dim

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (batch, H, W, input_dim) or (H, W, input_dim)
        v = self.lifting(x)  # (..., H, W, hidden_dim)
        for block in self.blocks:
            v = block(v)
        out = self.projection(v)  # (..., H, W, output_dim)
        return out

    def get_hidden_states(self, x: jnp.ndarray) -> dict:
        """Get intermediate hidden states for LUNO last-layer linearization."""
        states = {}
        v = self.lifting(x)
        for i, block in enumerate(self.blocks):
            states[f"pre_block_{i}"] = v
            v = block(v)
        states["pre_projection"] = v
        states["output"] = self.projection(v)
        return states

    def last_layer_params(self):
        """Extract last Fourier block parameters for last-layer Laplace."""
        last_block = self.blocks[-1]
        return {
            "R_real": last_block.spectral_conv.R_real.value,
            "R_imag": last_block.spectral_conv.R_imag.value,
            "W": last_block.W.value,
        }


def create_fno(
    spatial_dim: int,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    n_modes,
    n_blocks: int,
    rngs: nnx.Rngs,
):
    """Factory function to create 1D or 2D FNO."""
    if spatial_dim == 1:
        n_modes = n_modes if isinstance(n_modes, int) else n_modes[0]
        return FNO1d(input_dim, output_dim, hidden_dim, n_modes, n_blocks, rngs)
    elif spatial_dim == 2:
        return FNO2d(input_dim, output_dim, hidden_dim, tuple(n_modes), n_blocks, rngs)
    else:
        raise ValueError(f"Unsupported spatial dimension: {spatial_dim}")
