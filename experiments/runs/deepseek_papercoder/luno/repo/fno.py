"""
fno.py
=======
Fourier Neural Operator (FNO) module as described in Li et al. (2021) and
adapted for the LUNO reproduction experiments.

This file provides two Flax modules:
  - ``FourierBlock`` – one spectral–local transformation with truncation.
  - ``FourierNeuralOperator`` – the full FNO consisting of lifting, several
    Fourier blocks, and a final projection.
"""

from __future__ import annotations

import logging
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from config import ModelConfig  # safe import: no circular dependency

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: activation function resolver
# ---------------------------------------------------------------------------
def get_activation(name: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Return a JAX activation function by name.

    Parameters
    ----------
    name : str
        One of ``"gelu"``, ``"relu"``, (case‑insensitive).

    Returns
    -------
    callable
        The activation function.
    """
    name = name.lower()
    if name == "gelu":
        return jax.nn.gelu  # exact GELU
    if name == "relu":
        return jax.nn.relu
    raise ValueError(f"Unsupported activation '{name}'. Choose 'gelu' or 'relu'.")


# ===========================================================================
# Fourier Block
# ===========================================================================


class FourierBlock(nn.Module):
    """
    One Fourier layer of an FNO.

    It applies:
      1. Real‑to‑complex FFT along spatial dimensions,
      2. Truncation to the lowest ``modes`` frequencies,
      3. Linear transformation with complex weights ``R``,
      4. Zero‑padding back to original frequency shape,
      5. Inverse real FFT,
      6. Addition of a local linear skip connection (point‑wise dense),
      7. Activation.

    The layer preserves spatial dimensions and channels.
    """

    modes: int
    hidden_dim: int
    activation_name: str
    ndim: int  # 1 or 2

    def setup(self) -> None:
        """Initialise complex spectral weights and the local dense layer."""
        # Complex weight matrix R = R_real + i * R_imag
        if self.ndim == 1:
            shape = (self.modes, self.hidden_dim, self.hidden_dim)
        else:
            shape = (self.modes, self.modes, self.hidden_dim, self.hidden_dim)

        # Scaling factor following common FNO initialisations
        stddev = 1.0 / np.sqrt(self.hidden_dim * (self.modes ** self.ndim))

        self.R_real = self.param(
            "R_real",
            nn.initializers.normal(stddev),
            shape,
            dtype=jnp.float32,
        )
        self.R_imag = self.param(
            "R_imag",
            nn.initializers.normal(stddev),
            shape,
            dtype=jnp.float32,
        )

        # Local linear skip connection – no bias is standard,
        # corresponds to W^{(l)} in the paper.
        self.W = nn.Dense(features=self.hidden_dim, use_bias=False)

        self.activation_fn = get_activation(self.activation_name)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass for a single Fourier layer.

        Parameters
        ----------
        x : jnp.ndarray
            Input of shape ``(batch, *spatial_dims, hidden_dim)``.

        Returns
        -------
        jnp.ndarray
            Output of the same shape.
        """
        orig_spatial_shape = x.shape[1 : 1 + self.ndim]  # e.g., (Nx,) or (Nx,Ny)
        batch_size = x.shape[0]

        if self.ndim == 1:
            # ----------------------------------------------------------------
            # 1D transform
            # ----------------------------------------------------------------
            uf = jnp.fft.rfft(x, axis=1)                     # (B, Nfreq, C)
            uf = uf[:, : self.modes, :]                       # truncate

            R = self.R_real + 1j * self.R_imag
            out_ft = jnp.einsum("b k i, k i j -> b k j", uf, R)  # spectral mult

            # pad back to original frequency length
            freq_len = x.shape[1] // 2 + 1
            padded = jnp.zeros(
                (batch_size, freq_len, self.hidden_dim),
                dtype=jnp.complex64,
            )
            padded = padded.at[:, : self.modes, :].set(out_ft)

            x_global = jnp.fft.irfft(padded, n=x.shape[1], axis=1)  # back to real

        else:
            # ----------------------------------------------------------------
            # 2D transform
            # ----------------------------------------------------------------
            uf = jnp.fft.rfftn(x, axes=(1, 2))               # (B, Nx, Ny//2+1, C)
            uf = uf[:, : self.modes, : self.modes, :]        # keep low modes

            R = self.R_real + 1j * self.R_imag
            out_ft = jnp.einsum("b x y i, x y i j -> b x y j", uf, R)

            # original frequency shape after rfftn: (Nx, Ny//2+1)
            freq_shape = (x.shape[1], x.shape[2] // 2 + 1)
            padded = jnp.zeros(
                (batch_size,) + freq_shape + (self.hidden_dim,),
                dtype=jnp.complex64,
            )
            padded = padded.at[:, : self.modes, : self.modes, :].set(out_ft)

            x_global = jnp.fft.irfftn(
                padded, s=orig_spatial_shape, axes=(1, 2)
            )

        # Local skip connection
        x_local = self.W(x)                                   # (B, ..., hidden_dim)

        # Combine and activate
        return self.activation_fn(x_global + x_local)


# ===========================================================================
# Full FNO model
# ===========================================================================


class FourierNeuralOperator(nn.Module):
    """
    Resolution‑agnostic Fourier Neural Operator for next‑step prediction.

    Architecture:
      - Lifting layer (``Dense`` with no activation) that maps the input
        channels to ``hidden_dim``.
      - ``num_blocks`` Fourier layers, each performing a spectral‑local
        transformation and activation.
      - Projection layer (``Dense``) that maps back to the output scalar field.
    """

    model_config: ModelConfig
    ndim: int  # spatial dimensionality (1 or 2)

    def setup(self) -> None:
        cfg = self.model_config

        self.lifting = nn.Dense(
            features=cfg.hidden_dim,
            use_bias=cfg.use_bias,
        )

        self.blocks = [
            FourierBlock(
                modes=cfg.modes,
                hidden_dim=cfg.hidden_dim,
                activation_name=cfg.activation,
                ndim=self.ndim,
            )
            for _ in range(cfg.num_blocks)
        ]

        # Output is a scalar field
        self.projection = nn.Dense(
            features=1,
            use_bias=cfg.use_bias,
        )

    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """
        Forward pass through the FNO.

        Parameters
        ----------
        x : jnp.ndarray
            Input discretised function(s) of shape
            ``(batch, *spatial_dims, in_channels)``.
            The spatial dimensions must be even‑gridded (e.g., 256 for 1D,
            100x100 for 2D).
        train : bool, optional
            Flag for (optional) dropout / batch‑norm. Not used in this
            implementation (default ``False``).

        Returns
        -------
        jnp.ndarray
            Predicted solution at the next time step, shape
            ``(batch, *spatial_dims, 1)``.
        """
        del train  # not used, but kept for compatibility with common Flax patterns

        x = self.lifting(x)           # (B, ..., hidden_dim)

        for block in self.blocks:
            x = block(x)              # (B, ..., hidden_dim)

        x = self.projection(x)        # (B, ..., 1)
        return x
