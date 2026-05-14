"""
Fourier Neural Operator (FNO) implementation in JAX/Flax NNX.

Based on: Li et al. (2021) "Fourier Neural Operator for Parametric Partial Differential Equations"
Architecture follows the paper's description and the experimental setup in Appendix D.2:
  - 12 modes (per spatial dimension)
  - 18 hidden dimensions
  - 4 Fourier blocks
"""

import jax
import jax.numpy as jnp
import flax.nnx as nnx
from typing import Callable, Optional


class SpectralConv1d(nnx.Module):
    """1D Fourier spectral convolution layer.
    
    Applies a linear transformation in the frequency domain using the
    real fast Fourier transform (RFFT).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes

        # Complex weights for spectral convolution: shape (n_modes, out_channels, in_channels)
        scale = 1.0 / (in_channels * out_channels)
        key = rngs.params()
        real_key, imag_key = jax.random.split(key)
        self.weights_real = nnx.Param(
            jax.random.uniform(real_key, (n_modes, out_channels, in_channels)) * 2 * scale - scale
        )
        self.weights_imag = nnx.Param(
            jax.random.uniform(imag_key, (n_modes, out_channels, in_channels)) * 2 * scale - scale
        )

    def complex_mul1d(self, x_ft: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        """Complex multiplication: (batch, n_modes, in_ch) x (n_modes, out_ch, in_ch) -> (batch, n_modes, out_ch)"""
        return jnp.einsum("bki,koi->bko", x_ft, weights)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: (batch, n_x, in_channels)
        Returns:
            out: (batch, n_x, out_channels)
        """
        batch, n_x, _ = x.shape

        # FFT along spatial dimension
        x_ft = jnp.fft.rfft(x, axis=1)  # (batch, n_x//2+1, in_channels)

        # Truncate to n_modes
        x_ft_trunc = x_ft[:, :self.n_modes, :]  # (batch, n_modes, in_channels)

        # Complex weights
        weights = self.weights_real.value + 1j * self.weights_imag.value  # (n_modes, out_ch, in_ch)

        # Multiply in frequency domain
        out_ft_trunc = self.complex_mul1d(x_ft_trunc, weights)  # (batch, n_modes, out_channels)

        # Pad back to full frequency size
        out_ft = jnp.zeros((batch, n_x // 2 + 1, self.out_channels), dtype=jnp.complex64)
        out_ft = out_ft.at[:, :self.n_modes, :].set(out_ft_trunc)

        # Inverse FFT
        out = jnp.fft.irfft(out_ft, n=n_x, axis=1)  # (batch, n_x, out_channels)
        return out


class SpectralConv2d(nnx.Module):
    """2D Fourier spectral convolution layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes_x: int,
        n_modes_y: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes_x = n_modes_x
        self.n_modes_y = n_modes_y

        scale = 1.0 / (in_channels * out_channels)
        key = rngs.params()
        k1, k2, k3, k4 = jax.random.split(key, 4)

        # Two sets of weights for the two quadrants of the 2D FFT
        self.weights1_real = nnx.Param(
            jax.random.uniform(k1, (n_modes_x, n_modes_y, out_channels, in_channels)) * 2 * scale - scale
        )
        self.weights1_imag = nnx.Param(
            jax.random.uniform(k2, (n_modes_x, n_modes_y, out_channels, in_channels)) * 2 * scale - scale
        )
        self.weights2_real = nnx.Param(
            jax.random.uniform(k3, (n_modes_x, n_modes_y, out_channels, in_channels)) * 2 * scale - scale
        )
        self.weights2_imag = nnx.Param(
            jax.random.uniform(k4, (n_modes_x, n_modes_y, out_channels, in_channels)) * 2 * scale - scale
        )

    def complex_mul2d(self, x_ft: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        """(batch, mx, my, in_ch) x (mx, my, out_ch, in_ch) -> (batch, mx, my, out_ch)"""
        return jnp.einsum("bkli,kloi->bklo", x_ft, weights)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: (batch, n_x, n_y, in_channels)
        Returns:
            out: (batch, n_x, n_y, out_channels)
        """
        batch, n_x, n_y, _ = x.shape

        # 2D FFT
        x_ft = jnp.fft.rfft2(x, axes=(1, 2))  # (batch, n_x, n_y//2+1, in_channels)

        # Complex weights
        w1 = self.weights1_real.value + 1j * self.weights1_imag.value
        w2 = self.weights2_real.value + 1j * self.weights2_imag.value

        out_ft = jnp.zeros((batch, n_x, n_y // 2 + 1, self.out_channels), dtype=jnp.complex64)

        # Upper-left quadrant
        out_ft = out_ft.at[:, :self.n_modes_x, :self.n_modes_y, :].set(
            self.complex_mul2d(x_ft[:, :self.n_modes_x, :self.n_modes_y, :], w1)
        )
        # Lower-left quadrant
        out_ft = out_ft.at[:, -self.n_modes_x:, :self.n_modes_y, :].set(
            self.complex_mul2d(x_ft[:, -self.n_modes_x:, :self.n_modes_y, :], w2)
        )

        # Inverse 2D FFT
        out = jnp.fft.irfft2(out_ft, s=(n_x, n_y), axes=(1, 2))  # (batch, n_x, n_y, out_channels)
        return out


class FourierBlock1d(nnx.Module):
    """Single Fourier layer for 1D FNO.
    
    Combines spectral convolution with a local linear transform (W).
    """

    def __init__(
        self,
        channels: int,
        n_modes: int,
        activation: Callable = jax.nn.gelu,
        *,
        rngs: nnx.Rngs,
    ):
        self.spectral_conv = SpectralConv1d(channels, channels, n_modes, rngs=rngs)
        self.w = nnx.Linear(channels, channels, use_bias=True, rngs=rngs)
        self.activation = activation

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: (batch, n_x, channels)
        Returns:
            out: (batch, n_x, channels)
        """
        x1 = self.spectral_conv(x)
        x2 = self.w(x)
        return self.activation(x1 + x2)


class FourierBlock2d(nnx.Module):
    """Single Fourier layer for 2D FNO."""

    def __init__(
        self,
        channels: int,
        n_modes_x: int,
        n_modes_y: int,
        activation: Callable = jax.nn.gelu,
        *,
        rngs: nnx.Rngs,
    ):
        self.spectral_conv = SpectralConv2d(channels, channels, n_modes_x, n_modes_y, rngs=rngs)
        self.w = nnx.Linear(channels, channels, use_bias=True, rngs=rngs)
        self.activation = activation

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: (batch, n_x, n_y, channels)
        Returns:
            out: (batch, n_x, n_y, channels)
        """
        x1 = self.spectral_conv(x)
        x2 = self.w(x)
        return self.activation(x1 + x2)


class FNO1d(nnx.Module):
    """1D Fourier Neural Operator.
    
    Architecture from Li et al. (2021) with hyperparameters from Appendix D.2:
    - n_modes=12, hidden_channels=18, n_layers=4
    
    Input: (batch, n_x, in_channels) - discretized input function
    Output: (batch, n_x, out_channels) - discretized output function
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 18,
        n_modes: int = 12,
        n_layers: int = 4,
        activation: Callable = jax.nn.gelu,
        *,
        rngs: nnx.Rngs,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_modes = n_modes
        self.n_layers = n_layers

        # Lifting layer (p): maps input to hidden channels
        self.lifting = nnx.Linear(in_channels, hidden_channels, rngs=rngs)

        # Fourier blocks
        self.fourier_blocks = [
            FourierBlock1d(hidden_channels, n_modes, activation, rngs=rngs)
            for _ in range(n_layers)
        ]

        # Projection layer (q): maps hidden to output channels
        # Two-layer MLP as projection
        self.proj1 = nnx.Linear(hidden_channels, hidden_channels * 2, rngs=rngs)
        self.proj2 = nnx.Linear(hidden_channels * 2, out_channels, rngs=rngs)
        self.activation = activation

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: (batch, n_x, in_channels)
        Returns:
            out: (batch, n_x, out_channels)
        """
        # Lifting
        x = self.lifting(x)

        # Fourier blocks
        for block in self.fourier_blocks:
            x = block(x)

        # Projection
        x = self.activation(self.proj1(x))
        x = self.proj2(x)
        return x

    def get_last_layer_input(self, x: jnp.ndarray) -> jnp.ndarray:
        """Get the input to the last Fourier block (v^{L-1}).
        
        Used for last-layer LUNO.
        """
        x = self.lifting(x)
        for block in self.fourier_blocks[:-1]:
            x = block(x)
        return x


class FNO2d(nnx.Module):
    """2D Fourier Neural Operator.
    
    Architecture from Li et al. (2021) with hyperparameters from Appendix D.2:
    - n_modes=12, hidden_channels=18, n_layers=4
    
    Input: (batch, n_x, n_y, in_channels) - discretized input function
    Output: (batch, n_x, n_y, out_channels) - discretized output function
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 18,
        n_modes_x: int = 12,
        n_modes_y: int = 12,
        n_layers: int = 4,
        activation: Callable = jax.nn.gelu,
        *,
        rngs: nnx.Rngs,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_modes_x = n_modes_x
        self.n_modes_y = n_modes_y
        self.n_layers = n_layers

        # Lifting layer
        self.lifting = nnx.Linear(in_channels, hidden_channels, rngs=rngs)

        # Fourier blocks
        self.fourier_blocks = [
            FourierBlock2d(hidden_channels, n_modes_x, n_modes_y, activation, rngs=rngs)
            for _ in range(n_layers)
        ]

        # Projection layer
        self.proj1 = nnx.Linear(hidden_channels, hidden_channels * 2, rngs=rngs)
        self.proj2 = nnx.Linear(hidden_channels * 2, out_channels, rngs=rngs)
        self.activation = activation

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: (batch, n_x, n_y, in_channels)
        Returns:
            out: (batch, n_x, n_y, out_channels)
        """
        # Lifting
        x = self.lifting(x)

        # Fourier blocks
        for block in self.fourier_blocks:
            x = block(x)

        # Projection
        x = self.activation(self.proj1(x))
        x = self.proj2(x)
        return x

    def get_last_layer_input(self, x: jnp.ndarray) -> jnp.ndarray:
        """Get the input to the last Fourier block (v^{L-1}).
        
        Used for last-layer LUNO.
        """
        x = self.lifting(x)
        for block in self.fourier_blocks[:-1]:
            x = block(x)
        return x
