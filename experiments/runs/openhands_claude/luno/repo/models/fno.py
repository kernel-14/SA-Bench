"""
Fourier Neural Operator (FNO) implementation in JAX/Flax NNX.

Implements the FNO architecture from Li et al. (2021) as described in the LUNO paper:
  v^(l+1)(x) = σ( F^{-1}(R^(l) * F(v^(l)))_k + W^(l) v^(l)(x) )
  F(a, w)(x) = q(v^(L)(x), w_q)

Supports 1D and 2D spatial domains.
"""

from typing import Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


def get_activation(name: str) -> Callable:
    activations = {
        "gelu": jax.nn.gelu,
        "relu": jax.nn.relu,
        "tanh": jnp.tanh,
        "silu": jax.nn.silu,
    }
    return activations[name]


class SpectralConv1d(nnx.Module):
    """
    1D Fourier integral operator: F^{-1}(R * F(v)).

    R ∈ C^{k_max × d_out × d_in} are the learnable spectral weights.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        n_modes: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.d_in = d_in
        self.d_out = d_out
        self.n_modes = n_modes

        scale = 1.0 / (d_in * d_out)
        # R stored as real and imaginary parts: shape (n_modes, d_out, d_in)
        self.weights_real = nnx.Param(
            jax.random.uniform(rngs.params(), (n_modes, d_out, d_in), minval=-scale, maxval=scale)
        )
        self.weights_imag = nnx.Param(
            jax.random.uniform(rngs.params(), (n_modes, d_out, d_in), minval=-scale, maxval=scale)
        )

    def __call__(self, v: jax.Array) -> jax.Array:
        """
        Args:
            v: (batch, n_x, d_in)
        Returns:
            out: (batch, n_x, d_out)
        """
        batch, n_x, _ = v.shape
        R = self.weights_real.value + 1j * self.weights_imag.value  # (n_modes, d_out, d_in)

        v_hat = jnp.fft.rfft(v, axis=1)  # (batch, n_x//2+1, d_in)
        n_freq = v_hat.shape[1]
        k = min(self.n_modes, n_freq)

        # Truncate to k_max modes and apply spectral weights
        # v_hat_trunc: (batch, k, d_in) -> out_hat_trunc: (batch, k, d_out)
        v_hat_trunc = v_hat[:, :k, :]  # (batch, k, d_in)
        # einsum: bki, koi -> bko
        out_hat_trunc = jnp.einsum("bki,koi->bko", v_hat_trunc, R[:k])

        # Pad back to full frequency dimension
        out_hat = jnp.zeros((batch, n_freq, self.d_out), dtype=jnp.complex64)
        out_hat = out_hat.at[:, :k, :].set(out_hat_trunc)

        out = jnp.fft.irfft(out_hat, n=n_x, axis=1)  # (batch, n_x, d_out)
        return out


class SpectralConv2d(nnx.Module):
    """
    2D Fourier integral operator: F^{-1}(R * F(v)).

    R ∈ C^{k_max × k_max × d_out × d_in} are the learnable spectral weights.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        n_modes: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.d_in = d_in
        self.d_out = d_out
        self.n_modes = n_modes

        scale = 1.0 / (d_in * d_out)
        shape = (n_modes, n_modes, d_out, d_in)
        self.weights_real = nnx.Param(
            jax.random.uniform(rngs.params(), shape, minval=-scale, maxval=scale)
        )
        self.weights_imag = nnx.Param(
            jax.random.uniform(rngs.params(), shape, minval=-scale, maxval=scale)
        )

    def __call__(self, v: jax.Array) -> jax.Array:
        """
        Args:
            v: (batch, n_x, n_y, d_in)
        Returns:
            out: (batch, n_x, n_y, d_out)
        """
        batch, n_x, n_y, _ = v.shape
        R = self.weights_real.value + 1j * self.weights_imag.value  # (k, k, d_out, d_in)

        v_hat = jnp.fft.rfft2(v, axes=(1, 2))  # (batch, n_x, n_y//2+1, d_in)
        n_freq_x, n_freq_y = v_hat.shape[1], v_hat.shape[2]
        kx = min(self.n_modes, n_freq_x)
        ky = min(self.n_modes, n_freq_y)

        # Apply spectral weights to low-frequency modes
        # We handle both positive and negative x-frequencies
        out_hat = jnp.zeros((batch, n_freq_x, n_freq_y, self.d_out), dtype=jnp.complex64)

        # Lower-left block: positive x-frequencies, positive y-frequencies
        v_hat_ll = v_hat[:, :kx, :ky, :]
        out_ll = jnp.einsum("bkli,kloi->bklo", v_hat_ll, R[:kx, :ky])
        out_hat = out_hat.at[:, :kx, :ky, :].set(out_ll)

        # Upper-left block: negative x-frequencies, positive y-frequencies
        v_hat_ul = v_hat[:, -kx:, :ky, :]
        out_ul = jnp.einsum("bkli,kloi->bklo", v_hat_ul, R[:kx, :ky])
        out_hat = out_hat.at[:, -kx:, :ky, :].set(out_ul)

        out = jnp.fft.irfft2(out_hat, s=(n_x, n_y), axes=(1, 2))  # (batch, n_x, n_y, d_out)
        return out


class FourierLayer1d(nnx.Module):
    """
    Single Fourier layer for 1D FNO:
      v^(l+1)(x) = σ( F^{-1}(R^(l) * F(v^(l)))_k + W^(l) v^(l)(x) )
    """

    def __init__(
        self,
        d_v: int,
        n_modes: int,
        activation: Callable,
        *,
        rngs: nnx.Rngs,
    ):
        self.spectral_conv = SpectralConv1d(d_v, d_v, n_modes, rngs=rngs)
        self.local_linear = nnx.Linear(d_v, d_v, use_bias=True, rngs=rngs)
        self.activation = activation

    def __call__(self, v: jax.Array) -> jax.Array:
        """
        Args:
            v: (batch, n_x, d_v)
        Returns:
            v_next: (batch, n_x, d_v)
        """
        spectral_out = self.spectral_conv(v)
        local_out = self.local_linear(v)
        return self.activation(spectral_out + local_out)


class FourierLayer2d(nnx.Module):
    """
    Single Fourier layer for 2D FNO.
    """

    def __init__(
        self,
        d_v: int,
        n_modes: int,
        activation: Callable,
        *,
        rngs: nnx.Rngs,
    ):
        self.spectral_conv = SpectralConv2d(d_v, d_v, n_modes, rngs=rngs)
        self.local_linear = nnx.Linear(d_v, d_v, use_bias=True, rngs=rngs)
        self.activation = activation

    def __call__(self, v: jax.Array) -> jax.Array:
        """
        Args:
            v: (batch, n_x, n_y, d_v)
        Returns:
            v_next: (batch, n_x, n_y, d_v)
        """
        spectral_out = self.spectral_conv(v)
        local_out = self.local_linear(v)
        return self.activation(spectral_out + local_out)


class FNO1d(nnx.Module):
    """
    1D Fourier Neural Operator.

    Architecture:
      v^(1)(x) = p(a(x), w_p)                    [lifting]
      v^(l+1)(x) = σ(F^{-1}(R^(l)*F(v^(l))) + W^(l) v^(l)(x))  [Fourier layers]
      F(a,w)(x) = q(v^(L)(x), w_q)               [projection]

    Hyperparameters from paper: n_modes=12, d_v=18, n_layers=4, padding=2.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        n_modes: int = 12,
        d_v: int = 18,
        n_layers: int = 4,
        padding: int = 2,
        projection_hidden: int = 128,
        activation: str = "gelu",
        *,
        rngs: nnx.Rngs,
    ):
        self.d_in = d_in
        self.d_out = d_out
        self.n_modes = n_modes
        self.d_v = d_v
        self.n_layers = n_layers
        self.padding = padding
        act = get_activation(activation)

        # Lifting layer p: R^{d_in} -> R^{d_v}
        self.lifting = nnx.Linear(d_in, d_v, use_bias=True, rngs=rngs)

        # Fourier layers
        self.fourier_layers = [
            FourierLayer1d(d_v, n_modes, act, rngs=rngs)
            for _ in range(n_layers)
        ]

        # Projection layer q: R^{d_v} -> R^{d_out}
        self.proj1 = nnx.Linear(d_v, projection_hidden, use_bias=True, rngs=rngs)
        self.proj2 = nnx.Linear(projection_hidden, d_out, use_bias=True, rngs=rngs)
        self.proj_act = act

    def __call__(self, a: jax.Array) -> jax.Array:
        """
        Args:
            a: (batch, n_x, d_in) - discretized input function
        Returns:
            u: (batch, n_x, d_out) - predicted output function
        """
        # Pad spatial dimension to reduce boundary artifacts
        if self.padding > 0:
            a = jnp.pad(a, ((0, 0), (0, self.padding), (0, 0)), mode="constant")

        # Lifting
        v = self.lifting(a)  # (batch, n_x + padding, d_v)

        # Fourier layers
        for layer in self.fourier_layers:
            v = layer(v)

        # Remove padding
        if self.padding > 0:
            v = v[:, :-self.padding, :]

        # Projection
        out = self.proj_act(self.proj1(v))
        out = self.proj2(out)
        return out

    def get_last_layer_input(self, a: jax.Array) -> jax.Array:
        """
        Returns v^(L-1)(x) - the input to the last Fourier block.
        Used for last-layer LUNO.

        Args:
            a: (batch, n_x, d_in)
        Returns:
            v_Lm1: (batch, n_x, d_v) - hidden state before last Fourier layer
        """
        if self.padding > 0:
            a = jnp.pad(a, ((0, 0), (0, self.padding), (0, 0)), mode="constant")

        v = self.lifting(a)

        # Apply all but the last Fourier layer
        for layer in self.fourier_layers[:-1]:
            v = layer(v)

        if self.padding > 0:
            v = v[:, :-self.padding, :]

        return v

    def forward_from_last_layer_input(self, v_Lm1: jax.Array) -> jax.Array:
        """
        Forward pass from v^(L-1) through the last Fourier block and projection.

        Args:
            v_Lm1: (batch, n_x, d_v)
        Returns:
            u: (batch, n_x, d_out)
        """
        if self.padding > 0:
            v_Lm1 = jnp.pad(v_Lm1, ((0, 0), (0, self.padding), (0, 0)), mode="constant")

        v = self.fourier_layers[-1](v_Lm1)

        if self.padding > 0:
            v = v[:, :-self.padding, :]

        out = self.proj_act(self.proj1(v))
        out = self.proj2(out)
        return out


class FNO2d(nnx.Module):
    """
    2D Fourier Neural Operator.

    Same architecture as FNO1d but for 2D spatial domains.
    Hyperparameters from paper: n_modes=12, d_v=18, n_layers=4, padding=2.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        n_modes: int = 12,
        d_v: int = 18,
        n_layers: int = 4,
        padding: int = 2,
        projection_hidden: int = 128,
        activation: str = "gelu",
        *,
        rngs: nnx.Rngs,
    ):
        self.d_in = d_in
        self.d_out = d_out
        self.n_modes = n_modes
        self.d_v = d_v
        self.n_layers = n_layers
        self.padding = padding
        act = get_activation(activation)

        self.lifting = nnx.Linear(d_in, d_v, use_bias=True, rngs=rngs)

        self.fourier_layers = [
            FourierLayer2d(d_v, n_modes, act, rngs=rngs)
            for _ in range(n_layers)
        ]

        self.proj1 = nnx.Linear(d_v, projection_hidden, use_bias=True, rngs=rngs)
        self.proj2 = nnx.Linear(projection_hidden, d_out, use_bias=True, rngs=rngs)
        self.proj_act = act

    def __call__(self, a: jax.Array) -> jax.Array:
        """
        Args:
            a: (batch, n_x, n_y, d_in)
        Returns:
            u: (batch, n_x, n_y, d_out)
        """
        if self.padding > 0:
            a = jnp.pad(a, ((0, 0), (0, self.padding), (0, self.padding), (0, 0)), mode="constant")

        v = self.lifting(a)

        for layer in self.fourier_layers:
            v = layer(v)

        if self.padding > 0:
            v = v[:, :-self.padding, :-self.padding, :]

        out = self.proj_act(self.proj1(v))
        out = self.proj2(out)
        return out

    def get_last_layer_input(self, a: jax.Array) -> jax.Array:
        """
        Returns v^(L-1)(x) for last-layer LUNO.

        Args:
            a: (batch, n_x, n_y, d_in)
        Returns:
            v_Lm1: (batch, n_x, n_y, d_v)
        """
        if self.padding > 0:
            a = jnp.pad(a, ((0, 0), (0, self.padding), (0, self.padding), (0, 0)), mode="constant")

        v = self.lifting(a)

        for layer in self.fourier_layers[:-1]:
            v = layer(v)

        if self.padding > 0:
            v = v[:, :-self.padding, :-self.padding, :]

        return v

    def forward_from_last_layer_input(self, v_Lm1: jax.Array) -> jax.Array:
        """
        Forward from v^(L-1) through last Fourier block and projection.

        Args:
            v_Lm1: (batch, n_x, n_y, d_v)
        Returns:
            u: (batch, n_x, n_y, d_out)
        """
        if self.padding > 0:
            v_Lm1 = jnp.pad(v_Lm1, ((0, 0), (0, self.padding), (0, self.padding), (0, 0)), mode="constant")

        v = self.fourier_layers[-1](v_Lm1)

        if self.padding > 0:
            v = v[:, :-self.padding, :-self.padding, :]

        out = self.proj_act(self.proj1(v))
        out = self.proj2(out)
        return out
