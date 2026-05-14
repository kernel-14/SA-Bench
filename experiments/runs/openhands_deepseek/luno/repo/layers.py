"""Fourier Neural Operator layer primitives."""

from typing import Tuple, Optional
import jax
import jax.numpy as jnp
from flax import nnx


class SpectralConv1d(nnx.Module):
    """1D Fourier spectral convolution layer.

    Applies: F^{-1}(R * F(v)) where R is a complex weight matrix.
    Follows the FNO paper formulation (Li et al., 2021).
    """

    def __init__(self, in_dim: int, out_dim: int, n_modes: int, rngs: nnx.Rngs):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_modes = n_modes
        # Complex weights for the Fourier modes: shape (n_modes, out_dim, in_dim)
        # We store real and imaginary parts separately
        scale = 1.0 / jnp.sqrt(in_dim)
        self.R_real = nnx.Param(
            jax.random.normal(rngs.param(), (n_modes, out_dim, in_dim)) * scale
        )
        self.R_imag = nnx.Param(
            jax.random.normal(rngs.param(), (n_modes, out_dim, in_dim)) * scale
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (batch, N, in_dim) or (N, in_dim)
        needs_squeeze = x.ndim == 2
        if needs_squeeze:
            x = x[None, ...]

        batch_size, n_points, _ = x.shape
        # RFFT along spatial axis
        x_ft = jnp.fft.rfft(x, axis=1)  # (batch, N//2+1, in_dim)

        # Truncate to n_modes
        x_ft_trunc = x_ft[:, :self.n_modes, :]  # (batch, n_modes, in_dim)

        # Complex multiplication: R * F(v)
        R = self.R_real.value + 1j * self.R_imag.value  # (n_modes, out_dim, in_dim)
        R = R[None, ...]  # (1, n_modes, out_dim, in_dim)

        # out_ft[k, i] = sum_j R[k, i, j] * x_ft_trunc[batch, k, j]
        out_ft = jnp.einsum("bki,koi->bko", x_ft_trunc, R)  # (batch, n_modes, out_dim)

        # Pad with zeros to match original size
        out_ft_padded = jnp.zeros(
            (batch_size, n_points // 2 + 1, self.out_dim),
            dtype=jnp.complex64,
        )
        out_ft_padded = out_ft_padded.at[:, :self.n_modes, :].set(out_ft)

        # Inverse RFFT
        out = jnp.fft.irfft(out_ft_padded, n=n_points, axis=1)  # (batch, N, out_dim)

        if needs_squeeze:
            out = out[0]
        return out


class SpectralConv2d(nnx.Module):
    """2D Fourier spectral convolution layer."""

    def __init__(self, in_dim: int, out_dim: int, n_modes: Tuple[int, int], rngs: nnx.Rngs):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_modes1, self.n_modes2 = n_modes
        scale = 1.0 / jnp.sqrt(in_dim)
        self.R_real = nnx.Param(
            jax.random.normal(rngs.param(), (self.n_modes1, self.n_modes2, out_dim, in_dim)) * scale
        )
        self.R_imag = nnx.Param(
            jax.random.normal(rngs.param(), (self.n_modes1, self.n_modes2, out_dim, in_dim)) * scale
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (batch, H, W, in_dim) or (H, W, in_dim)
        needs_squeeze = x.ndim == 3
        if needs_squeeze:
            x = x[None, ...]

        batch_size, h, w, _ = x.shape
        x_ft = jnp.fft.rfft2(x, axes=(1, 2))  # (batch, H, W//2+1, in_dim)

        x_ft_trunc = x_ft[:, :self.n_modes1, :self.n_modes2, :]
        R = self.R_real.value + 1j * self.R_imag.value
        R = R[None, ...]  # (1, n1, n2, out_dim, in_dim)

        out_ft = jnp.einsum("bxyi,xyoi->bxyo", x_ft_trunc, R)

        out_ft_padded = jnp.zeros(
            (batch_size, h, w // 2 + 1, self.out_dim),
            dtype=jnp.complex64,
        )
        out_ft_padded = out_ft_padded.at[:, :self.n_modes1, :self.n_modes2, :].set(out_ft)

        out = jnp.fft.irfft2(out_ft_padded, s=(h, w), axes=(1, 2))

        if needs_squeeze:
            out = out[0]
        return out


class FourierBlock1d(nnx.Module):
    """One Fourier block for 1D FNO.

    v^{(l+1)}(x) = sigma( F^{-1}(R * F(v^{(l)})) + W @ v^{(l)}(x) )
    """

    def __init__(self, dim: int, n_modes: int, rngs: nnx.Rngs):
        self.dim = dim
        self.spectral_conv = SpectralConv1d(dim, dim, n_modes, rngs)
        # Linear mix W in the spatial domain
        scale = 1.0 / jnp.sqrt(dim)
        self.W = nnx.Param(
            jax.random.normal(rngs.param(), (dim, dim)) * scale
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # spectral path
        s = self.spectral_conv(x)  # (..., N, dim)
        # linear path: W @ x
        l = jnp.dot(x, self.W.value.T)  # (..., N, dim)
        # Combine and activate (GELU as typical for FNOs)
        return jax.nn.gelu(s + l)


class FourierBlock2d(nnx.Module):
    """One Fourier block for 2D FNO."""

    def __init__(self, dim: int, n_modes: Tuple[int, int], rngs: nnx.Rngs):
        self.dim = dim
        self.spectral_conv = SpectralConv2d(dim, dim, n_modes, rngs)
        scale = 1.0 / jnp.sqrt(dim)
        self.W = nnx.Param(
            jax.random.normal(rngs.param(), (dim, dim)) * scale
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        s = self.spectral_conv(x)
        l = jnp.dot(x, self.W.value.T)
        return jax.nn.gelu(s + l)


class LiftingLayer(nnx.Module):
    """Lifting layer: p(a(x), w_p) -> R^{d_v'}.

    A simple linear layer from input_dim to hidden_dim.
    """

    def __init__(self, in_dim: int, out_dim: int, rngs: nnx.Rngs):
        self.linear = nnx.Linear(in_dim, out_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.linear(x)


class ProjectionLayer(nnx.Module):
    """Projection layer: q(v^{(L)}(x), w_q) -> R^{d_U'}.

    A simple linear layer from hidden_dim to output_dim.
    """

    def __init__(self, in_dim: int, out_dim: int, rngs: nnx.Rngs):
        self.linear = nnx.Linear(in_dim, out_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.linear(x)
