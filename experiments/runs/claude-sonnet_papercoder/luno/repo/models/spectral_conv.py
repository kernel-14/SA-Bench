## models/spectral_conv.py
"""Spectral convolution layers for Fourier Neural Operators (FNOs).

This module implements the core frequency-domain linear transformation used in
each Fourier layer of the FNO architecture. The spectral convolution applies a
learned complex-valued weight matrix to the truncated Fourier coefficients of
the input, then transforms back to the spatial domain.

Mathematical foundation (paper Section 2.1, Example 2.1):

    v_i^{(l+1)}(x) = σ^(l)(
        Σ_j F^{-1}(R_{kij}^{(l)} · F(v_j^{(l)})_k)_{k=1}^{k_max}(x)
        + W_{ij}^{(l)} v_j^{(l)}(x)
    )

This module handles the spectral part:
    F^{-1}(R_{kij}^{(l)} · F(v_j^{(l)})_k)

The pointwise linear W_{ij}^{(l)} v_j^{(l)}(x) is handled in FourierBlock
(models/fno.py).

Paper references:
  - Section 2.1, Example 2.1: FNO architecture definition
  - Appendix C.1: Last-layer LUNO uses spectral conv weights R^{(L-1)}
  - config.yaml model.modes: 12, model.channels: 18

Design notes:
  - Uses Flax NNX API (not flax.linen) as specified in the design.
  - Weights are stored as nnx.Param with real/imaginary parts split for
    JAX float32 compatibility.
  - Weight names (weights_real, weights_imag) must match what
    uncertainty/ggn.py and uncertainty/luno.py expect for last-layer access.
  - No bias terms (absorbed into the pointwise linear layer in FourierBlock).
  - Channels-last format: [batch, spatial, channels] for 1D,
    [batch, H, W, channels] for 2D.
"""

from __future__ import annotations

import math
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import nnx


class SpectralConv1d(nnx.Module):
    """1D spectral convolution layer for Fourier Neural Operators.

    Applies a learned linear transformation in the frequency domain to a
    1D spatial input. The transformation is restricted to the first ``modes``
    Fourier modes (low-frequency truncation), giving the FNO its
    resolution-agnostic property.

    The complex weights R^{(l)} ∈ C^{k_max × d_v' × d_v'} are stored as two
    real arrays (real and imaginary parts) for JAX float32 compatibility.

    Attributes:
        in_channels: Number of input channels (d_v').
        out_channels: Number of output channels (d_v').
        modes: Number of Fourier modes to retain (k_max). Paper: 12.
        weights_real: Real part of spectral weights, shape
            ``[modes, in_channels, out_channels]``. Stored as ``nnx.Param``.
        weights_imag: Imaginary part of spectral weights, shape
            ``[modes, in_channels, out_channels]``. Stored as ``nnx.Param``.

    Example::

        rngs = nnx.Rngs(params=0)
        conv = SpectralConv1d(in_channels=18, out_channels=18, modes=12, rngs=rngs)
        x = jnp.ones([4, 260, 18])  # [batch, spatial_padded, channels]
        y = conv(x)
        # y.shape == (4, 260, 18)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialise the 1D spectral convolution layer.

        Weights are initialised with uniform distribution in
        ``[-1/sqrt(in_channels), 1/sqrt(in_channels)]``, following
        Xavier-style initialisation scaled for the input channel count.
        Real and imaginary parts are initialised independently using
        separate PRNG subkeys.

        Args:
            in_channels: Number of input channels. From ``config.model.channels``
                (default 18) for intermediate blocks; may differ for lifting/
                projection layers.
            out_channels: Number of output channels. Typically equals
                ``in_channels`` for intermediate Fourier blocks.
            modes: Number of Fourier modes to retain. From
                ``config.model.modes`` (default 12).
            rngs: Flax NNX random number generator state. Must provide a
                ``params`` key for weight initialisation.

        Raises:
            ValueError: If ``in_channels <= 0``, ``out_channels <= 0``, or
                ``modes <= 0``.
        """
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")
        if modes <= 0:
            raise ValueError(f"modes must be positive, got {modes}")

        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.modes: int = modes

        # Xavier-style initialisation limit
        limit: float = 1.0 / math.sqrt(in_channels)

        # Weight shape: [modes, in_channels, out_channels]
        weight_shape: Tuple[int, int, int] = (modes, in_channels, out_channels)

        # Initialise real and imaginary parts with separate PRNG keys
        key_real: jax.Array = rngs.params()
        key_imag: jax.Array = rngs.params()

        self.weights_real: nnx.Param = nnx.Param(
            jax.random.uniform(
                key_real,
                shape=weight_shape,
                minval=-limit,
                maxval=limit,
                dtype=jnp.float32,
            )
        )
        self.weights_imag: nnx.Param = nnx.Param(
            jax.random.uniform(
                key_imag,
                shape=weight_shape,
                minval=-limit,
                maxval=limit,
                dtype=jnp.float32,
            )
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the 1D spectral convolution.

        Computes the frequency-domain linear transformation:
        1. Apply real FFT along the spatial axis.
        2. Truncate to the first ``modes`` frequency components.
        3. Multiply by the complex weight matrix (einsum over channels).
        4. Pad back to the full frequency size with zeros.
        5. Apply inverse real FFT to recover the spatial representation.

        Args:
            x: Input tensor, shape ``[batch, spatial, in_channels]``.
                The spatial dimension can be any positive integer; the
                output will have the same spatial dimension.

        Returns:
            Output tensor, shape ``[batch, spatial, out_channels]``.
            The spatial dimension is preserved exactly (via the ``n``
            argument to ``irfft``).

        Notes:
            - The FFT is applied along ``axis=1`` (spatial axis).
            - Only the first ``modes`` frequency components are used;
              higher frequencies are zeroed out (low-pass filter).
            - The ``n=spatial`` argument to ``irfft`` ensures the output
              length matches the input length exactly, even for odd-length
              inputs.
        """
        batch: int = x.shape[0]
        spatial: int = x.shape[1]

        # ------------------------------------------------------------------
        # Step 1: Real FFT along spatial axis
        # x_ft shape: [batch, spatial//2 + 1, in_channels]
        # ------------------------------------------------------------------
        x_ft: jnp.ndarray = jnp.fft.rfft(x, axis=1)  # [batch, freq, in_channels]

        # ------------------------------------------------------------------
        # Step 2: Truncate to first `modes` frequency components
        # x_ft_trunc shape: [batch, modes, in_channels]
        # ------------------------------------------------------------------
        x_ft_trunc: jnp.ndarray = x_ft[:, : self.modes, :]  # [batch, modes, in_channels]

        # ------------------------------------------------------------------
        # Step 3: Complex multiplication with learned weights
        # Construct complex weight matrix from real and imaginary parts
        # weights_complex shape: [modes, in_channels, out_channels]
        # ------------------------------------------------------------------
        weights_complex: jnp.ndarray = (
            self.weights_real.value + 1j * self.weights_imag.value
        ).astype(jnp.complex64)  # [modes, in_channels, out_channels]

        # Apply complex einsum: 'bxi,xio->bxo'
        # b=batch, x=modes (frequency), i=in_channels, o=out_channels
        out_ft_trunc: jnp.ndarray = self._compl_mul1d(
            x_ft_trunc, weights_complex
        )  # [batch, modes, out_channels]

        # ------------------------------------------------------------------
        # Step 4: Pad back to full frequency size with zeros
        # Full frequency size for rfft of length `spatial`: spatial//2 + 1
        # ------------------------------------------------------------------
        freq_size: int = spatial // 2 + 1
        n_pad: int = freq_size - self.modes

        if n_pad > 0:
            # Append zeros for the high-frequency components
            zeros_pad: jnp.ndarray = jnp.zeros(
                (batch, n_pad, self.out_channels),
                dtype=jnp.complex64,
            )  # [batch, n_pad, out_channels]
            out_ft: jnp.ndarray = jnp.concatenate(
                [out_ft_trunc, zeros_pad], axis=1
            )  # [batch, freq_size, out_channels]
        else:
            # modes >= freq_size: no padding needed (unusual but handled)
            out_ft = out_ft_trunc[:, :freq_size, :]

        # ------------------------------------------------------------------
        # Step 5: Inverse real FFT to recover spatial representation
        # n=spatial ensures output length matches input length exactly
        # ------------------------------------------------------------------
        out: jnp.ndarray = jnp.fft.irfft(
            out_ft, n=spatial, axis=1
        )  # [batch, spatial, out_channels]

        return out

    @staticmethod
    def _compl_mul1d(
        a: jnp.ndarray,
        b: jnp.ndarray,
    ) -> jnp.ndarray:
        """Complex matrix multiplication for 1D spectral convolution.

        Computes the batched complex einsum ``'bxi,xio->bxo'`` where:
        - ``b``: batch dimension
        - ``x``: frequency mode index (k in the paper)
        - ``i``: input channel index
        - ``o``: output channel index

        JAX handles complex arithmetic natively in ``jnp.einsum``, so no
        manual real/imaginary decomposition is needed.

        Args:
            a: Complex input array, shape ``[batch, modes, in_channels]``.
                Typically the truncated FFT of the input.
            b: Complex weight array, shape ``[modes, in_channels, out_channels]``.
                The learned spectral filter R^{(l)}.

        Returns:
            Complex output array, shape ``[batch, modes, out_channels]``.
            Represents the filtered frequency coefficients.

        Example::

            a = jnp.ones((4, 12, 18), dtype=jnp.complex64)
            b = jnp.ones((12, 18, 18), dtype=jnp.complex64)
            c = SpectralConv1d._compl_mul1d(a, b)
            # c.shape == (4, 12, 18)
        """
        return jnp.einsum("bxi,xio->bxo", a, b)


class SpectralConv2d(nnx.Module):
    """2D spectral convolution layer for Fourier Neural Operators.

    Extends the 1D spectral convolution to two spatial dimensions. Uses
    ``rfft2``/``irfft2`` for efficiency. The 2D truncation retains both
    the low positive-frequency corner (top-left) and the low
    negative-frequency corner (bottom-left) of the frequency tensor to
    capture both positive and negative frequency contributions.

    The complex weights have shape
    ``[modes1, modes2, in_channels, out_channels]``, stored as two real
    arrays (real and imaginary parts).

    Attributes:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        modes1: Number of Fourier modes along the first spatial axis (H).
        modes2: Number of Fourier modes along the second spatial axis (W).
        weights_real: Real part of spectral weights, shape
            ``[modes1, modes2, in_channels, out_channels]``.
            Stored as ``nnx.Param``.
        weights_imag: Imaginary part of spectral weights, shape
            ``[modes1, modes2, in_channels, out_channels]``.
            Stored as ``nnx.Param``.

    Example::

        rngs = nnx.Rngs(params=0)
        conv = SpectralConv2d(
            in_channels=18, out_channels=18, modes1=12, modes2=12, rngs=rngs
        )
        x = jnp.ones([4, 104, 104, 18])  # [batch, H_padded, W_padded, channels]
        y = conv(x)
        # y.shape == (4, 104, 104, 18)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialise the 2D spectral convolution layer.

        Weights are initialised with uniform distribution in
        ``[-1/sqrt(in_channels), 1/sqrt(in_channels)]``. Real and imaginary
        parts are initialised independently.

        Args:
            in_channels: Number of input channels. From
                ``config.model.channels`` (default 18).
            out_channels: Number of output channels. Typically equals
                ``in_channels`` for intermediate Fourier blocks.
            modes1: Number of Fourier modes along the first spatial axis (H).
                From ``config.model.modes`` (default 12).
            modes2: Number of Fourier modes along the second spatial axis (W).
                From ``config.model.modes`` (default 12).
            rngs: Flax NNX random number generator state.

        Raises:
            ValueError: If any channel count or mode count is non-positive.
        """
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")
        if modes1 <= 0:
            raise ValueError(f"modes1 must be positive, got {modes1}")
        if modes2 <= 0:
            raise ValueError(f"modes2 must be positive, got {modes2}")

        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.modes1: int = modes1
        self.modes2: int = modes2

        # Xavier-style initialisation limit
        limit: float = 1.0 / math.sqrt(in_channels)

        # Weight shape: [modes1, modes2, in_channels, out_channels]
        weight_shape: Tuple[int, int, int, int] = (
            modes1, modes2, in_channels, out_channels
        )

        # Initialise real and imaginary parts with separate PRNG keys
        key_real: jax.Array = rngs.params()
        key_imag: jax.Array = rngs.params()

        self.weights_real: nnx.Param = nnx.Param(
            jax.random.uniform(
                key_real,
                shape=weight_shape,
                minval=-limit,
                maxval=limit,
                dtype=jnp.float32,
            )
        )
        self.weights_imag: nnx.Param = nnx.Param(
            jax.random.uniform(
                key_imag,
                shape=weight_shape,
                minval=-limit,
                maxval=limit,
                dtype=jnp.float32,
            )
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the 2D spectral convolution.

        Computes the 2D frequency-domain linear transformation:
        1. Apply 2D real FFT along the two spatial axes.
        2. Extract the low-frequency corners (positive and negative
           frequencies along axis 1; only positive along axis 2 due to rfft).
        3. Multiply each corner by the complex weight matrix.
        4. Reconstruct the full frequency tensor by placing results back
           into the appropriate corners.
        5. Apply 2D inverse real FFT to recover the spatial representation.

        The two-corner approach captures both positive and negative
        frequency contributions along the first spatial axis, which is
        important for non-symmetric signals.

        Args:
            x: Input tensor, shape ``[batch, H, W, in_channels]``.
                The spatial dimensions H and W can be any positive integers.

        Returns:
            Output tensor, shape ``[batch, H, W, out_channels]``.
            Both spatial dimensions are preserved exactly.

        Notes:
            - FFT is applied along ``axes=(1, 2)`` (H and W axes).
            - For rfft2 of shape ``[H, W]``, the output has shape
              ``[H, W//2 + 1]`` in the frequency domain.
            - The top-left corner ``[:modes1, :modes2]`` captures low
              positive frequencies in both dimensions.
            - The bottom-left corner ``[-modes1:, :modes2]`` captures low
              negative frequencies in H and low positive in W.
            - ``s=(H, W)`` in ``irfft2`` ensures exact spatial dimensions.
        """
        batch: int = x.shape[0]
        h: int = x.shape[1]
        w: int = x.shape[2]

        # ------------------------------------------------------------------
        # Step 1: 2D real FFT along spatial axes (1, 2)
        # x_ft shape: [batch, H, W//2+1, in_channels]
        # ------------------------------------------------------------------
        x_ft: jnp.ndarray = jnp.fft.rfft2(x, axes=(1, 2))
        # x_ft.shape: [batch, H, W//2+1, in_channels]

        # ------------------------------------------------------------------
        # Step 2: Build complex weights
        # weights_complex shape: [modes1, modes2, in_channels, out_channels]
        # ------------------------------------------------------------------
        weights_complex: jnp.ndarray = (
            self.weights_real.value + 1j * self.weights_imag.value
        ).astype(jnp.complex64)

        # ------------------------------------------------------------------
        # Step 3: Apply spectral filter to the two low-frequency corners
        #
        # Corner 1 (top-left): positive low frequencies in both H and W
        #   x_ft[:, :modes1, :modes2, :]
        #
        # Corner 2 (bottom-left): negative low frequencies in H,
        #   positive low frequencies in W
        #   x_ft[:, -modes1:, :modes2, :]
        # ------------------------------------------------------------------

        # Top-left corner: [batch, modes1, modes2, in_channels]
        x_ft_top: jnp.ndarray = x_ft[:, : self.modes1, : self.modes2, :]
        out_ft_top: jnp.ndarray = self._compl_mul2d(
            x_ft_top, weights_complex
        )  # [batch, modes1, modes2, out_channels]

        # Bottom-left corner: [batch, modes1, modes2, in_channels]
        x_ft_bot: jnp.ndarray = x_ft[:, -self.modes1 :, : self.modes2, :]
        out_ft_bot: jnp.ndarray = self._compl_mul2d(
            x_ft_bot, weights_complex
        )  # [batch, modes1, modes2, out_channels]

        # ------------------------------------------------------------------
        # Step 4: Reconstruct the full frequency tensor
        # Full rfft2 output shape: [batch, H, W//2+1, out_channels]
        # We place the filtered corners back and leave the rest as zeros.
        # ------------------------------------------------------------------
        freq_h: int = h
        freq_w: int = w // 2 + 1

        # Build the output frequency tensor using concatenation
        # to avoid in-place assignment (JAX arrays are immutable).
        #
        # Layout of out_ft along axis 1 (H frequency axis):
        #   [0 : modes1]        → out_ft_top  (positive low freqs)
        #   [modes1 : H-modes1] → zeros       (high freqs, zeroed out)
        #   [H-modes1 : H]      → out_ft_bot  (negative low freqs)
        #
        # Layout along axis 2 (W frequency axis, rfft):
        #   [0 : modes2]        → filtered values
        #   [modes2 : W//2+1]   → zeros       (high freqs, zeroed out)

        n_mid_h: int = freq_h - 2 * self.modes1
        n_pad_w: int = freq_w - self.modes2

        # Pad the W dimension of each corner with zeros
        if n_pad_w > 0:
            zeros_w_top: jnp.ndarray = jnp.zeros(
                (batch, self.modes1, n_pad_w, self.out_channels),
                dtype=jnp.complex64,
            )
            out_ft_top_padded: jnp.ndarray = jnp.concatenate(
                [out_ft_top, zeros_w_top], axis=2
            )  # [batch, modes1, freq_w, out_channels]

            zeros_w_bot: jnp.ndarray = jnp.zeros(
                (batch, self.modes1, n_pad_w, self.out_channels),
                dtype=jnp.complex64,
            )
            out_ft_bot_padded: jnp.ndarray = jnp.concatenate(
                [out_ft_bot, zeros_w_bot], axis=2
            )  # [batch, modes1, freq_w, out_channels]
        else:
            # modes2 >= freq_w: truncate (unusual but handled)
            out_ft_top_padded = out_ft_top[:, :, :freq_w, :]
            out_ft_bot_padded = out_ft_bot[:, :, :freq_w, :]

        # Build the H dimension: [top | zeros_mid | bottom]
        if n_mid_h > 0:
            zeros_mid_h: jnp.ndarray = jnp.zeros(
                (batch, n_mid_h, freq_w, self.out_channels),
                dtype=jnp.complex64,
            )
            out_ft: jnp.ndarray = jnp.concatenate(
                [out_ft_top_padded, zeros_mid_h, out_ft_bot_padded],
                axis=1,
            )  # [batch, freq_h, freq_w, out_channels]
        elif n_mid_h == 0:
            # Exactly 2 * modes1 == H: no middle section
            out_ft = jnp.concatenate(
                [out_ft_top_padded, out_ft_bot_padded],
                axis=1,
            )  # [batch, freq_h, freq_w, out_channels]
        else:
            # 2 * modes1 > H: overlap; use only top half
            # This is an edge case that should not occur with valid configs
            half_h: int = freq_h // 2
            out_ft = jnp.concatenate(
                [
                    out_ft_top_padded[:, :half_h, :, :],
                    out_ft_bot_padded[:, -(freq_h - half_h) :, :, :],
                ],
                axis=1,
            )

        # ------------------------------------------------------------------
        # Step 5: 2D inverse real FFT
        # s=(h, w) ensures exact spatial dimensions
        # ------------------------------------------------------------------
        out: jnp.ndarray = jnp.fft.irfft2(
            out_ft, s=(h, w), axes=(1, 2)
        )  # [batch, H, W, out_channels]

        return out

    @staticmethod
    def _compl_mul2d(
        a: jnp.ndarray,
        b: jnp.ndarray,
    ) -> jnp.ndarray:
        """Complex matrix multiplication for 2D spectral convolution.

        Computes the batched complex einsum ``'bxyi,xyio->bxyo'`` where:
        - ``b``: batch dimension
        - ``x``: first frequency mode index (k1 in the paper)
        - ``y``: second frequency mode index (k2 in the paper)
        - ``i``: input channel index
        - ``o``: output channel index

        Args:
            a: Complex input array, shape
                ``[batch, modes1, modes2, in_channels]``.
                Typically a corner of the truncated 2D FFT of the input.
            b: Complex weight array, shape
                ``[modes1, modes2, in_channels, out_channels]``.
                The learned 2D spectral filter.

        Returns:
            Complex output array, shape
            ``[batch, modes1, modes2, out_channels]``.
            Represents the filtered 2D frequency coefficients.

        Example::

            a = jnp.ones((4, 12, 12, 18), dtype=jnp.complex64)
            b = jnp.ones((12, 12, 18, 18), dtype=jnp.complex64)
            c = SpectralConv2d._compl_mul2d(a, b)
            # c.shape == (4, 12, 12, 18)
        """
        return jnp.einsum("bxyi,xyio->bxyo", a, b)
