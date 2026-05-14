## models/spectral_conv.py
"""Spectral convolution layers for Fourier Neural Operators (FNO).

Implements the core Fourier integral operator layer from Li et al. (2021):
    K(φ)v_t(x) = F⁻¹(R_φ · F(v_t))(x)

where R_φ is a learnable complex weight tensor truncated to the first k Fourier
modes. Two variants are provided:

  - SpectralConv1d: For ODEs and 1D PDEs, operating on [B, C, L] tensors.
  - SpectralConv2d: For 2D PDEs (PDE3) and 2D (x,t) inputs (PDE1, PDE2, PDE4),
                    operating on [B, C, H, W] tensors.

Both classes are purely linear operators — no activation function is applied
here. Activations are applied in FNOBlock1d/FNOBlock2d (defined in sc_fno.py)
after summing the spectral conv output with the pointwise linear skip connection.

Hyperparameters from Table C.7 of the SC-FNO paper:
  - modes = 8 for all cases (both 1D and 2D)
  - width = 20 (number of channels, set in FNO, not here)
  - n_fourier_layers = 4

References:
    - Li et al. (2021): "Fourier Neural Operator for Parametric Partial
      Differential Equations" (https://arxiv.org/abs/2010.08895)
    - Paper Table C.7: Hyperparameters for FNOs
    - config.yaml: model.width=20, modes_t=8, modes_x=8, modes_y=8
"""

from typing import Tuple

import torch
import torch.nn as nn


class SpectralConv1d(nn.Module):
    """Spectral convolution layer for 1D signals (ODEs and 1D PDEs).

    Implements the Fourier integral operator for 1D inputs:
        output(x) = F⁻¹(R_φ · F(input)[:modes])(x)

    where R_φ is a learnable complex weight tensor of shape
    [in_channels, out_channels, modes]. Only the first `modes` Fourier
    frequencies are retained (low-pass truncation), which dramatically
    reduces the number of learnable parameters while capturing the dominant
    spatial/temporal patterns.

    Used by FNOBlock1d in sc_fno.py for:
      - ODE1, ODE2: 1D time sequences, modes=8 (Table C.7)
      - PDE1, PDE2, PDE4: 2D (x,t) inputs treated as 1D along each dimension
        when using the 1D FNO variant

    Attributes:
        in_channels: Number of input feature channels.
        out_channels: Number of output feature channels.
        modes: Number of Fourier modes to retain (low-pass cutoff).
               From Table C.7: modes=8 for all cases.
        weights: Learnable complex weight tensor of shape
                 [in_channels, out_channels, modes], dtype=torch.cfloat.
                 Initialized with scale 1/(in_channels * out_channels).

    Example:
        >>> layer = SpectralConv1d(in_channels=20, out_channels=20, modes=8)
        >>> x = torch.randn(4, 20, 100)  # [B, C, L]
        >>> out = layer(x)
        >>> out.shape  # [4, 20, 100]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int = 8,
    ) -> None:
        """Initializes SpectralConv1d with learnable complex Fourier weights.

        Args:
            in_channels: Number of input feature channels. Typically equal to
                         the FNO width (20 from config.yaml model.width).
            out_channels: Number of output feature channels. Typically equal
                          to the FNO width (20 from config.yaml model.width).
            modes: Number of Fourier modes to retain in the truncated
                   frequency representation. From Table C.7: modes=8 for all
                   ODE and 1D PDE cases. Must satisfy modes <= L//2 + 1 where
                   L is the sequence length at runtime.
        """
        super().__init__()

        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.modes: int = modes

        # ------------------------------------------------------------------
        # Learnable complex weight tensor R_φ in the Fourier domain.
        # Shape: [in_channels, out_channels, modes]
        # dtype: torch.cfloat (complex64) — matches float32 precision of the
        #        rest of the model.
        #
        # Initialization: scale by 1/(in_channels * out_channels) to keep
        # activations stable. Using torch.rand (uniform [0,1]) rather than
        # torch.randn (Gaussian) follows the original FNO implementation
        # convention from Li et al. (2021).
        # ------------------------------------------------------------------
        scale: float = 1.0 / (in_channels * out_channels)
        self.weights: nn.Parameter = nn.Parameter(
            scale * torch.rand(
                in_channels,
                out_channels,
                modes,
                dtype=torch.cfloat,
            )
        )

    def _compl_mul1d(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """Batched complex multiplication over the in_channels dimension.

        Computes the contraction:
            output[batch, out_ch, mode] = Σ_in_ch a[batch, in_ch, mode] * b[in_ch, out_ch, mode]

        This is a batched matrix-vector product for each Fourier mode, where
        the matrix is b[:, :, mode] (shape [in_channels, out_channels]) and
        the vector is a[batch, :, mode] (shape [in_channels]).

        Args:
            a: Input frequency tensor, shape [B, in_channels, modes].
               Complex (cfloat). Represents the truncated Fourier transform
               of the input signal.
            b: Learnable weight tensor, shape [in_channels, out_channels, modes].
               Complex (cfloat). This is self.weights.

        Returns:
            Output frequency tensor, shape [B, out_channels, modes].
            Complex (cfloat). Represents the filtered frequency content.

        Example:
            >>> a = torch.randn(4, 20, 8, dtype=torch.cfloat)
            >>> b = torch.randn(20, 20, 8, dtype=torch.cfloat)
            >>> out = layer._compl_mul1d(a, b)
            >>> out.shape  # [4, 20, 8]
        """
        # einsum notation: "bim,iom->bom"
        #   b = batch dimension
        #   i = in_channels (contracted)
        #   o = out_channels
        #   m = modes (Fourier mode index)
        return torch.einsum("bim,iom->bom", a, b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the 1D spectral convolution to the input tensor.

        Pipeline:
          1. rfft along the last dimension (sequence length L)
          2. Truncate to first `modes` frequencies
          3. Apply learnable complex multiplication (R_φ · F(v_t))
          4. Zero-pad back to L//2 + 1 frequencies
          5. irfft to recover the spatial/temporal domain signal

        The `n=L` argument to irfft guarantees the output length matches the
        input length exactly, even for odd L values.

        Args:
            x: Input tensor of shape [B, in_channels, L] where:
               - B is the batch size
               - in_channels is the number of feature channels (FNO width)
               - L is the sequence length (time steps or spatial points)
               Must be real-valued (float32). The rfft internally converts
               to complex for the frequency-domain operations.

        Returns:
            Output tensor of shape [B, out_channels, L]. Real-valued (float32).
            Same spatial/temporal resolution as the input.

        Raises:
            RuntimeError: If modes > L//2 + 1 (more modes requested than
                          available frequencies for the given sequence length).

        Example:
            >>> layer = SpectralConv1d(20, 20, 8)
            >>> x = torch.randn(4, 20, 100)  # [B=4, C=20, L=100]
            >>> out = layer(x)
            >>> out.shape  # [4, 20, 100]
        """
        # Sequence length — needed for irfft's n argument.
        L: int = x.shape[-1]

        # ------------------------------------------------------------------
        # Step 1: Apply rfft along the last dimension.
        # rfft exploits conjugate symmetry of real signals, returning only
        # the non-negative frequencies: shape [B, in_channels, L//2 + 1].
        # ------------------------------------------------------------------
        x_ft: torch.Tensor = torch.fft.rfft(x, dim=-1)
        # x_ft shape: [B, in_channels, L//2 + 1], complex cfloat

        # ------------------------------------------------------------------
        # Step 2: Truncate to first `modes` frequencies and apply the
        # learnable complex multiplication R_φ · F(v_t).
        # x_ft[:, :, :modes] has shape [B, in_channels, modes].
        # ------------------------------------------------------------------
        # Validate that modes does not exceed available frequencies.
        n_freqs: int = L // 2 + 1
        if self.modes > n_freqs:
            raise RuntimeError(
                f"SpectralConv1d: modes={self.modes} exceeds the number of "
                f"available rfft frequencies={n_freqs} for sequence length L={L}. "
                f"Reduce modes or increase the sequence length."
            )

        # Apply complex multiplication: [B, in_channels, modes] × [in_channels, out_channels, modes]
        # → [B, out_channels, modes]
        x_ft_truncated: torch.Tensor = x_ft[:, :, :self.modes]
        out_modes: torch.Tensor = self._compl_mul1d(x_ft_truncated, self.weights)
        # out_modes shape: [B, out_channels, modes], complex cfloat

        # ------------------------------------------------------------------
        # Step 3: Zero-pad back to the full rfft frequency dimension.
        # Create a zero tensor of shape [B, out_channels, L//2 + 1] and
        # fill the first `modes` entries with the computed output modes.
        # The remaining frequencies are zero (high-frequency truncation).
        # ------------------------------------------------------------------
        out_ft: torch.Tensor = torch.zeros(
            x.shape[0],       # B
            self.out_channels,
            n_freqs,          # L//2 + 1
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, :self.modes] = out_modes

        # ------------------------------------------------------------------
        # Step 4: Apply irfft to recover the real-valued spatial/temporal signal.
        # n=L ensures the output length matches the input length exactly.
        # Without n=L, irfft may produce L-1 or L+1 outputs for odd L.
        # ------------------------------------------------------------------
        out: torch.Tensor = torch.fft.irfft(out_ft, n=L, dim=-1)
        # out shape: [B, out_channels, L], real float32

        return out


class SpectralConv2d(nn.Module):
    """Spectral convolution layer for 2D signals (2D PDEs and (x,t) inputs).

    Implements the Fourier integral operator for 2D inputs:
        output(x,y) = F⁻¹(R_φ · F(input)[:modes1, :modes2])(x,y)

    Uses two weight tensors (weights1, weights2) to capture Fourier modes in
    both the top-left and bottom-left corners of the 2D frequency domain.
    This is necessary because rfft2 returns only non-negative frequencies in
    the last dimension but both positive and negative frequencies in the first
    dimension — the two weight tensors capture the full set of low-frequency
    modes.

    Used by FNOBlock2d in sc_fno.py for:
      - PDE1, PDE2, PDE4: 2D (x,t) inputs, modes1=modes2=8 (Table C.7)
      - PDE3 (Navier-Stokes): 2D spatial (x,y) inputs, modes1=modes2=8

    Attributes:
        in_channels: Number of input feature channels.
        out_channels: Number of output feature channels.
        modes1: Number of Fourier modes in the first spatial dimension.
                From Table C.7: modes1=8 for all 2D cases.
        modes2: Number of Fourier modes in the second spatial dimension.
                From Table C.7: modes2=8 for all 2D cases.
        weights1: Learnable complex weight tensor for the top-left frequency
                  corner, shape [in_channels, out_channels, modes1, modes2],
                  dtype=torch.cfloat.
        weights2: Learnable complex weight tensor for the bottom-left frequency
                  corner, shape [in_channels, out_channels, modes1, modes2],
                  dtype=torch.cfloat.

    Example:
        >>> layer = SpectralConv2d(in_channels=20, out_channels=20, modes1=8, modes2=8)
        >>> x = torch.randn(4, 20, 64, 64)  # [B, C, H, W]
        >>> out = layer(x)
        >>> out.shape  # [4, 20, 64, 64]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int = 8,
        modes2: int = 8,
    ) -> None:
        """Initializes SpectralConv2d with two learnable complex Fourier weight tensors.

        Args:
            in_channels: Number of input feature channels. Typically equal to
                         the FNO width (20 from config.yaml model.width).
            out_channels: Number of output feature channels. Typically equal
                          to the FNO width (20 from config.yaml model.width).
            modes1: Number of Fourier modes to retain in the first spatial
                    dimension (H). From Table C.7: modes1=8 for all 2D cases.
                    Must satisfy modes1 <= H//2 + 1 at runtime.
            modes2: Number of Fourier modes to retain in the second spatial
                    dimension (W). From Table C.7: modes2=8 for all 2D cases.
                    Must satisfy modes2 <= W//2 + 1 at runtime.
        """
        super().__init__()

        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.modes1: int = modes1
        self.modes2: int = modes2

        # ------------------------------------------------------------------
        # Two learnable complex weight tensors for the 2D spectral conv.
        # Each has shape [in_channels, out_channels, modes1, modes2].
        #
        # weights1: Applied to the top-left corner of the 2D frequency domain
        #           (positive frequencies in both dimensions).
        # weights2: Applied to the bottom-left corner of the 2D frequency domain
        #           (negative frequencies in dim1, positive in dim2).
        #
        # This two-tensor design captures the full set of low-frequency modes
        # in the rfft2 output, which only returns non-negative frequencies in
        # the last dimension but both positive and negative in the first.
        #
        # Initialization: same convention as SpectralConv1d — scale by
        # 1/(in_channels * out_channels), use torch.rand (uniform [0,1]).
        # ------------------------------------------------------------------
        scale: float = 1.0 / (in_channels * out_channels)

        self.weights1: nn.Parameter = nn.Parameter(
            scale * torch.rand(
                in_channels,
                out_channels,
                modes1,
                modes2,
                dtype=torch.cfloat,
            )
        )

        self.weights2: nn.Parameter = nn.Parameter(
            scale * torch.rand(
                in_channels,
                out_channels,
                modes1,
                modes2,
                dtype=torch.cfloat,
            )
        )

    def _compl_mul2d(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """Batched complex multiplication over the in_channels dimension for 2D modes.

        Computes the contraction:
            output[batch, out_ch, x, y] = Σ_in_ch a[batch, in_ch, x, y] * b[in_ch, out_ch, x, y]

        This is a batched matrix-vector product for each (x, y) mode pair,
        where the matrix is b[:, :, x, y] (shape [in_channels, out_channels])
        and the vector is a[batch, :, x, y] (shape [in_channels]).

        Args:
            a: Input frequency tensor, shape [B, in_channels, modes1, modes2].
               Complex (cfloat). Represents a corner of the 2D rfft2 output.
            b: Learnable weight tensor, shape [in_channels, out_channels, modes1, modes2].
               Complex (cfloat). This is either self.weights1 or self.weights2.

        Returns:
            Output frequency tensor, shape [B, out_channels, modes1, modes2].
            Complex (cfloat). Represents the filtered 2D frequency content.

        Example:
            >>> a = torch.randn(4, 20, 8, 8, dtype=torch.cfloat)
            >>> b = torch.randn(20, 20, 8, 8, dtype=torch.cfloat)
            >>> out = layer._compl_mul2d(a, b)
            >>> out.shape  # [4, 20, 8, 8]
        """
        # einsum notation: "bixy,ioxy->boxy"
        #   b = batch dimension
        #   i = in_channels (contracted)
        #   o = out_channels
        #   x = first spatial/mode dimension (modes1)
        #   y = second spatial/mode dimension (modes2)
        return torch.einsum("bixy,ioxy->boxy", a, b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the 2D spectral convolution to the input tensor.

        Pipeline:
          1. rfft2 along the last two dimensions (H, W)
          2. Apply weights1 to the top-left frequency corner [:modes1, :modes2]
          3. Apply weights2 to the bottom-left frequency corner [-modes1:, :modes2]
          4. Zero-pad back to the full rfft2 frequency dimensions
          5. irfft2 to recover the 2D spatial signal

        The `s=(H, W)` argument to irfft2 guarantees the output spatial
        dimensions match the input exactly.

        Args:
            x: Input tensor of shape [B, in_channels, H, W] where:
               - B is the batch size
               - in_channels is the number of feature channels (FNO width)
               - H is the first spatial dimension (e.g., Sx or T)
               - W is the second spatial dimension (e.g., Sy or Sx)
               Must be real-valued (float32).

        Returns:
            Output tensor of shape [B, out_channels, H, W]. Real-valued (float32).
            Same spatial resolution as the input.

        Raises:
            RuntimeError: If modes1 > H//2 + 1 or modes2 > W//2 + 1.

        Example:
            >>> layer = SpectralConv2d(20, 20, 8, 8)
            >>> x = torch.randn(4, 20, 64, 64)  # [B=4, C=20, H=64, W=64]
            >>> out = layer(x)
            >>> out.shape  # [4, 20, 64, 64]
        """
        # Spatial dimensions — needed for irfft2's s argument.
        H: int = x.shape[-2]
        W: int = x.shape[-1]

        # ------------------------------------------------------------------
        # Step 1: Apply rfft2 along the last two dimensions.
        # rfft2 returns shape [B, in_channels, H, W//2 + 1] (complex cfloat).
        # The last dimension is halved due to conjugate symmetry of real signals.
        # The first spatial dimension H retains both positive and negative
        # frequencies (full FFT in that dimension).
        # ------------------------------------------------------------------
        x_ft: torch.Tensor = torch.fft.rfft2(x, dim=(-2, -1))
        # x_ft shape: [B, in_channels, H, W//2 + 1], complex cfloat

        # Validate mode counts against available frequencies.
        n_freqs_h: int = H
        n_freqs_w: int = W // 2 + 1

        if self.modes1 > n_freqs_h // 2 + 1:
            raise RuntimeError(
                f"SpectralConv2d: modes1={self.modes1} exceeds the number of "
                f"available frequencies in dim H={H}. "
                f"Reduce modes1 or increase H."
            )
        if self.modes2 > n_freqs_w:
            raise RuntimeError(
                f"SpectralConv2d: modes2={self.modes2} exceeds the number of "
                f"available rfft2 frequencies in dim W={W} (W//2+1={n_freqs_w}). "
                f"Reduce modes2 or increase W."
            )

        # ------------------------------------------------------------------
        # Step 2: Pre-allocate the output frequency tensor with zeros.
        # Shape: [B, out_channels, H, W//2 + 1], complex cfloat.
        # The zero entries correspond to high-frequency modes that are
        # discarded (low-pass truncation).
        # ------------------------------------------------------------------
        out_ft: torch.Tensor = torch.zeros(
            x.shape[0],        # B
            self.out_channels,
            n_freqs_h,         # H
            n_freqs_w,         # W//2 + 1
            dtype=torch.cfloat,
            device=x.device,
        )

        # ------------------------------------------------------------------
        # Step 3: Apply weights1 to the top-left frequency corner.
        # Top-left corner: x_ft[:, :, :modes1, :modes2]
        # These are the low-frequency modes with positive frequencies in
        # both the H and W dimensions.
        # ------------------------------------------------------------------
        out_ft[:, :, :self.modes1, :self.modes2] = self._compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2],
            self.weights1,
        )

        # ------------------------------------------------------------------
        # Step 4: Apply weights2 to the bottom-left frequency corner.
        # Bottom-left corner: x_ft[:, :, -modes1:, :modes2]
        # These are the low-frequency modes with negative frequencies in
        # the H dimension and positive frequencies in the W dimension.
        # This corner captures the conjugate-symmetric counterpart of the
        # top-left corner modes in the full 2D FFT.
        # ------------------------------------------------------------------
        out_ft[:, :, -self.modes1:, :self.modes2] = self._compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2],
            self.weights2,
        )

        # ------------------------------------------------------------------
        # Step 5: Apply irfft2 to recover the real-valued 2D spatial signal.
        # s=(H, W) ensures the output spatial dimensions match the input
        # exactly, regardless of whether H and W are odd or even.
        # ------------------------------------------------------------------
        out: torch.Tensor = torch.fft.irfft2(out_ft, s=(H, W), dim=(-2, -1))
        # out shape: [B, out_channels, H, W], real float32

        return out
