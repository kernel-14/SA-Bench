# models/fourier_layer.py
"""Multi-head Fourier layer for the MoE-POT architecture.

Implements the FourierLayer class, which approximates the kernel integral
operator in the Fourier domain using learnable complex weight matrices.
This is the first sub-layer in each MoEBlock, and its output z_0^l feeds
directly into the MoELayer.

From the paper (Section 4, "Fourier Layer"):
    "Using a multi-head architecture, the Fourier layer is designed to learn
    complex kernel-based integral transformations that approximate PDE solutions.
    We approximate (K_phi * z^l)(x) using h smaller MLPs:
        z_{0i}^l(x) = F^{-1}[W_{2,i}^l · σ(W_{1,i}^l · F[z_i^l] + b_{1,i}^l)
                              + b_{2,i}^l](x)
    where W_{1,i}^l, W_{2,i}^l ∈ R^{d_z/h × d_z/h} and σ(·) is GELU."

The skip connection via a pointwise Conv2d is added to the spectral output,
following the standard FNO design (local linear path + global spectral path).

From config.yaml:
    architecture.modes_x: 8          (Fourier modes in x, half of 16-token grid)
    architecture.modes_y: 8          (Fourier modes in y)
    architecture.patch_size: 8       (determines H'=W'=16 token grid)
    architecture.target_resolution: 128
    models.tiny.attn_dim: 512        (embed_dim for Tiny)
    models.tiny.num_heads: 4         (num_heads for Tiny, head_dim=128)
    models.small.attn_dim: 1024      (embed_dim for Small/Medium)
    models.small.num_heads: 8        (num_heads for Small/Medium, head_dim=128)

Data flow:
    Input:  (B, embed_dim, H'=16, W'=16)
    Output: (B, embed_dim, H'=16, W'=16)
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierLayer(nn.Module):
    """Multi-head spectral convolution layer for PDE operator learning.

    Implements a two-layer MLP in the Fourier domain per head, with GELU
    activation between layers. The spectral output is combined with a
    skip connection (pointwise Conv2d) to form the final output z_0^l.

    Architecture per head i:
        z_{0i}^l(x) = F^{-1}[W_{2,i} · GELU(W_{1,i} · F[z_i^l] + b_{1,i})
                              + b_{2,i}](x)

    Full layer:
        spectral_out = Concat(z_{01}^l, ..., z_{0h}^l)
        output = spectral_out + skip_conv(x)

    Key design choices:
      - Parameters stored as separate real/imag nn.Parameter tensors for
        optimizer compatibility and clean serialization.
      - FFT with norm='ortho' for unitary transform and training stability.
      - Low-frequency mode truncation (top-left corner of spectrum).
      - GELU applied to real and imaginary parts separately.
      - Vectorized einsum over all heads simultaneously for GPU efficiency.
      - Zero-padding back to full rfft2 size before irfft2.
      - irfft2 with explicit s=(H', W') to guarantee correct spatial dims.

    Attributes:
        embed_dim: Total feature dimension. Equals attn_dim from config.
            512 (Tiny), 1024 (Small/Medium).
        num_heads: Number of spectral heads h. 4 (Tiny), 8 (Small/Medium).
        modes_x: Number of Fourier modes retained in x-direction. Default 8
            (config.yaml architecture.modes_x).
        modes_y: Number of Fourier modes retained in y-direction. Default 8
            (config.yaml architecture.modes_y).
        head_dim: Feature dimension per head = embed_dim // num_heads.
            128 for both Tiny (512//4) and Small/Medium (1024//8).
        w1_real: Real part of first-layer weight W1.
            Shape: (num_heads, head_dim, head_dim, modes_x, modes_y).
        w1_imag: Imaginary part of first-layer weight W1.
            Shape: (num_heads, head_dim, head_dim, modes_x, modes_y).
        w2_real: Real part of second-layer weight W2.
            Shape: (num_heads, head_dim, head_dim, modes_x, modes_y).
        w2_imag: Imaginary part of second-layer weight W2.
            Shape: (num_heads, head_dim, head_dim, modes_x, modes_y).
        b1_real: Real part of first-layer bias b1.
            Shape: (num_heads, head_dim, modes_x, modes_y).
        b1_imag: Imaginary part of first-layer bias b1.
            Shape: (num_heads, head_dim, modes_x, modes_y).
        b2_real: Real part of second-layer bias b2.
            Shape: (num_heads, head_dim, modes_x, modes_y).
        b2_imag: Imaginary part of second-layer bias b2.
            Shape: (num_heads, head_dim, modes_x, modes_y).
        skip_conv: Conv2d(embed_dim, embed_dim, kernel_size=1) for the
            local linear skip connection (residual path).
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 4,
        modes_x: int = 8,
        modes_y: int = 8,
    ) -> None:
        """Initializes the FourierLayer with learnable complex spectral weights.

        Constructs all learnable parameters as pairs of real/imaginary
        nn.Parameter tensors. Initializes weights with scaled random normal
        values (scale = 1 / sqrt(head_dim)) and biases to zero.

        Args:
            embed_dim: Total feature dimension. Must be divisible by num_heads.
                Corresponds to attn_dim in config.yaml model configurations:
                  - Tiny:   512  (config.yaml models.tiny.attn_dim)
                  - Small:  1024 (config.yaml models.small.attn_dim)
                  - Medium: 1024 (config.yaml models.medium.attn_dim)
            num_heads: Number of spectral attention heads h. The channel
                dimension is split into num_heads groups of size head_dim.
                Corresponds to num_heads in config.yaml:
                  - Tiny:   4 (config.yaml models.tiny.num_heads)
                  - Small:  8 (config.yaml models.small.num_heads)
                  - Medium: 8 (config.yaml models.medium.num_heads)
            modes_x: Number of Fourier modes to retain in the x-direction
                (first spatial dimension). Default 8 (config.yaml
                architecture.modes_x). Must be <= H'//2 + 1 where H' is
                the token grid height (16 for patch_size=8, img_size=128).
            modes_y: Number of Fourier modes to retain in the y-direction
                (second spatial dimension, rfft2 output). Default 8
                (config.yaml architecture.modes_y). Must be <= W'//2 + 1
                where W'//2 + 1 = 9 for W'=16.

        Raises:
            ValueError: If embed_dim is not divisible by num_heads.
            ValueError: If embed_dim <= 0, num_heads <= 0, modes_x <= 0,
                or modes_y <= 0.
        """
        super().__init__()

        # --- Input validation ---
        if embed_dim <= 0:
            raise ValueError(
                f"embed_dim must be positive, got {embed_dim}."
            )
        if num_heads <= 0:
            raise ValueError(
                f"num_heads must be positive, got {num_heads}."
            )
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads "
                f"({num_heads}). Got remainder {embed_dim % num_heads}."
            )
        if modes_x <= 0:
            raise ValueError(
                f"modes_x must be positive, got {modes_x}."
            )
        if modes_y <= 0:
            raise ValueError(
                f"modes_y must be positive, got {modes_y}."
            )

        # Store configuration attributes.
        self.embed_dim: int = embed_dim
        self.num_heads: int = num_heads
        self.modes_x: int = modes_x
        self.modes_y: int = modes_y

        # Per-head feature dimension.
        # Tiny:   512 // 4  = 128
        # Small:  1024 // 8 = 128
        # Medium: 1024 // 8 = 128
        self.head_dim: int = embed_dim // num_heads

        # ----------------------------------------------------------------
        # Initialization scale for spectral weights
        # ----------------------------------------------------------------
        # Scale = 1 / sqrt(head_dim) follows the standard FNO initialization
        # convention, which prevents vanishing/exploding gradients in the
        # frequency domain. The paper does not specify initialization details,
        # so we follow the DPOT/FNO convention.
        init_scale: float = 1.0 / (self.head_dim ** 0.5)

        # ----------------------------------------------------------------
        # First-layer complex weights W1 ∈ C^{head_dim × head_dim}
        # ----------------------------------------------------------------
        # Shape: (num_heads, head_dim_out, head_dim_in, modes_x, modes_y)
        # Stored as separate real and imaginary nn.Parameter tensors.
        # The output dimension (head_dim_out) is first for einsum efficiency:
        #   einsum('b h i x y, h o i x y -> b h o x y', X, W1)
        # where i = head_dim_in, o = head_dim_out.
        self.w1_real: nn.Parameter = nn.Parameter(
            torch.randn(num_heads, self.head_dim, self.head_dim, modes_x, modes_y)
            * init_scale
        )
        self.w1_imag: nn.Parameter = nn.Parameter(
            torch.randn(num_heads, self.head_dim, self.head_dim, modes_x, modes_y)
            * init_scale
        )

        # ----------------------------------------------------------------
        # Second-layer complex weights W2 ∈ C^{head_dim × head_dim}
        # ----------------------------------------------------------------
        # Same shape as W1. Applied after GELU activation.
        self.w2_real: nn.Parameter = nn.Parameter(
            torch.randn(num_heads, self.head_dim, self.head_dim, modes_x, modes_y)
            * init_scale
        )
        self.w2_imag: nn.Parameter = nn.Parameter(
            torch.randn(num_heads, self.head_dim, self.head_dim, modes_x, modes_y)
            * init_scale
        )

        # ----------------------------------------------------------------
        # First-layer complex biases b1 ∈ C^{head_dim}
        # ----------------------------------------------------------------
        # Shape: (num_heads, head_dim, modes_x, modes_y)
        # Initialized to zero (standard bias initialization).
        # The bias is frequency-dependent (different b1 per mode), which
        # allows the model to learn mode-specific offsets.
        self.b1_real: nn.Parameter = nn.Parameter(
            torch.zeros(num_heads, self.head_dim, modes_x, modes_y)
        )
        self.b1_imag: nn.Parameter = nn.Parameter(
            torch.zeros(num_heads, self.head_dim, modes_x, modes_y)
        )

        # ----------------------------------------------------------------
        # Second-layer complex biases b2 ∈ C^{head_dim}
        # ----------------------------------------------------------------
        # Shape: (num_heads, head_dim, modes_x, modes_y)
        # Initialized to zero.
        self.b2_real: nn.Parameter = nn.Parameter(
            torch.zeros(num_heads, self.head_dim, modes_x, modes_y)
        )
        self.b2_imag: nn.Parameter = nn.Parameter(
            torch.zeros(num_heads, self.head_dim, modes_x, modes_y)
        )

        # ----------------------------------------------------------------
        # Skip connection (local linear path)
        # ----------------------------------------------------------------
        # Pointwise Conv2d(embed_dim, embed_dim, kernel_size=1) provides
        # a local linear transformation that complements the global spectral
        # path. This is the standard FNO residual design.
        # kernel_size=1 means no spatial mixing — purely channel-wise.
        # bias=True (default): standard for convolutional layers.
        self.skip_conv: nn.Conv2d = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=1,
            bias=True,
        )

    def _spectral_conv_head(
        self,
        x_head: torch.Tensor,
        head_idx: int,
        h_prime: int,
        w_prime: int,
    ) -> torch.Tensor:
        """Applies the two-layer spectral MLP for a single head.

        This method is provided for clarity and testing but is NOT called
        in the main forward pass, which uses the vectorized batched
        implementation for efficiency. It implements the per-head formula:
            z_{0i}^l(x) = F^{-1}[W2 · GELU(W1 · F[z_i^l] + b1) + b2](x)

        Args:
            x_head: Input feature map for this head, shape (B, head_dim, H', W').
            head_idx: Index of the head (0 to num_heads-1).
            h_prime: Token grid height H'.
            w_prime: Token grid width W'.

        Returns:
            Output feature map for this head, shape (B, head_dim, H', W').
        """
        batch_size: int = x_head.shape[0]

        # Apply rfft2 to get frequency domain representation.
        # Input:  (B, head_dim, H', W')
        # Output: (B, head_dim, H', W'//2 + 1) complex
        x_freq: torch.Tensor = torch.fft.rfft2(x_head, norm="ortho")

        # Truncate to modes_x × modes_y low-frequency modes.
        # x_freq shape: (B, head_dim, H', W'//2+1)
        # x_trunc shape: (B, head_dim, modes_x, modes_y)
        x_trunc: torch.Tensor = x_freq[
            :, :, : self.modes_x, : self.modes_y
        ]

        # Build complex weight matrices for this head.
        # W1 shape: (head_dim_out, head_dim_in, modes_x, modes_y) complex
        w1_complex: torch.Tensor = torch.complex(
            self.w1_real[head_idx], self.w1_imag[head_idx]
        )
        b1_complex: torch.Tensor = torch.complex(
            self.b1_real[head_idx], self.b1_imag[head_idx]
        )
        w2_complex: torch.Tensor = torch.complex(
            self.w2_real[head_idx], self.w2_imag[head_idx]
        )
        b2_complex: torch.Tensor = torch.complex(
            self.b2_real[head_idx], self.b2_imag[head_idx]
        )

        # First spectral linear layer: W1 · x_trunc + b1
        # x_trunc: (B, head_dim_in, modes_x, modes_y)
        # w1:      (head_dim_out, head_dim_in, modes_x, modes_y)
        # out1:    (B, head_dim_out, modes_x, modes_y)
        out1: torch.Tensor = torch.einsum(
            "b i x y, o i x y -> b o x y", x_trunc, w1_complex
        ) + b1_complex

        # Apply GELU to real and imaginary parts separately.
        out1_activated: torch.Tensor = torch.complex(
            F.gelu(out1.real), F.gelu(out1.imag)
        )

        # Second spectral linear layer: W2 · activated + b2
        out2: torch.Tensor = torch.einsum(
            "b i x y, o i x y -> b o x y", out1_activated, w2_complex
        ) + b2_complex

        # Zero-pad back to full rfft2 output size.
        w_freq_size: int = w_prime // 2 + 1
        out_padded: torch.Tensor = torch.zeros(
            batch_size,
            self.head_dim,
            h_prime,
            w_freq_size,
            dtype=torch.cfloat,
            device=x_head.device,
        )
        out_padded[:, :, : self.modes_x, : self.modes_y] = out2

        # Apply irfft2 to recover spatial domain.
        # Explicitly pass s=(H', W') to guarantee correct output size.
        out_spatial: torch.Tensor = torch.fft.irfft2(
            out_padded, s=(h_prime, w_prime), norm="ortho"
        )
        # Shape: (B, head_dim, H', W')

        return out_spatial

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the multi-head Fourier layer to the input feature map.

        Implements the vectorized batched computation across all heads
        simultaneously using einsum, avoiding Python-level loops over heads.

        Processing pipeline:
            (B, embed_dim, H', W')
            → split into heads: (B, num_heads, head_dim, H', W')
            → rfft2 per head: (B, num_heads, head_dim, H', W'//2+1) complex
            → truncate: (B, num_heads, head_dim, modes_x, modes_y) complex
            → W1 · x + b1: (B, num_heads, head_dim, modes_x, modes_y) complex
            → GELU(real), GELU(imag): same shape
            → W2 · activated + b2: same shape
            → zero-pad: (B, num_heads, head_dim, H', W'//2+1) complex
            → irfft2: (B, num_heads, head_dim, H', W') real
            → reshape: (B, embed_dim, H', W')
            → + skip_conv(x): (B, embed_dim, H', W')

        Args:
            x: Input feature map of shape (B, embed_dim, H', W') where:
                - B: Batch size (up to 20 for pre-training, config.yaml
                  pretraining.batch_size).
                - embed_dim: Must match self.embed_dim (attn_dim from config).
                - H': Token grid height, typically 16 (= 128 / patch_size=8,
                  config.yaml architecture.target_resolution=128,
                  architecture.patch_size=8).
                - W': Token grid width, typically 16.
                This is z^l(x) from the paper — the pre-norm input to the
                Fourier layer within each MoEBlock.

        Returns:
            Output feature map z_0^l of shape (B, embed_dim, H', W').
            This is passed to the MoELayer in MoEBlock.forward().
            The skip connection is included in this output.
        """
        batch_size: int = x.shape[0]
        h_prime: int = x.shape[2]
        w_prime: int = x.shape[3]

        # rfft2 output size in the last dimension.
        # torch.fft.rfft2 returns W'//2 + 1 complex values in the last dim.
        # For W'=16: w_freq_size = 9.
        w_freq_size: int = w_prime // 2 + 1

        # ----------------------------------------------------------------
        # Step 1: Split into heads
        # ----------------------------------------------------------------
        # Reshape channel dimension into (num_heads, head_dim).
        # Input:  (B, embed_dim, H', W')
        # Output: (B, num_heads, head_dim, H', W')
        x_heads: torch.Tensor = x.reshape(
            batch_size, self.num_heads, self.head_dim, h_prime, w_prime
        )

        # ----------------------------------------------------------------
        # Step 2: Apply rfft2 to all heads simultaneously
        # ----------------------------------------------------------------
        # Merge batch and head dimensions for batched rfft2.
        # (B, num_heads, head_dim, H', W')
        # → reshape to (B * num_heads, head_dim, H', W')
        # → rfft2 → (B * num_heads, head_dim, H', W'//2+1) complex
        # → reshape to (B, num_heads, head_dim, H', W'//2+1) complex
        x_heads_flat: torch.Tensor = x_heads.reshape(
            batch_size * self.num_heads, self.head_dim, h_prime, w_prime
        )
        x_freq_flat: torch.Tensor = torch.fft.rfft2(x_heads_flat, norm="ortho")
        # Shape: (B * num_heads, head_dim, H', W'//2+1) complex

        x_freq: torch.Tensor = x_freq_flat.reshape(
            batch_size, self.num_heads, self.head_dim, h_prime, w_freq_size
        )
        # Shape: (B, num_heads, head_dim, H', W'//2+1) complex

        # ----------------------------------------------------------------
        # Step 3: Truncate to low-frequency modes
        # ----------------------------------------------------------------
        # Retain only the top-left modes_x × modes_y corner of the spectrum.
        # These low-frequency modes capture the dominant spatial patterns.
        # x_trunc shape: (B, num_heads, head_dim, modes_x, modes_y) complex
        x_trunc: torch.Tensor = x_freq[
            :, :, :, : self.modes_x, : self.modes_y
        ]

        # ----------------------------------------------------------------
        # Step 4: Build complex weight and bias tensors for all heads
        # ----------------------------------------------------------------
        # Combine real and imaginary parts into complex tensors.
        # All shapes: (num_heads, head_dim, head_dim, modes_x, modes_y) complex
        w1_complex: torch.Tensor = torch.complex(self.w1_real, self.w1_imag)
        w2_complex: torch.Tensor = torch.complex(self.w2_real, self.w2_imag)

        # Bias shapes: (num_heads, head_dim, modes_x, modes_y) complex
        b1_complex: torch.Tensor = torch.complex(self.b1_real, self.b1_imag)
        b2_complex: torch.Tensor = torch.complex(self.b2_real, self.b2_imag)

        # ----------------------------------------------------------------
        # Step 5: First spectral linear layer — W1 · x_trunc + b1
        # ----------------------------------------------------------------
        # Vectorized einsum over all heads simultaneously.
        # x_trunc:   (B, num_heads, head_dim_in,  modes_x, modes_y) complex
        # w1_complex:(   num_heads, head_dim_out, head_dim_in, modes_x, modes_y) complex
        # out1:      (B, num_heads, head_dim_out, modes_x, modes_y) complex
        #
        # Einsum notation:
        #   b = batch, h = head, i = head_dim_in, o = head_dim_out,
        #   x = modes_x, y = modes_y
        out1: torch.Tensor = torch.einsum(
            "b h i x y, h o i x y -> b h o x y",
            x_trunc,
            w1_complex,
        )
        # Add bias: b1_complex shape (num_heads, head_dim, modes_x, modes_y)
        # Unsqueeze batch dim for broadcasting: (1, num_heads, head_dim, modes_x, modes_y)
        out1 = out1 + b1_complex.unsqueeze(0)
        # Shape: (B, num_heads, head_dim, modes_x, modes_y) complex

        # ----------------------------------------------------------------
        # Step 6: Apply GELU activation to real and imaginary parts
        # ----------------------------------------------------------------
        # Standard approach for complex-valued networks with real activations:
        # apply the activation independently to real and imaginary components.
        # This preserves the complex structure while introducing nonlinearity.
        out1_activated: torch.Tensor = torch.complex(
            F.gelu(out1.real),
            F.gelu(out1.imag),
        )
        # Shape: (B, num_heads, head_dim, modes_x, modes_y) complex

        # ----------------------------------------------------------------
        # Step 7: Second spectral linear layer — W2 · activated + b2
        # ----------------------------------------------------------------
        # Same einsum pattern as Step 5.
        # out2 shape: (B, num_heads, head_dim, modes_x, modes_y) complex
        out2: torch.Tensor = torch.einsum(
            "b h i x y, h o i x y -> b h o x y",
            out1_activated,
            w2_complex,
        )
        # Add bias.
        out2 = out2 + b2_complex.unsqueeze(0)
        # Shape: (B, num_heads, head_dim, modes_x, modes_y) complex

        # ----------------------------------------------------------------
        # Step 8: Zero-pad back to full rfft2 output size
        # ----------------------------------------------------------------
        # Create a zero tensor of the full rfft2 output size and insert
        # the computed modes into the low-frequency corner.
        # out_padded shape: (B, num_heads, head_dim, H', W'//2+1) complex
        out_padded: torch.Tensor = torch.zeros(
            batch_size,
            self.num_heads,
            self.head_dim,
            h_prime,
            w_freq_size,
            dtype=torch.cfloat,
            device=x.device,
        )
        # Insert computed modes into the low-frequency corner.
        out_padded[:, :, :, : self.modes_x, : self.modes_y] = out2

        # ----------------------------------------------------------------
        # Step 9: Apply irfft2 to recover spatial domain
        # ----------------------------------------------------------------
        # Merge batch and head dimensions for batched irfft2.
        # (B, num_heads, head_dim, H', W'//2+1)
        # → reshape to (B * num_heads, head_dim, H', W'//2+1)
        # → irfft2 with s=(H', W') → (B * num_heads, head_dim, H', W') real
        # → reshape to (B, num_heads, head_dim, H', W') real
        out_padded_flat: torch.Tensor = out_padded.reshape(
            batch_size * self.num_heads, self.head_dim, h_prime, w_freq_size
        )
        # Explicitly pass s=(H', W') to guarantee the correct output spatial
        # dimensions, since irfft2 infers size from the input by default
        # (which would give W'//2*2 = W' only if W' is even, but explicit
        # is safer and required when W' is odd or when padding was applied).
        out_spatial_flat: torch.Tensor = torch.fft.irfft2(
            out_padded_flat,
            s=(h_prime, w_prime),
            norm="ortho",
        )
        # Shape: (B * num_heads, head_dim, H', W') real float32

        out_spatial: torch.Tensor = out_spatial_flat.reshape(
            batch_size, self.num_heads, self.head_dim, h_prime, w_prime
        )
        # Shape: (B, num_heads, head_dim, H', W') real float32

        # ----------------------------------------------------------------
        # Step 10: Concatenate heads back to full embed_dim
        # ----------------------------------------------------------------
        # Reshape (B, num_heads, head_dim, H', W') → (B, embed_dim, H', W')
        # by merging the num_heads and head_dim dimensions.
        # contiguous() ensures the memory layout is correct after reshape.
        spectral_out: torch.Tensor = out_spatial.reshape(
            batch_size, self.embed_dim, h_prime, w_prime
        ).contiguous()
        # Shape: (B, embed_dim, H', W')

        # ----------------------------------------------------------------
        # Step 11: Add skip connection (local linear path)
        # ----------------------------------------------------------------
        # skip_conv is Conv2d(embed_dim, embed_dim, kernel_size=1).
        # It provides a local linear transformation that complements the
        # global spectral path, following the standard FNO design.
        # Both terms have shape (B, embed_dim, H', W').
        output: torch.Tensor = spectral_out + self.skip_conv(x)
        # Shape: (B, embed_dim, H', W')

        return output
