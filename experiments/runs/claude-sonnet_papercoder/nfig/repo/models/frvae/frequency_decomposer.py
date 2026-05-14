## models/frvae/frequency_decomposer.py
"""Frequency-guided Decomposer and Composer for the FR-VAE.

Implements the core frequency decomposition described in Section 3.1.1 of the
NFIG paper. Given a latent feature map f ∈ R^(B×C×H'×W') from the encoder,
this module decomposes it into n frequency-band components via 2D FFT masking,
and provides the inverse composition operation.

Paper equations (Section 3.1.1):
    Decomposer: f_hat_i = F^{-1}(F(f) ⊙ M_i),  ∀i ∈ {1, ..., n}
    Composer:   f_tilde = Σ_i T(f_hat_i, H', W')

Config values used (config.yaml frvae section):
    scale_factors:       [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
    latent_spatial_size: 16   (H' = W' = 16)
    num_frequency_bands: 10

The masks M_i are built lazily on first use (device-aware) and cached.
Frequency band boundaries σ_i are computed via FrequencyUtils following
the proportional allocation formula from Section 3.2.
"""

from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from utils.freq_utils import (
    build_frequency_masks,
    compute_frequency_boundaries,
    decompose_into_frequency_bands,
    compose_frequency_bands,
    verify_mask_reconstruction,
)


class FrequencyDecomposer(nn.Module):
    """Decomposes and composes latent feature maps in the frequency domain.

    Uses 2D FFT to split a feature map into n frequency-band components,
    where each band is defined by a radial frequency range [σ_{i-1}, σ_i).
    The frequency band widths are proportional to the number of tokens at
    each scale (Section 3.2), so lower-frequency bands are narrower and
    higher-frequency bands are wider.

    Masks are built lazily on the first call to decompose() and cached as
    non-parameter buffers. They are automatically rebuilt if the input
    tensor moves to a different device.

    Attributes:
        scale_factors: List of n scale factors defining token grid sizes.
            From config.frvae.scale_factors = [1,2,3,4,5,6,8,10,13,16].
        H_prime: Spatial height of the latent feature map (H' in paper).
            From config.frvae.latent_spatial_size = 16.
        W_prime: Spatial width of the latent feature map (W' in paper).
            From config.frvae.latent_spatial_size = 16.
        masks: List of n float32 tensors of shape (H', W'), or None if not
            yet built. Each mask is a binary frequency selection mask M_i.
        boundaries: List of n+1 frequency boundary values [σ_0, ..., σ_n].
            Pre-computed in __init__ for use in _build_masks.
        num_bands: Number of frequency bands n = len(scale_factors) = 10.
    """

    def __init__(
        self,
        scale_factors: List[int],
        H_prime: int,
        W_prime: int,
    ) -> None:
        """Initialize the FrequencyDecomposer.

        Pre-computes frequency band boundaries and defers mask construction
        to the first call to decompose() (device-aware lazy initialization).

        Args:
            scale_factors: List of n integer scale factors defining the token
                grid size for each frequency band. The i-th band uses a token
                grid of scale_factors[i] × scale_factors[i]. Must be strictly
                increasing and positive.
                From config.frvae.scale_factors = [1,2,3,4,5,6,8,10,13,16].
            H_prime: Spatial height of the latent feature map in pixels.
                From config.frvae.latent_spatial_size = 16.
                Must be positive.
            W_prime: Spatial width of the latent feature map in pixels.
                From config.frvae.latent_spatial_size = 16.
                Must be positive.

        Raises:
            ValueError: If scale_factors is empty, contains non-positive values,
                or is not strictly increasing.
            ValueError: If H_prime or W_prime are not positive integers.
            ValueError: If scale_factors[-1] does not equal H_prime (the highest-
                frequency band must cover the full feature map resolution, as
                required by config validation in utils/config.py).
        """
        super().__init__()

        # --- Input validation ---
        if not scale_factors:
            raise ValueError("scale_factors must be a non-empty list.")

        if any(s <= 0 for s in scale_factors):
            raise ValueError(
                f"All scale_factors must be positive integers. "
                f"Got: {scale_factors}"
            )

        for idx in range(1, len(scale_factors)):
            if scale_factors[idx] <= scale_factors[idx - 1]:
                raise ValueError(
                    f"scale_factors must be strictly increasing. "
                    f"Got scale_factors[{idx - 1}]={scale_factors[idx - 1]} >= "
                    f"scale_factors[{idx}]={scale_factors[idx]}. "
                    f"Full list: {scale_factors}"
                )

        if H_prime <= 0:
            raise ValueError(
                f"H_prime must be a positive integer, got {H_prime}."
            )

        if W_prime <= 0:
            raise ValueError(
                f"W_prime must be a positive integer, got {W_prime}."
            )

        if scale_factors[-1] != H_prime:
            raise ValueError(
                f"scale_factors[-1]={scale_factors[-1]} must equal H_prime={H_prime}. "
                "The highest-frequency band must cover the full feature map resolution. "
                "This is required for the residual quantizer to operate at full resolution "
                "in the final frequency band."
            )

        # --- Store configuration ---
        self.scale_factors: List[int] = list(scale_factors)
        self.H_prime: int = H_prime
        self.W_prime: int = W_prime
        self.num_bands: int = len(scale_factors)

        # --- Pre-compute frequency band boundaries ---
        # σ_i = σ_{i-1} + (h_i * w_i) / (Σ_j h_j * w_j) * σ_max
        # Returns list of n+1 values: [σ_0=0, σ_1, ..., σ_n=σ_max]
        # Computed on CPU; device-independent (pure Python floats).
        self.boundaries: List[float] = compute_frequency_boundaries(
            scale_factors=self.scale_factors,
            H=self.H_prime,
            W=self.W_prime,
        )

        # --- Lazy mask initialization ---
        # Masks are None until first call to decompose() or _build_masks().
        # This avoids device placement issues at construction time.
        self.masks: Optional[List[Tensor]] = None

        # Track the device of the currently cached masks for invalidation.
        # When the model is moved to a new device, masks are rebuilt.
        self._masks_device: Optional[torch.device] = None

    def _build_masks(self, device: torch.device) -> None:
        """Build and cache the n binary frequency selection masks M_i.

        Delegates to FrequencyUtils.build_frequency_masks() which constructs
        binary masks in the FFT domain (using PyTorch's fftfreq convention,
        where DC is at index [0,0] and frequencies are in [-0.5, 0.5)).

        The masks satisfy the partition-of-unity property:
            Σ_i M_i[u,v] = 1  for all (u,v)

        This ensures that decompose() followed by compose() is lossless:
            compose(decompose(f)) ≈ f  (up to float32 precision)

        After construction, masks are stored in self.masks and the device
        is recorded in self._masks_device for cache invalidation.

        Args:
            device: Target device for the mask tensors. Must match the device
                of the input tensor passed to decompose().

        Side effects:
            Sets self.masks to a list of n float32 tensors of shape (H', W').
            Sets self._masks_device to the given device.

        Raises:
            ValueError: If build_frequency_masks() detects gaps or overlaps
                in the frequency band partition (propagated from FrequencyUtils).
        """
        self.masks = build_frequency_masks(
            scale_factors=self.scale_factors,
            H=self.H_prime,
            W=self.W_prime,
            device=device,
        )
        self._masks_device = device

    def decompose(self, f: Tensor) -> List[Tensor]:
        """Decompose a latent feature map into n frequency-band components.

        Implements the Frequency-guided Decomposer from paper Section 3.1.1:
            f_hat_i = F^{-1}(F(f) ⊙ M_i),  ∀i ∈ {1, ..., n}

        The 2D FFT is computed once and reused for all n bands, making the
        decomposition O(n * H' * W') after the initial O(H' * W' * log(H'W'))
        FFT computation.

        The output components satisfy the reconstruction identity:
            Σ_i f_hat_i ≈ f  (up to float32 FFT round-trip precision)

        Args:
            f: Real-valued latent feature map of shape (B, C, H', W').
               For the default config: (B, 768, 16, 16).
               Values are in an unconstrained real range (encoder output).
               Must have spatial dimensions matching H_prime × W_prime.

        Returns:
            List of n tensors, each of shape (B, C, H', W'), real-valued.
            The i-th tensor f_hat_i contains only the frequency content of
            band i (radial frequencies in [σ_{i-1}, σ_i)).
            Ordered from lowest frequency (i=0, DC + near-DC) to highest
            frequency (i=n-1, high-frequency details).

        Raises:
            ValueError: If f.shape[-2:] does not match (H_prime, W_prime).
            RuntimeError: If f is not a real-valued tensor (complex inputs
                are not supported; the encoder always outputs real features).
        """
        # --- Validate input spatial dimensions ---
        if f.shape[-2] != self.H_prime or f.shape[-1] != self.W_prime:
            raise ValueError(
                f"Input spatial dimensions {f.shape[-2]}×{f.shape[-1]} do not match "
                f"expected H_prime×W_prime = {self.H_prime}×{self.W_prime}. "
                f"Full input shape: {tuple(f.shape)}."
            )

        if f.is_complex():
            raise RuntimeError(
                "FrequencyDecomposer.decompose() expects a real-valued input tensor. "
                f"Got complex tensor of dtype {f.dtype}. "
                "The encoder (VQGANEncoder) always outputs real-valued feature maps."
            )

        # --- Lazy mask construction / device cache invalidation ---
        # Rebuild masks if: (1) not yet built, or (2) input moved to new device.
        if self.masks is None or self._masks_device != f.device:
            self._build_masks(f.device)

        # --- Decompose into frequency bands ---
        # decompose_into_frequency_bands() computes FFT once and applies all masks.
        # Returns list of n real-valued tensors, each (B, C, H', W').
        components: List[Tensor] = decompose_into_frequency_bands(
            f=f,
            masks=self.masks,  # type: ignore[arg-type]  # masks is not None here
        )

        return components

    def compose(self, components: List[Tensor]) -> Tensor:
        """Reconstruct a feature map by summing frequency-band components.

        Implements the Frequency-guided Composer from paper Section 3.1.1:
            f_tilde = Σ_i T(f_hat_i, H', W')

        Since all components are already at H'×W' (FFT decomposition preserves
        spatial dimensions), T(·) is the identity and this reduces to a direct
        element-wise sum.

        Note: In ResidualQuantizer, the quantized representations v_i^q are at
        reduced resolution h_i×w_i and are upsampled to H'×W' before being
        passed to this method. The upsampling T(·) is handled by
        ResidualQuantizer._upsample(), not here.

        Args:
            components: List of n tensors, each of shape (B, C, H', W').
                Typically the output of decompose() (for lossless reconstruction)
                or upsampled quantized representations from ResidualQuantizer
                (for quantized reconstruction).
                All tensors must have identical shapes.

        Returns:
            Reconstructed feature map f_tilde of shape (B, C, H', W').
            When components = decompose(f), the result satisfies:
                f_tilde ≈ f  (up to float32 FFT round-trip precision)

        Raises:
            ValueError: If components is empty.
            ValueError: If component tensors have inconsistent shapes.
        """
        if not components:
            raise ValueError(
                "Cannot compose an empty list of frequency components. "
                f"Expected {self.num_bands} components."
            )

        # Validate shape consistency across all components.
        reference_shape = components[0].shape
        for idx, component in enumerate(components[1:], start=1):
            if component.shape != reference_shape:
                raise ValueError(
                    f"All frequency components must have the same shape. "
                    f"components[0].shape={reference_shape}, "
                    f"components[{idx}].shape={component.shape}."
                )

        # Direct element-wise sum: f_tilde = Σ_i f_hat_i
        f_tilde: Tensor = compose_frequency_bands(components)

        return f_tilde

    def verify_reconstruction(
        self,
        f: Tensor,
        atol: float = 1e-4,
    ) -> bool:
        """Verify that decompose followed by compose is lossless.

        Tests the reconstruction identity:
            compose(decompose(f)) ≈ f

        This is a correctness check for the frequency decomposition. Should
        be called during unit testing and optionally at initialization time
        to validate the mask construction.

        Args:
            f: Real-valued feature map of shape (B, C, H', W') to test.
            atol: Absolute tolerance for the reconstruction error.
                Default 1e-4 is appropriate for float32 FFT round-trip errors.

        Returns:
            True if the reconstruction error is within tolerance, False otherwise.
            A warning is issued if the check fails (via verify_mask_reconstruction).
        """
        # Ensure masks are built before verification.
        if self.masks is None or self._masks_device != f.device:
            self._build_masks(f.device)

        return verify_mask_reconstruction(
            f=f,
            masks=self.masks,  # type: ignore[arg-type]
            atol=atol,
        )

    def get_band_info(self) -> List[dict]:
        """Return metadata about each frequency band for logging/debugging.

        Provides a human-readable summary of each band's frequency range,
        token count, and proportion of the total frequency spectrum.

        Returns:
            List of n dicts, one per frequency band, each containing:
                - 'band_idx': int, band index (0 = lowest frequency)
                - 'scale_factor': int, token grid size (s_i)
                - 'num_tokens': int, number of tokens (s_i^2)
                - 'sigma_low': float, lower frequency boundary σ_{i-1}
                - 'sigma_high': float, upper frequency boundary σ_i
                - 'bandwidth': float, frequency range width σ_i - σ_{i-1}
        """
        band_info: List[dict] = []
        for i, scale in enumerate(self.scale_factors):
            band_info.append({
                "band_idx": i,
                "scale_factor": scale,
                "num_tokens": scale * scale,
                "sigma_low": self.boundaries[i],
                "sigma_high": self.boundaries[i + 1],
                "bandwidth": self.boundaries[i + 1] - self.boundaries[i],
            })
        return band_info

    def extra_repr(self) -> str:
        """Return a human-readable string with key decomposer configuration.

        Returns:
            String describing the decomposer's scale factors and spatial size.
        """
        total_tokens: int = sum(s * s for s in self.scale_factors)
        return (
            f"num_bands={self.num_bands}, "
            f"scale_factors={self.scale_factors}, "
            f"spatial_size={self.H_prime}×{self.W_prime}, "
            f"total_tokens={total_tokens}, "
            f"masks_built={self.masks is not None}"
        )
