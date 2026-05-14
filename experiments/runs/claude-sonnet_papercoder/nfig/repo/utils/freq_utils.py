## utils/freq_utils.py
"""Frequency domain utilities for the NFIG framework.

Provides FFT-based frequency mask construction, application, and visualization
helpers used by FrequencyDecomposer and ResidualQuantizer. All math follows
Section 3.1 and Appendix A of the NFIG paper.

Key functions:
    - compute_frequency_boundaries: Compute σ_i band boundaries (Section 3.2)
    - build_frequency_masks: Build binary spatial masks M_i in FFT domain
    - apply_fft_mask: Apply a single mask via FFT → multiply → IFFT
    - apply_fft_mask_from_spectrum: Apply mask to pre-computed FFT spectrum
    - visualize_frequency_spectrum: Log-magnitude spectrum for visualization
    - compute_power_spectral_density: Radially averaged PSD (Appendix B.2)
    - compute_frequency_keep_score: Weighted FKS metric (Appendix B.2)
"""

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def compute_frequency_boundaries(
    scale_factors: List[int],
    H: int,
    W: int,
) -> List[float]:
    """Compute cumulative frequency band boundaries σ_i.

    Implements the formula from paper Section 3.2:
        σ_i = σ_{i-1} + (h_i * w_i) / (Σ_j h_j * w_j) * σ_max

    where σ_max = sqrt((H/2)^2 + (W/2)^2) is the Nyquist radius in the
    shifted FFT domain (DC centered).

    Args:
        scale_factors: List of n scale factors [s_1, ..., s_n]. Each s_i
            defines a token grid of size s_i × s_i for frequency band i.
            From config: [1, 2, 3, 4, 5, 6, 8, 10, 13, 16].
        H: Spatial height of the feature map (H' in the paper). Typically 16.
        W: Spatial width of the feature map (W' in the paper). Typically 16.

    Returns:
        List of n+1 boundary values [σ_0=0.0, σ_1, ..., σ_n=σ_max].
        Band i covers the half-open interval [σ_{i-1}, σ_i) for i < n,
        and the closed interval [σ_{n-1}, σ_n] for the last band.

    Example:
        >>> boundaries = compute_frequency_boundaries([1,2,3,4,5,6,8,10,13,16], 16, 16)
        >>> len(boundaries)  # n+1 = 11
        11
        >>> boundaries[0]  # σ_0 = 0
        0.0
        >>> abs(boundaries[-1] - math.sqrt(8**2 + 8**2)) < 1e-6
        True
    """
    n = len(scale_factors)
    token_counts: List[int] = [s * s for s in scale_factors]
    total_tokens: int = sum(token_counts)

    # σ_max: maximum radial frequency in the shifted FFT domain.
    # The DC component is at the center (H/2, W/2), so the maximum
    # radial distance to a corner is sqrt((H/2)^2 + (W/2)^2).
    sigma_max: float = math.sqrt((H / 2.0) ** 2 + (W / 2.0) ** 2)

    boundaries: List[float] = [0.0]
    cumulative: float = 0.0
    for i in range(n):
        proportion: float = token_counts[i] / total_tokens
        cumulative += proportion * sigma_max
        boundaries.append(cumulative)

    # Force the last boundary to exactly σ_max to avoid floating-point gaps.
    boundaries[-1] = sigma_max

    # Validate monotonicity.
    for i in range(1, len(boundaries)):
        if boundaries[i] < boundaries[i - 1]:
            raise ValueError(
                f"Frequency boundaries are not monotonically increasing at index {i}: "
                f"{boundaries[i - 1]:.6f} -> {boundaries[i]:.6f}"
            )

    return boundaries


def build_frequency_masks(
    scale_factors: List[int],
    H: int,
    W: int,
    device: torch.device,
) -> List[Tensor]:
    """Build binary frequency selection masks M_i for each frequency band.

    Each mask M_i is a 2D binary map of shape (H, W) defined in the shifted
    FFT domain (DC at center). Mask i selects all spatial frequencies whose
    radial distance from the center falls within [σ_{i-1}, σ_i).

    The masks are mutually exclusive and collectively exhaustive:
        Σ_i M_i = ones(H, W)  (every frequency assigned to exactly one band)

    Args:
        scale_factors: List of n scale factors defining frequency band widths.
            From config: [1, 2, 3, 4, 5, 6, 8, 10, 13, 16].
        H: Feature map height (H' in paper). Typically 16.
        W: Feature map width (W' in paper). Typically 16.
        device: Target device for the returned tensors.

    Returns:
        List of n float32 tensors, each of shape (H, W), on the given device.
        Values are 0.0 or 1.0 (binary masks).

    Raises:
        ValueError: If masks do not sum to all-ones (coverage check fails).
    """
    boundaries: List[float] = compute_frequency_boundaries(scale_factors, H, W)
    n: int = len(scale_factors)

    # Build centered coordinate grid on CPU for numerical precision.
    # y_coords[i] = i - H/2, so the center pixel has coordinate 0.
    y_coords: Tensor = torch.arange(H, dtype=torch.float32) - H / 2.0
    x_coords: Tensor = torch.arange(W, dtype=torch.float32) - W / 2.0

    # Shape: (H, W)
    Y, X = torch.meshgrid(y_coords, x_coords, indexing="ij")
    radial_freq: Tensor = torch.sqrt(Y ** 2 + X ** 2)  # (H, W)

    masks: List[Tensor] = []
    for i in range(n):
        sigma_low: float = boundaries[i]
        sigma_high: float = boundaries[i + 1]

        if i < n - 1:
            # Half-open interval [σ_{i-1}, σ_i) for all but the last band.
            mask: Tensor = (radial_freq >= sigma_low) & (radial_freq < sigma_high)
        else:
            # Closed interval [σ_{n-1}, σ_n] for the last band to ensure
            # full coverage including the maximum frequency corner.
            mask = (radial_freq >= sigma_low) & (radial_freq <= sigma_high)

        masks.append(mask.float())

    # --- Correctness check: masks must partition the frequency plane ---
    mask_sum: Tensor = torch.stack(masks, dim=0).sum(dim=0)  # (H, W)
    expected_sum: Tensor = torch.ones(H, W, dtype=torch.float32)
    if not torch.allclose(mask_sum, expected_sum, atol=1e-5):
        max_deviation: float = (mask_sum - expected_sum).abs().max().item()
        raise ValueError(
            f"Frequency masks do not sum to all-ones. "
            f"Max deviation: {max_deviation:.2e}. "
            "This indicates gaps or overlaps in the frequency band partition."
        )

    # Move to target device.
    return [m.to(device) for m in masks]


def apply_fft_mask(f: Tensor, mask: Tensor) -> Tensor:
    """Extract a frequency band from a feature map using FFT masking.

    Implements: f_hat_i = F^{-1}(F(f) ⊙ M_i)  (paper Section 3.1)

    The FFT is applied over the spatial dimensions (H, W). Each channel
    is processed independently. The mask is broadcast over batch and channel
    dimensions.

    Args:
        f: Real-valued feature map of shape (B, C, H, W).
        mask: Binary frequency mask of shape (H, W), float32.

    Returns:
        Frequency-filtered feature map of shape (B, C, H, W), real-valued.
        The imaginary residual from IFFT is discarded (it is negligible for
        real inputs with symmetric masks).
    """
    # Compute 2D FFT over spatial dimensions.
    # Shape: (B, C, H, W) complex
    F_f: Tensor = torch.fft.fft2(f, dim=(-2, -1))

    # Shift DC to center for mask application.
    # Shape: (B, C, H, W) complex
    F_f_shifted: Tensor = torch.fft.fftshift(F_f, dim=(-2, -1))

    # Apply mask: broadcast (H, W) -> (1, 1, H, W).
    mask_expanded: Tensor = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    F_f_masked: Tensor = F_f_shifted * mask_expanded

    # Inverse shift to restore standard FFT layout.
    F_f_unshifted: Tensor = torch.fft.ifftshift(F_f_masked, dim=(-2, -1))

    # Inverse FFT and take real part.
    # The imaginary component is negligible (~1e-7) for real inputs with
    # symmetric (radial) masks.
    f_hat_i: Tensor = torch.fft.ifft2(F_f_unshifted, dim=(-2, -1)).real

    return f_hat_i


def apply_fft_mask_from_spectrum(
    F_f_shifted: Tensor,
    mask: Tensor,
) -> Tensor:
    """Extract a frequency band from a pre-computed shifted FFT spectrum.

    This is the performance-optimized variant of apply_fft_mask. When
    decomposing a feature map into n bands, the FFT is computed once and
    this function is called n times — avoiding n-1 redundant FFT computations.

    Args:
        F_f_shifted: Pre-computed shifted FFT spectrum of shape (B, C, H, W),
            complex-valued. Obtained via torch.fft.fftshift(torch.fft.fft2(f)).
        mask: Binary frequency mask of shape (H, W), float32.

    Returns:
        Frequency-filtered feature map of shape (B, C, H, W), real-valued.
    """
    # Apply mask: broadcast (H, W) -> (1, 1, H, W).
    mask_expanded: Tensor = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    F_f_masked: Tensor = F_f_shifted * mask_expanded

    # Inverse shift and IFFT.
    F_f_unshifted: Tensor = torch.fft.ifftshift(F_f_masked, dim=(-2, -1))
    f_hat_i: Tensor = torch.fft.ifft2(F_f_unshifted, dim=(-2, -1)).real

    return f_hat_i


def compute_shifted_fft(f: Tensor) -> Tensor:
    """Compute the 2D FFT of a feature map with DC shifted to center.

    This is a convenience function for the FrequencyDecomposer to compute
    the FFT once and reuse it across all frequency bands.

    Args:
        f: Real-valued feature map of shape (B, C, H, W).

    Returns:
        Shifted FFT spectrum of shape (B, C, H, W), complex-valued.
        DC component is at the center of the spatial dimensions.
    """
    F_f: Tensor = torch.fft.fft2(f, dim=(-2, -1))
    F_f_shifted: Tensor = torch.fft.fftshift(F_f, dim=(-2, -1))
    return F_f_shifted


def decompose_into_frequency_bands(
    f: Tensor,
    masks: List[Tensor],
) -> List[Tensor]:
    """Decompose a feature map into n frequency band components.

    Computes the FFT once and applies all n masks efficiently.
    This is the primary entry point for FrequencyDecomposer.decompose().

    Implements: f_hat_i = F^{-1}(F(f) ⊙ M_i) for all i  (paper Section 3.1)

    Reconstruction identity (verified up to float precision):
        Σ_i f_hat_i ≈ f

    Args:
        f: Real-valued feature map of shape (B, C, H, W).
        masks: List of n binary masks, each of shape (H, W).

    Returns:
        List of n tensors, each of shape (B, C, H, W), real-valued.
        The i-th tensor contains only the frequency content of band i.
    """
    # Compute FFT once for all bands.
    F_f_shifted: Tensor = compute_shifted_fft(f)

    # Apply each mask independently.
    components: List[Tensor] = [
        apply_fft_mask_from_spectrum(F_f_shifted, mask) for mask in masks
    ]

    return components


def compose_frequency_bands(components: List[Tensor]) -> Tensor:
    """Reconstruct a feature map by summing frequency band components.

    Implements: f_tilde = Σ_i T(f_hat_i, H', W')  (paper Section 3.1)

    When all components are at the same spatial resolution (no downsampling),
    this is a direct sum. The interpolation T(·) is handled by the caller
    (ResidualQuantizer) when components are at different resolutions.

    Args:
        components: List of n tensors, each of shape (B, C, H, W).
            All tensors must have the same shape.

    Returns:
        Reconstructed feature map of shape (B, C, H, W).
    """
    if not components:
        raise ValueError("Cannot compose an empty list of frequency components.")

    result: Tensor = components[0].clone()
    for component in components[1:]:
        result = result + component

    return result


def visualize_frequency_spectrum(
    f: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Compute log-magnitude frequency spectrum for visualization.

    Used for Figure 4 and Figure 5 in the paper. The log scaling compresses
    the dynamic range for display. Channels are averaged to produce a
    single-channel spatial map.

    Args:
        f: Feature map of shape (B, C, H, W) or (C, H, W).
        eps: Small constant for numerical stability in log computation.

    Returns:
        Log-magnitude spectrum of shape (B, H, W) or (H, W), float32.
        Values are log(1 + |F(f)|) averaged over channels.
    """
    squeeze_output: bool = False
    if f.dim() == 3:
        f = f.unsqueeze(0)  # (1, C, H, W)
        squeeze_output = True

    # Compute shifted FFT magnitude.
    F_f_shifted: Tensor = compute_shifted_fft(f)  # (B, C, H, W) complex
    magnitude: Tensor = F_f_shifted.abs()  # (B, C, H, W) float

    # Average over channels.
    magnitude_avg: Tensor = magnitude.mean(dim=1)  # (B, H, W)

    # Log scaling for visualization (matches standard spectrogram display).
    log_magnitude: Tensor = torch.log(1.0 + magnitude_avg + eps)

    if squeeze_output:
        log_magnitude = log_magnitude.squeeze(0)  # (H, W)

    return log_magnitude


def compute_power_spectral_density(
    f: Tensor,
    num_radial_bins: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """Compute radially averaged power spectral density (PSD).

    Used for the PSD comparison in Appendix B.2 (Table comparing VAR-16
    and NFIG). The PSD characterizes how signal energy is distributed
    across spatial frequencies.

    The power spectrum P(u,v) = |F(u,v)|^2 is averaged over all (u,v)
    pairs with the same radial frequency r = sqrt(u^2 + v^2).

    Args:
        f: Feature map of shape (B, C, H, W).
        num_radial_bins: Number of radial frequency bins. Defaults to
            min(H, W) // 2, which gives one bin per pixel radius.

    Returns:
        Tuple of:
            - radial_freqs: 1D tensor of radial frequency values (bin centers).
            - psd: 1D tensor of mean power at each radial frequency.
    """
    B, C, H, W = f.shape

    if num_radial_bins is None:
        num_radial_bins = min(H, W) // 2

    # Compute power spectrum: |F(f)|^2, averaged over batch and channels.
    F_f_shifted: Tensor = compute_shifted_fft(f)  # (B, C, H, W) complex
    power: Tensor = F_f_shifted.abs() ** 2  # (B, C, H, W)
    power_avg: Tensor = power.mean(dim=(0, 1))  # (H, W)

    # Build radial frequency grid (centered).
    y_coords: Tensor = (
        torch.arange(H, dtype=torch.float32, device=f.device) - H / 2.0
    )
    x_coords: Tensor = (
        torch.arange(W, dtype=torch.float32, device=f.device) - W / 2.0
    )
    Y, X = torch.meshgrid(y_coords, x_coords, indexing="ij")
    radial_freq: Tensor = torch.sqrt(Y ** 2 + X ** 2)  # (H, W)

    # Maximum radial frequency.
    sigma_max: float = math.sqrt((H / 2.0) ** 2 + (W / 2.0) ** 2)

    # Bin edges and centers.
    bin_edges: Tensor = torch.linspace(
        0.0, sigma_max, num_radial_bins + 1, device=f.device
    )
    bin_centers: Tensor = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Radially average the power spectrum.
    psd: Tensor = torch.zeros(num_radial_bins, device=f.device)
    counts: Tensor = torch.zeros(num_radial_bins, device=f.device)

    for bin_idx in range(num_radial_bins):
        low: float = bin_edges[bin_idx].item()
        high: float = bin_edges[bin_idx + 1].item()

        if bin_idx < num_radial_bins - 1:
            in_bin: Tensor = (radial_freq >= low) & (radial_freq < high)
        else:
            in_bin = (radial_freq >= low) & (radial_freq <= high)

        count: int = in_bin.sum().item()
        if count > 0:
            psd[bin_idx] = power_avg[in_bin].mean()
            counts[bin_idx] = count

    return bin_centers, psd


def compute_frequency_keep_score(
    f_real: Tensor,
    f_generated: Tensor,
    weights: Optional[List[float]] = None,
    num_bands: int = 3,
) -> Tuple[float, List[float]]:
    """Compute the Frequency Keep Score (FKS) metric.

    Measures weighted frequency similarity between real and generated feature
    maps across Low/Mid/High frequency bands. Used in Appendix B.2.

    From paper Appendix B.2:
        Weights: [0.57, 0.28, 0.15] for [Low, Mid, High] bands.
        FKS = Σ_k weight_k * similarity_k

    Similarity for each band is computed as the cosine similarity between
    the power spectra of real and generated images in that band.

    Args:
        f_real: Real image feature maps of shape (B, C, H, W).
        f_generated: Generated image feature maps of shape (B, C, H, W).
        weights: Band weights [w_low, w_mid, w_high]. Defaults to
            [0.57, 0.28, 0.15] from paper Appendix B.2.
        num_bands: Number of frequency bands to divide into. Default 3
            (Low, Mid, High) as in the paper.

    Returns:
        Tuple of:
            - fks: Overall weighted FKS score (float, higher is better).
            - band_scores: List of per-band similarity scores [Low, Mid, High].
    """
    if weights is None:
        # From paper Appendix B.2: weights emphasizing structural low-frequency info.
        weights = [0.57, 0.28, 0.15]

    if len(weights) != num_bands:
        raise ValueError(
            f"len(weights)={len(weights)} must equal num_bands={num_bands}"
        )

    if f_real.shape != f_generated.shape:
        raise ValueError(
            f"f_real.shape={f_real.shape} must match f_generated.shape={f_generated.shape}"
        )

    B, C, H, W = f_real.shape

    # Compute power spectra for real and generated.
    F_real_shifted: Tensor = compute_shifted_fft(f_real)
    F_gen_shifted: Tensor = compute_shifted_fft(f_generated)

    power_real: Tensor = F_real_shifted.abs() ** 2  # (B, C, H, W)
    power_gen: Tensor = F_gen_shifted.abs() ** 2  # (B, C, H, W)

    # Average over batch and channels.
    power_real_avg: Tensor = power_real.mean(dim=(0, 1))  # (H, W)
    power_gen_avg: Tensor = power_gen.mean(dim=(0, 1))  # (H, W)

    # Build radial frequency grid.
    y_coords: Tensor = (
        torch.arange(H, dtype=torch.float32, device=f_real.device) - H / 2.0
    )
    x_coords: Tensor = (
        torch.arange(W, dtype=torch.float32, device=f_real.device) - W / 2.0
    )
    Y, X = torch.meshgrid(y_coords, x_coords, indexing="ij")
    radial_freq: Tensor = torch.sqrt(Y ** 2 + X ** 2)  # (H, W)

    sigma_max: float = math.sqrt((H / 2.0) ** 2 + (W / 2.0) ** 2)

    # Divide into equal-width frequency bands.
    band_edges: List[float] = [
        i * sigma_max / num_bands for i in range(num_bands + 1)
    ]

    band_scores: List[float] = []
    for band_idx in range(num_bands):
        low: float = band_edges[band_idx]
        high: float = band_edges[band_idx + 1]

        if band_idx < num_bands - 1:
            in_band: Tensor = (radial_freq >= low) & (radial_freq < high)
        else:
            in_band = (radial_freq >= low) & (radial_freq <= high)

        if in_band.sum() == 0:
            band_scores.append(0.0)
            continue

        # Extract band power values.
        p_real_band: Tensor = power_real_avg[in_band]  # (N_band,)
        p_gen_band: Tensor = power_gen_avg[in_band]  # (N_band,)

        # Cosine similarity between power spectra in this band.
        dot: float = (p_real_band * p_gen_band).sum().item()
        norm_real: float = p_real_band.norm().item()
        norm_gen: float = p_gen_band.norm().item()

        if norm_real < 1e-10 or norm_gen < 1e-10:
            similarity: float = 0.0
        else:
            similarity = dot / (norm_real * norm_gen)
            # Clamp to [0, 1] for percentage display.
            similarity = max(0.0, min(1.0, similarity))

        band_scores.append(similarity)

    # Weighted sum.
    fks: float = sum(w * s for w, s in zip(weights, band_scores))

    return fks, band_scores


def verify_mask_reconstruction(
    f: Tensor,
    masks: List[Tensor],
    atol: float = 1e-4,
) -> bool:
    """Verify that decomposing and recomposing a feature map is lossless.

    Tests the reconstruction identity:
        Σ_i F^{-1}(F(f) ⊙ M_i) ≈ f

    This is a critical correctness check for the frequency decomposition.
    Should be called during unit testing and optionally at init time.

    Args:
        f: Real-valued feature map of shape (B, C, H, W).
        masks: List of n binary masks, each of shape (H, W).
        atol: Absolute tolerance for the reconstruction error. Default 1e-4
            is appropriate for float32 FFT round-trip errors.

    Returns:
        True if reconstruction error is within tolerance, False otherwise.
    """
    components: List[Tensor] = decompose_into_frequency_bands(f, masks)
    f_reconstructed: Tensor = compose_frequency_bands(components)

    max_error: float = (f - f_reconstructed).abs().max().item()
    mean_error: float = (f - f_reconstructed).abs().mean().item()

    is_valid: bool = max_error <= atol

    if not is_valid:
        import warnings
        warnings.warn(
            f"Frequency decomposition reconstruction error exceeds tolerance. "
            f"Max error: {max_error:.2e}, Mean error: {mean_error:.2e}, "
            f"Tolerance: {atol:.2e}. "
            "This may indicate numerical issues with the FFT masks."
        )

    return is_valid


def get_band_token_counts(scale_factors: List[int]) -> List[int]:
    """Compute the number of tokens per frequency band.

    Args:
        scale_factors: List of n scale factors. From config:
            [1, 2, 3, 4, 5, 6, 8, 10, 13, 16].

    Returns:
        List of n integers: [s_i^2 for s_i in scale_factors].
        For the default config: [1, 4, 9, 16, 25, 36, 64, 100, 169, 256].
        Sum = 680 (matches paper Section 4.1).
    """
    return [s * s for s in scale_factors]


def get_cumulative_token_offsets(scale_factors: List[int]) -> List[int]:
    """Compute cumulative token offsets for indexing into the flat token sequence.

    Used by the NFIG Transformer to locate tokens belonging to each frequency
    band within the flattened token sequence of length 680.

    Args:
        scale_factors: List of n scale factors. From config:
            [1, 2, 3, 4, 5, 6, 8, 10, 13, 16].

    Returns:
        List of n+1 integers: [0, s_1^2, s_1^2+s_2^2, ..., total_tokens].
        For the default config: [0, 1, 5, 14, 30, 55, 91, 155, 255, 424, 680].
    """
    token_counts: List[int] = get_band_token_counts(scale_factors)
    offsets: List[int] = [0]
    cumulative: int = 0
    for count in token_counts:
        cumulative += count
        offsets.append(cumulative)
    return offsets
