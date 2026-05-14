"""
Frequency-guided operations for the NFIG framework:
- FrequencyDecomposer: decompose feature maps into frequency bands via FFT
- FrequencyComposer: reconstruct from frequency components
- FrequencyResidualQuantizer: residual quantization with frequency guidance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


def create_frequency_mask(
    h: int, w: int, band_low: float, band_high: float, device: torch.device
) -> torch.Tensor:
    """
    Create a circular frequency mask for a given frequency band.
    Uses radial frequency: f_r = sqrt(u^2 + v^2), where u,v are normalized frequencies.
    band_low, band_high are fractions of max frequency (0 to 1).
    """
    u = torch.fft.fftfreq(h, device=device).view(1, h, 1)
    v = torch.fft.fftfreq(w, device=device).view(1, 1, w)
    f_r = torch.sqrt(u**2 + v**2)
    f_r = f_r / f_r.max()  # normalize to [0, 1]

    mask = ((f_r >= band_low) & (f_r < band_high)).float()
    return mask  # shape: (1, h, w)


class FrequencyDecomposer(nn.Module):
    """
    Decomposes a feature map into n frequency components using FFT-based masking.
    Following the paper: \hat{f}_i = F^{-1}(F(f) ⊙ M_i)
    """

    def __init__(
        self,
        feature_size: int = 16,
        scale_factors: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        num_channels: int = 256,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.scale_factors = scale_factors
        self.num_bands = len(scale_factors)
        self.num_channels = num_channels

        # Scale factors represent the resolution: h_i = s, w_i = s
        total_area = sum(s * s for s in scale_factors)

        self.register_buffer("band_boundaries", self._compute_band_boundaries(total_area))

    def _compute_band_boundaries(self, total_area: float) -> torch.Tensor:
        """Compute frequency band boundaries σ_i based on equation in paper.
        Scale factors s_i directly represent the resolution h_i = s_i, w_i = s_i.
        σ_i = σ_{i-1} + (h_i * w_i) / (Σ_j h_j * w_j) * σ_max
        """
        boundaries = [0.0]
        for i, s in enumerate(self.scale_factors):
            area = float(s * s)  # h_i * w_i = s * s
            increment = area / total_area
            boundaries.append(boundaries[-1] + increment)
        return torch.tensor(boundaries)

    def get_masks(self, device: torch.device) -> List[torch.Tensor]:
        """Generate frequency masks for all bands."""
        masks = []
        for i in range(self.num_bands):
            mask = create_frequency_mask(
                self.feature_size,
                self.feature_size,
                self.band_boundaries[i].item(),
                self.band_boundaries[i + 1].item(),
                device,
            )
            masks.append(mask)
        return masks

    def forward(self, f: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            f: Feature map of shape (B, C, H', W')
        Returns:
            List of frequency components, each shape (B, C, H', W')
        """
        B, C, H, W = f.shape
        F_f = torch.fft.fft2(f)
        masks = self.get_masks(f.device)

        components = []
        for mask in masks:
            M_i = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            F_hat_i = F_f * M_i.to(F_f.dtype)
            f_hat_i = torch.fft.ifft2(F_hat_i).real
            components.append(f_hat_i)

        return components


class FrequencyComposer(nn.Module):
    """
    Reconstructs the full feature map from frequency components.
    Following the paper: \tilde{f} = Σ_i T(\hat{f}_i, H', W')
    where T is interpolation to the target size.
    """

    def __init__(self, target_size: int = 16):
        super().__init__()
        self.target_size = target_size

    def forward(self, components: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            components: List of frequency component tensors.
                        Each can have different spatial dimensions.
        Returns:
            Reconstructed feature map of shape (B, C, H', W')
        """
        target = (self.target_size, self.target_size)
        resized = []
        for comp in components:
            if comp.shape[-2:] != target:
                comp = F.interpolate(comp, size=target, mode="bilinear", align_corners=False)
            resized.append(comp)
        return torch.stack(resized, dim=0).sum(dim=0)


class FrequencyResidualQuantizer(nn.Module):
    """
    Frequency-guided Residual Quantizer.
    Implements the residual token extraction and vector quantization
    described in Section 3.1.2 of the paper.

    The residual R_i accumulates the error between the accumulated frequency
    components up to level i and the learned features.

    R_0 = 0
    For i >= 0:
        v_i = argmin ||(R_{i-1} + \hat{f}_i) - upsample(v_i)||^2
        R_i = R_{i-1} + (\hat{f}_i - upsample(v_i^q))
    """

    def __init__(
        self,
        feature_size: int = 16,
        scale_factors: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        codebook_size: int = 4096,
        latent_dim: int = 32,
        num_channels: int = 256,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.scale_factors = scale_factors
        self.num_bands = len(scale_factors)
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.num_channels = num_channels

        # Learnable codebook Z ∈ R^{K × C}
        self.codebook = nn.Parameter(
            torch.randn(codebook_size, latent_dim) * 0.02
        )

        # Projection layers: from channel dim to latent dim and back
        self.enc_proj = nn.Conv2d(num_channels, latent_dim, kernel_size=1)
        self.dec_proj = nn.Conv2d(latent_dim, num_channels, kernel_size=1)

        # Small CNNs per scale to predict v_i from the residual + frequency component
        # Scale factor s_i directly gives the resolution h_i = w_i = s_i
        self.scale_encoders = nn.ModuleList()
        self.scale_decoders = nn.ModuleList()
        for s in scale_factors:
            self.scale_encoders.append(
                nn.Sequential(
                    nn.Conv2d(num_channels, latent_dim, kernel_size=3, padding=1),
                    nn.GroupNorm(8, latent_dim),
                    nn.ReLU(),
                    nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1),
                )
            )
            self.scale_decoders.append(
                nn.Sequential(
                    nn.Conv2d(latent_dim, num_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(8, num_channels),
                    nn.ReLU(),
                    nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
                )
            )

    def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Vector quantization: find nearest codebook entry for each spatial position.
        Args:
            z: (B, C_latent, h, w) continuous latent
        Returns:
            z_q: quantized latent
            tokens: codebook indices (B, h, w)
            commit_loss: commitment loss
        """
        B, C, h, w = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, C)  # (B*h*w, C)

        # Compute distances to codebook
        codebook = self.codebook  # (K, C)
        z_flat_norm = z_flat.pow(2).sum(dim=1, keepdim=True)  # (B*h*w, 1)
        cb_norm = codebook.pow(2).sum(dim=1)  # (K,)
        dist = (
            z_flat_norm
            + cb_norm.unsqueeze(0)
            - 2 * z_flat @ codebook.t()
        )  # (B*h*w, K)

        # Find nearest
        min_indices = dist.argmin(dim=1)  # (B*h*w,)
        z_q_flat = self.codebook[min_indices]  # (B*h*w, C)

        # Commitment loss
        commit_loss = F.mse_loss(z_flat, z_q_flat.detach())

        # Straight-through estimator
        z_q_flat = z_flat + (z_q_flat - z_flat).detach()

        z_q = z_q_flat.reshape(B, h, w, C).permute(0, 3, 1, 2)
        tokens = min_indices.reshape(B, h, w)

        return z_q, tokens, commit_loss

    def forward(
        self, frequency_components: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """
        Args:
            frequency_components: List of \hat{f}_i, each shape (B, C, H', W') where H'=W'=feature_size
        Returns:
            quantized_components: List of v_i^q at each scale, shapes vary (h_i, w_i)
            all_tokens: List of token tensors, shapes (B, h_i, w_i)
            total_commit_loss: Sum of commitment losses
        """
        B = frequency_components[0].shape[0]
        device = frequency_components[0].device
        target_hw = (self.feature_size, self.feature_size)

        residual = torch.zeros(
            B, self.num_channels, self.feature_size, self.feature_size, device=device
        )
        quantized_components = []
        all_tokens = []
        total_commit_loss = torch.tensor(0.0, device=device)

        for i, (s, f_hat_i) in enumerate(zip(self.scale_factors, frequency_components)):
            # Scale factor s directly gives the resolution: h_i = s, w_i = s
            h_i = s
            w_i = s

            # Accumulate: R_{i-1} + \hat{f}_i
            accumulated = residual + f_hat_i

            # Downsample accumulated to resolution (h_i, w_i)
            if (h_i, w_i) != target_hw:
                accumulated_ds = F.interpolate(
                    accumulated, size=(h_i, w_i), mode="bilinear", align_corners=False
                )
            else:
                accumulated_ds = accumulated

            # Encode to latent: v_i = argmin ||(R_{i-1} + \hat{f}_i) - upsample(v_i)||^2
            v_i = self.scale_encoders[i](accumulated_ds)  # (B, latent_dim, h_i, w_i)

            # Vector quantize
            v_i_q, tokens_i, commit_loss = self.quantize(v_i)

            # Decode back to channel dimension
            v_i_decoded = self.scale_decoders[i](v_i_q)  # (B, num_channels, h_i, w_i)

            # Upsample decoded to target size for residual computation
            if (h_i, w_i) != target_hw:
                v_i_upsampled = F.interpolate(
                    v_i_decoded, size=target_hw, mode="bilinear", align_corners=False
                )
            else:
                v_i_upsampled = v_i_decoded

            # Compute residual: R_i = R_{i-1} + (\hat{f}_i - upsample(v_i^q))
            residual = residual + (f_hat_i - v_i_upsampled)

            quantized_components.append(v_i_decoded)
            all_tokens.append(tokens_i)
            total_commit_loss = total_commit_loss + commit_loss

        return quantized_components, all_tokens, total_commit_loss


class VectorQuantizer(nn.Module):
    """Simple vector quantizer for the baseline (non-frequency) variant."""

    def __init__(self, codebook_size: int = 4096, latent_dim: int = 32):
        super().__init__()
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.codebook = nn.Parameter(torch.randn(codebook_size, latent_dim) * 0.02)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, h, w = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, C)

        codebook = self.codebook
        z_flat_norm = z_flat.pow(2).sum(dim=1, keepdim=True)
        cb_norm = codebook.pow(2).sum(dim=1)
        dist = z_flat_norm + cb_norm.unsqueeze(0) - 2 * z_flat @ codebook.t()

        min_indices = dist.argmin(dim=1)
        z_q_flat = self.codebook[min_indices]
        commit_loss = F.mse_loss(z_flat, z_q_flat.detach())
        z_q_flat = z_flat + (z_q_flat - z_flat).detach()

        z_q = z_q_flat.reshape(B, h, w, C).permute(0, 3, 1, 2)
        tokens = min_indices.reshape(B, h, w)

        return z_q, tokens, commit_loss


def create_radial_frequency_mask(
    size: int, scaling_factor: int, band_index: int, num_bands: int, device: torch.device
) -> torch.Tensor:
    """
    Alternative frequency mask creation based on the paper's frequency division strategy.
    The frequency bands are divided proportionally to the area of each scale.
    σ_i = σ_{i-1} + (h_i * w_i) / (Σ_j h_j * w_j) * σ_max
    where h_i = s_i, w_i = s_i (scale factor directly gives resolution).
    """
    scales = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    areas = [s * s for s in scales]
    total_area = sum(areas)

    boundaries = [0.0]
    for area in areas:
        boundaries.append(boundaries[-1] + area / total_area)

    band_low = boundaries[band_index]
    band_high = boundaries[band_index + 1]

    return create_frequency_mask(size, size, band_low, band_high, device)
