import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_frequency_masks(
    h: int,
    w: int,
    scale_factors: List[int],
    device: torch.device = torch.device("cpu"),
) -> List[torch.Tensor]:
    """
    Build binary frequency masks M_i for each frequency band.

    The frequency space is divided proportionally to the number of tokens
    at each scale level. For scale factor s_i, the band covers
    [sigma_{i-1}, sigma_i) where sigma_i is determined by the cumulative
    token count ratio (Eq. 6 in paper).

    Args:
        h, w: spatial dimensions of the feature map
        scale_factors: list of scale factors [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
        device: target device

    Returns:
        List of masks, each of shape (h, w), values in {0, 1}
    """
    total_tokens = sum(s * s for s in scale_factors)
    sigma_max = math.sqrt((h // 2) ** 2 + (w // 2) ** 2)

    # Build frequency coordinate grid (centered, after fftshift)
    fy = torch.fft.fftfreq(h, d=1.0) * h  # [-h/2, h/2)
    fx = torch.fft.fftfreq(w, d=1.0) * w
    fy = torch.fft.fftshift(torch.tensor(fy, dtype=torch.float32))
    fx = torch.fft.fftshift(torch.tensor(fx, dtype=torch.float32))
    grid_y, grid_x = torch.meshgrid(fy, fx, indexing="ij")
    freq_radius = torch.sqrt(grid_y ** 2 + grid_x ** 2)  # (h, w)

    masks = []
    sigma_prev = 0.0
    cumulative_tokens = 0

    for i, s in enumerate(scale_factors):
        cumulative_tokens += s * s
        if i < len(scale_factors) - 1:
            sigma_i = (cumulative_tokens / total_tokens) * sigma_max
        else:
            sigma_i = sigma_max + 1.0  # include all remaining

        mask = ((freq_radius >= sigma_prev) & (freq_radius < sigma_i)).float()
        masks.append(mask.to(device))
        sigma_prev = sigma_i

    return masks


class FrequencyDecomposer(nn.Module):
    """
    Decomposes a feature map f into n frequency components via FFT masking.

    f_hat_i = IFFT(FFT(f) * M_i)  (Eq. 1 in paper)
    """

    def __init__(self, h: int, w: int, scale_factors: List[int]):
        super().__init__()
        self.h = h
        self.w = w
        self.scale_factors = scale_factors
        self.n = len(scale_factors)
        # Register masks as buffers (not parameters)
        masks = build_frequency_masks(h, w, scale_factors)
        for i, mask in enumerate(masks):
            self.register_buffer(f"mask_{i}", mask)

    def get_mask(self, i: int) -> torch.Tensor:
        return getattr(self, f"mask_{i}")

    def forward(self, f: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            f: (B, C, H', W') feature map

        Returns:
            List of n frequency components, each (B, C, H', W')
        """
        B, C, H, W = f.shape
        # 2D FFT over spatial dims
        F_freq = torch.fft.fft2(f, norm="ortho")  # (B, C, H, W) complex
        F_shifted = torch.fft.fftshift(F_freq, dim=(-2, -1))

        components = []
        for i in range(self.n):
            mask = self.get_mask(i)  # (H, W)
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            F_masked = F_shifted * mask
            F_unshifted = torch.fft.ifftshift(F_masked, dim=(-2, -1))
            f_hat = torch.fft.ifft2(F_unshifted, norm="ortho").real
            components.append(f_hat)

        return components


class FrequencyComposer(nn.Module):
    """
    Reconstructs feature map from frequency components via interpolation and sum.

    f_tilde = sum_i T(f_hat_i, H', W')  (Eq. 2 in paper)
    """

    def __init__(self, target_h: int, target_w: int):
        super().__init__()
        self.target_h = target_h
        self.target_w = target_w

    def forward(self, components: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            components: list of (B, C, h_i, w_i) tensors at various resolutions

        Returns:
            (B, C, H', W') reconstructed feature map
        """
        result = None
        for comp in components:
            if comp.shape[-2] != self.target_h or comp.shape[-1] != self.target_w:
                comp_up = F.interpolate(
                    comp,
                    size=(self.target_h, self.target_w),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                comp_up = comp
            result = comp_up if result is None else result + comp_up
        return result


def downsample_feature(f: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Downsample feature map to (target_h, target_w) via bilinear interpolation."""
    if f.shape[-2] == target_h and f.shape[-1] == target_w:
        return f
    return F.interpolate(f, size=(target_h, target_w), mode="bilinear", align_corners=False)


def upsample_feature(f: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Upsample feature map to (target_h, target_w) via bilinear interpolation."""
    if f.shape[-2] == target_h and f.shape[-1] == target_w:
        return f
    return F.interpolate(f, size=(target_h, target_w), mode="bilinear", align_corners=False)
