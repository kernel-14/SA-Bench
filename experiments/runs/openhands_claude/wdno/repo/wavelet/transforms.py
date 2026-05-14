"""
Wavelet transform utilities for WDNO.

1D PDE data (e.g. Burgers', advection, compressible NS):
  - Data shape: [batch, T, X]
  - Apply 2D DWT over (T, X) using pytorch_wavelets
  - Wavelet: bior2.4, mode: periodization
  - Result: 4 coefficient arrays each of shape [batch, T//2+1, X//2+1] (approx)

2D PDE data (e.g. incompressible fluid, ERA5):
  - Data shape: [batch, T, H, W]
  - Apply 3D DWT over (T, H, W) using ptwt
  - Wavelet: bior1.3, mode: zero
  - Result: 8 coefficient arrays each of shape [batch, T//2+1, H//2+1, W//2+1] (approx)

The decomposition uses a single level (l0 = L), yielding:
  - 1 coarse (LL/LLL) coefficient array
  - 3 detail (LH, HL, HH) arrays for 2D
  - 7 detail arrays for 3D
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1D wavelet transform (applied along a single spatial axis)
# Used for initial conditions and 1D target states before concatenation
# ---------------------------------------------------------------------------

def apply_wavelet_1d(x: torch.Tensor, wavelet: str = "bior2.4", mode: str = "periodization") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Single-level 1D DWT along the last dimension.

    Args:
        x: [batch, N] or [batch, C, N]
        wavelet: wavelet name
        mode: padding mode

    Returns:
        (cA, cD): approximation and detail coefficients
    """
    try:
        import pywt
        import pywt.data
    except ImportError:
        raise ImportError("PyWavelets (pywt) is required for 1D transforms.")

    w = pywt.Wavelet(wavelet)
    lo = torch.tensor(w.dec_lo, dtype=x.dtype, device=x.device).flip(0)
    hi = torch.tensor(w.dec_hi, dtype=x.dtype, device=x.device).flip(0)

    squeeze = x.dim() == 2
    if squeeze:
        x = x.unsqueeze(1)  # [batch, 1, N]

    batch, C, N = x.shape
    filter_len = len(lo)

    # Pad for periodization
    pad = filter_len - 1
    x_pad = F.pad(x, (pad // 2, pad - pad // 2), mode="circular")

    lo_f = lo.view(1, 1, -1).expand(C, 1, -1)
    hi_f = hi.view(1, 1, -1).expand(C, 1, -1)

    cA = F.conv1d(x_pad, lo_f, stride=2, groups=C)
    cD = F.conv1d(x_pad, hi_f, stride=2, groups=C)

    if squeeze:
        cA = cA.squeeze(1)
        cD = cD.squeeze(1)

    return cA, cD


def inverse_wavelet_1d(cA: torch.Tensor, cD: torch.Tensor, wavelet: str = "bior2.4") -> torch.Tensor:
    """Single-level 1D IDWT."""
    try:
        import pywt
    except ImportError:
        raise ImportError("PyWavelets (pywt) is required.")

    w = pywt.Wavelet(wavelet)
    lo = torch.tensor(w.rec_lo, dtype=cA.dtype, device=cA.device)
    hi = torch.tensor(w.rec_hi, dtype=cA.dtype, device=cA.device)

    squeeze = cA.dim() == 2
    if squeeze:
        cA = cA.unsqueeze(1)
        cD = cD.unsqueeze(1)

    batch, C, N = cA.shape
    filter_len = len(lo)

    lo_f = lo.view(1, 1, -1).expand(C, 1, -1)
    hi_f = hi.view(1, 1, -1).expand(C, 1, -1)

    # Upsample
    cA_up = F.conv_transpose1d(cA, lo_f, stride=2, groups=C)
    cD_up = F.conv_transpose1d(cD, hi_f, stride=2, groups=C)

    rec = cA_up + cD_up
    # Trim to original size
    trim = filter_len - 2
    if trim > 0:
        rec = rec[..., trim // 2: -(trim - trim // 2)]

    if squeeze:
        rec = rec.squeeze(1)

    return rec


# ---------------------------------------------------------------------------
# 2D wavelet transform (for 1D PDE data treated as time-space 2D)
# Uses pytorch_wavelets for bior2.4 with periodization
# ---------------------------------------------------------------------------

class WaveletTransform2D(nn.Module):
    """
    Single-level 2D DWT using pytorch_wavelets.
    Applied to data of shape [batch, C, T, X].
    Returns 4 coefficient subbands: LL, LH, HL, HH.
    """

    def __init__(self, wavelet: str = "bior2.4", mode: str = "periodization"):
        super().__init__()
        self.wavelet = wavelet
        self.mode = mode
        self._dwt = None
        self._idwt = None

    def _get_dwt(self, device):
        if self._dwt is None:
            from pytorch_wavelets import DWTForward, DWTInverse
            self._dwt = DWTForward(J=1, wave=self.wavelet, mode=self.mode).to(device)
            self._idwt = DWTInverse(wave=self.wavelet, mode=self.mode).to(device)
        return self._dwt, self._idwt

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: [batch, C, T, X]
        Returns:
            list of 4 tensors each [batch, C, T', X']:
            [LL, LH, HL, HH]
        """
        dwt, _ = self._get_dwt(x.device)
        # pytorch_wavelets expects [batch, C, H, W]
        yl, yh = dwt(x)  # yl: [B,C,T',X'], yh: list of [B,C,3,T',X']
        # yh[0] has shape [B, C, 3, T', X'] for LH, HL, HH
        lh = yh[0][:, :, 0]
        hl = yh[0][:, :, 1]
        hh = yh[0][:, :, 2]
        return [yl, lh, hl, hh]

    def inverse(self, coeffs: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            coeffs: [LL, LH, HL, HH] each [batch, C, T', X']
        Returns:
            x: [batch, C, T, X]
        """
        _, idwt = self._get_dwt(coeffs[0].device)
        yl = coeffs[0]
        yh_tensor = torch.stack([coeffs[1], coeffs[2], coeffs[3]], dim=2)  # [B,C,3,T',X']
        return idwt((yl, [yh_tensor]))


def apply_wavelet_2d(x: torch.Tensor, wavelet: str = "bior2.4", mode: str = "periodization") -> torch.Tensor:
    """
    Apply 2D DWT and concatenate all 4 subbands along channel dim.

    Args:
        x: [batch, C, T, X]
    Returns:
        [batch, 4*C, T', X']
    """
    from pytorch_wavelets import DWTForward
    dwt = DWTForward(J=1, wave=wavelet, mode=mode).to(x.device)
    yl, yh = dwt(x)
    lh = yh[0][:, :, 0]
    hl = yh[0][:, :, 1]
    hh = yh[0][:, :, 2]
    return torch.cat([yl, lh, hl, hh], dim=1)


def inverse_wavelet_2d(x: torch.Tensor, wavelet: str = "bior2.4", mode: str = "periodization", n_channels: int = 1) -> torch.Tensor:
    """
    Inverse 2D DWT from concatenated subbands.

    Args:
        x: [batch, 4*C, T', X']
        n_channels: C (number of original channels)
    Returns:
        [batch, C, T, X]
    """
    from pytorch_wavelets import DWTInverse
    idwt = DWTInverse(wave=wavelet, mode=mode).to(x.device)
    C = n_channels
    yl = x[:, :C]
    lh = x[:, C:2*C]
    hl = x[:, 2*C:3*C]
    hh = x[:, 3*C:4*C]
    yh_tensor = torch.stack([lh, hl, hh], dim=2)
    return idwt((yl, [yh_tensor]))


# ---------------------------------------------------------------------------
# 3D wavelet transform (for 2D PDE data with time dimension)
# Uses ptwt for bior1.3 with zero padding
# ---------------------------------------------------------------------------

class WaveletTransform3D(nn.Module):
    """
    Single-level 3D DWT using ptwt (Pytorch Wavelet Toolbox).
    Applied to data of shape [batch, C, T, H, W].
    Returns 8 coefficient subbands.
    """

    def __init__(self, wavelet: str = "bior1.3", mode: str = "zero"):
        super().__init__()
        self.wavelet = wavelet
        self.mode = mode

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: [batch, C, T, H, W]
        Returns:
            list of 8 tensors each [batch, C, T', H', W']:
            [LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH]
        """
        import ptwt
        import pywt
        w = pywt.Wavelet(self.wavelet)
        # ptwt.wavedec3 returns (approx, details_list)
        # For J=1: details_list has one dict with keys 'aad','ada','add','daa','dad','dda','ddd'
        coeffs = ptwt.wavedec3(x, w, level=1, mode=self.mode)
        approx = coeffs[0]  # LLL: [B, C, T', H', W']
        detail_dict = coeffs[1]
        # Standard ordering: aad=LLH, ada=LHL, add=LHH, daa=HLL, dad=HLH, dda=HHL, ddd=HHH
        detail_list = [
            detail_dict['aad'],
            detail_dict['ada'],
            detail_dict['add'],
            detail_dict['daa'],
            detail_dict['dad'],
            detail_dict['dda'],
            detail_dict['ddd'],
        ]
        return [approx] + detail_list

    def inverse(self, coeffs: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            coeffs: list of 8 tensors [LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH]
        Returns:
            x: [batch, C, T, H, W]
        """
        import ptwt
        import pywt
        w = pywt.Wavelet(self.wavelet)
        approx = coeffs[0]
        detail_dict = {
            'aad': coeffs[1],
            'ada': coeffs[2],
            'add': coeffs[3],
            'daa': coeffs[4],
            'dad': coeffs[5],
            'dda': coeffs[6],
            'ddd': coeffs[7],
        }
        return ptwt.waverec3([approx, detail_dict], w)


def apply_wavelet_3d(x: torch.Tensor, wavelet: str = "bior1.3", mode: str = "zero") -> torch.Tensor:
    """
    Apply 3D DWT and concatenate all 8 subbands along channel dim.

    Args:
        x: [batch, C, T, H, W]
    Returns:
        [batch, 8*C, T', H', W']
    """
    import ptwt
    import pywt
    w = pywt.Wavelet(wavelet)
    coeffs = ptwt.wavedec3(x, w, level=1, mode=mode)
    approx = coeffs[0]
    detail_dict = coeffs[1]
    detail_list = [
        detail_dict['aad'],
        detail_dict['ada'],
        detail_dict['add'],
        detail_dict['daa'],
        detail_dict['dad'],
        detail_dict['dda'],
        detail_dict['ddd'],
    ]
    all_coeffs = [approx] + detail_list
    return torch.cat(all_coeffs, dim=1)


def inverse_wavelet_3d(x: torch.Tensor, wavelet: str = "bior1.3", mode: str = "zero", n_channels: int = 1) -> torch.Tensor:
    """
    Inverse 3D DWT from concatenated subbands.

    Args:
        x: [batch, 8*C, T', H', W']
        n_channels: C
    Returns:
        [batch, C, T, H, W]
    """
    import ptwt
    import pywt
    w = pywt.Wavelet(wavelet)
    C = n_channels
    approx = x[:, :C]
    detail_dict = {
        'aad': x[:, C:2*C],
        'ada': x[:, 2*C:3*C],
        'add': x[:, 3*C:4*C],
        'daa': x[:, 4*C:5*C],
        'dad': x[:, 5*C:6*C],
        'dda': x[:, 6*C:7*C],
        'ddd': x[:, 7*C:8*C],
    }
    return ptwt.waverec3([approx, detail_dict], w)


# ---------------------------------------------------------------------------
# Convenience wrappers that handle the full data preparation pipeline
# ---------------------------------------------------------------------------

class WaveletTransform1D(nn.Module):
    """Wrapper for 1D wavelet transform (applied along spatial axis only)."""

    def __init__(self, wavelet: str = "bior2.4", mode: str = "periodization"):
        super().__init__()
        self.wavelet = wavelet
        self.mode = mode

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: [batch, N] → (cA, cD) each [batch, N//2+...]"""
        return apply_wavelet_1d(x, self.wavelet, self.mode)

    def inverse(self, cA: torch.Tensor, cD: torch.Tensor) -> torch.Tensor:
        return inverse_wavelet_1d(cA, cD, self.wavelet)


def pad_to_match(low_res: torch.Tensor, high_res: torch.Tensor) -> torch.Tensor:
    """
    Duplicate low-resolution wavelet coefficients to match high-resolution size.
    Used in SRM training to align low-res and high-res coefficient shapes.
    Handles odd-dimension boundary by duplicating the last element.

    Args:
        low_res: [batch, C, ...] low-resolution coefficients
        high_res: [batch, C, ...] high-resolution coefficients (target shape)
    Returns:
        low_res_padded: same shape as high_res
    """
    target_shape = high_res.shape[2:]
    current_shape = low_res.shape[2:]

    # Compute repeat factors per dimension
    result = low_res
    for dim_idx, (cur, tgt) in enumerate(zip(current_shape, target_shape)):
        spatial_dim = dim_idx + 2  # account for batch and channel dims
        repeat_factor = math.ceil(tgt / cur)
        result = result.repeat_interleave(repeat_factor, dim=spatial_dim)
        # Trim to target size
        slices = [slice(None)] * result.dim()
        slices[spatial_dim] = slice(0, tgt)
        result = result[slices]

    return result
