"""
Wavelet transform utilities for WDNO.

Implements 1D, 2D, and 3D discrete wavelet transforms using:
- pytorch_wavelets for 1D/2D transforms (bior2.4 with periodization mode)
- ptwt (pytorch wavelet toolbox) for 3D transforms (bior1.3 with zero mode)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Union


class WaveletTransform1D(nn.Module):
    """
    1D Wavelet Transform module for 1D PDE data.
    """
    
    def __init__(self, wavelet="bior2.4", mode="periodization", level=1):
        super().__init__()
        self.wavelet = wavelet
        self.mode = mode
        self.level = level
        self._dwt = None
        self._idwt = None
    
    def _get_transforms(self, device):
        if self._dwt is None:
            try:
                import pytorch_wavelets as pw
                self._dwt = pw.DWT1DForward(wave=self.wavelet, J=self.level, mode=self.mode).to(device)
                self._idwt = pw.DWT1DInverse(wave=self.wavelet, mode=self.mode).to(device)
            except ImportError:
                raise ImportError("pytorch_wavelets is required for 1D wavelet transforms")
        return self._dwt, self._idwt
    
    def forward(self, x):
        dwt, _ = self._get_transforms(x.device)
        return dwt(x)
    
    def inverse(self, yl, yh):
        _, idwt = self._get_transforms(yl.device)
        return idwt((yl, yh))


class WaveletTransform2D(nn.Module):
    """
    2D Wavelet Transform module.
    
    For 1D PDE data (Burgers, NS) of shape (B, T, X), treats the
    spatiotemporal data as a 2D image and applies 2D DWT.
    """
    
    def __init__(self, wavelet="bior2.4", mode="periodization", level=1):
        super().__init__()
        self.wavelet = wavelet
        self.mode = mode
        self.level = level
        self._dwt = None
        self._idwt = None
    
    def _get_transforms(self, device):
        if self._dwt is None:
            try:
                import pytorch_wavelets as pw
                self._dwt = pw.DWTForward(wave=self.wavelet, J=self.level, mode=self.mode).to(device)
                self._idwt = pw.DWTInverse(wave=self.wavelet, mode=self.mode).to(device)
            except ImportError:
                raise ImportError("pytorch_wavelets is required for 2D wavelet transforms")
        return self._dwt, self._idwt
    
    def forward(self, x):
        """
        Apply 2D DWT.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            yl: Low-frequency coefficients of shape (B, C, H//2, W//2)
            yh: List of high-frequency coefficients, each (B, C, 3, H//2, W//2)
        """
        dwt, _ = self._get_transforms(x.device)
        return dwt(x)
    
    def inverse(self, yl, yh):
        """Apply inverse 2D DWT."""
        _, idwt = self._get_transforms(yl.device)
        return idwt((yl, yh))


class WaveletTransform3D(nn.Module):
    """
    3D Wavelet Transform module for 2D fluid data.
    
    For 2D fluid data of shape (B, T, H, W), applies 3D DWT.
    Uses bior1.3 wavelet with zero padding mode.
    """
    
    def __init__(self, wavelet="bior1.3", mode="zero", level=1):
        super().__init__()
        self.wavelet = wavelet
        self.mode = mode
        self.level = level
        self._wavelet_obj = None
        self._ptwt = None
    
    def _get_ptwt(self):
        if self._ptwt is None:
            try:
                import ptwt
                import pywt
                self._wavelet_obj = pywt.Wavelet(self.wavelet)
                self._ptwt = ptwt
            except ImportError:
                raise ImportError("ptwt is required for 3D wavelet transforms")
        return self._ptwt, self._wavelet_obj
    
    def forward(self, x):
        """
        Apply 3D DWT.
        
        Args:
            x: Input tensor of shape (B, C, T, H, W)
        
        Returns:
            coeffs: List of wavelet coefficients
        """
        ptwt, wavelet_obj = self._get_ptwt()
        return ptwt.wavedec3(x, wavelet_obj, level=self.level, mode=self.mode)
    
    def inverse(self, coeffs):
        """Apply inverse 3D DWT."""
        ptwt, wavelet_obj = self._get_ptwt()
        return ptwt.waverec3(coeffs, wavelet_obj)


def pack_wavelet_coeffs_2d(yl, yh):
    """
    Pack 2D wavelet coefficients into a single tensor.
    
    Args:
        yl: Low-frequency coefficients (B, C, H, W)
        yh: List of high-frequency coefficients, each (B, C, 3, H, W)
    
    Returns:
        packed: Packed tensor (B, 4*C, H, W)
    """
    B, C, H, W = yl.shape
    yh_reshaped = yh[0].reshape(B, C * 3, H, W)
    return torch.cat([yl, yh_reshaped], dim=1)


def unpack_wavelet_coeffs_2d(packed, n_channels):
    """
    Unpack 2D wavelet coefficients from a single tensor.
    
    Args:
        packed: Packed tensor (B, 4*C, H, W)
        n_channels: Number of original channels C
    
    Returns:
        yl: Low-frequency coefficients (B, C, H, W)
        yh: List of high-frequency coefficients
    """
    B, _, H, W = packed.shape
    yl = packed[:, :n_channels]
    yh_flat = packed[:, n_channels:]
    yh = [yh_flat.reshape(B, n_channels, 3, H, W)]
    return yl, yh


def pack_wavelet_coeffs_3d(coeffs_list):
    """
    Pack 3D wavelet coefficients into a single tensor.
    
    Args:
        coeffs_list: List from wavedec3, [ll, {hh_dict}]
    
    Returns:
        packed: Packed tensor (B, 8*C, T//2, H//2, W//2)
    """
    ll = coeffs_list[0]
    hh_dict = coeffs_list[1]
    
    # Stack all 7 high-freq subbands in sorted order
    hh_list = [hh_dict[key] for key in sorted(hh_dict.keys())]
    hh = torch.stack(hh_list, dim=2)  # (B, C, 7, T//2, H//2, W//2)
    B, C, _, T, H, W = hh.shape
    hh = hh.reshape(B, C * 7, T, H, W)
    
    return torch.cat([ll, hh], dim=1)


def unpack_wavelet_coeffs_3d(packed, n_channels):
    """
    Unpack 3D wavelet coefficients from a single tensor.
    
    Args:
        packed: Packed tensor (B, 8*C, T//2, H//2, W//2)
        n_channels: Number of original channels C
    
    Returns:
        coeffs_list: List compatible with waverec3
    """
    B, _, T, H, W = packed.shape
    ll = packed[:, :n_channels]
    hh_flat = packed[:, n_channels:]
    hh = hh_flat.reshape(B, n_channels, 7, T, H, W)
    
    # Reconstruct dict with sorted keys
    keys = ["aad", "ada", "add", "daa", "dad", "dda", "ddd"]
    hh_dict = {keys[i]: hh[:, :, i] for i in range(7)}
    
    return [ll, hh_dict]


def apply_2d_wavelet_transform(data, wavelet="bior2.4", mode="periodization"):
    """
    Apply 2D wavelet transform to 1D PDE data.
    
    For 1D PDE data of shape (B, T, X), treats it as a 2D image (T, X)
    and applies 2D DWT. Returns concatenated wavelet coefficients.
    
    Args:
        data: Input tensor of shape (B, T, X) or (B, C, T, X)
        wavelet: Wavelet basis
        mode: Padding mode
    
    Returns:
        coeffs: Concatenated wavelet coefficients of shape (B, 4*C, T//2, X//2)
        raw_coeffs: Tuple (yl, yh) for inverse transform
    """
    import pytorch_wavelets as pw
    dwt = pw.DWTForward(wave=wavelet, J=1, mode=mode).to(data.device)
    
    if data.dim() == 3:
        # (B, T, X) -> (B, 1, T, X)
        data = data.unsqueeze(1)
    
    yl, yh = dwt(data)
    coeffs = pack_wavelet_coeffs_2d(yl, yh)
    return coeffs, (yl, yh)


def apply_3d_wavelet_transform(data, wavelet="bior1.3", mode="zero"):
    """
    Apply 3D wavelet transform to 2D fluid data.
    
    Args:
        data: Input tensor of shape (B, C, T, H, W)
        wavelet: Wavelet basis
        mode: Padding mode
    
    Returns:
        coeffs: Concatenated wavelet coefficients
        raw_coeffs: Raw wavelet coefficients for inverse transform
    """
    import ptwt
    import pywt
    wavelet_obj = pywt.Wavelet(wavelet)
    
    coeffs_list = ptwt.wavedec3(data, wavelet_obj, level=1, mode=mode)
    coeffs = pack_wavelet_coeffs_3d(coeffs_list)
    return coeffs, coeffs_list


def inverse_2d_wavelet_transform(yl, yh, wavelet="bior2.4", mode="periodization"):
    """Apply inverse 2D wavelet transform."""
    import pytorch_wavelets as pw
    idwt = pw.DWTInverse(wave=wavelet, mode=mode).to(yl.device)
    return idwt((yl, yh))


def inverse_3d_wavelet_transform(coeffs_list, wavelet="bior1.3"):
    """Apply inverse 3D wavelet transform."""
    import ptwt
    import pywt
    wavelet_obj = pywt.Wavelet(wavelet)
    return ptwt.waverec3(coeffs_list, wavelet_obj)


def downsample_data(data, factor=2):
    """
    Downsample data by a given factor.
    
    Args:
        data: Input tensor
        factor: Downsampling factor (default: 2)
    
    Returns:
        Downsampled tensor
    """
    if data.dim() == 3:
        return data[:, ::factor, ::factor]
    elif data.dim() == 4:
        return data[:, :, ::factor, ::factor]
    elif data.dim() == 5:
        return data[:, :, ::factor, ::factor, ::factor]
    else:
        raise ValueError(f"Unsupported data dimension: {data.dim()}")


def duplicate_to_match_size(low_res, high_res_size):
    """
    Duplicate low-resolution data to match high-resolution size.
    
    As described in the paper: "we duplicate the low-resolution data to match
    the size of high-resolution data."
    
    Args:
        low_res: Low-resolution tensor
        high_res_size: Target size tuple
    
    Returns:
        Duplicated tensor matching high_res_size
    """
    if low_res.dim() == 4:
        return torch.nn.functional.interpolate(
            low_res, size=high_res_size[-2:], mode="nearest"
        )
    elif low_res.dim() == 5:
        return torch.nn.functional.interpolate(
            low_res, size=high_res_size[-3:], mode="nearest"
        )
    else:
        raise ValueError(f"Unsupported dimension: {low_res.dim()}")
