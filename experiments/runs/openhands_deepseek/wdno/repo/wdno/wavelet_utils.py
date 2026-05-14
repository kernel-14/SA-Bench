import torch
import torch.nn.functional as F
import numpy as np

try:
    import pytorch_wavelets.dwt.lowlevel as ll
    HAS_PYTORCH_WAVELETS = True
except ImportError:
    HAS_PYTORCH_WAVELETS = False

try:
    import ptwt
    HAS_PTWT = True
except ImportError:
    HAS_PTWT = False


class WaveletTransform1D:
    """1D wavelet transform for initial conditions and target states."""
    
    def __init__(self, wavelet='bior2.4', mode='periodization', level=None):
        self.wavelet = wavelet
        self.mode = mode
        self.level = level
    
    def decompose(self, x):
        """
        Args:
            x: (B, C, L) tensor
        Returns:
            coeffs: dict with 'cA' (coarse) and 'cD' (detail) keys
        """
        if HAS_PYTORCH_WAVELETS:
            coeffs = ll.dwt(x, self.wavelet, self.mode)
            return {'cA': coeffs[0], 'cD': coeffs[1]}
        else:
            raise ImportError("pytorch_wavelets required for 1D wavelet transform")
    
    def reconstruct(self, cA, cD):
        if HAS_PYTORCH_WAVELETS:
            return ll.idwt((cA, cD), self.wavelet, self.mode)
        else:
            raise ImportError("pytorch_wavelets required for 1D wavelet transform")


class WaveletTransform2D:
    """2D wavelet transform for spatiotemporal data (1D space + 1D time).
    Used for 1D Burgers, Advection, Navier-Stokes equations.
    """
    
    def __init__(self, wavelet='bior2.4', mode='periodization'):
        self.wavelet = wavelet
        self.mode = mode
    
    def decompose(self, x):
        """
        Args:
            x: (B, C, H, W) tensor (time x space)
        Returns:
            coeffs: tensor of shape (B, 4*C, H//2, W//2)
            [cA, cH, cV, cD] concatenated along channel dim
        """
        if HAS_PYTORCH_WAVELETS:
            coeffs = ll.dwt2(x, self.wavelet, self.mode)
            # coeffs is (cA, (cH, cV, cD))
            cA = coeffs[0]
            cH, cV, cD = coeffs[1]
            return torch.cat([cA, cH, cV, cD], dim=1)
        else:
            raise ImportError("pytorch_wavelets required for 2D wavelet transform")
    
    def decompose_separate(self, x):
        """Returns separate coarse and detail coefficients."""
        if HAS_PYTORCH_WAVELETS:
            coeffs = ll.dwt2(x, self.wavelet, self.mode)
            cA = coeffs[0]
            cH, cV, cD = coeffs[1]
            return cA, (cH, cV, cD)
        else:
            raise ImportError("pytorch_wavelets required for 2D wavelet transform")
    
    def reconstruct(self, coeffs):
        """
        Args:
            coeffs: tensor of shape (B, 4*C, H_out, W_out)
        Returns:
            reconstructed: (B, C, 2*H_out, 2*W_out)
        """
        if HAS_PYTORCH_WAVELETS:
            B, C4, H, W = coeffs.shape
            C = C4 // 4
            cA = coeffs[:, :C, :, :]
            cH = coeffs[:, C:2*C, :, :]
            cV = coeffs[:, 2*C:3*C, :, :]
            cD = coeffs[:, 3*C:4*C, :, :]
            return ll.idwt2((cA, (cH, cV, cD)), self.wavelet, self.mode)
        else:
            raise ImportError("pytorch_wavelets required for 2D wavelet transform")
    
    def reconstruct_from_components(self, cA, cH, cV, cD):
        if HAS_PYTORCH_WAVELETS:
            return ll.idwt2((cA, (cH, cV, cD)), self.wavelet, self.mode)
        else:
            raise ImportError("pytorch_wavelets required for 2D wavelet transform")


class WaveletTransform3D:
    """3D wavelet transform for spatiotemporal data (2D space + 1D time).
    Used for 2D incompressible fluid and ERA5.
    """
    
    def __init__(self, wavelet='bior1.3', mode='zero'):
        self.wavelet = wavelet
        self.mode = mode
    
    def decompose(self, x):
        """
        Args:
            x: (B, C, T, H, W) tensor
        Returns:
            coeffs: tensor of shape (B, 8*C, T//2, H//2, W//2)
            [cA, cH1, cH2, ..., cD7] concatenated along channel dim
        """
        if HAS_PTWT:
            coeffs = ptwt.wavedec3(x, self.wavelet, mode=self.mode, level=1)
            # coeffs is [cA, (cH1, cH2, cH3, cH4, cH5, cH6, cH7)]
            cA = coeffs[0]
            details = coeffs[1]
            all_coeffs = [cA] + list(details)
            return torch.cat(all_coeffs, dim=1)
        else:
            raise ImportError("ptwt required for 3D wavelet transform")
    
    def decompose_separate(self, x):
        """Returns separate coarse and detail coefficients."""
        if HAS_PTWT:
            coeffs = ptwt.wavedec3(x, self.wavelet, mode=self.mode, level=1)
            return coeffs[0], coeffs[1]
        else:
            raise ImportError("ptwt required for 3D wavelet transform")
    
    def reconstruct(self, coeffs):
        """
        Args:
            coeffs: tensor of shape (B, 8*C, T, H, W)
        Returns:
            reconstructed: (B, C, 2*T, 2*H, 2*W)
        """
        if HAS_PTWT:
            B, C8 = coeffs.shape[:2]
            C = C8 // 8
            cA = coeffs[:, :C]
            details = tuple([coeffs[:, (i+1)*C:(i+2)*C] for i in range(7)])
            return ptwt.waverec3((cA, details), self.wavelet, mode=self.mode)
        else:
            raise ImportError("ptwt required for 3D wavelet transform")
    
    def reconstruct_from_components(self, cA, details):
        if HAS_PTWT:
            return ptwt.waverec3((cA, details), self.wavelet, mode=self.mode)
        else:
            raise ImportError("ptwt required for 3D wavelet transform")


def duplicate_low_res_to_high_res(low_res, target_shape):
    """Duplicate low-resolution data to match high-resolution shape.
    
    Args:
        low_res: tensor of shape (B, C, *spatial_low)
        target_shape: tuple of target spatial dimensions
    
    Returns:
        duplicated: tensor of shape (B, C, *target_shape)
    """
    spatial_low = low_res.shape[2:]
    spatial_high = target_shape
    
    scale_factors = [h // l for h, l in zip(spatial_high, spatial_low)]
    
    result = low_res
    for dim_idx, scale in enumerate(scale_factors):
        result = result.repeat_interleave(scale, dim=dim_idx + 2)
    
    # Handle odd sizes: duplicate last element along each dimension if needed
    for dim_idx, (h, l) in enumerate(zip(spatial_high, spatial_low)):
        if result.shape[dim_idx + 2] < h:
            pad = [0] * (2 * len(spatial_high))
            pad[2 * (len(spatial_high) - 1 - dim_idx) + 1] = 1
            result = F.pad(result, pad, mode='replicate')
    
    return result


def pad_to_match(tensor, target_shape):
    """Pad tensor to match target spatial shape."""
    current_shape = tensor.shape[2:]
    padding = []
    for cur, tgt in zip(reversed(current_shape), reversed(target_shape)):
        diff = tgt - cur
        padding.extend([0, diff])
    return F.pad(tensor, padding, mode='replicate')
