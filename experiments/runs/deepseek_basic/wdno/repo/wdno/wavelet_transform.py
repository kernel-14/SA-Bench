"""
Wavelet transform utilities for WDNO.

Uses pytorch_wavelets (Cotter, 2019) for 1D/2D and ptwt (Wolter et al., 2024) for 3D transforms.
Supports biorthogonal wavelet families with periodization and zero-padding modes.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List


class WaveletTransform1D(nn.Module):
    """
    1D Discrete Wavelet Transform wrapper.

    Used for 1D initial conditions and target states.

    Args:
        wavelet: Wavelet basis name (e.g., 'bior2.4', 'bior1.3')
        mode: Padding mode ('periodization', 'zero', etc.)
    """

    def __init__(self, wavelet: str = 'bior2.4', mode: str = 'periodization'):
        super().__init__()
        self.wavelet = wavelet
        self.mode = mode
        self._ensure_package()

    def _ensure_package(self):
        try:
            import pytorch_wavelets
        except ImportError:
            raise ImportError(
                "pytorch_wavelets is required for wavelet transforms. "
                "Install with: pip install pytorch-wavelets"
            )

    def forward(self, x: torch.Tensor, levels: int = 1) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward wavelet transform.

        Args:
            x: Input tensor of shape (B, C, L)
            levels: Number of decomposition levels

        Returns:
            coeffs: Low-frequency (coarse) coefficients
            details: List of high-frequency (detail) coefficients at each level
        """
        import pytorch_wavelets as pw
        from pytorch_wavelets import DWT1DForward

        # Convert to batch if needed
        if x.dim() == 2:
            x = x.unsqueeze(0)

        B, C, L = x.shape
        dwt = DWT1DForward(wave=self.wavelet, J=levels, mode=self.mode)
        coeffs_list, detail_list = dwt(x)

        # coeffs_list: list of (B, C, L_l) coarse coefficients
        # detail_list: list of (B, C, H, L_l) highpass coefficients (H=1 for real wavelet)
        return coeffs_list, detail_list

    def inverse(self, coeffs: torch.Tensor, details: List[torch.Tensor]) -> torch.Tensor:
        """
        Inverse wavelet transform.

        Args:
            coeffs: Low-frequency coefficients
            details: List of high-frequency coefficients

        Returns:
            Reconstructed signal
        """
        import pytorch_wavelets as pw
        from pytorch_wavelets import DWT1DInverse

        idwt = DWT1DInverse(wave=self.wavelet, mode=self.mode)
        return idwt((coeffs, details))


class WaveletTransform2D(nn.Module):
    """
    2D Discrete Wavelet Transform wrapper.

    Used for spatiotemporal data from 1D PDEs (time × space).
    Produces 4 sets of coefficients per level: LL, LH, HL, HH.

    Uses pytorch_wavelets package (Cotter, 2019).

    Args:
        wavelet: Wavelet basis name ('bior2.4' for 1D Burgers/CFD)
        mode: Padding mode ('periodization' for 1D, 'zero' for 2D)
    """

    def __init__(self, wavelet: str = 'bior2.4', mode: str = 'periodization'):
        super().__init__()
        self.wavelet = wavelet
        self.mode = mode
        self._ensure_package()

    def _ensure_package(self):
        try:
            import pytorch_wavelets
        except ImportError:
            raise ImportError(
                "pytorch_wavelets is required for wavelet transforms. "
                "Install with: pip install pytorch-wavelets"
            )

    def forward(self, x: torch.Tensor, levels: int = 1) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward 2D wavelet transform.

        Args:
            x: Input tensor of shape (B, C, H, W)
            levels: Number of decomposition levels

        Returns:
            coeffs: List of coarse coefficients at each level, each (B, C, H_l, W_l)
            details: List of detail coefficients, each is a tuple (LH, HL, HH)
                     each of shape (B, C, H_l, W_l)
        """
        import pytorch_wavelets as pw
        from pytorch_wavelets import DWTForward

        B, C, H, W = x.shape
        dwt = DWTForward(wave=self.wavelet, J=levels, mode=self.mode)
        coeffs, detail_list = dwt(x)

        return coeffs, detail_list

    def inverse(self, coeffs: List[torch.Tensor], details: List[torch.Tensor]) -> torch.Tensor:
        """
        Inverse 2D wavelet transform.

        Args:
            coeffs: List of coarse coefficients
            details: List of detail coefficient tuples

        Returns:
            Reconstructed signal of shape (B, C, H, W)
        """
        import pytorch_wavelets as pw
        from pytorch_wavelets import DWTInverse

        idwt = DWTInverse(wave=self.wavelet, mode=self.mode)
        return idwt((coeffs, details))


class WaveletTransform3D(nn.Module):
    """
    3D Discrete Wavelet Transform wrapper.

    Used for 2D fluid data (time × height × width).
    Produces 8 sets of coefficients per level.

    Uses Pytorch Wavelet Toolbox (ptwt) (Wolter et al., 2024).

    Args:
        wavelet: Wavelet basis name ('bior1.3' for 2D incompressible fluid)
        mode: Padding mode ('zero' for 2D)
    """

    def __init__(self, wavelet: str = 'bior1.3', mode: str = 'zero'):
        super().__init__()
        self.wavelet = wavelet
        self.mode = mode
        self._ensure_package()

    def _ensure_package(self):
        try:
            import ptwt
        except ImportError:
            raise ImportError(
                "ptwt is required for 3D wavelet transforms. "
                "Install with: pip install ptwt"
            )

    def forward(self, x: torch.Tensor, levels: int = 1) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward 3D wavelet transform.

        Args:
            x: Input tensor of shape (B, C, T, H, W)
            levels: Number of decomposition levels

        Returns:
            coeffs: Coarse coefficients (low-pass)
            details: List of detail coefficient tuples
        """
        import ptwt

        B, C, T, H, W = x.shape

        # ptwt expects (B, T, H, W) for each channel or handles multi-channel
        # We'll process channel by channel and stack
        coeffs_list = []
        details_list = []

        for b in range(B):
            x_b = x[b]  # (C, T, H, W)
            coeffs_c = []
            details_c = []
            for c in range(C):
                x_bc = x_b[c]  # (T, H, W)
                coeffs_3d, details_3d = ptwt.wavedec3(
                    x_bc, self.wavelet, level=levels, mode=self.mode
                )
                coeffs_c.append(coeffs_3d)
                details_c.append(details_3d)

            # Stack across channels
            coeffs_batch = torch.stack(coeffs_c, dim=0)  # (C, T_l, H_l, W_l)
            coeffs_list.append(coeffs_batch)

            # details_c is a list of [level1_details, level2_details, ...]
            # Each level_details is a dict with keys 'aad', 'ada', 'daa', 'add', 'dad', 'dda', 'ddd'
            details_list.append(details_c)

        coeffs = torch.stack(coeffs_list, dim=0)  # (B, C, T_l, H_l, W_l)

        # Reorganize details: list over levels, each level is (B, C, ...)
        num_levels = len(details_list[0])
        reorg_details = []
        for lvl in range(num_levels):
            lvl_details = []
            for b in range(B):
                lvl_details.append(details_list[b][lvl])
            # For each level, we have a dict of detail subbands
            reorg_details.append(lvl_details)

        return coeffs, reorg_details

    def inverse(self, coeffs: torch.Tensor, details: List) -> torch.Tensor:
        """
        Inverse 3D wavelet transform.

        Args:
            coeffs: Coarse coefficients
            details: List of detail coefficients

        Returns:
            Reconstructed signal
        """
        import ptwt

        B, C = coeffs.shape[0], coeffs.shape[1]
        reconstructions = []

        for b in range(B):
            recon_c = []
            for c in range(C):
                c_coeff = coeffs[b, c]
                # Reconstruct details for this channel
                c_details = [details[lvl][b][c] for lvl in range(len(details))]
                recon = ptwt.waverec3(
                    c_coeff, c_details, self.wavelet
                )
                recon_c.append(recon)
            recon_b = torch.stack(recon_c, dim=0)  # (C, T, H, W)
            reconstructions.append(recon_b)

        return torch.stack(reconstructions, dim=0)  # (B, C, T, H, W)


def wavelet_coeffs_to_tensor(coeffs: List[torch.Tensor], details: List[torch.Tensor]) -> torch.Tensor:
    """
    Flatten wavelet coefficients into a single tensor for the diffusion model.

    For 2D wavelet: concatenates LL, LH, HL, HH into a single tensor along channel dim.

    Args:
        coeffs: List of coarse coefficients [level0] or [(B,C,H_l,W_l)]
        details: List of detail tuples [((LH,HL,HH), ...)]

    Returns:
        Concatenated tensor of shape (B, 4*C, H_l, W_l)
    """
    if isinstance(coeffs, list):
        ll = coeffs[0]  # (B, C, H_l, W_l)
    else:
        ll = coeffs

    if len(details) == 0:
        return ll

    # details[0] is the first level: tuple of (LH, HL, HH) each (B, C, H_l, W_l)
    level_details = details[0]
    lh, hl, hh = level_details

    # Concatenate along channel dimension
    return torch.cat([ll, lh, hl, hh], dim=1)


def wavelet_tensor_to_coeffs(tensor: torch.Tensor, n_channels: int) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Convert flattened tensor back to wavelet coefficients.

    Inverse of wavelet_coeffs_to_tensor.

    Args:
        tensor: Flattened tensor (B, 4*C, H_l, W_l)
        n_channels: Number of original channels

    Returns:
        coeffs: Coarse coefficients
        details: List of detail tuples
    """
    ll = tensor[:, :n_channels]
    lh = tensor[:, n_channels:2*n_channels]
    hl = tensor[:, 2*n_channels:3*n_channels]
    hh = tensor[:, 3*n_channels:4*n_channels]

    return [ll], [(lh, hl, hh)]


def duplicate_low_res_to_match(lo_data: torch.Tensor, hi_shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Duplicate low-resolution data to match high-resolution spatial dimensions.

    As described in the paper (Section 3.2, Training):
    'to align low-resolution with high-resolution data, we duplicate the
    low-resolution data to match the size of high-resolution data.'

    For boundary handling with odd dimensions, duplicate the last element
    along each dimension.

    Args:
        lo_data: Low-resolution tensor
        hi_shape: Target high-resolution shape

    Returns:
        Duplicated tensor matching hi_shape
    """
    if lo_data.shape == hi_shape:
        return lo_data

    # Compute duplication factors
    factors = []
    for i, (lo_s, hi_s) in enumerate(zip(lo_data.shape, hi_shape)):
        factor = hi_s // lo_s
        factors.append(factor)

    # Repeat along each spatial dimension
    result = lo_data
    for dim, factor in enumerate(factors):
        if factor > 1:
            result = result.repeat_interleave(factor, dim=dim)

    # Handle boundary: if shapes don't perfectly align due to odd sizes
    if list(result.shape) != list(hi_shape):
        result = _pad_to_match(result, hi_shape)

    return result


def _pad_to_match(tensor: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
    """Pad tensor to match target shape by duplicating boundary values."""
    padded = tensor
    for dim in range(len(target_shape)):
        if padded.shape[dim] < target_shape[dim]:
            pad_size = target_shape[dim] - padded.shape[dim]
            # Pad by repeating the last slice
            pad_slice = padded.select(dim, -1).unsqueeze(dim)
            repeats = [1] * padded.dim()
            repeats[dim] = pad_size
            pad_slice = pad_slice.repeat(*repeats)
            padded = torch.cat([padded, pad_slice], dim=dim)
    return padded
