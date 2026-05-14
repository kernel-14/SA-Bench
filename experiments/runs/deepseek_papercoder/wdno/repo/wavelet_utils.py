## wavelet_utils.py
"""
Wavelet transform utility class for the Wavelet Diffusion Neural Operator (WDNO).
Wraps ptwt (PyTorch Wavelet Toolbox) to perform 2D or 3D discrete wavelet transforms
with a single level of decomposition, returning coefficient lists in a consistent order.
Also provides a reconstruction error check.

Used by:
    - dataset.py  (pre‑computing wavelet coefficients)
    - wdno.py     (inference transforms)
"""

from typing import List

import torch
import torch.nn.functional as F
import ptwt


class WaveletTransform:
    """
    Performs forward and inverse discrete wavelet transforms using ptwt.
    Supports 2D transforms (for 1D spatio‑temporal data) and 3D transforms
    (for 2D spatio‑temporal + channel data).

    The decomposition is always single‑level (level=1) because WDNO uses
    L0 = L and only one set of detail coefficients.

    Attributes:
        wavelet (str):  Wavelet name (e.g., 'bior2.4', 'bior1.3').
        mode (str):     Signal extension mode (e.g., 'periodization', 'zero').
        ndim (int):     2 for 2D DWT, 3 for 3D DWT.
        level (int):    Decomposition level (fixed to 1).
        rec_tol (float): Reconstruction tolerance for check_reconstruction.
    """

    def __init__(self, wavelet: str, mode: str, ndim: int, rec_tol: float = 1.0e-6) -> None:
        """
        Args:
            wavelet: Wavelet type as accepted by ptwt (e.g., 'bior2.4', 'bior1.3', 'db4').
            mode:    Signal extension mode ('periodization', 'zero', 'symmetric', etc.).
            ndim:    Dimensionality of the transform: 2 for 2D, 3 for 3D.
            rec_tol: Maximum relative L2 reconstruction error considered acceptable.
        """
        if ndim not in (2, 3):
            raise ValueError(f"WaveletTransform only supports 2D or 3D transforms, got ndim={ndim}")

        self.wavelet = wavelet
        self.mode = mode
        self.ndim = ndim
        self.level = 1          # paper always uses single-level decomposition
        self.rec_tol = rec_tol

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Apply one-level wavelet decomposition to the input tensor.

        For ndim=2, input shape (H, W) → outputs [cA, cH, cV, cD] (4 tensors).
        For ndim=3, input shape (D, H, W) → outputs [cA, c1, ..., c7] (8 tensors).

        All tensors are returned on the same device as x and with consistent memory layout.

        Args:
            x: Input tensor of shape (H, W) for 2D or (D, H, W) for 3D.

        Returns:
            List of coefficient tensors, ordered from approximation to detail
            subbands. The order is guaranteed so that inverse() can reconstruct
            the original tensor.
        """
        if x.dim() != self.ndim:
            raise ValueError(
                f"Input tensor has {x.dim()} dimensions, but the transform is configured for "
                f"{self.ndim}D data (expecting shape {'(H, W)' if self.ndim == 2 else '(D, H, W)'})."
            )

        if self.ndim == 2:
            # wavedec2 returns [cA_n, (cH_n, cV_n, cD_n)] for level, with n decreasing
            # for level=1 it returns [cA1, (cH1, cV1, cD1)]
            coeffs = ptwt.wavedec2(x, self.wavelet, level=self.level, mode=self.mode)
            # flatten: [cA1, cH1, cV1, cD1]
            # coeffs[0] is approx, coeffs[1] is tuple of details
            flat = [coeffs[0]] + list(coeffs[1])
            return flat
        else:  # ndim == 3
            # wavedec3 returns [cA_n, (details_tuple)] for level n
            # for level=1: [cA1, (d1, d2, ..., d7)] where there are 7 detail subbands.
            coeffs = ptwt.wavedec3(x, self.wavelet, level=self.level, mode=self.mode)
            flat = [coeffs[0]] + list(coeffs[1])
            return flat

    def inverse(self, coeffs: List[torch.Tensor]) -> torch.Tensor:
        """
        Reconstruct the original tensor from the wavelet coefficient list.

        Expects the same list format as returned by forward():
            - 2D: [cA, cH, cV, cD]
            - 3D: [cA, c1, ..., c7]

        Args:
            coeffs: Wavelet coefficient tensors.

        Returns:
            Reconstructed tensor of shape (H, W) for 2D or (D, H, W) for 3D.
            The output device matches the device of the first coefficient.
        """
        if self.ndim == 2:
            if len(coeffs) != 4:
                raise ValueError(f"Expected 4 coefficients for 2D inverse, got {len(coeffs)}")
            # reconstruct nested structure: [approx, (h, v, d)]
            nested = [coeffs[0], (coeffs[1], coeffs[2], coeffs[3])]
            return ptwt.waverec2(nested, self.wavelet, mode=self.mode)
        else:  # ndim == 3
            if len(coeffs) != 8:
                raise ValueError(f"Expected 8 coefficients for 3D inverse, got {len(coeffs)}")
            nested = [coeffs[0], tuple(coeffs[1:8])]
            return ptwt.waverec3(nested, self.wavelet, mode=self.mode)

    def check_reconstruction(self, x: torch.Tensor) -> float:
        """
        Verify that the wavelet decomposition and reconstruction is nearly lossless.
        Computes the relative L2 norm of the reconstruction error.

        Args:
            x: Original input tensor (same shape as expected by forward()).

        Returns:
            float: Relative reconstruction error = ||x_rec - x||_2 / ||x||_2.
        """
        coeffs = self.forward(x)
        x_rec = self.inverse(coeffs)
        # Ensure sizes match (due to padding, the reconstructed may need trimming)
        # ptwt inverse usually handles trimming automatically, but we make sure.
        if x_rec.shape != x.shape:
            x_rec = self._crop_to_shape(x_rec, x.shape)
        rel_err = torch.norm(x_rec - x) / (torch.norm(x) + 1e-12)
        return rel_err.item()

    @staticmethod
    def _crop_to_shape(tensor: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        """Crop a tensor to match a smaller target shape along each dimension."""
        slices = tuple(slice(0, s) for s in target_shape)
        return tensor[slices]

