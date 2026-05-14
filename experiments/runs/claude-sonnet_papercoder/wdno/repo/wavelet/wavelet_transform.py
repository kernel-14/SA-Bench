```python
## wavelet/wavelet_transform.py
"""Wavelet transform module for WDNO (Wavelet Diffusion Neural Operator).

This module implements the core WaveletTransform class that converts raw PDE
state tensors into packed wavelet coefficient tensors and back. It is the
foundational component used by all data pipelines, model conditioning, and
super-resolution alignment in WDNO.

Key design decisions (from paper Appendix A, F.3, H.2):
    - 1D PDE experiments: 2D DWT on (time × space) using pytorch_wavelets
      with bior2.4 wavelet and periodization mode.
    - 2D PDE experiments: 3D DWT on (time × height × width) using ptwt
      with bior1.3 wavelet and zero mode.
    - Level is always 1 (l_0 = L, finest level), yielding exactly one coarse
      + one detail set at the finest level to preserve locality.
    - Reconstruction error target: relative L2 ~1e-7 (paper Table 3).

Paper sources:
    - Wavelet basis selection: Appendix A
    - 1D experiment transform: Appendix F.3 (bior2.4, periodization, 2D DWT)
    - 2D experiment transform: Appendix H.2 (bior1.3, zero, 3D DWT)
    - Reconstruction verification: Appendix A, Table 3
    - Coefficient shapes: Appendix F.3 ([41,60] per set), H.2 ([18,34,34] per set)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Fixed ordering of 3D DWT detail subband keys (ptwt convention).
# Must be consistent between forward() and inverse() to preserve channel
# semantics learned by the U-Net.
_PTWT_DETAIL_KEYS: Tuple[str, ...] = (
    "aad",
    "ada",
    "add",
    "daa",
    "dad",
    "dda",
    "ddd",
)

# Number of coefficient sets per transform dimensionality.
# 2D DWT at level=1: 1 coarse (cA) + 3 detail (cH, cV, cD) = 4 sets.
# 3D DWT at level=1: 1 coarse (cA) + 7 detail = 8 sets.
_NUM_COEFF_SETS: Dict[int, int] = {1: 4, 2: 8}


class WaveletTransform(nn.Module):
    """Wavelet transform layer for WDNO.

    Converts raw PDE state tensors to packed wavelet coefficient tensors
    and back. Supports both 1D PDE experiments (2D DWT via pytorch_wavelets)
    and 2D PDE experiments (3D DWT via ptwt).

    This class holds no learnable parameters and is instantiated once per
    experiment, reused across data loading, training, and inference.

    Attributes:
        wavelet: Wavelet basis string, e.g. 'bior2.4' or 'bior1.3'.
        mode: Padding mode, 'periodization' for 1D PDEs or 'zero' for 2D.
        level: Decomposition level (always 1, l_0=L finest level).
        spatial_dim: Spatial dimensionality of the PDE (1 or 2). Drives
            choice of 2D vs 3D DWT and library backend.
        num_coeff_sets: Number of coefficient sets after transform (4 for
            spatial_dim=1, 8 for spatial_dim=2).
    """

    def __init__(
        self,
        wavelet: str = "bior2.4",
        mode: str = "periodization",
        level: int = 1,
        spatial_dim: int = 1,
    ) -> None:
        """Initialize the WaveletTransform.

        Args:
            wavelet: Wavelet basis string. Paper uses 'bior2.4' for 1D PDE
                experiments (Burgers', advection, compressible NS) and
                'bior1.3' for 2D PDE experiments (fluid_2d, era5).
                Config: wavelet.burgers.wavelet_type / wavelet.fluid_2d.wavelet_type.
            mode: Padding mode for the DWT. 'periodization' for 1D PDE
                experiments (config: wavelet.burgers.padding_mode),
                'zero' for 2D PDE experiments
                (config: wavelet.fluid_2d.padding_mode).
            level: Decomposition level. Always 1 (l_0=L, finest level) per
                paper Section 3.1 and config: wavelet.*.level.
            spatial_dim: Spatial dimensionality of the PDE. 1 means apply
                2D DWT on (time × space) data using pytorch_wavelets;
                2 means apply 3D DWT on (time × height × width) data
                using ptwt. Config: experiment.spatial_dim.

        Raises:
            ValueError: If spatial_dim is not 1 or 2.
            ImportError: If the required wavelet library is not installed.
        """
        super().__init__()

        if spatial_dim not in (1, 2):
            raise ValueError(
                f"spatial_dim must be 1 or 2, got {spatial_dim}. "
                "Use 1 for 1D PDE experiments (Burgers', advection, "
                "compressible NS) and 2 for 2D PDE experiments (fluid_2d, era5)."
            )

        self.wavelet: str = wavelet
        self.mode: str = mode
        self.level: int = level
        self.spatial_dim: int = spatial_dim
        self.num_coeff_sets: int = _NUM_COEFF_SETS[spatial_dim - 1 + 1]

        if spatial_dim == 1:
            self._init_1d_transform()
        else:
            self._init_2d_transform()

        logger.info(
            "WaveletTransform initialized: spatial_dim=%d, wavelet=%s, "
            "mode=%s, level=%d, num_coeff_sets=%d",
            spatial_dim,
            wavelet,
            mode,
            level,
            self.num_coeff_sets,
        )

    def _init_1d_transform(self) -> None:
        """Initialize pytorch_wavelets DWT for 1D PDE experiments.

        Uses DWTForward/DWTInverse from pytorch_wavelets. The transform
        operates on 2D data (time × space) treated as a single-channel
        2D image.

        Raises:
            ImportError: If pytorch_wavelets is not installed.
        """
        try:
            from pytorch_wavelets import DWTForward, DWTInverse
        except ImportError as exc:
            raise ImportError(
                "pytorch_wavelets is required for 1D PDE experiments. "
                "Install with: pip install pytorch-wavelets"
            ) from exc

        # DWTForward expects [B, C, H, W] and returns (yl, [yh])
        # J=level=1 yields exactly one coarse + one detail level
        self.dwt_forward = DWTForward(
            J=self.level,
            wave=self.wavelet,
            mode=self.mode,
        )
        self.dwt_inverse = DWTInverse(
            wave=self.wavelet,
            mode=self.mode,
        )
        self.num_coeff_sets = 4  # cA, cH, cV, cD

    def _init_2d_transform(self) -> None:
        """Initialize ptwt for 2D PDE experiments.

        ptwt uses string-based wavelet specification and handles device
        placement automatically. No persistent objects needed.

        Raises:
            ImportError: If ptwt is not installed.
        """
        try:
            import ptwt  # noqa: F401 — verify import works
        except ImportError as exc:
            raise ImportError(
                "ptwt (pytorch-wavelet-toolbox) is required for 2D PDE "
                "experiments. Install with: pip install ptwt"
            ) from exc

        self.num_coeff_sets = 8  # cA + 7 detail subbands

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply DWT to raw PDE data and return packed wavelet coefficients.

        Dispatches to the appropriate DWT backend based on spatial_dim.

        For spatial_dim=1 (2D DWT):
            Input:  [B, T, X]  e.g. [N, 81, 120] for Burgers'
            Output: [B, 4, T_c, X_c]  e.g. [N, 4, 41, 60]

        For spatial_dim=2 (3D DWT):
            Input:  [B, T, H, W]  e.g. [N, 32, 64, 64] for fluid_2d
            Output: [B, 8, T_c, H_c, W_c]  e.g. [N, 8, 18, 34, 34]

        The output is a single tensor with all coefficient sets packed along
        the channel dimension (dim=1). This is the format expected by the
        U-Net denoising backbone.

        Args:
            x: Raw PDE state tensor. Shape [B, T, X] for 1D PDEs or
                [B, T, H, W] for 2D PDEs. Must be float32.

        Returns:
            Packed wavelet coefficient tensor with all sets concatenated
            along dim=1.

        Raises:
            ValueError: If x has unexpected number of dimensions.
        """
        if self.spatial_dim == 1:
            return self._forward_1d(x)
        else:
            return self._forward_2d(x)

    def _forward_1d(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 2D DWT to 1D PDE data (time × space).

        Args:
            x: Tensor of shape [B, T, X].

        Returns:
            Packed coefficients of shape [B, 4, T_c, X_c].
        """
        if x.dim() != 3:
            raise ValueError(
                f"Expected 3D input [B, T, X] for spatial_dim=1, got {x.dim()}D tensor "
                f"with shape {tuple(x.shape)}."
            )

        # pytorch_wavelets requires [B, C, H, W]; treat (T, X) as (H, W)
        x_4d = x.unsqueeze(1)  # [B, 1, T, X]

        # Move DWT modules to same device as input
        device = x.device
        self.dwt_forward = self.dwt_forward.to(device)

        # Apply forward DWT: returns (yl, [yh])
        # yl: [B, 1, T_c, X_c]  — coarse approximation
        # yh[0]: [B, 1, 3, T_c, X_c]  — 3 detail subbands at level 1
        yl, yh_list = self.dwt_forward(x_4d)

        # Extract individual coefficient sets, removing the channel dim (C=1)
        cA = yl.squeeze(1)              # [B, T_c, X_c]
        cH = yh_list[0][:, 0, 0, :, :]  # [B, T_c, X_c] horizontal detail
        cV = yh_list[0][:, 0, 1, :, :]  # [B, T_c, X_c] vertical detail
        cD = yh_list[0][:, 0, 2, :, :]  # [B, T_c, X_c] diagonal detail

        # Pack all 4 coefficient sets along channel dim
        return self._pack_coefficients([cA, cH, cV, cD])

    def _forward_2d(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 3D DWT to 2D PDE data (time × height × width).

        Args:
            x: Tensor of shape [B, T, H, W].

        Returns:
            Packed coefficients of shape [B, 8, T_c, H_c, W_c].
        """
        if x.dim() != 4:
            raise ValueError(
                f"Expected 4D input [B, T, H, W] for spatial_dim=2, got {x.dim()}D "
                f"tensor with shape {tuple(x.shape)}."
            )

        import ptwt

        # ptwt requires [B, C, D, H, W]; treat (T, H, W) as (D, H, W)
        x_5d = x.unsqueeze(1)  # [B, 1, T, H, W]

        # Apply 3D DWT
        # Returns: [cA, detail_dict] where
        #   cA: [B, 1, T_c, H_c, W_c]
        #   detail_dict: dict with 7 keys, each value [B, 1, T_c, H_c, W_c]
        coeffs = ptwt.wavedec3(
            x_5d,
            wavelet=self.wavelet,
            level=self.level,
            mode=self.mode,
        )

        # Extract coarse approximation
        cA = coeffs[0].squeeze(1)  # [B, T_c, H_c, W_c]

        # Extract 7 detail subbands in fixed alphabetical order
        detail_dict = coeffs[1]
        detail_tensors = [
            detail_dict[key].squeeze(1) for key in _PTWT_DETAIL_KEYS
        ]

        # Pack all 8 coefficient sets along channel dim
        all_coeffs = [cA] + detail_tensors
        return self._pack_coefficients(all_coeffs)

    def inverse(
        self,
        coeffs: torch.Tensor,
        original_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """Reconstruct raw PDE data from packed wavelet coefficients.

        Dispatches to the appropriate inverse DWT backend based on spatial_dim.

        For spatial_dim=1:
            Input:  [B, 4, T_c, X_c]
            Output: [B, T, X]  cropped to original_shape

        For spatial_dim=2:
            Input:  [B, 8, T_c, H_c, W_c]
            Output: [B, T, H, W]  cropped to original_shape

        Args:
            coeffs: Packed wavelet coefficient tensor as returned by forward().
            original_shape: Shape of the original input tensor before the
                forward transform. Used to crop the reconstruction to the
                correct size (padding modes may add extra elements).
                Format: (B, T, X) for 1D or (B, T, H, W) for 2D.

        Returns:
            Reconstructed PDE state tensor with shape matching original_shape.

        Raises:
            ValueError: If coeffs has unexpected number of dimensions.
        """
        if self.spatial_dim == 1:
            return self._inverse_1d(coeffs, original_shape)
        else:
            return self._inverse_2d(coeffs, original_shape)

    def _inverse_1d(
        self,
        coeffs: torch.Tensor,
        original_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """Apply inverse 2D DWT to reconstruct 1D PDE data.

        Args:
            coeffs: Packed coefficients of shape [B, 4, T_c, X_c].
            original_shape: Original input shape (B, T, X).

        Returns:
            Reconstructed tensor of shape (B, T, X).
        """
        # Unpack the 4 coefficient sets
        coeff_list = self._unpack_coefficients(coeffs, original_shape)
        cA, cH, cV, cD = coeff_list[0], coeff_list[1], coeff_list[2], coeff_list[3]

        # Reconstruct pytorch_wavelets format
        # yl: [B, 1, T_c, X_c]
        yl = cA.unsqueeze(1)

        # yh: [B, 1, 3, T_c, X_c] — stack detail subbands along dim=2
        yh = torch.stack([cH, cV, cD], dim=1).unsqueeze(1)
        # After stack: [B, 3, T_c, X_c] → unsqueeze(1) → [B, 1, 3, T_c, X_c]

        # Move inverse DWT to correct device
        device = coeffs.device
        self.dwt_inverse = self.dwt_inverse.to(device)

        # Apply inverse DWT: returns [B, 1, T, X]
        x_reconstructed = self.dwt_inverse((yl, [yh]))

        # Remove channel dimension: [B, T, X]
        x_reconstructed = x_reconstructed.squeeze(1)

        # Crop to original shape (padding may add extra elements)
        if len(original_shape) >= 3:
            x_reconstructed = x_reconstructed[
                :, : original_shape[1], : original_shape[2]
            ]

        return x_reconstructed

    def _inverse_2d(
        self,
        coeffs: torch.Tensor,
        original_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """Apply inverse 3D DWT to reconstruct 2D PDE data.

        Args:
            coeffs: Packed coefficients of shape [B, 8, T_c, H_c, W_c].
            original_shape: Original input shape (B, T, H, W).

        Returns:
            Reconstructed tensor of shape (B, T, H, W).
        """
        import ptwt

        # Unpack the 8 coefficient sets
        coeff_list = self._unpack_coefficients(coeffs, original_shape)
        cA = coeff_list[0]  # [B, T_c, H_c, W_c]
        detail_tensors = coeff_list[1:]  # 7 detail tensors

        # Reconstruct ptwt format
        # cA_5d: [B, 1, T_c, H_c, W_c]
        cA_5d = cA.unsqueeze(1)

        # detail_dict: each value [B, 1, T_c, H_c, W_c]
        detail_dict: Dict[str, torch.Tensor] = {
            key: detail_tensors[i].unsqueeze(1)
            for i, key in enumerate(_PTWT_DETAIL_KEYS)
        }

        # ptwt expects list: [cA, detail_dict]
        ptwt_coeffs = [cA_5d, detail_dict]

        # Apply inverse 3D DWT: returns [B, 1, T, H, W]
        x_reconstructed = ptwt.waverec3(ptwt_coeffs, wavelet=self.wavelet)

        # Remove channel dimension: [B, T, H, W]
        x_reconstructed = x_reconstructed.squeeze(1)

        # Crop to original shape
        if len(original_shape) >= 4:
            x_reconstructed = x_reconstructed[
                :,
                : original_shape[1],
                : original_shape[2],
                : original_shape[3],
            ]

        return x_reconstructed

    def _pack_coefficients(self, coeffs: List[torch.Tensor]) -> torch.Tensor:
        """Concatenate coefficient sets along the channel dimension (dim=1).

        Each coefficient set is a tensor of shape [B, *spatial_dims]. This
        method adds a channel dimension to each and concatenates them, producing
        a single multi-channel tensor suitable for the U-Net backbone.

        The channel ordering is fixed and must be consistent between forward()
        and inverse() calls. For 2D DWT: [cA, cH, cV, cD]. For 3D DWT:
        [cA, aad, ada, add, daa, dad, dda, ddd].

        Args:
            coeffs: List of coefficient tensors, each of shape
                [B, T_c, X_c] (1D case) or [B, T_c, H_c, W_c] (2D case).

        Returns:
            Packed tensor of shape [B, num_sets, T_c, X_c] or
            [B, num_sets, T_c, H_c, W_c].
        """
        # Add channel dimension to each coefficient set: [B, *dims] → [B, 1, *dims]
        coeffs_with_channel = [c.unsqueeze(1) for c in coeffs]
        # Concatenate along channel dim=1
        return torch.cat(coeffs_with_channel, dim=1)

    def _unpack_coefficients(
        self,
        packed: torch.Tensor,
        original_shape: Tuple[int, ...],
    ) -> List[torch.Tensor]:
        """Split packed coefficient tensor back into individual coefficient sets.

        Reverses the operation of _pack_coefficients. The channel ordering
        is preserved from the forward pass.

        Args:
            packed: Packed coefficient tensor of shape
                [B, num_sets, T_c, X_c] or [B, num_sets, T_c, H_c, W_c].
            original_shape: Original input shape before forward transform.
                Not used directly in unpacking but passed for API consistency
                with inverse() which uses it for cropping.

        Returns:
            List of coefficient tensors, each of shape [B, T_c, X_c] or
            [B, T_c, H_c, W_c]. Length equals num_coeff_sets (4 or 8).
        """
        num_sets = packed.shape[1]
        # Split along channel dim, removing the channel dimension
        return [packed[:, i, ...] for i in range(num_sets)]

    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute the shape of packed wavelet coefficients for a given input shape.

        Used by model builders to determine the number of input channels for
        the U-Net. Does not run the actual transform.

        For spatial_dim=1 with periodization mode:
            Input (B, T, X) → Output (B, 4, ceil(T/2), ceil(X/2))
            Burgers' [N, 81, 120] → [N, 4, 41, 60]  ✓ (paper Appendix F.3)

        For spatial_dim=2 with zero mode:
            Input (B, T, H, W) → Output (B, 8, T_c, H_c, W_c)
            Fluid [N, 32, 64, 64] → [N, 8, 18, 34, 34]  ✓ (paper Appendix H.2)
            Note: zero padding adds extra elements; T_c = T//2 + 2, H_c = H//2 + 2

        Args:
            input_shape: Shape of the raw PDE data tensor. Format:
                (B, T, X) for 1D PDEs or (B, T, H, W) for 2D PDEs.

        Returns:
            Shape of the packed wavelet coefficient tensor.

        Raises:
            ValueError: If input_shape has wrong number of dimensions for
                the configured spatial_dim.
        """
        if self.spatial_dim == 1:
            if len(input_shape) != 3:
                raise ValueError(
                    f"spatial_dim=1 expects input_shape of length 3 (B, T, X), "
                    f"got length {len(input_shape)}: {input_shape}."
                )
            B, T, X = input_shape
            # periodization mode: output size = ceil(N/2)
            T_c = (T + 1) // 2
            X_c = (X + 1) // 2
            return (B, self.num_coeff_sets, T_c, X_c)

        else:  # spatial_dim == 2
            if len(input_shape) != 4:
                raise ValueError(
                    f"spatial_dim=2 expects input_shape of length 4 (B, T, H, W), "
                    f"got length {len(input_shape)}: {input_shape}."
                )
            B, T, H, W = input_shape
            # zero mode: ptwt adds padding; empirically T_c = T//2 + 2 for bior1.3
            # Verified against paper: [32,64,64] → [18,34,34]
            # 32//2 + 2 = 18 ✓, 64//2 + 2 = 34 ✓
            T_c = T // 2 + 2
            H_c = H // 2 + 2
            W_c = W // 2 + 2
            return (B, self.num_coeff_sets, T_c, H_c, W_c)

    def verify_reconstruction(self, x: torch.Tensor) -> float:
        """Verify near-lossless reconstruction via relative L2 error.

        Computes the relative L2 error between the original tensor and the
        result of forward() followed by inverse(). Expected error is ~1e-7
        for bior2.4 (1D) and ~1e-7 for bior1.3 (2D) per paper Table 3.

        This method should be called in main.py on a small sample batch
        before training begins to confirm correct wavelet setup.

        Args:
            x: Sample PDE data tensor. Shape [B, T, X] for 1D or
                [B, T, H, W] for 2D. Should be a representative sample
                from the training set.

        Returns:
            Relative L2 error: ||x - inverse(forward(x))||_2 / ||x||_2.
            Expected value: ~1e-7 (paper Appendix A, Table 3).
        """
        with torch.no_grad():
            coeffs = self.forward(x)
            x_reconstructed = self.inverse(coeffs, tuple(x.shape))
            numerator = torch.norm(x - x_reconstructed)
            denominator = torch.norm(x)
            if denominator < 1e-12:
                return float(numerator.item())
            error = (numerator / denominator).item()

        logger.info(
            "Wavelet reconstruction relative L2 error: %.2e "
            "(expected ~1e-7 for bior wavelets, paper Table 3)",
            error,
        )
        return float(error)

    def repeat_1d_to_nd(
        self,
        coeffs_lower: torch.Tensor,
        target_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """Tile lower-dimensional wavelet coefficients to match a higher-dimensional shape.

        Used to prepare conditioning tensors for channel-wise concatenation
        with the main wavelet coefficients. Three cases are handled:

        Case 1 — 1D condition → 2D coefficient shape (1D PDE experiments):
            coeffs_lower: [B, C_1d, X_c]  (from 1D wavelet of u_0 or u*)
            target_shape: (B, C_nd, T_c, X_c)
            → tile along temporal dim: [B, C_1d, T_c, X_c]

        Case 2 — 2D condition → 3D coefficient shape (2D PDE experiments):
            coeffs_lower: [B, C_2d, H_c, W_c]  (from 2D wavelet of initial density)
            target_shape: (B, C_nd, T_c, H_c, W_c)
            → tile along temporal dim: [B, C_2d, T_c, H_c, W_c]

        Case 3 — 1D condition → 3D coefficient shape (smoke percentage in 2D fluid):
            coeffs_lower: [B, C_1d, T_c]  (from 1D wavelet of time series)
            target_shape: (B, C_nd, T_c, H_c, W_c)
            → tile along spatial dims: [B, C_1d, T_c, H_c, W_c]

        The tiling uses expand() (zero-copy view) rather than repeat() to
        avoid unnecessary memory allocation.

        Args:
            coeffs_lower: Lower-dimensional wavelet coefficient tensor.
                Shape depends on the case (see above).
            target_shape: Target shape tuple including batch and channel dims.
                The method tiles coeffs_lower to match the spatial/temporal
                dimensions of target_shape.

        Returns:
            Tiled tensor matching the spatial/temporal dimensions of
            target_shape, with the original channel count of coeffs_lower.

        Raises:
            ValueError: If the dimensionality mismatch cannot be resolved
                by any of the three supported cases.
        """
        input_ndim = coeffs_lower.dim()
        target_ndim = len(target_shape)

        # Case 1: [B, C, X_c] → [B, C, T_c, X_c]
        # 1D condition (u_0, u*) tiled to match 2D coefficient shape
        if input_ndim == 3 and target_ndim == 4:
            T_c = target_shape[2]
            # [B, C, X_c] → [B, C, 1, X_c] → [B, C, T_c, X_c]
            coeffs_expanded = coeffs_lower.unsqueeze(2)
            return coeffs_expanded.expand(-1, -1, T_c, -1).contiguous()

        # Case 2: [B, C, H_c, W_c] → [B, C, T_c, H_c, W_c]
        # 2D condition (initial density) tiled to match 3D coefficient shape
        elif input_ndim == 4 and target_ndim == 5:
            T_c = target_shape[2]
            # [B, C, H_c, W_c] → [B, C, 1, H_c, W_c] → [B, C, T_c, H_c, W_c]
            coeffs_expanded = coeffs_lower.unsqueeze(2)
            return coeffs_expanded.expand(-1, -1, T_c, -1, -1).contiguous()

        # Case 3: [B, C, T_c] → [B, C, T_c, H_c, W_c]
        # 1D time-series condition (smoke percentage) tiled to match 3D shape
        elif input_ndim == 3 and target_ndim == 5:
            H_c = target_shape[3]
            W_c = target_shape[4]
            # [B, C, T_c] → [B, C, T_c, 1, 1] → [B, C, T_c, H_c, W_c]
            coeffs_expanded = coeffs_lower.unsqueeze(-1).unsqueeze(-1)
            return coeffs_expanded.expand(-1, -1, -1, H_c, W_c).contiguous()

        else:
            raise ValueError(
                f"Cannot tile coeffs_lower of shape {tuple(coeffs_lower.shape)} "
                f"(ndim={input_ndim}) to match target_shape {target_shape} "
                f"(ndim={target_ndim}). Supported cases: "
                "3D→4D (1D cond to 2D co