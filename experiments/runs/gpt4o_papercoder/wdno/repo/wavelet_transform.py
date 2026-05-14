# wavelet_transform.py
"""
This module provides the WaveletTransform class to handle wavelet decomposition
and reconstruction operations using PyTorch Wavelets. 
The implementation follows the prescribed methodology for the WDNO paper.
"""

from typing import List
import torch
from torch import Tensor
from pytorch_wavelets import DWTForward, DWTInverse


class WaveletTransform:
    """
    A class to perform wavelet decomposition and reconstruction operations.
    It supports both 1D and 2D data types based on configurations provided.

    Attributes:
        wavelet_type (str): Wavelet basis to use (e.g., 'bior2.4' for 1D, 'bior1.3' for 2D).
        mode (str): Boundary handling mode (e.g., 'periodization' for 1D, 'zero' for 2D).
        forward_transform: PyTorch DWTForward operator.
        inverse_transform: PyTorch DWTInverse operator.
    """

    def __init__(self, wavelet_type: str = "bior2.4", mode: str = "periodization") -> None:
        """
        Initializes the WaveletTransform class with the given wavelet type and mode.

        Args:
            wavelet_type (str): The wavelet basis to use for decomposition and reconstruction.
                                Default is 'bior2.4' for 1D data (can switch to 'bior1.3' for 2D).
            mode (str): Boundary handling technique. Default is 'periodization'.

        Raises:
            ValueError: If the wavelet type or mode provided is invalid.
        """
        self.wavelet_type = wavelet_type
        self.mode = mode

        # Validate wavelet type and mode
        supported_wavelets = ["bior2.4", "bior1.3", "db", "sym"]
        supported_modes = ["periodization", "zero"]

        if self.wavelet_type not in supported_wavelets:
            raise ValueError(f"Unsupported wavelet type '{wavelet_type}'. "
                             f"Supported wavelets: {supported_wavelets}.")
        if self.mode not in supported_modes:
            raise ValueError(f"Unsupported wavelet mode '{mode}'. "
                             f"Supported modes: {supported_modes}.")

        # PyTorch wavelet forward and inverse transforms
        self.forward_transform = DWTForward(J=1, wave=self.wavelet_type, mode=self.mode)
        self.inverse_transform = DWTInverse(wave=self.wavelet_type, mode=self.mode)

    def apply_transform(self, data: Tensor) -> List[Tensor]:
        """
        Apply forward wavelet decomposition to the input data.

        Args:
            data (Tensor): Input PyTorch Tensor.
                           Expected dimensions:
                           - For 1D tasks: [batch_size, time_steps, spatial_points].
                           - For 2D tasks: [batch_size, time_steps, height, width].

        Returns:
            List[Tensor]: List of tensors representing the wavelet coefficients,
                          including low-frequency (coarse) and high-frequency (details).
                          The structure of the list depends on whether the input is 1D or 2D.

        Raises:
            ValueError: If the input tensor dimensions are invalid.
        """
        if data.dim() not in [3, 4]:
            raise ValueError(f"Input tensor dimensions must be 3 for 1D tasks or 4 for 2D tasks. "
                             f"Got {data.dim()} dimensions.")

        # Apply wavelet forward transform
        low_freq, high_freq = self.forward_transform(data)

        # Combine low-frequency and high-frequency coefficients into a single list
        if isinstance(high_freq, list):  # Handle variable-level decomposition details
            coeffs = [low_freq] + high_freq
        else:
            coeffs = [low_freq, high_freq]

        return coeffs

    def inverse_transform(self, coeffs: List[Tensor]) -> Tensor:
        """
        Perform inverse wavelet transform to reconstruct the original data.

        Args:
            coeffs (List[Tensor]): List of PyTorch tensors representing wavelet coefficients.
                                   This should include both low-frequency and detail coefficients.

        Returns:
            Tensor: The reconstructed tensor, matching the dimensions of the original input.

        Raises:
            ValueError: If the list of coefficients is empty or mismatched.
        """
        if not isinstance(coeffs, list) or len(coeffs) < 2:
            raise ValueError("Coefficients must be a non-empty list containing low-frequency "
                             "and high-frequency components.")

        # Separate low- and high-frequency components
        low_freq = coeffs[0]
        high_freq = coeffs[1:]  # Remaining are detail coefficients

        # Apply wavelet inverse transform
        reconstructed = self.inverse_transform((low_freq, high_freq))
        return reconstructed

    def __repr__(self) -> str:
        """
        Returns a string representation of the WaveletTransform configuration.

        Returns:
            str: Summary of wavelet transform configuration.
        """
        return (f"WaveletTransform(wavelet_type={self.wavelet_type}, "
                f"mode={self.mode})")


# Example Usage inside DatasetLoader (for integration testing):
# ---------------------------------------------------------------
# from wavelet_transform import WaveletTransform
# import torch
# 
# # Initialize wavelet transform for 1D use case
# wavelet_transform = WaveletTransform(wavelet_type="bior2.4", mode="periodization")
# 
# # Simulated 1D data: [batch_size, time_steps, spatial_points]
# sample_data = torch.randn(16, 81, 120)  # Example input for Burgers' equation
# 
# # Forward transform
# coeffs = wavelet_transform.apply_transform(sample_data)
# print(f"Coarse Coefficients Shape: {coeffs[0].shape}")
# for i, coeff in enumerate(coeffs[1:], 1):
#     print(f"Detail Coefficients Level-{i} Shape: {coeff.shape}")
# 
# # Inverse transform
# reconstructed_data = wavelet_transform.inverse_transform(coeffs)
# print(f"Reconstructed Data Shape: {reconstructed_data.shape}")
