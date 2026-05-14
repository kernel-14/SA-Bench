## pyramid_utils.py
import torch
import torch.nn.functional as F
from typing import Tuple

class PyramidUtils:
    """Utility class for spatial and temporal pyramid operations including compression, decompression, and noise addition."""

    @staticmethod
    def compress_frames(frames: torch.Tensor, levels: int) -> torch.Tensor:
        """
        Compress video frames by progressively reducing spatial resolution.

        Args:
            frames (torch.Tensor): Input tensor of shape (B, T, C, H, W), 
                                   where B=batch size, T=timesteps, C=channels, H=height, W=width.
            levels (int): Number of compression levels. Each level halves the resolution.

        Returns:
            torch.Tensor: Compressed tensor with reduced resolution (B, T, C, H/2^levels, W/2^levels).
        """
        assert levels >= 0, "Compression levels must be non-negative."
        _, _, _, height, width = frames.size()
        new_height, new_width = height // (2 ** levels), width // (2 ** levels)
        return F.interpolate(frames, size=(new_height, new_width), mode="bilinear", align_corners=False)

    @staticmethod
    def decompress(frames: torch.Tensor, levels: int) -> torch.Tensor:
        """
        Decompress video frames by progressively increasing spatial resolution.

        Args:
            frames (torch.Tensor): Compressed tensor input of shape (B, T, C, H', W'),
                                   where H' and W' represent the compressed height and width.
            levels (int): Number of decompression levels. Each level doubles the resolution.

        Returns:
            torch.Tensor: Decompressed tensor with increased resolution (B, T, C, H*2^levels, W*2^levels).
        """
        assert levels >= 0, "Decompression levels must be non-negative."
        _, _, _, height, width = frames.size()
        new_height, new_width = height * (2 ** levels), width * (2 ** levels)
        return F.interpolate(frames, size=(new_height, new_width), mode="bilinear", align_corners=False)

    @staticmethod
    def add_noise(tensor: torch.Tensor, noise_level: float) -> torch.Tensor:
        """
        Add Gaussian noise to the input tensor.

        Args:
            tensor (torch.Tensor): Input tensor of shape (B, T, C, H, W) or any compatible latent format.
            noise_level (float): Strength of Gaussian noise. Should be in the range [0, 1].

        Returns:
            torch.Tensor: Tensor after adding Gaussian noise, same shape as input.
        """
        assert 0 <= noise_level <= 1, "Noise level must be in [0, 1]."
        noise = torch.randn_like(tensor) * noise_level
        return tensor + noise

    @staticmethod
    def compute_corrective_noise(
        compressed_tensor: torch.Tensor, 
        scaling_factor: float, 
        covariance_structure: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply corrective Gaussian noise at pyramid stage jump points to ensure continuity of probability paths.

        Args:
            compressed_tensor (torch.Tensor): Input tensor from lower-resolution stage (B, T, C, H', W').
            scaling_factor (float): Scaling factor for renoising.
            covariance_structure (torch.Tensor): Covariance matrix defining correlated noise.

        Returns:
            torch.Tensor: Corrected tensor with renoising applied, compatible with higher resolution stages.
        """
        assert scaling_factor > 0, "Scaling factor must be positive."
        assert covariance_structure.shape[0] == compressed_tensor.size(-2), (
            "Covariance matrix size must match spatial dimensions of compressed tensor."
        )

        # Upsample to target resolution
        upsampled = F.interpolate(compressed_tensor, scale_factor=2, mode="nearest")
        
        # Generate Gaussian noise
        noise = torch.matmul(covariance_structure, torch.randn_like(upsampled))
        
        # Apply scaling and add noise
        corrected_tensor = (scaling_factor * upsampled) + noise
        return corrected_tensor

    @staticmethod
    def interpolate_flow_trajectory(
        start_tensor: torch.Tensor, 
        end_tensor: torch.Tensor, 
        timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform linear interpolation between flow trajectory tensors.

        Args:
            start_tensor (torch.Tensor): Starting point tensor (B, T, C, H, W).
            end_tensor (torch.Tensor): Endpoint tensor (B, T, C, H, W).
            timestep (torch.Tensor): Scalar specifying the interpolation position within [0, 1].

        Returns:
            torch.Tensor: Interpolated tensor at the given timestep.
        """
        assert start_tensor.shape == end_tensor.shape, "Start and end tensors must have the same shape."
        assert 0 <= timestep <= 1, "Timestep must be between 0 and 1."
        
        # Linear interpolation
        interpolated_tensor = timestep * end_tensor + (1 - timestep) * start_tensor
        return interpolated_tensor
