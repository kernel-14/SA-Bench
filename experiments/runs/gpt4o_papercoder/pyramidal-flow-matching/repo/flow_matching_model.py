# flow_matching_model.py

import torch
import torch.nn as nn
from typing import Dict, Tuple
from pyramid_utils import PyramidUtils

class FlowMatchingModel(nn.Module):
    """
    FlowMatchingModel implements spatial and temporal pyramid flow matching for efficient video generation.
    Handles forward pass and computation of flow matching losses across pyramid stages.
    """
    def __init__(self, num_stages: int = 3, params: Dict = None) -> None:
        """
        Initializes the flow matching model.

        Args:
            num_stages (int): Number of pyramid stages for spatial flow matching.
            params (dict): Flow matching parameters (e.g., noise levels, compression ratios, etc.).
        """
        super(FlowMatchingModel, self).__init__()
        self.num_stages = num_stages
        self.noise_level_range = params.get("flow_matching", {}).get("noise_level", [0.0, 0.33])
        self.downsample_ratio = params.get("vae", {}).get("downsampling_ratio", [8, 8, 8])

        # Velocity field model (trainable component)
        self.velocity_field = nn.Sequential(
            nn.Linear(512, 512),  # Compression into latent dimensionality if required
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        
    def forward(self, latent: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for generating flow matching predictions across spatial and temporal pyramid stages.

        Args:
            latent (torch.Tensor): Input latent tensor, typically the compressed representation of video frames.
            noise (torch.Tensor): Gaussian noise tensor for stochastic endpoints in the pyramid flow.

        Returns:
            torch.Tensor: Final high-resolution prediction after processing through pyramid stages.
        """
        batch_size, channels, height, width, timesteps = latent.size()
        current_latent = latent  # Initialize with input latent

        for stage in range(self.num_stages):
            # Compute endpoints for current stage
            compressed_latent = PyramidUtils.compress_frames(current_latent, levels=1)  # Downsample
            decompressed_latent = PyramidUtils.decompress(compressed_latent, levels=1)  # Upsample back

            # Add corrective noise for continuity at jump points
            corrective_noise = self._add_corrective_noise(decompressed_latent, stage)
            current_latent = decompressed_latent + corrective_noise

            # Generate flow noise paths
            flow_noise = self.velocity_field(current_latent)
            current_latent = flow_noise + noise  # Combine flow trajectory with input noise

        return current_latent

    def compute_loss(self, data: torch.Tensor, noise: torch.Tensor, timestep: int) -> float:
        """
        Computes the unified flow matching loss for spatial and temporal pyramid stages.

        Args:
            data (torch.Tensor): Ground-truth high-resolution latent frames.
            noise (torch.Tensor): Gaussian noise sampled for each pyramid stage endpoint.
            timestep (int): Current time step in the flow trajectory.

        Returns:
            float: Loss value based on flow matching objectives.
        """
        batch_size, channels, height, width, timesteps = data.size()
        loss = 0.0

        for stage in range(self.num_stages):
            # Sample Gaussian noise for current stage
            noise_sample = noise * torch.rand_like(noise).uniform_(*self.noise_level_range)

            # Compute start and end points for flow interpolation
            start = PyramidUtils.compress_frames(data, levels=stage + 1)
            end = PyramidUtils.decompress(start, levels=stage + 1)

            # Interpolate between noisy start and cleaner end points
            interpolated = self._interpolate(start, end, timestep / self.num_stages)

            # Compute loss component for this stage
            flow_noise = self.velocity_field(interpolated)
            stage_loss = torch.nn.functional.mse_loss(flow_noise, end - start)  # Flow Matching Loss
            loss += stage_loss

        # Average loss over all pyramid stages
        return loss / self.num_stages

    def _interpolate(self, start: torch.Tensor, end: torch.Tensor, t: float) -> torch.Tensor:
        """
        Perform linear interpolation between start and end tensors.

        Args:
            start (torch.Tensor): Starting tensor for interpolation.
            end (torch.Tensor): Ending tensor for interpolation.
            t (float): Interpolation factor, ranging between [0, 1].

        Returns:
            torch.Tensor: Interpolated tensor.
        """
        return t * end + (1 - t) * start

    def _add_corrective_noise(self, latent: torch.Tensor, stage: int) -> torch.Tensor:
        """
        Add corrective Gaussian noise for continuity across pyramid stages.

        Args:
            latent (torch.Tensor): Input tensor representing decompressed latent representation.
            stage (int): Current pyramid stage index.

        Returns:
            torch.Tensor: Tensor after corrective noise application.
        """
        # Upsample latent to match higher resolution
        upsampled_latent = PyramidUtils.decompress(latent, levels=1)
        
        # Generate noise with covariance-correcting factors
        noise = PyramidUtils.add_noise(upsampled_latent, self.noise_level_range[1])
        adjusted_latent = (1 / (stage + 1)) * upsampled_latent + noise

        return adjusted_latent


# Example Usage:
if __name__ == "__main__":
    from config import Config

    # Load config and initialize FlowMatchingModel
    config = Config("config.yaml")
    model_params = config.get_model_config()
    num_stages = model_params["flow_matching"]["num_stages"]

    flow_model = FlowMatchingModel(num_stages=num_stages, params=model_params)

    # Dummy Input (B, C, H, W, T)
    dummy_latent = torch.rand(2, 512, 16, 16, 8)  # Compressed latent tensor
    dummy_noise = torch.randn_like(dummy_latent)  # Gaussian noise
    
    # Forward pass
    prediction = flow_model(dummy_latent, dummy_noise)
    print("Prediction Shape:", prediction.shape)
    
    # Compute loss
    ground_truth = torch.rand_like(dummy_latent)
    loss = flow_model.compute_loss(ground_truth, dummy_noise, timestep=1)
    print("Loss:", loss)
