import torch
import torch.nn as nn
import random
from typing import Tuple, List, Dict, Any, Optional

# Assuming Config, VideoVAE, downsample, upsample are available from other modules
# To avoid circular imports, these are usually imported in main.py and passed around,
# or accessed through a global config object.
# For standalone testing/linting, we'll use minimal stubs if actual imports fail.
try:
    from config import Config
    from vae import VideoVAE
    from utils import downsample, upsample
except ImportError:
    print("Warning: Could not import Config, VideoVAE, downsample, upsample directly. Using stubs.")

    class Config:
        """Minimal stub for Config."""
        def __init__(self):
            self.model = self.ModelConfig()
            self.compute = self.ComputeConfig()

        class ModelConfig:
            pyramid_stages: int = 3
            spatial_pyramid_time_windows: List[Tuple[float, float]] = [
                (0.5, 1.0), (0.2, 2/3), (0.0, 1/3) # Example values (s_k, e_k) for k=0,1,2
            ]

        class ComputeConfig:
            device: str = "cpu"

    class VideoVAE(nn.Module):
        """Minimal stub for VideoVAE."""
        def __init__(self):
            super().__init__()
        # Placeholder methods to allow compilation, actual logic not implemented here
        def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return x, x, x
        def decode(self, latents: torch.Tensor) -> torch.Tensor:
            return latents

    def downsample(tensor: torch.Tensor, factor: int, mode: str = "trilinear") -> torch.Tensor:
        """Stub for utils.downsample."""
        if tensor.ndim == 5: # Video (B, C, T, H, W)
            return torch.nn.functional.interpolate(tensor, size=(tensor.shape[2] // factor, tensor.shape[3] // factor, tensor.shape[4] // factor), mode=mode, align_corners=False)
        elif tensor.ndim == 4: # Image (B, C, H, W)
            return torch.nn.functional.interpolate(tensor, size=(tensor.shape[2] // factor, tensor.shape[3] // factor), mode='bilinear', align_corners=False)
        else:
            raise NotImplementedError("Stub downsample for other tensor dimensions not implemented.")

    def upsample(tensor: torch.Tensor, factor: int, mode: str = "trilinear") -> torch.Tensor:
        """Stub for utils.upsample."""
        if tensor.ndim == 5: # Video (B, C, T, H, W)
            return torch.nn.functional.interpolate(tensor, size=(tensor.shape[2] * factor, tensor.shape[3] * factor, tensor.shape[4] * factor), mode=mode, align_corners=False)
        elif tensor.ndim == 4: # Image (B, C, H, W)
            return torch.nn.functional.interpolate(tensor, size=(tensor.shape[2] * factor, tensor.shape[3] * factor), mode='bilinear', align_corners=False)
        else:
            raise NotImplementedError("Stub upsample for other tensor dimensions not implemented.")


class PyramidFlowMatcher:
    """
    Manages the logic for Pyramidal Flow Matching, including calculation of pyramid
    endpoints for training and the renoising process during inference.
    """
    def __init__(self, config: Config, vae: VideoVAE):
        """
        Initializes the PyramidFlowMatcher.

        Args:
            config (Config): The global configuration object.
            vae (VideoVAE): An instance of the VideoVAE for latent space operations.
        """
        self.config = config
        self.vae = vae
        self.K = config.model.pyramid_stages
        self.spatial_pyramid_timesteps = config.model.spatial_pyramid_time_windows
        self.device = torch.device(config.compute.device)

        if not self.spatial_pyramid_timesteps:
            # If config derivation failed or was not intended, derive them here as fallback
            # The config.py is designed to derive these, so this block should ideally not be hit.
            # But as a fallback for robustness, or if a user provides K but no windows.
            self._derive_pyramid_timesteps_fallback()

        # Basic validation of derived timesteps
        if len(self.spatial_pyramid_timesteps) != self.K:
            raise ValueError(f"Number of pyramid time windows ({len(self.spatial_pyramid_timesteps)}) "
                             f"does not match K ({self.K}) from config.")
        for k_idx, (s_k, e_k) in enumerate(self.spatial_pyramid_timesteps):
            if not (0.0 <= s_k < e_k <= 1.0):
                raise ValueError(f"Invalid time window (s_k={s_k}, e_k={e_k}) for stage {k_idx}. "
                                 "Must satisfy 0 <= s_k < e_k <= 1.")
            if k_idx == self.K - 1 and s_k != 0.0:
                print(f"Warning: Coarsest stage (k={k_idx}) s_k is {s_k}, but expected 0.0 for starting from noise.")


    def _derive_pyramid_timesteps_fallback(self):
        """
        Fallback method to derive (s_k, e_k) tuples for each pyramid stage k.
        This follows the derivation logic from the plan, assuming k=0 is finest, k=K-1 is coarsest.
        This method is only called if config.model.spatial_pyramid_time_windows is empty.
        """
        K = self.K
        derived_time_windows = []

        if K > 0:
            e_values: Dict[int, float] = {}
            e_values[0] = 1.0 # Finest stage ends at t=1.0
            for k_idx in range(1, K):
                e_values[k_idx] = 1.0 - (k_idx / K) # e.g., K=3: e0=1.0, e1=2/3, e2=1/3
            
            s_values: Dict[int, float] = {}
            s_values[K-1] = 0.0 # Coarsest stage starts at t=0.0 (pure noise)
            
            # Calculate s_k for k from 0 to K-2 using the derived relation
            # e_{k+1} = (2 * s_k) / (1 + s_k)  => s_k = e_{k+1} / (2 - e_{k+1})
            for k_idx in range(K-2, -1, -1): # Iterate k_idx from K-2 down to 0
                s_k = e_values[k_idx+1] / (2.0 - e_values[k_idx+1])
                s_values[k_idx] = s_k

            # Assemble (s_k, e_k) for each stage, ordered k=0 (finest) to k=K-1 (coarsest)
            for k_idx in range(K):
                derived_time_windows.append((s_values[k_idx], e_values[k_idx]))
        
        self.spatial_pyramid_timesteps = derived_time_windows
        print(f"Derived pyramid time windows (fallback): {self.spatial_pyramid_timesteps}")

    def sample_pyramid_stage(self) -> int:
        """
        Randomly samples a pyramid stage index for training.

        Returns:
            int: The sampled pyramid stage index (0 to K-1).
        """
        return random.randint(0, self.K - 1)

    def get_pyramid_timesteps(self, k_stage: int) -> Tuple[float, float]:
        """
        Retrieves the (s_k, e_k) tuple for a specified pyramid stage.

        Args:
            k_stage (int): The index of the pyramid stage (0 for finest, K-1 for coarsest).

        Returns:
            Tuple[float, float]: A tuple (s_k, e_k) representing the start and end times
                                 for the given stage k.
        """
        if not (0 <= k_stage < self.K):
            raise ValueError(f"k_stage must be between 0 and {self.K-1}, but got {k_stage}.")
        return self.spatial_pyramid_timesteps[k_stage]

    def compute_pyramid_endpoints(
        self, x1: torch.Tensor, k_stage: int, t_prime: float, noise: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the endpoints (hat_x_ek, hat_x_sk) and the target velocity vector for the
        flow matching objective during training, based on Eqs. (9) and (10) from the paper.

        Args:
            x1 (torch.Tensor): The clean, full-resolution data latent from the VAE.
                               Shape: (B, C, T_latent, H_latent, W_latent).
            k_stage (int): The current pyramid stage index (0 for finest, K-1 for coarsest).
            t_prime (float): The rescaled timestep (0 to 1) for linear interpolation within the stage.
            noise (torch.Tensor): A full-resolution noise tensor (matching x1's shape)
                                  sampled from N(0, I).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - hat_x_ek (torch.Tensor): The end point latent for the current stage.
            - hat_x_sk (torch.Tensor): The start point latent for the current stage.
            - target_vector (torch.Tensor): The target velocity vector (hat_x_ek - hat_x_sk).
        """
        s_k, e_k = self.get_pyramid_timesteps(k_stage)

        # Downsampling factor for the current stage's resolution (2^k_stage)
        down_factor_current_stage = self.get_down_factor(k_stage)
        # Downsampling factor for the resolution 'below' the current stage (2^(k_stage+1))
        down_factor_next_coarser_stage = self.get_down_factor(k_stage + 1)
        
        # Determine interpolation mode based on tensor dimensions
        interp_mode = 'trilinear' if x1.ndim == 5 else 'bilinear'

        # Downsample x1 and noise to the resolution corresponding to the current stage (e_k)
        downsampled_x1_ek = downsample(x1, down_factor_current_stage, mode=interp_mode)
        downsampled_noise_ek = downsample(noise, down_factor_current_stage, mode=interp_mode)

        # Compute hat_x_ek (End Point, Eq. 9)
        hat_x_ek = e_k * downsampled_x1_ek + (1 - e_k) * downsampled_noise_ek

        # Prepare Up(Down(x1, 2^(k+1))) for hat_x_sk (Start Point, Eq. 10)
        # Downsample x1 to the resolution of the next coarser stage (2^(k_stage+1))
        downsampled_x1_sk_base = downsample(x1, down_factor_next_coarser_stage, mode=interp_mode)
        # Upsample it back to the resolution of the current stage (by a factor of 2)
        upsampled_downsampled_x1_sk = upsample(downsampled_x1_sk_base, factor=2, mode=interp_mode)
        
        # Compute hat_x_sk (Start Point, Eq. 10)
        hat_x_sk = s_k * upsampled_downsampled_x1_sk + (1 - s_k) * downsampled_noise_ek

        # Calculate the target velocity vector
        target_vector = hat_x_ek - hat_x_sk
        
        return hat_x_ek, hat_x_sk, target_vector

    def apply_renoising(self, prev_latent_ek: torch.Tensor, current_k: int, prev_ek_time: float) -> torch.Tensor:
        """
        Implements the renoising step during inference (Eq. 15), transitioning from
        the endpoint of a coarser stage to the starting point of a finer stage.

        Args:
            prev_latent_ek (torch.Tensor): The generated latent (hat_x_e_{k+1} in the paper's notation)
                                           from the *previous*, coarser resolution pyramid stage.
            current_k (int): The index of the *current*, finer resolution stage (k in the paper's notation)
                             for which we are computing the starting point hat_x_sk.
            prev_ek_time (float): The end time 'e_{k+1}' of the previous (coarser) stage.

        Returns:
            torch.Tensor: The computed starting latent (hat_x_sk) for the current (finer) stage.
        """
        s_current_k, _ = self.get_pyramid_timesteps(current_k)
        
        # Verify the consistency of prev_ek_time with s_current_k as per Appendix A
        # e_{k+1} = (2 * s_k) / (1 + s_k)
        expected_prev_ek_time = (2.0 * s_current_k) / (1.0 + s_current_k)
        if not torch.isclose(torch.tensor(prev_ek_time), torch.tensor(expected_prev_ek_time), atol=1e-4):
            print(f"Warning: Renoising time consistency check failed for stage k={current_k}. "
                  f"Actual prev_ek_time={prev_ek_time:.4f}, expected={expected_prev_ek_time:.4f} "
                  f"based on s_k={s_current_k:.4f}.")

        # Determine interpolation mode based on tensor dimensions
        interp_mode = 'trilinear' if prev_latent_ek.ndim == 5 else 'bilinear'

        # Upsample the latent from the coarser stage's resolution to the current stage's resolution (factor=2)
        upsampled_latent = upsample(prev_latent_ek, factor=2, mode=interp_mode)

        # Calculate coefficients for Eq. (15)
        # hat_x_sk = (1 + s_k) / 2 * Up(hat_x_e_{k+1}) + (sqrt(3) * (1 - s_k)) / 2 * n'
        sk_coeff = (1.0 + s_current_k) / 2.0
        alpha_n_prime_coeff = (torch.sqrt(torch.tensor(3.0, device=self.device)) * (1.0 - s_current_k)) / 2.0

        # Sample corrective noise n'
        n_prime = torch.randn_like(upsampled_latent, device=self.device)

        # Compute hat_x_sk
        next_start_latent_sk = sk_coeff * upsampled_latent + alpha_n_prime_coeff * n_prime

        return next_start_latent_sk

    def get_down_factor(self, k_stage: int) -> int:
        """
        Returns the spatial downsampling factor for a given pyramid stage k_stage.
        Stage k=0 means no downsampling (factor 1). Stage k=1 means 2x downsampling (factor 2).
        Stage k=2 means 4x downsampling (factor 4).
        Generally, factor = 2^k_stage.

        Args:
            k_stage (int): The index of the pyramid stage.

        Returns:
            int: The downsampling factor (2^k_stage).
        """
        if k_stage < 0:
            raise ValueError(f"Pyramid stage index k_stage cannot be negative, but got {k_stage}.")
        return 2**k_stage


if __name__ == "__main__":
    print("--- Testing PyramidFlowMatcher ---")

    # 1. Setup dummy components
    dummy_config = Config()
    dummy_vae = VideoVAE() # Use stub VAE

    # Ensure config.py derived values are as expected for K=3
    # k=0: (0.5, 1.0)
    # k=1: (0.2, 2/3)
    # k=2: (0.0, 1/3)
    expected_windows = [(0.5, 1.0), (0.2, 2/3), (0.0, 1/3)]
    dummy_config.model.spatial_pyramid_time_windows = expected_windows

    pyramid_matcher = PyramidFlowMatcher(dummy_config, dummy_vae)
    print(f"Initialized PyramidFlowMatcher with K={pyramid_matcher.K}")
    print(f"Spatial pyramid time windows: {pyramid_matcher.spatial_pyramid_timesteps}")
    assert len(pyramid_matcher.spatial_pyramid_timesteps) == 3

    # 2. Test sample_pyramid_stage
    sampled_stage = pyramid_matcher.sample_pyramid_stage()
    print(f"Sampled pyramid stage: {sampled_stage}")
    assert 0 <= sampled_stage < pyramid_matcher.K

    # 3. Test get_pyramid_timesteps
    s0, e0 = pyramid_matcher.get_pyramid_timesteps(0)
    s1, e1 = pyramid_matcher.get_pyramid_timesteps(1)
    s2, e2 = pyramid_matcher.get_pyramid_timesteps(2)
    print(f"Stage 0 timesteps: s={s0:.4f}, e={e0:.4f}")
    print(f"Stage 1 timesteps: s={s1:.4f}, e={e1:.4f}")
    print(f"Stage 2 timesteps: s={s2:.4f}, e={e2:.4f}")
    assert torch.isclose(torch.tensor(s0), torch.tensor(0.5))
    assert torch.isclose(torch.tensor(e0), torch.tensor(1.0))
    assert torch.isclose(torch.tensor(s1), torch.tensor(0.2))
    assert torch.isclose(torch.tensor(e1), torch.tensor(2/3))
    assert torch.isclose(torch.tensor(s2), torch.tensor(0.0))
    assert torch.isclose(torch.tensor(e2), torch.tensor(1/3))

    # 4. Test get_down_factor
    factor0 = pyramid_matcher.get_down_factor(0)
    factor1 = pyramid_matcher.get_down_factor(1)
    factor2 = pyramid_matcher.get_down_factor(2)
    print(f"Down factor for stage 0: {factor0}, stage 1: {factor1}, stage 2: {factor2}")
    assert factor0 == 1
    assert factor1 == 2
    assert factor2 == 4

    # 5. Test compute_pyramid_endpoints
    print("\nTesting compute_pyramid_endpoints...")
    batch_size = 1
    latent_channels = 4
    # Example full-resolution latent (e.g., from VAE input 64x64, 8x8x8 compression)
    T_latent, H_latent, W_latent = 2, 16, 16
    x1_full_res = torch.randn(batch_size, latent_channels, T_latent, H_latent, W_latent, device=pyramid_matcher.device)
    noise_full_res = torch.randn_like(x1_full_res, device=pyramid_matcher.device)
    t_prime = 0.5 # Example rescaled timestep

    # Compute for k_stage = 0 (finest)
    hat_x_ek_0, hat_x_sk_0, target_vec_0 = pyramid_matcher.compute_pyramid_endpoints(
        x1_full_res, k_stage=0, t_prime=t_prime, noise=noise_full_res
    )
    print(f"Stage 0: hat_x_ek_0 shape: {hat_x_ek_0.shape}, hat_x_sk_0 shape: {hat_x_sk_0.shape}")
    assert hat_x_ek_0.shape == (batch_size, latent_channels, T_latent, H_latent, W_latent)
    assert hat_x_sk_0.shape == (batch_size, latent_channels, T_latent, H_latent, W_latent)

    # Compute for k_stage = 1 (middle)
    # Target resolution: T_latent/2, H_latent/2, W_latent/2
    hat_x_ek_1, hat_x_sk_1, target_vec_1 = pyramid_matcher.compute_pyramid_endpoints(
        x1_full_res, k_stage=1, t_prime=t_prime, noise=noise_full_res
    )
    print(f"Stage 1: hat_x_ek_1 shape: {hat_x_ek_1.shape}, hat_x_sk_1 shape: {hat_x_sk_1.shape}")
    assert hat_x_ek_1.shape == (batch_size, latent_channels, T_latent // 2, H_latent // 2, W_latent // 2)
    assert hat_x_sk_1.shape == (batch_size, latent_channels, T_latent // 2, H_latent // 2, W_latent // 2)

    # Compute for k_stage = 2 (coarsest)
    # Target resolution: T_latent/4, H_latent/4, W_latent/4
    hat_x_ek_2, hat_x_sk_2, target_vec_2 = pyramid_matcher.compute_pyramid_endpoints(
        x1_full_res, k_stage=2, t_prime=t_prime, noise=noise_full_res
    )
    print(f"Stage 2: hat_x_ek_2 shape: {hat_x_ek_2.shape}, hat_x_sk_2 shape: {hat_x_sk_2.shape}")
    assert hat_x_ek_2.shape == (batch_size, latent_channels, T_latent // 4, H_latent // 4, W_latent // 4)
    assert hat_x_sk_2.shape == (batch_size, latent_channels, T_latent // 4, H_latent // 4, W_latent // 4)


    # 6. Test apply_renoising (inference step)
    print("\nTesting apply_renoising...")
    # Inference proceeds from coarser to finer.
    # Start with coarsest stage (k=2) output (hat_x_e_2), derive start for middle stage (k=1) (hat_x_s_1)
    next_start_latent_s1 = pyramid_matcher.apply_renoising(
        prev_latent_ek=hat_x_ek_2, current_k=1, prev_ek_time=e2
    )
    print(f"Renoised s1 latent shape (from e2): {next_start_latent_s1.shape}")
    assert next_start_latent_s1.shape == hat_x_ek_1.shape # Should match resolution of stage 1

    # Take middle stage (k=1) output (hat_x_e_1), derive start for finest stage (k=0) (hat_x_s_0)
    next_start_latent_s0 = pyramid_matcher.apply_renoising(
        prev_latent_ek=hat_x_ek_1, current_k=0, prev_ek_time=e1
    )
    print(f"Renoised s0 latent shape (from e1): {next_start_latent_s0.shape}")
    assert next_start_latent_s0.shape == hat_x_ek_0.shape # Should match resolution of stage 0

    print("\nAll PyramidFlowMatcher tests completed successfully!")
