"""
Pyramidal Flow Matching for Efficient Video Generative Modeling
Core algorithm implementation based on the paper.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class PyramidalFlowMatching:
    """
    Implements the Pyramidal Flow Matching algorithm from the paper.
    
    The algorithm divides the denoising trajectory into K pyramid stages,
    where each stage operates at a different spatial resolution. Only the
    final stage operates at full resolution.
    
    Key equations from the paper:
    - Eq. (5): Piecewise flow within each pyramid stage
    - Eq. (9-10): Coupled endpoint sampling for training
    - Eq. (11): Unified flow matching objective
    - Eq. (15): Renoising rule at jump points during inference
    """
    
    def __init__(
        self,
        num_stages: int = 3,
        stage_time_windows: Optional[List[Tuple[float, float]]] = None,
    ):
        """
        Args:
            num_stages: Number of pyramid stages K. Default is 3 as in the paper.
            stage_time_windows: List of (s_k, e_k) time windows for each stage.
                If None, uses uniform partitioning of [0, 1].
        """
        self.num_stages = num_stages
        
        if stage_time_windows is None:
            # Uniform partitioning of [0, 1] into K stages
            # Stage 0 is the lowest resolution (most compressed)
            # Stage K-1 is the full resolution
            step = 1.0 / num_stages
            self.stage_time_windows = [
                (k * step, (k + 1) * step) for k in range(num_stages)
            ]
        else:
            assert len(stage_time_windows) == num_stages
            self.stage_time_windows = stage_time_windows
    
    def get_stage_for_timestep(self, t: float) -> int:
        """Get the pyramid stage index for a given timestep t in [0, 1]."""
        for k, (s_k, e_k) in enumerate(self.stage_time_windows):
            if s_k <= t <= e_k:
                return k
        return self.num_stages - 1
    
    def rescale_timestep(self, t: float, stage: int) -> float:
        """
        Rescale timestep t to t' = (t - s_k) / (e_k - s_k) within stage k.
        """
        s_k, e_k = self.stage_time_windows[stage]
        return (t - s_k) / (e_k - s_k)
    
    def downsample(self, x: torch.Tensor, factor: int) -> torch.Tensor:
        """
        Downsample spatial dimensions by given factor using bilinear interpolation.
        
        Args:
            x: Tensor of shape (B, C, H, W) or (B, C, T, H, W)
            factor: Downsampling factor
        
        Returns:
            Downsampled tensor
        """
        if factor == 1:
            return x
        
        if x.dim() == 4:
            # Image: (B, C, H, W)
            B, C, H, W = x.shape
            new_H, new_W = H // factor, W // factor
            return F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)
        elif x.dim() == 5:
            # Video: (B, C, T, H, W)
            B, C, T, H, W = x.shape
            new_H, new_W = H // factor, W // factor
            # Reshape to apply 2D interpolation
            x_reshaped = x.view(B * T, C, H, W)
            x_down = F.interpolate(x_reshaped, size=(new_H, new_W), mode='bilinear', align_corners=False)
            return x_down.view(B, C, T, new_H, new_W)
        else:
            raise ValueError(f"Unsupported tensor shape: {x.shape}")
    
    def upsample(self, x: torch.Tensor, target_size: Tuple) -> torch.Tensor:
        """
        Upsample spatial dimensions to target size using nearest neighbor.
        
        Args:
            x: Tensor of shape (B, C, H, W) or (B, C, T, H, W)
            target_size: Target (H, W) size
        
        Returns:
            Upsampled tensor
        """
        if x.dim() == 4:
            return F.interpolate(x, size=target_size, mode='nearest')
        elif x.dim() == 5:
            B, C, T, H, W = x.shape
            x_reshaped = x.view(B * T, C, H, W)
            x_up = F.interpolate(x_reshaped, size=target_size, mode='nearest')
            return x_up.view(B, C, T, *target_size)
        else:
            raise ValueError(f"Unsupported tensor shape: {x.shape}")
    
    def sample_training_pair(
        self,
        x1: torch.Tensor,
        stage: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a training pair (x_sk, x_ek) for a given pyramid stage.
        
        Implements Eqs. (9) and (10) from the paper:
        - End: x_ek = e_k * Down(x1, 2^k) + (1 - e_k) * n
        - Start: x_sk = s_k * Up(Down(x1, 2^(k+1))) + (1 - s_k) * n
        
        The noise n is shared (coupled) between endpoints to improve
        trajectory straightness.
        
        Args:
            x1: Clean data latent at full resolution (B, C, H, W) or (B, C, T, H, W)
            stage: Pyramid stage index k (0 = lowest resolution, K-1 = full resolution)
        
        Returns:
            Tuple of (x_sk, x_ek, t_prime, target_velocity)
            where t_prime is a random rescaled timestep within the stage
        """
        s_k, e_k = self.stage_time_windows[stage]
        
        # Downsampling factors: stage k uses factor 2^k
        # Stage 0 (lowest res): factor 2^(K-1)
        # Stage K-1 (full res): factor 1
        # The k-th stage endpoint is at resolution 2^k relative to full
        # Actually: stage k endpoint is Down(x1, 2^(K-1-k)) based on paper notation
        # where K stages go from 0 (lowest) to K-1 (full)
        
        # From paper: k-th time window endpoint is Down(x1, 2^k)
        # and start is Up(Down(x1, 2^(k+1)))
        # Here k=0 is the last stage (full resolution), k=K-1 is first stage
        # Let's use the paper's convention where k goes from K-1 down to 0
        
        # For stage index s (0=lowest res, K-1=full res):
        # k in paper = K-1-s
        k = self.num_stages - 1 - stage
        
        # Sample shared noise n ~ N(0, I) at the resolution of the endpoint
        # Endpoint resolution: Down(x1, 2^k)
        down_factor_end = 2 ** k
        x1_down_end = self.downsample(x1, down_factor_end)
        
        # Sample noise at endpoint resolution
        n = torch.randn_like(x1_down_end)
        
        # Compute endpoint: x_ek = e_k * Down(x1, 2^k) + (1 - e_k) * n
        x_ek = e_k * x1_down_end + (1 - e_k) * n
        
        # Compute start point: x_sk = s_k * Up(Down(x1, 2^(k+1))) + (1 - s_k) * n
        if k + 1 <= self.num_stages:
            down_factor_start = 2 ** (k + 1)
            x1_down_start = self.downsample(x1, down_factor_start)
            # Upsample to endpoint resolution
            if x1.dim() == 4:
                target_size = x1_down_end.shape[-2:]
            else:
                target_size = x1_down_end.shape[-2:]
            x1_up_start = self.upsample(x1_down_start, target_size)
        else:
            # Last stage: start is at same resolution as end
            x1_up_start = x1_down_end
        
        x_sk = s_k * x1_up_start + (1 - s_k) * n
        
        # Sample a random timestep t' within [0, 1] for this stage
        t_prime = torch.rand(x1.shape[0], device=x1.device)
        
        # Interpolate to get x_t at the sampled timestep
        # x_t = t' * x_ek + (1 - t') * x_sk  (linear interpolation within stage)
        t_prime_expanded = t_prime.view(-1, *([1] * (x1.dim() - 1)))
        x_t = t_prime_expanded * x_ek + (1 - t_prime_expanded) * x_sk
        
        # Target velocity: u_t(x_t | x1) = x_ek - x_sk
        target_velocity = x_ek - x_sk
        
        return x_sk, x_ek, x_t, t_prime, target_velocity
    
    def compute_flow_loss(
        self,
        predicted_velocity: torch.Tensor,
        target_velocity: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the flow matching loss (Eq. 11 from paper):
        E_{k,t,(x_ek, x_sk)} || v_t(x_t) - (x_ek - x_sk) ||^2
        
        Args:
            predicted_velocity: Model's predicted velocity
            target_velocity: Target velocity (x_ek - x_sk)
        
        Returns:
            MSE loss
        """
        return F.mse_loss(predicted_velocity, target_velocity)
    
    def renoise_at_jump_point(
        self,
        x_ek_plus1: torch.Tensor,
        s_k: float,
        target_size: Optional[Tuple] = None,
    ) -> torch.Tensor:
        """
        Apply the renoising rule at jump points between pyramid stages.
        
        Implements Eq. (15) from the paper:
        x_sk = (1 + s_k) / 2 * Up(x_{e_{k+1}}) + sqrt(3) * (1 - s_k) / 2 * n'
        
        with e_{k+1} = 2 * s_k / (1 + s_k)
        
        This ensures continuity of the probability path between stages.
        
        Args:
            x_ek_plus1: Endpoint of the previous (lower resolution) stage
            s_k: Starting timestep of the current stage
            target_size: Target spatial size for upsampling
        
        Returns:
            Starting point x_sk for the current stage
        """
        # Upsample the previous endpoint
        if target_size is not None:
            x_up = self.upsample(x_ek_plus1, target_size)
        else:
            # Double the spatial resolution
            if x_ek_plus1.dim() == 4:
                B, C, H, W = x_ek_plus1.shape
                x_up = self.upsample(x_ek_plus1, (H * 2, W * 2))
            else:
                B, C, T, H, W = x_ek_plus1.shape
                x_up = self.upsample(x_ek_plus1, (H * 2, W * 2))
        
        # Sample corrective noise n' ~ N(0, I)
        # The corrective noise has a special covariance structure (Eq. 14)
        # but for practical implementation we use standard Gaussian
        n_prime = torch.randn_like(x_up)
        
        # Apply renoising rule: Eq. (15)
        # x_sk = (1 + s_k) / 2 * Up(x_{e_{k+1}}) + sqrt(3) * (1 - s_k) / 2 * n'
        x_sk = (1 + s_k) / 2 * x_up + math.sqrt(3) * (1 - s_k) / 2 * n_prime
        
        return x_sk
    
    def get_jump_point_timestep(self, s_k: float) -> float:
        """
        Get the endpoint timestep e_{k+1} that corresponds to starting point s_k.
        
        From Eq. (26) in the paper: e_{k+1} = 2 * s_k / (1 + s_k)
        
        Args:
            s_k: Starting timestep of stage k
        
        Returns:
            e_{k+1}: Endpoint timestep of stage k+1
        """
        return 2 * s_k / (1 + s_k)


class TemporalPyramidCondition:
    """
    Implements the temporal pyramid condition for autoregressive video generation.
    
    From Section 3.3 of the paper: uses progressively compressed, lower-resolution
    history as conditions to reduce computational overhead.
    
    The history condition is:
    ... -> Down(x_{t'}^{i-2}, 2^{k+1}) -> Down(x_{t'}^{i-1}, 2^k) -> x_t^i
    
    where t' indicates small noise added to history latents during training.
    """
    
    def __init__(
        self,
        num_history_stages: int = 2,
        noise_strength_range: Tuple[float, float] = (0.0, 1/3),
    ):
        """
        Args:
            num_history_stages: Number of history frames with different resolutions
            noise_strength_range: Range for corruption noise strength during training
                (uniformly sampled from this range as per paper)
        """
        self.num_history_stages = num_history_stages
        self.noise_strength_range = noise_strength_range
    
    def prepare_history_condition(
        self,
        history_latents: List[torch.Tensor],
        current_stage: int,
        num_pyramid_stages: int,
        training: bool = True,
    ) -> List[torch.Tensor]:
        """
        Prepare the temporal pyramid history condition.
        
        Args:
            history_latents: List of clean history latents [x1^{i-N}, ..., x1^{i-1}]
            current_stage: Current pyramid stage (0=lowest, K-1=full resolution)
            num_pyramid_stages: Total number of pyramid stages K
            training: Whether in training mode (adds noise to history)
        
        Returns:
            List of compressed history latents at different resolutions
        """
        compressed_history = []
        
        for idx, hist_latent in enumerate(history_latents):
            # Determine compression factor for this history frame
            # More recent frames are less compressed
            # From paper: Down(x^{i-2}, 2^{k+1}) -> Down(x^{i-1}, 2^k)
            history_position = len(history_latents) - 1 - idx  # 0 = most recent
            
            # Compression factor increases for older frames
            k = current_stage + history_position + 1
            k = min(k, num_pyramid_stages)  # Cap at max stages
            down_factor = 2 ** k
            
            # Downsample the history latent
            if hist_latent.dim() == 4:
                B, C, H, W = hist_latent.shape
                new_H, new_W = max(H // down_factor, 1), max(W // down_factor, 1)
                hist_compressed = F.interpolate(
                    hist_latent, size=(new_H, new_W), mode='bilinear', align_corners=False
                )
            elif hist_latent.dim() == 5:
                B, C, T, H, W = hist_latent.shape
                new_H, new_W = max(H // down_factor, 1), max(W // down_factor, 1)
                hist_reshaped = hist_latent.view(B * T, C, H, W)
                hist_down = F.interpolate(
                    hist_reshaped, size=(new_H, new_W), mode='bilinear', align_corners=False
                )
                hist_compressed = hist_down.view(B, C, T, new_H, new_W)
            else:
                hist_compressed = hist_latent
            
            # Add corruption noise during training to mitigate error accumulation
            if training:
                noise_strength = torch.rand(1).item() * (
                    self.noise_strength_range[1] - self.noise_strength_range[0]
                ) + self.noise_strength_range[0]
                
                noise = torch.randn_like(hist_compressed)
                hist_compressed = (
                    math.sqrt(1 - noise_strength ** 2) * hist_compressed
                    + noise_strength * noise
                )
            
            compressed_history.append(hist_compressed)
        
        return compressed_history
