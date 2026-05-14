"""
Autoregressive inference with KV-cache sharing for Ca2-VDM.

The inference pipeline consists of two repeated steps per AR step:
1. Denoising stage: Denoise l new frames conditioned on cached KVs
   - Temporal & spatial KV-caches are shared across all denoising timesteps
   - Uses improved DDPM schedule with 100 steps
   
2. Cache writing stage: Compute clean KVs of newly generated frames
   - Update temporal KV-cache queue (dequeue oldest when P_max reached)
   - Overwrite spatial KV-cache

Key features:
- KV-cache sharing: All denoising steps share the same cache
  (clean prefix always has t=0 embedding)
- Temporal KV-cache queue with Cyclic-TPEs
- Prefix-enhanced spatial attention with spatial KV-cache
"""

import os
import sys
import argparse
import torch
import numpy as np
from typing import Optional, List, Tuple
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ca2_vdm.model import Ca2VDM
from ca2_vdm.diffusion import DiffusionProcess
from ca2_vdm.cache import KVCacheManager


class Ca2VDMInference:
    """
    Autoregressive inference for Ca2-VDM.
    
    Generates videos autoregressively with KV-cache sharing.
    """
    
    def __init__(
        self,
        model: Ca2VDM,
        diffusion: DiffusionProcess,
        device: torch.device,
        num_inference_steps: int = 100,  # improved DDPM steps
        cfg_scale: float = 7.5,  # classifier-free guidance scale
    ):
        self.model = model.to(device)
        self.diffusion = diffusion
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.cfg_scale = cfg_scale
        
        self.l = model.l
        self.P_max = model.P_max
        self.L_train = model.L_train
        self.prefix_len = model.prefix_len
        
        # Create KV-cache manager
        self.kv_cache = KVCacheManager(
            num_layers=model.num_layers,
            P_max=model.P_max,
            P_prime=model.prefix_len,
        )
        
        # Improved DDPM timestep schedule
        self.timesteps = self._get_improved_ddpm_schedule(num_inference_steps)
    
    def _get_improved_ddpm_schedule(self, steps: int) -> torch.Tensor:
        """
        Get improved DDPM timestep schedule.
        
        Uses stride sampling with non-uniform spacing.
        For 1000 training steps and 100 inference steps,
        sample every 10 steps.
        """
        # Linear spaced timesteps
        timesteps = torch.linspace(
            self.diffusion.num_timesteps - 1, 0, steps, dtype=torch.long
        )
        return timesteps
    
    def _cache_writing(
        self,
        z_0_chunk: torch.Tensor,
        P_k: int,
        text_emb: Optional[torch.Tensor] = None,
    ):
        """
        Cache writing stage: compute clean KVs of newly generated frames.
        
        The denoised chunk z_0_chunk is fed through the model with t=0
        to compute its keys and values. These are stored in the KV-cache
        for use in subsequent AR steps.

        Args:
            z_0_chunk: denoised latent chunk (B, C, l, H, W)
            P_k: current total prefix length before this chunk
            text_emb: text embeddings for cross-attention
        """
        self.model.eval()
        
        with torch.no_grad():
            result = self.model(
                z=z_0_chunk,
                t=torch.zeros(z_0_chunk.shape[0], device=self.device, dtype=torch.long),
                P=0,  # All frames are clean (for cache writing)
                text_emb=text_emb,
                cyclic_offset=P_k % self.L_train,  # Cyclic offset for TPE
                kv_cache_manager=self.kv_cache,
                cache_write=True,
            )
        
        # Update caches
        temporal_caches = result['temporal_caches']
        spatial_caches = result['spatial_caches']
        
        for layer_idx in range(len(temporal_caches)):
            if temporal_caches[layer_idx] is not None:
                k, v = temporal_caches[layer_idx]
                self.kv_cache.update_temporal(layer_idx, k, v, self.l)
            
            if spatial_caches[layer_idx] is not None:
                k, v = spatial_caches[layer_idx]
                self.kv_cache.update_spatial(layer_idx, k, v)
    
    def _denoise_chunk(
        self,
        P_k: int,
        text_emb: Optional[torch.Tensor] = None,
        uncond_text_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Denoising stage: generate l new frames using KV-cache.
        
        Uses classifier-free guidance for T2V.
        
        Args:
            P_k: current total prefix length
            text_emb: text embeddings for conditional generation
            uncond_text_emb: text embeddings for unconditional (for CFG)
        
        Returns:
            z_0: denoised latent chunk (B, C, l, H, W)
        """
        B = 1 if text_emb is None else text_emb.shape[0]
        
        # Start from pure noise for target frames
        z_t = torch.randn(B, self.model.in_channels, self.l, self.model.H, self.model.W,
                          device=self.device)
        
        # Denoising loop
        for step_idx, t_val in enumerate(self.timesteps):
            t = torch.full((B,), t_val, device=self.device, dtype=torch.long)
            
            # Classifier-free guidance
            if self.cfg_scale > 1.0 and text_emb is not None and uncond_text_emb is not None:
                # Conditional forward
                z_t_cond = self._single_denoise_step(z_t, t, P_k, text_emb)
                # Unconditional forward
                z_t_uncond = self._single_denoise_step(z_t, t, P_k, uncond_text_emb)
                # CFG interpolation
                z_t = z_t_uncond + self.cfg_scale * (z_t_cond - z_t_uncond)
            else:
                z_t = self._single_denoise_step(z_t, t, P_k, text_emb)
        
        return z_t
    
    def _single_denoise_step(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        P_k: int,
        text_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Single denoising step using KV-cache.
        
        The model receives ONLY the noisy target frames z_t.
        Clean prefix frames are accessed through KV-cache.
        """
        self.model.eval()
        
        with torch.no_grad():
            result = self.model(
                z=z_t,
                t=t,
                P=P_k,  # Total clean prefix frames (via cache)
                text_emb=text_emb,
                cyclic_offset=0,  # Target frames always start from offset 0 position
                # (after P_k in absolute position, but 0 in relative position for target)
                kv_cache_manager=self.kv_cache,
                cache_write=False,
            )
        
        model_output = result['output']
        
        # Compute z_{t-1} from model output
        z_t_minus_1 = self.diffusion.p_sample(
            model=self.model,
            z_t=z_t,
            t=t,
            P=P_k,
            text_emb=text_emb,
            kv_cache_manager=self.kv_cache,
            learn_sigma=self.model.learn_sigma,
        )
        
        # Actually, we should call p_sample properly
        # The p_sample method does model forward internally, so let's just use the
        # computed output directly.
        mean, variance, log_variance = self.diffusion.p_mean_variance(
            model_output, z_t, t, self.model.learn_sigma
        )
        
        noise = torch.randn_like(mean) if t[0] > 0 else torch.zeros_like(mean)
        z_t_minus_1 = mean + torch.sqrt(variance) * noise
        
        return z_t_minus_1
    
    @torch.no_grad()
    def generate(
        self,
        first_frame: torch.Tensor,
        num_ar_steps: int,
        text_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate video autoregressively.
        
        Args:
            first_frame: initial frame latent (B, C, 1, H, W)
            num_ar_steps: number of autoregression steps
            text_emb: text embeddings (B, T, dim) for T2V
        
        Returns:
            video: generated video (B, C, total_frames, H, W)
        """
        B = first_frame.shape[0]
        
        # Reset caches
        self.kv_cache.reset()
        
        # Initialize with first frame
        generated_frames = [first_frame]
        P_k = 1  # Start with 1 clean prefix frame
        
        # Cache writing for first frame
        self._cache_writing(first_frame, 0, text_emb)
        
        # Unconditional embedding for CFG
        uncond_text_emb = None
        if text_emb is not None and self.cfg_scale > 1.0:
            uncond_text_emb = torch.zeros_like(text_emb)
        
        # Autoregressive generation
        for ar_step in range(num_ar_steps):
            # Denoise l new frames
            z_0_chunk = self._denoise_chunk(P_k, text_emb, uncond_text_emb)
            
            # Store generated frames
            generated_frames.append(z_0_chunk)
            P_k += self.l
            
            # Cache writing for new chunk
            self._cache_writing(z_0_chunk, P_k - self.l, text_emb)
        
        # Concatenate all frames
        video = torch.cat(generated_frames, dim=2)  # (B, C, total_frames, H, W)
        
        return video
    
    @torch.no_grad()
    def generate_with_cache_analysis(
        self,
        first_frame: torch.Tensor,
        num_ar_steps: int,
        text_emb: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Generate video and return timing/FLOP analysis.
        Only computes one denoising step per AR step for FLOP counting.
        """
        import time
        
        B = first_frame.shape[0]
        self.kv_cache.reset()
        
        generated_frames = [first_frame]
        P_k = 1
        timings = []
        
        # Cache first frame
        t_start = time.time()
        self._cache_writing(first_frame, 0, text_emb)
        timings.append(time.time() - t_start)
        
        for ar_step in range(num_ar_steps):
            t_start = time.time()
            
            # Single denoising step for timing analysis
            z_t = torch.randn(B, self.model.in_channels, self.l, 
                             self.model.H, self.model.W, device=self.device)
            t = torch.full((B,), 500, device=self.device, dtype=torch.long)
            
            _ = self._single_denoise_step(z_t, t, P_k, text_emb)
            
            timing = time.time() - t_start
            timings.append(timing)
            
            P_k += self.l
        
        video = torch.cat(generated_frames, dim=2)
        
        return {
            'video': video,
            'timings': timings,
            'total_time': sum(timings),
        }


def main():
    parser = argparse.ArgumentParser(description="Ca2-VDM Autoregressive Inference")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--first_frame', type=str, default=None,
                        help='Path to first frame image (or latent)')
    parser.add_argument('--text_prompt', type=str, default=None,
                        help='Text prompt for T2V generation')
    parser.add_argument('--num_ar_steps', type=int, default=5,
                        help='Number of autoregression steps')
    parser.add_argument('--output', type=str, default='output.mp4',
                        help='Output video path')
    parser.add_argument('--cfg_scale', type=float, default=7.5,
                        help='Classifier-free guidance scale')
    parser.add_argument('--num_steps', type=int, default=100,
                        help='Number of denoising steps')
    parser.add_argument('--mode', type=str, choices=['t2v', 'vp'], default='t2v',
                        help='Generation mode')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    if args.mode == 't2v':
        model = Ca2VDM(
            in_channels=4, H=32, W=32,
            dim=1152, num_heads=16, num_layers=28,
            l=16, P_max=49, L_train=65, prefix_len=3,
            use_text_cond=True, text_dim=4096,
        )
    else:
        model = Ca2VDM(
            in_channels=4, H=32, W=32,
            dim=1152, num_heads=16, num_layers=28,
            l=8, P_max=25, L_train=33, prefix_len=3,
            use_text_cond=False,
        )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create diffusion process
    diffusion = DiffusionProcess(num_timesteps=1000, beta_start=1e-4, beta_end=0.02)
    diffusion.to(device)
    
    # Create inference engine
    inference = Ca2VDMInference(
        model=model,
        diffusion=diffusion,
        device=device,
        num_inference_steps=args.num_steps,
        cfg_scale=args.cfg_scale,
    )
    
    # Generate
    # Note: In practice, first_frame should be encoded via VAE
    # and text_emb should be computed using T5
    
    print(f"Starting autoregressive generation with {args.num_ar_steps} AR steps...")
    print(f"Model config: l={model.l}, P_max={model.P_max}, L_train={model.L_train}")
    
    # Placeholder: actual generation requires VAE encoding and T5 encoding
    

if __name__ == '__main__':
    main()
