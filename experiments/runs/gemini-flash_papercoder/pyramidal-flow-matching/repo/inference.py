```python
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from PIL import Image
import numpy as np # For potential conversion from tensor to numpy for saving/viewing if needed
from collections import deque # For efficiently managing history frames

# Project-specific imports
try:
    from config import Config
    from model import PyramidalFlowMatchingModel, TextEncoder
    from vae import VideoVAE
    from pyramid_logic import PyramidFlowMatcher
    from utils import get_default_device, downsample, upsample
except ImportError as e:
    # Minimal Stubs for testing inference.py independently if other modules are not yet complete
    print(f"Failed to import project modules for inference.py: {e}. Using stub classes.")

    class Config:
        def __init__(self):
            self.model = self.ModelConfig()
            self.compute = self.ComputeConfig()
            self.inference = self.InferenceConfig()
            self.data_paths = self.DataPathsConfig()

        class ModelConfig:
            pyramid_stages: int = 3
            text_encoder: Any = None # Placeholder for TextEncoder config
            vae: Any = None # Placeholder for VAE config
            dit_params: Any = None # Placeholder for DiT params

        class ComputeConfig:
            device: str = "cpu"
            mixed_precision: str = "no"

        class InferenceConfig:
            guidance_scale: float = 7.0
            num_inference_steps: int = 50
            ode_solver: str = "euler" # Default to a simple solver
            output_resolution: Tuple[int, int] = (256, 256)
            output_fps: int = 24
            output_duration: int = 5
            max_output_duration: int = 10
            seed: int = 42

        class DataPathsConfig:
            # Placeholder, not directly used in inference logic but good to have
            pass

    class PyramidalFlowMatchingModel(torch.nn.Module):
        def __init__(self, config: Config, text_encoders: Any):
            super().__init__()
            # Dummy model to simulate DiT output
            self.config = config
            self.text_encoders = text_encoders
            # Assuming DiT operates on flattened sequence (B, N_tokens, hidden_size)
            # The output velocity field would be (B, N_tokens, in_channels)
            # which needs to be reshaped to (B, in_channels, T, H, W)
            # For simplicity, let's assume `in_channels` is 4.
            self.dummy_output_channels = 4 
            # Dummy output size will be based on the pyramid stage k.
            # Example: 16x16 for k=0, 8x8 for k=1, 4x4 for k=2 with 8x8x8 VAE
            
        def forward(self, latent_xt: torch.Tensor, time_batch: torch.Tensor,
                    text_cond_t5: torch.Tensor, text_cond_clip: torch.Tensor,
                    history_cond: Optional[List[torch.Tensor]] = None,
                    pyramid_stage_k: int = 0) -> torch.Tensor:
            # Simulate velocity field output matching latent_xt shape
            # Assuming latent_xt is (B, C, T, H, W)
            return torch.randn_like(latent_xt)


    class VideoVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            # Simulate encoding to a 4-channel latent, downsampled 8x
            latent_channels = 4
            if x.ndim == 5: # video (B, C, T, H, W)
                t_out = max(1, x.shape[2] // 8)
                h_out = max(1, x.shape[3] // 8)
                w_out = max(1, x.shape[4] // 8)
                return torch.zeros(x.shape[0], latent_channels, t_out, h_out, w_out), None, torch.randn(x.shape[0], latent_channels, t_out, h_out, w_out)
            elif x.ndim == 4: # image (B, C, H, W)
                h_out = max(1, x.shape[2] // 8)
                w_out = max(1, x.shape[3] // 8)
                return torch.zeros(x.shape[0], latent_channels, h_out, w_out), None, torch.randn(x.shape[0], latent_channels, h_out, w_out)
            else:
                raise ValueError("Unsupported input dimensions for VAE stub.")
        def decode(self, latents: torch.Tensor) -> torch.Tensor:
            # Simulate decoding
            if latents.ndim == 5:
                return torch.randn(latents.shape[0], 3, latents.shape[2]*8, latents.shape[3]*8, latents.shape[4]*8)
            elif latents.ndim == 4:
                return torch.randn(latents.shape[0], 3, latents.shape[2]*8, latents.shape[3]*8)
            else:
                raise ValueError("Unsupported latent dimensions for VAE stub.")

    class PyramidFlowMatcher:
        def __init__(self, config: Config, vae: VideoVAE):
            self.config = config
            self.vae = vae
            self.K = config.model.pyramid_stages
            # Example time windows, finest to coarsest
            # k=0: (0.5, 1.0)
            # k=1: (0.2, 2/3)
            # k=2: (0.0, 1/3)
            self.spatial_pyramid_timesteps = [(0.5, 1.0), (0.2, 2/3), (0.0, 1/3)]
            self.device = get_default_device()

        def get_pyramid_timesteps(self, k_stage: int) -> Tuple[float, float]:
            # Return from finest (k=0) to coarsest (k=K-1)
            # Map k_stage (0=finest, K-1=coarsest) to index in the list
            if not (0 <= k_stage < self.K):
                raise ValueError(f"k_stage must be between 0 and {self.K-1}, but got {k_stage}.")
            return self.spatial_pyramid_timesteps[k_stage]

        def apply_renoising(self, prev_latent_ek: torch.Tensor, current_k: int, prev_ek_time: float) -> torch.Tensor:
            # Stub for renoising
            # Needs to upsample prev_latent_ek by factor 2
            upsampled_latent = upsample(prev_latent_ek, factor=2, mode='trilinear' if prev_latent_ek.ndim==5 else 'bilinear')
            
            s_current_k, _ = self.get_pyramid_timesteps(current_k)
            # Coefficients from paper (Eq. 15)
            sk_coeff = (1.0 + s_current_k) / 2.0
            alpha_n_prime_coeff = (torch.sqrt(torch.tensor(3.0, device=prev_latent_ek.device)) * (1.0 - s_current_k)) / 2.0
            n_prime = torch.randn_like(upsampled_latent, device=prev_latent_ek.device)
            next_start_latent_sk = sk_coeff * upsampled_latent + alpha_n_prime_coeff * n_prime

            return next_start_latent_sk


        def get_down_factor(self, k_stage: int) -> int:
            return 2**k_stage

    class TextEncoder(torch.nn.Module):
        def __init__(self, config: Config):
            super().__init__()
            self.max_text_length = 77
        def get_text_embeddings(self, prompts: List[str], device: torch.device) -> Dict[str, torch.Tensor]:
            batch_size = len(prompts)
            t5_embed_dim = 768
            clip_embed_dim = 1024
            return {
                't5': torch.randn(batch_size, self.max_text_length, t5_embed_dim, device=device),
                'clip': torch.randn(batch_size, self.max_text_length, clip_embed_dim, device=device)
            }
    
    # Stubs for utils functions
    def get_default_device() -> torch.device:
        return torch.device("cpu")

    def downsample(tensor: torch.Tensor, factor: int, mode: str = "trilinear") -> torch.Tensor:
        if tensor.ndim == 5: # Video (B, C, T, H, W)
            t_out = max(1, tensor.shape[2] // factor)
            h_out = max(1, tensor.shape[3] // factor)
            w_out = max(1, tensor.shape[4] // factor)
            return F.interpolate(tensor, size=(t_out, h_out, w_out), mode=mode, align_corners=False)
        elif tensor.ndim == 4: # Image (B, C, H, W)
            h_out = max(1, tensor.shape[2] // factor)
            w_out = max(1, tensor.shape[3] // factor)
            return F.interpolate(tensor, size=(h_out, w_out), mode='bilinear', align_corners=False)
        else:
            raise NotImplementedError("Stub downsample for other tensor dimensions not implemented.")

    def upsample(tensor: torch.Tensor, factor: int, mode: str = "trilinear") -> torch.Tensor:
        if tensor.ndim == 5: # Video (B, C, T, H, W)
            return F.interpolate(tensor, size=(tensor.shape[2] * factor, tensor.shape[3] * factor, tensor.shape[4] * factor), mode=mode, align_corners=False)
        elif tensor.ndim == 4: # Image (B, C, H, W)
            return F.interpolate(tensor, size=(tensor.shape[2] * factor, tensor.shape[3] * factor), mode='bilinear', align_corners=False)
        else:
            raise NotImplementedError("Stub upsample for other tensor dimensions not implemented.")


logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class VideoGenerator:
    """
    Generates video frames autoregressively using a trained Pyramidal Flow Matching model.
    """

    def __init__(self, config: Config, model: PyramidalFlowMatchingModel, vae: VideoVAE, pyramid_logic: PyramidFlowMatcher):
        """
        Initializes the VideoGenerator.

        Args:
            config (Config): The global configuration object.
            model (PyramidalFlowMatchingModel): The trained flow matching model.
            vae (VideoVAE): The trained variational autoencoder.
            pyramid_logic (PyramidFlowMatcher): The logic handler for pyramid operations.
        """
        self.config = config
        self.model = model
        self.vae = vae
        self.pyramid_logic = pyramid_logic

        self.device = get_default_device() if not torch.cuda.is_available() else torch.device(config.compute.device)

        self.model.to(self.device)
        self.vae.to(self.device)
        self.model.eval()
        self.vae.eval()

        self.pyramid_k_stages = config.model.pyramid_stages
        self.num_inference_steps = config.inference.num_inference_steps
        self.guidance_scale = config.inference.guidance_scale
        self.ode_solver_type = config.inference.ode_solver

        # Cache for unconditional text embeddings for CFG
        self.uncond_text_t5: Optional[torch.Tensor] = None
        self.uncond_text_clip: Optional[torch.Tensor] = None

        # Deque for efficiently managing generated clean latents for history conditioning
        # Stores full-resolution clean latents (B, C_latent, H_latent, W_latent) for each frame.
        self.generated_latents_list: deque = deque(maxlen=2) # Stores x_1^{i-1}, x_1^{i-2}


        logger.info(f"VideoGenerator initialized on device: {self.device}")

    def _prepare_conditions(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepares conditional and unconditional text embeddings for classifier-free guidance.

        Args:
            prompt (str): The text prompt for generation.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            (cond_t5_embeds, uncond_t5_embeds, cond_clip_embeds, uncond_clip_embeds)
            Each tensor is of shape (1, max_text_length, embed_dim).
        """
        # Conditional embeddings
        cond_embeddings_dict = self.model.text_encoders.get_text_embeddings([prompt], self.device)
        cond_t5_embeds = cond_embeddings_dict['t5']
        cond_clip_embeds = cond_embeddings_dict['clip']

        # Unconditional embeddings (cache them)
        if self.uncond_text_t5 is None or self.uncond_text_clip is None:
            uncond_embeddings_dict = self.model.text_encoders.get_text_embeddings([""], self.device)
            self.uncond_text_t5 = uncond_embeddings_dict['t5']
            self.uncond_text_clip = uncond_embeddings_dict['clip']
            
        uncond_t5_embeds = self.uncond_text_t5
        uncond_clip_embeds = self.uncond_text_clip

        # Ensure all are 1 in batch dimension
        # If TextEncoder returns (Batch, SequenceLength, EmbeddingDim) and Batch was 1, it's (1, ...)
        # If using accelerator, it might be (batch_size_per_gpu, ...). Assuming single batch for inference.
        if cond_t5_embeds.shape[0] != 1:
            cond_t5_embeds = cond_t5_embeds[:1] # Take first item if batch > 1
            cond_clip_embeds = cond_clip_embeds[:1]
            uncond_t5_embeds = uncond_t5_embeds[:1]
            uncond_clip_embeds = uncond_clip_embeds[:1]

        return cond_t5_embeds, uncond_t5_embeds, cond_clip_embeds, uncond_clip_embeds


    def _run_ode_solver(
        self,
        current_latent: torch.Tensor,
        start_time: float,
        end_time: float,
        text_cond_t5: torch.Tensor,
        uncond_text_t5: torch.Tensor,
        text_cond_clip: torch.Tensor,
        uncond_text_clip: torch.Tensor,
        history_cond_list: Optional[List[torch.Tensor]] = None,
        pyramid_stage_k: int = 0,
        guidance_scale: float = 7.0
    ) -> torch.Tensor:
        """
        Integrates the ODE within a single pyramid stage using a basic Euler solver.

        Args:
            current_latent (torch.Tensor): The starting latent for this ODE integration (e.g., hat_x_sk).
            start_time (float): The start time (s_k) for this ODE integration.
            end_time (float): The end time (e_k) for this ODE integration.
            text_cond_t5 (torch.Tensor): Conditional T5 embeddings.
            uncond_text_t5 (torch.Tensor): Unconditional T5 embeddings.
            text_cond_clip (torch.Tensor): Conditional CLIP embeddings.
            uncond_text_clip (torch.Tensor): Unconditional CLIP embeddings.
            history_cond_list (Optional[List[torch.Tensor]]): List of downsampled history latents.
            pyramid_stage_k (int): The current pyramid stage index.
            guidance_scale (float): Classifier-free guidance scale.

        Returns:
            torch.Tensor: The latent state after ODE integration (e.g., hat_x_ek).
        """
        dt = (end_time - start_time) / self.num_inference_steps
        current_t = start_time
        
        for _ in range(self.num_inference_steps):
            # Prepare for CFG: concatenate current_latent and current_t
            # Replicate current_latent for unconditional and conditional paths (batch size of 2)
            latent_xt_cfg = torch.cat([current_latent, current_latent], dim=0)
            time_batch_cfg = torch.tensor([current_t, current_t], device=self.device, dtype=torch.float32)

            # Concatenate text embeddings for CFG batch
            full_text_t5 = torch.cat([uncond_text_t5, text_cond_t5], dim=0)
            full_text_clip = torch.cat([uncond_text_clip, text_cond_clip], dim=0)

            # Prepare history conditions for CFG if available
            history_cond_cfg_list = None
            if history_cond_list:
                history_cond_cfg_list = []
                for hist_latent in history_cond_list:
                    history_cond_cfg_list.append(torch.cat([hist_latent, hist_latent], dim=0))

            # Model forward pass for both conditional and unconditional
            # The model is expected to handle the batch of 2
            velocity_field_cfg = self.model(
                latent_xt_cfg,
                time_batch_cfg,
                full_text_t5,
                full_text_clip,
                history_cond=history_cond_cfg_list,
                pyramid_stage_k=pyramid