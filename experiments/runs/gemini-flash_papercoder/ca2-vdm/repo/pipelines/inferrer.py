import os
import logging
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F

from diffusers.models import AutoencoderKL
from transformers import CLIPTextModel, AutoTokenizer

from config import Config
from models.ca2_vdm_model import Ca2VDM
from utils.diffusion_schedulers import DiffusionScheduler
from utils.kv_cache_manager import KVCacheManager

logger = logging.getLogger(__name__)

class Inferrer:
    """
    Handles autoregressive video generation using the Ca2-VDM model.
    It manages the denoising process, KV-cache updates, and video assembly.
    """

    def __init__(
        self,
        model: Ca2VDM,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        scheduler: DiffusionScheduler,
        config: Config
    ):
        """
        Initializes the Inferrer with necessary components and configurations.

        Args:
            model (Ca2VDM): An instance of the Ca2VDM model.
            vae (AutoencoderKL): The pretrained VAE for latent-pixel conversion.
            text_encoder (CLIPTextModel): The pretrained T5 text encoder.
            scheduler (DiffusionScheduler): The diffusion scheduler for denoising steps.
            config (Config): The global configuration object.
        """
        self.model = model
        self.vae = vae
        self.text_encoder = text_encoder
        self.scheduler = scheduler
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Inferrer using device: {self.device}")

        # Move models to device
        self.model.to(self.device)
        self.vae.to(self.device)
        self.text_encoder.to(self.device)

        # KVCacheManager is initialized inside generate_video_autoregressive because
        # chunk_length and max_prefix_length are task/stage-specific and are set
        # dynamically in the config object via set_active_task.
        # This instance variable will be populated once per generation call.
        self.kv_cache_manager: Optional[KVCacheManager] = None

    @torch.no_grad()
    def generate_video_autoregressive(
        self,
        first_frame_latents: torch.Tensor, # (1, C, H, W)
        text_prompt_embeddings: torch.Tensor, # (1, N_tokens, D_text_emb)
        num_ar_steps: int,
        task_mode: str, # e.g., "t2v_internvid", "vp_skytimelapse"
        uncond_text_embeddings: Optional[torch.Tensor] = None # (1, N_tokens, D_text_emb) for CFG
    ) -> List[Image.Image]:
        """
        Generates a video autoregressively for a specified number of AR steps.

        Args:
            first_frame_latents (torch.Tensor): The latent representation of the initial frame,
                                                provided as a clean (non-noisy) input.
                                                Shape: (1, C_latent, H_latent, W_latent).
            text_prompt_embeddings (torch.Tensor): Embeddings of the text prompt.
                                                   Shape: (1, N_tokens, D_text_emb).
            num_ar_steps (int): The number of autoregressive steps to perform (generates
                                num_ar_steps * chunk_length new frames).
            task_mode (str): Specifies the task (e.g., 't2v_internvid' or 'vp_skytimelapse')
                             to fetch correct `chunk_length`, `max_prefix_length`, and
                             `max_train_video_length` from config.
            uncond_text_embeddings (Optional[torch.Tensor]): Unconditional text embeddings for
                                                             Classifier-Free Guidance.
                                                             If None, CFG is not applied.
                                                             Shape: (1, N_tokens, D_text_emb).

        Returns:
            List[PIL.Image.Image]: A list of PIL.Image.Image objects,
                                   representing the generated video frames.
        """
        self.model.eval()
        self.vae.eval()
        self.text_encoder.eval()

        # Set active task in config to load relevant parameters
        stage_name_for_config = "ca2_vdm_stage2" if task_mode == "t2v_internvid" else "ca2_vdm"
        self.config.set_active_task(task_mode, stage_name_for_config)

        chunk_length: int = self.config.chunk_length
        max_prefix_length: int = self.config.max_prefix_length
        max_train_video_length: int = self.config.max_train_video_length
        num_inference_steps: int = self.config.num_inference_steps
        guidance_scale: float = self.config.guidance_scale
        use_prefix_enhancement: bool = self.config.use_prefix_enhancement

        # Initialize KV Cache Manager
        # Temporal and spatial cache dims are dynamically inferred by KVCacheManager
        self.kv_cache_manager = KVCacheManager(
            max_temporal_cache_len=max_prefix_length,
            chunk_len=chunk_length,
            temporal_cache_dims=(), # Placeholder
            spatial_cache_dims=(),  # Placeholder
            device=self.device
        )
        logger.info(f"KVCacheManager initialized with max_temporal_cache_len={max_prefix_length}, chunk_len={chunk_length}")

        generated_latents_history: List[torch.Tensor] = [] # Stores (C, L_chunk, H, W)

        # 1. Process the `first_frame_latents` (Initial Prefix)
        first_frame_latents = first_frame_latents.to(self.device) # Shape: (1, C, H, W)
        
        # Unsqueeze to (1, C, 1, H, W) for get_clean_kvs which expects (B, C, L, H, W)
        first_frame_latents_with_L = first_frame_latents.unsqueeze(2)

        # Generate TPE indices for this single frame (global index 0, fixed offset=0 for inference)
        # _get_tpe expects a tensor of indices and total length
        tpe_initial_frame_indices = self.model._get_tpe(
            torch.tensor([0], device=self.device, dtype=torch.long), # Current frame's global index
            max_train_video_length, # L_train
            random_offset=0 # Fixed offset for inference
        ).unsqueeze(0) # Unsqueeze to match batch dim for `get_clean_kvs`
        
        # Compute Initial KVs for the first (clean) frame
        initial_temporal_kvs_dict, initial_spatial_kvs_dict = self.model.get_clean_kvs(
            latents=first_frame_latents_with_L,
            tpe_indices=tpe_initial_frame_indices,
            text_embeddings=text_prompt_embeddings, # Use original text embeddings
            is_conditioning=True # This frame is clean and serves as condition
        )
        self.kv_cache_manager.update_temporal_cache(initial_temporal_kvs_dict)
        if use_prefix_enhancement:
            self.kv_cache_manager.update_spatial_cache(initial_spatial_kvs_dict)
        logger.info(f"Initialized KV caches with first frame.")

        # Store first frame latent in history as (C, 1, H, W)
        generated_latents_history.append(first_frame_latents.squeeze(0).unsqueeze(1))


        # 2. Autoregressive Generation Loop
        current_global_frame_idx: int = 1 # Tracks the global index of the first frame in the current chunk to generate

        # Get inference timesteps once
        inference_timesteps: List[int] = self.scheduler.get_ddim_timesteps(num_inference_steps)

        for ar_step_idx in tqdm(range(num_ar_steps), desc="Autoregressive Generation"):
            logger.debug(f"AR Step {ar_step_idx + 1}/{num_ar_steps}, Generating chunk starting at global frame {current_global_frame_idx}")

            # Initialize noisy latents for the current chunk
            # Shape: (1, C_latent, chunk_length, H_latent, W_latent)
            current_noisy_latents = torch.randn(
                1, first_frame_latents.shape[1], chunk_length,
                first_frame_latents.shape[2], first_frame_latents.shape[3],
                device=self.device
            )

            # Generate TPE indices for the current chunk to be denoised
            tpe_base_indices = torch.arange(
                current_global_frame_idx,
                current_global_frame_idx + chunk_length,
                device=self.device,
                dtype=torch.long
            )
            chunk_tpe_indices = self.model._get_tpe(
                tpe_base_indices,
                max_train_video_length,
                random_offset=0
            ).unsqueeze(0) # Unsqueeze to match batch dim for `denoise_step`

            # Denoising sub-loop for the current chunk
            for t in inference_timesteps:
                # 2.1. Conditional pass
                cond_noise_pred = self.model.denoise_step(
                    noisy_latents=current_noisy_latents,
                    timestep=t,
                    text_embeddings=text_prompt_embeddings,
                    kv_cache_manager=self.kv_cache_manager,
                    chunk_tpe_indices=chunk_tpe_indices, # Pass TPEs for the current chunk
                    current_ar_step_idx=ar_step_idx # Pass current AR step index
                )

                # 2.2. Unconditional pass (if CFG is enabled)
                if uncond_text_embeddings is not None and guidance_scale > 1.0:
                    uncond_noise_pred = self.model.denoise_step(
                        noisy_latents=current_noisy_latents,
                        timestep=t,
                        text_embeddings=uncond_text_embeddings, # Use unconditional embeddings
                        kv_cache_manager=self.kv_cache_manager,
                        chunk_tpe_indices=chunk_tpe_indices,
                        current_ar_step_idx=ar_step_idx
                    )
                    # Apply Classifier-Free Guidance
                    noise_pred = uncond_noise_pred + guidance_scale * (cond_noise_pred - uncond_noise_pred)
                else:
                    noise_pred = cond_noise_pred
                
                # Update current noisy latents using the scheduler
                current_noisy_latents, _ = self.scheduler.step(noise_pred, t, current_noisy_latents)

            denoised_chunk_latents = current_noisy_latents # This is now the clean latent chunk

            # Store denoised chunk in history (shape: (C, chunk_length, H, W))
            generated_latents_history.append(denoised_chunk_latents.squeeze(0))

            # 3. Cache Writing Stage for the newly denoised chunk
            new_temporal_kvs_dict, new_spatial_kvs_dict = self.model.get_clean_kvs(
                latents=denoised_chunk_latents,
                tpe_indices=chunk_tpe_indices,
                text_embeddings=text_prompt_embeddings, # Still use conditional text for KV computation if applicable
                is_conditioning=True
            )
            self.kv_cache_manager.update_temporal_cache(new_temporal_kvs_dict)
            if use_prefix_enhancement:
                self.kv_cache_manager.update_spatial_cache(new_spatial_kvs_dict)
            
            # Update global frame index for the next AR step
            current_global_frame_idx += chunk_length

        # 4. Final Video Assembly and Decoding
        # Concatenate all generated latent chunks along the time dimension (dim=1)
        # Each item in generated_latents_history is (C, L_chunk, H, W)
        concatenated_latents = torch.cat(generated_latents_history, dim=1) # Shape: (C, Total_Frames, H, W)

        # Add batch dimension for VAE decoder: (1, C, Total_Frames, H, W)
        # Also apply VAE scaling factor
        final_latents_for_decode = concatenated_latents.unsqueeze(0) / self.vae.config.scaling_factor

        # Decode to pixel space
        decoded_pixels = self.vae.decode(final_latents_for_decode).sample # Shape: (1, C, Total_Frames, H_pixel, W_pixel)
        
        # Post-process decoded pixels
        decoded_pixels = (decoded_pixels / 2 + 0.5).clamp(0, 1) # Unnormalize from [-1, 1] to [0, 1]
        decoded_pixels = decoded_pixels.cpu().permute(0, 2, 3, 4, 1).numpy() # (1, Total_Frames, H, W, C)

        output_images: List[Image.Image] = []
        for i in range(decoded_pixels.shape[1]):
            frame_np = (decoded_pixels[0, i] * 255).astype(np.uint8)
            output_images.append(Image.fromarray(frame_np))

        logger.info(f"Generated {len(output_images)} frames in total.")
        return output_images

