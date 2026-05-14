
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import math
import random
from typing import Dict, Any, List, Optional, Tuple

from accelerate import Accelerator
from accelerate.utils import set_seed
from einops import rearrange, repeat

from .config import Config, ModelConfig, DiffusionConfig
from .model import Ca2VDM
from .modules import TimestepEmbedding, T5TextEncoder, CausalVQVAE
from .train import GaussianDiffusion # Reuse GaussianDiffusion from train.py

class KVCacheQueue:
    def __init__(self, max_length: int, num_attention_layers: int, kv_shape_template: Tuple):
        self.max_length = max_length # P_max
        self.num_attention_layers = num_attention_layers
        self.kv_shape_template = kv_shape_template # (HW, C) for temporal KV
        
        # Each item in the list corresponds to an attention layer's cache.
        # Each cache is a dict {'k': tensor (P_k, HW, C), 'v': tensor (P_k, HW, C)}
        self.temporal_kv_caches: List[Dict[str, torch.Tensor]] = [
            {'k': torch.empty(0, *kv_shape_template),
             'v': torch.empty(0, *kv_shape_template)} for _ in range(num_attention_layers)
        ]
        
        # Spatial KV cache: (P', HW, C) - only stores features of the last P' frames.
        # This is a single tensor, not per layer. It's overwritten.
        # Assumed to be features, not K/V projections.
        self.spatial_kv_cache_features: Optional[torch.Tensor] = None # (P', HW, C_latent)
    
    def enqueue_temporal_kv(self, new_temporal_kv_list: List[Dict[str, torch.Tensor]]):
        """
        Enqueues new temporal K/V from a generated chunk into the cache queues.
        new_temporal_kv_list: List of {'k': (L_chunk, HW, C), 'v': (L_chunk, HW, C)} for each layer.
        """
        for i, new_kv in enumerate(new_temporal_kv_list):
            current_k = self.temporal_kv_caches[i]['k']
            current_v = self.temporal_kv_caches[i]['v']
            
            # Concatenate new K/V
            updated_k = torch.cat((current_k, new_kv['k']), dim=0)
            updated_v = torch.cat((current_v, new_kv['v']), dim=0)
            
            # Dequeue if max_length is exceeded
            if updated_k.shape[0] > self.max_length:
                updated_k = updated_k[-self.max_length:]
                updated_v = updated_v[-self.max_length:]
            
            self.temporal_kv_caches[i]['k'] = updated_k
            self.temporal_kv_caches[i]['v'] = updated_v

    def update_spatial_kv_features(self, new_spatial_features: torch.Tensor):
        """
        Updates the spatial KV cache features (P' frames).
        new_spatial_features: (P', C_latent, H, W) -> will be reshaped to (P', HW, C_model)
        """
        self.spatial_kv_cache_features = new_spatial_features
    
    def get_current_temporal_kv(self) -> List[Dict[str, torch.Tensor]]:
        return self.temporal_kv_caches
    
    def get_current_spatial_features(self) -> Optional[torch.Tensor]:
        return self.spatial_kv_cache_features

class VideoGenerator:
    def __init__(self, config: Config, device: str):
        self.config = config
        self.device = device
        
        self.model = Ca2VDM(config.model).to(device)
        self.diffusion = GaussianDiffusion(config.diffusion, device)
        
        # Placeholders for pre-trained VAE and Text Encoder
        self.vae = CausalVQVAE() # In a real impl, load from diffusers
        self.text_encoder = T5TextEncoder() # In a real impl, load from transformers
        
        self.model.eval()

    def _get_num_attention_layers(self) -> int:
        num_attn_layers = 0
        for block_pair in self.model.down_blocks:
            if not isinstance(block_pair, torch.nn.Conv3d): # Not downsampling conv
                num_attn_layers += 1 # ResNetBlock is followed by CausalAttentionBlock
        num_attn_layers += 1 # Middle block CausalAttentionBlock
        for block_pair in self.model.up_blocks:
            if not isinstance(block_pair, torch.nn.ConvTranspose3d): # Not upsampling conv
                num_attn_layers += 1
        return num_attn_layers

    @torch.no_grad()
    def generate_autoregressive(self,
                                initial_frame_latent: torch.Tensor, # (1, C, H, W)
                                total_generation_length: int, # Total frames to generate beyond initial
                                text_prompt: Optional[str] = None,
                                num_ar_steps: int = 0, # How many chunks to generate
                                guidance_scale: float = 7.5
                                ) -> torch.Tensor:
        """
        Generates a long video autoregressively.
        initial_frame_latent: The VAE-encoded latent of the first frame (z_0^0). (1, C_latent, H_latent, W_latent)
        total_generation_length: Total number of frames to generate, including the initial frame.
        """
        
        L_chunk = self.config.model.chunk_length
        P_max = self.config.model.max_condition_frames
        P_prime = self.config.model.prefix_enhancement_frames # For spatial KV cache
        
        # Text conditioning
        text_embedding = None
        if text_prompt is not None and self.config.training.task_type == "text_to_video":
            # In a real impl, pass through text_encoder
            text_embedding = torch.randn(1, 77, self.config.model.context_dim).to(self.device) # Dummy
        
        # Initialize KV-cache queues
        # KV shape template for temporal cache: (HW, C_model)
        kv_shape_template = (self.config.model.resolution // 8 * self.config.model.resolution // 8, self.config.model.model_channels)
        num_attn_layers = self._get_num_attention_layers()
        kv_queue = KVCacheQueue(P_max, num_attn_layers, kv_shape_template)
        
        # Start with the initial frame as the first generated content
        # It serves as the initial clean prefix.
        generated_latents = initial_frame_latent.unsqueeze(0) # (1, 1, C, H, W)
        
        # 1. Cache writing for the initial frame (t=0)
        # timesteps is (B,). Here B=1. Timestep is 0 for initial frame.
        # tpe_indices for initial frame (relative to its own position 0)
        tpe_indices_initial_frame = torch.tensor([0], device=self.device).unsqueeze(0) # (1, 1)
        initial_frame_kv_input = initial_frame_latent.unsqueeze(0) # (1, 1, C, H, W)

        # `clean_prefix_frames_len` for initial frame is 1.
        clean_prefix_frames_len_initial = torch.tensor([1], dtype=torch.long, device=self.device)

        # Get KVs for the initial frame.
        initial_temporal_kvs, initial_spatial_features = self.model.get_kv_caches(
            initial_frame_kv_input,
            timesteps=torch.tensor([0], device=self.device), # t=0 for clean frames
            clean_prefix_frames_len=clean_prefix_frames_len_initial, # For the input to get_kv_caches, P=L=1
            text_context=text_embedding,
            tpe_indices=tpe_indices_initial_frame
        )
        kv_queue.enqueue_temporal_kv(initial_temporal_kvs)

        # Update spatial KV cache with the initial frame's features (if P_prime > 0)
        if P_prime > 0 and initial_spatial_features:
            # `initial_spatial_features` is list containing one (P_prime, HW, C) if from `get_kv_caches`
            # For the first frame, it is only 1 frame long, so P' needs to be handled
            # The `get_kv_caches` returns the last `P_prime` frames from its input `x`.
            # If `initial_frame_kv_input` is (1, 1, C, H, W), then `initial_spatial_features` would be (1, HW, C).
            # We need to ensure `P_prime` is considered.
            
            # The paper says `spatial_kv_cache` is `P'` frames.
            # So if initial frame is just 1 frame, its features become the first element in spatial cache.
            # When we have enough frames, it's the last P' frames.
            # For now, let's assume `initial_frame_latent` itself, if P_prime=1.
            
            # The `initial_spatial_features` is `x` slice from `get_kv_caches`.
            # `x` is `(B, L, C, H, W)`. `initial_spatial_features` is `(P', HW, C_model)`.
            # For 1 frame, it should be `(1, HW, C_model)`.
            
            # `spatial_kv_features = initial_spatial_features[0]` is `(1, HW, C_model)`
            
            # If P_prime > 1, we need to pad or handle less than P_prime frames.
            # For now, let's assume `P_prime` can be <= number of frames provided.
            
            # The spatial cache for prefix enhancement is `P'` frames.
            # If we only have 1 frame, it might be (1, HW, C).
            # The `attn2` expects (P', HW, C) as `clean_prefix_frames`.
            
            # Let's assume that if we have fewer than P_prime frames, we just use what we have,
            # or pad with zeros if strictly P_prime is expected.
            
            # For the first frame, `spatial_kv_features` would be `(1, HW, C_model)`.
            # If `P_prime > 1`, we need to decide how to fill `P_prime - 1` spots.
            # Let's just use the features from the initial frame as spatial cache.
            
            if initial_spatial_features and len(initial_spatial_features) > 0:
                # `initial_spatial_features` will be a list of one tensor: `[(1, HW, C_model)]`
                kv_queue.update_spatial_kv_features(initial_spatial_features[0])

        current_num_frames = 1 # Initial frame already processed

        # Cyclic TPEs: Base sequence for TPEs (0 to P_max + L_chunk - 1)
        base_tpe_sequence = torch.arange(0, P_max + L_chunk, device=self.device)
        
        # Autoregressive generation loop
        for ar_step in tqdm(range(num_ar_steps), desc="Autoregressive Generation"):
            # Prepare input for denoising stage
            # Noisy input for the next chunk
            noisy_chunk = torch.randn(1, L_chunk, self.config.model.latent_channels,
                                      self.config.model.resolution // 8,
                                      self.config.model.resolution // 8, device=self.device)
            
            # Determine TPE indices for the current chunk
            # The paper says: "During inference, the TPEs are assigned chunk-by-chunk as the autoregression progresses.
            # ...the denoising target will be assigned those TPEs indexed from the beginning." (Cyclic-TPEs).
            # `base_tpe_sequence` has length `P_max + L_chunk`.
            # The TPEs for the current `L_chunk` frames are taken from `0` to `L_chunk-1` (relative to the current cyclic window).
            
            # The TPEs are for the *entire* sequence used by the model at a given AR step.
            # This sequence is `P_k + L_chunk`.
            # For the conditional part (P_k frames), their TPEs are already bound in the cache.
            # For the *current chunk* being generated (L_chunk frames), their TPEs are from the beginning.
            
            # So, `tpe_indices_full` in `p_sample_loop` should contain indices for `P_k` and then `L_chunk`.
            # The `tpe_indices_full` should be for the `L_chunk` frames only, which will then be combined inside `p_sample_loop`.
            
            # Let's align with how `p_sample_loop` expects `tpe_indices_full`
            # `tpe_indices_full` in `p_sample_loop` refers to the `tpe_indices` for the current `L_chunk` of noisy frames.
            # These are for `L_chunk` frames, and should be based on `base_tpe_sequence` with cyclic shift.
            
            # TPE for the current chunk (L_chunk frames)
            # If current total frames generated so far is `current_num_frames`, and we add `L_chunk`.
            # The TPE indices for the *new* `L_chunk` frames should be relative to the entire AR process.
            # So, for frame `j` in the new chunk, its global index is `current_num_frames + j`.
            # Its cyclic TPE index is `(current_num_frames + j) % (P_max + L_chunk)`.
            
            # For a batch of 1 video:
            current_chunk_tpe_indices = torch.tensor([(current_num_frames + j) % (P_max + L_chunk) for j in range(L_chunk)], device=self.device).unsqueeze(0) # (1, L_chunk)
            
            # P_k is the actual number of frames currently in the temporal KV cache.
            actual_P_k = kv_queue.get_current_temporal_kv()[0]['k'].shape[0] # Length of K in the first layer's cache
            
            # Causal mask for the current chunk. Shape (L_chunk, actual_P_k + L_chunk)
            # Mask should be `-inf` if `i < j` where `i` is current frame index, `j` is all frames (cached + current).
            causal_mask_for_chunk = torch.triu(torch.ones(L_chunk, actual_P_k + L_chunk, device=self.device, dtype=torch.bool), diagonal=actual_P_k + 1)
            causal_mask_for_chunk = causal_mask_for_chunk.masked_fill(causal_mask_for_chunk, -torch.inf)
            
            # Denoising Stage
            denoised_chunk = self.diffusion.p_sample_loop(
                self.model,
                shape=noisy_chunk.shape,
                text_context=text_embedding,
                tpe_indices_full=current_chunk_tpe_indices, # These are for the L_chunk frames
                num_frames_in_chunk=L_chunk,
                num_condition_frames_in_ar_step=actual_P_k, # P_k for the current step
                max_condition_frames_in_cache=P_max,
                guidance_scale=guidance_scale,
                temporal_kv_cache_queues=kv_queue.get_current_temporal_kv(),
                spatial_kv_cache_features=kv_queue.get_current_spatial_features(),
                causal_mask=causal_mask_for_chunk
            )
            
            generated_latents = torch.cat((generated_latents, denoised_chunk), dim=1) # (1, current_num_frames + L_chunk, C, H, W)
            current_num_frames += L_chunk

            # Cache Writing Stage
            # The `denoised_chunk` (z_0^{P_k:P_k+l}) is input to model again to compute KVs.
            
            # Timesteps for cache writing are all 0
            timesteps_cache_writing = torch.tensor([0] * L_chunk, device=self.device) # (L_chunk,)
            timesteps_cache_writing = timesteps_cache_writing.unsqueeze(0) # (1, L_chunk) for B=1
            
            # TPE indices for cache writing are the same as for denoising target.
            tpe_indices_cache_writing = tpe_indices_for_chunk
            
            # `clean_prefix_frames_len` for the `denoised_chunk` is its own length, L_chunk.
            clean_prefix_frames_len_chunk = torch.tensor([L_chunk], dtype=torch.long, device=self.device)

            new_temporal_kvs, new_spatial_features_for_cache = self.model.get_kv_caches(
                denoised_chunk, # (1, L_chunk, C, H, W)
                timesteps=timesteps_cache_writing[:, 0], # Pass `0` for t_emb, should be (B,)
                clean_prefix_frames_len=clean_prefix_frames_len_chunk,
                text_context=text_embedding,
                tpe_indices=tpe_indices_cache_writing
            )
            kv_queue.enqueue_temporal_kv(new_temporal_kvs)
            
            if P_prime > 0 and new_spatial_features_for_cache and len(new_spatial_features_for_cache) > 0:
                # `new_spatial_features_for_cache` is a list with one item: (P_prime, HW, C_model)
                # Or (L_chunk, HW, C_model) if L_chunk < P_prime.
                kv_queue.update_spatial_kv_features(new_spatial_features_for_cache[0])
            
            if current_num_frames >= total_generation_length:
                break
        
        # Decode the generated latents to pixel space
        # In a real impl, vae.decode(generated_latents)
        generated_video_pixels = torch.randn(generated_latents.shape[0], generated_latents.shape[1],
                                             3, self.config.data.image_size, self.config.data.image_size) # Dummy
        
        return generated_video_pixels

def generate_video_main():
    config = Config()
    accelerator = Accelerator(
        mixed_precision=config.system.mixed_precision,
        log_with="tensorboard",
        project_dir=os.path.join(config.system.output_dir, "inference_logs")
    )
    
    set_seed(config.system.seed)
    
    # Load trained model weights
    # For now, this is a placeholder. In a real scenario, you'd load from a checkpoint.
    # accelerator.load_state(os.path.join(config.system.output_dir, "final_checkpoint"))
    
    generator = VideoGenerator(config, accelerator.device)
    
    # Example usage
    # Initial frame latent (dummy)
    initial_frame_latent = torch.randn(1, config.model.latent_channels,
                                       config.model.resolution // 8,
                                       config.model.resolution // 8, device=accelerator.device)
    
    # Define total frames to generate
    total_frames_to_generate = 80 # As in Table 5
    num_ar_steps = (total_frames_to_generate - 1) // config.model.chunk_length # Minus 1 for initial frame
    if (total_frames_to_generate - 1) % config.model.chunk_length != 0:
        num_ar_steps += 1
    
    text_prompt = "a dog running in a park" if config.training.task_type == "text_to_video" else None
    
    accelerator.print(f"Generating video with {total_frames_to_generate} frames in {num_ar_steps} AR steps...")
    
    generated_video = generator.generate_autoregressive(
        initial_frame_latent=initial_frame_latent,
        total_generation_length=total_frames_to_generate,
        text_prompt=text_prompt,
        num_ar_steps=num_ar_steps,
        guidance_scale=config.diffusion.guidance_scale
    )
    
    accelerator.print(f"Generated video shape: {generated_video.shape}")
    # Save the generated video (placeholder)
    # torch.save(generated_video, os.path.join(config.system.output_dir, "generated_video.pt"))
    accelerator.print(f"Video generation complete. Output saved to {config.system.output_dir}/generated_video.pt (dummy)")

if __name__ == "__main__":
    generate_video_main()

