import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Tuple, Any, Optional, Callable, Dict, Union
import numpy as np
import logging

# Local imports
from config import Config
from tokenizer import VAETokenizer, CLIPTextEncoder
from model import HiMARModel # Import the HiMARModel
from utils import get_noise_scheduler, _predict_original_from_noise # Import utility functions

logger = logging.getLogger(__name__)

class Generator:
    """
    Handles the generation of images using the trained Hi-MAR model.
    It orchestrates the two-phase hierarchical denoising process and
    integrates Classifier-Free Guidance (CFG).
    """
    def __init__(
        self,
        model: HiMARModel,
        tokenizer: VAETokenizer,
        clip_encoder: Optional[CLIPTextEncoder],
        config: Dict[str, Any], # Full global config from config.py
        device: str
    ):
        """
        Initializes the Generator.

        Args:
            model: The HiMARModel instance (typically the EMA model) for generation.
            tokenizer: An instance of VAETokenizer to encode/decode visual latents.
            clip_encoder: An instance of CLIPTextEncoder (optional, for text-to-image tasks).
            config: The full loaded configuration dictionary from config.py.
            device: The computational device ('cuda' or 'cpu').
        """
        self.model = model.to(device)
        self.model.eval() # Ensure model is in evaluation mode
        self.tokenizer = tokenizer
        self.clip_encoder = clip_encoder
        self.device = device

        # Retrieve generation and tokenizer specific configurations
        self.generation_cfg: Dict[str, Any] = Config.get_generation_config()
        self.tokenizer_cfg: Dict[str, Any] = Config.get_tokenizer_config()
        # Ensure we have the model config to get variant details like hidden_size
        self.model_cfg: Dict[str, Any] = Config.get_model_config(config['model_config']['variant'])

        # Initialize noise scheduler
        self.noise_scheduler_num_timesteps: int = 1000 # Default DDPM timesteps, common for such models
        self.noise_scheduler_fn: Callable = get_noise_scheduler(
            self.generation_cfg['noise_scheduler_type'],
            num_train_timesteps=self.noise_scheduler_num_timesteps
        )

        self.phase1_inference_steps: int = self.generation_cfg['inference_steps']['phase1']
        self.phase2_inference_steps: int = self.generation_cfg['inference_steps']['phase2']
        self.default_guidance_scale: float = self.generation_cfg['guidance_scale']
        # The paper specifies phase1_cfg_on_for_eval behavior directly for 'w/o CFG' imageNet eval
        # It's in generation_cfg because it's a generation-time setting.
        self.phase1_cfg_on_for_eval: bool = self.generation_cfg['phase1_cfg_on_for_eval']

        logger.info("Generator initialized.")

    @torch.no_grad()
    def _denoise_step(
        self,
        latent: torch.Tensor,
        t_curr: int,
        t_prev: int,
        model_output_noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Performs a single denoising step using the DDPM formulation.

        Args:
            latent: The current noisy latent tensor (x_t), shape (B, N, D_latent).
            t_curr: Current discrete timestep (0 to num_train_timesteps-1).
            t_prev: Previous discrete timestep (0 to num_train_timesteps-1, or -1 for the last step).
            model_output_noise: The noise predicted by the model (epsilon_theta(x_t, t)),
                                shape (B, N, D_latent).

        Returns:
            The denoised latent tensor (x_{t_prev}), shape (B, N, D_latent).
        """
        # Ensure timesteps are tensors
        t_curr_tensor = torch.tensor([t_curr], device=self.device, dtype=torch.long)
        t_prev_tensor = torch.tensor([t_prev], device=self.device, dtype=torch.long)

        # Get noise schedule parameters for current timestep
        # alpha_prod_t is sqrt_alpha_prod_t, one_minus_alpha_prod_t is sqrt_one_minus_alpha_prod_t
        sqrt_alpha_prod_t, sqrt_one_minus_alpha_prod_t = self.noise_scheduler_fn(t_curr_tensor)
        
        # Reshape for broadcasting
        sqrt_alpha_prod_t = sqrt_alpha_prod_t.view(-1, 1, 1)
        sqrt_one_minus_alpha_prod_t = sqrt_one_minus_alpha_prod_t.view(-1, 1, 1)

        # Predict x_0 (the original, clean latent) from x_t and predicted noise epsilon
        # x_0 = (x_t - sqrt(1-alpha_prod_t) * epsilon) / sqrt(alpha_prod_t)
        x0_pred = (latent - sqrt_one_minus_alpha_prod_t * model_output_noise) / sqrt_alpha_prod_t

        # If t_prev < 0 (i.e., we are at the last step, going from t=0 to x_0), return x0_pred
        if t_prev < 0:
            return x0_pred

        # Get noise schedule parameters for previous timestep
        sqrt_alpha_prod_t_prev, sqrt_one_minus_alpha_prod_t_prev = self.noise_scheduler_fn(t_prev_tensor)
        
        # Reshape for broadcasting
        sqrt_alpha_prod_t_prev = sqrt_alpha_prod_t_prev.view(-1, 1, 1)
        sqrt_one_minus_alpha_prod_t_prev = sqrt_one_minus_alpha_prod_t_prev.view(-1, 1, 1)

        # Reconstruct x_{t_prev} by adding noise back to x0_pred
        # x_{t-1} = sqrt(alpha_prod_t_prev) * x_0 + sqrt(1-alpha_prod_t_prev) * z
        # where z is a new noise sample
        denoised_latent = sqrt_alpha_prod_t_prev * x0_pred + sqrt_one_minus_alpha_prod_t_prev * torch.randn_like(latent)
        return denoised_latent

    @torch.no_grad()
    def generate_samples(
        self,
        conditions: Any, # List[int] for class, List[str] for text
        num_samples: int,
        guidance_scale: Optional[float] = None,
        phase2_cfg_off_for_eval: bool = False,
        low_res_steps: Optional[int] = None,
        high_res_steps: Optional[int] = None
    ) -> List[Image.Image]:
        """
        Generates a batch of images using the Hi-MAR model with a two-phase process.

        Args:
            conditions: Conditioning information (class IDs as List[int] or text prompts as List[str]).
                        This list should have `num_samples` entries.
            num_samples: The number of images to generate.
            guidance_scale: The Classifier-Free Guidance scale. If None, uses default from config.
                            If 0, CFG is effectively off.
            phase2_cfg_off_for_eval: If True, CFG is disabled for Phase 2, matching the paper's
                                     "w/o CFG" evaluation for dense tokens.
            low_res_steps: Number of denoising steps for Phase 1. If None, uses config default.
            high_res_steps: Number of denoising steps for Phase 2. If None, uses config default.

        Returns:
            A list of generated PIL.Image.Image objects.
        """
        # Default steps if not provided
        low_res_steps = low_res_steps if low_res_steps is not None else self.phase1_inference_steps
        high_res_steps = high_res_steps if high_res_steps is not None else self.phase2_inference_steps
        guidance_scale = guidance_scale if guidance_scale is not None else self.default_guidance_scale

        is_class_conditional: bool = isinstance(conditions[0], int) if conditions else False # Assuming homogeneous list

        # 0. Prepare Conditions
        # Replicate conditions if a single condition is given for multiple samples
        if len(conditions) == 1 and num_samples > 1:
            conditions = conditions * num_samples
        if len(conditions) != num_samples:
            raise ValueError(f"Number of conditions ({len(conditions)}) must match num_samples ({num_samples}).")

        # Get conditional embeddings
        # The model's _get_condition_tokens expects a list of class IDs or a batched tensor of text embeddings.
        # It handles conversion to (B, L_cond, D_model) format.
        conditional_embeddings: torch.Tensor = self.model._get_condition_tokens(conditions, num_samples)
        
        do_cfg_overall: bool = guidance_scale > 0

        # Phase 1 CFG logic: The paper says CFG is ON for Phase 1 even in "w/o CFG" eval for dense tokens.
        # So `do_cfg_phase1` is true if `do_cfg_overall` is true AND `phase1_cfg_on_for_eval` is true.
        # Or, if `phase2_cfg_off_for_eval` is True, but `do_cfg_overall` is true, then Phase 1 still uses CFG.
        # Let's simplify: Phase 1 CFG is ON if `do_cfg_overall` is true (and it's explicitly allowed by `phase1_cfg_on_for_eval`).
        # This parameter seems to mainly exist to allow explicitly turning off phase 1 CFG for ablation studies,
        # but for standard evaluation, it's ON.
        do_cfg_phase1: bool = do_cfg_overall and self.phase1_cfg_on_for_eval
        
        # Phase 2 CFG logic: CFG is ON if `do_cfg_overall` is true AND `phase2_cfg_off_for_eval` is false.
        do_cfg_phase2: bool = do_cfg_overall and (not phase2_cfg_off_for_eval)

        unconditional_embeddings: Optional[torch.Tensor] = None
        if do_cfg_phase1 or do_cfg_phase2:
            unconditional_embeddings = self.model.get_null_conditions(num_samples, self.device)
        
        # Determine model_conditions for Phase 1. If CFG is active for P1, we concatenate cond and uncond.
        model_conditions_ph1 = torch.cat([conditional_embeddings, unconditional_embeddings], dim=0) \
                               if do_cfg_phase1 else conditional_embeddings
        
        # Get latent dimensions
        latent_h_low, latent_w_low = self.tokenizer.get_latent_hw('low')
        latent_h_high, latent_w_high = self.tokenizer.get_latent_hw('high')

        # Inference timesteps (discrete steps, from num_train_timesteps-1 down to 0)
        # These are the actual indices for the noise scheduler.
        inference_timesteps_ph1: List[int] = list(np.linspace(self.noise_scheduler_num_timesteps - 1, 0, low_res_steps + 1, dtype=int))
        inference_timesteps_ph2: List[int] = list(np.linspace(self.noise_scheduler_num_timesteps - 1, 0, high_res_steps + 1, dtype=int))

        logger.info(f"Generating {num_samples} samples with CFG={do_cfg_overall}, GS={guidance_scale}")
        logger.info(f"Phase 1 steps: {low_res_steps}, Phase 2 steps: {high_res_steps}")
        logger.info(f"Phase 1 CFG active: {do_cfg_phase1}, Phase 2 CFG active: {do_cfg_phase2}")

        # --- 1. Phase 1: Low-Resolution Generation ---
        # Initialize with pure Gaussian noise
        low_res_latents = torch.randn(
            num_samples, latent_h_low * latent_w_low, self.tokenizer_cfg['latent_channels'], device=self.device
        )
        
        # dummy_mask_indices is not used in inference, as we predict noise for all tokens.
        # It's required by the signature of `forward_phase1` but typically ignored in eval mode.
        dummy_mask_indices = torch.zeros(
            num_samples * (2 if do_cfg_phase1 else 1), latent_h_low * latent_w_low, dtype=torch.bool, device=self.device
        )

        phase1_pivot_features: torch.Tensor # Will store the final blended transformer output for Phase 2 pivots

        for i in range(low_res_steps):
            t_curr = inference_timesteps_ph1[i]
            t_prev = inference_timesteps_ph1[i+1] if i < low_res_steps - 1 else -1

            # Prepare current latents for model input. Duplicate if CFG is on.
            model_input_latents_ph1 = low_res_latents
            
            if do_cfg_phase1:
                model_input_latents_ph1_cfg_batch = torch.cat([model_input_latents_ph1, model_input_latents_ph1], dim=0)
                timesteps_cfg_batch = torch.full((num_samples * 2,), t_curr, device=self.device, dtype=torch.long)
            else:
                model_input_latents_ph1_cfg_batch = model_input_latents_ph1
                timesteps_cfg_batch = torch.full((num_samples,), t_curr, device=self.device, dtype=torch.long)
            
            # Predict noise and get transformer output
            predicted_noise_ph1_cfg, transformer_output_ph1_cfg = self.model.forward_phase1(
                masked_low_res_tokens=model_input_latents_ph1_cfg_batch,
                conditions=model_conditions_ph1,
                timesteps=timesteps_cfg_batch,
                mask_indices=dummy_mask_indices, # Not used in inference for this purpose
                scale_id=0 # Low-resolution scale
            )

            # Apply CFG if enabled for Phase 1
            if do_cfg_phase1:
                cond_noise_pred_ph1, uncond_noise_pred_ph1 = predicted_noise_ph1_cfg.chunk(2)
                cond_transformer_output_ph1, uncond_transformer_output_ph1 = transformer_output_ph1_cfg.chunk(2)

                model_output_noise_ph1 = uncond_noise_pred_ph1 + guidance_scale * (cond_noise_pred_ph1 - uncond_noise_pred_ph1)
                
                # Blend transformer outputs to be used as pivots for Phase 2
                phase1_pivot_features = uncond_transformer_output_ph1 + guidance_scale * (cond_transformer_output_ph1 - uncond_transformer_output_ph1)
            else:
                model_output_noise_ph1 = predicted_noise_ph1_cfg
                phase1_pivot_features = transformer_output_ph1_cfg # Use directly if no CFG for phase 1

            # Denoise step
            low_res_latents = self._denoise_step(
                model_input_latents_ph1, t_curr, t_prev, model_output_noise_ph1
            )
        
        # --- 2. Phase 2: High-Resolution Generation ---
        # Initialize with pure Gaussian noise
        high_res_latents = torch.randn(
            num_samples, latent_h_high * latent_w_high, self.tokenizer_cfg['latent_channels'], device=self.device
        )
        
        # dummy_mask_indices_high is not used in inference.
        dummy_mask_indices_high = torch.zeros(
            num_samples * (2 if do_cfg_phase2 else 1), latent_h_high * latent_w_high, dtype=torch.bool, device=self.device
        )

        for i in range(high_res_steps):
            t_curr = inference_timesteps_ph2[i]
            t_prev = inference_timesteps_ph2[i+1] if i < high_res_steps - 1 else -1

            model_input_latents_ph2 = high_res_latents
            
            # Prepare CFG-related inputs for Phase 2
            current_low_res_transformer_output_for_ph2 = phase1_pivot_features # This is the conditional pivot (possibly CFG-blended)
            
            # For the unconditional branch of Phase 2 CFG, we need unconditional pivots from Phase 1.
            # We obtain these by re-running Phase 1 forward with unconditional inputs using the *final* low-res latents.
            # A simpler approach is to generate the unconditional_pivots_ph1 once at the start of Phase 2
            # because the phase1_pivot_features are derived from the *final* low_res_latents.
            
            if do_cfg_phase2:
                model_input_latents_ph2_cfg_batch = torch.cat([model_input_latents_ph2, model_input_latents_ph2], dim=0)
                # Unconditional pivots are derived from the unconditional_embeddings for phase 1.
                # Since phase1_pivot_features might be CFG-blended, we need the pure unconditional equivalent.
                # The easiest is to use a null condition for the `low_res_transformer_output` arg of forward_phase2.
                # However, the design requires `low_res_transformer_output` as argument to forward_phase2 which acts as pivots.
                # So we must provide `unconditional_pivots_ph1` here.
                # Let's derive it using `model.forward_phase1` with `unconditional_embeddings`
                _, unconditional_pivots_ph1_for_cfg = self.model.forward_phase1(
                    masked_low_res_tokens=low_res_latents, # Use the final denoised low-res latents
                    conditions=unconditional_embeddings, # Pass the unconditional conditions
                    timesteps=torch.full((num_samples,), 0, device=self.device, dtype=torch.long), # Use timestep 0 for fully denoised context
                    mask_indices=dummy_mask_indices[:num_samples], # Only need single batch of mask indices
                    scale_id=0
                )
                model_pivots_for_ph2_cfg = torch.cat([current_low_res_transformer_output_for_ph2, unconditional_pivots_ph1_for_cfg], dim=0)
                model_conditions_ph2_cfg = torch.cat([conditional_embeddings, unconditional_embeddings], dim=0)
                timesteps_cfg_batch = torch.full((num_samples * 2,), t_curr, device=self.device, dtype=torch.long)
            else:
                model_input_latents_ph2_cfg_batch = model_input_latents_ph2
                model_pivots_for_ph2_cfg = current_low_res_transformer_output_for_ph2
                model_conditions_ph2_cfg = conditional_embeddings
                timesteps_cfg_batch = torch.full((num_samples,), t_curr, device=self.device, dtype=torch.long)

            # Predict noise for Phase 2
            predicted_noise_ph2_cfg = self.model.forward_phase2(
                masked_high_res_tokens=model_input_latents_ph2_cfg_batch, # input sequence to the transformer
                noisy_high_res_tokens=model_input_latents_ph2_cfg_batch, # input to the Diffusion Transformer head (y^t)
                low_res_transformer_output=model_pivots_for_ph2_cfg,
                conditions=model_conditions_ph2_cfg,
                timesteps=timesteps_cfg_batch,
                mask_indices=dummy_mask_indices_high, # Not used in inference for this purpose
                scale_id=1 # High-resolution scale
            )

            # Apply CFG if enabled for Phase 2
            if do_cfg_phase2:
                cond_noise_pred_ph2, uncond_noise_pred_ph2 = predicted_noise_ph2_cfg.chunk(2)
                model_output_noise_ph2 = uncond_noise_pred_ph2 + guidance_scale * (cond_noise_pred_ph2 - uncond_noise_pred_ph2)
            else:
                model_output_noise_ph2 = predicted_noise_ph2_cfg

            # Denoise step for high-resolution latents
            high_res_latents = self._denoise_step(
                model_input_latents_ph2, t_curr, t_prev, model_output_noise_ph2
            )

        # --- 3. Decode and Return ---
        decoded_images_tensor = self.tokenizer.decode(high_res_latents) # Returns images in [0, 1] range

        # Convert to PIL Images
        generated_pil_images: List[Image.Image] = []
        for i in range(num_samples):
            # C, H, W -> H, W, C for numpy conversion
            img_tensor = decoded_images_tensor[i].cpu().permute(1, 2, 0).numpy() 
            # Scale to 0-255 and convert to uint8
            img_pil = Image.fromarray((img_tensor * 255).astype(np.uint8))
            generated_pil_images.append(img_pil)

        return generated_pil_images

