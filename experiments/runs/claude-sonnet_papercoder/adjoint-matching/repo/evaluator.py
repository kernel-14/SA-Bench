```python
## evaluator.py
"""Evaluator class for Adjoint Matching fine-tuning experiments.

This module implements the Evaluator class responsible for:
1. Generating images from fine-tuned models using ODE (σ=0) or memoryless SDE sampling
2. Computing all five evaluation metrics from the paper (Table 2):
   - ClipScore (text-image consistency, Hessel et al., 2021)
   - PickScore (human preference, Kirstain et al., 2023)
   - HPSv2 (generalization to unseen reward, Wu et al., 2023b)
   - DreamSim Diversity (sample diversity, Fu et al., 2023)
   - ImageReward (training reward monitoring, Xu et al., 2023)
3. Running the full evaluation suite including CFG ablation (Table 5)

Configuration alignment (config.yaml):
    evaluation.clipscore.model: "ViT-L-14"
    evaluation.clipscore.pretrained: "openai"
    evaluation.pickscore.model: "yuvalkirstain/PickScore_v1"
    evaluation.dreamsim_diversity.num_images_per_prompt: 25
    evaluation.dreamsim_diversity.num_prompts_for_diversity: 40
    evaluation.num_test_prompts: 1000
    inference.guidance_weights: [0.0, 1.0, 4.0]
    inference.num_steps: 40
    sampling.K: 40
    sampling.h: 0.025
    model.vae_scale_factor: 0.18215
    model.num_train_timesteps: 1000
    model.latent_channels: 4
    model.latent_height: 64
    model.latent_width: 64

Dependencies:
    - config.py: Config
    - noise_schedule.py: NoiseSchedule
    - utils.py: get_unet_timestep, latents_to_pil, set_seed
    - torch, torch.nn, open_clip, transformers, hpsv2, dreamsim, ImageReward
    - PIL.Image, numpy, math, typing, logging
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image as PILImage

from config import Config
from noise_schedule import NoiseSchedule
from utils import get_unet_timestep, latents_to_pil, set_seed

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants (from config.yaml)
# ---------------------------------------------------------------------------

# Number of UNet training timesteps (config.yaml model.num_train_timesteps: 1000)
_NUM_TRAIN_TIMESTEPS: int = 1000

# VAE latent scaling factor (config.yaml model.vae_scale_factor: 0.18215)
_VAE_SCALE_FACTOR: float = 0.18215

# Minimum sigma value to prevent division by zero
_SIGMA_MIN: float = 1e-6

# ClipScore scaling factor (standard: cosine_sim * 100, clamped to >= 0)
_CLIPSCORE_SCALE: float = 100.0

# Default evaluation batch size for metric computation (to avoid OOM)
_EVAL_BATCH_SIZE: int = 20


class Evaluator:
    """Evaluates fine-tuned Flow Matching models using paper metrics.

    Generates images from the fine-tuned model and computes all five
    evaluation metrics reported in Table 2 of the paper. Supports both
    ODE (σ=0) and memoryless SDE sampling modes, as well as CFG ablation.

    Metric models are loaded lazily on first use to avoid OOM when only
    a subset of metrics is needed.

    Attributes:
        config: Full experiment configuration.
        v_theta: Fine-tuned UNet velocity field (eval mode).
        v_base: Frozen base UNet velocity field (eval mode).
        vae: Frozen AutoencoderKL for latent→pixel decoding.
        text_encoder: Frozen CLIPTextModel for text conditioning.
        tokenizer: CLIPTokenizer for text tokenization.
        noise_schedule: NoiseSchedule for sigma and timestep computation.
        device: PyTorch device for tensor placement.
        dtype: Model inference dtype (bfloat16 matching training).

    Lazily loaded metric models (None until first use):
        _clip_model: open_clip ViT-L-14 model.
        _clip_preprocess: open_clip image preprocessing transform.
        _pickscore_processor: PickScore AutoProcessor.
        _pickscore_model: PickScore AutoModel.
        _imagereward_model: ImageReward model instance.
        _dreamsim_model: DreamSim model instance.
        _dreamsim_preprocess: DreamSim preprocessing transform.

    Example:
        >>> evaluator = Evaluator(config, v_theta, v_base, vae, text_encoder, tokenizer)
        >>> images = evaluator.generate_images(prompts, sigma_t_zero=True)
        >>> clipscore = evaluator.compute_clipscore(images, prompts)
        >>> results = evaluator.run_evaluation_suite(test_prompts)
    """

    def __init__(
        self,
        config: Config,
        v_theta: nn.Module,
        v_base: nn.Module,
        vae: nn.Module,
        text_encoder: nn.Module,
        tokenizer: Any,
    ) -> None:
        """Initialize the evaluator with all model components.

        Sets all models to eval mode and initializes the noise schedule.
        Metric models are NOT loaded here — they are loaded lazily on
        first use to minimize memory footprint.

        Args:
            config: Fully initialized Config object with all hyperparameters.
                Key fields used:
                    config.h (sampling.h: 0.025)
                    config.K (sampling.K: 40)
                    config.device (training.device)
                    config.precision (training.precision: "bfloat16")
                    config.vae_scale_factor (model.vae_scale_factor: 0.18215)
                    config.latent_channels (model.latent_channels: 4)
                    config.latent_height (model.latent_height: 64)
                    config.latent_width (model.latent_width: 64)
                    config.clipscore_model (evaluation.clipscore.model: "ViT-L-14")
                    config.clipscore_pretrained (evaluation.clipscore.pretrained: "openai")
                    config.pickscore_model (evaluation.pickscore.model: "yuvalkirstain/PickScore_v1")
                    config.num_images_per_prompt_diversity (evaluation.dreamsim_diversity.num_images_per_prompt: 25)
                    config.num_prompts_for_diversity (evaluation.dreamsim_diversity.num_prompts_for_diversity: 40)
                    config.guidance_weight (inference.guidance_weight: 0.0)
                    config.num_steps (inference.num_steps: 40)
            v_theta: Fine-tuned UNet velocity field. Set to eval mode.
                Parameters may have requires_grad=True (not modified here).
            v_base: Frozen base UNet velocity field. Set to eval mode.
                Parameters must have requires_grad=False.
            vae: Frozen AutoencoderKL. Set to eval mode.
                Must be in float32 for numerical stability in decode.
            text_encoder: Frozen CLIPTextModel. Set to eval mode.
            tokenizer: CLIPTokenizer for text tokenization.
                Has .model_max_length attribute (77 for SD1.5).
        """
        self.config: Config = config
        self.v_theta: nn.Module = v_theta
        self.v_base: nn.Module = v_base
        self.vae: nn.Module = vae
        self.text_encoder: nn.Module = text_encoder
        self.tokenizer: Any = tokenizer

        # Device and dtype setup
        self.device: torch.device = torch.device(config.device)
        self.dtype: torch.dtype = (
            torch.bfloat16
            if config.precision == "bfloat16"
            else (
                torch.float16
                if config.precision == "float16"
                else torch.float32
            )
        )

        # Noise schedule for sigma computation and timestep generation
        # config.yaml sampling.h: 0.025 (= 1/K = 1/40)
        self.noise_schedule: NoiseSchedule = NoiseSchedule(h=config.h)

        # Set all models to eval mode — no dropout, batch norm in eval
        self.v_theta.eval()
        self.v_base.eval()
        self.vae.eval()
        self.text_encoder.eval()

        # Lazy-loaded metric models (None until first use)
        self._clip_model: Optional[Any] = None
        self._clip_preprocess: Optional[Any] = None
        self._pickscore_processor: Optional[Any] = None
        self._pickscore_model: Optional[Any] = None
        self._imagereward_model: Optional[Any] = None
        self._dreamsim_model: Optional[Any] = None
        self._dreamsim_preprocess: Optional[Any] = None

        logger.info(
            "Evaluator initialized: device=%s, dtype=%s, K=%d, h=%.4f",
            str(self.device),
            str(self.dtype),
            config.K,
            config.h,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _encode_text(
        self,
        prompts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text prompts to CLIP encoder hidden states.

        Produces both conditional (text) and unconditional (empty string)
        embeddings for CFG support.

        Args:
            prompts: List of text prompt strings, length = batch_size.

        Returns:
            Tuple (text_emb, uncond_emb) where:
                text_emb: Conditional embeddings (B, 77, 768) in self.dtype.
                uncond_emb: Unconditional embeddings (B, 77, 768) in self.dtype.
        """
        batch_size: int = len(prompts)
        max_length: int = getattr(self.tokenizer, "model_max_length", 77)

        # Tokenize conditional prompts
        cond_tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        # Tokenize unconditional (empty string) prompts for CFG
        uncond_tokens = self.tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        # Encode with frozen text encoder (no gradient)
        with torch.no_grad():
            text_emb: torch.Tensor = self.text_encoder(
                cond_tokens
            ).last_hidden_state.to(dtype=self.dtype)
            uncond_emb: torch.Tensor = self.text_encoder(
                uncond_tokens
            ).last_hidden_state.to(dtype=self.dtype)

        return text_emb, uncond_emb

    def _call_unet(
        self,
        model: nn.Module,
        X_t: torch.Tensor,
        t: float,
        text_emb: torch.Tensor,
        batch_size: int,
        h: float,
    ) -> torch.Tensor:
        """Perform a single UNet forward pass and return velocity prediction.

        Handles dtype conversion and timestep tensor creation. The SD1.5 UNet
        uses ε-prediction (noise prediction). We convert to velocity using:
            v(x, t) = (x - (1-t)*ε) / t   [for α_t=t, β_t=1-t]

        This conversion is consistent with the Flow Matching framework where
        the velocity field v satisfies dX_t/dt = v(X_t, t).

        Args:
            model: UNet model (v_theta or v_base).
            X_t: Latent tensor of shape (B, C, H, W).
            t: Continuous time value in (0, 1].
            text_emb: CLIP text embeddings of shape (B, seq_len, hidden_dim).
            batch_size: Number of samples in the batch.
            h: Step size (used for timestep offset in sigma computation).

        Returns:
            Velocity prediction tensor of shape (B, C, H, W) in self.dtype.
        """
        model_dtype: torch.dtype = self.dtype
        device: torch.device = self.device

        # Convert continuous t to UNet integer timestep
        # Diffusers convention: t=0 (noise) → 1000, t=1 (clean) → 0
        timestep_int: int = get_unet_timestep(
            t_continuous=t,
            num_train_timesteps=_NUM_TRAIN_TIMESTEPS,
        )
        timestep_tensor: torch.Tensor = torch.tensor(
            [timestep_int] * batch_size,
            dtype=torch.long,
            device=device,
        )

        X_t_input: torch.Tensor = X_t.to(dtype=model_dtype, device=device)
        text_emb_input: torch.Tensor = text_emb.to(
            dtype=model_dtype, device=device
        )

        unet_output = model(
            X_t_input,
            timestep_tensor,
            encoder_hidden_states=text_emb_input,
            return_dict=True,
        )
        # UNet output is ε-prediction (noise prediction) for SD1.5
        eps_pred: torch.Tensor = unet_output.sample  # (B, C, H, W)

        # Convert ε-prediction to velocity for Flow Matching framework
        # For α_t = t, β_t = 1-t:
        #   v(x, t) = (x - β_t * ε) / α_t = (x - (1-t)*ε) / t
        # This is the velocity that satisfies dX_t/dt = v(X_t, t)
        # Clamp t to avoid division by zero (t >= h = 0.025 in practice)
        t_clamped: float = max(t, h)
        beta_t: float = 1.0 - t_clamped
        alpha_t: float = t_clamped

        v_pred: torch.Tensor = (
            X_t_input - beta_t * eps_pred
        ) / alpha_t

        return v_pred

    def _decode_latents_to_pil(
        self,
        latents: torch.Tensor,
    ) -> List[PILImage.Image]:
        """Decode latent tensors to PIL images via the VAE decoder.

        Args:
            latents: Tensor of shape (B, 4, 64, 64) in UNet latent space.
                May be in bfloat16; cast to float32 internally for stability.

        Returns:
            List of B PIL.Image.Image objects in RGB format.
        """
        return latents_to_pil(
            latents=latents,
            vae=self.vae,
            vae_scale_factor=self.config.vae_scale_factor,
        )

    # ------------------------------------------------------------------
    # Image generation
    # ------------------------------------------------------------------

    def generate_images(
        self,
        prompts: List[str],
        sigma_t_zero: bool = True,
        guidance_weight: float = 0.0,
        num_steps: int = 40,
    ) -> List[PILImage.Image]:
        """Generate images from the fine-tuned model.

        Implements Euler-Maruyama integration of the Flow Matching SDE/ODE.
        Supports two sampling modes:
            - ODE (sigma_t_zero=True): pure drift, no noise
              X_{t+h} = X_t + h * v_eff(X_t, t)
            - Memoryless SDE (sigma_t_zero=False): with noise
              X_{t+h} = X_t + h*(2*v_eff - κ_t*X_t) + sqrt(h)*σ(t)*ε

        Supports Classifier-Free Guidance (CFG) with the formula from Section 7:
            v_guided = (1+w)*v_theta(X_t, t | text) - w*v_base(X_t, t | "")

        Note: The unconditional branch uses v_base (not v_theta) as stated
        in the paper: "v(x,t) is an unconditional image model" (Section 7).

        All operations run under torch.no_grad() — no gradients needed.

        Args:
            prompts: List of text prompt strings, length = batch_size.
                Processed in mini-batches of _EVAL_BATCH_SIZE to avoid OOM.
            sigma_t_zero: If True, use ODE sampling (σ=0, primary evaluation).
                If False, use memoryless SDE sampling (σ=√(2η_t)).
                From config.yaml: sampling.inference_sigma: 0.0 (primary).
                Table 2 reports both modes for each method.
            guidance_weight: CFG guidance weight w.
                0.0 = no guidance (primary evaluation, config.yaml
                inference.guidance_weight: 0.0).
                1.0, 4.0 = CFG ablation (config.yaml
                inference.guidance_weights: [0.0, 1.0, 4.0]).
            num_steps: Number of Euler integration steps.
                Default 40 matches training (config.yaml sampling.K: 40).
                For Table 8 ablation: also tested at 10, 20, 100, 200.

        Returns:
            List of PIL.Image.Image objects in RGB format, length = len(prompts).
            Each image is 512×512 pixels for SD1.5.
        """
        all_images: List[PILImage.Image] = []

        # Process in mini-batches to avoid OOM with large prompt sets
        for batch_start in range(0, len(prompts), _EVAL_BATCH_SIZE):
            batch_prompts: List[str] = prompts[batch_start: batch_start + _EVAL_BATCH_SIZE]
            batch_images: List[PILImage.Image] = self._generate_images_batch(
                prompts=batch_prompts,
                sigma_t_zero=sigma_t_zero,
                guidance_weight=guidance_weight,
                num_steps=num_steps,
            )
            all_images.extend(batch_images)

        return all_images

    def _generate_images_batch(
        self,
        prompts: List[str],
        sigma_t_zero: bool,
        guidance_weight: float,
        num_steps: int,
    ) -> List[PILImage.Image]:
        """Generate images for a single mini-batch of prompts.

        Internal method called by generate_images() for each mini-batch.

        Args:
            prompts: Mini-batch of text prompts.
            sigma_t_zero: ODE (True) or SDE (False) sampling.
            guidance_weight: CFG weight w.
            num_steps: Number of Euler steps.

        Returns:
            List of PIL images for this mini-batch.
        """
        batch_size: int = len(prompts)

        # Compute step size for this evaluation (may differ from training h)
        # config.yaml sampling.h: 0.025 for K=40; for other num_steps, recompute
        h_eval: float = 1.0 / float(num_steps)

        # Build timestep list: [h, 2h, ..., 1.0] with num_steps elements
        # Use a local NoiseSchedule with the evaluation h
        eval_schedule: NoiseSchedule = NoiseSchedule(h=h_eval)
        timesteps: List[float] = eval_schedule.get_timesteps(K=num_steps)

        # ------------------------------------------------------------------
        # Step 1: Encode text prompts
        # ------------------------------------------------------------------
        text_emb, uncond_emb = self._encode_text(prompts)
        # text_emb: (B, 77, 768) in self.dtype

        # ------------------------------------------------------------------
        # Step 2: Sample initial noise X_0 ~ N(0, I)
        # config.yaml model.latent_channels: 4, latent_height: 64, latent_width: 64
        # ------------------------------------------------------------------
        X_t: torch.Tensor = torch.randn(
            batch_size,
            self.config.latent_channels,
            self.config.latent_height,
            self.config.latent_width,
            device=self.device,
            dtype=self.dtype,
        )

        # ------------------------------------------------------------------
        # Step 3: Euler-Maruyama integration loop
        # Iterate from t=h to t=1-h (K-1 steps), then final step to t=1
        # ------------------------------------------------------------------
        with torch.no_grad():
            for i, t in enumerate(timesteps[:-1]):
                # t is the current time; next time is timesteps[i+1]
                # We step from t to t+h

                # ----------------------------------------------------------
                # Compute effective velocity (with or without CFG)
                # ----------------------------------------------------------
                if guidance_weight > 0.0:
                    # CFG: v_guided = (1+w)*v_cond - w*v_uncond
                    # v_cond: fine-tuned conditional model
                    # v_uncond: BASE model with empty text (Section 7)
                    v_cond: torch.Tensor = self._call_unet(
                        model=self.v_theta,
                        X_t=X_t,
                        t=t,
                        text_emb=text_emb,
                        batch_size=batch_size,
                        h=h_eval,
                    )
                    v_uncond: torch.Tensor = self._call_unet(
                        model=self.v_base,  # Base model for unconditional
                        X_t=X_t,
                        t=t,
                        text_emb=uncond_emb,
                        batch_size=batch_size,
                        h=h_eval,
                    )
                    # CFG formula from Section 7:
                    # v_guided = (1+w)*v_cond - w*v_uncond
                    v_eff: torch.Tensor = (
                        (1.0 + guidance_weight) * v_cond
                        - guidance_weight * v_uncond
                    )
                else:
                    # No CFG: use fine-tuned conditional model directly
                    v_eff = self._call_unet(
                        model=self.v_theta,
                        X_t=X_t,
                        t=t,
                        text_emb=text_emb,
                        batch_size=batch_size,
                        h=h_eval,
                    )

                # ----------------------------------------------------------
                # Euler step based on sampling mode
                # ----------------------------------------------------------
                if sigma_t_zero:
                    # ODE sampling (σ=0): pure drift
                    # X_{t+h} = X_t + h * v_eff(X_t, t)
                    # Standard Flow Matching ODE (equation 3 of paper)
                    X_t = X_t + h_eval * v_eff

                else:
                    # Memoryless SDE sampling: with noise
                    # Drift: 2*v_eff - κ_t * X_t (equation 40 of paper)
                    # Noise: sqrt(h) * σ(t) * ε
                    kappa_t: float = eval_schedule.kappa(t)
                    drift: torch.Tensor = 2.0 * v_eff - kappa_t * X_t

                    # σ(t) = sqrt(2*(1-t+h)/(t+h)) with practical offset
                    sigma_t: float = eval_schedule.sigma_memoryless(
                        t=t, h=h_eval
                    )
                    sigma_t = max(sigma_t, _SIGMA_MIN)

                    noise: torch.Tensor = torch.randn_like(X_t)
                    X_t = (
                        X_t
                        + h_eval * drift
                        + math.sqrt(h_eval) * sigma_t * noise
                    )

                # Clamp latents to prevent extreme values
                X_t = X_t.clamp(-10.0, 10.0)

            # ------------------------------------------------------------------
            # Final step: t = 1-h → t = 1 (always ODE, no noise at terminal)
            # This matches the noiseless terminal step in TrajectorySampler
            # ------------------------------------------------------------------
            t_final: float = timesteps[-2] if len(timesteps) >= 2 else timesteps[-1]
            v_final: torch.Tensor
            if guidance_weight > 0.0:
                v_cond_final: torch.Tensor = self._call_unet(
                    model=self.v_theta,
                    X_t=X_t,
                    t=t_final,
                    text_emb=text_emb,
                    batch_size=batch_size,
                    h=h_eval,
                )
                v_uncond_final: torch.Tensor = self._call_unet(
                    model=self.v_base,
                    X_t=X_t,
                    t=t_final,
                    text_emb=uncond_emb,
                    batch_size=batch_size,
                    h=h_eval,
                )
                v_final = (
                    (1.0 + guidance_weight) * v_cond_final
                    - guidance_weight * v_uncond_final
                )
            else:
                v_final = self._call_unet(
                    model=self.v_theta,
                    X_t=X_t,
                    t=t_final,
                    text_emb=text_emb,
                    batch_size=batch_size,
                    h=h_eval,
                )
            # Final noiseless step (always ODE regardless of sigma_t_zero)
            X_1: torch.Tensor = X_t + h_eval * v_final

        # ------------------------------------------------------------------
        # Step 4: Decode final latent to PIL images
        # ------------------------------------------------------------------
        pil_images: List[PILImage.Image] = self._decode_latents_to_pil(X_1)

        return pil_images

    # ------------------------------------------------------------------
    # Metric computation: ClipScore
    # ------------------------------------------------------------------

    def _load_clip_model(self) -> None:
        """Lazily load the CLIP model for ClipScore computation.

        Loads ViT-L-14 with OpenAI pretrained weights using open_clip.
        Configuration from config.yaml:
            evaluation.clipscore.model: "ViT-L-14"
            evaluation.clipscore.pretrained: "openai"
        """
        if self._clip_model is not None:
            return

        try:
            import open_clip  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "open_clip is required for ClipScore computation. "
                "Install with: pip install open-clip-torch==2.23.0"
            ) from exc

        logger.info(
            "Loading CLIP model '%s' (pretrained='%s') for ClipScore...",
            self.config.clipscore_model,
            self.config.clipscore_pretrained,
        )

        # config.yaml evaluation.clipscore.model: "ViT-L-14"
        # config.yaml evaluation.clipscore.pretrained: "openai"
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            self.config.clipscore_model,
            pretrained=self.config.clipscore_pretrained,
        )
        clip_model = clip_model.to(self.device)
        clip_model.eval()

        # Freeze CLIP model — never fine-tuned
        for param in clip_model.parameters():
            param.requires_grad_(False)

        self._clip_model = clip_model
        self._clip_preprocess = clip_preprocess

        logger.info("CLIP model loaded for ClipScore.")

    def compute_clipscore(
        self,
        images: List[PILImage.Image],
        prompts: List[str],
    ) -> float:
        """Compute mean ClipScore (text-image consistency).

        Implements the standard ClipScore metric (Hessel et al., 2021):
            ClipScore = max(100 * cosine_similarity(image_emb, text_emb), 0)

        Uses open_clip with ViT-L-14 OpenAI pretrained weights.
        Paper reports values in the range 24-32 (Table 2).

        Configuration (config.yaml):
            evaluation.clipscore.model: "ViT-L-14"
            evaluation.clipscore.pretrained: "openai"

        Args:
            images: List of PIL.Image.Image objects in RGB format.
                Length must equal len(prompts).
            prompts: List of text prompt strings corresponding to each image.
                Length must equal len(images).

        Returns:
            Mean ClipScore as a Python float. Range approximately 0-100,
            with paper values in 24-32 range.

        Raises:
            ImportError: If open_clip is not installed.
            AssertionError: If len(images) != len(prompts).
        """
        assert len(images) == len(prompts), (
            f"Number of images ({len(images)}) must equal "
            f"number of prompts ({len(prompts)})."
        )

        if len(images) == 0:
            return 0.0

        # Lazy load CLIP model
        self._load_clip_model()

        import open_clip  # type: ignore[import]

        all_scores: List[float] = []

        # Process in mini-batches to avoid OOM
        for batch_start in range(0, len(images), _EVAL_BATCH_SIZE):
            batch_images: List[PILImage.Image] = images[
                batch_start: batch_start + _EVAL_BATCH_SIZE
            ]
            batch_prompts: List[str] = prompts[
                batch_start: batch_start + _EVAL_BATCH_SIZE
            ]

            # Preprocess images: apply CLIP preprocessing transform
            preprocessed_images: List[torch.Tensor] = []
            for img in batch_images:
                img_rgb: PILImage.Image = img.convert("RGB")
                preprocessed: torch.Tensor = self._clip_preprocess(img_rgb)
                preprocessed_images.append(preprocessed)

            # Stack to batch tensor: (B, 3, 224, 224)
            image_tensor: torch.Tensor = torch.stack(preprocessed_images).to(
                device=self.device
            )

            # Tokenize text prompts
            text_tokens: torch.Tensor = open_clip.tokenize(batch_prompts).to(
                device=self.device
            )

            with torch.no_grad():
                # Encode images and text
                image_features: torch.Tensor = self._clip_model.encode_image(
                    image_tensor
                )
                text_features: torch.Tensor = self._clip_model.encode_text(
                    text_tokens
                )

                # L2 normalize features
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True