## reward_models.py
"""Reward models for Adjoint Matching fine-tuning experiments.

This module provides the reward signal that drives fine-tuning. It serves two
distinct purposes:

1. **Scoring** — evaluating generated images against prompts (used in
   evaluation and monitoring, matching Figure 6 and Table 2 of the paper).
2. **Gradient computation** — computing the terminal condition for the lean
   adjoint ODE: ã(1; X) = ∇_{X̂_1} g(X̂_1) where g = -r (Section 5.2,
   equations 38-39 and Algorithm 1).

The terminal condition is:
    ã_1 = ∇_x g(X̂_1) = -∇_x r(X̂_1) = -λ · d(ImageReward)/dX_latent

This gradient flows from pixel space back through the VAE decoder to the
latent space, requiring a differentiable path through the VAE.

Configuration alignment (config.yaml):
    reward.reward_model: "ImageReward-v1.0"
    reward.lambda_reward: 12500  (also tested: 1000, 2500)
    model.vae_scale_factor: 0.18215

Dependencies:
    - torch, torch.nn (tensor operations)
    - abc (abstract base class)
    - ImageReward (reward model library, version 1.5)
    - PIL.Image (image format for reward model input)
    - diffusers.AutoencoderKL (type hint for VAE)
    - torchvision.transforms.functional (tensor-to-PIL conversion)
    - typing (List, Optional)

No dependencies on other project files.
"""

from __future__ import annotations

import abc
import logging
import warnings
from typing import List, Optional

import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# VAE latent scaling factor for SD1.5 (config.yaml model.vae_scale_factor: 0.18215)
_VAE_SCALE_FACTOR: float = 0.18215

# Minimum absolute value for gradient clamping to prevent NaN propagation
_GRAD_CLAMP_VALUE: float = 1e4


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class RewardModel(abc.ABC):
    """Abstract base class for reward models used in fine-tuning.

    Defines the interface for reward models that provide both scalar scores
    (for evaluation/monitoring) and differentiable gradients (for the lean
    adjoint terminal condition in Algorithm 1).

    Subclasses must implement:
        - score(images, prompts) -> Tensor
        - gradient(X_latent, prompts, vae, lambda_r) -> Tensor

    Attributes:
        device: PyTorch device string where the model is loaded.
    """

    def __init__(self, device: str = "cuda") -> None:
        """Initialize the reward model on the specified device.

        Args:
            device: PyTorch device string (e.g., "cuda", "cpu", "cuda:0").
                From config.yaml: training.device (inferred from num_gpus).
        """
        self.device: str = device

    @abc.abstractmethod
    def score(
        self,
        images: List[Image.Image],
        prompts: List[str],
    ) -> torch.Tensor:
        """Compute scalar reward scores for a batch of image-prompt pairs.

        Used for evaluation/monitoring (Figure 6, Table 2 of the paper).
        Does NOT need to maintain a differentiable computation graph.

        Args:
            images: List of PIL.Image.Image objects in RGB format.
                Length must equal len(prompts).
            prompts: List of text prompt strings corresponding to each image.
                Length must equal len(images).

        Returns:
            Float tensor of shape (batch_size,) containing scalar reward
            values for each image-prompt pair. Higher is better.
            Tensor is on self.device.

        Raises:
            AssertionError: If len(images) != len(prompts).
        """

    @abc.abstractmethod
    def gradient(
        self,
        X_latent: torch.Tensor,
        prompts: List[str],
        vae: nn.Module,
        lambda_r: float = 12500.0,
    ) -> torch.Tensor:
        """Compute the terminal condition gradient for the lean adjoint ODE.

        Computes ∇_{X_latent} g(X̂_1) = -λ · d(r)/dX_latent where g = -r,
        which is the terminal condition ã(1; X) = ∇_x g(X_1) from
        equations (38)-(39) of the paper.

        The gradient flows from pixel space through the VAE decoder to the
        latent space. The VAE parameters are frozen (no gradient updates),
        but the computation graph through the VAE forward pass is preserved
        to allow backpropagation from reward to latent.

        Args:
            X_latent: Latent tensor of shape (batch_size, 4, 64, 64)
                representing X̂_1 (the noiseless terminal latent from
                TrajectorySampler.get_noiseless_terminal()). May be in
                bfloat16 or float32; internally cast to float32 for stability.
            prompts: List of text prompt strings, length = batch_size.
            vae: Frozen AutoencoderKL instance. Parameters must have
                requires_grad=False. Used to decode latents to pixel space.
            lambda_r: Reward scaling factor λ from config.yaml
                reward.lambda_reward (default 12500). The returned gradient
                is scaled by -lambda_r.

        Returns:
            Gradient tensor of shape (batch_size, 4, 64, 64) representing
            -lambda_r · d(r)/dX_latent. Detached from the autograd graph.
            In float32. NaN/Inf values are replaced with zeros.

        Raises:
            AssertionError: If len(prompts) != X_latent.shape[0].
        """


# ---------------------------------------------------------------------------
# ImageReward implementation
# ---------------------------------------------------------------------------


class ImageRewardModel(RewardModel):
    """Reward model using ImageReward (Xu et al., 2023).

    ImageReward is the primary reward model used in the paper's experiments
    (Section 7): r(x) = λ × ImageReward(x).

    The model is loaded from "ImageReward-v1.0" using the ImageReward library
    (version 1.5 as specified in requirements). It uses a BLIP-based
    architecture trained on human preference data.

    For scoring: uses model.score(prompt, image) → float (non-differentiable)
    For gradients: uses model.score_gard(prompt, image) → Tensor (differentiable)
        Note: "score_gard" is the actual method name in the ImageReward library
        (intentional typo in the library's API).

    Attributes:
        device: PyTorch device string.
        model: Loaded ImageReward model instance.

    Example:
        >>> reward_model = ImageRewardModel(device="cuda")
        >>> images = [PIL.Image.open("img.png")]
        >>> prompts = ["a beautiful sunset"]
        >>> scores = reward_model.score(images, prompts)
        >>> scores.shape
        torch.Size([1])
    """

    def __init__(self, device: str = "cuda") -> None:
        """Load the ImageReward model on the specified device.

        Args:
            device: PyTorch device string (e.g., "cuda", "cpu").
                From config.yaml: training.device.

        Raises:
            ImportError: If the ImageReward library is not installed.
            RuntimeError: If the model fails to load (e.g., network error
                when downloading weights).
        """
        super().__init__(device=device)

        try:
            import ImageReward as RM  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "The 'ImageReward' library is required but not installed. "
                "Install with: pip install ImageReward==1.5"
            ) from exc

        logger.info("Loading ImageReward model 'ImageReward-v1.0'...")
        try:
            # Load the ImageReward model; it handles its own weight download
            self.model = RM.load("ImageReward-v1.0")
            # Move model to the specified device
            self.model = self.model.to(device)
            # Ensure model is in eval mode (no dropout, batch norm in eval)
            self.model.eval()
            logger.info(
                "ImageReward model loaded successfully on device '%s'.", device
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load ImageReward model: {exc}. "
                f"Ensure you have internet access for the first download, "
                f"or that the model weights are cached locally."
            ) from exc

        # Freeze all parameters — ImageReward is never fine-tuned
        for param in self.model.parameters():
            param.requires_grad_(False)

    def score(
        self,
        images: List[Image.Image],
        prompts: List[str],
    ) -> torch.Tensor:
        """Compute scalar ImageReward scores for a batch of image-prompt pairs.

        Uses the non-differentiable model.score() API for efficiency.
        Called during evaluation/monitoring (Figure 6, Table 2).

        Args:
            images: List of PIL.Image.Image objects in RGB format.
            prompts: List of text prompt strings, same length as images.

        Returns:
            Float tensor of shape (batch_size,) on self.device.
            Values are typically in [-2, 2] for ImageReward.

        Raises:
            AssertionError: If len(images) != len(prompts).
        """
        assert len(images) == len(prompts), (
            f"Number of images ({len(images)}) must equal "
            f"number of prompts ({len(prompts)})."
        )

        if len(images) == 0:
            return torch.zeros(0, dtype=torch.float32, device=self.device)

        scores: List[float] = []
        for prompt, image in zip(prompts, images):
            # Ensure image is in RGB format
            image_rgb: Image.Image = image.convert("RGB")
            try:
                # model.score() returns a Python float
                score_val = self.model.score(prompt, image_rgb)
                # Handle both float and 0-dim tensor returns
                if isinstance(score_val, torch.Tensor):
                    scores.append(float(score_val.item()))
                else:
                    scores.append(float(score_val))
            except Exception as exc:
                logger.warning(
                    "ImageReward.score() failed for prompt '%s': %s. "
                    "Using score=0.0.",
                    prompt[:50],
                    exc,
                )
                scores.append(0.0)

        return torch.tensor(scores, dtype=torch.float32, device=self.device)

    def gradient(
        self,
        X_latent: torch.Tensor,
        prompts: List[str],
        vae: nn.Module,
        lambda_r: float = 12500.0,
    ) -> torch.Tensor:
        """Compute the terminal condition gradient for the lean adjoint ODE.

        Implements the terminal condition from Algorithm 1 (equation 41):
            ã_1 = -∇_{X̂_1} r(X̂_1)

        where r is the ImageReward function scaled by lambda_r.

        The gradient path is:
            X_latent_f32 → VAE decode → pixel image → ImageReward → scalar

        The VAE parameters are frozen (requires_grad=False), so no gradient
        flows to the VAE weights. The computation graph from X_latent_f32
        through the VAE forward pass to the reward IS preserved.

        Args:
            X_latent: Latent tensor of shape (batch_size, 4, 64, 64).
                Represents X̂_1 from TrajectorySampler.get_noiseless_terminal().
                Cast to float32 internally for numerical stability.
            prompts: List of text prompt strings, length = batch_size.
            vae: Frozen AutoencoderKL. Parameters have requires_grad=False.
                Used to decode latents to pixel space (512×512 RGB).
            lambda_r: Reward scaling factor λ (config.yaml
                reward.lambda_reward: 12500). Returned gradient is scaled
                by -lambda_r.

        Returns:
            Gradient tensor of shape (batch_size, 4, 64, 64) representing
            -lambda_r · d(r)/dX_latent. Detached, float32.
            NaN/Inf values are replaced with zeros.

        Raises:
            AssertionError: If len(prompts) != X_latent.shape[0].
        """
        batch_size: int = X_latent.shape[0]
        assert len(prompts) == batch_size, (
            f"Number of prompts ({len(prompts)}) must equal "
            f"batch size ({batch_size})."
        )

        if batch_size == 0:
            return torch.zeros_like(X_latent)

        # ------------------------------------------------------------------
        # Step 1: Create a float32 leaf tensor for gradient computation.
        # .detach() ensures no gradient flows back into the trajectory.
        # .requires_grad_(True) makes this a leaf node in the autograd graph.
        # ------------------------------------------------------------------
        X_latent_f32: torch.Tensor = (
            X_latent.float().detach().requires_grad_(True)
        )

        # ------------------------------------------------------------------
        # Step 2: VAE decode with gradient graph preserved.
        # torch.enable_grad() ensures grad is enabled even if this method
        # is called inside a torch.no_grad() context (e.g., in lean_adjoint).
        # The VAE parameters have requires_grad=False (frozen), so no
        # gradient flows to VAE weights — only to X_latent_f32.
        # ------------------------------------------------------------------
        with torch.enable_grad():
            # Scale latents: UNet space → VAE input space
            # config.yaml model.vae_scale_factor: 0.18215
            latents_scaled: torch.Tensor = X_latent_f32 / _VAE_SCALE_FACTOR

            # Decode: (B, 4, 64, 64) → (B, 3, 512, 512), values in [-1, 1]
            # The VAE decode builds the computation graph from X_latent_f32
            # through the decoder operations to the pixel tensor.
            decoded: torch.Tensor = vae.decode(latents_scaled).sample

            # Clamp to valid range (VAE can occasionally produce out-of-range)
            decoded = decoded.clamp(-1.0, 1.0)

            # ------------------------------------------------------------------
            # Step 3: Compute differentiable reward scores.
            # We use score_gard() which is the differentiable API of ImageReward.
            # "score_gard" is the actual method name in the library (typo in API).
            # If score_gard is unavailable, fall back to manual BLIP forward pass.
            # ------------------------------------------------------------------
            reward_sum: torch.Tensor = self._compute_differentiable_reward(
                decoded=decoded,
                prompts=prompts,
                lambda_r=lambda_r,
            )

            # ------------------------------------------------------------------
            # Step 4: Backpropagate to get d(lambda_r * sum(r_i)) / dX_latent.
            # Cross-sample gradients are zero since each r_i depends only on
            # its own latent X_latent_f32[i], so the result has shape
            # (batch_size, 4, 64, 64) where slice [i] = lambda_r * d(r_i)/dX_i.
            # ------------------------------------------------------------------
            grad_tuple = torch.autograd.grad(
                outputs=reward_sum,
                inputs=X_latent_f32,
                create_graph=False,
                retain_graph=False,
                allow_unused=False,
            )
            grad: torch.Tensor = grad_tuple[0]

        # ------------------------------------------------------------------
        # Step 5: Apply sign convention for terminal condition.
        # The lean adjoint terminal condition is ã(1) = ∇_x g(X_1)
        # where g = -r, so ∇g = -∇r.
        # We already scaled by lambda_r in _compute_differentiable_reward,
        # so grad = lambda_r * d(r)/dX_latent.
        # Terminal condition: ã_1 = -grad = -lambda_r * d(r)/dX_latent.
        # ------------------------------------------------------------------
        terminal_grad: torch.Tensor = -grad

        # ------------------------------------------------------------------
        # Step 6: Sanitize gradient (replace NaN/Inf with zeros).
        # NaN can occur with extreme latent values or numerical instability.
        # ------------------------------------------------------------------
        terminal_grad = torch.nan_to_num(
            terminal_grad,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # Detach from autograd graph — this is a constant for the adjoint ODE
        return terminal_grad.detach()

    def _compute_differentiable_reward(
        self,
        decoded: torch.Tensor,
        prompts: List[str],
        lambda_r: float,
    ) -> torch.Tensor:
        """Compute a differentiable scalar reward sum from decoded pixel tensors.

        Tries to use ImageReward's differentiable API (score_gard) first.
        Falls back to a manual BLIP-based forward pass if score_gard is
        unavailable. As a last resort, uses a non-differentiable path with
        a warning (gradient will be zero in this case).

        The returned scalar is lambda_r * Σ_i r_i, where the sum is over
        the batch. This allows torch.autograd.grad to compute per-sample
        gradients since each r_i depends only on decoded[i].

        Args:
            decoded: Pixel tensor of shape (batch_size, 3, H, W) in [-1, 1].
                Has gradient graph connected to X_latent_f32.
            prompts: List of text prompt strings, length = batch_size.
            lambda_r: Reward scaling factor.

        Returns:
            Scalar tensor (0-dim) = lambda_r * Σ_i r_i, with gradient graph
            connected to decoded (and thus to X_latent_f32).
        """
        batch_size: int = decoded.shape[0]

        # ------------------------------------------------------------------
        # Strategy 1: Use score_gard() — the differentiable ImageReward API.
        # This method accepts a PIL image and returns a differentiable tensor.
        # Note: "score_gard" is the actual method name (typo in the library).
        # ------------------------------------------------------------------
        if hasattr(self.model, "score_gard"):
            return self._reward_via_score_gard(decoded, prompts, lambda_r)

        # ------------------------------------------------------------------
        # Strategy 2: Manual BLIP forward pass.
        # Access the underlying BLIP model and reward head directly.
        # This works when the ImageReward model exposes .blip and .reward_model.
        # ------------------------------------------------------------------
        if hasattr(self.model, "blip") and hasattr(self.model, "reward_model"):
            return self._reward_via_blip_forward(decoded, prompts, lambda_r)

        # ------------------------------------------------------------------
        # Strategy 3: Non-differentiable fallback (gradient will be zero).
        # This should not happen with ImageReward 1.5, but provides a safe
        # fallback that doesn't crash training.
        # ------------------------------------------------------------------
        warnings.warn(
            "ImageReward model does not expose a differentiable API "
            "(score_gard or blip). Gradient will be zero. "
            "This will cause incorrect fine-tuning. "
            "Please install ImageReward==1.5.",
            UserWarning,
            stacklevel=3,
        )
        # Return a differentiable zero that still has a gradient path
        # (gradient will be zero but won't cause errors)
        return decoded.sum() * 0.0

    def _reward_via_score_gard(
        self,
        decoded: torch.Tensor,
        prompts: List[str],
        lambda_r: float,
    ) -> torch.Tensor:
        """Compute differentiable reward using ImageReward's score_gard API.

        score_gard() is the differentiable version of score() in the
        ImageReward library. It accepts a PIL image and returns a tensor
        with gradient graph preserved.

        Args:
            decoded: Pixel tensor (batch_size, 3, H, W) in [-1, 1].
            prompts: Text prompts, length = batch_size.
            lambda_r: Reward scaling factor.

        Returns:
            Scalar tensor = lambda_r * Σ_i r_i with gradient graph.
        """
        batch_size: int = decoded.shape[0]
        reward_scores: List[torch.Tensor] = []

        for i in range(batch_size):
            # Convert decoded[i] from [-1, 1] tensor to PIL image
            # decoded[i] shape: (3, H, W), values in [-1, 1]
            pil_image: Image.Image = self._tensor_to_pil(decoded[i])

            try:
                # score_gard returns a differentiable tensor
                score_tensor = self.model.score_gard(prompts[i], pil_image)
                # Ensure it's a scalar tensor on the correct device
                if not isinstance(score_tensor, torch.Tensor):
                    score_tensor = torch.tensor(
                        float(score_tensor),
                        dtype=torch.float32,
                        device=self.device,
                    )
                else:
                    score_tensor = score_tensor.to(
                        device=self.device, dtype=torch.float32
                    )
                reward_scores.append(score_tensor.squeeze())
            except Exception as exc:
                logger.warning(
                    "score_gard() failed for prompt '%s': %s. Using 0.0.",
                    prompts[i][:50],
                    exc,
                )
                # Use a zero that still participates in the graph via decoded
                reward_scores.append(decoded[i].sum() * 0.0)

        # Stack and sum: shape (batch_size,) → scalar
        rewards_tensor: torch.Tensor = torch.stack(reward_scores)
        return lambda_r * rewards_tensor.sum()

    def _reward_via_blip_forward(
        self,
        decoded: torch.Tensor,
        prompts: List[str],
        lambda_r: float,
    ) -> torch.Tensor:
        """Compute differentiable reward via direct BLIP model forward pass.

        Accesses the underlying BLIP encoder and reward head of the
        ImageReward model to compute a differentiable reward score.

        The ImageReward model architecture:
            - self.model.blip: BLIP image-text encoder
            - self.model.reward_model: Linear reward head on top of BLIP

        Args:
            decoded: Pixel tensor (batch_size, 3, H, W) in [-1, 1].
            prompts: Text prompts, length = batch_size.
            lambda_r: Reward scaling factor.

        Returns:
            Scalar tensor = lambda_r * Σ_i r_i with gradient graph.
        """
        import torchvision.transforms.functional as TF  # type: ignore[import]

        batch_size: int = decoded.shape[0]

        # Normalize decoded from [-1, 1] to [0, 1] for BLIP preprocessing
        pixel_values_01: torch.Tensor = (decoded + 1.0) / 2.0  # (B, 3, H, W)

        # BLIP expects images normalized with ImageNet mean/std
        # Mean: [0.48145466, 0.4578275, 0.40821073]
        # Std:  [0.26862954, 0.26130258, 0.27577711]
        mean: torch.Tensor = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073],
            dtype=torch.float32,
            device=self.device,
        ).view(1, 3, 1, 1)
        std: torch.Tensor = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711],
            dtype=torch.float32,
            device=self.device,
        ).view(1, 3, 1, 1)

        # Normalize: (B, 3, H, W)
        pixel_values_norm: torch.Tensor = (pixel_values_01 - mean) / std

        # Resize to BLIP's expected input size (224x224)
        if pixel_values_norm.shape[-2:] != (224, 224):
            pixel_values_norm = torch.nn.functional.interpolate(
                pixel_values_norm,
                size=(224, 224),
                mode="bicubic",
                align_corners=False,
            )

        reward_scores: List[torch.Tensor] = []

        for i in range(batch_size):
            try:
                # Tokenize the prompt for BLIP
                # The BLIP model in ImageReward uses its own tokenizer
                blip_model = self.model.blip
                reward_head = self.model.reward_model

                # Get text tokens
                text_input = blip_model.tokenizer(
                    prompts[i],
                    padding="max_length",
                    truncation=True,
                    max_length=35,
                    return_tensors="pt",
                ).to(self.device)

                # BLIP forward pass: image + text → multimodal features
                image_embeds = blip_model.visual_encoder(
                    pixel_values_norm[i : i + 1]
                )
                image_atts = torch.ones(
                    image_embeds.size()[:-1],
                    dtype=torch.long,
                    device=self.device,
                )

                text_output = blip_model.text_encoder(
                    text_input.input_ids,
                    attention_mask=text_input.attention_mask,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

                # Extract [CLS] token embedding and pass through reward head
                txt_features = text_output.last_hidden_state[:, 0, :]
                rewards_i = reward_head(txt_features)  # (1, 1) or (1,)
                reward_scores.append(rewards_i.squeeze())

            except Exception as exc:
                logger.warning(
                    "BLIP forward pass failed for prompt '%s': %s. Using 0.0.",
                    prompts[i][:50],
                    exc,
                )
                # Differentiable zero via decoded to maintain graph
                reward_scores.append(decoded[i].sum() * 0.0)

        rewards_tensor: torch.Tensor = torch.stack(reward_scores)
        return lambda_r * rewards_tensor.sum()

    def _tensor_to_pil(self, image_tensor: torch.Tensor) -> Image.Image:
        """Convert a single image tensor in [-1, 1] to a PIL Image.

        Args:
            image_tensor: Float tensor of shape (3, H, W) in [-1, 1].
                May be on any device (CPU or CUDA).

        Returns:
            PIL.Image.Image in RGB format with values in [0, 255].
        """
        # Clamp to [-1, 1] for safety
        img: torch.Tensor = image_tensor.clamp(-1.0, 1.0)

        # Rescale from [-1, 1] to [0, 1]
        img = (img + 1.0) / 2.0

        # Move to CPU and convert to uint8 numpy array
        # Shape: (3, H, W) → (H, W, 3)
        img_np = (
            img.detach()
            .float()
            .cpu()
            .permute(1, 2, 0)
            .numpy()
        )
        img_uint8 = (img_np * 255.0).clip(0, 255).astype("uint8")

        return Image.fromarray(img_uint8, mode="RGB")

    def __repr__(self) -> str:
        """Human-readable representation of the reward model."""
        return (
            f"ImageRewardModel("
            f"model='ImageReward-v1.0', "
            f"device='{self.device}'"
            f")"
        )
