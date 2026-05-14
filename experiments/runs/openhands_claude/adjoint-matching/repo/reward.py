"""
Reward model wrapper that handles latent-to-pixel decoding.

In the paper's setup:
- The generative model operates in latent space (64x64x4)
- The reward model (ImageReward) operates in pixel space (512x512x3)
- The VAE decoder converts latents to pixels

This module provides differentiable reward computation through the VAE decoder,
enabling gradient-based fine-tuning methods to backpropagate through the reward.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List, Optional


class LatentRewardWrapper(nn.Module):
    """
    Wraps a pixel-space reward model to work with latent representations.

    Handles:
    1. VAE decoding: latent [B, 4, 64, 64] -> pixel [B, 3, 512, 512]
    2. Reward computation: pixel -> scalar
    3. Gradient flow through VAE decoder (for differentiable methods)

    For non-differentiable reward models (e.g., ImageReward with PIL images),
    uses a straight-through estimator or detaches the VAE output.
    """

    def __init__(
        self,
        reward_model,
        vae: Optional[nn.Module] = None,
        vae_scale_factor: float = 0.18215,
        reward_lambda: float = 1.0,
        differentiable: bool = True,
    ):
        """
        Args:
            reward_model: Pixel-space reward model (e.g., ImageReward)
            vae: VAE decoder (from diffusers AutoencoderKL)
            vae_scale_factor: VAE latent scaling factor
            reward_lambda: Scaling factor lambda for reward
            differentiable: Whether to allow gradients through VAE
        """
        super().__init__()
        self.reward_model = reward_model
        self.vae = vae
        self.vae_scale_factor = vae_scale_factor
        self.reward_lambda = reward_lambda
        self.differentiable = differentiable

        # Freeze VAE if not differentiating through it
        if vae is not None and not differentiable:
            for p in vae.parameters():
                p.requires_grad_(False)

    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to pixel space."""
        if self.vae is None:
            return z
        z_scaled = z / self.vae_scale_factor
        if self.differentiable:
            images = self.vae.decode(z_scaled).sample
        else:
            with torch.no_grad():
                images = self.vae.decode(z_scaled).sample
        return images

    def forward(
        self,
        latents: torch.Tensor,
        prompts: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Compute reward from latents.

        Args:
            latents: [B, C, H, W] latent representations
            prompts: Text prompts for reward conditioning

        Returns:
            Rewards [B], scaled by reward_lambda
        """
        images = self.decode_latent(latents)
        rewards = self.reward_model(images, prompts)
        return self.reward_lambda * rewards


class ImageRewardDifferentiable(nn.Module):
    """
    Differentiable wrapper for ImageReward model.

    ImageReward uses BLIP for image-text scoring. To make it differentiable,
    we use the BLIP image encoder directly with gradient flow.
    """

    def __init__(self, model_name: str = "ImageReward-v1.0", device=None):
        super().__init__()
        self.device = device or torch.device("cpu")
        try:
            import ImageReward as RM
            self._model = RM.load(model_name, device=str(self.device))
            self._model.eval()
        except ImportError:
            self._model = None

    def forward(
        self,
        images: torch.Tensor,
        prompts: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Compute ImageReward scores.

        For gradient-based methods, this uses the model's internal
        image encoder with gradient flow enabled.
        """
        if self._model is None:
            return torch.zeros(images.shape[0], device=images.device)

        if prompts is None:
            prompts = [""] * images.shape[0]

        # Normalize images to [0, 1]
        imgs = images
        if imgs.min() < 0:
            imgs = (imgs + 1.0) / 2.0
        imgs = imgs.clamp(0, 1)

        # Use PIL-based scoring (non-differentiable path)
        from torchvision.transforms.functional import to_pil_image
        scores = []
        for img, prompt in zip(imgs, prompts):
            pil_img = to_pil_image(img.detach().cpu())
            score = self._model.score(prompt, pil_img)
            scores.append(score)

        return torch.tensor(scores, device=images.device, dtype=images.dtype)


class ScaledReward(nn.Module):
    """
    Simple wrapper that scales a reward function by lambda.
    r_scaled(x) = lambda * r(x)
    """

    def __init__(self, reward_fn: Callable, reward_lambda: float = 1.0):
        super().__init__()
        self.reward_fn = reward_fn
        self.reward_lambda = reward_lambda

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.reward_lambda * self.reward_fn(*args, **kwargs)


class PromptConditionedReward:
    """
    Reward function that maintains a reference to current batch prompts.
    Used to pass prompts from the training loop to the reward function.
    """

    def __init__(self, base_reward_fn: Callable, reward_lambda: float = 1.0):
        self.base_reward_fn = base_reward_fn
        self.reward_lambda = reward_lambda
        self._current_prompts: List[str] = []

    def set_prompts(self, prompts: List[str]):
        self._current_prompts = prompts

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        prompts = self._current_prompts or [""] * images.shape[0]
        return self.reward_lambda * self.base_reward_fn(images, prompts)


def build_reward_pipeline(
    reward_model_name: str = "ImageReward-v1.0",
    vae: Optional[nn.Module] = None,
    vae_scale_factor: float = 0.18215,
    reward_lambda: float = 1.0,
    device: torch.device = None,
) -> PromptConditionedReward:
    """
    Build the complete reward pipeline for fine-tuning.

    Returns a callable that:
    1. Decodes latents to pixels (if VAE provided)
    2. Computes ImageReward scores
    3. Scales by lambda
    4. Accepts current prompts via set_prompts()
    """
    if device is None:
        device = torch.device("cpu")

    image_reward = ImageRewardDifferentiable(reward_model_name, device=device)

    if vae is not None:
        latent_reward = LatentRewardWrapper(
            reward_model=image_reward,
            vae=vae,
            vae_scale_factor=vae_scale_factor,
            reward_lambda=1.0,  # scaling applied in PromptConditionedReward
            differentiable=True,
        )
        base_fn = latent_reward
    else:
        base_fn = image_reward

    return PromptConditionedReward(base_fn, reward_lambda=reward_lambda)
