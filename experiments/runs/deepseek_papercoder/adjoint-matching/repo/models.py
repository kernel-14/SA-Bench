## models.py

import copy
import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel
from PIL import Image
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer


class BaseModels:
    """
    Holds all pre‑trained, frozen components required for fine‑tuning and evaluation.

    The class loads a VAE, a flow‑matching U‑Net, a CLIP text encoder,
    and the ImageReward reward model.  It also provides utility methods
    to create velocity callables and reward‑gradient callables used by the
    fine‑tuning loop.

    Args:
        config: An object containing all configuration values read from
                ``config.yaml``.  Expected attributes are described in the
                constructor body.

    Attributes:
        vae:             Pre‑trained VAE (AutoencoderKL).
        flow_model:      Pre‑trained flow‑matching U‑Net (UNet2DConditionModel).
        clip_text_encoder:  Frozen CLIP text encoder.
        tokenizer:       Tokenizer matching the CLIP model.
        reward_model:    ImageReward model instance.
        unconditional_embedding:  Encoded hidden states for the empty prompt,
                                  used for classifier‑free guidance.
    """

    def __init__(self, config: Any) -> None:
        """
        Load all base models from disk or Hugging Face Hub.

        Raises:
            FileNotFoundError: if a required checkpoint is missing.
            RuntimeError:      if a model cannot be instantiated.
        """
        self._config = config

        # ---- 1. VAE ----
        self.vae = AutoencoderKL.from_pretrained(config.model.vae_model_name)

        # ---- 2. Flow Matching U‑Net ----
        # Architecture mirroring Stable Diffusion’s U‑Net; we load the config
        # from a public model and then inject custom FM weights.
        self.flow_model = self._load_flow_model(config)

        # ---- 3. CLIP text encoder & tokenizer ----
        self.clip_text_encoder = CLIPTextModel.from_pretrained(
            config.model.clip_model_name
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            config.model.clip_model_name
        )

        # ---- 4. ImageReward ----
        from ImageReward import ImageReward

        self.reward_model = ImageReward.load(config.model.reward_model_name)

        # ---- 5. Unconditional embedding for classifier-free guidance ----
        empty_prompt = [""]
        self.unconditional_embedding = self.encode_prompts(empty_prompt)

        # ---- Freeze EVERYTHING ----
        self.freeze_base()

    def _load_flow_model(self, config: Any) -> UNet2DConditionModel:
        """
        Instantiate a UNet2DConditionModel and load the provided FM state dict.

        The architecture is taken from a standard Stable Diffusion U‑Net;
        the checkpoint must contain compatible weights.
        """
        checkpoint_path = config.model.flow_model_checkpoint
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Flow matching checkpoint not found: {checkpoint_path}"
            )

        # Obtain the architecture configuration from a public model.
        # (caution: requires internet on first call; cached afterwards)
        unet_config = UNet2DConditionModel.load_config(
            "runwayml/stable-diffusion-v1-5", subfolder="unet"
        )
        flow_model = UNet2DConditionModel.from_config(unet_config)

        # Load custom FM weights.
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        flow_model.load_state_dict(state_dict)
        return flow_model

    def freeze_base(self) -> None:
        """Switch all base models to eval mode and disable gradient tracking."""
        for model in [
            self.vae,
            self.flow_model,
            self.clip_text_encoder,
            self.reward_model,
        ]:
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)

    def encode_prompts(self, prompts: List[str]) -> torch.Tensor:
        """
        Tokenize a list of prompts and produce encoder hidden states.

        Args:
            prompts: List of text strings.

        Returns:
            A tensor of shape ``(len(prompts), max_length, embedding_dim)``.
        """
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            encoder_output = self.clip_text_encoder(
                text_inputs.input_ids.to(self.vae.device)
            )
        return encoder_output.last_hidden_state

    def make_velocity_fn(
        self,
        model: nn.Module,
        encoder_hidden_states: torch.Tensor,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Create a callable that evaluates the velocity field of a model.

        The returned callable expects a batch of latents and a time tensor
        (scalar or 1‑D tensor of shape ``(batch,)``).

        Args:
            model:                 An instance of UNet2DConditionModel (either the
                                   frozen base model or the wrapped fine‑tuned model).
            encoder_hidden_states: Text conditioning of shape
                                   ``(batch, seq_len, embed_dim)``.

        Returns:
            A function ``v(x, t)`` returning a tensor of the same shape as ``x``.
        """
        # Ensure the batch dimensions agree.
        batch_size = encoder_hidden_states.shape[0]

        def velocity_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            # t may arrive as a scalar; convert to a 1‑D tensor of length B.
            if not torch.is_tensor(t):
                t = torch.full((batch_size,), t, device=x.device, dtype=x.dtype)
            # The U‑Net expects timesteps in its own way; some internal
            # embedding will map the raw value.  For FM, t is in [0, 1].
            return model(x, t, encoder_hidden_states=encoder_hidden_states)

        return velocity_fn

    def get_reward_grad_fn(
        self, prompt: str, lambda_: float
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Build a callable that computes ∇_x (λ · reward(decoded(x), prompt)).

        The returned function is suitable for initialising the lean adjoint
        terminal condition.  It expects a batch of latents of size **1**.

        Args:
            prompt:   The text prompt associated with the trajectory.
            lambda_:  Scaling coefficient for the reward (λ from the paper).

        Returns:
            A function ``grad_fn(x_hat1)`` that returns a detached gradient
            tensor of the same shape as ``x_hat1``.
        """
        # Pre‑compute reward model transforms
        reward_transform = transforms.Compose(
            [
                transforms.Resize((224, 224), interpolation=Image.LANCZOS),
                transforms.ToTensor(),
            ]
        )

        def grad_fn(x_hat1: torch.Tensor) -> torch.Tensor:
            assert x_hat1.shape[0] == 1, "Batch size must be 1 for reward grad"
            x_hat1.requires_grad_(True)

            # Decode latent to pixel space.
            with torch.no_grad():
                decoded = self.vae.decode(x_hat1).sample  # (1, 3, H, W)
                # The VAE output is around [-1, 1]; bring to [0, 1] for PIL.
                image_tensor = (decoded / 2 + 0.5).clamp(0, 1)
            # Convert to PIL and pre‑process for ImageReward.
            pil_img = transforms.ToPILImage()(image_tensor[0].cpu())
            pil_img = reward_transform(pil_img)  # now a tensor (C, 224, 224)
            # Move back to device
            pil_img = pil_img.unsqueeze(0).to(x_hat1.device)  # (1, C, 224, 224)

            # Compute reward (ImageReward.score expects PIL image; we already
            # have a pre‑processed tensor, so we call the underlying neural net).
            # We bypass the PIL interface for gradient computation.
            with torch.no_grad():
                # ImageReward’s internal model expects images in a specific range.
                # We use the model’s own pre‑processing, which is a ViT that takes
                # tensors after normalisation.  To avoid fragile internal calls,
                # we re‑process through the full pipeline but detach.
                reward_tensor = self.reward_model.model(pil_img)  # (1,)
            # Scale by λ.
            reward = lambda_ * reward_tensor.sum()  # ensures scalar

            # Gradient w.r.t. input latent.
            grad = torch.autograd.grad(reward, x_hat1, create_graph=False)[0]
            x_hat1.requires_grad_(False)
            return grad.detach()

        return grad_fn

    def decode_latent(self, latent: torch.Tensor) -> Image.Image:
        """
        Helper to decode a single latent to a PIL image (used in evaluation).

        Args:
            latent: A tensor of shape ``(1, 4, 64, 64)``.

        Returns:
            A PIL image of size 512×512.
        """
        with torch.no_grad():
            image = self.vae.decode(latent).sample
            image = (image / 2 + 0.5).clamp(0, 1)
        pil_image = transforms.ToPILImage()(image[0].cpu())
        return pil_image.resize((512, 512), Image.LANCZOS)


class FineTunedModel(nn.Module):
    """
    Wrapper around the flow‑matching U‑Net that holds the trainable parameters.

    At construction, a deep copy of the base model is made; only the copy’s
    parameters receive gradient updates during fine‑tuning.

    Args:
        base_flow_model: The frozen base FM U‑Net.
        config:          Configuration object (reserved for future use).

    Attributes:
        unet: The copied, trainable U‑Net.
    """

    def __init__(self, base_flow_model: nn.Module, config: Any) -> None:
        super().__init__()
        self._config = config
        # Deep copy ensures independent parameters while preserving structure.
        self.unet = copy.deepcopy(base_flow_model)
        # Enable gradient tracking on the fine‑tuned copy.
        for param in self.unet.parameters():
            param.requires_grad_(True)
        # The wrapper itself has no extra parameters.

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate the fine‑tuned velocity field.

        Args:
            x:                    Latent tensor of shape ``(B, C, H, W)``.
            t:                    Timesteps (raw float values in [0,1]),
                                  shape ``(B,)``.
            encoder_hidden_states: Text conditioning of shape
                                  ``(B, seq_len, embed_dim)``.

        Returns:
            Velocity field v_θ(x, t) of the same shape as x.
        """
        return self.unet(x, t, encoder_hidden_states=encoder_hidden_states)
