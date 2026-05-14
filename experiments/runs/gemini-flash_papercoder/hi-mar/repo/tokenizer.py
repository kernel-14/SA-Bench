import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from diffusers.models import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
from typing import List, Tuple, Any, Optional

# To avoid circular import, we won't import Config directly here.
# Instead, the __init__ methods will receive the necessary config values as arguments.

class VAETokenizer:
    """
    Encapsulates the Variational Autoencoder (VAE) for converting images to and from
    continuous latent tokens. Handles different resolutions for latent representations.
    """
    def __init__(
        self,
        vae_path: str,
        latent_channels: int,
        high_res_image_size: int,
        low_res_image_size: int,
        device: str,
        vae_scaling_factor: Optional[float] = None
    ):
        """
        Initializes the VAETokenizer.

        Args:
            vae_path: Path to the pre-trained VAE model (e.g., a Hugging Face model ID or local path).
            latent_channels: The channel dimension of the latent tokens.
            high_res_image_size: The target size for high-resolution images (e.g., 256).
            low_res_image_size: The target size for low-resolution images (e.g., 128).
            device: The device to load the VAE onto ('cuda' or 'cpu').
            vae_scaling_factor: Optional, explicit scaling factor for VAE latents. If None,
                                it will try to retrieve it from the loaded VAE config.
        """
        self.device = device
        self.latent_channels = latent_channels
        self.high_res_image_size = high_res_image_size
        self.low_res_image_size = low_res_image_size

        try:
            # Assume it's a diffusers compatible path or model ID.
            # If vae_path is a local directory, AutoencoderKL.from_pretrained will load it.
            self.vae_model = AutoencoderKL.from_pretrained(vae_path, subfolder="vae")
        except Exception as e:
            # Fallback for simpler VAE loading if subfolder is not present or other issues
            print(f"Warning: Could not load VAE with subfolder 'vae' from {vae_path}. Trying direct load. Error: {e}")
            self.vae_model = AutoencoderKL.from_pretrained(vae_path)

        self.vae_model.to(self.device)
        self.vae_model.eval()
        for param in self.vae_model.parameters():
            param.requires_grad = False

        self.vae_scaling_factor = vae_scaling_factor
        if self.vae_scaling_factor is None and hasattr(self.vae_model.config, 'scaling_factor'):
            self.vae_scaling_factor = self.vae_model.config.scaling_factor
        elif self.vae_scaling_factor is None:
            # Default scaling factor if not found in config. Common for stable diffusion VAEs.
            print("Warning: VAE scaling factor not found in model config. Defaulting to 0.18215.")
            self.vae_scaling_factor = 0.18215 # A common default for Stable Diffusion VAEs

        # Assuming VAE downsampling factor is 8x based on 256 -> 32 and 128 -> 16 latents
        self.vae_downsampling_factor = 8

        # Define transform for VAE input (assuming VAE expects normalized images [-1, 1])
        self.normalize_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]), # Normalize to [-1, 1]
        ])

    def encode(self, images: torch.Tensor, resolution_key: str) -> torch.Tensor:
        """
        Encodes a batch of images into their continuous latent token representations.

        Args:
            images: A torch.Tensor of shape (B, C, H, W), expected to be in [0, 1] range.
            resolution_key: A string, either 'low' or 'high', indicating the target
                            resolution of the latent tokens.

        Returns:
            A torch.Tensor of shape (B, N_tokens, latent_channels) representing
            the encoded latent tokens.
        """
        if images.max() > 1.001 or images.min() < -0.001: # Check if images are already normalized to [-1,1]
             # If images are not [-1,1], apply the normalization that transform.ToTensor() -> [0,1] then [-1,1]
            images = self.normalize_transform(images).to(self.device) # Assume input is PIL Image or [0,255]
        else: # If they are already in [0,1], just move to device. Data module should handle this.
            images = images.to(self.device)
        
        original_h, original_w = images.shape[2:]

        # Resize images to high_res_image_size if they are smaller for consistent VAE input
        # The VAE is typically trained on a fixed input resolution (e.g., 256x256).
        if original_h != self.high_res_image_size or original_w != self.high_res_image_size:
            images = F.interpolate(
                images,
                size=(self.high_res_image_size, self.high_res_image_size),
                mode='bicubic',
                align_corners=False
            )

        # Encode with VAE
        with torch.no_grad():
            posterior = self.vae_model.encode(images).latent_dist
            latents = posterior.sample()
            latents = latents * self.vae_scaling_factor

        # If low-resolution tokens are requested, downsample the latent grid
        if resolution_key == 'low':
            target_latent_h = self.low_res_image_size // self.vae_downsampling_factor
            target_latent_w = self.low_res_image_size // self.vae_downsampling_factor
            current_latent_h, current_latent_w = latents.shape[2:]

            if current_latent_h != target_latent_h or current_latent_w != target_latent_w:
                latents = F.interpolate(
                    latents,
                    size=(target_latent_h, target_latent_w),
                    mode='area' # 'area' is good for downsampling to avoid aliasing
                )

        # Reshape from (B, C, H_latent, W_latent) to (B, N_tokens, C)
        batch_size, channels, latent_h, latent_w = latents.shape
        tokens = latents.permute(0, 2, 3, 1).reshape(batch_size, latent_h * latent_w, channels)
        return tokens

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decodes a batch of latent token sequences back into pixel-space images.

        Args:
            latents: A torch.Tensor of shape (B, N_tokens, latent_channels).

        Returns:
            A torch.Tensor of shape (B, C, H, W) representing the decoded images,
            in the range [0, 1].
        """
        # Determine the latent grid dimensions
        num_tokens = latents.shape[1]
        latent_h, latent_w = self.get_latent_hw_from_num_tokens(num_tokens)
        
        # Reshape from (B, N_tokens, latent_channels) to (B, latent_channels, H_latent, W_latent)
        latents_grid = latents.reshape(latents.shape[0], latent_h, latent_w, self.latent_channels).permute(0, 3, 1, 2)

        # Unscale latents
        latents_grid = latents_grid / self.vae_scaling_factor

        # Decode with VAE
        with torch.no_grad():
            images = self.vae_model.decode(latents_grid).sample

        # Post-processing: clamp to [0, 1] range after denormalizing from [-1, 1]
        images = (images / 2 + 0.5).clamp(0, 1)
        return images

    def get_latent_hw(self, resolution_key: str) -> Tuple[int, int]:
        """
        Calculates the height and width of the latent grid for a given resolution key.

        Args:
            resolution_key: 'low' or 'high'.

        Returns:
            A tuple (height, width) of the latent grid.
        """
        if resolution_key == 'high':
            latent_h = self.high_res_image_size // self.vae_downsampling_factor
            latent_w = self.high_res_image_size // self.vae_downsampling_factor
        elif resolution_key == 'low':
            latent_h = self.low_res_image_size // self.vae_downsampling_factor
            latent_w = self.low_res_image_size // self.vae_downsampling_factor
        else:
            raise ValueError(f"Invalid resolution_key: {resolution_key}. Must be 'low' or 'high'.")
        return latent_h, latent_w

    def get_latent_hw_from_num_tokens(self, num_tokens: int) -> Tuple[int, int]:
        """
        Infers the latent grid height and width from the total number of tokens.
        Assumes square latent grids.
        """
        side = int(num_tokens**0.5)
        if side * side != num_tokens:
            raise ValueError(f"Number of tokens {num_tokens} does not correspond to a square latent grid.")
        return side, side


class CLIPTextEncoder:
    """
    Encapsulates the CLIP text encoder for converting text prompts into embeddings.
    """
    def __init__(self, model_name: str, device: str):
        """
        Initializes the CLIPTextEncoder.

        Args:
            model_name: The name of the pre-trained CLIP model (e.g., "openai/clip-vit-large-patch14").
            device: The device to load the CLIP model onto ('cuda' or 'cpu').
        """
        self.device = device
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Encodes a list of text strings into CLIP embeddings.

        Args:
            texts: A list of strings (text prompts).

        Returns:
            A torch.Tensor of shape (batch_size, sequence_length, embedding_dim)
            representing the contextual text embeddings.
        """
        with torch.no_grad():
            encoded_inputs = self.tokenizer(
                texts,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoded_inputs.input_ids.to(self.device)
            attention_mask = encoded_inputs.attention_mask.to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # The last_hidden_state is typically used for text embeddings in models like CLIP
            last_hidden_state = outputs.last_hidden_state
        return last_hidden_state

