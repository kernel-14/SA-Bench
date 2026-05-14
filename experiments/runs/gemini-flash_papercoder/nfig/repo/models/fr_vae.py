import math
import os
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# Assuming Config class is available from config.py
# Assuming utility functions are available from utils.py
from config import Config
from utils import (
    get_frequency_masks,
    init_weights,
    interpolate_feature_map,
    straight_through_estimator,
)


class _ResidualBlock(nn.Module):
    """
    A standard residual block with two convolutional layers, normalization, and activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = "group",
        activation: nn.Module = nn.SiLU(),
    ):
        """
        Initializes a residual block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            norm_type: Type of normalization ('group' or 'batch'). Default is 'group'.
            activation: Activation function to use. Default is SiLU.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation

        if norm_type == "group":
            self.norm1 = nn.GroupNorm(
                num_groups=min(32, in_channels), num_channels=in_channels, eps=1e-6
            )
            self.norm2 = nn.GroupNorm(
                num_groups=min(32, out_channels), num_channels=out_channels, eps=1e-6
            )
        elif norm_type == "batch":
            self.norm1 = nn.BatchNorm2d(in_channels)
            self.norm2 = nn.BatchNorm2d(out_channels)
        else:
            raise ValueError(f"Unknown normalization type: {norm_type}")

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Skip connection for differing input/output channels
        self.nin_shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the residual block.
        """
        h = x
        h = self.norm1(h)
        h = self.activation(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.activation(h)
        h = self.conv2(h)

        return self.nin_shortcut(x) + h


class _Encoder(nn.Module):
    """
    Convolutional encoder for the FR-VAE, downsampling an image to a latent feature map.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,  # Latent C
        image_size: int,
        latent_size: int,  # H', W'
        num_res_blocks: int = 2,
        ch_mults: List[int] = None,
        base_channels: int = 128,  # Base number of channels for first block
        dino_v2_pretrained_path: Optional[str] = None,
    ):
        """
        Initializes the encoder.

        Args:
            in_channels: Number of input image channels (e.g., 3 for RGB).
            out_channels: Number of output latent channels (C).
            image_size: Input image height/width (e.g., 256).
            latent_size: Target latent feature map height/width (H', W').
            num_res_blocks: Number of residual blocks at each resolution level.
            ch_mults: List of channel multipliers for each downsampling stage.
                      The length of this list determines the number of downsampling steps.
            base_channels: Initial number of channels after the first convolution.
            dino_v2_pretrained_path: Path to DINOv2 pretrained weights.
                                     Currently, this implementation only supports direct
                                     convolutional encoder initialization. A DINOv2 ViT
                                     would require specific adaptation not covered here.
        """
        super().__init__()

        if ch_mults is None:
            ch_mults = [1, 2, 4, 8]

        self.num_resolutions = len(ch_mults)
        self.current_resolution = image_size
        self.latent_size = latent_size

        # Validate that image_size can be downsampled to latent_size
        expected_latent_size = image_size // (2**self.num_resolutions)
        if expected_latent_size != latent_size:
            raise ValueError(
                f"Image size {image_size} with {self.num_resolutions} downsamples"
                f" (ch_mults length) results in latent size {expected_latent_size},"
                f" but expected {latent_size}."
            )

        blocks = []

        # Initial convolution
        blocks.append(nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1))

        # Downsampling blocks with residual connections
        in_ch = base_channels
        for i, mult in enumerate(ch_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                blocks.append(_ResidualBlock(in_ch, out_ch))
                in_ch = out_ch

            if i < self.num_resolutions - 1:  # Only downsample if not the last stage
                blocks.append(nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1))
                self.current_resolution //= 2

        # Final residual blocks
        for _ in range(num_res_blocks):
            blocks.append(_ResidualBlock(in_ch, out_channels))

        # Final output convolution
        blocks.append(nn.GroupNorm(min(32, out_channels), out_channels))
        blocks.append(nn.SiLU())
        blocks.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

        self.model = nn.Sequential(*blocks)

        # Initialize weights (DINOv2 adaptation is commented out for now)
        self.apply(init_weights)
        if dino_v2_pretrained_path and os.path.exists(dino_v2_pretrained_path):
            print(
                f"Warning: DINOv2 pretraining for convolutional encoder is not directly "
                f"supported by this implementation. {dino_v2_pretrained_path} will be ignored."
            )
            # Future work: Implement DINOv2 feature distillation or a compatible loading mechanism.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the encoder.

        Args:
            x: Input image tensor (B, C_in, H, W).

        Returns:
            Latent feature map (B, C_latent, H', W').
        """
        if x.ndim != 4:
            raise ValueError(f"Input to Encoder must be 4D (B, C, H, W), but got {x.ndim}D.")
        return self.model(x)


class _Decoder(nn.Module):
    """
    Convolutional decoder for the FR-VAE, upsampling a latent feature map back to an image.
    """

    def __init__(
        self,
        in_channels: int,  # Latent C
        out_channels: int,  # Output image channels (e.g., 3 for RGB)
        image_size: int,
        latent_size: int,  # H', W'
        num_res_blocks: int = 2,
        ch_mults: List[int] = None,
        base_channels: int = 128,
    ):
        """
        Initializes the decoder.

        Args:
            in_channels: Number of input latent channels (C).
            out_channels: Number of output image channels.
            image_size: Target image height/width.
            latent_size: Input latent feature map height/width.
            num_res_blocks: Number of residual blocks at each resolution level.
            ch_mults: List of channel multipliers (reversed from encoder for upsampling).
            base_channels: Initial number of channels for the highest resolution latent.
        """
        super().__init__()

        if ch_mults is None:
            ch_mults = [8, 4, 2, 1]  # Reversed from encoder

        self.num_resolutions = len(ch_mults)
        self.current_resolution = latent_size

        blocks = []

        # Initial convolution and residual blocks
        in_ch = base_channels * ch_mults[0]
        blocks.append(nn.Conv2d(in_channels, in_ch, kernel_size=3, padding=1))
        for _ in range(num_res_blocks):
            blocks.append(_ResidualBlock(in_ch, in_ch))

        # Upsampling blocks with residual connections
        for i, mult in enumerate(ch_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                blocks.append(_ResidualBlock(in_ch, out_ch))
                in_ch = out_ch

            if i < self.num_resolutions - 1:  # Only upsample if not the last stage
                blocks.append(nn.ConvTranspose2d(in_ch, in_ch, kernel_size=4, stride=2, padding=1))
                self.current_resolution *= 2

        # Final residual blocks
        for _ in range(num_res_blocks):
            blocks.append(_ResidualBlock(in_ch, in_ch))

        # Final output convolution
        blocks.append(nn.GroupNorm(min(32, in_ch), in_ch))
        blocks.append(nn.SiLU())
        blocks.append(nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1))

        self.model = nn.Sequential(*blocks)
        self.apply(init_weights)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the decoder.

        Args:
            z: Latent feature map (B, C_latent, H', W').

        Returns:
            Reconstructed image tensor (B, C_out, H, W).
        """
        if z.ndim != 4:
            raise ValueError(f"Input to Decoder must be 4D (B, C, H, W), but got {z.ndim}D.")
        return self.model(z)


class _FrequencyDecomposer(nn.Module):
    """
    Decomposes a latent feature map into multiple frequency components
    using 2D FFT, frequency masks, and inverse FFT.
    """

    def __init__(
        self,
        H_prime: int,
        W_prime: int,
        n_bands: int,
        token_dims: List[Tuple[int, int]],
        device: torch.device,
    ):
        """
        Initializes the frequency decomposer.

        Args:
            H_prime: Height of the latent feature map.
            W_prime: Width of the latent feature map.
            n_bands: Number of frequency bands.
            token_dims: List of (h_i, w_i) for each band, used to calculate mask cutoffs.
            device: The device to generate the masks on.
        """
        super().__init__()
        self.H_prime = H_prime
        self.W_prime = W_prime
        self.n_bands = n_bands
        self.token_dims = token_dims

        # Generate and store masks (assuming they are fixed after initialization)
        self.masks: List[torch.Tensor] = []
        # Masks are (H_prime, W_prime), need to be expanded for (B, C, H', W')
        raw_masks = get_frequency_masks(H_prime, W_prime, n_bands, token_dims, device)
        for mask in raw_masks:
            self.masks.append(mask.unsqueeze(0).unsqueeze(0))  # Shape (1, 1, H', W')

    def apply_masks(self, f: torch.Tensor) -> List[torch.Tensor]:
        """
        Applies frequency masks to the latent feature map in the Fourier domain.

        Args:
            f: Latent feature map (B, C, H', W').

        Returns:
            A list of `n_bands` tensors, where each tensor is a frequency component
            (B, C, H', W').
        """
        if f.ndim != 4:
            raise ValueError(f"Input to Decomposer must be 4D (B, C, H, W), but got {f.ndim}D.")

        # Move masks to the same device as input feature map
        current_masks = [mask.to(f.device) for mask in self.masks]

        # Apply 2D FFT
        # torch.fft.fft2d operates on the last two dimensions
        F_complex = torch.fft.fft2d(f)
        F_shifted = torch.fft.fftshift(F_complex, dim=(-2, -1))

        hat_f_i_list = []
        for mask_i in current_masks:
            # Mask_i is (1, 1, H', W'), F_shifted is (B, C, H', W')
            masked_F_shifted = F_shifted * mask_i
            masked_F = torch.fft.ifftshift(masked_F_shifted, dim=(-2, -1))
            hat_f_i = torch.fft.ifft2d(masked_F).real  # Take real part as input `f` is real
            hat_f_i_list.append(hat_f_i)

        return hat_f_i_list


class _FrequencyQuantizer(nn.Module):
    """
    Performs residual vector quantization for each frequency band.
    """

    def __init__(
        self,
        codebook_size: int,
        embedding_dim: int,
        n_bands: int,
        token_dims: List[Tuple[int, int]],
        H_prime: int,
        W_prime: int,
        codebook_loss_beta: float = 0.25,
    ):
        """
        Initializes the frequency quantizer.

        Args:
            codebook_size: Size of the learnable codebook (K).
            embedding_dim: Dimension of each codebook vector (C).
            n_bands: Number of frequency bands.
            token_dims: List of (h_i, w_i) for each band.
            H_prime: Original height of the latent feature map.
            W_prime: Original width of the latent feature map.
            codebook_loss_beta: Weight for the codebook loss term.
        """
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, embedding_dim)
        self.token_dims = token_dims
        self.H_prime = H_prime
        self.W_prime = W_prime
        self.codebook_loss_beta = codebook_loss_beta

    def quantize_band(
        self, input_to_quantize: torch.Tensor, band_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantizes a single frequency band's feature map.

        Args:
            input_to_quantize: The feature map to quantize (B, C, H', W'), which is (R_{i-1} + hat_f_i).
            band_idx: The index of the current frequency band.

        Returns:
            A tuple containing:
            - v_i_quantized_downsampled: Quantized feature map for this band, at its target resolution (B, C, h_i, w_i).
            - indices: Codebook indices for this band (B, h_i, w_i).
            - total_codebook_loss: The sum of commitment and codebook losses for this band.
            - v_i_quantized_upsampled: Quantized feature map upsampled to (H', W') for residual calculation (B, C, H', W').
        """
        if input_to_quantize.ndim != 4:
            raise ValueError(
                f"Input to Quantizer must be 4D (B, C, H, W), but got {input_to_quantize.ndim}D."
            )

        B, C_in, H_prime, W_prime = input_to_quantize.shape
        target_h, target_w = self.token_dims[band_idx]

        # 1. Downsample `input_to_quantize` to `v_i_continuous` at target resolution
        v_i_continuous = interpolate_feature_map(
            input_to_quantize, (target_h, target_w), mode="bilinear", align_corners=False
        )  # (B, C, h_i, w_i)

        # 2. Vector Quantization
        # Reshape to (Batch * h_i * w_i, C) for distance computation
        v_i_continuous_flat = rearrange(v_i_continuous, "b c h w -> (b h w) c")

        # Compute distances to codebook embeddings
        # (N, 1, C) - (1, K, C) -> (N, K, C) -> sum over C -> (N, K)
        distances = torch.sum(
            (v_i_continuous_flat.unsqueeze(1) - self.codebook.weight) ** 2, dim=2
        )

        # Find the closest codebook entry for each vector
        indices = torch.argmin(distances, dim=1)  # (N)

        # Lookup the quantized embeddings
        v_i_quantized_flat = self.codebook.weight[indices]  # (N, C)

        # Apply Straight-Through Estimator (STE)
        v_i_quantized_downsampled = v_i_continuous_flat + (
            v_i_quantized_flat - v_i_continuous_flat
        ).detach()
        # Reshape back to (B, C, h_i, w_i)
        v_i_quantized_downsampled = rearrange(
            v_i_quantized_downsampled, "(b h w) c -> b c h w", b=B, h=target_h, w=target_w
        )

        # Reshape indices back to (B, h_i, w_i)
        indices = indices.view(B, target_h, target_w)

        # 3. Compute Codebook Loss
        # Commitment loss: encourages the encoder output to be close to the chosen codebook entry
        commitment_loss = torch.mean(
            (v_i_quantized_downsampled.detach() - v_i_continuous) ** 2
        )
        # Codebook loss: encourages codebook entries to move towards the encoder output
        codebook_loss = torch.mean((v_i_quantized_downsampled - v_i_continuous.detach()) ** 2)
        total_codebook_loss = commitment_loss + self.codebook_loss_beta * codebook_loss

        # 4. Upsample `v_i_quantized_downsampled` to `H_prime, W_prime` for residual calculation
        v_i_quantized_upsampled = interpolate_feature_map(
            v_i_quantized_downsampled,
            (H_prime, W_prime),
            mode="bilinear",
            align_corners=False,
        )  # (B, C, H', W')

        return (
            v_i_quantized_downsampled,
            indices,
            total_codebook_loss,
            v_i_quantized_upsampled,
        )


class _FrequencyComposer(nn.Module):
    """
    Reconstructs the full latent feature map by summing upsampled quantized frequency components.
    """

    def __init__(self, H_prime: int, W_prime: int):
        """
        Initializes the frequency composer.

        Args:
            H_prime: Target height of the composed feature map.
            W_prime: Target width of the composed feature map.
        """
        super().__init__()
        self.H_prime = H_prime
        self.W_prime = W_prime

    def forward(self, v_q_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Composes the list of quantized feature maps into a single latent feature map.

        Args:
            v_q_list: A list of quantized feature maps. Each tensor in the list is
                      expected to be (B, C, h_i, w_i) for a specific frequency band.

        Returns:
            The composed latent feature map (B, C, H_prime, W_prime).
        """
        if not v_q_list:
            raise ValueError("Input v_q_list cannot be empty.")
        if any(v_q.ndim != 4 for v_q in v_q_list):
            raise ValueError("All tensors in v_q_list must be 4D (B, C, h, w).")

        # Get batch size and channel dimension from the first element
        B, C = v_q_list[0].shape[0], v_q_list[0].shape[1]

        # Initialize the composed feature map with zeros
        tilde_f = torch.zeros(B, C, self.H_prime, self.W_prime, device=v_q_list[0].device)

        # Upsample and sum each quantized component
        for v_q in v_q_list:
            upsampled_v_q = interpolate_feature_map(
                v_q, (self.H_prime, self.W_prime), mode="bilinear", align_corners=False
            )
            tilde_f = tilde_f + upsampled_v_q

        return tilde_f


class FRVAE(nn.Module):
    """
    Frequency-guided Residual-quantized VAE (FR-VAE).
    This is the core image tokenizer for the NFIG framework.
    """

    def __init__(
        self, config: Config, DINOv2_base_pretrained_path: Optional[str] = None
    ):
        """
        Initializes the FR-VAE model.

        Args:
            config: Configuration object.
            DINOv2_base_pretrained_path: Path to DINOv2 pretrained weights (currently a placeholder).
        """
        super().__init__()
        self.config = config

        # --- Configuration extraction ---
        self.image_size = config.data.image_size
        self.latent_dim_channels = config.fr_vae.latent_dim_channels
        self.encoder_latent_size = config.fr_vae.encoder_latent_size
        self.codebook_size = config.fr_vae.codebook_size
        self.n_bands = config.fr_vae.freq_bands.num_bands
        self.codebook_loss_beta = config.fr_vae_training.codebook_loss_beta

        # Validate total_quantized_tokens and derive token_dims_list
        # Based on paper's description for 10 bands and 680 tokens
        # Sum of (1*1 + 2*2 + 3*3 + 4*4 + 5*5 + 6*6 + 8*8 + 10*10 + 13*13 + 16*16) = 680
        # This derivation is based on the problem description's hint and matches 'total_quantized_tokens: 680'
        # The 'scaling_factors' in config.yaml are indicative of resolution increases,
        # and directly correspond to these (h,w) pairs when interpreted as (scale*1, scale*1).
        # Note: The last band (16x16) should match the full latent size (encoder_latent_size).
        self.token_dims_list: List[Tuple[int, int]] = [
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
            (6, 6),
            (8, 8),
            (10, 10),
            (13, 13),
            (16, 16),
        ]

        actual_total_tokens = sum(h * w for h, w in self.token_dims_list)
        if actual_total_tokens != config.fr_vae.freq_bands.total_quantized_tokens:
            raise ValueError(
                f"Derived total tokens {actual_total_tokens} does not match config's "
                f"total_quantized_tokens {config.fr_vae.freq_bands.total_quantized_tokens}. "
                "Please verify token_dims_list derivation logic."
            )
        if (
            self.token_dims_list[-1][0] != self.encoder_latent_size
            or self.token_dims_list[-1][1] != self.encoder_latent_size
        ):
            raise ValueError(
                f"The last token dimension {self.token_dims_list[-1]} must match "
                f"encoder_latent_size {self.encoder_latent_size} for the highest frequency band."
            )

        # --- Sub-module instantiation ---
        base_channels = 128  # Common for VQGAN-style models
        ch_mults_encoder = [1, 2, 4, 8]  # Example multipliers, tuned for 256->16 downsampling
        ch_mults_decoder = [8, 4, 2, 1]  # Reversed for decoder

        self.encoder = _Encoder(
            in_channels=3,
            out_channels=self.latent_dim_channels,
            image_size=self.image_size,
            latent_size=self.encoder_latent_size,
            num_res_blocks=2,  # Default from plan
            ch_mults=ch_mults_encoder,
            base_channels=base_channels,
            dino_v2_pretrained_path=DINOv2_base_pretrained_path,
        )

        self.decoder = _Decoder(
            in_channels=self.latent_dim_channels,
            out_channels=3,
            image_size=self.image_size,
            latent_size=self.encoder_latent_size,
            num_res_blocks=2,  # Default from plan
            ch_mults=ch_mults_decoder,
            base_channels=base_channels,
        )

        # Decomposer needs to know the device to create masks
        # We can create a dummy tensor to infer the current device during __init__
        # or pass it explicitly if known. Let's pass the default device config.
        # However, masks should be generated on the device where actual computations happen.
        # A better approach is to instantiate masks when the model is on device, or move them.
        # For simplicity, during init, I'll pass a generic CPU device, and then ensure they are
        # moved to the correct device when `apply_masks` is called.
        initial_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.decomposer = _FrequencyDecomposer(
            H_prime=self.encoder_latent_size,
            W_prime=self.encoder_latent_size,
            n_bands=self.n_bands,
            token_dims=self.token_dims_list,
            device=initial_device, # Masks will be moved to actual device in forward
        )

        self.quantizer = _FrequencyQuantizer(
            codebook_size=self.codebook_size,
            embedding_dim=self.latent_dim_channels,
            n_bands=self.n_bands,
            token_dims=self.token_dims_list,
            H_prime=self.encoder_latent_size,
            W_prime=self.encoder_latent_size,
            codebook_loss_beta=self.codebook_loss_beta,
        )

        self.composer = _FrequencyComposer(
            H_prime=self.encoder_latent_size, W_prime=self.encoder_latent_size
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes an input image into a latent feature map.

        Args:
            x: Input image tensor (B, 3, H, W).

        Returns:
            Latent feature map (B, C, H', W').
        """
        return self.encoder(x)

    def decompose(self, f: torch.Tensor) -> List[torch.Tensor]:
        """
        Decomposes a latent feature map into frequency components.

        Args:
            f: Latent feature map (B, C, H', W').

        Returns:
            A list of `n_bands` frequency component tensors (B, C, H', W').
        """
        return self.decomposer.apply_masks(f)

    def quantize(
        self, f_components: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Performs residual quantization across all frequency components.

        Args:
            f_components: List of `n_bands` frequency component tensors (B, C, H', W').

        Returns:
            A tuple containing:
            - v_q_list: List of quantized feature maps for each band (B, C, h_i, w_i).
            - v_q_indices_list: List of codebook indices for each band (B, h_i, w_i).
            - commit_loss_list: List of codebook losses for each band.
        """
        v_q_list: List[torch.Tensor] = []
        v_q_indices_list: List[torch.Tensor] = []
        commit_loss_list: List[torch.Tensor] = []

        R_prev = torch.zeros_like(f_components[0])  # Initialize R_{0} with zeros
        for band_idx, hat_f_i in enumerate(f_components):
            # Move R_prev to the current device of hat_f_i if necessary
            R_prev = R_prev.to(hat_f_i.device)

            # (R_{i-1} + hat_f_i)
            input_to_quantize_for_this_band = R_prev + hat_f_i

            (
                v_q_downsampled,
                indices,
                codebook_loss,
                v_q_upsampled_for_residual,
            ) = self.quantizer.quantize_band(input_to_quantize_for_this_band, band_idx)

            v_q_list.append(v_q_downsampled)
            v_q_indices_list.append(indices)
            commit_loss_list.append(codebook_loss)

            # R_i = (R_{i-1} + hat_f_i) - Z(v_i, H', W')
            R_prev = input_to_quantize_for_this_band - v_q_upsampled_for_residual

        return v_q_list, v_q_indices_list, commit_loss_list

    def compose(self, v_q_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Composes the list of quantized feature maps back into a single latent feature map.

        Args:
            v_q_list: List of quantized feature maps (B, C, h_i, w_i).

        Returns:
            Composed latent feature map (B, C, H', W').
        """
        return self.composer(v_q_list)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes a latent feature map into an image.

        Args:
            z: Latent feature map (B, C, H', W').

        Returns:
            Reconstructed image (B, 3, H, W).
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Performs the full FR-VAE forward pass for training.

        Args:
            x: Input image tensor (B, 3, H, W).

        Returns:
            A dictionary containing:
            - 'hat_I': Reconstructed image (B, 3, H, W).
            - 'tilde_f': Composed latent feature map (B, C, H', W').
            - 'f': Original encoded latent feature map (B, C, H', W').
            - 'v_q_indices_list': List of codebook indices for each band (B, h_i, w_i).
            - 'commit_loss_list': List of codebook losses for each band.
        """
        # Encode image
        f = self.encode(x)  # (B, C, H', W')

        # Decompose into frequency components
        f_components = self.decomposer.apply_masks(f)  # List of (B, C, H', W')

        # Quantize frequency components residually
        v_q_list, v_q_indices_list, commit_loss_list = self.quantize(f_components)

        # Compose quantized feature maps
        tilde_f = self.compose(v_q_list)  # (B, C, H', W')

        # Decode to image
        hat_I = self.decode(tilde_f)  # (B, 3, H, W)

        return {
            "hat_I": hat_I,
            "tilde_f": tilde_f,
            "f": f,
            "v_q_indices_list": v_q_indices_list,
            "commit_loss_list": commit_loss_list,
        }

    @torch.no_grad()
    def get_tokens(self, image: torch.Tensor) -> torch.Tensor:
        """
        Tokenizes an image (or a batch of images) into a flattened sequence of codebook indices.
        Used for preparing data for the NFIG Transformer.

        Args:
            image: Input image tensor (B, 3, H, W).

        Returns:
            A tensor of flattened codebook indices (B, total_sequence_length).
        """
        if image.ndim != 4:
            raise ValueError(f"Input image must be 4D (B, C, H, W), but got {image.ndim}D.")

        # Ensure model is in eval mode
        self.eval()

        # Encode image
        f = self.encode(image)  # (B, C, H', W')

        # Decompose into frequency components
        f_components = self.decomposer.apply_masks(f)  # List of (B, C, H', W')

        all_token_indices: List[torch.Tensor] = []
        R_prev = torch.zeros_like(f_components[0])  # Initialize R_{0} with zeros

        for band_idx, hat_f_i in enumerate(f_components):
            R_prev = R_prev.to(hat_f_i.device)
            input_to_quantize_for_this_band = R_prev + hat_f_i

            (
                _,  # v_q_downsampled (not needed for tokens)
                indices,
                _,  # codebook_loss (not needed for tokens)
                v_q_upsampled_for_residual,
            ) = self.quantizer.quantize_band(input_to_quantize_for_this_band, band_idx)

            # Flatten indices and add to list
            all_token_indices.append(indices.view(indices.shape[0], -1))

            # R_i = (R_{i-1} + hat_f_i) - Z(v_i, H', W')
            R_prev = input_to_quantize_for_this_band - v_q_upsampled_for_residual

        # Concatenate all flattened token indices
        # Resulting shape: (B, total_sequence_length)
        return torch.cat(all_token_indices, dim=1)

    @torch.no_grad()
    def decode_from_tokens(self, token_indices_batch: torch.Tensor) -> torch.Tensor:
        """
        Decodes a batch of token sequences back into images.

        Args:
            token_indices_batch: A tensor of shape (B, total_sequence_length)
                                 containing codebook indices for each image.

        Returns:
            Reconstructed images (B, 3, H, W).
        """
        self.eval()

        B, total_sequence_length = token_indices_batch.shape
        if total_sequence_length != sum(h * w for h, w in self.token_dims_list):
            raise ValueError(
                f"Expected token sequence length {sum(h*w for h,w in self.token_dims_list)}, "
                f"but got {total_sequence_length}."
            )

        v_q_list: List[torch.Tensor] = []
        start_idx = 0
        for band_idx, (h_i, w_i) in enumerate(self.token_dims_list):
            num_tokens_in_band = h_i * w_i
            band_tokens_flat = token_indices_batch[:, start_idx : start_idx + num_tokens_in_band]
            band_tokens_reshaped = band_tokens_flat.view(B, h_i, w_i)  # (B, h_i, w_i)

            # Lookup codebook embeddings for these tokens
            # codebook.weight is (codebook_size, embedding_dim)
            v_q_downsampled = self.quantizer.codebook.weight[band_tokens_reshaped]  # (B, h_i, w_i, C)
            v_q_downsampled = rearrange(v_q_downsampled, 'b h w c -> b c h w') # (B, C, h_i, w_i)
            
            v_q_list.append(v_q_downsampled)
            start_idx += num_tokens_in_band

        # Compose quantized feature maps
        tilde_f = self.composer(v_q_list)  # (B, C, H', W')

        # Decode to image
        hat_I = self.decode(tilde_f)  # (B, 3, H, W)
        return hat_I

