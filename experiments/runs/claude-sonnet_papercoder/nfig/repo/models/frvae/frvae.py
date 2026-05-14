## models/frvae/frvae.py
"""Frequency-guided Residual-quantized VAE (FR-VAE) for the NFIG framework.

Coordinates the full FR-VAE pipeline: encoder → frequency decomposition →
residual quantization → composition → decoder. Exposes three primary interfaces:

  - forward(x): Full training-time pass returning (x_hat, f, f_tilde).
  - get_tokens(x): Tokenization for NFIGTrainer (frozen model, no grad).
  - tokens_to_image(token_indices): Decoding for NFIGSampler.

Paper references:
  - Section 3.1: FR-VAE architecture overview
  - Section 3.1.1: Frequency-guided Decomposer/Composer
  - Section 3.1.2: Frequency-guided Residual Quantization
  - Appendix B.1: Loss function details

Config values used (config.yaml frvae section):
  encoder_model:        "vit_base_patch14_dinov2"
  pretrained_encoder:   true
  image_size:           256
  latent_spatial_size:  16
  latent_channels:      768
  codebook_size:        4096
  codebook_dim:         768
  scale_factors:        [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
  total_tokens:         680
  num_frequency_bands:  10
"""

from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from models.frvae.decoder import VQGANDecoder
from models.frvae.encoder import VQGANEncoder
from models.frvae.frequency_decomposer import FrequencyDecomposer
from models.frvae.residual_quantizer import ResidualQuantizer
from utils.config import FRVAEConfig


class FRVAE(nn.Module):
    """Frequency-guided Residual-quantized VAE.

    Integrates the encoder, frequency decomposer, residual quantizer, and
    decoder into a unified model. The discriminator (DINODiscriminator) is
    intentionally excluded — it is instantiated separately in FRVAETrainer
    to keep this class focused on the generative model and to allow clean
    inference without adversarial components.

    Data flow (training):
        x [B,3,256,256]
          → encoder → f [B,768,16,16]
          → decomposer.decompose → [f_hat_0..f_hat_9], each [B,768,16,16]
          → quantizer.encode_all → token_indices, quantized_upsampled
          → decomposer.compose(quantized_upsampled) → f_tilde [B,768,16,16]
          → decoder → x_hat [B,3,256,256]

    Data flow (tokenization, frozen):
        x → encoder → decomposer.decompose → quantizer.encode_all
          → token_indices [10 × [B, h_i, w_i]]

    Data flow (decoding, inference):
        token_indices → quantizer.decode_all → f_tilde → decoder → x_hat

    Attributes:
        encoder: DINOv2-based spatial feature extractor (VQGANEncoder).
        decomposer: FFT-based frequency band splitter (FrequencyDecomposer).
        quantizer: Progressive residual vector quantizer (ResidualQuantizer).
        decoder: Convolutional image reconstructor (VQGANDecoder).
        config: Stored FRVAEConfig for downstream access by trainers.
    """

    def __init__(self, config: FRVAEConfig) -> None:
        """Initialize the FR-VAE by wiring together all submodules.

        Submodule initialization order:
          1. encoder  — no intra-FRVAE dependencies
          2. decomposer — depends only on scale_factors and spatial size
          3. quantizer  — depends on scale_factors, spatial size, codebook params
          4. decoder    — depends only on latent_channels and image_size

        Args:
            config: FRVAEConfig dataclass populated from config.yaml frvae section.
                Key values:
                  - encoder_model:       "vit_base_patch14_dinov2"
                  - pretrained_encoder:  True
                  - image_size:          256
                  - latent_spatial_size: 16  (H' = W')
                  - latent_channels:     768 (C)
                  - codebook_size:       4096 (K)
                  - codebook_dim:        768  (must equal latent_channels)
                  - scale_factors:       [1,2,3,4,5,6,8,10,13,16]
                  - commitment_loss_weight: 0.25

        Raises:
            ValueError: If config.codebook_dim != config.latent_channels.
                The encoder output channel dimension must match the codebook
                vector dimension for nearest-neighbor lookup to be meaningful.
            ValueError: If config.scale_factors[-1] != config.latent_spatial_size.
                The highest-frequency band must operate at full latent resolution.
        """
        super().__init__()

        # --- Validate critical alignment constraints ---
        if config.codebook_dim != config.latent_channels:
            raise ValueError(
                f"config.codebook_dim={config.codebook_dim} must equal "
                f"config.latent_channels={config.latent_channels}. "
                "The encoder output channel dimension must match the codebook "
                "vector dimension for nearest-neighbor lookup to be meaningful."
            )

        if config.scale_factors[-1] != config.latent_spatial_size:
            raise ValueError(
                f"config.scale_factors[-1]={config.scale_factors[-1]} must equal "
                f"config.latent_spatial_size={config.latent_spatial_size}. "
                "The highest-frequency band must operate at full latent resolution."
            )

        # --- Store config for downstream access ---
        self.config: FRVAEConfig = config

        # --- 1. Encoder: image → spatial latent feature map ---
        # Maps x [B, 3, 256, 256] → f [B, 768, 16, 16].
        # DINOv2-base backbone initialized with pretrained weights.
        # Internal resize to 224×224 ensures exactly 16×16 patch tokens.
        self.encoder: VQGANEncoder = VQGANEncoder(
            model_name=config.encoder_model,
            latent_channels=config.latent_channels,
            pretrained=config.pretrained_encoder,
        )

        # --- 2. Frequency Decomposer: f → {f_hat_i} ---
        # Decomposes the latent feature map into n=10 frequency-band components
        # via 2D FFT masking. Masks are built lazily on first use (device-aware).
        # Frequency band boundaries σ_i are proportional to token counts at each scale.
        self.decomposer: FrequencyDecomposer = FrequencyDecomposer(
            scale_factors=config.scale_factors,
            H_prime=config.latent_spatial_size,
            W_prime=config.latent_spatial_size,
        )

        # --- 3. Residual Quantizer: {f_hat_i} → token_indices + quantized ---
        # Progressive residual quantization with a shared codebook Z ∈ R^(K×C).
        # Each frequency band captures what previous bands failed to represent.
        # Total tokens: Σ s_i^2 = 1+4+9+16+25+36+64+100+169+256 = 680.
        self.quantizer: ResidualQuantizer = ResidualQuantizer(
            codebook_size=config.codebook_size,
            codebook_dim=config.codebook_dim,
            scale_factors=config.scale_factors,
            H_prime=config.latent_spatial_size,
            W_prime=config.latent_spatial_size,
            commitment_loss_weight=config.commitment_loss_weight,
        )

        # --- 4. Decoder: f_tilde → reconstructed image ---
        # VQ-GAN style convolutional decoder.
        # Maps f_tilde [B, 768, 16, 16] → x_hat [B, 3, 256, 256].
        # 4 stages of 2× upsampling: 16→32→64→128→256.
        self.decoder: VQGANDecoder = VQGANDecoder(
            latent_channels=config.latent_channels,
            image_size=config.image_size,
        )

    def encode(
        self,
        x: Tensor,
    ) -> Tuple[List[Tensor], List[Tensor], Tensor]:
        """Encode an image batch into discrete token indices and quantized features.

        Implements the full encoding pipeline:
            x → encoder → f → decompose → {f_hat_i}
              → encode_all → token_indices, quantized_upsampled
              → compose → f_tilde

        This method is used by the trainer when it needs both the token indices
        (for transformer training targets) and the composed quantized feature map
        (for the frequency quantization loss and decoder input).

        Args:
            x: Input image batch of shape (B, 3, H, W) with values in [-1, 1].
               Typically H = W = 256 (config.frvae.image_size).

        Returns:
            Tuple of three elements:
                - token_indices_list: List of n=10 integer tensors.
                  token_indices_list[i] has shape (B, h_i, w_i) where
                  h_i = w_i = scale_factors[i]. Values in [0, K-1].
                  Shapes: (B,1,1), (B,2,2), ..., (B,16,16).

                - quantized_upsampled_list: List of n=10 float tensors.
                  quantized_upsampled_list[i] has shape (B, C, H', W').
                  These are T(v_i^q, H', W') — quantized representations
                  upsampled to full latent resolution.

                - f_tilde: Composed quantized feature map of shape (B, C, H', W').
                  f_tilde = Σ_i T(v_i^q, H', W') — the sum of all upsampled
                  quantized representations. Passed to the decoder.
        """
        # Step 1: Encode image to latent feature map.
        # f: [B, C, H', W'] = [B, 768, 16, 16]
        f: Tensor = self.encoder(x)

        # Step 2: Decompose into frequency-band components.
        # freq_components: list of n tensors, each [B, C, H', W']
        # freq_components[i] = f_hat_i = F^{-1}(F(f) ⊙ M_i)
        freq_components: List[Tensor] = self.decomposer.decompose(f)

        # Step 3: Progressive residual quantization.
        # token_indices_list[i]: [B, h_i, w_i] — discrete indices in [0, K-1]
        # quantized_upsampled_list[i]: [B, C, H', W'] — T(v_i^q, H', W')
        token_indices_list: List[Tensor]
        quantized_upsampled_list: List[Tensor]
        token_indices_list, quantized_upsampled_list = self.quantizer.encode_all(
            freq_components
        )

        # Step 4: Compose quantized representations into a single feature map.
        # f_tilde = Σ_i T(v_i^q, H', W'): [B, C, H', W']
        f_tilde: Tensor = self.decomposer.compose(quantized_upsampled_list)

        return token_indices_list, quantized_upsampled_list, f_tilde

    def decode(self, token_indices: List[Tensor]) -> Tensor:
        """Decode a list of token index tensors into a reconstructed image.

        Converts discrete token indices back to continuous feature vectors via
        codebook lookup, composes them into a latent feature map, and decodes
        to pixel space.

        Args:
            token_indices: List of n=10 integer tensors.
                token_indices[i] has shape (B, h_i, w_i) where
                h_i = w_i = scale_factors[i]. Values in [0, K-1].

        Returns:
            Reconstructed image batch x_hat of shape (B, 3, H, W).
            Values are in [-1, 1] (tanh output from decoder).
        """
        # Step 1: Reconstruct composed feature map from token indices.
        # f_tilde = Σ_i T(lookup(Z, token_indices_i), H', W')
        # Shape: [B, C, H', W'] = [B, 768, 16, 16]
        f_tilde: Tensor = self.quantizer.decode_all(token_indices)

        # Step 2: Decode to image space.
        # x_hat: [B, 3, H, W] = [B, 3, 256, 256], values in [-1, 1]
        x_hat: Tensor = self.decoder(f_tilde)

        return x_hat

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Full training-time forward pass through the FR-VAE.

        Implements the complete encode-quantize-decode pipeline and returns
        the tensors needed by FRVAETrainer to compute all loss components:

        Loss components (paper Appendix B.1):
            L = ||I - I_hat||^2           ← reconstruction loss: (x, x_hat)
              + ||f_hat - f_hat_rec||^2   ← freq quantization loss: (f, f_tilde)
              + L_p(I)                    ← LPIPS perceptual loss: (x, x_hat)
              + 0.5 * L_g(I)             ← GAN loss: discriminator(x_hat)

        The codebook/commitment losses are computed separately by the trainer
        via self.model.quantizer.get_codebook_loss() using the (v_i, v_i_q)
        pairs stored in self.quantizer._last_v_pairs after encode_all().

        Args:
            x: Input image batch of shape (B, 3, H, W) with values in [-1, 1].
               Typically H = W = 256 (config.frvae.image_size = 256).

        Returns:
            Tuple of three tensors:
                - x_hat: Reconstructed image of shape (B, 3, H, W), values in [-1, 1].
                  Used for: reconstruction loss, LPIPS loss, GAN generator loss.

                - f: Raw encoder output of shape (B, C, H', W') = (B, 768, 16, 16).
                  Used for: frequency quantization loss ||f - f_tilde||^2.
                  This is the original latent feature map before quantization.

                - f_tilde: Composed quantized feature map of shape (B, C, H', W').
                  f_tilde = Σ_i T(v_i^q, H', W').
                  Used for: frequency quantization loss ||f - f_tilde||^2.
                  This is the quantized approximation of f.
        """
        # Step 1: Encode image to latent feature map.
        # f: [B, C, H', W'] = [B, 768, 16, 16]
        # Stored separately so it can be returned for the frequency quantization loss.
        f: Tensor = self.encoder(x)

        # Step 2: Decompose into frequency-band components.
        # freq_components[i] = f_hat_i: [B, C, H', W']
        freq_components: List[Tensor] = self.decomposer.decompose(f)

        # Step 3: Progressive residual quantization.
        # token_indices_list[i]: [B, h_i, w_i] — discrete indices (not returned)
        # quantized_upsampled_list[i]: [B, C, H', W'] — T(v_i^q, H', W')
        # Side effect: populates self.quantizer._last_v_pairs for loss computation.
        _token_indices_list: List[Tensor]
        quantized_upsampled_list: List[Tensor]
        _token_indices_list, quantized_upsampled_list = self.quantizer.encode_all(
            freq_components
        )

        # Step 4: Compose quantized representations.
        # f_tilde = Σ_i T(v_i^q, H', W'): [B, C, H', W']
        f_tilde: Tensor = self.decomposer.compose(quantized_upsampled_list)

        # Step 5: Decode to image space.
        # x_hat: [B, 3, H, W] = [B, 3, 256, 256], values in [-1, 1]
        x_hat: Tensor = self.decoder(f_tilde)

        return x_hat, f, f_tilde

    @torch.no_grad()
    def get_tokens(self, x: Tensor) -> List[Tensor]:
        """Extract discrete token indices from an image batch (no gradient).

        Called by NFIGTrainer._tokenize_batch() with the FR-VAE frozen.
        Returns only the token indices needed for transformer training targets.
        The quantized feature maps and composed f_tilde are discarded.

        This method always runs under torch.no_grad() (enforced via decorator)
        to prevent accidental gradient accumulation when the FR-VAE is used
        as a frozen tokenizer during NFIG Transformer training.

        The caller (NFIGTrainer) is responsible for:
          - Calling self.frvae.eval() before tokenization
          - Calling self.frvae.requires_grad_(False) to freeze all parameters

        Args:
            x: Input image batch of shape (B, 3, H, W) with values in [-1, 1].
               Typically H = W = 256 (config.frvae.image_size = 256).

        Returns:
            List of n=10 integer tensors (token indices).
            token_indices_list[i] has shape (B, h_i, w_i) where
            h_i = w_i = scale_factors[i].
            Shapes: (B,1,1), (B,2,2), (B,3,3), (B,4,4), (B,5,5),
                    (B,6,6), (B,8,8), (B,10,10), (B,13,13), (B,16,16).
            Values are integers in [0, K-1] = [0, 4095].
        """
        # Step 1: Encode image to latent feature map.
        f: Tensor = self.encoder(x)

        # Step 2: Decompose into frequency-band components.
        freq_components: List[Tensor] = self.decomposer.decompose(f)

        # Step 3: Quantize and extract token indices.
        # Discard quantized_upsampled_list — not needed for tokenization.
        token_indices_list: List[Tensor]
        token_indices_list, _ = self.quantizer.encode_all(freq_components)

        return token_indices_list

    @torch.no_grad()
    def tokens_to_image(self, token_indices: List[Tensor]) -> Tensor:
        """Decode a list of token index tensors into generated images (no gradient).

        Called by NFIGSampler after all 10 frequency bands have been
        autoregressively generated. Converts discrete token indices back to
        pixel-space images via codebook lookup and the VQ-GAN decoder.

        This method always runs under torch.no_grad() (enforced via decorator)
        since it is used exclusively during inference/sampling.

        Args:
            token_indices: List of n=10 integer tensors from the autoregressive
                sampler. token_indices[i] has shape (B, h_i, w_i) where
                h_i = w_i = scale_factors[i]. Values in [0, K-1].
                Shapes: (B,1,1), (B,2,2), ..., (B,16,16).

        Returns:
            Generated image batch x_hat of shape (B, 3, H, W).
            Values are in [-1, 1] (tanh output from VQGANDecoder).
            H = W = config.frvae.image_size = 256.
        """
        # Step 1: Reconstruct composed feature map from token indices.
        # f_tilde = Σ_i T(lookup(Z, token_indices_i), H', W')
        # Shape: [B, C, H', W'] = [B, 768, 16, 16]
        f_tilde: Tensor = self.quantizer.decode_all(token_indices)

        # Step 2: Decode to image space.
        # x_hat: [B, 3, H, W] = [B, 3, 256, 256], values in [-1, 1]
        x_hat: Tensor = self.decoder(f_tilde)

        return x_hat

    def freeze(self) -> None:
        """Freeze all FR-VAE parameters (disable gradient computation).

        Called by NFIGTrainer before using the FR-VAE as a frozen tokenizer.
        After calling this method, no parameters will be updated during
        NFIG Transformer training.

        Also sets the model to eval mode to disable any dropout/batch norm
        training behavior.
        """
        self.requires_grad_(False)
        self.eval()

    def unfreeze(self) -> None:
        """Unfreeze all FR-VAE parameters (enable gradient computation).

        Called to resume FR-VAE fine-tuning after a frozen phase.
        Sets the model back to train mode.
        """
        self.requires_grad_(True)
        self.train()

    def get_last_codebook_loss(self) -> Tensor:
        """Compute the total codebook + commitment loss from the last forward pass.

        Aggregates the codebook and commitment losses across all n=10 frequency
        bands using the (v_i, v_i_q) pairs stored in self.quantizer._last_v_pairs
        after the most recent encode_all() call.

        This is a convenience method for FRVAETrainer to avoid directly accessing
        the quantizer's internal state. It should be called immediately after
        forward() to ensure _last_v_pairs is populated from the current batch.

        Loss formula (standard VQ-VAE, applied per frequency band):
            L_vq_i = ||sg(v_i) - v_i_q||^2 + beta * ||v_i - sg(v_i_q)||^2

        Total: L_vq = Σ_i L_vq_i

        Returns:
            Scalar tensor: sum of codebook + commitment losses across all bands.
            Requires grad (flows through the encoder via commitment loss).

        Raises:
            RuntimeError: If called before forward() (no _last_v_pairs available).
        """
        if not self.quantizer._last_v_pairs:
            raise RuntimeError(
                "get_last_codebook_loss() called before forward(). "
                "The quantizer's _last_v_pairs is empty. "
                "Call forward() or encode_all() first to populate it."
            )

        total_loss: Tensor = torch.tensor(
            0.0,
            device=next(self.parameters()).device,
            dtype=next(self.parameters()).dtype,
        )

        for v_i, v_i_q in self.quantizer._last_v_pairs:
            total_loss = total_loss + self.quantizer.get_codebook_loss(v_i, v_i_q)

        return total_loss

    def get_num_tokens(self) -> int:
        """Return the total number of tokens in the FR-VAE token sequence.

        Computed as Σ_i scale_factors[i]^2 = 680 for the default config.
        This matches config.frvae.total_tokens = 680 from config.yaml.

        Returns:
            Total token count (680 for the default scale_factors).
        """
        return sum(s * s for s in self.config.scale_factors)

    def get_token_shapes(self) -> List[Tuple[int, int]]:
        """Return the spatial shape of the token grid for each frequency band.

        Returns:
            List of n=10 (h_i, w_i) tuples where h_i = w_i = scale_factors[i].
            For the default config:
                [(1,1), (2,2), (3,3), (4,4), (5,5),
                 (6,6), (8,8), (10,10), (13,13), (16,16)]
        """
        return [(s, s) for s in self.config.scale_factors]

    def extra_repr(self) -> str:
        """Return a human-readable string with key FR-VAE configuration.

        Returns:
            String describing the FR-VAE's key dimensions and token counts.
        """
        total_tokens: int = self.get_num_tokens()
        return (
            f"latent_channels={self.config.latent_channels}, "
            f"latent_spatial_size={self.config.latent_spatial_size}×"
            f"{self.config.latent_spatial_size}, "
            f"codebook_size={self.config.codebook_size}, "
            f"num_frequency_bands={self.config.num_frequency_bands}, "
            f"scale_factors={self.config.scale_factors}, "
            f"total_tokens={total_tokens}"
        )
