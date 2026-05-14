"""
models/fr_vae.py

Frequency-guided Residual-quantized VAE (FR-VAE) for the NFIG reproduction.

Classes:
    FrequencyMaskGenerator – generates binary masks for frequency decomposition.
    ResidualQuantizer – per-level encoding, vector quantisation, and upsampling.
    FRVAE – full tokenizer combining DINOv2 encoder, frequency decomposer,
            residual quantizer, and CNN decoder.

All hyperparameters are read from a configuration dictionary compatible with
the project's config.yaml.  Missing details are filled with sensible defaults
based on XQGAN/VQ-GAN.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import VisionTransformer
from torchvision.models.vision_transformer import _load_weights  # noqa: F401
from torchvision.models.vision_transformer import interpolate_embeddings
from torchvision.models.vision_transformer import _DINO_V2_WEIGHTS


# ------------------------------------------------------------------
# 1. FrequencyMaskGenerator
# ------------------------------------------------------------------

class FrequencyMaskGenerator(nn.Module):
    """
    Pre‑computes binary masks for radial frequency band decomposition.

    Args:
        H_prime: spatial height of the feature map (e.g. 16).
        W_prime: spatial width of the feature map (e.g. 16).
        scale_sizes: list of target resolutions (h_i = w_i) for each band.
    """

    def __init__(
        self,
        H_prime: int,
        W_prime: int,
        scale_sizes: List[int],
    ) -> None:
        super().__init__()
        self.H_prime = H_prime
        self.W_prime = W_prime
        self.scale_sizes = scale_sizes
        self.n_bands = len(scale_sizes)

        # Pre‑compute masks and register as buffer (non‑trainable)
        masks = self._generate_masks()
        for i, mask in enumerate(masks):
            self.register_buffer(f"mask_{i}", mask, persistent=False)

    def _generate_masks(self) -> List[torch.Tensor]:
        """
        Build the list of binary frequency masks.

        Returns:
            A list of tensors, each of shape (1, 1, H_prime, W_prime).
        """
        H, W = self.H_prime, self.W_prime
        total_tokens = sum(s * s for s in self.scale_sizes)

        # Radial distances from zero‑frequency centre (after fftshift)
        ys = torch.arange(H, dtype=torch.float32) - H // 2
        xs = torch.arange(W, dtype=torch.float32) - W // 2
        Y, X = torch.meshgrid(ys, xs, indexing="ij")
        R = torch.sqrt(X * X + Y * Y)                     # (H, W)
        sigma_max = R.max().item()

        masks: List[torch.Tensor] = []
        sigma_prev = 0.0

        for i, s in enumerate(self.scale_sizes):
            tokens_this = s * s
            sigma_i = sigma_prev + (tokens_this / total_tokens) * sigma_max

            # Final band includes all remaining frequencies
            if i == self.n_bands - 1:
                mask = R >= sigma_prev
            else:
                mask = (R >= sigma_prev) & (R < sigma_i)

            # Shape (1, 1, H, W) for easy broadcasting over batch & channel
            mask = mask.to(torch.float32).unsqueeze(0).unsqueeze(0)
            masks.append(mask)

            sigma_prev = sigma_i

        return masks

    def get_masks(self) -> List[torch.Tensor]:
        """Return the pre‑computed frequency masks."""
        return [getattr(self, f"mask_{i}") for i in range(self.n_bands)]


# ------------------------------------------------------------------
# 2. ResidualQuantizer
# ------------------------------------------------------------------

class ResidualQuantizer(nn.Module):
    """
    Per‑level encoding, vector quantisation, and upsampling.

    Args:
        C: latent feature dimension (same as tokenizer.latent_dim).
        codebook_size: number of entries in the shared codebook.
        commitment_cost: β for VQ‑VAE commitment loss.
        scale_sizes: list of per‑level output spatial sizes (h_i, w_i).
        H_prime: original feature map height (e.g. 16).
        W_prime: original feature map width.
    """

    def __init__(
        self,
        C: int,
        codebook_size: int,
        commitment_cost: float,
        scale_sizes: List[int],
        H_prime: int,
        W_prime: int,
    ) -> None:
        super().__init__()
        self.C = C
        self.codebook_size = codebook_size
        self.commitment_cost = commitment_cost
        self.scale_sizes = scale_sizes
        self.H_prime = H_prime
        self.W_prime = W_prime

        # Shared learnable codebook
        self.codebook = nn.Parameter(torch.empty(codebook_size, C))
        nn.init.kaiming_uniform_(self.codebook)

        # Per‑level encoders: each maps (B, C, H_prime, W_prime) -> (B, C, h_i, w_i)
        self.level_encoders = nn.ModuleList()
        for h_i in scale_sizes:
            # Simple CNN with adaptive pooling to target resolution
            encoder = nn.Sequential(
                nn.Conv2d(C, C, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(C),
                nn.ReLU(inplace=True),
                nn.Conv2d(C, C, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(C),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((h_i, h_i)),
            )
            self.level_encoders.append(encoder)

    def forward(
        self,
        target: torch.Tensor,
        level_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode, quantise, and upsample for one frequency band.

        Args:
            target: feature map of shape (B, C, H_prime, W_prime).
            level_idx: index of the current band.

        Returns:
            u_i  : upsampled quantized feature, shape (B, C, H_prime, W_prime).
            token_indices : token index map, shape (B, h_i, w_i).
            v_i  : pre‑quantized continuous feature, (B, C, h_i, w_i).
            v_i_q: quantized continuous feature, (B, C, h_i, w_i).
        """
        B, C, Hp, Wp = target.shape
        encoder = self.level_encoders[level_idx]
        v_i = encoder(target)                             # (B, C, h_i, w_i)

        # ---- Vector quantisation ----
        B, C, h, w = v_i.shape
        v_flat = v_i.permute(0, 2, 3, 1).reshape(-1, C)  # (B*h*w, C)

        # Distances to codebook (||v - z||^2)
        sq_v = (v_flat ** 2).sum(dim=1, keepdim=True)    # (N, 1)
        sq_z = (self.codebook ** 2).sum(dim=1)           # (K,)
        distances = sq_v + sq_z - 2.0 * v_flat @ self.codebook.T  # (N, K)

        token_indices_flat = torch.argmin(distances, dim=1)        # (N,)
        z_q_flat = self.codebook[token_indices_flat]               # (N, C)

        # Straight‑through estimator
        v_i_q_flat = v_flat + (z_q_flat - v_flat).detach()
        v_i_q = v_i_q_flat.reshape(B, h, w, C).permute(0, 3, 1, 2)  # (B, C, h, w)

        # Token indices as spatial map
        tokens = token_indices_flat.reshape(B, h, w)               # (B, h, w)

        # ---- Upsample to original spatial size ----
        u_i = F.interpolate(
            v_i_q,
            size=(Hp, Wp),
            mode="bilinear",
            align_corners=False,
        )                                                   # (B, C, Hp, Wp)

        return u_i, tokens, v_i, v_i_q


# ------------------------------------------------------------------
# 3. FRVAE – the full tokenizer
# ------------------------------------------------------------------

class FRVAE(nn.Module):
    """
    Frequency-guided Residual-quantized VAE.

    Args:
        config: configuration dictionary containing at minimum the keys:
            data.image_size, tokenizer.latent_dim, tokenizer.spatial_dim,
            tokenizer.scale_sizes, tokenizer.codebook_size,
            tokenizer.commitment_cost.
    """

    def __init__(self, config: Dict) -> None:
        super().__init__()
        # Unpack configuration values with defaults
        self.image_size = config.get("data", {}).get("image_size", 256)
        self.latent_dim = config["tokenizer"].get("latent_dim", 256)
        self.H_prime = config["tokenizer"].get("spatial_dim", 16)
        self.W_prime = self.H_prime
        self.scale_sizes = config["tokenizer"].get(
            "scale_sizes",
            [1, 2, 3, 4, 5, 6, 8, 10, 13, 16],
        )
        self.codebook_size = config["tokenizer"].get("codebook_size", 4096)
        self.commitment_cost = config["tokenizer"].get("commitment_cost", 0.25)

        # ---- Encoder build ----
        self.encoder = self._build_dino_encoder()

        # ---- Frequency mask generator ----
        self.mask_generator = FrequencyMaskGenerator(
            H_prime=self.H_prime,
            W_prime=self.W_prime,
            scale_sizes=self.scale_sizes,
        )

        # ---- Residual quantizer ----
        self.quantizer = ResidualQuantizer(
            C=self.latent_dim,
            codebook_size=self.codebook_size,
            commitment_cost=self.commitment_cost,
            scale_sizes=self.scale_sizes,
            H_prime=self.H_prime,
            W_prime=self.W_prime,
        )

        # ---- Decoder ----
        self.decoder = self._build_decoder()

    def _build_dino_encoder(self) -> nn.Module:
        """
        Build a DINOv2‑base encoder with patch size 16 and output feature map
        of shape (latent_dim, H', W').

        The encoder consists of:
            - A ViT backbone (12 layers, 768‑dim) initialised from DINOv2‑base (patch14).
            - A 1×1 convolution to reduce the channel dimension to latent_dim.
        """
        # Build ViT with pixel‑patch embedding of size 16 (matching the desired grid)
        vit = VisionTransformer(
            image_size=self.image_size,
            patch_size=16,               # ensure 16×16 grid for 256x256 input
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,               # 4 * 768
            num_classes=0,               # no classification head
        )

        # Load DINOv2‑base pretrained weights (patch14) and interpolate up to patch16
        try:
            from torchvision.models import vit_b_16
            state_dict = _DINO_V2_WEIGHTS["dinov2_vitb14"].get_state_dict(progress=True)
        except Exception:
            raise RuntimeError("Could not load DINOv2‑base weights; check internet or PATH.")

        # Interpolate patch embedding
        orig_patch_size = 14
        old_conv_weight = state_dict["conv_proj.weight"]   # (768, 3, 14, 14)
        # Rescale to 16x16 kernel
        state_dict["conv_proj.weight"] = F.interpolate(
            old_conv_weight, size=16, mode="bilinear", align_corners=False
        )
        # Adjust positional embeddings: original are for 14x14 grid (256 tokens)
        old_pos = state_dict["encoder.pos_embedding"]
        # Reshape to 2D: (1, N, 768) -> (1, 14, 14, 768) -> permute -> (1, 768, 14, 14)
        n, dim = old_pos.shape[1], old_pos.shape[2]
        h = w = int(math.sqrt(n))
        old_pos_2d = old_pos.reshape(1, h, w, dim).permute(0, 3, 1, 2).contiguous()
        new_pos_2d = F.interpolate(
            old_pos_2d, size=(16, 16), mode="bilinear", align_corners=False
        )
        new_pos = new_pos_2d.permute(0, 2, 3, 1).reshape(1, 256, dim).contiguous()
        state_dict["encoder.pos_embedding"] = new_pos

        vit.load_state_dict(state_dict, strict=False)

        # Remove the classification token and the classification head (already num_classes=0)
        vit.heads = nn.Identity()

        # Add a 1×1 projection to reduce channel from 768 to latent_dim
        proj = nn.Conv2d(768, self.latent_dim, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(proj.weight, mode="fan_out", nonlinearity="relu")

        encoder = nn.Sequential(vit, proj)
        return encoder

    def _build_decoder(self) -> nn.Module:
        """
        Build a CNN decoder that upscales from (latent_dim, 16, 16) to (3, 256, 256).
        Follows a typical VQ‑GAN decoder structure.
        """
        decoder = nn.Sequential(
            # 16 -> 32
            nn.ConvTranspose2d(self.latent_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 32 -> 64
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 64 -> 128
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 128 -> 256
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 256 -> 256 (output layer)
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),   # output in [-1, 1] matching input normalisation
        )
        return decoder

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode an RGB image to the latent feature map f.

        Args:
            image: (B, 3, 256, 256) normalised in [-1, 1].

        Returns:
            f: feature map of shape (B, latent_dim, 16, 16).
        """
        # The encoder is a nn.Sequential: ViT + Conv2d.
        # The ViT outputs a sequence of patches; we need to reshape to 2D.
        # For this ViT (no cls token, patch_size=16), output is (B, 256, 768) because
        # num_patches = 16*16 = 256.
        vit_features = self.encoder[0](image)                # (B, 256, 768)
        # Reshape to (B, 768, 16, 16)
        vit_features = vit_features.permute(0, 2, 1).reshape(
            -1, 768, self.H_prime, self.W_prime
        )
        # 1×1 projection
        f = self.encoder[1](vit_features)                     # (B, latent_dim, 16, 16)
        return f

    def decode(self, f_tilde: torch.Tensor) -> torch.Tensor:
        """
        Decode the quantized composite feature map back to an image.

        Args:
            f_tilde: (B, latent_dim, H', W').

        Returns:
            reconnected image, shape (B, 3, 256, 256), in [-1, 1].
        """
        return self.decoder(f_tilde)

    def quantize(
        self, v_i: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vector quantisation helper that operates on a single feature map.

        Args:
            v_i: continuous feature map (B, C, h, w).

        Returns:
            v_i_q: quantized feature map (B, C, h, w).
            tokens: token index map (B, h, w).
        """
        B, C, h, w = v_i.shape
        v_flat = v_i.permute(0, 2, 3, 1).reshape(-1, C)

        sq_v = (v_flat ** 2).sum(dim=1, keepdim=True)
        sq_z = (self.quantizer.codebook ** 2).sum(dim=1)
        distances = sq_v + sq_z - 2.0 * v_flat @ self.quantizer.codebook.T

        token_indices = torch.argmin(distances, dim=1)
        z_q = self.quantizer.codebook[token_indices]

        v_i_q_flat = v_flat + (z_q - v_flat).detach()
        v_i_q = v_i_q_flat.reshape(B, h, w, C).permute(0, 3, 1, 2)
        tokens = token_indices.reshape(B, h, w)

        return v_i_q, tokens

    def frequency_decompose(self, f: torch.Tensor) -> List[torch.Tensor]:
        """
        Decompose the latent feature map into band‑limited components.

        Args:
            f: (B, C, H', W').

        Returns:
            List of tensors, each of shape (B, C, H', W'), one per frequency band.
        """
        # FFT to frequency domain
        F = torch.fft.fft2(f)
        F_shifted = torch.fft.fftshift(F, dim=(-2, -1))

        components = []
        for mask in self.mask_generator.get_masks():
            mask = mask.to(device=f.device, dtype=torch.float32)
            # mask shape: (1, 1, H', W')
            F_i = F_shifted * mask
            F_i_unshifted = torch.fft.ifftshift(F_i, dim=(-2, -1))
            f_i = torch.fft.ifft2(F_i_unshifted)
            components.append(f_i.real)

        return components

    def get_masks(self) -> List[torch.Tensor]:
        """Expose the frequency masks."""
        return self.mask_generator.get_masks()

    def forward(
        self, image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full reconstruction pass.

        Args:
            image: (B, 3, 256, 256) in [-1, 1].

        Returns:
            recon_image: reconstructed image (B, 3, 256, 256).
            f_tilde: composite quantized feature map (B, C, H', W').
            f: original encoder feature map (B, C, H', W').
        """
        # 1. Encode
        f = self.encode(image)                               # (B, C, H', W')

        # 2. Frequency decomposition
        f_hat_list = self.frequency_decompose(f)             # list of length n

        # 3. Residual quantisation loop
        f_tilde = torch.zeros_like(f)
        R_prev = torch.zeros_like(f)

        for i, f_hat_i in enumerate(f_hat_list):
            if i == 0:
                target = f_hat_i
            else:
                target = R_prev + f_hat_i

            u_i, _, _, _ = self.quantizer(target, level_idx=i)
            f_tilde = f_tilde + u_i

            # Update residual: R_i = R_{i-1} + (f_hat_i - u_i)
            R_prev = R_prev + (f_hat_i - u_i)

        # 4. Decode
        recon_image = self.decode(f_tilde)
        return recon_image, f_tilde, f

    def tokenize(self, image: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract discrete token indices for all frequency bands.

        Args:
            image: (B, 3, 256, 256) in [-1, 1].

        Returns:
            A list of tensors, each of shape (B, h_i * w_i) containing flattened
            token indices for the corresponding frequency band.
        """
        with torch.no_grad():
            f = self.encode(image)
            f_hat_list = self.frequency_decompose(f)

            R_prev = torch.zeros_like(f)
            all_tokens = []          # list of per‑batch token maps

            for i, f_hat_i in enumerate(f_hat_list):
                if i == 0:
                    target = f_hat_i
                else:
                    target = R_prev + f_hat_i

                # Pass through the per‑level encoder, then quantise
                encoder = self.quantizer.level_encoders[i]
                v_i = encoder(target)                      # (B, C, h_i, w_i)
                _, tokens = self.quantize(v_i)             # (B, h_i, w_i)

                # Compute upsampled version for residual update (without gradient)
                _, _, _, v_i_q = self.quantizer(target, level_idx=i)
                u_i = F.interpolate(
                    v_i_q,
                    size=(self.H_prime, self.W_prime),
                    mode="bilinear",
                    align_corners=False,
                )

                R_prev = R_prev + (f_hat_i - u_i)

                # Flatten token map to (B, h_i*w_i)
                all_tokens.append(tokens.reshape(image.size(0), -1))

        return all_tokens

