from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.frequency import FrequencyComposer, FrequencyDecomposer, upsample_feature
from modules.layers import Decoder, Encoder
from models.quantizer import FrequencyResidualQuantizer


class FRVAE(nn.Module):
    """
    Frequency-guided Residual-quantized VAE (FR-VAE).

    Architecture (Section 3.1):
    - Encoder E: image -> feature map f ∈ R^{H'×W'×C}
    - FrequencyDecomposer: f -> {f_hat_i} via FFT masking
    - FrequencyResidualQuantizer: {f_hat_i} -> {v_q_i}, {indices_i}
    - FrequencyComposer: {v_q_i_upsampled} -> f_tilde
    - Decoder D: f_tilde -> reconstructed image

    The encoder is initialized with DINOv2-base pretrained weights.
    """

    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 3,
        z_channels: int = 256,
        ch: int = 128,
        ch_mult: Tuple[int, ...] = (1, 1, 2, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions: Tuple[int, ...] = (16,),
        dropout: float = 0.0,
        codebook_size: int = 4096,
        scale_factors: List[int] = None,
        feature_map_size: int = 16,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        if scale_factors is None:
            scale_factors = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]

        self.image_size = image_size
        self.feature_map_size = feature_map_size
        self.scale_factors = scale_factors
        self.z_channels = z_channels

        self.encoder = Encoder(
            in_channels=in_channels,
            z_channels=z_channels,
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
        )

        self.decomposer = FrequencyDecomposer(
            h=feature_map_size,
            w=feature_map_size,
            scale_factors=scale_factors,
        )

        self.quantizer = FrequencyResidualQuantizer(
            codebook_size=codebook_size,
            codebook_dim=z_channels,
            scale_factors=scale_factors,
            feature_map_size=feature_map_size,
            commitment_cost=commitment_cost,
        )

        self.composer = FrequencyComposer(
            target_h=feature_map_size,
            target_w=feature_map_size,
        )

        self.decoder = Decoder(
            out_channels=in_channels,
            z_channels=z_channels,
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
        )

    def encode(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """
        Encode image to frequency token indices.

        Args:
            x: (B, 3, H, W) input image

        Returns:
            indices_list: list of n token index tensors (B, h_i*w_i)
            v_q_list: list of quantized feature maps at each scale
            vq_loss: scalar VQ loss
        """
        f = self.encoder(x)  # (B, C, H', W')
        components = self.decomposer(f)  # list of (B, C, H', W')
        v_q_list, indices_list, residuals, vq_loss = self.quantizer(components)
        return indices_list, v_q_list, vq_loss

    def decode_from_indices(self, indices_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Decode from token indices to image.

        Args:
            indices_list: list of (B, h_i*w_i) index tensors

        Returns:
            (B, 3, H, W) reconstructed image
        """
        f_tilde = self.quantizer.decode_indices(indices_list)
        return self.decoder(f_tilde)

    def decode_from_quantized(self, v_q_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Decode from quantized feature maps to image.

        Args:
            v_q_list: list of (B, C, h_i, w_i) quantized feature maps

        Returns:
            (B, 3, H, W) reconstructed image
        """
        H = self.feature_map_size
        W = self.feature_map_size
        # Upsample and sum all quantized components
        upsampled = [upsample_feature(v, H, W) for v in v_q_list]
        f_tilde = sum(upsampled)
        return self.decoder(f_tilde)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, List[torch.Tensor]]:
        """
        Full forward pass for training.

        Args:
            x: (B, 3, H, W) input image

        Returns:
            x_rec: (B, 3, H, W) reconstructed image
            indices_list: list of token index tensors
            vq_loss: scalar VQ loss
            components: list of frequency components (for frequency loss)
        """
        f = self.encoder(x)
        components = self.decomposer(f)
        v_q_list, indices_list, residuals, vq_loss = self.quantizer(components)

        # Reconstruct feature map from quantized components
        H = self.feature_map_size
        W = self.feature_map_size
        upsampled = [upsample_feature(v, H, W) for v in v_q_list]
        f_tilde = sum(upsampled)

        x_rec = self.decoder(f_tilde)

        return x_rec, indices_list, vq_loss, components, f, f_tilde

    def get_last_layer_weight(self) -> torch.Tensor:
        return self.decoder.conv_out.weight

    @property
    def total_tokens(self) -> int:
        return sum(s * s for s in self.scale_factors)
