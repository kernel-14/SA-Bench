## models/memory_encoder.py
"""Memory encoder for SAM 2: converts frame predictions into compact memory features.

This module implements the MemoryEncoder that transforms the current frame's
predicted mask and unconditioned frame embedding into a spatial memory feature
stored in the MemoryBank for future frame conditioning.

Key design decisions from the paper (Section 4, Appendix D.1):
    - Reuses unconditioned Hiera+FPN frame embeddings (no separate encoder)
    - Downsamples predicted mask via convolutional module to stride-16 spatial size
    - Element-wise sums downsampled mask features with frame embedding
    - Applies light-weight convolutional fusion layers
    - Projects from fpn_out_channels (256) to memory_feature_dim (64)
    - Adds learned occlusion embedding for frames predicted as occluded

Config references:
    model.fpn_out_channels: 256      → in_dim parameter
    model.memory_feature_dim: 64     → out_dim parameter
    model.input_resolution: 1024     → determines stride-16 spatial size = 64×64

Paper references:
    Section 4: "The memory encoder generates a memory by downsampling the output
        mask using a convolutional module and summing it element-wise with the
        unconditioned frame embedding from the image-encoder, followed by
        light-weight convolutional layers to fuse the information."
    Appendix D.1: "we project the memory features in our memory bank to a
        dimension of 64."
    Appendix D.1: "we also add a learned occlusion embedding to the memory
        features of those frames that are predicted to be occluded (invisible)
        by the occlusion prediction head."
    Appendix D.1: "Our memory encoder does not use an additional image encoder
        and instead reuses the image embeddings produced by the Hiera encoder."
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LayerNorm2d helper (channels-first format)
# ---------------------------------------------------------------------------


class LayerNorm2d(nn.Module):
    """Layer normalization for 2D feature maps in [B, C, H, W] format.

    Permutes to channels-last, applies LayerNorm over C, then permutes back.
    Used in both mask_downsampler and fuser within MemoryEncoder.

    Args:
        num_channels: Number of channels C to normalize over.
        eps: Epsilon for numerical stability. Defaults to 1e-6.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm: nn.LayerNorm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization over the channel dimension.

        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            Normalized tensor of shape [B, C, H, W].
        """
        x = x.permute(0, 2, 3, 1)   # [B, H, W, C]
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)   # [B, C, H, W]
        return x


# ---------------------------------------------------------------------------
# MemoryEncoder
# ---------------------------------------------------------------------------


class MemoryEncoder(nn.Module):
    """Encodes frame predictions into compact spatial memory features.

    Converts the current frame's predicted mask and unconditioned frame
    embedding into a 64-dim spatial memory feature map stored in the
    MemoryBank. Called at the end of every processed frame in the streaming
    inference loop.

    Processing pipeline (Section 4, Appendix D.1):
        1. Downsample predicted mask (full resolution) to stride-16 spatial
           size via learned strided convolutions (mask_downsampler)
        2. Element-wise sum with unconditioned frame embedding (256-dim, stride-16)
        3. Apply light-weight convolutional fusion layers (fuser)
        4. Project from in_dim (256) to out_dim (64) via 1×1 conv (out_proj)
        5. Optionally add learned occlusion embedding if frame is occluded

    The output retains the H×W spatial structure of the frame embedding
    (e.g., 64×64 for 1024 input resolution), just with reduced channel depth.

    Attributes:
        mask_downsampler: nn.Sequential of strided convolutions downsampling
            the predicted mask from full resolution to stride-16 spatial size
            with in_dim output channels.
        fuser: nn.Sequential of light-weight conv layers fusing the summed
            mask features and frame embedding.
        out_proj: nn.Conv2d(in_dim, out_dim, kernel_size=1) for final
            projection to memory_feature_dim.
        occlusion_embedding: nn.Parameter of shape [1, out_dim, 1, 1] added
            to memory features of occluded frames.

    Args:
        in_dim: Input channel dimension of the frame embedding. Defaults to 256
            (config.model.fpn_out_channels).
        out_dim: Output channel dimension of the memory feature. Defaults to 64
            (config.model.memory_feature_dim).

    Example:
        encoder = MemoryEncoder(in_dim=256, out_dim=64)
        frame_embed = torch.randn(2, 256, 64, 64)   # stride-16 FPN output
        mask = torch.randn(2, 1, 1024, 1024)         # full-resolution mask
        memory = encoder(frame_embed, mask, is_occluded=False)
        # memory: [2, 64, 64, 64]
        memory_occ = encoder(frame_embed, mask, is_occluded=True)
        # memory_occ: [2, 64, 64, 64] with occlusion embedding added
    """

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 64,
    ) -> None:
        super().__init__()

        self.in_dim: int = in_dim
        self.out_dim: int = out_dim

        # ------------------------------------------------------------------
        # Mask downsampler: 16× spatial reduction via strided convolutions
        #
        # Downsamples the predicted mask from full resolution (e.g., 1024×1024)
        # to stride-16 spatial size (e.g., 64×64) to match the frame embedding.
        # Uses 4 stride-2 convolutions for 16× total downsampling.
        #
        # Channel progression: 1 → 4 → 16 → 64 → in_dim (256)
        # This "light-weight" design keeps intermediate channels small while
        # expanding to in_dim for the element-wise sum with frame_embedding.
        #
        # Paper: "downsampling the output mask using a convolutional module"
        # ------------------------------------------------------------------
        self.mask_downsampler: nn.Sequential = nn.Sequential(
            # Stage 1: 1 → 4 channels, 2× spatial downscale
            nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1, bias=False),
            LayerNorm2d(4),
            nn.ReLU(inplace=True),
            # Stage 2: 4 → 16 channels, 2× spatial downscale
            nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=1, bias=False),
            LayerNorm2d(16),
            nn.ReLU(inplace=True),
            # Stage 3: 16 → 64 channels, 2× spatial downscale
            nn.Conv2d(16, 64, kernel_size=3, stride=2, padding=1, bias=False),
            LayerNorm2d(64),
            nn.ReLU(inplace=True),
            # Stage 4: 64 → in_dim channels, 2× spatial downscale
            # Output matches frame_embedding spatial size and channel depth
            nn.Conv2d(64, in_dim, kernel_size=3, stride=2, padding=1, bias=False),
            LayerNorm2d(in_dim),
            nn.ReLU(inplace=True),
        )

        # ------------------------------------------------------------------
        # Fuser: light-weight convolutional layers to fuse mask + frame features
        #
        # Operates on the element-wise sum of downsampled mask features and
        # the unconditioned frame embedding. Uses two conv layers for fusion.
        #
        # Paper: "followed by light-weight convolutional layers to fuse the
        # information"
        # ------------------------------------------------------------------
        self.fuser: nn.Sequential = nn.Sequential(
            # Fusion conv 1: refine the summed features
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(in_dim),
            nn.ReLU(inplace=True),
            # Fusion conv 2: further refinement before projection
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(in_dim),
            nn.ReLU(inplace=True),
        )

        # ------------------------------------------------------------------
        # Output projection: in_dim (256) → out_dim (64)
        #
        # 1×1 convolution for channel-wise projection to memory_feature_dim.
        # Paper: "we project the memory features in our memory bank to a
        # dimension of 64" (Appendix D.1)
        # Config: model.memory_feature_dim: 64
        # ------------------------------------------------------------------
        self.out_proj: nn.Conv2d = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=1,
            bias=True,
        )

        # ------------------------------------------------------------------
        # Occlusion embedding: learned additive embedding for occluded frames
        #
        # Shape [1, out_dim, 1, 1] broadcasts over [B, out_dim, H, W].
        # Added to memory features AFTER projection to out_dim space.
        # Paper: "we also add a learned occlusion embedding to the memory
        # features of those frames that are predicted to be occluded"
        # (Appendix D.1)
        # ------------------------------------------------------------------
        self.occlusion_embedding: nn.Parameter = nn.Parameter(
            torch.zeros(1, out_dim, 1, 1)
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for all convolutional and linear layers.

        Uses Kaiming normal initialization for conv layers (appropriate for
        ReLU activations) and zeros for biases. The occlusion_embedding is
        initialized to zero so it has no effect at the start of training.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Occlusion embedding initialized to zero (no effect at training start)
        nn.init.zeros_(self.occlusion_embedding)

    def forward(
        self,
        frame_embedding: torch.Tensor,
        mask: torch.Tensor,
        is_occluded: bool = False,
    ) -> torch.Tensor:
        """Encode frame prediction into a compact spatial memory feature.

        Implements the three-step pipeline from Section 4:
            1. Downsample mask → match frame_embedding spatial size
            2. Element-wise sum with frame_embedding
            3. Fuse → project → optionally add occlusion embedding

        The frame_embedding must be the UNCONDITIONED output from the Hiera
        image encoder + FPN, NOT the memory-attention-conditioned embedding
        that feeds into the mask decoder. This is the "unconditioned frame
        embedding from the image-encoder" referenced in Section 4.

        Args:
            frame_embedding: Unconditioned frame embedding from HieraImageEncoder,
                shape [B, in_dim, H_e, W_e] where H_e = W_e = input_resolution/16.
                For 1024 input: [B, 256, 64, 64].
                For 512 input (ablations): [B, 256, 32, 32].
            mask: Predicted mask from MaskDecoder (best selected mask),
                shape [B, 1, H_img, W_img] at full input resolution.
                Values can be binary {0, 1}, probabilities [0, 1], or logits.
                For 1024 input: [B, 1, 1024, 1024].
                For 512 input (ablations): [B, 1, 512, 512].
            is_occluded: If True, add the learned occlusion embedding to the
                output memory feature. Derived from SAM2FrameOutput.occlusion_score
                by the caller (SAM2Model). Defaults to False.

        Returns:
            Memory feature tensor of shape [B, out_dim, H_e, W_e].
            For 1024 input: [B, 64, 64, 64].
            For 512 input (ablations): [B, 64, 32, 32].
            This tensor is passed to MemoryBank.add_memory() for storage.

        Raises:
            ValueError: If frame_embedding and downsampled mask have mismatched
                spatial dimensions after downsampling.
        """
        # ------------------------------------------------------------------
        # Step 1: Downsample mask to match frame_embedding spatial dimensions
        #
        # mask: [B, 1, H_img, W_img] → mask_features: [B, in_dim, H_e, W_e]
        #
        # The mask_downsampler applies 4 stride-2 convolutions for 16× total
        # downsampling. For 1024 input: 1024 → 512 → 256 → 128 → 64.
        # For 512 input: 512 → 256 → 128 → 64 → 32.
        #
        # If the mask resolution is not exactly 16× the frame embedding size
        # (e.g., due to non-power-of-2 input sizes), we apply adaptive pooling
        # after the conv downsampling to ensure exact spatial alignment.
        # ------------------------------------------------------------------
        mask_float: torch.Tensor = mask.float()
        mask_features: torch.Tensor = self.mask_downsampler(mask_float)
        # mask_features: [B, in_dim, H_mask_down, W_mask_down]

        # Ensure spatial dimensions match frame_embedding exactly
        # This handles edge cases where input resolution is not a power of 2
        target_h: int = frame_embedding.shape[2]
        target_w: int = frame_embedding.shape[3]
        actual_h: int = mask_features.shape[2]
        actual_w: int = mask_features.shape[3]

        if actual_h != target_h or actual_w != target_w:
            logger.debug(
                "MemoryEncoder: mask_features spatial size (%d, %d) != "
                "frame_embedding spatial size (%d, %d). Applying adaptive pooling.",
                actual_h, actual_w, target_h, target_w,
            )
            mask_features = F.adaptive_avg_pool2d(
                mask_features, output_size=(target_h, target_w)
            )

        # ------------------------------------------------------------------
        # Step 2: Element-wise sum with unconditioned frame embedding
        #
        # Both tensors are [B, in_dim, H_e, W_e] after alignment.
        # Paper: "summing it element-wise with the unconditioned frame
        # embedding from the image-encoder"
        # ------------------------------------------------------------------
        fused: torch.Tensor = frame_embedding + mask_features
        # fused: [B, in_dim, H_e, W_e]

        # ------------------------------------------------------------------
        # Step 3: Apply light-weight fusion convolutional layers
        #
        # Refines the fused representation before projection.
        # Paper: "followed by light-weight convolutional layers to fuse
        # the information"
        # ------------------------------------------------------------------
        fused = self.fuser(fused)
        # fused: [B, in_dim, H_e, W_e]

        # ------------------------------------------------------------------
        # Step 4: Project to memory_feature_dim (64)
        #
        # 1×1 conv reduces channels from in_dim (256) to out_dim (64).
        # Paper: "we project the memory features in our memory bank to a
        # dimension of 64" (Appendix D.1)
        # ------------------------------------------------------------------
        memory: torch.Tensor = self.out_proj(fused)
        # memory: [B, out_dim, H_e, W_e]

        # ------------------------------------------------------------------
        # Step 5: Add occlusion embedding for occluded frames
        #
        # The occlusion_embedding is a learned [1, out_dim, 1, 1] parameter
        # that broadcasts over [B, out_dim, H_e, W_e].
        # Paper: "we also add a learned occlusion embedding to the memory
        # features of those frames that are predicted to be occluded"
        # (Appendix D.1)
        # ------------------------------------------------------------------
        if is_occluded:
            memory = memory + self.occlusion_embedding

        return memory  # [B, out_dim, H_e, W_e]

    def encode_for_storage(
        self,
        frame_embedding: torch.Tensor,
        mask: torch.Tensor,
        occlusion_score: float,
        occlusion_threshold: float = 0.5,
    ) -> torch.Tensor:
        """Convenience method that thresholds occlusion_score internally.

        Wraps forward() by converting a continuous occlusion score to a
        boolean flag using the provided threshold. This is the typical call
        pattern from SAM2Model.forward_video_frame().

        Args:
            frame_embedding: Unconditioned frame embedding, shape
                [B, in_dim, H_e, W_e].
            mask: Predicted mask, shape [B, 1, H_img, W_img].
            occlusion_score: Continuous occlusion probability in [0, 1] from
                the MaskDecoder's occlusion_prediction_head (after sigmoid).
            occlusion_threshold: Threshold above which the frame is considered
                occluded. Defaults to 0.5.

        Returns:
            Memory feature tensor of shape [B, out_dim, H_e, W_e].
        """
        is_occluded: bool = float(occlusion_score) > occlusion_threshold
        return self.forward(frame_embedding, mask, is_occluded=is_occluded)
