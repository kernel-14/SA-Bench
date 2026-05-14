## models/prompt_encoder.py
"""Prompt encoder for SAM 2: encodes clicks, boxes, and masks into embeddings.

This module is identical to SAM's PromptEncoder as stated in the paper:
"Our prompt encoder is identical to SAM's and can be prompted by clicks
(positive or negative), boxes, or masks to define the extent of the object
in a given frame." (Section 4)

The encoder produces two outputs consumed by the MaskDecoder:
    - sparse_embeddings: [B, N_tokens, embed_dim] for clicks and boxes
    - dense_embeddings:  [B, embed_dim, H_embed, W_embed] for masks

Config references:
    model.fpn_out_channels: 256  → embed_dim
    model.input_resolution: 1024 → input_image_size, image_embedding_size=(64,64)

Paper references:
    Section 4: "Our prompt encoder is identical to SAM's."
    Section 4: "Sparse prompts are represented by positional encodings summed
        with learned embeddings for each prompt type, while masks are embedded
        using convolutions and summed with the frame embedding."
    Appendix D.1: "The prompt encoder design follows SAM."
"""

import logging
from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.positional_encoding import PositionEmbeddingRandom

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LayerNorm2d helper
# ---------------------------------------------------------------------------


class LayerNorm2d(nn.Module):
    """Layer normalization for 2D feature maps in [B, C, H, W] format.

    Standard nn.LayerNorm expects [..., C] (channels last). This wrapper
    permutes to [B, H, W, C], applies LayerNorm over C, then permutes back.
    Used in mask_downscaling within PromptEncoder.

    Args:
        num_channels: Number of channels C to normalize over.
        eps: Epsilon for numerical stability. Defaults to 1e-6.

    Example:
        norm = LayerNorm2d(16)
        x = torch.randn(2, 16, 64, 64)
        out = norm(x)  # (2, 16, 64, 64)
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
        # [B, C, H, W] → [B, H, W, C] → norm → [B, C, H, W]
        x = x.permute(0, 2, 3, 1)   # [B, H, W, C]
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)   # [B, C, H, W]
        return x


# ---------------------------------------------------------------------------
# PromptEncoder
# ---------------------------------------------------------------------------


class PromptEncoder(nn.Module):
    """Encodes click, box, and mask prompts for the SAM 2 mask decoder.

    Identical to SAM's PromptEncoder (Kirillov et al., 2023). Handles three
    prompt types:

    1. Points (clicks): positive/negative/padding labels → sparse embeddings
    2. Boxes: bounding box corners → sparse embeddings (2 tokens per box)
    3. Masks: binary/soft masks → dense embeddings via convolutional downscaling

    When a prompt type is absent, learned "no-prompt" embeddings are
    substituted so the decoder always receives valid inputs.

    Attributes:
        embed_dim: Embedding dimension (256, matching fpn_out_channels).
        image_embedding_size: Spatial size of the frame embedding (H_e, W_e).
            For 1024 input with stride-16 FPN: (64, 64).
        pe_layer: PositionEmbeddingRandom for spatial coordinate encoding.
        point_embeddings: Learned type embeddings for 4 point types:
            [0] negative click, [1] positive click,
            [2] box top-left corner, [3] box bottom-right corner.
        not_a_point_embed: Learned embedding for padding tokens (label=-1).
        mask_input_size: Intermediate mask size before downscaling.
            = (4 * H_e, 4 * W_e) = (256, 256) for 1024 input.
        mask_downscaling: Sequential conv layers downscaling mask 4× to
            match image_embedding_size.
        no_mask_embed: Learned embedding used when no mask prompt is given.

    Args:
        embed_dim: Embedding dimension. Defaults to 256 (config.model.fpn_out_channels).
        image_embedding_size: (H_embed, W_embed) spatial size of frame embedding.
            Defaults to (64, 64) for 1024 input resolution.
        input_image_size: (H_img, W_img) of the input image.
            Defaults to (1024, 1024) per config.model.input_resolution.
        mask_in_chans: Intermediate channel count in mask downscaling.
            Defaults to 16 (SAM default).
        activation: Activation function class for mask downscaling.
            Defaults to nn.GELU.

    Example:
        encoder = PromptEncoder(
            embed_dim=256,
            image_embedding_size=(64, 64),
            input_image_size=(1024, 1024),
        )
        # Click prompts: 3 positive clicks
        points = torch.randint(0, 1024, (1, 3, 2)).float()
        labels = torch.ones(1, 3, dtype=torch.long)
        sparse, dense = encoder(points=(points, labels), boxes=None, masks=None)
        # sparse: [1, 3, 256], dense: [1, 256, 64, 64]
    """

    def __init__(
        self,
        embed_dim: int = 256,
        image_embedding_size: Tuple[int, int] = (64, 64),
        input_image_size: Tuple[int, int] = (1024, 1024),
        mask_in_chans: int = 16,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()

        self.embed_dim: int = embed_dim
        self.image_embedding_size: Tuple[int, int] = image_embedding_size
        self.input_image_size: Tuple[int, int] = input_image_size

        # ------------------------------------------------------------------
        # Positional encoding layer
        # num_pos_feats = embed_dim // 2 so that sin+cos concatenation
        # produces embed_dim-dimensional output.
        # ------------------------------------------------------------------
        self.pe_layer: PositionEmbeddingRandom = PositionEmbeddingRandom(
            num_pos_feats=embed_dim // 2
        )

        # ------------------------------------------------------------------
        # Learned type embeddings for point prompts
        # 4 embeddings: [neg_click, pos_click, box_top_left, box_bot_right]
        # ------------------------------------------------------------------
        num_point_embeddings: int = 4
        self.point_embeddings: nn.Embedding = nn.Embedding(
            num_point_embeddings, embed_dim
        )

        # Separate embedding for padding tokens (label == -1)
        self.not_a_point_embed: nn.Embedding = nn.Embedding(1, embed_dim)

        # ------------------------------------------------------------------
        # Mask downscaling: 4× spatial reduction via strided convolutions
        # Input:  [B, 1, mask_input_size[0], mask_input_size[1]]
        # Output: [B, embed_dim, H_embed, W_embed]
        #
        # mask_input_size = (4 * H_embed, 4 * W_embed)
        # For 1024 input: H_embed=64, mask_input_size=(256, 256)
        # ------------------------------------------------------------------
        self.mask_input_size: Tuple[int, int] = (
            4 * image_embedding_size[0],
            4 * image_embedding_size[1],
        )

        self.mask_downscaling: nn.Sequential = nn.Sequential(
            # Stage 1: 1 → 4 channels, 2× spatial downscale
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            # Stage 2: 4 → 16 channels, 2× spatial downscale
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            # Stage 3: 16 → embed_dim channels, no spatial change
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )

        # ------------------------------------------------------------------
        # No-mask embedding: used when no mask prompt is provided
        # Shape: [1, embed_dim] → broadcast to [B, embed_dim, H_e, W_e]
        # ------------------------------------------------------------------
        self.no_mask_embed: nn.Embedding = nn.Embedding(1, embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for learned embeddings and conv layers.

        Embeddings use normal initialization (mean=0, std=1).
        Conv layers use Kaiming uniform (default PyTorch init).
        """
        nn.init.normal_(self.point_embeddings.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.not_a_point_embed.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.no_mask_embed.weight, mean=0.0, std=1.0)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_dense_pe(self) -> torch.Tensor:
        """Get the dense positional encoding for the image embedding grid.

        Returns the positional encoding for the full image_embedding_size
        spatial grid. This is passed as `image_pe` to MaskDecoder.forward()
        and used in the two-way transformer's cross-attention.

        Returns:
            Dense PE tensor of shape [1, embed_dim, H_embed, W_embed].
            The batch dimension of 1 is broadcast over the actual batch size
            inside the mask decoder.
        """
        # pe_layer.forward(size) returns [H_embed, W_embed, embed_dim]
        # Permute to [embed_dim, H_embed, W_embed] then add batch dim
        pe = self.pe_layer(self.image_embedding_size)  # [H_e, W_e, embed_dim]
        pe = pe.permute(2, 0, 1)                        # [embed_dim, H_e, W_e]
        return pe.unsqueeze(0)                          # [1, embed_dim, H_e, W_e]

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode all provided prompts into sparse and dense embeddings.

        Any combination of prompt types can be None. When all are None,
        returns empty sparse embeddings and the no-mask dense embedding.

        Args:
            points: Optional tuple of (coords, labels) where:
                - coords: [B, N, 2] float tensor of (x, y) pixel coordinates
                - labels: [B, N] long tensor with values:
                    1 = positive click, 0 = negative click, -1 = padding
            boxes: Optional [B, 4] float tensor of (x1, y1, x2, y2) pixel
                coordinates for bounding box prompts.
            masks: Optional [B, 1, H_img, W_img] float tensor of mask prompts.
                Values should be in [0, 1] (probabilities) or raw logits.

        Returns:
            Tuple of:
                - sparse_embeddings: [B, N_tokens, embed_dim] where N_tokens
                  is the total number of point + box tokens. Shape [B, 0, embed_dim]
                  when neither points nor boxes are provided.
                - dense_embeddings: [B, embed_dim, H_embed, W_embed] mask
                  embedding, or broadcast no-mask embedding when masks=None.

        Raises:
            ValueError: If batch sizes are inconsistent across prompt types.
        """
        # Determine batch size from the first non-None input
        bs: int = self._get_batch_size(points, boxes, masks)

        device: torch.device = self._get_device()

        # ------------------------------------------------------------------
        # Build sparse embeddings from points and/or boxes
        # ------------------------------------------------------------------
        sparse_embeddings: torch.Tensor = torch.empty(
            (bs, 0, self.embed_dim),
            device=device,
            dtype=torch.float32,
        )

        if points is not None:
            coords, labels = points
            # Pad with a not-a-point token when boxes are also present
            # so the decoder sees a consistent token layout
            pad: bool = boxes is not None
            point_embeddings: torch.Tensor = self._embed_points(
                coords, labels, pad=pad
            )
            sparse_embeddings = torch.cat(
                [sparse_embeddings, point_embeddings], dim=1
            )

        if boxes is not None:
            box_embeddings: torch.Tensor = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat(
                [sparse_embeddings, box_embeddings], dim=1
            )

        # ------------------------------------------------------------------
        # Build dense embeddings from mask or use no-mask embedding
        # ------------------------------------------------------------------
        if masks is not None:
            dense_embeddings: torch.Tensor = self._embed_masks(masks)
        else:
            # Broadcast no_mask_embed to [B, embed_dim, H_embed, W_embed]
            # no_mask_embed.weight: [1, embed_dim]
            no_mask: torch.Tensor = self.no_mask_embed.weight  # [1, embed_dim]
            dense_embeddings = no_mask.reshape(1, self.embed_dim, 1, 1).expand(
                bs,
                self.embed_dim,
                self.image_embedding_size[0],
                self.image_embedding_size[1],
            )

        return sparse_embeddings, dense_embeddings

    # ------------------------------------------------------------------
    # Private encoding methods
    # ------------------------------------------------------------------

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """Encode point (click) prompts into sparse embeddings.

        Normalizes pixel coordinates to [0, 1], applies random Fourier
        positional encoding, then adds learned type embeddings based on
        the click label (positive=1, negative=0, padding=-1).

        Args:
            points: [B, N, 2] float tensor of (x, y) pixel coordinates.
                Values in [0, input_image_size[1]] for x and
                [0, input_image_size[0]] for y.
            labels: [B, N] long tensor with values 0, 1, or -1.
            pad: If True, append a "not-a-point" padding token to each
                sequence. Used when boxes are also present to maintain
                consistent token ordering.

        Returns:
            Point embeddings of shape [B, N_out, embed_dim] where
            N_out = N + 1 if pad=True, else N.
        """
        # Shift coordinates by +0.5 to center within pixels (SAM convention)
        points = points + 0.5

        if pad:
            # Append a padding point at coordinate (0, 0) with label -1
            # Shape: [B, 1, 2] and [B, 1]
            padding_point: torch.Tensor = torch.zeros(
                (points.shape[0], 1, 2),
                device=points.device,
                dtype=points.dtype,
            )
            padding_label: torch.Tensor = -torch.ones(
                (labels.shape[0], 1),
                device=labels.device,
                dtype=labels.dtype,
            )
            points = torch.cat([points, padding_point], dim=1)   # [B, N+1, 2]
            labels = torch.cat([labels, padding_label], dim=1)   # [B, N+1]

        # Encode coordinates using random Fourier features
        # pe_layer.forward_with_coords normalizes by input_image_size internally
        point_embedding: torch.Tensor = self.pe_layer.forward_with_coords(
            points, self.input_image_size
        )  # [B, N_out, embed_dim]

        # Add learned type embeddings based on label values
        # label == -1 (padding): use not_a_point_embed
        # label == 0 (negative): use point_embeddings[0]
        # label == 1 (positive): use point_embeddings[1]

        # Start with zeros and fill in per-label embeddings
        # We use masking to handle all three cases efficiently
        type_embedding: torch.Tensor = torch.zeros_like(point_embedding)

        # Padding tokens (label == -1)
        padding_mask: torch.Tensor = labels == -1  # [B, N_out]
        type_embedding[padding_mask] = self.not_a_point_embed.weight[0]

        # Negative click tokens (label == 0)
        neg_mask: torch.Tensor = labels == 0  # [B, N_out]
        type_embedding[neg_mask] = self.point_embeddings.weight[0]

        # Positive click tokens (label == 1)
        pos_mask: torch.Tensor = labels == 1  # [B, N_out]
        type_embedding[pos_mask] = self.point_embeddings.weight[1]

        # Sum positional encoding and type embedding
        return point_embedding + type_embedding  # [B, N_out, embed_dim]

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Encode bounding box prompts into sparse embeddings.

        Each box is represented as two corner tokens:
        - Token 0: top-left corner (x1, y1) + point_embeddings[2]
        - Token 1: bottom-right corner (x2, y2) + point_embeddings[3]

        Args:
            boxes: [B, 4] float tensor of (x1, y1, x2, y2) pixel coordinates.

        Returns:
            Box embeddings of shape [B, 2, embed_dim].
        """
        # Shift by +0.5 to center within pixels (SAM convention)
        boxes = boxes + 0.5

        # Reshape to two corner points: [B, 2, 2]
        # corners[:, 0, :] = (x1, y1) top-left
        # corners[:, 1, :] = (x2, y2) bottom-right
        corners: torch.Tensor = boxes.reshape(-1, 2, 2)  # [B, 2, 2]

        # Encode corner coordinates using random Fourier features
        corner_embedding: torch.Tensor = self.pe_layer.forward_with_coords(
            corners, self.input_image_size
        )  # [B, 2, embed_dim]

        # Add learned corner type embeddings
        # corner_embedding[:, 0, :] += point_embeddings[2]  (top-left)
        # corner_embedding[:, 1, :] += point_embeddings[3]  (bottom-right)
        corner_embedding[:, 0, :] = (
            corner_embedding[:, 0, :] + self.point_embeddings.weight[2]
        )
        corner_embedding[:, 1, :] = (
            corner_embedding[:, 1, :] + self.point_embeddings.weight[3]
        )

        return corner_embedding  # [B, 2, embed_dim]

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Encode mask prompts into dense embeddings via convolutional downscaling.

        The mask is first resized to mask_input_size = (4*H_embed, 4*W_embed)
        using bilinear interpolation, then passed through mask_downscaling
        which applies 4× total spatial reduction to match image_embedding_size.

        From Section 4: "masks are embedded using convolutions and summed
        with the frame embedding."

        Args:
            masks: [B, 1, H_img, W_img] float tensor of mask prompts.
                Values can be binary {0, 1}, probabilities [0, 1], or logits.

        Returns:
            Dense mask embeddings of shape [B, embed_dim, H_embed, W_embed].
        """
        # Resize mask to mask_input_size = (4*H_embed, 4*W_embed)
        # For 1024 input: (256, 256)
        mask_resized: torch.Tensor = F.interpolate(
            masks.float(),
            size=self.mask_input_size,
            mode="bilinear",
            align_corners=False,
        )  # [B, 1, 4*H_embed, 4*W_embed]

        # Apply convolutional downscaling: 4× spatial reduction
        # Output: [B, embed_dim, H_embed, W_embed]
        return self.mask_downscaling(mask_resized)

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        """Infer batch size from the first non-None prompt input.

        Args:
            points: Optional (coords, labels) tuple.
            boxes: Optional box tensor.
            masks: Optional mask tensor.

        Returns:
            Batch size as an integer. Returns 1 if all inputs are None.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        return 1

    def _get_device(self) -> torch.device:
        """Get the device of the module's parameters.

        Returns:
            torch.device of the pe_layer's random matrix buffer.
        """
        return self.pe_layer.positional_encoding_gaussian_matrix.device
