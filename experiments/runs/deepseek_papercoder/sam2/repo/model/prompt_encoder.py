# model/prompt_encoder.py
"""
SAM 2 prompt encoder.

Implements a prompt encoder identical to SAM's, which transforms user prompts
(clicks, boxes, masks) into embeddings suitable for the mask decoder.

The module provides three independent methods:
- encode_clicks : encodes positive/negative point clicks.
- encode_boxes   : encodes bounding boxes as corner point embeddings.
- encode_masks   : applies a lightweight convolutional network to down‑sample
                   an optional mask prompt to the feature map size, or returns
                   a learned "no mask" embedding when no mask is given.

All designs follow Sections 4 and D.1 of the SAM 2 paper and are consistent
with the configuration file (`config.yaml`).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionEmbeddingRandom(nn.Module):
    """
    Random Fourier feature positional encoding for 2D coordinates.

    Maps normalised (x, y) coordinates to a high‑dimensional sinusoidal
    embedding using a fixed random projection matrix.  The output dimension
    is `2 * num_pos_feats`.

    This matches the positional encoding used in the original SAM prompt
    encoder and is applied to clicks and box corners.
    """

    def __init__(self, num_pos_feats: int = 128) -> None:
        """
        Args:
            num_pos_feats: half the output embedding dimension.
        """
        super().__init__()
        self.num_pos_feats = num_pos_feats
        # Fixed random Gaussian matrix (2, num_pos_feats)
        self.register_buffer(
            "gaussian_matrix", torch.randn(2, num_pos_feats), persistent=False
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, N, 2) tensor of normalised (x, y) coordinates in [0, 1].

        Returns:
            (B, N, 2 * num_pos_feats) sinusoidal embedding.
        """
        # Project coordinates: (B, N, 2) @ (2, num_pos_feats) -> (B, N, num_pos_feats)
        proj = torch.matmul(coords, self.gaussian_matrix.to(coords.dtype))  # keeps dtype

        # Apply sine and cosine and concatenate
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class PromptEncoder(nn.Module):
    """
    Prompt encoder for SAM 2.

    It processes clicks, boxes and masks into embeddings of a common
    dimensionality (`embed_dim`).  Sparse prompts (clicks, boxes) result in
    token sequences; dense prompts (masks) produce spatial feature maps that
    are added element‑wise to the image embedding.

    Args:
        embed_dim: Feature dimension for all output embeddings.
        image_embedding_size: Spatial size (H_feat, W_feat) of the image
            embedding that the mask embedding must match.  For the default
            configuration (1024 × 1024 input, stride 16) this is (64, 64).
        input_resolution: Input image resolution (square). Used to normalise
            pixel coordinates.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        image_embedding_size: Tuple[int, int] = (64, 64),
        input_resolution: int = 1024,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.input_resolution = input_resolution
        self.image_embedding_size = image_embedding_size
        H_feat, W_feat = image_embedding_size

        # Positional encoding for sparse prompts (clicks, box corners)
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim must be divisible by 2, got {embed_dim}")
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        # Learnable type embeddings (added to positional encodings)
        self.positive_click_embed = nn.Parameter(torch.zeros(1, embed_dim))
        self.negative_click_embed = nn.Parameter(torch.zeros(1, embed_dim))
        self.box_embed = nn.Parameter(torch.zeros(1, embed_dim))

        # --- Mask embedding network ---
        # Strided convolutions that downsample the mask from input_resolution²
        # to image_embedding_size.
        # Total stride required: input_resolution / H_feat = 16.
        # We use four Conv2d layers each with stride 2.
        channels = [1, embed_dim // 16, embed_dim // 8, embed_dim // 4, embed_dim]
        layers = []
        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            layers.append(
                nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)
            )
            if i < len(channels) - 2:  # no activation after the last conv
                layers.append(nn.ReLU(inplace=True))
        self.mask_downscaling = nn.Sequential(*layers)

        # Learnable "no mask" embedding – used when no mask prompt is provided
        self.no_mask_embed = nn.Parameter(torch.zeros(1, embed_dim, 1, 1))

        # Initialise parameters
        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_clicks(
        self, coords: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode point clicks.

        Args:
            coords: (B, N, 2) tensor of click pixel coordinates in
                [0, input_resolution).
            labels: (B, N) tensor, with 1 for positive clicks and 0 for
                negative clicks.

        Returns:
            Sparse prompt tokens of shape (B, N, embed_dim).  If N == 0,
            returns an empty tensor (B, 0, embed_dim).
        """
        if coords.numel() == 0:
            return torch.empty(
                coords.shape[0], 0, self.embed_dim, device=coords.device, dtype=coords.dtype
            )

        # Normalise to [0, 1]
        coords = coords.float() / self.input_resolution

        # Positional encoding
        pe = self.pe_layer(coords)  # (B, N, embed_dim)

        # Select the appropriate learned embedding per point
        labels = labels.long()
        # Build a mask for positive clicks: (B, N)
        is_positive = labels == 1
        is_negative = labels == 0
        # Gather the correct type embedding
        type_embed = torch.zeros_like(pe)
        # Broadcast the learnable 1D embeddings across the batch and points
        type_embed[is_positive] = self.positive_click_embed.to(pe.dtype)
        type_embed[is_negative] = self.negative_click_embed.to(pe.dtype)
        # (any other label value is treated as negative; this matches SAM's behaviour)

        return pe + type_embed

    def encode_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Encode bounding boxes.

        Each box is represented by the embeddings of its two corner points
        (top‑left and bottom‑right), each receiving the same box‑type
        embedding.

        Args:
            boxes: (B, N, 4) tensor in pixel coordinates (x1, y1, x2, y2).

        Returns:
            Sparse prompt tokens of shape (B, 2*N, embed_dim).
        """
        if boxes.numel() == 0:
            return torch.empty(
                boxes.shape[0], 0, self.embed_dim, device=boxes.device, dtype=boxes.dtype
            )

        B, N, _ = boxes.shape
        # Reshape to (B, 2*N, 2): (x1,y1), (x2,y2)
        corners = boxes.view(B, N * 2, 2).float()
        # Normalise
        corners = corners / self.input_resolution

        # Positional encoding
        pe = self.pe_layer(corners)  # (B, 2*N, embed_dim)

        # Box type embedding (same for both corners)
        type_embed = self.box_embed.to(pe.dtype).expand(B, 2 * N, self.embed_dim)

        return pe + type_embed

    def encode_masks(self, masks: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Encode a mask prompt (or lack thereof).

        When `masks` is provided, it is down‑sampled to the feature map size
        and transformed by a small convolutional network.  Otherwise, a
        learned "no mask" embedding is returned.

        Args:
            masks: (B, 1, H_in, W_in) binary mask tensor with H_in == W_in
                == input_resolution, or None.

        Returns:
            Dense embedding of shape (B, embed_dim, H_feat, W_feat) that
            will be added to the image embedding.
        """
        B = (
            masks.shape[0]
            if masks is not None
            else self.no_mask_embed.shape[0]  # determine batch from parameter
        )
        # Handle the no‑mask case
        if masks is None:
            return self.no_mask_embed.expand(B, self.embed_dim, *self.image_embedding_size)

        # Check spatial size; we assume the caller already resized to input_resolution
        H_in, W_in = masks.shape[2], masks.shape[3]
        if (H_in, W_in) != (self.input_resolution, self.input_resolution):
            # If not, resize to ensure correct downscaling; use nearest to preserve binary nature
            masks = F.interpolate(
                masks.float(),
                size=(self.input_resolution, self.input_resolution),
                mode="nearest",
            )

        # Downscale the mask
        mask_embed = self.mask_downscaling(masks)  # (B, embed_dim, H_feat, W_feat)

        # Optionally verify the output size matches image_embedding_size
        assert (
            mask_embed.shape[2:] == self.image_embedding_size
        ), f"Mask embedding size {mask_embed.shape[2:]} != {self.image_embedding_size}"

        return mask_embed

    def forward(
        self,
        coords: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Convenience forward that mimics the original SAM interface.

        This method encodes all provided prompt types and returns sparse
        tokens (concatenated clicks and boxes) and a dense mask embedding.

        Returns:
            sparse_embeddings: (B, total_prompt_tokens, embed_dim) or None if
                no sparse prompts were given.
            dense_embedding: (B, embed_dim, H_feat, W_feat) always returned.
        """
        sparse_embeddings = []

        # Encode clicks if present
        if coords is not None and labels is not None:
            click_emb = self.encode_clicks(coords, labels)
            sparse_embeddings.append(click_emb)

        # Encode boxes if present
        if boxes is not None:
            box_emb = self.encode_boxes(boxes)
            sparse_embeddings.append(box_emb)

        # Concatenate all sparse features along the token dimension
        if sparse_embeddings:
            sparse_concat = torch.cat(sparse_embeddings, dim=1)
        else:
            sparse_concat = None

        # Encode mask (or no‑mask)
        dense_embedding = self.encode_masks(masks)

        return sparse_concat, dense_embedding

