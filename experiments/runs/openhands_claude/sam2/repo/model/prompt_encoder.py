"""
Prompt encoder for SAM 2.

Identical to SAM's prompt encoder (Kirillov et al., 2023).
Encodes three types of prompts:
  - Sparse: positive/negative clicks (points) and bounding boxes
  - Dense: masks (embedded via convolutions, summed with frame embedding)

Sparse prompts → positional encodings + learned type embeddings.
Dense prompts  → convolutional embedding summed with the frame embedding.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies (Fourier features).
    Used to encode point coordinates.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: Tensor) -> Tensor:
        """coords: (..., 2) in [0, 1] → (..., 2*num_pos_feats)."""
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> Tensor:
        """Generate positional encoding for a grid of size (H, W)."""
        H, W = size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((H, W), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / H
        x_embed = x_embed / W
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # (2*num_pos_feats, H, W)

    def forward_with_coords(self, coords_input: Tensor, image_size: Tuple[int, int]) -> Tensor:
        """Encode point coordinates. coords_input: (B, N, 2) in pixel space."""
        coords = coords_input.clone().float()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords)  # (B, N, 2*num_pos_feats)


class PromptEncoder(nn.Module):
    """
    Encodes prompts for input to SAM 2's mask decoder.

    Sparse prompts (points, boxes) → (B, N_sparse, embed_dim) tokens.
    Dense prompts (masks) → (B, embed_dim, H/4, W/4) embedding.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        image_embedding_size: Tuple[int, int] = (64, 64),
        input_image_size: Tuple[int, int] = (1024, 1024),
        mask_in_chans: int = 16,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size

        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        # Learned embeddings for prompt types
        self.num_point_embeddings = 4  # pos click, neg click, top-left box, bottom-right box
        point_embeddings = [nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        # Dense mask embedding
        self.mask_input_size = (
            4 * image_embedding_size[0],
            4 * image_embedding_size[1],
        )
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            nn.LayerNorm([mask_in_chans // 4, image_embedding_size[0] * 2, image_embedding_size[1] * 2]),
            nn.GELU(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            nn.LayerNorm([mask_in_chans, image_embedding_size[0], image_embedding_size[1]]),
            nn.GELU(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> Tensor:
        """Return positional encoding for the image embedding grid."""
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(self, points: Tensor, labels: Tensor, pad: bool) -> Tensor:
        """
        points: (B, N, 2) pixel coordinates
        labels: (B, N) — 1=positive, 0=negative, -1=padding
        """
        points = points + 0.5  # shift to pixel center
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)

        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        # Add type embeddings
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def _embed_boxes(self, boxes: Tensor) -> Tensor:
        """boxes: (B, 4) in xyxy pixel format."""
        boxes = boxes + 0.5  # shift to pixel center
        coords = boxes.reshape(-1, 2, 2)  # (B, 2, 2): [[x1,y1],[x2,y2]]
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding  # (B, 2, embed_dim)

    def _embed_masks(self, masks: Tensor) -> Tensor:
        """masks: (B, 1, H, W) binary masks."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding  # (B, embed_dim, H/4, W/4)

    def forward(
        self,
        points: Optional[Tuple[Tensor, Tensor]] = None,
        boxes: Optional[Tensor] = None,
        masks: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            sparse_embeddings: (B, N_sparse, embed_dim)
            dense_embeddings:  (B, embed_dim, H_emb, W_emb)
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())

        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)

        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings

    def _get_batch_size(
        self,
        points: Optional[Tuple[Tensor, Tensor]],
        boxes: Optional[Tensor],
        masks: Optional[Tensor],
    ) -> int:
        if points is not None:
            return points[0].shape[0]
        if boxes is not None:
            return boxes.shape[0]
        if masks is not None:
            return masks.shape[0]
        return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device
