"""
Prompt Encoder for SAM 2.

Identical to SAM's prompt encoder. Can be prompted by clicks
(positive or negative), boxes, or masks to define the extent
of the object in a given frame.

Sparse prompts (points, boxes) are represented by positional
encodings summed with learned embeddings for each prompt type.

Dense prompts (masks) are embedded using convolutions and summed
with the frame embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PromptEncoder(nn.Module):
    """Prompt encoder for SAM 2, identical to SAM's design."""

    def __init__(
        self,
        embed_dim: int = 256,
        image_embedding_size: Tuple[int, int] = (64, 64),
        input_image_size: Tuple[int, int] = (1024, 1024),
        mask_in_chans: int = 16,
        activation: nn.Module = nn.GELU,
    ):
        """
        Args:
            embed_dim: embedding dimension
            image_embedding_size: spatial size of image embeddings
            input_image_size: spatial size of input images
            mask_in_chans: number of input channels for mask embedding
            activation: activation function
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size
        self.mask_in_chans = mask_in_chans

        # Learned embeddings for point types
        self.point_embeddings = nn.ModuleList([
            nn.Embedding(1, embed_dim),  # positive point
            nn.Embedding(1, embed_dim),  # negative point
        ])

        # A special "not-a-point" embedding
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        # Box embedding (encoded as top-left and bottom-right point embeddings)
        # We use two corner embeddings
        self.box_corner_embeddings = nn.ModuleList([
            nn.Embedding(1, embed_dim),  # top-left
            nn.Embedding(1, embed_dim),  # bottom-right
        ])

        # Mask downsampling network
        self.mask_downsampling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            nn.LayerNorm([mask_in_chans // 4, input_image_size[0] // 2, input_image_size[1] // 2]),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            nn.LayerNorm([mask_in_chans, input_image_size[0] // 4, input_image_size[1] // 4]),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )

        # PE for sparse prompts
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """Embed point prompts.
        Args:
            points: [B, N, 2] point coordinates in [0, 1] range
            labels: [B, N] point labels (1=positive, 0=negative)
            pad: whether to pad with not-a-point embedding
        """
        points = points + 0.5  # Shift to 0.5 - 1.5 range for PE
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)

        # Use label to select positive/negative embedding
        # labels: 1 -> positive (index 0), 0 -> negative (index 1)
        point_embedding = point_embedding + torch.where(
            labels.unsqueeze(-1) == 1,
            self.point_embeddings[0].weight,  # positive
            self.point_embeddings[1].weight,  # negative
        )

        if pad:
            padding_point = self.not_a_point_embed.weight.unsqueeze(0).expand(
                point_embedding.shape[0], 1, self.embed_dim
            )
            point_embedding = torch.cat([point_embedding, padding_point], dim=1)

        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embed box prompts.
        Args:
            boxes: [B, N, 4] box coordinates in [0, 1] range (x1, y1, x2, y2)
        """
        # Split into top-left and bottom-right corners
        tl = boxes[..., :2]  # top-left
        br = boxes[..., 2:]  # bottom-right
        box_embedding = self.pe_layer.forward_with_coords(tl + 0.5, self.input_image_size)
        box_embedding = box_embedding + self.box_corner_embeddings[0].weight
        box_embedding_br = self.pe_layer.forward_with_coords(br + 0.5, self.input_image_size)
        box_embedding_br = box_embedding_br + self.box_corner_embeddings[1].weight
        box_embedding = torch.cat([box_embedding, box_embedding_br], dim=1)
        return box_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embed dense mask prompts.
        Args:
            masks: [B, H, W] or [B, 1, H, W] binary masks
        """
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)  # [B, 1, H, W]
        mask_embedding = self.mask_downsampling(masks)
        # mask_embedding: [B, embed_dim, H/4, W/4]
        return mask_embedding

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_dense_pe(self, shape: Tuple[int, int]) -> torch.Tensor:
        """Get dense positional encodings for image embeddings."""
        return self.pe_layer(shape).unsqueeze(0)

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        boxes: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode prompts.
        Args:
            points: tuple of (point_coords, point_labels)
                point_coords: [B, N, 2] coordinates in [0, 1] range
                point_labels: [B, N] labels (1=positive, 0=negative)
            boxes: [B, N, 4] box coordinates in [0, 1] range
            masks: [B, H, W] or [B, 1, H, W] dense masks

        Returns:
            sparse_embeddings: [B, N_tokens, embed_dim]
            dense_embeddings: [B, embed_dim, H/16, W/16] (or None if no mask)
        """
        sparse_embeddings = None
        dense_embeddings = None

        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            if sparse_embeddings is None:
                sparse_embeddings = point_embeddings
            else:
                sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)

        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            if sparse_embeddings is None:
                sparse_embeddings = box_embeddings
            else:
                sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)

        if sparse_embeddings is None:
            # No sparse prompts -> use a single not-a-point embedding
            sparse_embeddings = self.not_a_point_embed.weight.unsqueeze(0).unsqueeze(0)

        return sparse_embeddings, dense_embeddings


class PositionEmbeddingRandom(nn.Module):
    """Position encoding using random spatial frequencies."""

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None):
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0, 1]."""
        # coords: [..., 2] assuming coords are in [0, 1]^2 square
        coords = 2 * coords - 1  # map to [-1, 1]
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * torch.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, dtype=torch.float32),
            torch.arange(w, dtype=torch.float32),
            indexing="ij",
        )
        # Normalize to [0, 1]
        grid_y = grid_y / h
        grid_x = grid_x / w
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.stack([grid_x, grid_y], dim=-1).to(device)
        return self._pe_encoding(grid)

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0, 1]."""
        coords = coords_input.clone()
        coords[..., 0] = coords[..., 0] / image_size[1]
        coords[..., 1] = coords[..., 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float32))
