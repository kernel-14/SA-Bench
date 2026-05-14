import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class PromptEncoder(nn.Module):
    """
    Encodes various prompt inputs to be integrated with image features.

    The prompt encoder design largely follows the original Segment Anything Model (SAM).
    It processes sparse prompts (points, boxes) and dense prompts (masks).

    - Sparse prompts are encoded using positional encodings summed with learned embeddings.
    - Masks are embedded using convolutions and summed with the frame embedding.
    """

    def __init__(self,
                 embed_dim: int,
                 image_size: int,
                 input_image_size: Tuple[int, int],
                 mask_in_channels: int,
                 activation: Type[nn.Module] = nn.GELU,
                 num_point_embeddings: int = 4, # 0: padding, 1: positive, 2: negative, 3: box_top_left, 4: box_bottom_right
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.input_image_size = input_image_size
        self.mask_in_channels = mask_in_channels

        self.point_embeddings = nn.Embedding(num_point_embeddings, embed_dim)
        self.no_mask_embed = nn.Embedding(1, embed_dim)

        # Positional encoding for sparse prompts (points and boxes)
        # This would typically be a fixed sinusoidal positional embedding or learned.
        # For simplicity, we'll use a simple coordinate-based positional embedding.
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        # Mask embedding layers
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_channels // 4, kernel_size=2, stride=2),
            nn.LayerNorm(mask_in_channels // 4),
            activation(),
            nn.Conv2d(mask_in_channels // 4, mask_in_channels // 2, kernel_size=2, stride=2),
            nn.LayerNorm(mask_in_channels // 2),
            activation(),
            nn.Conv2d(mask_in_channels // 2, embed_dim, kernel_size=1),
        )
        self.mask_layer_norm = nn.LayerNorm(embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """Get dense positional embeddings for image features."""
        return self.pe_layer(self.input_image_size).unsqueeze(0) # (1, H, W, C)

    def _embed_points(self, points: torch.Tensor, labels: torch.Tensor, pad: bool)
        points = points + 0.5 # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = torch.ones((labels.shape[0], 1), device=labels.device) * -1 # -1 for padding
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding = point_embedding + self.point_embeddings(labels)
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor)
        boxes = boxes + 0.5 # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2) # (B, 2, 2)
        box_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        box_embedding_tl = box_embedding[:, 0, :] + self.point_embeddings(torch.tensor(3, device=boxes.device))
        box_embedding_br = box_embedding[:, 1, :] + self.point_embeddings(torch.tensor(4, device=boxes.device))
        return torch.cat([box_embedding_tl.unsqueeze(1), box_embedding_br.unsqueeze(1)], dim=1)

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        mask_embedding = self.mask_downscaling(masks)
        return self.mask_layer_norm(mask_embedding)

    def forward(
        self,
        points: Tuple[torch.Tensor, torch.Tensor] = None,
        boxes: torch.Tensor = None,
        masks: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            points (tuple(torch.Tensor, torch.Tensor)): A tuple of (N, B, 2) point coordinates
                and (N, B) point labels. Labels: 0 is negative, 1 is positive, -1 is padding.
            boxes (torch.Tensor): A Bx4 tensor given a box prompt in XYXY format.
            masks (torch.Tensor): A Bx1xHxW dense mask input.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The sparse embeddings (B, N, C) and dense embeddings (B, C, H, W).
        """
        sparse_embeddings = torch.empty((masks.shape[0] if masks is not None else 1, 0, self.embed_dim), device=self.point_embeddings.weight.device)
        if points is not None:
            # points: (B, num_points, 2), labels: (B, num_points)
            sparse_embeddings = torch.cat([sparse_embeddings, self._embed_points(points[0], points[1], False)], dim=1)
        if boxes is not None:
            sparse_embeddings = torch.cat([sparse_embeddings, self._embed_boxes(boxes)], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, self.embed_dim, 1, 1).expand(masks.shape[0] if masks is not None else 1, -1, *self.get_dense_pe().shape[-3:-1])

        return sparse_embeddings, dense_embeddings

class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding for sparse prompts, similar to SAM.
    Generates random positional embeddings that are then scaled.
    """
    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None:
            scale = 2 * math.pi
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((num_pos_feats, 2)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (B, N, 2) or (N, 2)
        # Gaussian projection
        coords = coords @ self.positional_encoding_gaussian_matrix.to(coords.dtype)
        # Sinusoidal encoding
        coords = 2 * math.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """
        Generate dense positional embeddings for a given image size.
        """
        h, w = size
        grid = torch.ones((h, w), device=self.positional_encoding_gaussian_matrix.device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe # (H, W, C)

    def forward_with_coords(self, coords_input: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        """
        Positionally encode point prompts.
        Args:
          coords_input (torch.Tensor): Point coordinates (B, N, 2).
          image_size (tuple(int, int)): The size of the image the coordinates refer to.
        """
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(self.positional_encoding_gaussian_matrix.device))


# For type hinting, if not directly imported:
from typing import Type, Optional
import math

# Example usage (for testing the module structure)
if __name__ == "__main__":
    embed_dim = 256
    image_size = 1024 # Original image size
    input_image_size = (image_size // 16, image_size // 16) # Image embedding size
    mask_in_channels = 16 # Example, could be different

    prompt_encoder = PromptEncoder(
        embed_dim=embed_dim,
        image_size=image_size,
        input_image_size=input_image_size,
        mask_in_channels=mask_in_channels,
    )

    batch_size = 1
    # Test points
    points_coords = torch.randint(0, image_size, (batch_size, 2, 2), dtype=torch.float)
    points_labels = torch.randint(0, 2, (batch_size, 2), dtype=torch.long) # 0: negative, 1: positive
    sparse_embed_points, dense_embed_points = prompt_encoder(points=(points_coords, points_labels))
    print(f"Sparse embeddings (points) shape: {sparse_embed_points.shape}")
    print(f"Dense embeddings (points) shape: {dense_embed_points.shape}")

    # Test boxes
    boxes = torch.randint(0, image_size, (batch_size, 4), dtype=torch.float)
    sparse_embed_boxes, dense_embed_boxes = prompt_encoder(boxes=boxes)
    print(f"Sparse embeddings (boxes) shape: {sparse_embed_boxes.shape}")
    print(f"Dense embeddings (boxes) shape: {dense_embed_boxes.shape}")

    # Test masks
    masks = torch.randint(0, 2, (batch_size, 1, image_size // 4, image_size // 4), dtype=torch.float)
    sparse_embed_masks, dense_embed_masks = prompt_encoder(masks=masks)
    print(f"Sparse embeddings (masks) shape: {sparse_embed_masks.shape}")
    print(f"Dense embeddings (masks) shape: {dense_embed_masks.shape}")

    # Test all together
    sparse_embed_all, dense_embed_all = prompt_encoder(points=(points_coords, points_labels), boxes=boxes, masks=masks)
    print(f"Sparse embeddings (all) shape: {sparse_embed_all.shape}")
    print(f"Dense embeddings (all) shape: {dense_embed_all.shape}")

    # Test dense PE
    dense_pe = prompt_encoder.get_dense_pe()
    print(f"Dense PE shape: {dense_pe.shape}")
