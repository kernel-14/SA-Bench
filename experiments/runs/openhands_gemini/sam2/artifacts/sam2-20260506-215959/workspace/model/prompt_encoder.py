
import torch
import torch.nn as nn
from typing import Tuple, Optional
from .layers import MLP

class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    Copied from Segment Anything Model (SAM) codebase.
    """
    def __init__(self, num_pos_feats: int = 128, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((num_pos_feats, 2)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Positionally encode points that are normalized to [0,1].
        """
        # assuming coords are already normalized
        coords = coords @ self.positional_encoding_gaussian_matrix
        return torch.cat((torch.sin(coords), torch.cos(coords)), dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """
        Generates positional encoding for a grid of a given size.
        """
        h, w = size
        grid = torch.ones((h, w), device=self.positional_encoding_gaussian_matrix.device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C H W


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_size: int,
        input_image_size: int,
        mask_in_chans: int,
        num_point_embeddings: int,
        kernel_size: int = 1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.input_image_size = input_image_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.point_embeddings = nn.ModuleList(
            [nn.Embedding(1, embed_dim) for _ in range(num_point_embeddings)]
        )
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.box_encoder = MLP(4, embed_dim, embed_dim) # 4 coords (x1,y1,x2,y2) -> embed_dim

        self.mask_encoder = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(1, mask_in_chans // 4),
            nn.GELU(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(1, mask_in_chans),
            nn.GELU(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=kernel_size, padding=kernel_size // 2),
        )

        self.num_point_embeddings = num_point_embeddings

    def _embed_points(self, points, labels, pad):
        """
        Embeds point prompts.
        points: (B, N, 2) tensor of point coordinates
        labels: (B, N) tensor of labels (0=negative, 1=positive, -1=padding)
        pad: (B, 1, 2) tensor, if pad is provided, it means there are no points for this batch.
        """
        point_embedding = self.point_embeddings[0].weight # Positive point embedding
        negative_point_embedding = self.point_embeddings[1].weight # Negative point embedding

        # Handle padding for points
        if pad is not None:
            point_embedding = torch.cat([point_embedding, self.not_a_point_embed.weight], dim=0)
            negative_point_embedding = torch.cat([negative_point_embedding, self.not_a_point_embed.weight], dim=0)
            labels = torch.cat([labels, -torch.ones_like(labels[:, :1])], dim=1) # Append -1 for padding

        point_embeddings = point_embedding[labels.long()] + negative_point_embedding[(1 - labels).long()]
        point_embeddings[labels == -1] = self.not_a_point_embed.weight

        # Reshape points for PE
        points_pe = self.pe_layer._pe_encoding(points.reshape(-1, 2)).reshape(points.shape[0], points.shape[1], -1)

        return points_pe + point_embeddings

    def _embed_boxes(self, boxes):
        """
        Embeds box prompts.
        boxes: (B, 4) tensor of box coordinates
        """
        boxes_pe = self.pe_layer._pe_encoding(boxes.reshape(-1, 2)).reshape(boxes.shape[0], 2, -1)
        box_embedding = self.box_encoder(boxes)
        return box_embedding + boxes_pe.sum(1) # Sum over points of the box

    def _embed_masks(self, masks, image_embedding_size):
        """
        Embeds mask prompts.
        masks: (B, 1, H, W) tensor of masks
        image_embedding_size: (H_embed, W_embed)
        """
        mask_embedding = self.mask_encoder(masks)
        pe = self.pe_layer(image_embedding_size).unsqueeze(0)
        return mask_embedding + pe

    def forward(
        self,
        points: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
        image_embedding_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds prompts for input to the mask decoder.

        Arguments:
          points (torch.Tensor or None): A BxNx2 tensor of point prompts.
          labels (torch.Tensor or None): A BxN tensor of point labels.
          boxes (torch.Tensor or None): A Bx4 tensor of box prompts.
          masks (torch.Tensor or None): A Bx1xHxW tensor of mask inputs.
          image_embedding_size (tuple(int, int)): The size of the image embedding.

        Returns:
          torch.Tensor: The embedded sparse prompts.
          torch.Tensor: The embedded dense prompts.
        """
        sparse_embeddings = torch.empty((points.shape[0] if points is not None else 1, 0, self.embed_dim), device=image_embedding_size[0].device)
        if points is not None:
            sparse_embeddings = torch.cat([sparse_embeddings, self._embed_points(points, labels, None)], dim=1)
        if boxes is not None:
            sparse_embeddings = torch.cat([sparse_embeddings, self._embed_boxes(boxes)], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks, image_embedding_size)
        else:
            # Create a placeholder for dense_embeddings if no mask is provided
            dense_embeddings = self.not_a_point_embed.weight.reshape(1, 1, -1).repeat(
                sparse_embeddings.shape[0], image_embedding_size[0], image_embedding_size[1]
            ).permute(0, 2, 3, 1) # (B, H_embed, W_embed, C)
            dense_embeddings = dense_embeddings.reshape(
                sparse_embeddings.shape[0], image_embedding_size[0], image_embedding_size[1], self.embed_dim
            ).permute(0, 3, 1, 2) # (B, C, H_embed, W_embed)


        return sparse_embeddings, dense_embeddings

