"""Prompt encoder for SAM 2 - identical to SAM's prompt encoder.

Handles sparse prompts (points/positive clicks, negative clicks, bounding boxes)
and dense prompts (masks).

Sparse prompts are represented by positional encodings summed with learned embeddings
for each prompt type. Masks are embedded using convolutions and summed with the frame
embedding.
"""

import random
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import PromptEncoderConfig


class PositionEmbeddingRandom(nn.Module):
    """Position encoding using random spatial frequencies (from SAM)."""
    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None):
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer("positional_encoding_gaussian_matrix",
                            scale * torch.randn((2, num_pos_feats)))

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = 2 * coords - 1  # assuming coords in [0, 1]
        coords = coords @ self.positional_encoding_gaussian_matrix.to(coords.dtype)
        coords = 2 * torch.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Args: coords [B, N, 2] in [0, 1] range. Returns [B, N, C]."""
        return self._pe_encoding(coords)

    def forward_with_coords(self, coords_input: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        """Args: coords_input [B, N, 2] in pixel coordinates. Returns [B, N, C]."""
        coords_input = coords_input.float()
        coords = coords_input.clone()
        coords[..., 0] = coords[..., 0] / image_size[1]
        coords[..., 1] = coords[..., 1] / image_size[0]
        return self._pe_encoding(coords)


class PromptEncoder(nn.Module):
    """SAM-style prompt encoder for clicks, boxes, and masks.

    Architecture:
    - Sparse embeddings: learned embeddings for each prompt type + positional encoding
    - Dense embeddings: mask processed by convolutions, summed with image embeddings
    """
    def __init__(self, config: PromptEncoderConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.image_embedding_size = config.image_embedding_size
        self.input_image_size = config.image_embedding_size

        self.pe_layer = PositionEmbeddingRandom(config.embed_dim // 2)

        # Number of point types: foreground (positive), background (negative), box corner
        self.num_point_embeddings = 3

        self.point_embeddings = nn.Embedding(1, config.embed_dim)
        self.not_a_point_embed = nn.Embedding(1, config.embed_dim)

        # Mask downscaling
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, config.mask_input_channels // 4, kernel_size=2, stride=2),
            nn.LayerNorm([config.mask_input_channels // 4, config.image_embedding_size[0] // 2, config.image_embedding_size[1] // 2]),
            nn.GELU(),
            nn.Conv2d(config.mask_input_channels // 4, config.mask_input_channels, kernel_size=2, stride=2),
            nn.LayerNorm([config.mask_input_channels, config.image_embedding_size[0] // 4, config.image_embedding_size[1] // 4]),
            nn.GELU(),
            nn.Conv2d(config.mask_input_channels, config.mask_output_channels, kernel_size=1),
        )

        self.no_mask_embed = nn.Embedding(1, config.mask_output_channels)

    def _embed_points(self, points: torch.Tensor, labels: torch.Tensor, pad: bool = True) -> torch.Tensor:
        """Embed sparse point prompts.

        Args:
            points: [B, N, 2] in pixel coordinates
            labels: [B, N] - 1 for foreground, 0 for background, -1 for padding
            pad: whether to add padding token

        Returns:
            sparse_embeddings: [B, N+1, C] if pad else [B, N, C]
        """
        points = points + 0.5  # shift to center of pixel
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)

        # Mask to distinguish point types
        label_mask = torch.zeros_like(labels, dtype=torch.float)
        label_mask[labels == 1] = 1.0  # foreground
        label_mask[labels == 0] = 0.0  # background

        point_embedding = point_embedding + self.point_embeddings.weight[0:1] * label_mask.unsqueeze(-1)
        point_embedding = point_embedding + self.point_embeddings.weight[0:1] * (1 - label_mask).unsqueeze(-1)

        if pad:
            padding_point = self.not_a_point_embed.weight[0:1].expand(points.shape[0], 1, -1)
            point_embedding = torch.cat([point_embedding, padding_point], dim=1)

        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embed bounding box prompts.

        Args:
            boxes: [B, N, 4] in (x1, y1, x2, y2) pixel coordinates

        Returns:
            box_embeddings: [B, N*2, C] (two corners per box)
        """
        B, N = boxes.shape[:2]
        corners = boxes.reshape(B, N, 2, 2)
        corners = corners.reshape(B, N * 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(corners, self.input_image_size)
        corner_embedding = corner_embedding + self.point_embeddings.weight[0:1]
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embed dense mask prompts.

        Args:
            masks: [B, 1, H, W] binary masks

        Returns:
            mask_embeddings: [B, C, H', W']
        """
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_dense_pe(self) -> torch.Tensor:
        """Get dense positional encoding for the full image grid."""
        h, w = self.image_embedding_size
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, dtype=torch.float32),
            torch.arange(w, dtype=torch.float32),
            indexing="ij",
        )
        coords = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        coords = coords.unsqueeze(0)  # [1, H, W, 2]
        return self.pe_layer(coords / torch.tensor([w, h], dtype=torch.float32)).permute(0, 3, 1, 2)

    def forward(self, coords: Optional[torch.Tensor], labels: Optional[torch.Tensor],
                boxes: Optional[torch.Tensor], masks: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed all prompt types.

        Args:
            coords: [B, N, 2] click coordinates in pixels, or None
            labels: [B, N] click labels (1=fg, 0=bg, -1=pad), or None
            boxes: [B, N, 4] bounding boxes in pixels, or None
            masks: [B, 1, H, W] mask prompts, or None

        Returns:
            sparse_embeddings: [B, N_total, C]
            dense_embeddings: [B, C, H', W']
        """
        B = coords.shape[0] if coords is not None else boxes.shape[0] if boxes is not None else masks.shape[0]

        sparse_embeddings = torch.empty(B, 0, self.embed_dim, device=self._get_device())
        if coords is not None and labels is not None:
            coords_embed = self._embed_points(coords, labels)
            sparse_embeddings = torch.cat([sparse_embeddings, coords_embed], dim=1)

        if boxes is not None:
            box_embed = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embed], dim=1)

        if sparse_embeddings.shape[1] == 0:
            sparse_embeddings = self.not_a_point_embed.weight[0:1].expand(B, 1, -1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                B, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings

    def _get_device(self):
        return self.point_embeddings.weight.device


def sample_prompts_from_ground_truth(
    masks: torch.Tensor,
    mask_prompt_prob: float = 0.5,
    click_prompt_prob: float = 0.25,
    box_prompt_prob: float = 0.25,
    num_clicks: int = 1,
    max_clicks: int = 1,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Sample initial prompts from ground truth masks during training.

    Following SAM 2 training: initial prompts can be:
    - Mask with probability 0.5
    - Positive click with probability 0.25
    - Bounding box with probability 0.25

    Args:
        masks: [B, H, W] ground truth binary masks
        mask_prompt_prob: probability of using mask prompt
        click_prompt_prob: probability of using click prompt
        box_prompt_prob: probability of using box prompt
        num_clicks: number of clicks to sample
        max_clicks: maximum number of correction clicks

    Returns:
        coords, labels, boxes, masks - None for unused prompt types
    """
    B, H, W = masks.shape
    device = masks.device
    prompt_type = random.choices(
        ["mask", "click", "box"],
        weights=[mask_prompt_prob, click_prompt_prob, box_prompt_prob],
        k=B,
    )

    coords_list, labels_list, boxes_list, masks_list = [], [], [], []

    for b in range(B):
        gt_mask = masks[b]
        if gt_mask.sum() == 0:
            # No valid mask - skip
            coords_list.append(torch.zeros(0, 2, device=device))
            labels_list.append(torch.zeros(0, dtype=torch.long, device=device))
            boxes_list.append(torch.zeros(0, 4, device=device))
            masks_list.append(torch.zeros(1, H, W, device=device))
            continue

        if prompt_type[b] == "mask":
            coords_list.append(torch.zeros(0, 2, device=device))
            labels_list.append(torch.zeros(0, dtype=torch.long, device=device))
            boxes_list.append(torch.zeros(0, 4, device=device))
            masks_list.append(gt_mask.unsqueeze(0))
        elif prompt_type[b] == "click":
            # Sample positive click from center of mass region
            ys, xs = torch.where(gt_mask)
            if len(ys) > 0:
                idx = random.randint(0, len(ys) - 1)
                cy, cx = ys[idx].float(), xs[idx].float()
            else:
                cy, cx = H / 2, W / 2
            coords_list.append(torch.tensor([[cx, cy]], device=device))
            labels_list.append(torch.tensor([1], dtype=torch.long, device=device))
            boxes_list.append(torch.zeros(0, 4, device=device))
            masks_list.append(torch.zeros(1, H, W, device=device))
        else:  # box
            ys, xs = torch.where(gt_mask)
            if len(ys) > 0:
                x1, x2 = xs.min().float(), xs.max().float()
                y1, y2 = ys.min().float(), ys.max().float()
            else:
                x1, y1, x2, y2 = 0.0, 0.0, float(W), float(H)
            boxes_list.append(torch.tensor([[x1, y1, x2, y2]], device=device))
            coords_list.append(torch.zeros(0, 2, device=device))
            labels_list.append(torch.zeros(0, dtype=torch.long, device=device))
            masks_list.append(torch.zeros(1, H, W, device=device))

    # Pad to max sizes
    max_coords = max(c.shape[0] for c in coords_list) if coords_list else 0
    max_boxes = max(b.shape[0] for b in boxes_list) if boxes_list else 0

    # Return None if all empty
    if max_coords == 0 and max_boxes == 0:
        return None, None, None, masks

    # Pad coords
    if max_coords > 0:
        padded_coords, padded_labels = [], []
        for c, l in zip(coords_list, labels_list):
            if c.shape[0] < max_coords:
                pad = torch.zeros(max_coords - c.shape[0], 2, device=device)
                pad_label = torch.full((max_coords - l.shape[0],), -1, dtype=torch.long, device=device)
                c = torch.cat([c, pad], dim=0)
                l = torch.cat([l, pad_label], dim=0)
            padded_coords.append(c)
            padded_labels.append(l)
        coords = torch.stack(padded_coords)
        labels = torch.stack(padded_labels)
    else:
        coords, labels = None, None

    if max_boxes > 0:
        padded_boxes = []
        for b_item in boxes_list:
            if b_item.shape[0] < max_boxes:
                pad = torch.zeros(max_boxes - b_item.shape[0], 4, device=device)
                b_item = torch.cat([b_item, pad], dim=0)
            padded_boxes.append(b_item)
        boxes = torch.stack(padded_boxes)
    else:
        boxes = None

    # Check if all mask prompts are empty
    mask_prompts = torch.stack(masks_list)
    if (mask_prompts.sum(dim=[1, 2, 3]) == 0).all():
        masks = None
    else:
        masks = mask_prompts

    return coords, labels, boxes, masks
