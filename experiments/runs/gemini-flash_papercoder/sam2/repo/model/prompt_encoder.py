import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Any, Dict, List, Optional, Tuple, Union

# Placeholder for Config type hint to avoid circular import with config.py
# In a real project, this would be 'from config import Config'
Config = Any


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random Fourier features as used in SAM.
    This generates a 2D positional embedding for (x, y) coordinates.
    """

    def __init__(self, num_pos_feats: int = 128, scale: Optional[float] = None):
        """
        Initializes the random positional embedding layer.

        Args:
            num_pos_feats (int): Number of positional features (half of the embedding dimension).
            scale (Optional[float]): Scale factor for positional encoding. If None, it's inferred.
        """
        super().__init__()
        self.num_pos_feats = num_pos_feats
        if scale is None:
            scale = 1.0
        # Initialize a fixed Gaussian matrix for projection.
        # Shape (2, num_pos_feats) because we are encoding 2D coordinates (x, y).
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Applies positional encoding to normalized coordinates.

        Args:
            coords (torch.Tensor): Normalized coordinates (range [-1, 1]) of shape (B, N, 2).

        Returns:
            torch.Tensor: Positional embeddings of shape (B, N, 2 * num_pos_feats).
        """
        # coords is (B, N, 2)
        # positional_encoding_gaussian_matrix is (2, num_pos_feats)
        # Matrix multiplication results in (B, N, num_pos_feats)
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * math.pi * coords
        # Apply sin and cos functions element-wise and concatenate them.
        # This doubles the last dimension to 2 * num_pos_feats.
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, coords: torch.Tensor, input_image_size: Tuple[int, int]) -> torch.Tensor:
        """
        Generates positional embeddings for a batch of coordinates.

        Args:
            coords (torch.Tensor): Input coordinates (x, y) in pixel space.
                                   Shape: (B, N, 2). Can be float or long.
            input_image_size (Tuple[int, int]): (H, W) of the input image
                                                to normalize coordinates.

        Returns:
            torch.Tensor: Positional embeddings of shape (B, N, 2 * num_pos_feats).
        """
        h, w = input_image_size
        
        # Normalize coordinates to [-1, 1] based on the input image size.
        # We assume coordinates are 0-indexed.
        coords_normalized = coords.clone().float()
        coords_normalized[..., 0] = coords_normalized[..., 0] / (w / 2.0) - 1.0 # Normalize x to [-1, 1]
        coords_normalized[..., 1] = coords_normalized[..., 1] / (h / 2.0) - 1.0 # Normalize y to [-1, 1]
        
        return self._pe_encoding(coords_normalized)


class PromptEncoder(nn.Module):
    """
    Encodes various types of user prompts (points, bounding boxes, masks) into
    a unified tensor of embeddings, which are then fed into the MaskDecoder.
    """

    def __init__(self, config: Config):
        """
        Initializes the PromptEncoder.

        Args:
            config (Config): The global configuration object.
        """
        super().__init__()
        self._config = config

        # Retrieve hidden dimension from config, defaulting to 256.
        self.hidden_dim: int = self._config.get("model.memory_attention.hidden_dim", 256)
        
        # Determine the input image size for positional encoding normalization.
        # Prioritize full_train resolution, then pretrain, then default to 1024x1024.
        default_resolution = 1024
        if self._config.get("training.full_train.enabled", False):
            res = self._config.get("training.full_train.resolution", default_resolution)
            self.input_image_size = (res, res)
        elif self._config.get("training.pretrain.enabled", False):
            res = self._config.get("training.pretrain.resolution", default_resolution)
            self.input_image_size = (res, res)
        else:
            self.input_image_size = (default_resolution, default_resolution)

        # Positional embedding layer (random Fourier features).
        # num_pos_feats is half of hidden_dim because it outputs both sin and cos.
        self.positional_embedding_layer = PositionEmbeddingRandom(
            num_pos_feats=self.hidden_dim // 2
        )

        # Learnable embeddings for point types:
        # Index 0 for negative points, 1 for positive points, 2 for padding/absent points.
        self.point_embeddings = nn.Embedding(3, self.hidden_dim)
        
        # Learnable embeddings for box corners:
        # Index 0 for top-left corner, 1 for bottom-right corner.
        self.box_embeddings = nn.Embedding(2, self.hidden_dim)
        
        # Learned token to signal the presence of a mask prompt.
        # This parameter will be expanded to match batch size during forward pass.
        self.mask_input_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))

        # Learned embedding used when no prompts are provided (e.g., during evaluation
        # where the model needs a default token stream even with no prompts).
        # This parameter will also be expanded to match batch size.
        self.no_mask_embed = nn.Parameter(torch.randn(1, 1, self.hidden_dim))

    def forward(
        self,
        points: Optional[torch.Tensor], # (B, N_points, 3) -> (x, y, label)
        boxes: Optional[torch.Tensor],  # (B, N_boxes, 4) -> (x_min, y_min, x_max, y_max)
        masks: Optional[torch.Tensor],  # (B, 1, H, W)
    ) -> torch.Tensor:
        """
        Converts various types of user prompts into a unified tensor of embeddings.

        Args:
            points (Optional[torch.Tensor]): Point prompts. Shape (B, N_points, 3),
                                             where last dim is (x, y, label). Labels: 1=positive, 0=negative.
                                             Can be None or contain padding (label=-1).
            boxes (Optional[torch.Tensor]): Bounding box prompts. Shape (B, N_boxes, 4),
                                            where last dim is (x_min, y_min, x_max, y_max).
                                            Can be None.
            masks (Optional[torch.Tensor]): Mask prompt. Shape (B, 1, H, W).
                                            Can be None or an empty tensor.

        Returns:
            torch.Tensor: Concatenated prompt embeddings. Shape (B, N_total_tokens, hidden_dim).
        """
        all_prompt_embeddings: List[torch.Tensor] = []
        batch_size: int = 1 # Initialize batch_size, will update from first valid input

        # Process points if provided and not empty
        if points is not None and points.numel() > 0:
            batch_size = points.shape[0]
            coords = points[:, :, :2] # Extract (x, y) coordinates
            labels = points[:, :, 2].long() # Extract labels (0, 1, or -1 for padding)
            
            # Generate 2D positional embeddings for the point coordinates
            point_pos_embed = self.positional_embedding_layer(coords, self.input_image_size)
            
            # Retrieve learnable type embeddings for the labels
            point_type_embed = self.point_embeddings(labels)
            
            # Combine positional and type embeddings by summation
            point_embeddings = point_pos_embed + point_type_embed
            all_prompt_embeddings.append(point_embeddings)

        # Process boxes if provided and not empty
        if boxes is not None and boxes.numel() > 0:
            batch_size = boxes.shape[0]
            num_boxes = boxes.shape[1]

            # Extract top-left and bottom-right corner coordinates
            top_left_coords = boxes[:, :, :2]  # (B, N_boxes, 2)
            bottom_right_coords = boxes[:, :, 2:] # (B, N_boxes, 2)

            # Generate positional embeddings for both corners
            top_left_pos_embed = self.positional_embedding_layer(top_left_coords, self.input_image_size)
            bottom_right_pos_embed = self.positional_embedding_layer(bottom_right_coords, self.input_image_size)

            # Retrieve type embeddings for top-left (label 0) and bottom-right (label 1)
            # Create label tensors with correct batch and box dimensions
            top_left_type_embed = self.box_embeddings(
                torch.zeros(batch_size, num_boxes, dtype=torch.long, device=boxes.device)
            ) # Shape (B, N_boxes, hidden_dim)
            bottom_right_type_embed = self.box_embeddings(
                torch.ones(batch_size, num_boxes, dtype=torch.long, device=boxes.device)
            ) # Shape (B, N_boxes, hidden_dim)

            # Combine positional and type embeddings for each corner
            box_embeddings_top_left = top_left_pos_embed + top_left_type_embed
            box_embeddings_bottom_right = bottom_right_pos_embed + bottom_right_type_embed
            
            # Concatenate top-left and bottom-right embeddings along the token dimension (dim=1)
            # This results in (B, 2 * N_boxes, hidden_dim)
            box_embeddings = torch.cat([box_embeddings_top_left, box_embeddings_bottom_right], dim=1)
            all_prompt_embeddings.append(box_embeddings)

        # Process masks (signal its presence with a dedicated token)
        # Check if masks tensor is provided and has a valid batch dimension (B > 0).
        # Note: The actual spatial embedding of the mask itself is handled elsewhere,
        # this encoder only provides a token indicating its presence.
        if masks is not None and masks.numel() > 0 and masks.shape[0] > 0:
            batch_size = masks.shape[0]
            # Expand the mask_input_token to match the current batch size.
            # mask_input_token is (1, 1, hidden_dim), expand to (B, 1, hidden_dim).
            mask_token_embedding = self.mask_input_token.expand(batch_size, -1, -1)
            all_prompt_embeddings.append(mask_token_embedding)

        # Concatenate all collected prompt embeddings.
        if len(all_prompt_embeddings) == 0:
            # If no prompts were provided, return the 'no_mask_embed'.
            # If batch_size was never set (e.g., all inputs were None), default to 1.
            effective_batch_size = batch_size if 'batch_size' in locals() else 1
            return self.no_mask_embed.expand(effective_batch_size, -1, -1)
        else:
            return torch.cat(all_prompt_embeddings, dim=1)

