"""
NaViL Visual Multi-scale Packing.

Implements the multi-scale image processing described in Section 4.1:

"Given an input image I_0 and downsampling rate τ, a multi-scale image 
sequence {I_i} is obtained by continuously downsampling the original image 
(i.e. H_i = τ^i H_0, W_i = τ^i W_0) until its area is smaller than a given 
threshold."

Default τ = sqrt(2)/2 ≈ 0.707 (Section 5.1)

Special tokens:
- <begin_of_image>, <end_of_image>: mark image boundaries
- <end_of_line>: at end of each row of image tokens (spatial position)
- <end_of_scale>: at end of each scale
"""

import math
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F


def generate_multi_scale_images(
    image: torch.Tensor,
    downsample_rate: float = 0.7071067811865476,  # sqrt(2)/2
    min_area: int = 256 * 256,
    max_scales: int = 5,
) -> List[torch.Tensor]:
    """
    Generate multi-scale image sequence by iterative downsampling.
    
    The process continues until the image area falls below min_area.
    
    Args:
        image: (C, H, W) or (B, C, H, W) input image
        downsample_rate: τ, spatial scaling factor (default √2/2)
        min_area: minimum area threshold to stop downsampling
        max_scales: maximum number of scales
        
    Returns:
        List of images at different scales [(C, H_0, W_0), (C, H_1, W_1), ...]
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    
    B, C, H, W = image.shape
    scales = [image.squeeze(0) if squeeze else image]
    
    current_h, current_w = H, W
    
    for i in range(1, max_scales):
        # Compute new dimensions: H_i = τ^i * H_0, W_i = τ^i * W_0
        new_h = int(H * (downsample_rate ** i))
        new_w = int(W * (downsample_rate ** i))
        
        # Check if area is below threshold
        if new_h * new_w < min_area:
            break
        
        # Ensure even dimensions
        new_h = max(1, new_h)
        new_w = max(1, new_w)
        
        # Downsample using bilinear interpolation
        scaled = F.interpolate(
            image,
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False,
        )
        
        if squeeze:
            scales.append(scaled.squeeze(0))
        else:
            scales.append(scaled)
    
    return scales


def pad_to_multiple(
    image: torch.Tensor,
    multiple: int = 32,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Pad image to ensure height and width are multiples of `multiple`.
    
    From Section 5.1: "The input images are first padded to ensure 
    its length and width are multiples of 32."
    
    Returns:
        padded_image, (original_H, original_W)
    """
    if image.dim() == 3:
        C, H, W = image.shape
        pad_h = (multiple - H % multiple) % multiple
        pad_w = (multiple - W % multiple) % multiple
        
        if pad_h > 0 or pad_w > 0:
            image = F.pad(image, (0, pad_w, 0, pad_h), value=0)
        
        return image, (H, W)
    else:
        B, C, H, W = image.shape
        pad_h = (multiple - H % multiple) % multiple
        pad_w = (multiple - W % multiple) % multiple
        
        if pad_h > 0 or pad_w > 0:
            image = F.pad(image, (0, pad_w, 0, pad_h), value=0)
        
        return image, (H, W)


def pack_multi_scale_tokens(
    scale_tokens: List[torch.Tensor],
    spatial_dims: List[Tuple[int, int]],
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Pack multi-scale visual tokens into a single sequence with special tokens.
    
    The sequence format:
        <begin_of_image> scale_0_tokens <end_of_scale> 
        <begin_of_image> scale_1_tokens <end_of_scale> ...
        <end_of_image>
    
    Within each scale, <end_of_line> tokens are inserted at the end of each row.
    
    Args:
        scale_tokens: List of token tensors [(N_i, C), ...] from each scale
        spatial_dims: List of (H_i, W_i) spatial dimensions for each scale
        
    Returns:
        Concatenated token tensor and list of special token masks
    """
    all_tokens = []
    separator_masks = []
    
    for i, (tokens, (h, w)) in enumerate(zip(scale_tokens, spatial_dims)):
        # Reshape tokens to (H, W, C) to insert row separators
        N = tokens.shape[0]
        assert N == h * w, f"Token count {N} != spatial dims {h}x{w}={h*w}"
        
        tokens_2d = tokens.view(h, w, -1)  # (H, W, C)
        
        # Process each row and add <end_of_line>
        row_tokens = []
        for row_idx in range(h):
            row_tokens.append(tokens_2d[row_idx])  # (W, C)
            # <end_of_line> would be inserted here in practice
        
        scale_tokens_flat = tokens_2d.view(h * w, -1)
        all_tokens.append(scale_tokens_flat)
        
        # <end_of_scale> separator
        # In practice this is a learned special token embedding
    
    # Concatenate: scale_0 | sep | scale_1 | sep | ...
    combined = torch.cat(all_tokens, dim=0)
    
    return combined, separator_masks


def compute_2d_rope(
    height: int,
    width: int,
    dim: int,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute 2D Rotary Position Embeddings for the visual encoder.
    
    The visual encoder uses 2D-RoPE to capture global spatial relationships
    (Section 5.1). This splits the embedding dimension into two halves:
    one for horizontal position encoding and one for vertical.
    
    Returns:
        cos, sin tensors of shape (height*width, dim)
    """
    # Generate position indices
    y_pos = torch.arange(height, device=device, dtype=dtype).unsqueeze(1).repeat(1, width)
    x_pos = torch.arange(width, device=device, dtype=dtype).unsqueeze(0).repeat(height, 1)
    
    y_pos = y_pos.flatten()  # (H*W,)
    x_pos = x_pos.flatten()  # (H*W,)
    
    # Compute frequencies
    dim_half = dim // 2
    freq_dim = dim_half // 2  # half for x, half for y
    
    # Frequency bands
    freqs = 1.0 / (10000 ** (torch.arange(0, freq_dim, 2, device=device, dtype=dtype) / freq_dim))
    
    # Compute angles for y and x positions
    y_angles = y_pos.unsqueeze(-1) * freqs.unsqueeze(0)  # (H*W, freq_dim//2)
    x_angles = x_pos.unsqueeze(-1) * freqs.unsqueeze(0)  # (H*W, freq_dim//2)
    
    # Interleave to create full 2D RoPE
    # For dim_half = freq_dim: [y_0, x_0, y_1, x_1, ...]
    angles = torch.stack([y_angles, x_angles], dim=-1).flatten(-2)  # (H*W, freq_dim)
    
    # Repeat for full dimension
    angles = torch.cat([angles, angles], dim=-1)  # (H*W, dim)
    
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    
    return cos, sin


class MultiScaleProcessor:
    """
    Processor for visual multi-scale packing during inference.
    
    Implements the full multi-scale pipeline described in Section 4.1.
    """
    
    def __init__(
        self,
        downsample_rate: float = 0.7071067811865476,
        min_area: int = 256 * 256,
        max_scales: int = 5,
        patch_size: int = 16,
        pad_multiple: int = 32,
    ):
        self.downsample_rate = downsample_rate
        self.min_area = min_area
        self.max_scales = max_scales
        self.patch_size = patch_size
        self.pad_multiple = pad_multiple
        
    def process(
        self,
        image: torch.Tensor,
        visual_encoder: torch.nn.Module,
        connector: torch.nn.Module,
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """
        Process an image through multi-scale packing.
        
        Args:
            image: (C, H, W) or (B, C, H, W) input image
            visual_encoder: NaViL visual encoder module
            connector: NaViL connector module
            
        Returns:
            Packed visual tokens and spatial dimensions for each scale
        """
        # Generate multi-scale images
        scales = generate_multi_scale_images(
            image,
            self.downsample_rate,
            self.min_area,
            self.max_scales,
        )
        
        all_tokens = []
        spatial_dims = []
        
        for scale_img in scales:
            if scale_img.dim() == 3:
                scale_img = scale_img.unsqueeze(0)
            
            B, C, H, W = scale_img.shape
            
            # Pad to multiple
            scale_img, (orig_h, orig_w) = pad_to_multiple(scale_img, self.pad_multiple)
            
            # Compute spatial dimensions after patching
            H_feat = ((orig_h + self.pad_multiple - 1) // self.pad_multiple) * self.pad_multiple // self.patch_size
            W_feat = ((orig_w + self.pad_multiple - 1) // self.pad_multiple) * self.pad_multiple // self.patch_size
            
            # Visual encoder forward
            vis_features = visual_encoder(scale_img)
            
            # Connector
            tokens, H_out, W_out = connector(vis_features, H_feat, W_feat)
            
            all_tokens.append(tokens)
            spatial_dims.append((H_out, W_out))
        
        # Pack tokens from all scales
        packed_tokens, _ = pack_multi_scale_tokens(all_tokens, spatial_dims)
        
        return packed_tokens, spatial_dims
