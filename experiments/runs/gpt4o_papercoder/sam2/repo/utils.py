"""
utils.py

Utility functions and shared modules for SAM 2 implementation. Includes functionality for data augmentation,
positional encoding, metric computation, prompt simulation, and memory management.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
from typing import List, Dict, Tuple, Optional


def augment_video_sequence(frames: torch.Tensor, augmentations: Dict) -> torch.Tensor:
    """
    Apply augmentations (e.g., horizontal flip, random crop, affine transform) to video sequences.

    Args:
        frames (torch.Tensor): Video frames tensor [B, T, C, H, W].
        augmentations (Dict): Augmentation configurations (e.g., {"horizontal_flip": True, ...}).

    Returns:
        torch.Tensor: Augmented video frames [B, T, C, H, W].
    """
    B, T, C, H, W = frames.shape
    aug_frames = frames.clone()

    for t in range(T):
        if augmentations.get("horizontal_flip", False) and random.random() < 0.5:
            aug_frames[:, t] = torch.flip(aug_frames[:, t], dims=[3])  # Horizontal flip on width

        if augmentations.get("random_crop", False):
            crop_size = int(0.9 * H)
            top = random.randint(0, H - crop_size)
            left = random.randint(0, W - crop_size)
            aug_frames[:, t] = aug_frames[:, t, :, top:top + crop_size, left:left + crop_size]

        if augmentations.get("affine_transform", False):
            angle = random.uniform(-10, 10)
            scale = random.uniform(0.9, 1.1)
            grid = F.affine_grid(torch.tensor([[scale, 0, 0], [0, scale, angle]], device=frames.device),
                                 aug_frames[:, t].unsqueeze(0).size())
            aug_frames[:, t] = F.grid_sample(aug_frames[:, t].unsqueeze(0), grid)

        if augmentations.get("color_jitter", False):
            # Simulate basic brightness adjustment
            aug_frames[:, t] = torch.clamp(aug_frames[:, t] * random.uniform(0.8, 1.2), 0, 1)

    return aug_frames


def apply_mosaic_transform(video: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create mosaic augmentation for training on small objects and multiple similar-looking entities.

    Args:
        video (torch.Tensor): Video frames tensor [B, T, C, H, W].
        mask (torch.Tensor): Segmentation masks tensor [B, T, H, W].

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Mosaic augmented video and mask.
    """
    B, T, C, H, W = video.shape
    grid_size = 2  # 2x2 tiling for now; can be extended to higher grids
    tile_h, tile_w = H // grid_size, W // grid_size

    mosaic_video = torch.zeros(B, T, C, H, W, device=video.device)
    mosaic_mask = torch.zeros(B, T, H, W, device=mask.device)

    for i in range(grid_size):
        for j in range(grid_size):
            top, left = i * tile_h, j * tile_w
            selected_idx = random.randint(0, B - 1)
            video_tile = F.interpolate(video[selected_idx:selected_idx + 1], size=(tile_h, tile_w), mode="bilinear")
            mask_tile = F.interpolate(mask[selected_idx:selected_idx + 1], size=(tile_h, tile_w), mode="nearest")

            mosaic_video[:, :, :, top:top + tile_h, left:left + tile_w] = video_tile
            mosaic_mask[:, :, top:top + tile_h, left:left + tile_w] = mask_tile

    return mosaic_video, mosaic_mask


def generate_abs_pos_encoding(size: int) -> torch.Tensor:
    """
    Generate absolute positional encoding for a given spatial size.

    Args:
        size (int): Spatial size of input frames.

    Returns:
        torch.Tensor: Positional encoding tensor [H, W, embedding_dim].
    """
    position = torch.arange(size).unsqueeze(0).repeat(size, 1)
    encoding = torch.sin(position.float() * (3.1416 * 2 / size))
    return encoding.unsqueeze(0).unsqueeze(0)  # Return in [1, 1, H, W]


def apply_rope_encoding(features: torch.Tensor, spatial_dim: Tuple[int, int]) -> torch.Tensor:
    """
    Apply 2D Rotary Positional Encoding (RoPE) for spatial dimensions.

    Args:
        features (torch.Tensor): Input features [B, C, H, W].
        spatial_dim (Tuple[int, int]): (Height, Width) dimensions.

    Returns:
        torch.Tensor: Rotationally encoded features [B, C, H, W].
    """
    H, W = spatial_dim
    positional_features = generate_abs_pos_encoding(H).repeat(1, W, 1, 1)
    encoded_features = features + positional_features.to(features.device)
    return encoded_features


def generate_prompts(mask: torch.Tensor, strategy: str = "clicks") -> torch.Tensor:
    """
    Generate simulated user prompts (clicks, boxes, masks) based on mask region.

    Args:
        mask (torch.Tensor): Ground truth mask tensor [B, H, W].
        strategy (str): Strategy for prompting ('clicks', 'boxes', 'masks').

    Returns:
        torch.Tensor: Generated prompts tensor.
    """
    B, H, W = mask.shape
    prompts = torch.zeros(B, 3)  # Placeholder: [x, y, type]
    if strategy == "clicks":
        center_h = H // 2
        center_w = W // 2
        prompts[:, 0] = center_h
        prompts[:, 1] = center_w
    return prompts


def resolve_ambiguity(predictions: List[torch.Tensor], io_u_scores: List[float]) -> torch.Tensor:
    """
    Resolve ambiguous prompts by selecting the mask with the highest IoU score.

    Args:
        predictions (List[torch.Tensor]): List of predicted masks.
        io_u_scores (List[float]): List of IoU scores for the predicted masks.

    Returns:
        torch.Tensor: Mask with the highest IoU score.
    """
    idx = torch.argmax(torch.tensor(io_u_scores))
    return predictions[idx]


def compute_t_and_f(predictions: torch.Tensor, ground_truth: torch.Tensor) -> float:
    """
    Compute True Positive (T) and False Negative (F) metric.

    Args:
        predictions (torch.Tensor): Predicted mask tensors [B, H, W].
        ground_truth (torch.Tensor): Ground truth mask tensors [B, H, W].

    Returns:
        float: T&F metric (higher is better).
    """
    true_positive = torch.sum(predictions * ground_truth).float()
    false_negative = torch.sum((1 - predictions) * ground_truth).float()
    return true_positive / (true_positive + false_negative + 1e-8)  # Avoid zero denominator


def compute_mean_iou(predictions: torch.Tensor, ground_truth: torch.Tensor) -> float:
    """
    Compute Mean IoU (Intersection-over-Union) metric.

    Args:
        predictions (torch.Tensor): Predicted mask tensors [B, H, W].
        ground_truth (torch.Tensor): Ground truth mask tensors [B, H, W].

    Returns:
        float: Mean IoU metric.
    """
    intersection = torch.sum(predictions * ground_truth).float()
    union = torch.sum((predictions + ground_truth > 0)).float()
    return intersection / (union + 1e-8)  # Prevent division by zero


def measure_fps(model, test_loader: DataLoader, device: str = "cuda") -> float:
    """
    Measure Frames-Per-Second (FPS) benchmark for real-time segmentation.

    Args:
        model (torch.nn.Module): SAM 2 model instance.
        test_loader (DataLoader): DataLoader with test dataset.
        device (str): Device to run evaluation on ('cuda' or 'cpu').

    Returns:
        float: FPS value.
    """
    model.eval()
    total_frames = 0
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()

    with torch.no_grad():
        for batch in test_loader:
            video_frames = batch['frames'].to(device)
            for t in range(video_frames.size(1)):  # Iterate over frames
                model(video_frames[:, t])  # Forward pass
                total_frames += 1

    end_time.record()
    torch.cuda.synchronize()
    elapsed_time = start_time.elapsed_time(end_time) / 1000.0  # In seconds
    return total_frames / elapsed_time


def update_memory(fifo_queue: List[torch.Tensor], new_memory: torch.Tensor, max_size: int = 6) -> List[torch.Tensor]:
    """
    Update FIFO memory queue based on new memory embeddings.

    Args:
        fifo_queue (List[torch.Tensor]): Current memory queue.
        new_memory (torch.Tensor): New memory embedding to add.
        max_size (int): Maximum size of the memory queue.

    Returns:
        List[torch.Tensor]: Updated memory queue.
    """
    if len(fifo_queue) >= max_size:
        fifo_queue.pop(0)  # Remove oldest entry
    fifo_queue.append(new_memory)
    return fifo_queue


def downsample_memory(memory: torch.Tensor, scale: int = 4) -> torch.Tensor:
    """
    Downsample memory embeddings for computational efficiency.

    Args:
        memory (torch.Tensor): Memory embeddings tensor [B, H, W].
        scale (int): Downsampling factor.

    Returns:
        torch.Tensor: Downsampled memory tensor.
    """
    return F.avg_pool2d(memory, kernel_size=scale)
