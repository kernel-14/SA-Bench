```python
## utils/visualization.py
"""Visualization utilities for SAM 2 reproduction.

This module provides qualitative visualization tools for debugging and result
inspection throughout the SAM 2 reproduction pipeline. It serves three
distinct use cases:

1. Training debugging — visualizing prompt simulation, mask predictions vs.
   ground truth during training.
2. Evaluation inspection — overlaying predicted masklets on video frames to
   inspect J&F quality.
3. Dataset verification — confirming SA-V masklet annotations are loaded
   correctly.

All functions are stateless and operate on numpy arrays or PyTorch tensors.
No imports from other project modules are needed — this keeps visualization
fully decoupled and usable at any stage.

Config references:
    sav_dataset.annotation_fps: 6  → default fps=6 in save_video_with_masklets
    model.mask_threshold: 0.0      → threshold in _to_numpy_masks
    Figure 4, Figure 11: "each masklet has a unique color"
"""

import colorsys
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# Frames: Tensor[T, C, H, W] or ndarray[T, H, W, C] or ndarray[T, C, H, W]
FrameInput = Union[torch.Tensor, np.ndarray]

# Masks: Tensor[T, N, H, W] or ndarray[T, N, H, W], binary float or bool
MaskInput = Union[torch.Tensor, np.ndarray]

# Clicks: Tensor[K, 2] or ndarray[K, 2] in (x, y) pixel coordinates
ClickInput = Union[torch.Tensor, np.ndarray]

# Click labels: Tensor[K] or ndarray[K], 1=positive, 0=negative
LabelInput = Union[torch.Tensor, np.ndarray]

# Per-frame clicks: dict mapping frame_idx -> (clicks_array, labels_array)
ClickDict = Dict[int, Tuple[np.ndarray, np.ndarray]]

# ---------------------------------------------------------------------------
# Internal conversion utilities
# ---------------------------------------------------------------------------


def _to_numpy_frames(frames: FrameInput) -> np.ndarray:
    """Convert any frame input to ndarray[T, H, W, 3] uint8.

    Handles:
    - torch.Tensor of shape [T, C, H, W] or [C, H, W] (single frame)
    - numpy.ndarray of shape [T, H, W, C] or [T, C, H, W] or [H, W, C]
    - Float inputs in [0, 1]: multiplied by 255 and clipped
    - uint8 inputs: returned as-is after shape normalization

    Args:
        frames: Video frames in any supported format.

    Returns:
        ndarray of shape [T, H, W, 3], dtype uint8.

    Raises:
        ValueError: If the input cannot be interpreted as RGB frames.
    """
    if isinstance(frames, torch.Tensor):
        arr = frames.detach().cpu().float().numpy()
    else:
        arr = np.asarray(frames, dtype=np.float32)

    # Handle single frame (no T dimension): add batch dim
    if arr.ndim == 3:
        arr = arr[np.newaxis]  # [1, H, W, C] or [1, C, H, W]

    if arr.ndim != 4:
        raise ValueError(
            f"Expected 3D or 4D frame input, got shape {arr.shape}."
        )

    T = arr.shape[0]

    # Detect channel layout: [T, C, H, W] vs [T, H, W, C]
    # Heuristic: if dim 1 is 1 or 3 and dim 3 is not 1 or 3, assume CHW
    if arr.shape[1] in (1, 3) and arr.shape[3] not in (1, 3):
        # [T, C, H, W] → [T, H, W, C]
        arr = arr.transpose(0, 2, 3, 1)

    # Ensure 3 channels (handle grayscale by repeating)
    if arr.shape[3] == 1:
        arr = np.repeat(arr, 3, axis=3)
    elif arr.shape[3] != 3:
        raise ValueError(
            f"Expected 1 or 3 channels in last dimension, got {arr.shape[3]}."
        )

    # Normalize to uint8
    if arr.dtype != np.uint8:
        # Float in [0, 1] → scale to [0, 255]
        if arr.max() <= 1.0 + 1e-6:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr  # [T, H, W, 3] uint8


def _to_numpy_masks(
    masks: MaskInput,
    mask_threshold: float = 0.0,
) -> np.ndarray:
    """Convert mask input to ndarray[T, N, H, W] bool.

    Handles:
    - torch.Tensor of shape [T, N, H, W], [T, H, W], [N, H, W], or [H, W]
    - numpy.ndarray of the same shapes
    - Float logits: thresholded at mask_threshold (default 0.0 per config)
    - Float probabilities in [0, 1]: thresholded at 0.5
    - Bool arrays: passed through directly

    Config reference: model.mask_threshold: 0.0

    Args:
        masks: Segmentation masks in any supported format.
        mask_threshold: Threshold for converting logits/probabilities to binary.
            Use 0.0 for raw logits (paper default), 0.5 for probabilities.

    Returns:
        ndarray of shape [T, N, H, W], dtype bool.
    """
    if isinstance(masks, torch.Tensor):
        arr = masks.detach().cpu().float().numpy()
    else:
        arr = np.asarray(masks, dtype=np.float32)

    # Normalize dimensions to [T, N, H, W]
    if arr.ndim == 2:
        # [H, W] → [1, 1, H, W]
        arr = arr[np.newaxis, np.newaxis]
    elif arr.ndim == 3:
        # Ambiguous: could be [T, H, W] (single object) or [N, H, W]
        # Treat as [T, H, W] → [T, 1, H, W]
        arr = arr[:, np.newaxis]
    elif arr.ndim == 4:
        pass  # Already [T, N, H, W]
    else:
        raise ValueError(
            f"Expected 2D, 3D, or 4D mask input, got shape {arr.shape}."
        )

    # Convert to binary bool
    if arr.dtype == bool:
        return arr

    # Determine if values are logits or probabilities
    # Logits can be negative; probabilities are in [0, 1]
    if arr.min() < -0.5 or arr.max() > 1.5:
        # Likely raw logits — apply sigmoid then threshold at 0.5
        arr = 1.0 / (1.0 + np.exp(-arr))
        return arr >= 0.5
    else:
        # Probabilities or already binary — threshold at mask_threshold
        return arr >= (0.5 if mask_threshold == 0.0 else mask_threshold)


def _generate_color_palette(n_colors: int) -> List[Tuple[int, int, int]]:
    """Generate n_colors visually distinct RGB colors using HSV color space.

    Uses evenly spaced hues with fixed saturation=0.8 and value=0.9 to
    produce bright, distinguishable colors. This matches the paper's
    visualization style where "each masklet has a unique color" (Figure 4,
    Figure 11).

    Args:
        n_colors: Number of distinct colors to generate. Must be >= 1.

    Returns:
        List of (R, G, B) tuples with values in [0, 255].
    """
    if n_colors <= 0:
        return []

    colors: List[Tuple[int, int, int]] = []
    for i in range(n_colors):
        hue = i / n_colors
        saturation = 0.8
        value = 0.9
        r_f, g_f, b_f = colorsys.hsv_to_rgb(hue, saturation, value)
        r = int(round(r_f * 255))
        g = int(round(g_f * 255))
        b = int(round(b_f * 255))
        colors.append((r, g, b))

    return colors


def _normalize_clicks_to_dict(
    clicks: Optional[Union[ClickInput, ClickDict]],
    click_labels: Optional[LabelInput],
    num_frames: int,
) -> ClickDict:
    """Normalize click inputs to a dict mapping frame_idx -> (coords, labels).

    Handles two input formats:
    1. A single array of clicks (applied to frame 0 only)
    2. A dict mapping frame_idx -> (clicks_array, labels_array)

    Args:
        clicks: Click coordinates as array [K, 2] or dict.
        click_labels: Click labels as array [K] (only used when clicks is array).
        num_frames: Total number of frames (for validation).

    Returns:
        Dict mapping frame_idx (int) to (coords_ndarray[K, 2], labels_ndarray[K]).
        Empty dict if clicks is None.
    """
    if clicks is None:
        return {}

    if isinstance(clicks, dict):
        # Already in dict format — normalize arrays inside
        result: ClickDict = {}
        for frame_idx, (c, l) in clicks.items():
            if isinstance(c, torch.Tensor):
                c = c.detach().cpu().numpy()
            if isinstance(l, torch.Tensor):
                l = l.detach().cpu().numpy()
            result[int(frame_idx)] = (
                np.asarray(c, dtype=np.float32),
                np.asarray(l, dtype=np.int32),
            )
        return result

    # Single array of clicks — apply to frame 0
    if isinstance(clicks, torch.Tensor):
        clicks_np = clicks.detach().cpu().numpy()
    else:
        clicks_np = np.asarray(clicks, dtype=np.float32)

    if click_labels is not None:
        if isinstance(click_labels, torch.Tensor):
            labels_np = click_labels.detach().cpu().numpy()
        else:
            labels_np = np.asarray(click_labels, dtype=np.int32)
    else:
        # Default all clicks to positive
        labels_np = np.ones(len(clicks_np), dtype=np.int32)

    return {0: (clicks_np, labels_np)}


def _draw_score_overlay(
    frame: np.ndarray,
    iou_scores: Optional[np.ndarray],
    occlusion_scores: Optional[np.ndarray],
) -> np.ndarray:
    """Draw IoU and occlusion score text overlay on a single frame.

    Renders scores as text in the top-left corner with a semi-transparent
    dark background rectangle for readability. Stacks one line per object
    for multi-object scenarios.

    Args:
        frame: Single frame as ndarray[H, W, 3] uint8.
        iou_scores: Per-object IoU scores as ndarray[N] float, or None.
        occlusion_scores: Per-object occlusion scores as ndarray[N] float, or None.

    Returns:
        Frame with score text overlaid, ndarray[H, W, 3] uint8.
    """
    result = frame.copy()
    h, w = result.shape[:2]

    # Determine number of objects
    n_objects = 0
    if iou_scores is not None:
        n_objects = max(n_objects, len(iou_scores))
    if occlusion_scores is not None:
        n_objects = max(n_objects, len(occlusion_scores))

    if n_objects == 0:
        return result

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    font_thickness = 1
    line_height = 18
    padding = 4

    lines: List[str] = []
    for obj_idx in range(n_objects):
        parts: List[str] = [f"Obj {obj_idx}:"]
        if iou_scores is not None and obj_idx < len(iou_scores):
            parts.append(f"IoU={iou_scores[obj_idx]:.3f}")
        if occlusion_scores is not None and obj_idx < len(occlusion_scores):
            parts.append(f"Occ={occlusion_scores[obj_idx]:.3f}")
        lines.append(" ".join(parts))

    # Compute background rectangle size
    max_text_width = 0
    for line in lines:
        (text_w, _), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
        max_text_width = max(max_text_width, text_w)

    rect_h = len(lines) * line_height + 2 * padding
    rect_w = max_text_width + 2 * padding

    # Draw semi-transparent background
    overlay = result.copy()
    cv2.rectangle(overlay, (0, 0), (rect_w, rect_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, result, 0.4, 0, result)

    # Draw text lines
    for i, line in enumerate(lines):
        y_pos = padding + (i + 1) * line_height - 4
        cv2.putText(
            result,
            line,
            (padding, y_pos),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def overlay_masks_on_frames(
    frames: FrameInput,
    masks: MaskInput,
    alpha: float = 0.5,
    colors: Optional[List[Tuple[int, int, int]]] = None,
) -> np.ndarray:
    """Blend segmentation masks onto video frames with distinct per-object colors.

    This is the core visualization primitive used by all other functions.
    Each object receives a unique color from a deterministic palette, matching
    the paper's visualization style where "each masklet has a unique color"
    (Figure 4, Figure 11 of the SAM 2 paper).

    The alpha blending is applied only within the mask region — outside the
    mask, original frame pixels are preserved exactly.

    Args:
        frames: Video frames. Accepted formats:
            - torch.Tensor of shape [T, C, H, W] or [C, H, W]
            - numpy.ndarray of shape [T, H, W, C] or [T, C, H, W] or [H, W, C]
            - Float in [0, 1] or uint8 in [0, 255]
        masks: Segmentation masks. Accepted formats:
            - torch.Tensor or numpy.ndarray of shape [T, N, H, W]
            - Single-object: [T, H, W] (N dimension added automatically)
            - Binary bool, float probabilities, or raw logits
        alpha: Blending factor for mask overlay. 0.0 = no overlay, 1.0 = solid
            color. Defaults to 0.5 for semi-transparent visualization.
        colors: Optional list of (R, G, B) tuples for each object. If None,
            a deterministic palette is generated automatically. Length must
            be >= N (number of objects).

    Returns:
        ndarray of shape [T, H, W, 3], dtype uint8, with masks blended in.

    Raises:
        ValueError: If frames and masks have different numbers of frames (T).
    """
    # Normalize inputs
    frames_np = _to_numpy_frames(frames)   # [T, H, W, 3] uint8
    masks_np = _to_numpy_masks(masks)      # [T, N, H, W] bool

    T_frames = frames_np.shape[0]
    T_masks = masks_np.shape[0]

    if T_frames != T_masks:
        raise ValueError(
            f"frames has {T_frames} frames but masks has {T_masks} frames. "
            "They must match."
        )

    T, H, W, _ = frames_np.shape
    N = masks_np.shape[1]

    # Generate or validate color palette
    if colors is None:
        palette = _generate_color_palette(max(N, 20))
    else:
        if len(colors) < N:
            # Extend with auto-generated colors if not enough provided
            extra = _generate_color_palette(N - len(colors))
            palette = list(colors) + extra
        else:
            palette = list(colors)

    # Allocate output array
    result = frames_np.copy()  # [T, H, W, 3] uint8

    for t in range(T):
        frame = result[t].astype(np.float32)  # [H, W, 3]

        for n in range(N):
            mask = masks_np[t, n]  # [H, W] bool

            if not mask.any():
                # Empty mask — skip blending for this object on this frame
                continue

            color = palette[n % len(palette)]
            color_arr = np.array(color, dtype=np.float32)  # [3] RGB

            # Alpha blend only within the mask region
            # result[mask] = (1 - alpha) * frame[mask] + alpha * color
            frame[mask] = (
                (1.0 - alpha) * frame[mask] + alpha * color_arr
            )

        result[t] = np.clip(frame, 0, 255).astype(np.uint8)

    return result  # [T, H, W, 3] uint8


def draw_clicks(
    frames: FrameInput,
    clicks: Union[ClickInput, ClickDict],
    click_labels: Optional[LabelInput] = None,
    radius: int = 8,
    positive_color: Tuple[int, int, int] = (0, 255, 0),
    negative_color: Tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """Draw click prompts on frames as colored circles.

    Green circles for positive clicks, red circles for negative clicks,
    matching the paper's convention: "Green/red dots indicate positive/negative
    prompts respectively" (Section 3, Figure 2 of the SAM 2 paper).

    Each click is drawn as a filled colored circle with a white border ring
    for visibility against any background.

    Args:
        frames: Video frames. Accepted formats same as overlay_masks_on_frames.
        clicks: Click coordinates in one of two formats:
            1. Array of shape [K, 2] in (x, y) pixel coordinates — applied
               to frame 0 only.
            2. Dict mapping frame_idx (int) to (coords_array[K, 2],
               labels_array[K]) for multi-frame click visualization.
        click_labels: Click labels as array [K] with values 1 (positive) or
            0 (negative). Only used when clicks is an array (format 1).
            Defaults to all-positive if None.
        radius: Radius of the click circle in pixels. Defaults to 8.
        positive_color: RGB color for positive clicks. Defaults to green
            (0, 255, 0) matching Figure 2 of the paper.
        negative_color: RGB color for negative clicks. Defaults to red
            (255, 0, 0) matching Figure 2 of the paper.

    Returns:
        ndarray of shape [T, H, W, 3], dtype uint8, with clicks drawn.
    """
    frames_np = _to_numpy_frames(frames)  # [T, H, W, 3] uint8
    T, H, W, _ = frames_np.shape

    # Normalize clicks to dict format
    click_dict = _normalize_clicks_to_dict(clicks, click_labels, T)

    result = frames_np.copy()

    for frame_idx, (coords, labels) in click_dict.items():
        if frame_idx < 0 or frame_idx >= T:
            logger.warning(
                "Click frame_idx %d is out of range [0, %d). Skipping.",
                frame_idx,
                T,
            )
            continue

        frame_bgr = cv2.cvtColor(result[frame_idx], cv2.COLOR_RGB2BGR)

        for k in range(len(coords)):
            x = int(round(float(coords[k, 0])))
            y = int(round(float(coords[k, 1])))
            label = int(labels[k])

            # Clip to valid image bounds
            x = max(0, min(x, W - 1))
            y = max(0, min(y, H - 1))

            # Choose color based on label
            if label == 1:
                color_bgr = (
                    positive_color[2],
                    positive_color[1],
                    positive_color[0],
                )  # RGB → BGR
            else:
                color_bgr = (
                    negative_color[2],
                    negative_color[1],
                    negative_color[0],
                )  # RGB → BGR

            # Draw white border ring for visibility
            cv2.circle(frame_bgr, (x, y), radius + 2, (255, 255, 255), 2)
            # Draw filled colored circle
            cv2.circle(frame_bgr, (x, y), radius, color_bgr, -1)

        result[frame_idx] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    return result  # [T, H, W, 3] uint8


def save_video_with_masklets(
    frames: FrameInput,
    masks: MaskInput,
    output_path: str,
    fps: int = 6,
    clicks: Optional[Union[ClickInput, ClickDict]] = None,
    click_labels: Optional[LabelInput] = None,
    iou_scores: Optional[Union[torch.Tensor, np.ndarray]] = None,
    occlusion_scores: Optional[Union[torch.Tensor, np.ndarray]] = None,
) -> None:
    """Compose and save a visualization video with overlaid masklets.

    Combines mask overlay, optional click annotations, and optional score
    text overlays into a complete visualization. Supports MP4, AVI, GIF,
    and directory (individual PNG frames) output formats.

    The default fps=6 matches SA-V's annotation rate: "annotated at 6 FPS"
    (Section 5.1 of the paper). Config reference: sav_dataset.annotation_fps: 6

    Args:
        frames: Video frames. Accepted formats same as overlay_masks_on_frames.
        masks: Segmentation masks. Accepted formats same as overlay_masks_on_frames.
        output_path: Output file path. Supported formats:
            - ".mp4": H.264 video (recommended)
            - ".avi": AVI video
            - ".gif": Animated GIF
            - Directory path (no extension): saves individual PNG frames
        fps: Frames per second for video output. Defaults to 6 (SA-V rate).
        clicks: Optional click annotations. Same format as draw_clicks().
        click_labels: Optional click labels. Same format as draw_clicks().
        iou_scores: Optional per-object IoU scores as Tensor[N] or ndarray[N].
            Displayed as text overlay on each frame.
        occlusion_scores: Optional per-object occlusion scores as Tensor[N]
            or ndarray[N]. Displayed as text overlay on each frame.

    Returns:
        None. Side effect is the saved file or directory.
    """
    # Step 1: Compose visualization
    overlaid = overlay_masks_on_frames(frames, masks)  # [T, H, W, 3] uint8

    # Step 2: Draw clicks if provided
    if clicks is not None:
        overlaid = draw_clicks(overlaid, clicks, click_labels)

    # Step 3: Normalize score arrays
    iou_np: Optional[np.ndarray] = None
    occ_np: Optional[np.ndarray] = None

    if iou_scores is not None:
        if isinstance(iou_scores, torch.Tensor):
            iou_np = iou_scores.detach().cpu().float().numpy()
        else:
            iou_np = np.asarray(iou_scores, dtype=np.float32)

    if occlusion_scores is not None:
        if isinstance(occlusion_scores, torch.Tensor):
            occ_np = occlusion_scores.detach().cpu().float().numpy()
        else:
            occ_np = np.asarray(occlusion_scores, dtype=np.float32)

    # Step 4: Draw score overlays per frame
    if iou_np is not None or occ_np is not None:
        T = overlaid.shape[0]
        for t in range(T):
            overlaid[t] = _draw_score_overlay(overlaid[t], iou_np, occ_np)

    # Step 5: Create output directory
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    T, H, W, _ = overlaid.shape
    suffix = output_path_obj.suffix.lower()

    # Step 6: Write output in the appropriate format
    if suffix in (".mp4", ".avi"):
        _write_video(overlaid, output_path, fps=fps, suffix=suffix)
    elif suffix == ".gif":
        _write_gif(overlaid, output_path, fps=fps)
    else:
        # No extension or unknown extension → save as directory of PNG frames
        _write_frames_to_dir(overlaid, output_path)


def _write_video(
    frames_rgb: np.ndarray,
    output_path: str,
    fps: int,
    suffix: str,
) -> None:
    """Write frames to a video file using OpenCV VideoWriter.

    Args:
        frames_rgb: ndarray[T, H, W, 3] uint8 in RGB order.
        output_path: Output file path.
        fps: Frames per second.
        suffix: File extension (".mp4" or ".avi").
    """
    T, H, W, _ = frames_rgb.shape

    # Select codec
    if suffix == ".mp4":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")

    writer = cv2.VideoWriter(output_path, fourcc, float(fps), (W, H))

    if not writer.isOpened():
        # Fallback codec
        logger.warning(
            "Primary codec failed for %s. Falling back to XVID.", output_path
        )
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        fallback_path = str(Path(output_path).with_suffix(".avi"))
        writer = cv2.VideoWriter(fallback_path, fourcc, float(fps), (W, H))
        output_path = fallback_path

    for t in range(T):
        # OpenCV expects BGR
        frame_bgr = cv2.cvtColor(frames_rgb[t], cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()
    logger.info("Saved video to %s (%d frames @ %d fps)", output_path, T, fps)


def _write_gif(
    frames_rgb: np.ndarray,
    output_path: str,
    fps: int,
) -> None:
    """Write frames to an animated GIF using PIL.

    Args:
        frames_rgb: ndarray[T, H, W, 3] uint8 in RGB order.
        output_path: Output .gif file path.
        fps: Frames per second (converted to duration in ms per frame).
    """
    T = frames_rgb.shape[0]
    duration_ms = int(1000 / max(fps, 1))

    pil_frames = [Image.fromarray(frames_rgb[t]) for t in range(T)]

    if len(pil_frames) == 0:
        logger.warning("No frames to write to GIF: %s", output_path)
        return

    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,
        duration=duration_ms,
        optimize=False,
    )
    logger.info("Saved GIF to %s (%d frames @ %d fps)", output_path, T, fps)


def _write_frames_to_dir(
    frames_rgb: np.ndarray,
    output_dir: str,
) -> None:
    """Save individual frames as PNG files in a directory.

    Files are named frame_00000.png, frame_00001.png, etc.

    Args:
        frames_rgb: ndarray[T, H, W, 3] uint8 in RGB order.
        output_dir: Directory path to save frames into.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    T = frames_rgb.shape[0]

    for t in range(T):
        frame_path = os.path.join(output_dir, f"frame_{t:05d}.png")
        # PIL saves in RGB order directly
        Image.fromarray(frames_rgb[t]).save(frame_path)

    logger.info("Saved %d frames to directory %s", T, output_dir)


def visualize_training_sample(
    frames: FrameInput,
    gt_masks: MaskInput,
    pred_masks: MaskInput,
    prompts: Optional[Union[ClickInput, ClickDict]],
    output_path: str,
    prompt_labels: Optional[LabelInput] = None,
) -> None:
    """Create a side-by-side GT vs. prediction visualization for training debugging.

    Shows GT masks (left half) and predicted masks (right half) with click
    prompts overlaid on both. Useful for verifying prompt simulation and
    loss computation during training.

    Args:
        frames: Video frames. Accepted formats same as overlay_masks_on_frames.
        gt_masks: Ground-truth segmentation masks.
        pred_masks: Predicted segmentation masks from the model.
        prompts: Optional click prompts to overlay. Same format as draw_clicks().
        output_path: Output file path. Supports .png, .jpg, .pdf.
        prompt_labels: Optional click labels for the prompts array.

    Returns:
        None. Side effect is the saved image file.
    """
    # Compose GT visualization
    gt_overlaid = overlay_masks_on_frames(frames, gt_masks)    # [T, H, W, 3]
    pred_overlaid = overlay_masks_on_frames(frames, pred_masks)  # [T, H, W, 3]

    # Draw clicks on both
    if prompts is not None:
        gt_overlaid = draw_clicks(gt_overlaid, prompts, prompt_labels)
        pred_overlaid = draw_clicks(pred_overlaid, prompts, prompt_labels)

    T, H, W, _ = gt_overlaid.shape

    #