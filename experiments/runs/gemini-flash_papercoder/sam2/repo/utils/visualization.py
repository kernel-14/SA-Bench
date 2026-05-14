"""
utils/visualization.py

This module provides functions for visualizing segmentation masks,
user prompts, and internal model states for debugging and analysis.
"""

import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Optional, Dict, Any

# Ensure matplotlib is set to a non-interactive backend for server environments
plt.switch_backend('Agg')


def save_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    output_path: str,
    color: Tuple[int, int, int] = (0, 255, 0),  # Green in RGB
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlays a segmentation mask on the original image and saves the result.

    Args:
        image (np.ndarray): The original image, expected to be in HWC (Height, Width, Channels)
                            format with 3 channels (RGB). Pixel values 0-255.
        mask (np.ndarray): A binary or boolean segmentation mask (HW). Values should be 0/1 or False/True.
        output_path (str): The full path including filename and extension (e.g., '.png')
                           where the overlaid image will be saved.
        color (Tuple[int, int, int], optional): The color to use for the mask overlay in RGB.
                                                Defaults to green (0, 255, 0).
        alpha (float, optional): The transparency factor for the mask overlay.
                                 A value between 0.0 (fully transparent) and 1.0 (fully opaque).
                                 Defaults to 0.5.

    Returns:
        np.ndarray: The overlaid image in RGB format.

    Raises:
        ValueError: If image and mask dimensions are incompatible.
    """
    if image.shape[:2] != mask.shape:
        raise ValueError(
            f"Image and mask dimensions are incompatible. Image: {image.shape[:2]}, Mask: {mask.shape}"
        )
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    # Ensure mask is boolean or uint8
    mask_bool = mask.astype(bool)

    # Create a colored version of the mask
    colored_mask = np.zeros_like(image, dtype=np.uint8)
    colored_mask[mask_bool] = color

    # Blend the original image with the colored mask
    # Convert image and colored_mask to float for blending, then back to uint8
    overlaid_image = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)

    # Save the image
    # OpenCV expects BGR for imwrite, convert from RGB
    cv2.imwrite(output_path, cv2.cvtColor(overlaid_image, cv2.COLOR_RGB2BGR))
    return overlaid_image


def plot_prompts(
    image: np.ndarray,
    points: Optional[List[Tuple[int, int, int]]] = None,
    boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    masks: Optional[List[np.ndarray]] = None,
    output_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Optional[plt.Figure]:
    """
    Visualizes various types of prompts (points, bounding boxes, input masks) on an image.

    Args:
        image (np.ndarray): The original image (HWC, RGB). Pixel values 0-255.
        points (Optional[List[Tuple[int, int, int]]]): List of point prompts (x, y, label).
                                                        label=1 for positive (green), 0 for negative (red).
        boxes (Optional[List[Tuple[int, int, int, int]]]): List of bounding box prompts (x1, y1, x2, y2).
        masks (Optional[List[np.ndarray]]): List of binary input masks (HW) to be overlaid.
        output_path (Optional[str]): If provided, the plot will be saved to this path.
                                      If None, the plot will be displayed (or added to `ax`).
        ax (Optional[plt.Axes]): A matplotlib Axes object to draw on. If None, a new figure and axes will be created.

    Returns:
        Optional[plt.Figure]: The matplotlib Figure object if a new one was created, else None.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(image.shape[1] / 100, image.shape[0] / 100))
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        new_fig_created = True
    else:
        fig = ax.get_figure()
        new_fig_created = False

    ax.imshow(image)
    ax.set_axis_off()

    # Plot masks
    if masks is not None:
        for i, input_mask in enumerate(masks):
            if input_mask is not None:
                # Create a semi-transparent overlay for the mask
                # Use a consistent color or cycle through colors if multiple masks
                mask_color = np.array([0, 0.5, 1])  # Semi-transparent blue in RGB
                mask_rgba = np.zeros((*input_mask.shape, 4), dtype=float)
                mask_rgba[input_mask.astype(bool)] = [*mask_color, 0.3] # RGB and alpha
                ax.imshow(mask_rgba)

    # Plot points
    if points is not None:
        for x, y, label in points:
            color = 'lightgreen' if label == 1 else 'red'
            marker = 'o'
            ax.plot(x, y, marker, color=color, markersize=10, markeredgewidth=1, markeredgecolor='black')

    # Plot boxes
    if boxes is not None:
        for x1, y1, x2, y2 in boxes:
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor='blue',
                facecolor='none',
                linestyle='--',
            )
            ax.add_patch(rect)

    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        if new_fig_created:
            plt.close(fig)
        return None
    elif new_fig_created:
        # If no output path and a new figure was created, we return it to the caller
        # The caller is responsible for plt.show() or plt.close() if they desire
        return fig
    return None


def plot_memory_state(
    memory_features: List[torch.Tensor],
    object_pointers: List[torch.Tensor],
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """
    Provides a visual representation of the internal memory states
    (feature maps and object pointers) of the SAM 2 model.

    Args:
        memory_features (List[torch.Tensor]): A list of tensors, each (C, H_mem, W_mem)
                                               representing a spatial memory feature map.
        object_pointers (List[torch.Tensor]): A list of tensors, each (D_op,)
                                               representing an object pointer vector.
        output_path (Optional[str]): If provided, the plot will be saved to this path.
                                      If None, the plot will be displayed.

    Returns:
        Optional[plt.Figure]: The matplotlib Figure object if a new one was created, else None.
    """
    if not memory_features and not object_pointers:
        print("No memory features or object pointers to plot.")
        return None

    num_mem_features = len(memory_features)
    num_obj_pointers = len(object_pointers)

    # Determine optimal subplot grid
    # A simple layout: memory features in one row, object pointers below
    num_cols_mem = max(1, num_mem_features)
    num_cols_ptr = max(1, num_obj_pointers)
    
    # Heuristic for figure size:
    fig_height = 4 + (2 if num_mem_features > 0 else 0) + (2 if num_obj_pointers > 0 else 0)
    fig_width = max(num_cols_mem, num_cols_ptr) * 3

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(
        nrows=(1 if num_mem_features > 0 else 0) + (1 if num_obj_pointers > 0 else 0),
        ncols=max(num_cols_mem, num_cols_ptr),
        height_ratios=[1] * ((1 if num_mem_features > 0 else 0) + (1 if num_obj_pointers > 0 else 0))
    )
    
    row_idx = 0

    # Visualize Memory Features
    if num_mem_features > 0:
        for i, mem_feat in enumerate(memory_features):
            ax = fig.add_subplot(gs[row_idx, i % num_cols_mem])
            # Convert to numpy and reduce dimensionality for visualization (mean across channels)
            mem_feat_np = mem_feat.detach().cpu().float().numpy()
            if mem_feat_np.ndim == 3: # (C, H, W)
                vis_data = mem_feat_np.mean(axis=0) # Mean across channels
            elif mem_feat_np.ndim == 2: # (H, W) if it's already reduced
                vis_data = mem_feat_np
            else:
                print(f"Warning: Unexpected memory feature dimension {mem_feat_np.ndim} for visualization.")
                vis_data = np.zeros((64, 64)) # Placeholder
            
            ax.imshow(vis_data, cmap='viridis')
            ax.set_title(f'Mem Feat {i+1}')
            ax.axis('off')
        row_idx += 1

    # Visualize Object Pointers
    if num_obj_pointers > 0:
        if num_obj_pointers == 1:
            ax = fig.add_subplot(gs[row_idx, :]) # Span full width
            obj_ptr_np = object_pointers[0].detach().cpu().float().numpy()
            ax.plot(obj_ptr_np)
            ax.set_title('Object Pointer 1')
            ax.set_xlabel('Dimension Index')
            ax.set_ylabel('Value')
        else:
            for i, obj_ptr in enumerate(object_pointers):
                ax = fig.add_subplot(gs[row_idx, i % num_cols_ptr])
                obj_ptr_np = obj_ptr.detach().cpu().float().numpy()
                ax.plot(obj_ptr_np)
                ax.set_title(f'Obj Ptr {i+1}')
                ax.axis('off') # Hide axis for smaller plots
        row_idx += 1

    fig.suptitle('SAM 2 Memory State Visualization', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle

    if output_path is not None:
        plt.savefig(output_path)
        plt.close(fig)
        return None
    else:
        # The caller is responsible for plt.show() or plt.close() if they desire
        return fig


def visualize_video_segmentation(
    video_frames: List[np.ndarray],
    predicted_masklets: List[np.ndarray],
    output_dir: str,
    filename_prefix: str,
    as_video: bool = True,
    fps: int = 10,
    mask_color: Tuple[int, int, int] = (0, 255, 0),  # Green in RGB
    mask_alpha: float = 0.5,
) -> None:
    """
    Creates a visual output (either a sequence of images or a video file)
    of a model's segmentation predictions across an entire video sequence.

    Args:
        video_frames (List[np.ndarray]): A list of original video frames,
                                         each a HWC (RGB) numpy array.
        predicted_masklets (List[np.ndarray]): A list of binary masklets,
                                                each a HW numpy array,
                                                corresponding to `video_frames`.
        output_dir (str): The directory where the outputs (images or video) will be saved.
        filename_prefix (str): A prefix for output filenames (e.g., "frame_0001.png" or "output_video.mp4").
        as_video (bool, optional): If True, save as a single video file. If False, save as individual images.
                                   Defaults to True.
        fps (int, optional): Frames per second for the output video. Defaults to 10.
        mask_color (Tuple[int, int, int], optional): RGB color for the overlaid mask. Defaults to green.
        mask_alpha (float, optional): Transparency for the overlaid mask. Defaults to 0.5.

    Raises:
        ValueError: If the number of video frames and masklets do not match.
    """
    if len(video_frames) != len(predicted_masklets):
        raise ValueError(
            f"Number of video frames ({len(video_frames)}) must match "
            f"number of predicted masklets ({len(predicted_masklets)})."
        )

    os.makedirs(output_dir, exist_ok=True)

    overlaid_frames: List[np.ndarray] = []
    for i, (frame, masklet) in enumerate(zip(video_frames, predicted_masklets)):
        frame_output_path = os.path.join(output_dir, f"{filename_prefix}_{i:04d}.png")
        # save_mask_overlay internally saves a BGR image, but returns an RGB image
        overlaid_img = save_mask_overlay(frame, masklet, frame_output_path, mask_color, mask_alpha)
        overlaid_frames.append(overlaid_img)
    
    if as_video and overlaid_frames:
        height, width, _ = overlaid_frames[0].shape
        video_path = os.path.join(output_dir, f"{filename_prefix}.mp4")

        # Use XVID for wider compatibility or MP4V if XVID doesn't work.
        # FourCC code for XVID is 'XVID'
        # FourCC code for H.264 (AVC) is 'mp4v' for MPEG-4, or 'avc1' for H.264 (may require specific codecs)
        # Choosing 'mp4v' or 'XVID' as common choices.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        if not out.isOpened():
            print(f"Error: Could not open video writer for {video_path}. Check FourCC code or file path.")
            # Fallback to saving individual frames if video writer fails
            print("Falling back to saving individual frames only.")
            return

        for img in overlaid_frames:
            # cv2.VideoWriter expects BGR, convert from RGB
            out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"Video saved to: {video_path}")
    elif not as_video:
        print(f"Individual frames saved to: {output_dir}/{filename_prefix}_*.png")
    else:
        print("No frames to save for video segmentation visualization.")

