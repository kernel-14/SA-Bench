"""
Automatic masklet generation for SA-V dataset.

From Section 5.1 (Auto masklet generation):
- Prompt SAM 2 with a regular grid of points in the first frame
- Generate candidate masklets
- Apply post-processing: remove tiny disconnected components, fill holes
- Send to verification step for filtering

Grid configurations (from Appendix E.1):
- 32x32 grid on first frame
- 16x16 grid on 4 zoomed crops (2x2 overlapped window)
- 4x4 grid on 16 zoomed crops (4x4 overlapped window)

Post-processing:
- Remove disconnected components with area < 200 pixels
- Fill holes with area < 200 pixels
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def generate_grid_points(
    height: int,
    width: int,
    grid_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a regular grid of points for automatic masklet generation.

    Args:
        height: image height
        width: image width
        grid_size: number of points per dimension

    Returns:
        coords: [N, 2] array of (x, y) coordinates
        labels: [N] array of labels (all 1 = positive)
    """
    y_coords = np.linspace(0, height - 1, grid_size + 2)[1:-1]
    x_coords = np.linspace(0, width - 1, grid_size + 2)[1:-1]

    xx, yy = np.meshgrid(x_coords, y_coords)
    coords = np.stack([xx.flatten(), yy.flatten()], axis=1)
    labels = np.ones(len(coords), dtype=np.int64)

    return coords, labels


def remove_small_components(mask: np.ndarray, min_area: int = 200) -> np.ndarray:
    """
    Remove disconnected components with area smaller than min_area pixels.

    Args:
        mask: binary mask
        min_area: minimum component area in pixels

    Returns:
        cleaned mask
    """
    from scipy import ndimage

    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return mask

    component_sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    small_components = np.where(np.array(component_sizes) < min_area)[0] + 1

    cleaned = mask.copy()
    for comp_id in small_components:
        cleaned[labeled == comp_id] = 0

    return cleaned


def fill_small_holes(mask: np.ndarray, max_hole_area: int = 200) -> np.ndarray:
    """
    Fill holes in segmentation masks if the hole area is less than max_hole_area pixels.

    Args:
        mask: binary mask
        max_hole_area: maximum hole area to fill

    Returns:
        mask with small holes filled
    """
    from scipy import ndimage

    # Invert mask to find holes
    inverted = 1 - mask

    # Label connected components in inverted mask
    labeled, num_features = ndimage.label(inverted)
    if num_features == 0:
        return mask

    # Find components that are holes (not connected to border)
    border_labels = set()
    border_labels.update(labeled[0, :].tolist())
    border_labels.update(labeled[-1, :].tolist())
    border_labels.update(labeled[:, 0].tolist())
    border_labels.update(labeled[:, -1].tolist())
    border_labels.discard(0)

    filled = mask.copy()
    for comp_id in range(1, num_features + 1):
        if comp_id not in border_labels:
            hole_area = (labeled == comp_id).sum()
            if hole_area < max_hole_area:
                filled[labeled == comp_id] = 1

    return filled


def postprocess_mask(mask: np.ndarray, min_area: int = 200) -> np.ndarray:
    """
    Apply post-processing to a segmentation mask.

    From Appendix E.1:
    1. Remove tiny disconnected components with areas smaller than 200 pixels
    2. Fill in holes in segmentation masks if the area of the hole is less than 200 pixels

    Args:
        mask: binary mask
        min_area: minimum area threshold

    Returns:
        post-processed mask
    """
    mask = remove_small_components(mask, min_area=min_area)
    mask = fill_small_holes(mask, max_hole_area=min_area)
    return mask


class AutoMaskletGenerator:
    """
    Generates automatic masklets for SA-V dataset creation.

    Uses SAM 2 with grid prompts to generate candidate masklets,
    then applies post-processing.
    """

    def __init__(
        self,
        model,
        device: torch.device,
        image_size: int = 1024,
        min_mask_area: int = 200,
        stability_score_threshold: float = 0.85,
        iou_threshold: float = 0.7,
    ):
        self.model = model
        self.device = device
        self.image_size = image_size
        self.min_mask_area = min_mask_area
        self.stability_score_threshold = stability_score_threshold
        self.iou_threshold = iou_threshold

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for model input."""
        from PIL import Image
        from torchvision.transforms import functional as TF

        img = Image.fromarray(frame)
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img_tensor = TF.to_tensor(img)
        img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return img_tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def generate_masks_for_frame(
        self,
        frame: np.ndarray,
        grid_size: int = 32,
    ) -> List[Dict]:
        """
        Generate masks for a single frame using grid prompts.

        Args:
            frame: [H, W, 3] RGB frame
            grid_size: number of grid points per dimension

        Returns:
            list of mask dicts with 'mask', 'iou_pred', 'stability_score'
        """
        H, W = frame.shape[:2]
        frame_tensor = self._preprocess_frame(frame)

        # Generate grid points
        coords, labels = generate_grid_points(H, W, grid_size)

        # Scale coordinates to image_size
        scale_x = self.image_size / W
        scale_y = self.image_size / H
        coords_scaled = coords.copy()
        coords_scaled[:, 0] *= scale_x
        coords_scaled[:, 1] *= scale_y

        all_masks = []

        # Process points in batches
        batch_size = 64
        for i in range(0, len(coords_scaled), batch_size):
            batch_coords = coords_scaled[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]

            # Process each point individually
            for j in range(len(batch_coords)):
                coord = torch.tensor([[batch_coords[j]]], dtype=torch.float32, device=self.device)
                label = torch.tensor([[batch_labels[j]]], dtype=torch.long, device=self.device)

                outputs = self.model.forward_image(
                    img=frame_tensor,
                    points=(coord, label),
                    multimask_output=True,
                )

                pred_masks = outputs['masks']  # [1, num_masks, H, W]
                iou_pred = outputs['iou_predictions']  # [1, num_masks]

                # Select best mask
                best_idx = iou_pred[0].argmax().item()
                best_mask = pred_masks[0, best_idx]
                best_iou = iou_pred[0, best_idx].item()

                # Convert to binary mask
                mask_binary = (best_mask > 0).cpu().numpy().astype(np.uint8)

                # Resize to original frame size
                mask_resized = F.interpolate(
                    torch.from_numpy(mask_binary).float().unsqueeze(0).unsqueeze(0),
                    size=(H, W),
                    mode='nearest',
                ).squeeze().numpy().astype(np.uint8)

                # Apply post-processing
                mask_processed = postprocess_mask(mask_resized, min_area=self.min_mask_area)

                if mask_processed.sum() > self.min_mask_area:
                    all_masks.append({
                        'mask': mask_processed,
                        'iou_pred': best_iou,
                        'point': coords[i + j],
                    })

        return all_masks

    @torch.no_grad()
    def generate_masklets(
        self,
        frames: List[np.ndarray],
        first_frame_idx: int = 0,
    ) -> List[Dict]:
        """
        Generate automatic masklets for a video.

        Uses the grid prompting strategy from Appendix E.1:
        - 32x32 grid on first frame
        - 16x16 grid on 4 zoomed crops (2x2 overlapped window)
        - 4x4 grid on 16 zoomed crops (4x4 overlapped window)

        Args:
            frames: list of video frames
            first_frame_idx: index of first frame to use for prompting

        Returns:
            list of masklet dicts
        """
        first_frame = frames[first_frame_idx]
        H, W = first_frame.shape[:2]

        # Generate masks with different grid configurations
        all_first_frame_masks = []

        # 32x32 grid on full frame
        masks_32 = self.generate_masks_for_frame(first_frame, grid_size=32)
        all_first_frame_masks.extend(masks_32)

        # 16x16 grid on 4 zoomed crops (2x2 overlapped window)
        for row in range(2):
            for col in range(2):
                y_start = row * H // 2
                y_end = min(y_start + H * 3 // 4, H)
                x_start = col * W // 2
                x_end = min(x_start + W * 3 // 4, W)

                crop = first_frame[y_start:y_end, x_start:x_end]
                if crop.shape[0] > 0 and crop.shape[1] > 0:
                    crop_masks = self.generate_masks_for_frame(crop, grid_size=16)
                    # Adjust coordinates back to full frame
                    for m in crop_masks:
                        # Resize mask back to full frame
                        full_mask = np.zeros((H, W), dtype=np.uint8)
                        crop_h, crop_w = m['mask'].shape
                        full_mask[y_start:y_start + crop_h, x_start:x_start + crop_w] = m['mask']
                        m['mask'] = full_mask
                    all_first_frame_masks.extend(crop_masks)

        # 4x4 grid on 16 zoomed crops (4x4 overlapped window)
        for row in range(4):
            for col in range(4):
                y_start = row * H // 4
                y_end = min(y_start + H // 2, H)
                x_start = col * W // 4
                x_end = min(x_start + W // 2, W)

                crop = first_frame[y_start:y_end, x_start:x_end]
                if crop.shape[0] > 0 and crop.shape[1] > 0:
                    crop_masks = self.generate_masks_for_frame(crop, grid_size=4)
                    for m in crop_masks:
                        full_mask = np.zeros((H, W), dtype=np.uint8)
                        crop_h, crop_w = m['mask'].shape
                        full_mask[y_start:y_start + crop_h, x_start:x_start + crop_w] = m['mask']
                        m['mask'] = full_mask
                    all_first_frame_masks.extend(crop_masks)

        # Filter by IoU threshold
        filtered_masks = [
            m for m in all_first_frame_masks
            if m['iou_pred'] >= self.iou_threshold
        ]

        # Remove duplicate masks using NMS
        filtered_masks = self._nms_masks(filtered_masks)

        # Propagate each mask through the video
        masklets = []
        for mask_info in filtered_masks:
            masklet = self._propagate_mask(
                frames=frames,
                first_frame_mask=mask_info['mask'],
                first_frame_idx=first_frame_idx,
            )
            if masklet is not None:
                masklets.append(masklet)

        return masklets

    def _nms_masks(self, masks: List[Dict], iou_threshold: float = 0.7) -> List[Dict]:
        """Apply non-maximum suppression to remove duplicate masks."""
        if not masks:
            return masks

        # Sort by IoU prediction (descending)
        masks = sorted(masks, key=lambda x: x['iou_pred'], reverse=True)

        keep = []
        suppressed = set()

        for i, mask_i in enumerate(masks):
            if i in suppressed:
                continue
            keep.append(mask_i)

            for j, mask_j in enumerate(masks[i + 1:], start=i + 1):
                if j in suppressed:
                    continue
                # Compute IoU between masks
                intersection = (mask_i['mask'] & mask_j['mask']).sum()
                union = (mask_i['mask'] | mask_j['mask']).sum()
                if union > 0 and intersection / union > iou_threshold:
                    suppressed.add(j)

        return keep

    @torch.no_grad()
    def _propagate_mask(
        self,
        frames: List[np.ndarray],
        first_frame_mask: np.ndarray,
        first_frame_idx: int = 0,
    ) -> Optional[Dict]:
        """
        Propagate a mask from the first frame through the video.

        Args:
            frames: list of video frames
            first_frame_mask: binary mask for first frame
            first_frame_idx: index of first frame

        Returns:
            masklet dict with 'masks' list and 'iou_preds' list
        """
        T = len(frames)
        memory_bank = {
            'recent_feats': [],
            'recent_pos': [],
            'prompted_feats': [],
            'prompted_pos': [],
            'object_ptrs': [],
        }

        masklet_masks = [None] * T
        masklet_iou_preds = [None] * T

        for t in range(first_frame_idx, T):
            frame = frames[t]
            H, W = frame.shape[:2]
            frame_tensor = self._preprocess_frame(frame)

            # Use mask prompt on first frame
            masks_input = None
            if t == first_frame_idx:
                mask_tensor = torch.from_numpy(first_frame_mask).float()
                mask_tensor = F.interpolate(
                    mask_tensor.unsqueeze(0).unsqueeze(0),
                    size=(self.image_size, self.image_size),
                    mode='nearest',
                ).to(self.device)
                masks_input = mask_tensor

            outputs, memory = self.model.forward_video_frame(
                img=frame_tensor,
                memory_bank=memory_bank if len(memory_bank['recent_feats']) > 0 else None,
                masks=masks_input,
                multimask_output=False,
            )

            # Get predicted mask
            pred_logit = outputs['low_res_masks'][0, 0]
            iou_pred = outputs['iou_predictions'][0, 0].item()

            pred_logit_resized = F.interpolate(
                pred_logit.unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode='bilinear',
                align_corners=False,
            ).squeeze()

            pred_mask = (pred_logit_resized.cpu().numpy() > 0).astype(np.uint8)
            pred_mask = postprocess_mask(pred_mask, min_area=self.min_mask_area)

            masklet_masks[t] = pred_mask
            masklet_iou_preds[t] = iou_pred

            # Update memory bank
            B_f, C_m, H_m, W_m = memory.shape
            from sam2.modeling.prompt_encoder import PositionEmbeddingRandom
            pe_layer = PositionEmbeddingRandom(self.model.hidden_dim // 2)
            pos = pe_layer((H_m, W_m)).unsqueeze(0).expand(B_f, -1, -1, -1).to(self.device)

            memory_bank['recent_feats'].append(memory)
            memory_bank['recent_pos'].append(pos)

            if len(memory_bank['recent_feats']) > self.model.num_maskmem:
                memory_bank['recent_feats'].pop(0)
                memory_bank['recent_pos'].pop(0)

            if 'mask_tokens' in outputs:
                obj_ptr = outputs['mask_tokens'][:, 0, :]
                memory_bank['object_ptrs'].append(obj_ptr)
                if len(memory_bank['object_ptrs']) > self.model.num_maskmem:
                    memory_bank['object_ptrs'].pop(0)

        return {
            'masks': masklet_masks,
            'iou_preds': masklet_iou_preds,
            'first_frame_idx': first_frame_idx,
        }
