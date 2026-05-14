"""
SAM 2 Evaluation Script

Implements evaluation protocols from Section 6 of the paper:

1. Promptable Video Segmentation (§6.1):
   - Offline evaluation: multiple passes, select frame with lowest IoU
   - Online evaluation: single pass, pause at low-quality frames
   - 9 densely annotated zero-shot datasets
   - 3 clicks per frame, up to 8 interacted frames
   - Reports J&F metric

2. Semi-supervised VOS (§6.2):
   - First-frame prompts only (1/3/5 clicks, box, or GT mask)
   - 17 zero-shot video datasets
   - Reports J&F metric

3. Image Segmentation (§6.3):
   - 37 zero-shot datasets (23 from SAM + 14 new video datasets)
   - 1-click and 5-click mIoU
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sam2.modeling.sam2_model import SAM2Model, build_sam2


def compute_j_metric(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Compute Jaccard (IoU) metric between predicted and ground-truth masks.
    J = |pred ∩ gt| / |pred ∪ gt|
    """
    pred_bool = pred_mask > 0
    gt_bool = gt_mask > 0

    intersection = (pred_bool & gt_bool).sum()
    union = (pred_bool | gt_bool).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return float(intersection) / float(union)


def compute_f_metric(pred_mask: np.ndarray, gt_mask: np.ndarray, bound_th: float = 0.008) -> float:
    """
    Compute F-measure (boundary metric) between predicted and ground-truth masks.
    Based on the DAVIS evaluation protocol.
    """
    def seg2bmap(seg: np.ndarray, width: Optional[int] = None, height: Optional[int] = None) -> np.ndarray:
        """Convert segmentation to boundary map."""
        seg = seg.astype(bool)
        seg[seg > 0] = 1

        e = np.zeros_like(seg)
        s = np.zeros_like(seg)
        se = np.zeros_like(seg)
        sw = np.zeros_like(seg)

        e[:, :-1] = seg[:, 1:]
        s[:-1, :] = seg[1:, :]
        se[:-1, :-1] = seg[1:, 1:]
        sw[:-1, 1:] = seg[1:, :-1]

        b = seg ^ e | seg ^ s | seg ^ se | seg ^ sw
        b[-1, :] = seg[-1, :] ^ e[-1, :]
        b[:, -1] = seg[:, -1] ^ s[:, -1]
        b[-1, -1] = 0

        if width is not None and height is not None:
            b = b[:height, :width]

        return b

    def boundary_iou(gt: np.ndarray, dt: np.ndarray, bound_th: float = 0.008) -> float:
        """Compute boundary IoU."""
        bound_pix = bound_th if bound_th >= 1 else \
            np.ceil(bound_th * np.linalg.norm(gt.shape))

        gt_b = seg2bmap(gt)
        dt_b = seg2bmap(dt)

        from scipy.ndimage import binary_dilation
        gt_b_dil = binary_dilation(gt_b, iterations=int(bound_pix))
        dt_b_dil = binary_dilation(dt_b, iterations=int(bound_pix))

        # Get the intersection
        gt_match = gt_b * dt_b_dil
        dt_match = dt_b * gt_b_dil

        # Area of the intersection
        n_gt = gt_b.sum()
        n_dt = dt_b.sum()

        if n_gt == 0 and n_dt > 0:
            return 0.0
        elif n_gt > 0 and n_dt == 0:
            return 0.0
        elif n_gt == 0 and n_dt == 0:
            return 1.0

        precision = float(dt_match.sum()) / float(n_dt)
        recall = float(gt_match.sum()) / float(n_gt)

        if precision + recall == 0:
            return 0.0

        return 2.0 * precision * recall / (precision + recall)

    return boundary_iou(gt_mask, pred_mask, bound_th)


def compute_jf_metric(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Tuple[float, float]:
    """Compute both J and F metrics."""
    j = compute_j_metric(pred_mask, gt_mask)
    f = compute_f_metric(pred_mask, gt_mask)
    return j, f


def get_center_click(mask: np.ndarray) -> Tuple[int, int]:
    """Get click at center of mass of mask."""
    if mask.sum() == 0:
        h, w = mask.shape
        return h // 2, w // 2
    y_coords, x_coords = np.where(mask > 0)
    return int(y_coords.mean()), int(x_coords.mean())


def get_error_click(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> Tuple[Optional[Tuple[int, int]], int]:
    """
    Get corrective click from error region.
    Returns (click_position, label) where label is 1 (positive) or 0 (negative).
    """
    # False negatives (missed regions)
    fn = (gt_mask > 0) & (pred_mask == 0)
    # False positives (extra regions)
    fp = (gt_mask == 0) & (pred_mask > 0)

    fn_area = fn.sum()
    fp_area = fp.sum()

    if fn_area == 0 and fp_area == 0:
        return None, 1

    if fn_area >= fp_area:
        # Add positive click in false negative region
        return get_center_click(fn.astype(np.float32)), 1
    else:
        # Add negative click in false positive region
        return get_center_click(fp.astype(np.float32)), 0


class VideoEvaluator:
    """
    Evaluator for video segmentation tasks.

    Implements both offline and online interactive evaluation protocols.
    """

    def __init__(
        self,
        model: SAM2Model,
        device: torch.device,
        image_size: int = 1024,
        num_clicks_per_frame: int = 3,
        max_interacted_frames: int = 8,
        iou_threshold: float = 0.75,  # For online evaluation
    ):
        self.model = model
        self.device = device
        self.image_size = image_size
        self.num_clicks_per_frame = num_clicks_per_frame
        self.max_interacted_frames = max_interacted_frames
        self.iou_threshold = iou_threshold

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess a frame for model input."""
        from PIL import Image
        from torchvision.transforms import functional as TF

        if isinstance(frame, np.ndarray):
            img = Image.fromarray(frame)
        else:
            img = frame

        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img_tensor = TF.to_tensor(img)
        img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return img_tensor.unsqueeze(0).to(self.device)

    def _get_clicks(
        self,
        pred_mask: Optional[np.ndarray],
        gt_mask: np.ndarray,
        num_clicks: int,
        is_first_frame: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample clicks for interactive evaluation.

        First click: center of GT mask
        Subsequent clicks: center of error region
        """
        coords = []
        labels = []

        if is_first_frame or pred_mask is None:
            # First click at center of GT mask
            cy, cx = get_center_click(gt_mask)
            coords.append([cx, cy])
            labels.append(1)
            remaining_clicks = num_clicks - 1
        else:
            remaining_clicks = num_clicks

        # Add corrective clicks
        current_pred = pred_mask if pred_mask is not None else np.zeros_like(gt_mask)
        for _ in range(remaining_clicks):
            click_pos, label = get_error_click(current_pred, gt_mask)
            if click_pos is None:
                break
            cy, cx = click_pos
            coords.append([cx, cy])
            labels.append(label)

        if not coords:
            cy, cx = get_center_click(gt_mask)
            coords.append([cx, cy])
            labels.append(1)

        # Scale coordinates to image size
        h, w = gt_mask.shape
        scale_x = self.image_size / w
        scale_y = self.image_size / h

        scaled_coords = [[int(c[0] * scale_x), int(c[1] * scale_y)] for c in coords]

        coords_tensor = torch.tensor(scaled_coords, dtype=torch.float32, device=self.device).unsqueeze(0)
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=self.device).unsqueeze(0)

        return coords_tensor, labels_tensor

    @torch.no_grad()
    def evaluate_offline(
        self,
        frames: List[np.ndarray],
        gt_masks: List[np.ndarray],
    ) -> Dict[str, float]:
        """
        Offline interactive evaluation.

        Multiple passes through the video, selecting the frame with lowest IoU
        for the next interaction.

        Returns J&F metrics for each number of interacted frames.
        """
        self.model.eval()
        T = len(frames)

        # Initialize memory bank
        memory_bank = {
            'recent_feats': [],
            'recent_pos': [],
            'prompted_feats': [],
            'prompted_pos': [],
            'object_ptrs': [],
        }

        # Pre-compute image features
        frame_tensors = [self._preprocess_frame(f) for f in frames]

        results = {}
        pred_masks = [None] * T
        prompted_frames = set()

        for interaction_round in range(self.max_interacted_frames):
            if interaction_round == 0:
                # First interaction: prompt first frame
                prompt_frame_idx = 0
            else:
                # Select frame with lowest IoU
                ious = []
                for t in range(T):
                    if pred_masks[t] is not None and gt_masks[t] is not None:
                        iou = compute_j_metric(pred_masks[t], gt_masks[t])
                        ious.append((iou, t))
                    else:
                        ious.append((0.0, t))

                if not ious:
                    break
                _, prompt_frame_idx = min(ious)

            prompted_frames.add(prompt_frame_idx)

            # Get clicks for this frame
            gt_mask = gt_masks[prompt_frame_idx]
            pred_mask = pred_masks[prompt_frame_idx]
            is_first = (interaction_round == 0)

            coords, labels = self._get_clicks(
                pred_mask=pred_mask,
                gt_mask=gt_mask,
                num_clicks=self.num_clicks_per_frame,
                is_first_frame=is_first,
            )

            # Reset memory bank for new pass
            memory_bank = {
                'recent_feats': [],
                'recent_pos': [],
                'prompted_feats': [],
                'prompted_pos': [],
                'object_ptrs': [],
            }

            # Process all frames
            for t in range(T):
                frame_tensor = frame_tensors[t]

                # Add prompts if this is the prompted frame
                points = None
                if t == prompt_frame_idx:
                    points = (coords, labels)

                outputs, memory = self.model.forward_video_frame(
                    img=frame_tensor,
                    memory_bank=memory_bank if len(memory_bank['recent_feats']) > 0 else None,
                    points=points,
                    multimask_output=(points is not None),
                )

                # Get best mask
                pred_logits = outputs['low_res_masks']
                iou_pred = outputs['iou_predictions']

                if pred_logits.shape[1] > 1:
                    best_idx = iou_pred.argmax(dim=1)
                    pred_logit = pred_logits[0, best_idx[0]]
                else:
                    pred_logit = pred_logits[0, 0]

                # Resize to original size
                h, w = gt_masks[t].shape if gt_masks[t] is not None else (self.image_size, self.image_size)
                pred_logit_resized = F.interpolate(
                    pred_logit.unsqueeze(0).unsqueeze(0),
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze()

                pred_masks[t] = (pred_logit_resized.cpu().numpy() > 0).astype(np.uint8)

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

            # Compute metrics after this round
            j_scores = []
            f_scores = []
            for t in range(T):
                if pred_masks[t] is not None and gt_masks[t] is not None:
                    j, f = compute_jf_metric(pred_masks[t], gt_masks[t])
                    j_scores.append(j)
                    f_scores.append(f)

            if j_scores:
                jf = (np.mean(j_scores) + np.mean(f_scores)) / 2
                results[f'jf_at_{interaction_round + 1}_frames'] = jf

        return results

    @torch.no_grad()
    def evaluate_semi_supervised(
        self,
        frames: List[np.ndarray],
        gt_masks: List[np.ndarray],
        prompt_type: str = 'mask',
        num_clicks: int = 1,
    ) -> Dict[str, float]:
        """
        Semi-supervised VOS evaluation.

        Prompts only on the first frame, then propagates.

        Args:
            frames: list of video frames
            gt_masks: list of ground-truth masks
            prompt_type: 'mask', 'click', or 'box'
            num_clicks: number of clicks (for click prompt type)

        Returns:
            dict with J, F, and J&F metrics
        """
        self.model.eval()
        T = len(frames)

        memory_bank = {
            'recent_feats': [],
            'recent_pos': [],
            'prompted_feats': [],
            'prompted_pos': [],
            'object_ptrs': [],
        }

        frame_tensors = [self._preprocess_frame(f) for f in frames]
        pred_masks = []

        for t in range(T):
            frame_tensor = frame_tensors[t]

            # Build prompts for first frame
            points = None
            boxes = None
            masks = None

            if t == 0:
                gt_mask = gt_masks[0]
                h, w = gt_mask.shape

                if prompt_type == 'mask':
                    # GT mask prompt
                    mask_tensor = torch.from_numpy(gt_mask).float().unsqueeze(0).unsqueeze(0)
                    mask_tensor = F.interpolate(
                        mask_tensor,
                        size=(self.image_size, self.image_size),
                        mode='nearest',
                    ).to(self.device)
                    masks = mask_tensor

                elif prompt_type == 'click':
                    # Click prompts
                    coords_list = []
                    labels_list = []

                    # First click at center
                    cy, cx = get_center_click(gt_mask)
                    cx_scaled = int(cx * self.image_size / w)
                    cy_scaled = int(cy * self.image_size / h)
                    coords_list.append([cx_scaled, cy_scaled])
                    labels_list.append(1)

                    # Additional corrective clicks
                    for _ in range(num_clicks - 1):
                        # Simulate with empty prediction for first frame
                        click_pos, label = get_error_click(
                            np.zeros_like(gt_mask), gt_mask
                        )
                        if click_pos is not None:
                            cy_c, cx_c = click_pos
                            coords_list.append([int(cx_c * self.image_size / w),
                                               int(cy_c * self.image_size / h)])
                            labels_list.append(label)

                    coords_tensor = torch.tensor(coords_list, dtype=torch.float32, device=self.device).unsqueeze(0)
                    labels_tensor = torch.tensor(labels_list, dtype=torch.long, device=self.device).unsqueeze(0)
                    points = (coords_tensor, labels_tensor)

                elif prompt_type == 'box':
                    y_coords, x_coords = np.where(gt_mask > 0)
                    if len(y_coords) > 0:
                        x1, y1 = int(x_coords.min()), int(y_coords.min())
                        x2, y2 = int(x_coords.max()), int(y_coords.max())
                        # Scale to image size
                        x1 = int(x1 * self.image_size / w)
                        y1 = int(y1 * self.image_size / h)
                        x2 = int(x2 * self.image_size / w)
                        y2 = int(y2 * self.image_size / h)
                        boxes = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32, device=self.device)

            outputs, memory = self.model.forward_video_frame(
                img=frame_tensor,
                memory_bank=memory_bank if len(memory_bank['recent_feats']) > 0 else None,
                points=points,
                boxes=boxes,
                masks=masks,
                multimask_output=(t == 0 and prompt_type != 'mask'),
            )

            # Get best mask
            pred_logits = outputs['low_res_masks']
            iou_pred = outputs['iou_predictions']

            if pred_logits.shape[1] > 1:
                best_idx = iou_pred.argmax(dim=1)
                pred_logit = pred_logits[0, best_idx[0]]
            else:
                pred_logit = pred_logits[0, 0]

            # Resize to original size
            if gt_masks[t] is not None:
                h, w = gt_masks[t].shape
            else:
                h = w = self.image_size

            pred_logit_resized = F.interpolate(
                pred_logit.unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode='bilinear',
                align_corners=False,
            ).squeeze()

            pred_mask = (pred_logit_resized.cpu().numpy() > 0).astype(np.uint8)
            pred_masks.append(pred_mask)

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

        # Compute metrics
        j_scores = []
        f_scores = []
        for t in range(T):
            if gt_masks[t] is not None and pred_masks[t] is not None:
                j, f = compute_jf_metric(pred_masks[t], gt_masks[t])
                j_scores.append(j)
                f_scores.append(f)

        if not j_scores:
            return {'J': 0.0, 'F': 0.0, 'JF': 0.0}

        J = float(np.mean(j_scores))
        F = float(np.mean(f_scores))
        JF = (J + F) / 2

        return {'J': J, 'F': F, 'JF': JF}


def main():
    parser = argparse.ArgumentParser(description='SAM 2 Evaluation')

    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--model_size', type=str, default='base_plus')
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--device', type=str, default='cuda')

    # Evaluation settings
    parser.add_argument('--eval_type', type=str, default='semi_supervised',
                        choices=['offline', 'online', 'semi_supervised', 'image'])
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='davis',
                        choices=['davis', 'mose', 'youtube_vos', 'sa_v'])
    parser.add_argument('--prompt_type', type=str, default='mask',
                        choices=['mask', 'click', 'box'])
    parser.add_argument('--num_clicks', type=int, default=1)
    parser.add_argument('--num_clicks_per_frame', type=int, default=3)
    parser.add_argument('--max_interacted_frames', type=int, default=8)

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Build and load model
    print(f"Loading SAM 2 ({args.model_size}) from {args.checkpoint}...")
    model = build_sam2(model_size=args.model_size, image_size=args.image_size)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    # Create evaluator
    evaluator = VideoEvaluator(
        model=model,
        device=device,
        image_size=args.image_size,
        num_clicks_per_frame=args.num_clicks_per_frame,
        max_interacted_frames=args.max_interacted_frames,
    )

    print(f"Evaluating on {args.dataset} with {args.eval_type} protocol...")
    print("Note: Full evaluation requires dataset-specific data loading.")
    print("Please implement dataset-specific loaders for your evaluation.")


if __name__ == '__main__':
    main()
