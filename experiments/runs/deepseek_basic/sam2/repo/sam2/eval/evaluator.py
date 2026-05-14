"""
SAM 2 Evaluator.

Implements evaluation protocols from the paper:
- Semi-supervised VOS evaluation (prompts only on first frame)
- Interactive offline evaluation (multiple passes, select worst frame)
- Interactive online evaluation (single forward pass, pause at low-quality)
- Image segmentation evaluation (mIoU)
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
from .metrics import JFMetric, compute_iou, compute_miou

from ..model.sam2 import SAM2


class SAM2Evaluator:
    """
    Evaluator for SAM 2 on both image and video tasks.
    """

    def __init__(self, model: SAM2, device: str = "cuda"):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def evaluate_vos(
        self,
        frames: torch.Tensor,
        gt_masks: torch.Tensor,
        initial_prompt_type: str = "mask",
        initial_prompt_data: Optional[Dict] = None,
        num_clicks: int = 3,
    ) -> JFMetric:
        """
        Evaluate semi-supervised VOS: prompts only on first frame.

        Args:
            frames: [T, 3, H, W] video frames
            gt_masks: [T, H, W] ground truth masks
            initial_prompt_type: "mask", "click", or "box"
            initial_prompt_data: optional precomputed prompt data
            num_clicks: number of clicks for click-based prompting

        Returns:
            JFMetric with scores
        """
        T = frames.shape[0]
        metric = JFMetric()

        # Reset memory state
        self.model._initialize_state()

        # Process frames
        for t in range(T):
            frame = frames[t:t+1].to(self.device)

            points = None
            boxes = None
            mask_prompts = None

            if t == 0:
                # Apply initial prompt on first frame
                if initial_prompt_type == "mask":
                    mask_prompts = gt_masks[t:t+1].to(self.device)
                elif initial_prompt_type == "click":
                    # Sample clicks from ground truth mask
                    pts, lbls = self._sample_clicks_from_mask(
                        gt_masks[t], num_clicks, initial=True
                    )
                    points = (pts.unsqueeze(0).to(self.device),
                             lbls.unsqueeze(0).to(self.device))
                elif initial_prompt_type == "box":
                    boxes = self._mask_to_bbox(gt_masks[t]).unsqueeze(0).unsqueeze(0).to(self.device)

            output = self.model(
                frame=frame,
                points=points,
                boxes=boxes,
                masks=mask_prompts,
                multimask_output=False,
                is_first_frame=(t == 0),
            )

            # Use the predicted mask for metric
            pred_mask = output["masks"][0, 0]  # [H, W]
            gt_mask = gt_masks[t].to(self.device)

            metric.update(pred_mask, gt_mask)

        return metric

    @torch.no_grad()
    def evaluate_offline_interactive(
        self,
        frames: torch.Tensor,
        gt_masks: torch.Tensor,
        max_interacted_frames: int = 8,
        num_clicks_per_frame: int = 3,
    ) -> List[float]:
        """
        Offline interactive evaluation (Section 6.1, Appendix F.1.2):
        Multiple passes through the video, select frame with lowest IoU for correction.

        Returns:
            List of J&F scores for each number of interacted frames [1..max_interacted_frames]
        """
        T = frames.shape[0]
        results = []

        for N_frames in range(1, max_interacted_frames + 1):
            self.model._initialize_state()
            prompted_frames = set()
            all_prompts = {}  # frame_idx -> (points, labels)

            for round_idx in range(N_frames):
                # Process all frames
                frame_preds = []
                for t in range(T):
                    frame = frames[t:t+1].to(self.device)
                    points = None
                    boxes = None
                    mask_prompts = None

                    if t in all_prompts:
                        pts, lbls = all_prompts[t]
                        points = (pts.unsqueeze(0).to(self.device),
                                 lbls.unsqueeze(0).to(self.device))

                    output = self.model(
                        frame=frame,
                        points=points,
                        boxes=boxes,
                        masks=mask_prompts,
                        multimask_output=False,
                        is_first_frame=(t == 0 and round_idx == 0),
                    )
                    frame_preds.append(output["masks"][0, 0])

                # Find frame with lowest IoU for next round
                if round_idx < N_frames - 1:
                    worst_frame = 0
                    worst_iou = 1.0
                    for t in range(T):
                        if t not in prompted_frames or round_idx == 0:
                            iou = compute_iou(frame_preds[t], gt_masks[t].to(self.device))
                            if iou < worst_iou:
                                worst_iou = iou
                                worst_frame = t

                    # Add correction clicks at the worst frame
                    pts, lbls = self._sample_correction_clicks(
                        gt_masks[worst_frame],
                        frame_preds[worst_frame],
                        num_clicks_per_frame,
                    )
                    all_prompts[worst_frame] = (pts, lbls)
                    prompted_frames.add(worst_frame)

            # Compute J&F for this N_frames
            metric = JFMetric()
            for t in range(T):
                metric.update(
                    frame_preds[t].cpu(),
                    gt_masks[t],
                )
            results.append(metric.get_jf())

        return results

    @torch.no_grad()
    def evaluate_online_interactive(
        self,
        frames: torch.Tensor,
        gt_masks: torch.Tensor,
        max_interacted_frames: int = 8,
        num_clicks_per_frame: int = 3,
        iou_threshold: float = 0.75,
    ) -> List[float]:
        """
        Online interactive evaluation (Section 6.1, Appendix F.1.2):
        Single forward pass, pause at frames with IoU < 0.75 for corrections.

        Returns:
            List of J&F scores for each number of interacted frames
        """
        T = frames.shape[0]
        results = []

        for N_frames in range(1, max_interacted_frames + 1):
            self.model._initialize_state()
            frames_interacted = 0  # Start with first frame interaction
            all_preds = []

            for t in range(T):
                frame = frames[t:t+1].to(self.device)

                points = None
                boxes = None
                mask_prompts = None

                if t == 0:
                    # Initial prompt on first frame
                    pts, lbls = self._sample_clicks_from_mask(
                        gt_masks[t], num_clicks_per_frame, initial=True
                    )
                    points = (pts.unsqueeze(0).to(self.device),
                             lbls.unsqueeze(0).to(self.device))
                    frames_interacted = 1

                output = self.model(
                    frame=frame,
                    points=points,
                    boxes=boxes,
                    masks=mask_prompts,
                    multimask_output=False,
                    is_first_frame=(t == 0),
                )

                pred_mask = output["masks"][0, 0]
                all_preds.append(pred_mask)

                # Check if we need to pause and correct
                iou = compute_iou(pred_mask, gt_masks[t].to(self.device))
                if iou < iou_threshold and frames_interacted < N_frames and t > 0:
                    pts, lbls = self._sample_correction_clicks(
                        gt_masks[t], pred_mask, num_clicks_per_frame
                    )
                    # Re-run with correction
                    output = self.model(
                        frame=frame,
                        points=(pts.unsqueeze(0).to(self.device),
                               lbls.unsqueeze(0).to(self.device)),
                        boxes=None,
                        masks=None,
                        multimask_output=False,
                        is_first_frame=False,
                    )
                    all_preds[-1] = output["masks"][0, 0]
                    frames_interacted += 1

            # Compute J&F
            metric = JFMetric()
            for t in range(T):
                metric.update(all_preds[t].cpu(), gt_masks[t])
            results.append(metric.get_jf())

        return results

    @torch.no_grad()
    def evaluate_image(
        self,
        images: torch.Tensor,
        gt_masks: torch.Tensor,
        num_clicks: int = 1,
    ) -> float:
        """
        Evaluate on images (SA task) with click prompts.

        Args:
            images: [B, 3, H, W] images
            gt_masks: [B, N_masks, H, W] ground truth masks
            num_clicks: number of clicks for clicking-based evaluation

        Returns:
            mIoU score
        """
        B = images.shape[0]
        all_ious = []

        for b in range(B):
            image = images[b:b+1].to(self.device)
            N_masks = gt_masks.shape[1]

            for n in range(N_masks):
                gt = gt_masks[b, n]
                if gt.max() < 0.5:
                    continue

                # Sample click(s) from ground truth
                pts, lbls = self._sample_clicks_from_mask(gt, num_clicks, initial=True)

                output = self.model(
                    frame=image,
                    points=(pts.unsqueeze(0).to(self.device),
                           lbls.unsqueeze(0).to(self.device)),
                    multimask_output=False,
                    is_first_frame=True,
                )

                pred = output["masks"][0, 0]
                iou = compute_iou(pred.cpu(), gt)
                all_ious.append(iou)

        return sum(all_ious) / max(1, len(all_ious))

    def _sample_clicks_from_mask(
        self,
        mask: torch.Tensor,
        num_clicks: int,
        initial: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample clicks from mask for evaluation."""
        H, W = mask.shape
        fg = mask > 0.5

        if fg.sum() == 0:
            return torch.tensor([[0.5, 0.5]]), torch.tensor([1])

        ys, xs = torch.where(fg)

        points = []
        labels = []

        if initial:
            # First click at object center
            center_y = ys.float().mean() / H
            center_x = xs.float().mean() / W
            points.append([center_x, center_y])
            labels.append(1)

            # Additional clicks at error regions (simplified)
            for _ in range(num_clicks - 1):
                idx = torch.randint(0, len(ys), (1,)).item()
                points.append([xs[idx].float() / W, ys[idx].float() / H])
                labels.append(1)
        else:
            # All clicks at random foreground points
            for _ in range(num_clicks):
                idx = torch.randint(0, len(ys), (1,)).item()
                points.append([xs[idx].float() / W, ys[idx].float() / H])
                labels.append(1)

        return torch.tensor(points), torch.tensor(labels)

    def _sample_correction_clicks(
        self,
        gt_mask: torch.Tensor,
        pred_mask: torch.Tensor,
        num_clicks: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample correction clicks from error region."""
        H, W = gt_mask.shape
        pred_binary = (pred_mask > 0.5).float()
        gt_binary = (gt_mask > 0.5).float()

        # Error regions
        fp = (gt_binary == 0) & (pred_binary == 1)
        fn = (gt_binary == 1) & (pred_binary == 0)

        points = []
        labels = []

        for i in range(num_clicks):
            if i % 2 == 0 and fn.sum() > 0:
                ys, xs = torch.where(fn)
                idx = torch.randint(0, len(ys), (1,)).item()
                points.append([xs[idx].float() / W, ys[idx].float() / H])
                labels.append(1)
            elif fp.sum() > 0:
                ys, xs = torch.where(fp)
                idx = torch.randint(0, len(ys), (1,)).item()
                points.append([xs[idx].float() / W, ys[idx].float() / H])
                labels.append(0)
            else:
                points.append([0.5, 0.5])
                labels.append(1)

        return torch.tensor(points), torch.tensor(labels)

    def _mask_to_bbox(self, mask: torch.Tensor) -> torch.Tensor:
        """Convert mask to bounding box."""
        H, W = mask.shape
        fg = mask > 0.5
        if fg.sum() == 0:
            return torch.tensor([0.0, 0.0, 1.0, 1.0])
        ys, xs = torch.where(fg)
        return torch.tensor([
            xs.float().min() / W,
            ys.float().min() / H,
            xs.float().max() / W,
            ys.float().max() / H,
        ])


class InteractiveEvaluator:
    """
    Evaluator for interactive PVS task (Figure 5).
    """

    def __init__(self, model: SAM2, device: str = "cuda"):
        self.evaluator = SAM2Evaluator(model, device)

    def evaluate_offline(
        self,
        video_frames: List[torch.Tensor],
        video_gts: List[torch.Tensor],
        max_frames: int = 8,
    ) -> List[float]:
        """Run offline interactive evaluation over multiple videos."""
        all_results = [[] for _ in range(max_frames)]
        for frames, gts in zip(video_frames, video_gts):
            results = self.evaluator.evaluate_offline_interactive(
                frames, gts, max_frames, num_clicks_per_frame=3
            )
            for i, r in enumerate(results):
                all_results[i].append(r)

        # Average over videos
        return [sum(r) / len(r) for r in all_results]

    def evaluate_online(
        self,
        video_frames: List[torch.Tensor],
        video_gts: List[torch.Tensor],
        max_frames: int = 8,
    ) -> List[float]:
        """Run online interactive evaluation over multiple videos."""
        all_results = [[] for _ in range(max_frames)]
        for frames, gts in zip(video_frames, video_gts):
            results = self.evaluator.evaluate_online_interactive(
                frames, gts, max_frames, num_clicks_per_frame=3
            )
            for i, r in enumerate(results):
                all_results[i].append(r)

        return [sum(r) / len(r) for r in all_results]
