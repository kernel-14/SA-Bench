"""
Interactive Prompt Sampler for SAM 2 training.

Simulates interactive prompting of the model during training (Section 4, Appendix D.2.2):
- Samples sequences of 8 frames (or 16 for fine-tuning)
- Randomly selects up to 2 frames to prompt
- Initial prompts: ground-truth mask (50%), positive click (25%), or bounding box (25%)
- Probabilistically receives corrective clicks sampled from error region between
  ground-truth mask and model predictions
- 7 correction clicks during training (vs SAM's 8)
- Reverse temporal order with 50% probability for bi-directional propagation
- Mosaic transform: with 10% probability, tile video into 2x2 grid
"""

import torch
import numpy as np
from typing import Optional, Tuple, List, Dict
import random


class InteractivePromptSampler:
    """
    Simulates interactive user prompts during SAM 2 training.
    """

    def __init__(
        self,
        num_frames: int = 8,
        max_prompted_frames: int = 2,
        num_correction_clicks: int = 7,
        mask_prompt_prob: float = 0.5,
        click_prompt_prob: float = 0.25,
        box_prompt_prob: float = 0.25,
        reverse_prob: float = 0.5,
        mosaic_prob: float = 0.1,
        random_click_prob: float = 0.1,
    ):
        self.num_frames = num_frames
        self.max_prompted_frames = max_prompted_frames
        self.num_correction_clicks = num_correction_clicks
        self.mask_prompt_prob = mask_prompt_prob
        self.click_prompt_prob = click_prompt_prob
        self.box_prompt_prob = box_prompt_prob
        self.reverse_prob = reverse_prob
        self.mosaic_prob = mosaic_prob
        self.random_click_prob = random_click_prob

    def sample_initial_prompts(
        self,
        gt_masks: torch.Tensor,
    ) -> Dict:
        """
        Sample initial prompts for the first prompted frame.

        Args:
            gt_masks: [T, H, W] ground truth masklets

        Returns:
            dict with prompt type and prompt data
        """
        prompt_type_rand = random.random()
        t = 0  # First frame

        if prompt_type_rand < self.mask_prompt_prob:
            # Ground-truth mask prompt
            return {
                "type": "mask",
                "frame_idx": t,
                "mask": gt_masks[t],
            }
        elif prompt_type_rand < self.mask_prompt_prob + self.click_prompt_prob:
            # Positive click from ground-truth mask center
            return {
                "type": "click",
                "frame_idx": t,
                "points": self._sample_click_from_mask(gt_masks[t], positive=True),
                "labels": torch.tensor([1]),
            }
        else:
            # Bounding box from ground-truth mask
            return {
                "type": "box",
                "frame_idx": t,
                "box": self._mask_to_bbox(gt_masks[t]),
            }

    def _sample_click_from_mask(
        self,
        mask: torch.Tensor,
        positive: bool = True,
    ) -> torch.Tensor:
        """Sample a click from the mask.
        Args:
            mask: [H, W] binary mask
            positive: if True, sample from foreground; if False, from background
        """
        H, W = mask.shape
        if positive:
            # Sample from center of foreground region(s)
            fg = mask > 0.5
            if fg.sum() == 0:
                return torch.tensor([0.5, 0.5])  # default center
            ys, xs = torch.where(fg)
            center_y = ys.float().mean() / H
            center_x = xs.float().mean() / W
            return torch.tensor([center_x, center_y])
        else:
            # Sample from background
            bg = mask <= 0.5
            if bg.sum() == 0:
                return torch.tensor([0.0, 0.0])
            ys, xs = torch.where(bg)
            idx = random.randint(0, len(ys) - 1)
            return torch.tensor([xs[idx].float() / W, ys[idx].float() / H])

    def _sample_correction_clicks(
        self,
        gt_mask: torch.Tensor,
        pred_mask: torch.Tensor,
        num_clicks: int = 3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample correction clicks from error region.
        Args:
            gt_mask: [H, W] ground truth mask
            pred_mask: [H, W] predicted mask (binary)
            num_clicks: number of correction clicks to sample
        Returns:
            points [N, 2], labels [N]
        """
        H, W = gt_mask.shape
        pred_binary = (torch.sigmoid(pred_mask) if not pred_mask.dtype == torch.bool else pred_mask) > 0.5

        # Find error regions
        fp = (~gt_mask.bool()) & pred_binary  # false positive: predicted FG but GT is BG
        fn = gt_mask.bool() & ~pred_binary  # false negative: GT FG but predicted BG

        points = []
        labels = []

        for _ in range(num_clicks):
            # Alternate between correcting FP and FN
            if len(points) % 2 == 0 and fn.sum() > 0:
                # Correct a false negative (add positive click)
                ys, xs = torch.where(fn)
                idx = random.randint(0, len(ys) - 1)
                points.append([xs[idx].float() / W, ys[idx].float() / H])
                labels.append(1)  # positive
            elif fp.sum() > 0:
                # Correct a false positive (add negative click)
                ys, xs = torch.where(fp)
                idx = random.randint(0, len(ys) - 1)
                points.append([xs[idx].float() / W, ys[idx].float() / H])
                labels.append(0)  # negative
            elif fn.sum() > 0:
                ys, xs = torch.where(fn)
                idx = random.randint(0, len(ys) - 1)
                points.append([xs[idx].float() / W, ys[idx].float() / H])
                labels.append(1)
            else:
                # Default to center positive
                points.append([0.5, 0.5])
                labels.append(1)

        return torch.tensor(points), torch.tensor(labels)

    def _mask_to_bbox(self, mask: torch.Tensor) -> torch.Tensor:
        """Convert mask to bounding box [x1, y1, x2, y2] in [0, 1] range."""
        H, W = mask.shape
        fg = mask > 0.5
        if fg.sum() == 0:
            return torch.tensor([0.0, 0.0, 1.0, 1.0])
        ys, xs = torch.where(fg)
        x1 = xs.float().min() / W
        y1 = ys.float().min() / H
        x2 = xs.float().max() / W
        y2 = ys.float().max() / H
        return torch.tensor([x1, y1, x2, y2])

    def apply_mosaic_transform(
        self,
        frames: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Apply mosaic transform: tile the video into a 2x2 grid.
        Args:
            frames: [T, 3, H, W]
            masks: [T, H, W] ground truth masklets
        Returns:
            frames, masks, quadrant (0-3) containing target object
        """
        T, C, H, W = frames.shape
        new_H, new_W = H * 2, W * 2

        new_frames = torch.zeros(T, C, new_H, new_W, device=frames.device)
        new_masks = torch.zeros(T, new_H, new_W, device=masks.device)

        # Tile the same video 4 times in 2x2 grid
        quadrant = random.randint(0, 3)
        qy = (quadrant // 2) * H
        qx = (quadrant % 2) * W

        for q in range(4):
            qy_idx = (q // 2) * H
            qx_idx = (q % 2) * W
            new_frames[:, :, qy_idx:qy_idx+H, qx_idx:qx_idx+W] = frames
            new_masks[:, qy_idx:qy_idx+H, qx_idx:qx_idx+W] = masks

        # Only keep mask for the selected quadrant
        full_mask = new_masks.clone()
        new_masks = torch.zeros_like(new_masks)
        new_masks[:, qy:qy+H, qx:qx+W] = full_mask[:, qy:qy+H, qx:qx+W]

        return new_frames, new_masks, quadrant

    def sample_training_sequence(
        self,
        frames: torch.Tensor,
        gt_masks: torch.Tensor,
        model_predictions: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Dict:
        """
        Sample a full training sequence with interactive prompts.

        Args:
            frames: [T, 3, H, W] video frames
            gt_masks: [T, H, W] ground truth masklets
            model_predictions: optional model predictions for correction clicks

        Returns:
            dict with prompts and targets for training
        """
        T = frames.shape[0]

        # Take a random sequence of num_frames
        if T > self.num_frames:
            start = random.randint(0, T - self.num_frames)
            frames = frames[start:start + self.num_frames]
            gt_masks = gt_masks[start:start + self.num_frames]

        # Reverse temporal order with 50% probability
        if random.random() < self.reverse_prob:
            frames = torch.flip(frames, dims=[0])
            gt_masks = torch.flip(gt_masks, dims=[0])

        # Apply mosaic transform with 10% probability
        mosaic_quadrant = None
        if random.random() < self.mosaic_prob:
            frames, gt_masks, mosaic_quadrant = self.apply_mosaic_transform(frames, gt_masks)

        # Sample initial prompt
        initial_prompt = self.sample_initial_prompts(gt_masks)

        # Determine which frames to prompt (up to 2, including the first)
        num_prompted = min(self.max_prompted_frames, self.num_frames)
        prompted_frames = [0]  # Always prompt first frame
        if num_prompted > 1 and self.num_frames > 1:
            additional = random.sample(range(1, self.num_frames), num_prompted - 1)
            prompted_frames.extend(additional)

        return {
            "frames": frames,
            "gt_masks": gt_masks,
            "initial_prompt": initial_prompt,
            "prompted_frames": sorted(prompted_frames),
            "num_correction_clicks": self.num_correction_clicks,
            "reversed": False,  # Flag set above
            "mosaic_quadrant": mosaic_quadrant,
        }
