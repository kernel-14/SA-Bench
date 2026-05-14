"""
SAM 2 evaluation script.

Implements three evaluation protocols from the paper:

1. Interactive offline evaluation (Section 6.1, Appendix F.1.2):
   - Multiple passes over the video.
   - Each pass: select frame with lowest IoU, add 3 clicks.
   - Report average J&F over N_frame = 1..8 interacted frames.

2. Interactive online evaluation (Section 6.1, Appendix F.1.2):
   - Single forward pass through the video.
   - Pause when IoU < 0.75, add 3 correction clicks.
   - New prompts only affect frames after the current paused frame.

3. Semi-supervised VOS evaluation (Section 6.2, Appendix F.1.3):
   - Prompts (click/box/mask) only on the first frame.
   - Propagate to all subsequent frames.
   - Report J&F on 17 zero-shot datasets.

4. Image segmentation evaluation (Section 6.3, Appendix F.4):
   - 1-click and 5-click mIoU on 23/37 zero-shot datasets.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from tqdm import tqdm

from config import SAM2Config, get_config
from data.utils import get_error_center, get_mask_center, prompts_to_tensors
from model import build_sam2
from utils import (
    compute_jf_dataset, compute_jf_sequence, compute_miou,
    load_checkpoint, setup_logger,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mask_to_binary(logits: Tensor, threshold: float = 0.0) -> np.ndarray:
    """Convert mask logits to binary numpy array."""
    return (logits.squeeze() > threshold).cpu().numpy().astype(bool)


def select_best_mask(pred_masks: Tensor, iou_pred: Tensor) -> Tensor:
    """Select the mask with highest predicted IoU."""
    best_idx = iou_pred.argmax(dim=1)  # (B,)
    B = pred_masks.shape[0]
    return pred_masks[torch.arange(B), best_idx].unsqueeze(1)  # (B, 1, H, W)


def build_click_prompt(
    gt_mask: np.ndarray,
    pred_mask: Optional[np.ndarray],
    num_clicks: int,
    image_size: Tuple[int, int],
    device: torch.device,
) -> Tuple[Optional[Tuple[Tensor, Tensor]], None, None]:
    """Build click prompt tensors for a frame."""
    H, W = image_size
    points = []
    labels = []

    if pred_mask is None:
        # Initial click: center of GT mask
        cy, cx = get_mask_center(gt_mask)
        points.append([cx, cy])
        labels.append(1)
        # Add more clicks from error region (no prediction yet, so all GT)
        for _ in range(num_clicks - 1):
            points.append([cx, cy])
            labels.append(1)
    else:
        # Correction clicks from error region
        current_pred = pred_mask.copy()
        for _ in range(num_clicks):
            cy, cx, label = get_error_center(current_pred, gt_mask)
            points.append([cx, cy])
            labels.append(label)

    coords = torch.tensor(points, dtype=torch.float32, device=device).unsqueeze(0)
    lbls = torch.tensor(labels, dtype=torch.long, device=device).unsqueeze(0)
    return (coords, lbls), None, None


# ---------------------------------------------------------------------------
# Interactive offline evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_interactive_offline(
    model,
    video_frames: List[np.ndarray],
    video_gt_masks: List[np.ndarray],
    device: torch.device,
    num_clicks: int = 3,
    max_frames: int = 8,
    image_size: int = 1024,
) -> Dict[str, List[float]]:
    """
    Offline interactive evaluation for a single object in a video.

    Args:
        video_frames:   list of (H, W, 3) uint8 arrays
        video_gt_masks: list of (H, W) bool arrays
        num_clicks:     clicks per interacted frame (default 3)
        max_frames:     maximum number of interacted frames

    Returns dict with 'jf_per_nframes': list of J&F scores for n=1..max_frames.
    """
    T = len(video_frames)
    model.eval()
    model.reset_state(0)

    # Precompute image embeddings
    from data.dataset import normalize, IMAGENET_MEAN, IMAGENET_STD
    frames_tensor = []
    for frame in video_frames:
        f = torch.from_numpy(frame).float() / 255.0
        f = f.permute(2, 0, 1)  # (3, H, W)
        f = F.interpolate(f.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False).squeeze(0)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        f = (f - mean) / std
        frames_tensor.append(f.to(device))

    jf_per_nframes = []
    prompted_frame_indices = []
    all_pred_masks = [None] * T  # Current predictions

    for n_frame in range(1, max_frames + 1):
        # Select frame to interact with
        if n_frame == 1:
            interact_frame = 0
        else:
            # Select frame with lowest IoU
            ious = []
            for t in range(T):
                if all_pred_masks[t] is not None and video_gt_masks[t] is not None:
                    gt = video_gt_masks[t].astype(bool)
                    pred = all_pred_masks[t].astype(bool)
                    inter = (pred & gt).sum()
                    union = (pred | gt).sum()
                    iou = inter / (union + 1e-6) if union > 0 else 1.0
                    ious.append((iou, t))
                else:
                    ious.append((0.0, t))
            interact_frame = min(ious, key=lambda x: x[0])[1]

        prompted_frame_indices.append(interact_frame)

        # Re-run the entire video with all prompts so far
        model.reset_state(0)
        recent_memories: List[Tensor] = []
        prompted_memories: List[Tensor] = []
        object_pointers: List[Tensor] = []
        new_pred_masks = [None] * T

        for t in range(T):
            frame_t = frames_tensor[t].unsqueeze(0)
            gt_mask_t = video_gt_masks[t]

            points_t = boxes_t = masks_t = None
            if t in prompted_frame_indices:
                pred_t = all_pred_masks[t] if t != interact_frame else None
                points_t, boxes_t, masks_t = build_click_prompt(
                    gt_mask_t, pred_t, num_clicks, (image_size, image_size), device
                )

            output = model(
                frames=frame_t,
                points=points_t,
                boxes=boxes_t,
                masks=masks_t,
                recent_memories=recent_memories[-6:],
                prompted_memories=prompted_memories,
                object_pointers=object_pointers,
                multimask_output=(points_t is not None),
            )

            pred_masks = output["masks"]
            iou_pred = output["iou_pred"]
            best_mask = select_best_mask(pred_masks, iou_pred)
            new_pred_masks[t] = mask_to_binary(best_mask)

            memory = output["memory"]
            pointer = output["pointer_tokens"]
            if t in prompted_frame_indices:
                prompted_memories.append(memory)
            else:
                recent_memories.append(memory)
            object_pointers.append(pointer)

        all_pred_masks = new_pred_masks

        # Compute J&F
        valid_preds = [p for p in all_pred_masks if p is not None]
        valid_gts = [g for g in video_gt_masks if g is not None]
        if valid_preds and valid_gts:
            scores = compute_jf_sequence(valid_preds, valid_gts)
            jf_per_nframes.append(scores["JF"])
        else:
            jf_per_nframes.append(0.0)

    return {"jf_per_nframes": jf_per_nframes}


# ---------------------------------------------------------------------------
# Interactive online evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_interactive_online(
    model,
    video_frames: List[np.ndarray],
    video_gt_masks: List[np.ndarray],
    device: torch.device,
    num_clicks: int = 3,
    max_frames: int = 8,
    iou_threshold: float = 0.75,
    image_size: int = 1024,
) -> Dict[str, List[float]]:
    """
    Online interactive evaluation: single forward pass, pause at low-quality frames.
    """
    from data.dataset import IMAGENET_MEAN, IMAGENET_STD
    T = len(video_frames)
    model.eval()
    model.reset_state(0)

    frames_tensor = []
    for frame in video_frames:
        f = torch.from_numpy(frame).float() / 255.0
        f = f.permute(2, 0, 1)
        f = F.interpolate(f.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False).squeeze(0)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        f = (f - mean) / std
        frames_tensor.append(f.to(device))

    jf_per_nframes = []
    all_pred_masks = [None] * T
    recent_memories: List[Tensor] = []
    prompted_memories: List[Tensor] = []
    object_pointers: List[Tensor] = []
    num_prompted = 0

    for t in range(T):
        frame_t = frames_tensor[t].unsqueeze(0)
        gt_mask_t = video_gt_masks[t]

        # Check if we should prompt this frame
        should_prompt = (t == 0)
        if t > 0 and all_pred_masks[t - 1] is not None and num_prompted < max_frames:
            # Check IoU of previous frame
            prev_pred = all_pred_masks[t - 1]
            prev_gt = video_gt_masks[t - 1]
            if prev_gt is not None and prev_pred is not None:
                inter = (prev_pred & prev_gt).sum()
                union = (prev_pred | prev_gt).sum()
                iou = inter / (union + 1e-6) if union > 0 else 1.0
                if iou < iou_threshold:
                    should_prompt = True

        points_t = boxes_t = masks_t = None
        if should_prompt and num_prompted < max_frames:
            pred_t = all_pred_masks[t]
            points_t, boxes_t, masks_t = build_click_prompt(
                gt_mask_t, pred_t, num_clicks, (image_size, image_size), device
            )
            num_prompted += 1

        output = model(
            frames=frame_t,
            points=points_t,
            boxes=boxes_t,
            masks=masks_t,
            recent_memories=recent_memories[-6:],
            prompted_memories=prompted_memories,
            object_pointers=object_pointers,
            multimask_output=(points_t is not None),
        )

        pred_masks = output["masks"]
        iou_pred = output["iou_pred"]
        best_mask = select_best_mask(pred_masks, iou_pred)
        all_pred_masks[t] = mask_to_binary(best_mask)

        memory = output["memory"]
        pointer = output["pointer_tokens"]
        if should_prompt:
            prompted_memories.append(memory)
        else:
            recent_memories.append(memory)
        object_pointers.append(pointer)

    # Compute J&F at each number of prompted frames
    # (simplified: report final J&F)
    valid_preds = [p for p in all_pred_masks if p is not None]
    valid_gts = [g for g in video_gt_masks if g is not None]
    if valid_preds and valid_gts:
        scores = compute_jf_sequence(valid_preds, valid_gts)
        jf = scores["JF"]
    else:
        jf = 0.0

    return {"jf": jf, "num_prompted": num_prompted}


# ---------------------------------------------------------------------------
# Semi-supervised VOS evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_semisupervised_vos(
    model,
    video_frames: List[np.ndarray],
    video_gt_masks: List[np.ndarray],
    device: torch.device,
    prompt_type: str = "mask",
    num_clicks: int = 1,
    image_size: int = 1024,
) -> Dict[str, float]:
    """
    Semi-supervised VOS: prompt only on first frame, propagate to all frames.

    prompt_type: "mask", "1-click", "3-click", "5-click", "box"
    """
    from data.dataset import IMAGENET_MEAN, IMAGENET_STD
    T = len(video_frames)
    model.eval()
    model.reset_state(0)

    frames_tensor = []
    for frame in video_frames:
        f = torch.from_numpy(frame).float() / 255.0
        f = f.permute(2, 0, 1)
        f = F.interpolate(f.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False).squeeze(0)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        f = (f - mean) / std
        frames_tensor.append(f.to(device))

    all_pred_masks = []
    recent_memories: List[Tensor] = []
    prompted_memories: List[Tensor] = []
    object_pointers: List[Tensor] = []

    for t in range(T):
        frame_t = frames_tensor[t].unsqueeze(0)
        gt_mask_t = video_gt_masks[t]

        points_t = boxes_t = masks_t = None
        if t == 0:
            # Build first-frame prompt
            if prompt_type == "mask":
                gt_np = gt_mask_t.astype(np.float32)
                masks_t = torch.from_numpy(gt_np).float().unsqueeze(0).unsqueeze(0).to(device)
            elif "click" in prompt_type:
                n_clicks = int(prompt_type.split("-")[0])
                points_t, boxes_t, masks_t = build_click_prompt(
                    gt_mask_t, None, n_clicks, (image_size, image_size), device
                )
            elif prompt_type == "box":
                ys, xs = np.where(gt_mask_t)
                if len(ys) > 0:
                    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                    boxes_t = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32, device=device)

        output = model(
            frames=frame_t,
            points=points_t,
            boxes=boxes_t,
            masks=masks_t,
            recent_memories=recent_memories[-6:],
            prompted_memories=prompted_memories,
            object_pointers=object_pointers,
            multimask_output=(t == 0 and prompt_type != "mask"),
        )

        pred_masks = output["masks"]
        iou_pred = output["iou_pred"]
        best_mask = select_best_mask(pred_masks, iou_pred)
        all_pred_masks.append(mask_to_binary(best_mask))

        memory = output["memory"]
        pointer = output["pointer_tokens"]
        if t == 0:
            prompted_memories.append(memory)
        else:
            recent_memories.append(memory)
        object_pointers.append(pointer)

    scores = compute_jf_sequence(all_pred_masks, video_gt_masks)
    return scores


# ---------------------------------------------------------------------------
# Image segmentation evaluation (mIoU)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_image_segmentation(
    model,
    images: List[np.ndarray],
    gt_masks: List[np.ndarray],
    device: torch.device,
    num_clicks: int = 1,
    image_size: int = 1024,
) -> float:
    """
    Evaluate image segmentation with click prompts.
    Returns mIoU over all (image, mask) pairs.
    """
    from data.dataset import IMAGENET_MEAN, IMAGENET_STD
    model.eval()
    pred_masks_all = []
    gt_masks_all = []

    for img, gt_mask in zip(images, gt_masks):
        # Preprocess image
        f = torch.from_numpy(img).float() / 255.0
        f = f.permute(2, 0, 1)
        f = F.interpolate(f.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False).squeeze(0)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        f = (f - mean) / std
        frame_t = f.unsqueeze(0).to(device)

        # Resize GT mask
        gt_resized = np.array(Image.fromarray(gt_mask.astype(np.uint8) * 255).resize(
            (image_size, image_size), Image.NEAREST
        )).astype(bool)

        # Build click prompts
        current_pred = None
        for click_i in range(num_clicks):
            points_t, _, _ = build_click_prompt(
                gt_resized, current_pred, 1, (image_size, image_size), device
            )

            output = model(
                frames=frame_t,
                points=points_t,
                boxes=None,
                masks=None,
                recent_memories=[],
                prompted_memories=[],
                object_pointers=[],
                multimask_output=(click_i == 0),
            )

            pred_masks = output["masks"]
            iou_pred = output["iou_pred"]
            best_mask = select_best_mask(pred_masks, iou_pred)
            current_pred = mask_to_binary(best_mask)

        pred_masks_all.append(current_pred)
        gt_masks_all.append(gt_resized)

    return compute_miou(pred_masks_all, gt_masks_all)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM 2 Evaluation")
    parser.add_argument("--mode", choices=["offline", "online", "vos", "image"], default="vos")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="DAVIS")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--encoder_variant", type=str, default="B+")
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--num_clicks", type=int, default=3)
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--prompt_type", type=str, default="mask",
                        choices=["mask", "1-click", "3-click", "5-click", "box"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="./eval_output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("sam2.eval")
    device = torch.device(args.device)

    cfg = get_config(args.encoder_variant)
    model = build_sam2(
        variant=args.encoder_variant,
        image_size=args.image_size,
        embed_dim=cfg.embed_dim,
        memory_dim=cfg.memory_dim,
        num_memory_attention_layers=cfg.num_memory_attention_layers,
        max_recent_frames=cfg.max_recent_frames,
    ).to(device)

    load_checkpoint(args.checkpoint, model, strict=True)
    model.eval()
    logger.info(f"Loaded checkpoint from {args.checkpoint}")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "vos":
        # Semi-supervised VOS evaluation
        from data.dataset import VOSDataset
        dataset = VOSDataset(
            args.data_root, split=args.split,
            image_size=args.image_size, num_frames=100
        )
        all_jf = []
        for sample in tqdm(dataset.samples, desc="Evaluating VOS"):
            frames = [np.array(Image.open(f).convert("RGB")) for f in sample["frames"]]
            gt_masks = []
            for ann_path in sample["annotations"]:
                if os.path.exists(ann_path):
                    ann = np.array(Image.open(ann_path))
                    gt_masks.append(ann > 0)
                else:
                    gt_masks.append(np.zeros((frames[0].shape[0], frames[0].shape[1]), dtype=bool))

            scores = evaluate_semisupervised_vos(
                model, frames, gt_masks, device,
                prompt_type=args.prompt_type,
                image_size=args.image_size,
            )
            all_jf.append(scores["JF"])

        mean_jf = np.mean(all_jf)
        logger.info(f"Semi-supervised VOS J&F: {mean_jf:.4f}")

    elif args.mode == "offline":
        from data.dataset import VOSDataset
        dataset = VOSDataset(args.data_root, split=args.split, image_size=args.image_size, num_frames=100)
        all_jf_curves = []
        for sample in tqdm(dataset.samples[:50], desc="Evaluating offline"):
            frames = [np.array(Image.open(f).convert("RGB")) for f in sample["frames"]]
            gt_masks = []
            for ann_path in sample["annotations"]:
                if os.path.exists(ann_path):
                    ann = np.array(Image.open(ann_path))
                    gt_masks.append(ann > 0)
                else:
                    gt_masks.append(np.zeros((frames[0].shape[0], frames[0].shape[1]), dtype=bool))

            result = evaluate_interactive_offline(
                model, frames, gt_masks, device,
                num_clicks=args.num_clicks,
                max_frames=args.max_frames,
                image_size=args.image_size,
            )
            all_jf_curves.append(result["jf_per_nframes"])

        mean_curve = np.mean(all_jf_curves, axis=0)
        for i, jf in enumerate(mean_curve):
            logger.info(f"N_frames={i+1}: J&F={jf:.4f}")

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
