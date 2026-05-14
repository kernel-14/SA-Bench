"""
SAM 2 Trainer.

Implements the training procedure described in Appendix D.2:
- Pre-training on SA-1B (static images)
- Full training on mixed video + image data
- Alternating training strategy: batches sampled from either
  image or video datasets proportionally to data source size
- Fine-tuning stage with 16-frame sequences

Optimization (from Table 12):
- AdamW optimizer with beta1=0.9, beta2=0.999
- Weight decay: 0.1
- Gradient clipping: l2 norm, max 0.1
- Reciprocal square-root LR schedule (timescale=1000)
- Linear warmup (1k iterations) and cooldown (5k iterations)
- Layer-wise decay: 0.8 (T,S), 0.9 (B+), 0.925 (L)
- Drop path: 0.1 (T,S), 0.2 (B+), 0.3 (L)
- Precision: bfloat16
- Batch size: 256 (pre-training), 128 (full training)
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional, Dict, List, Any
import math
import time
from collections import defaultdict

from ..model.sam2 import SAM2
from .losses import SAM2Loss
from .interactive_sampler import InteractivePromptSampler


def get_layerwise_lr_groups(
    model: SAM2,
    base_lr: float,
    layer_decay: float,
) -> List[Dict]:
    """
    Create parameter groups with layer-wise learning rate decay.
    Layers closer to input get lower learning rates.
    """
    # Group parameters by depth
    param_groups = []
    # Image encoder (deepest layers get highest LR)
    encoder_params = list(model.image_encoder.named_parameters())
    encoder_params.reverse()  # Start from deepest layer

    num_layers = len(encoder_params)
    for i, (name, param) in enumerate(encoder_params):
        lr_scale = layer_decay ** (num_layers - i - 1)
        param_groups.append({
            "params": [param],
            "lr": base_lr * lr_scale,
            "name": f"encoder.{name}",
        })

    # Other parts get base learning rate
    other_params = []
    for name, param in model.named_parameters():
        if not name.startswith("image_encoder"):
            other_params.append(param)

    param_groups.append({
        "params": other_params,
        "lr": base_lr,
        "name": "other",
    })

    return param_groups


def reciprocal_sqrt_schedule(
    step: int,
    warmup_steps: int = 1000,
    cooldown_steps: int = 5000,
    total_steps: int = 90000,
    timescale: float = 1000.0,
) -> float:
    """Reciprocal square-root learning rate schedule with warmup and cooldown."""
    if step < warmup_steps:
        # Linear warmup
        return step / max(1, warmup_steps)
    elif step > total_steps - cooldown_steps:
        # Linear cooldown
        progress = (total_steps - step) / max(1, cooldown_steps)
        warmup_end_lr = 1.0 / math.sqrt(max(1, (warmup_steps / timescale)))
        return warmup_end_lr * progress
    else:
        # Reciprocal sqrt
        return 1.0 / math.sqrt(max(1, step / timescale))


class SAM2Trainer:
    """
    Trainer for SAM 2.

    Manages pre-training and full training phases.
    """

    def __init__(
        self,
        model: SAM2,
        loss_fn: Optional[SAM2Loss] = None,
        base_lr: float = 4e-4,
        weight_decay: float = 0.1,
        max_grad_norm: float = 0.1,
        warmup_steps: int = 1000,
        cooldown_steps: int = 5000,
        total_steps: int = 90000,
        timescale: float = 1000.0,
        layer_decay: float = 0.9,
        device: str = "cuda",
        use_amp: bool = True,
    ):
        self.model = model
        self.device = device
        self.use_amp = use_amp

        if loss_fn is None:
            loss_fn = SAM2Loss()
        self.loss_fn = loss_fn

        # Setup optimizer with layer-wise decay
        param_groups = get_layerwise_lr_groups(model, base_lr, layer_decay)
        self.optimizer = AdamW(
            param_groups,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        # LR scheduler
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: reciprocal_sqrt_schedule(
                step, warmup_steps, cooldown_steps, total_steps, timescale
            ),
        )

        self.max_grad_norm = max_grad_norm
        self.total_steps = total_steps
        self.current_step = 0

        # Metrics tracking
        self.metrics = defaultdict(list)

        # Move model to device
        self.model.to(device)

    def train_step_image(
        self,
        images: torch.Tensor,
        gt_masks: torch.Tensor,
        points: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Single training step on image data (SA-1B style).
        Simulates interactive prompting on a single frame.
        """
        self.model.train()
        self.current_step += 1

        images = images.to(self.device)
        gt_masks = gt_masks.to(self.device)
        B = images.shape[0]

        with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=torch.bfloat16):
            # Forward pass
            output = self.model(
                frame=images,
                points=(points, labels) if points is not None else None,
                boxes=boxes,
                masks=gt_masks[:, 0] if gt_masks.dim() == 4 else None,  # use first mask
                multimask_output=True,
                is_first_frame=True,
            )

            # Compute loss
            total_loss, loss_dict = self.loss_fn(
                pred_masks=output["masks"],
                pred_iou=output["iou_predictions"],
                pred_occlusion=output["occlusion_prediction"],
                gt_masks=gt_masks if gt_masks.dim() == 4 else gt_masks.unsqueeze(1),
                gt_present=torch.ones(B, device=self.device),
            )

        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()

        return {k: v.item() for k, v in loss_dict.items()}

    def train_step_video(
        self,
        frames: torch.Tensor,
        gt_masks: torch.Tensor,
        prompt_sampler: InteractivePromptSampler,
    ) -> Dict[str, float]:
        """
        Single training step on video data.
        Uses interactive prompt sampling for a sequence of frames.
        """
        self.model.train()
        self.current_step += 1

        frames = frames.to(self.device)
        gt_masks = gt_masks.to(self.device)
        T, C, H, W = frames.shape
        B = 1  # Single video per step (batch of frames)

        # Sample training sequence
        seq = prompt_sampler.sample_training_sequence(frames, gt_masks)
        seq_frames = seq["frames"]  # [num_frames, 3, H, W]
        seq_masks = seq["gt_masks"]  # [num_frames, H, W]

        total_loss = 0.0
        all_loss_dicts = []

        with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=torch.bfloat16):
            # Process frames sequentially, building up memory
            self.model._initialize_state()
            for t in range(seq_frames.shape[0]):
                frame = seq_frames[t:t+1]
                gt_mask = seq_masks[t:t+1]

                is_prompted = t in seq["prompted_frames"]

                # Prepare prompts
                points = None
                boxes = None
                mask_prompts = None

                if is_prompted:
                    if seq["initial_prompt"]["frame_idx"] == t:
                        prompt = seq["initial_prompt"]
                        if prompt["type"] == "mask":
                            mask_prompts = prompt["mask"].unsqueeze(0)
                        elif prompt["type"] == "click":
                            points = (
                                prompt["points"].unsqueeze(0),
                                prompt["labels"].unsqueeze(0).long(),
                            )
                        elif prompt["type"] == "box":
                            boxes = prompt["box"].unsqueeze(0).unsqueeze(0)

                output = self.model(
                    frame=frame,
                    points=points,
                    boxes=boxes,
                    masks=mask_prompts,
                    multimask_output=True,
                    is_first_frame=(t == 0),
                )

                # Determine if object is present in this frame
                gt_present = (gt_mask.max() > 0.5).float()

                # Compute loss for this frame
                loss, loss_dict = self.loss_fn(
                    pred_masks=output["masks"],
                    pred_iou=output["iou_predictions"],
                    pred_occlusion=output["occlusion_prediction"],
                    gt_masks=gt_mask.unsqueeze(1),
                    gt_present=gt_present,
                )
                total_loss = total_loss + loss
                all_loss_dicts.append(loss_dict)

        # Average losses
        avg_loss_dict = {}
        for key in all_loss_dicts[0].keys():
            avg_loss_dict[key] = sum(d[key] for d in all_loss_dicts) / len(all_loss_dicts)

        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()

        return {k: v.item() for k, v in avg_loss_dict.items()}

    def train_epoch(
        self,
        dataloader: Any,
        is_video: bool = False,
        prompt_sampler: Optional[InteractivePromptSampler] = None,
        log_every: int = 100,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_metrics = defaultdict(list)

        for batch_idx, batch in enumerate(dataloader):
            if is_video:
                metrics = self.train_step_video(
                    batch["frames"],
                    batch["masks"],
                    prompt_sampler,
                )
            else:
                metrics = self.train_step_image(
                    batch["images"],
                    batch["masks"],
                    points=batch.get("points"),
                    labels=batch.get("labels"),
                    boxes=batch.get("boxes"),
                )

            for k, v in metrics.items():
                epoch_metrics[k].append(v)
                self.metrics[k].append(v)

            if batch_idx % log_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"Step {self.current_step}/{self.total_steps}, "
                      f"LR: {lr:.6f}, Loss: {metrics.get('total_loss', 'N/A')}")

        # Average epoch metrics
        avg_metrics = {k: sum(v) / len(v) for k, v in epoch_metrics.items()}
        return avg_metrics

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "current_step": self.current_step,
            "metrics": dict(self.metrics),
        }, path)
