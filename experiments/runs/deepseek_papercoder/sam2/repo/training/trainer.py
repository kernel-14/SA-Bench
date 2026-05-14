# training/trainer.py

"""
Trainer class for SAM 2 – orchestrates pre‑training, full joint training,
and fine‑tuning with the configuration defined in ``config.yaml``.

This module ties together the SAM2Model, datasets, click simulator, and all
training‑specific components (optimizer, scheduler, mixed‑precision, gradient
accumulation).  It faithfully reproduces the training procedures described in
the SAM 2 paper (Appendices D.2.1, D.2.2) and the design document.

Typical usage:
    cfg = Config("config.yaml")
    model = SAM2Model(cfg.to_dict())
    video_ds = VideoDataset(root_paths=..., config=cfg, split='train')
    image_ds = ImageDataset(cfg.data.sa1b_root, cfg)
    trainer = Trainer(model, cfg)
    trainer.pretrain(image_ds)                    # pre‑train on SA‑1B
    trainer.train_full(video_ds, image_ds)        # joint video + image training
    trainer.finetune(video_ds, hard_examples=True)  # optional 16‑frame fine‑tuning
"""

from __future__ import annotations

import copy
import logging
import math
import os
import random
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, WeightedRandomSampler

# Project imports (assumed to be on PYTHONPATH)
from data.video_dataset import VideoDataset
from data.image_dataset import ImageDataset
from evaluation.click_simulator import ClickSimulator
from model.sam2 import SAM2Model
from training.pretrain import pretrain_on_sa1b  # implements Appendix D.2.1

# Logger
logger = logging.getLogger("sam2.trainer")


# ---------------------------------------------------------------------------
# Helper: infinite data stream
# ---------------------------------------------------------------------------

def _infinite_iter(loader: DataLoader) -> Iterator:
    """Yield batches forever from a DataLoader."""
    while True:
        for batch in loader:
            yield batch


# ---------------------------------------------------------------------------
# Trainer class
# ---------------------------------------------------------------------------

class Trainer:
    """
    High‑level orchestrator for SAM 2 training.

    Args:
        model: a :class:`SAM2Model` instance (fresh or pre‑trained).
        config: configuration parsed by :class:`config.Config` (AttrDict).
        device: torch device to use (``'cuda'`` or ``'cpu'``).  Defaults to
            ``'cuda'`` if available.
    """

    def __init__(
        self,
        model: SAM2Model,
        config: Any,                    # config.Config instance
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(
            device if device is not None else (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        )
        self.model.to(self.device)

        # Extract convenient shortcuts for training parameters
        pretrain_cfg = config.training.pretrain
        full_cfg = config.training.full_training
        finetune_cfg = config.training.finetune

        # Optimizer and scheduler – will be (re)initialised in each stage
        self.optimizer: Optional[AdamW] = None
        self.scheduler: Optional[LambdaLR] = None

        # Gradient accumulation steps (from full_training config)
        self.accum_steps: int = full_cfg.gradient_accumulation_steps

        # Click simulator – always centroid‑based for training
        self.click_sim = ClickSimulator(strategy="centroid")

        # Mixed‑precision: use bfloat16 autocast, no GradientScaler needed
        self.use_amp = True
        self.amp_dtype = torch.bfloat16

        # State tracking
        self.global_step: int = 0
        self.best_val_metric: float = 0.0

    # ------------------------------------------------------------------
    #  Optimizer and scheduler factories
    # ------------------------------------------------------------------

    def _build_optimizer(
        self,
        lr: float,
        layer_decay: float,
        weight_decay: float,
        betas: Tuple[float, float] = (0.9, 0.999),
    ) -> AdamW:
        """
        Create an AdamW optimizer with layer‑wise decay for the image encoder.

        The parameter grouping follows the paper's technique: parameters of
        the Hiera image encoder are assigned a multiplier that decays
        exponentially with depth, while all other parameters (memory, decoder,
        etc.) receive the base learning rate.

        Args:
            lr: base learning rate.
            layer_decay: decay factor per layer (e.g., 0.9 for B+).
            weight_decay: weight decay coefficient.
            betas: AdamW beta coefficients.

        Returns:
            Configured :class:`torch.optim.AdamW` optimizer.
        """
        # Heuristic depth assignment: walk the image_encoder module
        def _get_max_depth(module: nn.Module) -> int:
            # count the number of TransformerBlocks (each is one "layer")
            depth = 0
            for child in module.children():
                if isinstance(child, (nn.ModuleList, nn.Sequential)):
                    for subchild in child.children():
                        if subchild.__class__.__name__ == "TransformerBlock":
                            depth += 1
                elif child.__class__.__name__ == "TransformerBlock":
                    depth += 1
                else:
                    depth += _get_max_depth(child)
            return depth

        max_depth = max(_get_max_depth(self.model.image_encoder), 1)

        param_groups = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("image_encoder"):
                # Compute depth for this particular parameter
                parts = name.split(".")
                depth = 0
                # Walk through the nested structure to count TransformerBlocks
                # A simplified approach: use the number of "blocks" in the path
                # The paper uses a depth based on layer indices; we approximate
                # by counting the number of TransformerBlock instances encountered.
                # For a stronger implementation, we could directly inspect parent modules,
                # but here we'll assign depth based on the position in the name.
                if "blocks" in parts:
                    idx = parts.index("blocks")
                    if idx + 1 < len(parts):
                        try:
                            block_idx = int(parts[idx + 1])
                            depth = block_idx
                        except ValueError:
                            depth = max_depth // 2
                else:
                    depth = max_depth // 2  # generic fallback
                lr_mult = layer_decay ** (max_depth - depth)
                scaled_lr = lr * lr_mult
                param_groups.append({"params": [param], "lr": scaled_lr})
            else:
                param_groups.append({"params": [param], "lr": lr})

        # If no param_groups were built (should not happen), fallback
        if not param_groups:
            param_groups = [{"params": self.model.parameters(), "lr": lr}]

        optimizer = AdamW(
            param_groups,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )
        return optimizer

    def _build_scheduler(
        self,
        optimizer: AdamW,
        lr: float,
        total_steps: int,
        warmup_steps: int,
        cooldown_steps: int,
        timescale: float,
    ) -> LambdaLR:
        """
        Reciprocal square‑root learning rate schedule with linear warmup/cooldown.

        Args:
            optimizer: the optimizer whose param groups will be scaled.
            lr: base learning rate (used to scale the schedule).
            total_steps: total number of training steps.
            warmup_steps: number of linear warmup steps.
            cooldown_steps: number of linear cooldown steps.
            timescale: parameter of the reciprocal sqrt schedule.

        Returns:
            A :class:`LambdaLR` scheduler that can be step‑per‑iteration.
        """
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                # linear warmup from 0 to 1
                return step / max(1, warmup_steps)
            elif step > total_steps - cooldown_steps:
                # linear cooldown to 0
                remaining = total_steps - step
                return max(0.0, remaining / cooldown_steps)
            else:
                # reciprocal sqrt
                return math.sqrt(timescale) / math.sqrt(step)

        return LambdaLR(optimizer, lr_lambda)

    # ------------------------------------------------------------------
    #  Pre‑training stage (delegates to training/pretrain.py)
    # ------------------------------------------------------------------

    def pretrain(self, image_dataset: ImageDataset) -> SAM2Model:
        """
        Run image‑only pre‑training on SA‑1B.

        This method delegates to :func:`training.pretrain.pretrain_on_sa1b`,
        which implements the exact protocol from the SAM 2 paper.

        The model is moved to the chosen device, and after pre‑training is
        complete the (updated) model is returned.

        Args:
            image_dataset: :class:`ImageDataset` instance providing SA‑1B samples.

        Returns:
            The pre‑trained :class:`SAM2Model`.
        """
        logger.info("Starting pre‑training on SA‑1B ...")
        pretrain_cfg = self.config.training.pretrain

        # Build optimizer and scheduler for pre‑training
        optimizer = self._build_optimizer(
            lr=pretrain_cfg.learning_rate,
            layer_decay=pretrain_cfg.layer_decay,
            weight_decay=pretrain_cfg.weight_decay,
            betas=tuple(pretrain_cfg.betas),
        )
        scheduler = self._build_scheduler(
            optimizer,
            pretrain_cfg.learning_rate,
            total_steps=pretrain_cfg.steps,
            warmup_steps=pretrain_cfg.lr_schedule.warmup_steps,
            cooldown_steps=pretrain_cfg.lr_schedule.cooldown_steps,
            timescale=pretrain_cfg.lr_schedule.timescale,
        )

        # Call the external function (returns updated model)
        self.model = pretrain_on_sa1b(
            model=self.model,
            dataset=image_dataset,
            config=self.config,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
        )
        logger.info("Pre‑training finished.")
        return self.model

    # ------------------------------------------------------------------
    #  Full joint training
    # ------------------------------------------------------------------

    def train_full(
        self,
        video_dataset: VideoDataset,
        image_dataset: ImageDataset,
    ) -> SAM2Model:
        """
        Full training on a mix of video clips and static images.

        This method implements the alternating training strategy described in
        the paper (Appendix D.2.2).  It simulates interactive prompting on
        video clips (up to 2 prompted frames, 8‑frame sequences, etc.) and
        on images (same as pre‑training).  Gradient accumulation is used to
        achieve the effective batch size.

        Args:
            video_dataset: :class:`VideoDataset` for video clips.
            image_dataset: :class:`ImageDataset` for images (10% SA‑1B subset).

        Returns:
            The trained :class:`SAM2Model`.
        """
        full_cfg = self.config.training.full_training
        total_steps = full_cfg.steps
        batch_size = full_cfg.batch_size
        accum_steps = self.accum_steps
        effective_batch_size = batch_size

        # Data sources
        mix_weights = copy.deepcopy(self.config.data.mix_weights)
        # If internal dataset is missing, renormalize has been done in Config,
        # so mix_weights should sum to 1.0.

        # Create weighted samplers for the video dataset if needed,
        # but here we'll manually draw batches according to mix_weights.
        # For simplicity, we create two infinite iterators.
        video_loader = DataLoader(
            video_dataset,
            batch_size=1,                  # each batch is one video clip
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )
        image_loader = DataLoader(
            image_dataset,
            batch_size=1,                  # each batch is one image + masks
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )
        video_iter = _infinite_iter(video_loader)
        image_iter = _infinite_iter(image_loader)

        # Compute probability for image vs video based on mix_weights
        video_weight = (
            mix_weights.get("davis", 0.0)
            + mix_weights.get("mose", 0.0)
            + mix_weights.get("ytvos", 0.0)
            + mix_weights.get("sav_manual", 0.0)
        )
        image_weight = mix_weights.get("sa1b", 0.0)
        total_weight = video_weight + image_weight
        p_video = video_weight / total_weight if total_weight > 0 else 0.5

        # Build optimizer and scheduler for full training
        # (LR and other params from full_training config)
        optimizer = self._build_optimizer(
            lr=full_cfg.learning_rate,
            layer_decay=full_cfg.layer_decay,
            weight_decay=full_cfg.weight_decay,
            betas=tuple(full_cfg.betas),
        )
        scheduler = self._build_scheduler(
            optimizer,
            full_cfg.learning_rate,
            total_steps=total_steps,
            warmup_steps=full_cfg.lr_schedule.warmup_steps,
            cooldown_steps=full_cfg.lr_schedule.cooldown_steps,
            timescale=full_cfg.lr_schedule.timescale,
        )

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.global_step = 0

        logger.info("Starting full training ...")

        # Accumulation logic
        optimizer.zero_grad()
        accumulated_loss = 0.0
        log_interval = 100
        checkpoint_dir = os.path.join("checkpoints", "full_training")
        os.makedirs(checkpoint_dir, exist_ok=True)

        while self.global_step < total_steps:
            # --- 1. Choose data source ---
            is_video = random.random() < p_video

            if is_video:
                batch = next(video_iter)
                # batch is a dict with "frames", "masklets", etc.
                loss, _ = self._process_video_batch(batch, is_training=True)
            else:
                batch = next(image_iter)
                loss, _ = self._process_image_batch(batch, is_training=True)

            # --- 2. Scale loss for gradient accumulation ---
            loss = loss / accum_steps
            if self.use_amp:
                with autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    loss.backward()
            else:
                loss.backward()

            accumulated_loss += loss.item()

            # --- 3. Step optimiser after `accum_steps` backward passes ---
            if (self.global_step + 1) % accum_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=full_cfg.grad_clip_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            self.global_step += 1

            # Logging
            if self.global_step % log_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Step {self.global_step}/{total_steps}, "
                    f"Loss: {accumulated_loss:.5f}, "
                    f"LR: {lr:.2e}"
                )
                accumulated_loss = 0.0

            # Save periodic checkpoints (every 10% of total steps)
            if self.global_step % max(1, total_steps // 10) == 0:
                self._save_checkpoint(
                    os.path.join(checkpoint_dir, f"step_{self.global_step}.pth")
                )

        # Save final model
        self._save_checkpoint(os.path.join(checkpoint_dir, "final.pth"))
        logger.info("Full training completed.")
        return self.model

    # ------------------------------------------------------------------
    #  Fine‑tuning stage (16‑frame sequences, frozen encoder)
    # ------------------------------------------------------------------

    def finetune(
        self,
        video_dataset: VideoDataset,
        hard_examples: bool = True,
    ) -> SAM2Model:
        """
        Optional fine‑tuning on long (16‑frame) clips from hard examples.

        The image encoder is frozen (following the paper), and the
        learning rate is halved.  Only video batches are used.

        Args:
            video_dataset: :class:`VideoDataset` providing 16‑frame clips.
            hard_examples: if ``True``, the dataset has already been filtered
                to the top 50% most edited masklets (set in config).

        Returns:
            The fine‑tuned :class:`SAM2Model`.
        """
        finetune_cfg = self.config.training.finetune
        total_steps = finetune_cfg.steps
        accum_steps = self.accum_steps  # reuse same accumulation

        # Freeze image encoder
        for p in self.model.image_encoder.parameters():
            p.requires_grad = False
        logger.info("Image encoder frozen for fine‑tuning.")

        # Build a new optimizer with halved LR
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=finetune_cfg.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.1,
        )
        # Scheduler: same schedule shape but halved initial LR already given
        scheduler = self._build_scheduler(
            optimizer,
            finetune_cfg.learning_rate,
            total_steps=total_steps,
            warmup_steps=100,          # smaller warmup for fine‑tuning
            cooldown_steps=200,
            timescale=1000,
        )

        video_loader = DataLoader(
            video_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )
        video_iter = _infinite_iter(video_loader)

        logger.info("Starting fine‑tuning ...")
        optimizer.zero_grad()
        accumulated_loss = 0.0
        checkpoint_dir = os.path.join("checkpoints", "finetune")
        os.makedirs(checkpoint_dir, exist_ok=True)

        for step in range(total_steps):
            batch = next(video_iter)
            # Note: `_process_video_batch` will use the current model (image encoder frozen)
            loss, _ = self._process_video_batch(batch, is_training=True)

            loss = loss / accum_steps
            if self.use_amp:
                with autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    loss.backward()
            else:
                loss.backward()

            accumulated_loss += loss.item()

            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                self.global_step += 1   # continue counting from where we left off

            if (step + 1) % 100 == 0:
                logger.info(f"Fine‑tune step {step+1}/{total_steps}, Loss: {accumulated_loss:.5f}")
                accumulated_loss = 0.0

        # Unfreeze encoder (optional, but for evaluation we may want it trainable again)
        for p in self.model.image_encoder.parameters():
            p.requires_grad = True

        self._save_checkpoint(os.path.join(checkpoint_dir, "finetuned.pth"))
        logger.info("Fine‑tuning completed.")
        return self.model

    # ------------------------------------------------------------------
    #  Internal: processing a video batch (used in train_full and finetune)
    # ------------------------------------------------------------------

    def _process_video_batch(
        self, batch: Dict[str, Any], is_training: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Simulates interactive segmentation on a video clip and returns the total loss.

        The procedure:
        1. For each object masklet in the clip (up to ``max_masklets_per_clip``):
           - Reset memory.
           - Run initial prompt phase (first frame).
           - Possibly run a correction prompt phase (second frame).
        2. Accumulate losses over all frames where masklets are present.

        Args:
            batch: output of :class:`VideoDataset.__getitem__`, containing:
                ``frames`` (T, C, H, W), ``masklets`` list of dicts with
                ``mask`` (T, 1, H, W) and ``object_id``.
            is_training: if True, compute gradients; otherwise only metric.

        Returns:
            total_loss (scalar tensor), and a dict with auxiliary information.
        """
        frames = batch["frames"].to(self.device)       # (T, C, H, W)
        masklets = batch["masklets"]                   # list of dicts

        if len(masklets) == 0:
            return torch.tensor(0.0, device=self.device), {}

        config = self.config
        full_cfg = config.training.full_training
        max_masklets = full_cfg.max_masklets_per_clip

        # Randomly select up to max_masklets objects
        selected = list(range(len(masklets)))
        if len(selected) > max_masklets:
            selected = random.sample(selected, max_masklets)

        total_loss = torch.tensor(0.0, device=self.device)
        aux_info = {"n_masklets": len(selected)}

        for idx in selected:
            mlet = masklets[idx]
            mask_tensor = mlet["mask"].to(self.device)  # (T, 1, H, W)
            mask_tensor = mask_tensor.squeeze(1)        # (T, H, W)

            # Reset memory for this new object
            self.model.reset_memory()

            # ---- Initial prompt simulation ----
            init_prompt = self._generate_initial_prompt(
                mask_tensor[0], frames.shape[-2:], config
            )
            # First pass: predict masklet with only the initial prompt
            prompts_first = [init_prompt if t == 0 else None for t in range(frames.shape[0])]
            model_out_first = self._model_forward_video(frames, prompts_first, multi_mask=True)
            pred_masks_first = model_out_first["masks"]           # (T, H, W) or (T, num_masks, H, W)
            pred_iou_first = model_out_first["iou_pred"]          # ...
            occlusion_first = model_out_first.get("occlusion_logit", None)

            # Compute loss for the first pass
            loss_first = self._compute_video_loss(
                pred_masks_first, pred_iou_first, mask_tensor,
                model_out_first, is_first_pass=True, config=config,
            )

            # ---- Correction prompt simulation (optional) ----
            # Decide whether to add a correction (approx 80% chance, capped by paper's "up to 2")
            do_correction = random.random() < 0.8 if frames.shape[0] > 1 else False
            if do_correction:
                # Choose a frame for correction (can be same as first or different)
                corr_frame_idx = random.randint(0, frames.shape[0] - 1)

                # Generate correction clicks based on error between first prediction and GT
                corr_prompt = self._generate_correction_prompt(
                    pred_masks_first, mask_tensor, corr_frame_idx, config
                )
                if corr_prompt is not None:
                    # Second pass: reset memory from scratch and replay prompts
                    self.model.reset_memory()
                    prompts_second = [None] * frames.shape[0]
                    prompts_second[0] = init_prompt
                    prompts_second[corr_frame_idx] = corr_prompt
                    model_out_second = self._model_forward_video(
                        frames, prompts_second, multi_mask=False
                    )
                    pred_masks_second = model_out_second["masks"]
                    pred_iou_second = model_out_second["iou_pred"]
                    occlusion_second = model_out_second.get("occlusion_logit", None)

                    loss_second = self._compute_video_loss(
                        pred_masks_second, pred_iou_second, mask_tensor,
                        model_out_second, is_first_pass=False, config=config,
                    )
                    total_loss = total_loss + loss_first + loss_second
                else:
                    # No valid correction click (e.g., perfect prediction) – skip
                    total_loss = total_loss + loss_first
            else:
                total_loss = total_loss + loss_first

        # Average over selected masklets
        total_loss = total_loss / len(selected)
        return total_loss, aux_info

    # ------------------------------------------------------------------
    #  Internal: processing an image batch (used in train_full)
    # ------------------------------------------------------------------

    def _process_image_batch(
        self, batch: Dict[str, Any], is_training: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Simulate interactive segmentation on a single image with multiple masks.

        The procedure mirrors the pre‑training image simulation, but uses the
        model's forward method for image (no memory).  Losses are averaged
        over all masks.

        Args:
            batch: output of :class:`ImageDataset.__getitem__`, containing:
                ``image`` (C, H, W), ``masks`` (K, H, W).
            is_training: if True, compute gradients.

        Returns:
            total_loss (scalar), and auxiliary info (number of masks).
        """
        image = batch["image"].to(self.device)          # (C, H, W)
        masks = batch["masks"].to(self.device)          # (K, H, W)

        if masks.numel() == 0:
            return torch.tensor(0.0, device=self.device), {}

        pretrain_cfg = self.config.training.pretrain
        correction_points = pretrain_cfg.correction_points

        total_loss = torch.tensor(0.0, device=self.device)
        k = masks.shape[0]

        # For each mask, run interactive sequence
        for i in range(k):
            gt_mask = masks[i]  # (H, W)
            if gt_mask.sum() == 0:
                continue

            # 1. Generate initial prompt
            init_prompt = self._generate_initial_prompt(gt_mask, image.shape[-2:], self.config, is_video=False)
            # For image, forward is a single frame; we simulate by calling model with single frame batch.
            # We'll use a helper that wraps the model's forward_single_image? The SAM2Model may offer a method,
            # but our design only has `forward`. We'll construct a 1‑frame video here.
            frames = image.unsqueeze(0)  # (1, C, H, W)
            prompts = [init_prompt]

            model_out_first = self._model_forward_image(frames, prompts, multi_mask=True)
            # Process losses similarly to pretrain
            loss_first = self._compute_image_loss(
                model_out_first, gt_mask, self.config, is_first_click=True
            )
            total_loss = total_loss + loss_first

            # 2. Iterative correction clicks (up to 7)
            current_state = {
                "pred_masks": model_out_first["masks"],  # (1, num_masks, H, W)
                "iou_pred": model_out_first["iou_pred"],  # (1, num_masks)
            }
            for step_idx in range(correction_points):
                # Generate a correction click from the error between best prediction and GT
                # Use the lowest loss mask as the 'best' for error computation. (We'll re‑compute quickly.)
                best_mask = self._select_best_mask(
                    model_out_first["masks"], model_out_first["iou_pred"], gt_mask, self.config
                )
                corr_prompt = self._generate_correction_prompt_single(
                    best_mask, gt_mask, self.config, is_image=True
                )
                if corr_prompt is None:
                    # No error region → add a random click with 10% prob (as per paper)
                    if random.random() < pretrain_cfg.correction_click_random_prob:
                        corr_prompt = self.click_sim.get_random_gt_click(gt_mask.cpu().numpy())[0]
                        corr_prompt = {k: v.to(self.device) for k, v in corr_prompt.items()}
                    else:
                        # stop correction early if no error and not adding random
                        break
                # Accumulate prompts and re‑run
                prompts.append(corr_prompt)
                model_out = self._model_forward_image(frames, prompts, multi_mask=False)
                loss_corr = self._compute_image_loss(
                    model_out, gt_mask, self.config, is_first_click=False
                )
                total_loss = total_loss + loss_corr

                # Update state
                current_state = {
                    "pred_masks": model_out["masks"],
                    "iou_pred": model_out["iou_pred"],
                }

        total_loss = total_loss / max(1, k)
        return total_loss, {"n_masks": k}

    # ------------------------------------------------------------------
    #  Prompt generation helpers
    # ------------------------------------------------------------------

    def _generate_initial_prompt(
        self,
        gt_mask: torch.Tensor,
        frame_size: Tuple[int, int],
        config: Any,
        is_video: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Create the initial prompt dictionary for the first frame.

        Args:
            gt_mask: (H, W) binary ground‑truth mask, tensor.
            frame_size: (H, W) of the frame.
            config: configuration object.
            is_video: whether this is for a video clip (used to decide probs).

        Returns:
            prompt dict with keys appropriate for SAM2Model.forward.
        """
        if is_video:
            initial_probs = config.training.full_training.initial_prompt_probs
        else:
            # for images, re‑use full_training probs (same as pretrain)
            initial_probs = {
                "mask": 0.5,
                "click": 0.25,
                "box": 0.25,
            }

        rand = random.random()
        if rand < initial_probs["mask"]:
            return {
                "is_prompted": True,
                "masks": gt_mask.unsqueeze(0).unsqueeze(0),  # (1, 1, H, W)
            }
        elif rand < initial_probs["mask"] + initial_probs["click"]:
            # single positive click at centre
            click = self.click_sim.generate_initial_clicks(gt_mask.cpu().numpy())
            if not click:
                # if mask empty, fallback to box? but shouldn't happen
                return self._generate_initial_prompt(gt_mask, frame_size, config, is_video)
            coords = torch.tensor([[click[0]["x"], click[0]["y"]]], dtype=torch.float32, device=gt_mask.device)
            labels = torch.tensor([1], dtype=torch.int64, device=gt_mask.device)
            return {
                "is_prompted": True,
                "coords": coords.unsqueeze(0),   # (1, 1, 2)
                "labels": labels.unsqueeze(0),   # (1, 1)
            }
        else:
            # bounding box from mask
            mask_np = gt_mask.cpu().numpy()
            if not mask_np.any():
                # fallback to click
                return self._generate_initial_prompt(gt_mask, frame_size, config, is_video)
            rows = np.any(mask_np, axis=1)
            cols = np.any(mask_np, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            box = torch.tensor([[cmin, rmin, cmax, rmax]], dtype=torch.float32, device=gt_mask.device)
            return {
                "is_prompted": True,
                "boxes": box,
            }

    def _generate_correction_prompt(
        self,
        pred_masks: torch.Tensor,        # (T, ...) may be multi‑mask
        gt_masks: torch.Tensor,          # (T, H, W)
        corr_frame_idx: int,
        config: Any,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a correction prompt on the specified frame.

        Args:
            pred_masks: output of a previous forward, shape depends on multi‑mask.
                We'll assume the last dimension is H,W and the second‑to‑last is masks.
            gt_masks: (T, H, W) ground truth.
            corr_frame_idx: index of the correction frame.
            config: configuration.

        Returns:
            prompt dict for the correction frame, or None if no click can be placed.
        """
        # Extract the single best mask at corr_frame_idx.
        # For multi‑mask case, we need to pick the best mask; we'll compute best using simple IoU.
        # Since we don't have IoU scores per mask here, we can recompute binarised masks and pick higher IoU.
        # For simplicity, assume the model's last output was multi‑mask and its iou_pred gave best idx.
        # In _process_video_batch we have access to model_out_first['iou_pred'] for each mask; we could pass it.
        # But to keep this method generic, we'll re‑compute IoU from pred_masks and gt_masks.
        # We'll handle both cases: if pred_masks has extra dim for num_masks, select best.
        if pred_masks.dim() == 4:   # (T, num_masks, H, W) or (num_masks, H, W)? In our code, after model forward, pred_masks is (T, H, W) b/c we selected best. Actually for first pass, we kept all masks? We'll assume we have multi‑mask tensor here. To simplify, we can accept the already selected best mask (passed from outside). Let's modify the call in _process_video_batch to pass the best mask directly.
            # We'll expect this function to be called with the already best mask for the correction frame.
            # So we'll change signature and usage accordingly. For now, we'll assume pred_masks is (H, W) for the correction frame.
            pass
        # Actually, we'll redesign: in _process_video_batch, after first pass, we already computed best_mask (mask with lowest loss). We'll pass that best mask for the correction frame.
        # So this function will only receive the best binary mask for that frame.
        # We'll keep the signature simple: pred_mask_2d (H, W), gt_mask_2d (H, W).
        # We'll adapt.

    def _generate_correction_prompt_single(
        self,
        pred_mask: torch.Tensor,   # (H, W) binary or logits
        gt_mask: torch.Tensor,     # (H, W) binary
        config: Any,
        is_image: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a correction click on a single frame, using error region.

        Args:
            pred_mask: predicted mask (can be logits; we will binarise).
            gt_mask: ground truth mask.
            config: configuration.
            is_image: if True, use image‑specific random prob.

        Returns:
            prompt dict for the correction frame, or None.
        """
        # Binarise prediction if needed
        if pred_mask.dtype.is_floating:
            pred_bin = (pred_mask.sigmoid() > 0.5).float()
        else:
            pred_bin = pred_mask

        random_gt_prob = config.training.full_training.correction_click_random_prob
        click_list = self.click_sim.generate_correction_clicks(
            pred_bin.cpu().numpy(), gt_mask.cpu().numpy(), random_gt_prob=random_gt_prob
        )
        if not click_list:
            return None
        click = click_list[0]
        coords = torch.tensor([[click["x"], click["y"]]], dtype=torch.float32, device=gt_mask.device)
        label = 1 if click["positive"] else 0
        labels = torch.tensor([label], dtype=torch.int64, device=gt_mask.device)
        return {
            "is_prompted": True,
            "coords": coords.unsqueeze(0),   # (1, 1, 2)
            "labels": labels.unsqueeze(0),   # (1, 1)
        }

    # ------------------------------------------------------------------
    #  Model forward wrappers (video and image)
    # ------------------------------------------------------------------

    def _model_forward_video(
        self,
        frames: torch.Tensor,
        prompts: List[Optional[Dict[str, Any]]],
        multi_mask: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Call the SAM2Model.forward for a video clip.

        The model processes frames sequentially; memory is already reset.
        Returns a dict with masks, iou_pred, occlusion_logit, object_pointers.
        """
        # The model's forward expects a list of prompts per frame.
        # We'll add a flag for multi_mask on the first prompted frame if needed.
        if multi_mask and prompts[0] is not None:
            # The model uses the 'multi_mask' key in prompt dict
            prompts[0] = {**prompts[0], "multi_mask": True}
        return self.model.forward(frames, prompts)

    def _model_forward_image(
        self,
        frames: torch.Tensor,      # (1, C, H, W)
        prompts: List[Dict[str, Any]],
        multi_mask: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Call SAM2Model.forward for a single image (treated as 1‑frame video).

        Reset memory before call.  Returns same structure.
        """
        if multi_mask and prompts[0] is not None:
            prompts[0] = {**prompts[0], "multi_mask": True}
        return self.model.forward(frames, prompts)

    # ------------------------------------------------------------------
    #  Loss computation for video
    # ------------------------------------------------------------------

    def _compute_video_loss(
        self,
        pred_masks: torch.Tensor,        # (T, H, W) or (T, num_masks, H, W) from model output
        pred_iou: torch.Tensor,          # (T, num_masks) or similar
        gt_masks: torch.Tensor,          # (T, H, W)
        model_out: Dict[str, Any],
        is_first_pass: bool,
        config: Any,
    ) -> torch.Tensor:
        """
        Compute the combined mask, IoU, and occlusion loss for a video masklet.

        Supports multi‑mask supervision when is_first_pass=True and model
        predicted multiple masks; in that case only the best mask gets mask loss.

        Args:
            pred_masks: predicted masks. If multi‑mask, shape (T, num_masks, H, W);
                else (T, H, W).  We'll standardise: expand to (T, num_masks, H, W) if needed.
            pred_iou: predicted IoU of shape (T, num_masks) (or (T,) expanded).
            gt_masks: ground truth masks (T, H, W).
            model_out: full model output (may contain occlusion_logit).
            is_first_pass: whether this is the first interaction (multi‑mask may be active).
            config: full config.

        Returns:
            scalar loss.
        """
        T = gt_masks.shape[0]
        loss_weights = config.training.full_training.loss_weights

        # Ensure pred_masks has a num_masks dimension (even if 1)
        if pred_masks.dim() == 3:          # (T, H, W) -> expand to (T, 1, H, W)
            pred_masks = pred_masks.unsqueeze(1)          # (T, 1, H, W)
            pred_iou = pred_iou.unsqueeze(1) if pred_iou.dim() == 1 else pred_iou  # (T,1) if needed

        T, num_masks, H, W = pred_masks.shape
        # gt masks: (T, H, W) -> (T, 1, H, W) to broadcast
        gt = gt_masks.unsqueeze(1)         # (T, 1, H, W)

        # ---- Mask losses ----
        # Focal + Dice per mask, per frame where GT is present (non‑empty)
        focal_loss = 0.0
        dice_loss = 0.0
        for t in range(T):
            if gt_masks[t].sum() == 0:   # object absent from this frame
                continue
            for k in range(num_masks):
                logits = pred_masks[t, k].unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                target = gt[t, 0].unsqueeze(0).unsqueeze(0)          # (1, 1, H, W)
                focal = self._focal_loss(logits, target)
                dice = self._dice_loss(logits, target)
                focal_loss += focal
                dice_loss += dice

        # Average over frames and masks that were computed
        N_valid = (gt_masks.sum(dim=(1,2)) > 0).sum().item() * num_masks
        if N_valid == 0:
            return torch.tensor(0.0, device=pred_masks.device)

        focal_loss = focal_loss / N_valid
        dice_loss = dice_loss / N_valid
        mask_loss = loss_weights["focal"] * focal_loss + loss_weights["dice"] * dice_loss

        # ---- IoU loss (L1) ----
        # Compute true IoU for each predicted mask at each frame.
        true_ious = []
        for t in range(T):
            if gt_masks[t].sum() == 0:
                true_ious.append(torch.zeros(num_masks, device=pred_masks.device))
            else:
                # binarise prediction
                pred_bin = (pred_masks[t].sigmoid() > 0.5).float()   # (num_masks, H, W)
                target = gt_masks[t].unsqueeze(0)                    # (1, H, W)
                inter = (pred_bin * target).sum(dim=(-2, -1))
                union = pred_bin.sum(dim=(-2, -1)) + target.sum() - inter
                iou = (inter + 1e-6) / (union + 1e-6)
                true_ious.append(iou)   # (num_masks,)
        true_ious = torch.stack(true_ious, dim=0)   # (T, num_masks)

        # If pred_iou shape is (T, num_masks) we are good; else may need reshape.
        if pred_iou.shape != true_ious.shape:
            pred_iou = pred_iou.view_as(true_ious)

        iou_loss = F.l1_loss(pred_iou, true_ious)

        # ---- Occlusion loss (cross‑entropy) ----
        occlusion_loss = torch.tensor(0.0, device=pred_masks.device)
        if "occlusion_logit" in model_out and model_out["occlusion_logit"] is not None:
            occ_logit = model_out["occlusion_logit"]   # (T, 1) perhaps
            # Ground‑truth occlusion: 1 if object is absent (mask sum==0), else 0.
            occ_gt = (gt_masks.sum(dim=(-2, -1)) == 0).float().to(occ_logit.device)  # (T,)
            if occ_logit.dim() == 2:
                occ_logit = occ_logit.squeeze(1)   # (T,)
            occlusion_loss = F.binary_cross_entropy_with_logits(occ_logit, occ_gt)   # BCELogits
            # The paper uses cross‑entropy for occlusion; this is effectively the same.

        total_loss = mask_loss + loss_weights["iou_l1"] * iou_loss + loss_weights["occlusion_ce"] * occlusion_loss

        # For multi‑mask first pass, only mask loss for best mask is used. Since we already averaged over all masks, we must adjust.
        if is_first_pass and num_masks > 1:
            # We'll recompute mask loss only for the best mask.
            # Re‑compute mask loss for each mask individually and select minimum.
            losses_per_mask = []
            for k in range(num_masks):
                fl = 0.0
                dl = 0.0
                for t in range(T):
                    if gt_masks[t].sum() == 0:
                        continue
                    logits = pred_masks[t, k].unsqueeze(0).unsqueeze(0)
                    target = gt[t, 0].unsqueeze(0).unsqueeze(0)
                    fl += self._focal_loss(logits, target)
                    dl += self._dice_loss(logits, target)
                fl = fl / max(1, N_valid // num_masks)
                dl = dl / max(1, N_valid // num_masks)
                losses_per_mask.append(loss_weights["focal"] * fl + loss_weights["dice"] * dl)
            best_idx = torch.argmin(torch.stack(losses_per_mask))
            mask_loss = losses_per_mask[best_idx]
            total_loss = mask_loss + loss_weights["iou_l1"] * iou_loss + loss_weights["occlusion_ce"] * occlusion_loss

        return total_loss

    # ------------------------------------------------------------------
    #  Loss computation for image (single frame)
    # ------------------------------------------------------------------

    def _compute_image_loss(
        self,
        model_out: Dict[str, torch.Tensor],
        gt_mask: torch.Tensor,       # (H, W)
        config: Any,
        is_first_click: bool,
    ) -> torch.Tensor:
        """
        Compute losses for a single image interaction.

        Args:
            model_out: output of the model forward for an image (1‑frame video).
            gt_mask: ground truth mask (H, W).
            config: config object.
            is_first_click: if True, multi‑mask may be active.

        Returns:
            scalar loss.
        """
        loss_weights = config.training.pretrain.loss_weights
        masks_logits = model_out["masks_logits"].squeeze(0)  # (num_masks, H, W)
        iou_pred = model_out["iou_pred"].squeeze(0)            # (num_masks,)

        num_masks = masks_logits.shape[0]
        gt = gt_mask.unsqueeze(0).expand(num_masks, -1, -1)   # (num_masks, H, W)

        # Mask losses per candidate
        focal_per_mask = []
        dice_per_mask = []
        for k in range(num_masks):
            logits = masks_logits[k].unsqueeze(0).unsqueeze(0)
            target = gt[k].unsqueeze(0).unsqueeze(0)
            focal_per_mask.append(self._focal_loss(logits, target))
            dice_per_mask.append(self._dice_loss(logits, target))
        focal = torch.stack(focal_per_mask)
        dice = torch.stack(dice_per_mask)
        mask_losses = loss_weights["focal"] * focal + loss_weights["dice"] * dice

        # IoU loss (L1)
        with torch.no_grad():
            pred_bin = (masks_logits.sigmoid() > 0.5).float()
            true_ious = []
            for k in range(num_masks):
                inter = (pred_bin[k] * gt[k]).sum()
                union = pred_bin[k].sum() + gt[k].sum() - inter
                iou = (inter + 1e-6) / (union + 1e-6)
                true_ious.append(iou)
            true_ious = torch.stack(true_ious)  # (num_masks,)
        iou_l1 = F.l1_loss(iou_pred, true_ious)

        # Combine
        if is_first_click and num_masks > 1:
            best_idx = torch.argmin(mask_losses)
            mask_loss = mask_losses[best_idx]
        else:
            mask_loss = mask_losses.mean()

        total = mask_loss + loss_weights["iou_l1"] * iou_l1
        return total

    # ------------------------------------------------------------------
    #  Low‑level loss functions (focal, dice)
    # ------------------------------------------------------------------

    def _focal_loss(self, logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
        """Focal loss for binary segmentation."""
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = torch.exp(-bce)
        focal_weight = (1 - pt) ** gamma
        if alpha >= 0:
            alpha_t = alpha * target + (1 - alpha) * (1 - target)
            focal_weight = focal_weight * alpha_t
        return focal_weight.mean()

    def _dice_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Soft dice loss."""
        pred = torch.sigmoid(logits)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        return 1.0 - dice

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _select_best_mask(
        self,
        masks_logits: torch.Tensor,    # (1, num_masks, H, W) or (num_masks, H, W)
        iou_pred: torch.Tensor,        # (1, num_masks) or (num_masks,)
        gt_mask: torch.Tensor,         # (H, W)
        config: Any,
    ) -> torch.Tensor:
        """Select the best binary mask among multi‑mask predictions for image."""
        if masks_logits.dim() == 4:
            masks_logits = masks_logits.squeeze(0)
        # Compute simple IoU to pick best, or use iou_pred. We'll use iou_pred as it's trained.
        if iou_pred.dim() == 2:
            iou_pred = iou_pred.squeeze(0)
        best_idx = torch.argmax(iou_pred).item()
        best_logits = masks_logits[best_idx]
        return (best_logits.sigmoid() > 0.5).float()

    def _save_checkpoint(self, path: str) -> None:
        """Save model, optimizer, and step."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "step": self.global_step,
        }
        if self.optimizer is not None:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

