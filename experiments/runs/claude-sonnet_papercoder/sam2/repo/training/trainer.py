```python
## training/trainer.py
"""Full training pipeline for SAM 2: Stage 2 joint image/video training and
Stage 3 16-frame fine-tuning.

This module implements the Trainer class responsible for:
    - Stage 2: 200k-iteration joint image/video training with alternating
      batch sampling proportional to dataset size
    - Stage 3: 50k-iteration fine-tuning on 16-frame sequences with frozen
      image encoder and half learning rate

Config references (config.yaml training and finetuning sections):
    training.num_iterations: 200000
    training.num_frames: 8
    training.max_prompted_frames: 2
    training.max_masklets_per_sequence: 3
    training.temporal_reversal_prob: 0.50
    training.mosaic_prob: 0.10
    training.prompt_probabilities.*
    training.correction_click_random_prob: 0.10
    training.losses.*
    finetuning.num_iterations: 50000
    finetuning.num_frames: 16
    finetuning.learning_rate_multiplier: 0.5
    finetuning.freeze_image_encoder: true
    finetuning.most_edited_fraction: 0.50

Paper references:
    Section 4: "we sample sequences of 8 frames and randomly select up to 2
        frames to prompt"
    Appendix D.2.2: "in each training iteration, we sample a full batch either
        from the image or video dataset, with their sampling probabilities
        proportional to the size of each data source"
    Appendix D.2.2: "We reverse the temporal order with a probability of 50%"
    Appendix D.2.2: "With 10% probability, we tile the same training video
        into a 2×2 grid"
    Appendix D.2.2: "we sort our masklets by number of edited frames and only
        consider the top 50% most edited masklets for training"
"""

import logging
import os
import random
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from datasets import PromptInput, VideoSample
from datasets.prompt_sampler import PromptSampler
from models.memory_bank import MemoryBank, MemoryBankOutput
from models.sam2 import SAM2FrameOutput, SAM2Model
from training.losses import SAM2Losses
from training.optimizer import build_optimizer, build_scheduler
from utils.click_sampler import ClickSampler
from utils.mask_utils import MaskUtils
from utils.misc import (
    clip_gradients,
    ensure_dir,
    format_metrics,
    get_autocast_context,
    is_main_process,
    load_checkpoint,
    log_metrics,
    save_checkpoint,
    setup_logger,
    setup_tensorboard,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default hyperparameters from config.yaml training section
# ---------------------------------------------------------------------------

_DEFAULT_NUM_ITERATIONS: int = 200000
_DEFAULT_NUM_FRAMES: int = 8
_DEFAULT_MAX_PROMPTED_FRAMES: int = 2
_DEFAULT_MAX_MASKLETS: int = 3
_DEFAULT_TEMPORAL_REVERSAL_PROB: float = 0.50
_DEFAULT_MOSAIC_PROB: float = 0.10
_DEFAULT_GT_MASK_PROB: float = 0.50
_DEFAULT_CLICK_PROB: float = 0.25
_DEFAULT_BOX_PROB: float = 0.25
_DEFAULT_CORRECTION_RANDOM_PROB: float = 0.10
_DEFAULT_FOCAL_WEIGHT: float = 20.0
_DEFAULT_DICE_WEIGHT: float = 1.0
_DEFAULT_IOU_WEIGHT: float = 1.0
_DEFAULT_OCCLUSION_WEIGHT: float = 1.0
_DEFAULT_BASE_LR: float = 4.0e-4
_DEFAULT_WEIGHT_DECAY: float = 0.1
_DEFAULT_GRAD_CLIP_MAX: float = 0.1
_DEFAULT_GRAD_CLIP_TYPE: str = "l2"
_DEFAULT_TIMESCALE: int = 1000
_DEFAULT_WARMUP_ITERS: int = 1000
_DEFAULT_COOLDOWN_ITERS: int = 5000
_DEFAULT_PRECISION: str = "bfloat16"
_DEFAULT_ENCODER_TYPE: str = "hiera_b_plus"
_DEFAULT_SA1B_FRACTION: float = 0.155
_DEFAULT_LOG_INTERVAL: int = 50
_DEFAULT_CHECKPOINT_INTERVAL: int = 5000

# Fine-tuning defaults
_DEFAULT_FINETUNE_ITERATIONS: int = 50000
_DEFAULT_FINETUNE_NUM_FRAMES: int = 16
_DEFAULT_FINETUNE_LR_MULTIPLIER: float = 0.5
_DEFAULT_FINETUNE_MOST_EDITED_FRACTION: float = 0.50


class Trainer:
    """Joint image/video trainer for SAM 2 Stage 2 and Stage 3.

    Manages the complete training pipeline including:
        - Alternating image/video batch sampling proportional to dataset size
        - Interactive prompt simulation over 8-frame video sequences
        - Mosaic transform (10% probability) for small-object training
        - Temporal reversal (50% probability) for bi-directional generalization
        - Multi-object inference with shared encoder features
        - bfloat16 mixed precision training
        - Checkpoint save/load for training resumption
        - 16-frame fine-tuning with frozen image encoder

    Args:
        model: Initialized SAM2Model instance. Should be loaded with
            pre-training checkpoint before Stage 2 training.
        config: Training configuration dict from config.yaml. Expected to
            contain 'training' and 'finetuning' sub-dicts.
        device: Target device string (e.g., "cuda:0", "cuda", "cpu").
        image_dataloader: DataLoader for SA-1B image batches. Each batch is
            a dict with "image", "masks", "valid_mask", "num_masks" keys.
        video_dataloader: DataLoader for video batches (SA-V + DAVIS + MOSE
            + YouTubeVOS). Each batch is a VideoSample.
        checkpoint_dir: Directory for saving training checkpoints.
            Defaults to "checkpoints/train".
        log_dir: Directory for TensorBoard event files.
            Defaults to "logs/train".
        log_interval: Log metrics every this many steps. Defaults to 50.
        checkpoint_interval: Save checkpoint every this many steps.
            Defaults to 5000.

    Example:
        config = OmegaConf.to_container(cfg, resolve=True)
        model = SAM2Model(SAM2Config.from_dict(cfg))
        trainer = Trainer(
            model=model,
            config=config,
            device="cuda:0",
            image_dataloader=sa1b_loader,
            video_dataloader=video_loader,
        )
        trainer.train(num_iterations=200000)
        trainer.finetune_16frames(num_iterations=50000)
    """

    def __init__(
        self,
        model: SAM2Model,
        config: Dict[str, Any],
        device: str = "cuda",
        image_dataloader: Optional[DataLoader] = None,
        video_dataloader: Optional[DataLoader] = None,
        checkpoint_dir: str = "checkpoints/train",
        log_dir: str = "logs/train",
        log_interval: int = _DEFAULT_LOG_INTERVAL,
        checkpoint_interval: int = _DEFAULT_CHECKPOINT_INTERVAL,
    ) -> None:
        self.model: SAM2Model = model
        self.config: Dict[str, Any] = config
        self.device: str = device
        self.image_dataloader: Optional[DataLoader] = image_dataloader
        self.video_dataloader: Optional[DataLoader] = video_dataloader
        self.checkpoint_dir: str = checkpoint_dir
        self.log_dir: str = log_dir
        self.log_interval: int = log_interval
        self.checkpoint_interval: int = checkpoint_interval

        # Move model to device
        self.model = self.model.to(device)

        # ------------------------------------------------------------------
        # Extract hyperparameters from config with defaults
        # ------------------------------------------------------------------
        training_cfg: Dict[str, Any] = config.get("training", config)
        finetuning_cfg: Dict[str, Any] = config.get("finetuning", {})
        optimizer_cfg: Dict[str, Any] = training_cfg.get("optimizer", {})
        scheduler_cfg: Dict[str, Any] = training_cfg.get("scheduler", {})
        losses_cfg: Dict[str, Any] = training_cfg.get("losses", {})
        prompt_probs_cfg: Dict[str, Any] = training_cfg.get(
            "prompt_probabilities", {}
        )
        layer_decay_cfg: Dict[str, Any] = config.get(
            "pretrain", {}
        ).get("layer_wise_decay", {})

        # Training hyperparameters
        self.num_frames: int = int(
            training_cfg.get("num_frames", _DEFAULT_NUM_FRAMES)
        )
        self.max_prompted_frames: int = int(
            training_cfg.get("max_prompted_frames", _DEFAULT_MAX_PROMPTED_FRAMES)
        )
        self.max_masklets_per_sequence: int = int(
            training_cfg.get("max_masklets_per_sequence", _DEFAULT_MAX_MASKLETS)
        )
        self.temporal_reversal_prob: float = float(
            training_cfg.get("temporal_reversal_prob", _DEFAULT_TEMPORAL_REVERSAL_PROB)
        )
        self.mosaic_prob: float = float(
            training_cfg.get("mosaic_prob", _DEFAULT_MOSAIC_PROB)
        )
        self.correction_click_random_prob: float = float(
            training_cfg.get(
                "correction_click_random_prob", _DEFAULT_CORRECTION_RANDOM_PROB
            )
        )
        self.precision: str = str(
            training_cfg.get("precision", _DEFAULT_PRECISION)
        )

        # Data mixture: probability of sampling an image batch vs video batch
        data_mixture: Dict[str, Any] = training_cfg.get(
            "data_mixture_with_oss",
            training_cfg.get("data_mixture_released", {}),
        )
        self.sa1b_fraction: float = float(
            data_mixture.get("sa1b_fraction", _DEFAULT_SA1B_FRACTION)
        )
        # p_image = sa1b_fraction; p_video = 1 - sa1b_fraction
        self.p_image: float = self.sa1b_fraction
        self.p_video: float = 1.0 - self.sa1b_fraction

        # Optimizer hyperparameters
        self.base_lr: float = float(
            optimizer_cfg.get("learning_rate", _DEFAULT_BASE_LR)
        )
        self.weight_decay: float = float(
            optimizer_cfg.get("weight_decay", _DEFAULT_WEIGHT_DECAY)
        )
        self.beta1: float = float(optimizer_cfg.get("beta1", 0.9))
        self.beta2: float = float(optimizer_cfg.get("beta2", 0.999))
        self.grad_clip_max: float = float(
            optimizer_cfg.get("gradient_clip_max", _DEFAULT_GRAD_CLIP_MAX)
        )
        self.grad_clip_type: str = str(
            optimizer_cfg.get("gradient_clip_type", _DEFAULT_GRAD_CLIP_TYPE)
        )

        # Scheduler hyperparameters
        self.timescale: int = int(
            scheduler_cfg.get("timescale", _DEFAULT_TIMESCALE)
        )
        self.warmup_iters: int = int(
            scheduler_cfg.get("warmup_iters", _DEFAULT_WARMUP_ITERS)
        )
        self.cooldown_iters: int = int(
            scheduler_cfg.get("cooldown_iters", _DEFAULT_COOLDOWN_ITERS)
        )

        # Loss weights
        self.focal_weight: float = float(
            losses_cfg.get("focal_weight", _DEFAULT_FOCAL_WEIGHT)
        )
        self.dice_weight: float = float(
            losses_cfg.get("dice_weight", _DEFAULT_DICE_WEIGHT)
        )
        self.iou_weight: float = float(
            losses_cfg.get("iou_weight", _DEFAULT_IOU_WEIGHT)
        )
        self.occlusion_weight: float = float(
            losses_cfg.get("occlusion_weight", _DEFAULT_OCCLUSION_WEIGHT)
        )

        # Prompt probabilities
        self.gt_mask_prob: float = float(
            prompt_probs_cfg.get("gt_mask", _DEFAULT_GT_MASK_PROB)
        )
        self.click_prob: float = float(
            prompt_probs_cfg.get("positive_click", _DEFAULT_CLICK_PROB)
        )
        self.box_prob: float = float(
            prompt_probs_cfg.get("bounding_box", _DEFAULT_BOX_PROB)
        )

        # Encoder type for layer-wise decay
        model_cfg: Dict[str, Any] = config.get("model", {})
        self.encoder_type: str = str(
            model_cfg.get("image_encoder_type", _DEFAULT_ENCODER_TYPE)
        )

        # Fine-tuning hyperparameters
        self.finetune_num_iterations: int = int(
            finetuning_cfg.get("num_iterations", _DEFAULT_FINETUNE_ITERATIONS)
        )
        self.finetune_num_frames: int = int(
            finetuning_cfg.get("num_frames", _DEFAULT_FINETUNE_NUM_FRAMES)
        )
        self.finetune_lr_multiplier: float = float(
            finetuning_cfg.get(
                "learning_rate_multiplier", _DEFAULT_FINETUNE_LR_MULTIPLIER
            )
        )
        self.finetune_freeze_encoder: bool = bool(
            finetuning_cfg.get("freeze_image_encoder", True)
        )
        self.finetune_most_edited_fraction: float = float(
            finetuning_cfg.get(
                "most_edited_fraction", _DEFAULT_FINETUNE_MOST_EDITED_FRACTION
            )
        )

        # ------------------------------------------------------------------
        # Build optimizer with layer-wise LR decay
        # ------------------------------------------------------------------
        self.optimizer: torch.optim.AdamW = build_optimizer(
            model=self.model,
            base_lr=self.base_lr,
            weight_decay=self.weight_decay,
            beta1=self.beta1,
            beta2=self.beta2,
            encoder_type=self.encoder_type,
            layer_decay_rates=layer_decay_cfg if layer_decay_cfg else None,
            lr_multiplier=1.0,
        )

        # ------------------------------------------------------------------
        # Build LR scheduler (will be rebuilt with correct total_steps in train())
        # ------------------------------------------------------------------
        self.scheduler: Optional[LambdaLR] = None

        # ------------------------------------------------------------------
        # Instantiate loss function
        # ------------------------------------------------------------------
        self.losses: SAM2Losses = SAM2Losses(
            focal_weight=self.focal_weight,
            dice_weight=self.dice_weight,
            iou_weight=self.iou_weight,
            occlusion_weight=self.occlusion_weight,
        )

        # ------------------------------------------------------------------
        # Instantiate prompt sampler and click sampler
        # ------------------------------------------------------------------
        self.prompt_sampler: PromptSampler = PromptSampler(
            gt_mask_prob=self.gt_mask_prob,
            click_prob=self.click_prob,
            box_prob=self.box_prob,
            correction_click_random_prob=self.correction_click_random_prob,
        )
        self.click_sampler: ClickSampler = ClickSampler()
        self.mask_utils: MaskUtils = MaskUtils()

        # ------------------------------------------------------------------
        # TensorBoard writer (main process only)
        # ------------------------------------------------------------------
        self.writer: Optional[SummaryWriter] = None
        if is_main_process():
            ensure_dir(log_dir)
            ensure_dir(checkpoint_dir)
            self.writer = setup_tensorboard(log_dir=log_dir, rank=0)

        # Training state
        self.global_step: int = 0
        self._best_loss: float = float("inf")

        # Infinite dataloader iterators (initialized lazily in train())
        self._image_iter: Optional[Iterator] = None
        self._video_iter: Optional[Iterator] = None

        logger.info(
            "Trainer initialized: encoder_type=%s, base_lr=%.2e, "
            "p_image=%.3f, p_video=%.3f, num_frames=%d, precision=%s",
            self.encoder_type,
            self.base_lr,
            self.p_image,
            self.p_video,
            self.num_frames,
            self.precision,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, num_iterations: int = _DEFAULT_NUM_ITERATIONS) -> None:
        """Run the full Stage 2 joint image/video training loop.

        Iterates for num_iterations steps, alternating between image and
        video batches with probability proportional to dataset size.

        Args:
            num_iterations: Total number of training iterations.
                Defaults to 200000 (config: training.num_iterations).
        """
        # Build scheduler with correct total_steps
        self.scheduler = build_scheduler(
            optimizer=self.optimizer,
            total_steps=num_iterations,
            warmup_iters=self.warmup_iters,
            cooldown_iters=self.cooldown_iters,
            timescale=self.timescale,
            last_epoch=-1,
        )

        # Resume from checkpoint if available
        latest_ckpt: str = os.path.join(self.checkpoint_dir, "latest.pth")
        if os.path.isfile(latest_ckpt):
            logger.info("Trainer: Resuming from checkpoint %s", latest_ckpt)
            start_step: int = self.load_checkpoint(latest_ckpt)
            self.global_step = start_step
            logger.info("Trainer: Resumed from step %d", start_step)
        else:
            start_step = 0
            self.global_step = 0

        # Initialize infinite dataloader iterators
        self._image_iter = self._make_infinite_iter(self.image_dataloader)
        self._video_iter = self._make_infinite_iter(self.video_dataloader)

        self.model.train()

        logger.info(
            "Trainer: Starting training from step %d to %d",
            start_step,
            num_iterations,
        )

        for step in range(start_step, num_iterations):
            self.global_step = step

            # ------------------------------------------------------------------
            # Decide modality: image or video
            # Sampling probability proportional to dataset size
            # ------------------------------------------------------------------
            use_image: bool = (random.random() < self.p_image)

            try:
                if use_image and self._image_iter is not None:
                    batch = next(self._image_iter)
                    batch = self._move_batch_to_device_dict(batch)
                    loss_dict: Dict[str, float] = self._train_image_step(batch)
                    loss_dict["modality"] = 0.0  # 0 = image for logging
                elif self._video_iter is not None:
                    batch = next(self._video_iter)
                    batch = self._move_video_sample_to_device(batch)
                    loss_dict = self._train_video_step(batch)
                    loss_dict["modality"] = 1.0  # 1 = video for logging
                else:
                    logger.warning(
                        "Trainer: No dataloader available at step %d. Skipping.",
                        step,
                    )
                    continue
            except StopIteration:
                # Reinitialize exhausted iterator
                if use_image:
                    self._image_iter = self._make_infinite_iter(
                        self.image_dataloader
                    )
                    batch = next(self._image_iter)
                    batch = self._move_batch_to_device_dict(batch)
                    loss_dict = self._train_image_step(batch)
                    loss_dict["modality"] = 0.0
                else:
                    self._video_iter = self._make_infinite_iter(
                        self.video_dataloader
                    )
                    batch = next(self._video_iter)
                    batch = self._move_video_sample_to_device(batch)
                    loss_dict = self._train_video_step(batch)
                    loss_dict["modality"] = 1.0
            except Exception as exc:
                logger.warning(
                    "Trainer: Training step %d failed: %s. Skipping.",
                    step,
                    exc,
                )
                self.optimizer.zero_grad()
                continue

            # ------------------------------------------------------------------
            # Gradient clipping, optimizer step, scheduler step
            # ------------------------------------------------------------------
            grad_norm: float = clip_gradients(
                model=self.model,
                clip_type=self.grad_clip_type,
                clip_max=self.grad_clip_max,
            )
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()

            loss_dict["grad_norm"] = grad_norm
            loss_dict["lr"] = self._get_current_lr()

            # ------------------------------------------------------------------
            # Logging
            # ------------------------------------------------------------------
            if step % self.log_interval == 0 and is_main_process():
                log_metrics(
                    writer=self.writer,
                    metrics=loss_dict,
                    step=step,
                    prefix="train",
                )
                logger.info(
                    "Train step %d/%d | %s",
                    step,
                    num_iterations,
                    format_metrics(loss_dict),
                )

            # ------------------------------------------------------------------
            # Checkpoint saving
            # ------------------------------------------------------------------
            if step % self.checkpoint_interval == 0 and is_main_process():
                is_best: bool = (
                    loss_dict.get("total", float("inf")) < self._best_loss
                )
                if is_best:
                    self._best_loss = loss_dict.get("total", float("inf"))
                self.save_checkpoint(path=self.checkpoint_dir, step=step)
                logger.info("Trainer: Saved checkpoint at step %d", step)

        # Final checkpoint
        if is_main_process():
            self.save_checkpoint(
                path=self.checkpoint_dir, step=num_iterations
            )
            logger.info(
                "Trainer: Stage 2 training complete. Final checkpoint saved."
            )

        if self.writer is not None:
            self.writer.close()

    def _train_video_step(
        self, batch: VideoSample
    ) -> Dict[str, float]:
        """Execute a single training step on one video batch.

        Processes an 8-frame video sequence with interactive prompt simulation,
        running the full streaming inference pipeline per object.

        Args:
            batch: VideoSample with frames [T, C, H, W], masks [T, N, H, W],
                is_occluded List[bool], num_objects int.

        Returns:
            Dict[str, float] with keys: "total", "focal", "dice", "iou",
            "occlusion". All values are Python floats.
        """
        # ------------------------------------------------------------------
        # Step 1: Apply mosaic transform (10% probability, train only)
        # ------------------------------------------------------------------
        if random.random() < self.mosaic_prob:
            batch = self._apply_mosaic_transform(batch)

        # ------------------------------------------------------------------
        # Step 2: Apply temporal reversal (50% probability)
        # ------------------------------------------------------------------
        if random.random() < self.temporal_reversal_prob:
            batch = self._apply_temporal_reversal(batch)

        frames: Tensor = batch.frames  # [T, C, H, W]
        masks: Tensor = batch.masks    # [T, N, H, W]
        T: int = frames.shape[0]
        N: int = masks.shape[1]

        # ------------------------------------------------------------------
        # Step 3: Select up to max_masklets_per_sequence objects
        # ------------------------------------------------------------------
        num_objects: int = batch.num_objects
        if num_objects > self.max_masklets_per_sequence:
            selected_obj_indices: List[int] = random.sample(
                range(num_objects), self.max_masklets_per_sequence
            )
        else:
            selected_obj_indices = list(range(num_objects))

        # ------------------------------------------------------------------
        # Step 4: Cache image encoder outputs for all T frames
        # (shared across all objects — encoder runs once per frame)
        # ------------------------------------------------------------------
        autocast_ctx = get_autocast_context(self.precision)

        frame_embeds: List[Tensor] = []
        skip_features_list: List[List[Tensor]] = []

        with autocast_ctx:
            for t in range(T):
                frame_t: Tensor = frames[t].unsqueeze(0)  # [1, C, H, W]
                fe, sf = self.model.forward_image(frame_t)
                frame_embeds.append(fe)
                skip_features_list.append(sf)

        # ------------------------------------------------------------------
        # Step 5: Process each selected object independently
        # ------------------------------------------------------------------
        total_focal: float = 0.0
        total_dice: float = 0.0
        total_iou: float = 0.0
        total_occlusion: float = 0.0
        total_combined: float = 0.0
        num_processed: int = 0

        for obj_idx in selected_obj_indices:
            # Extract GT masks for this object: [T, H, W]
            gt_masks_obj: Tensor = masks[:, obj_idx, :, :]  # [T, H, W]

            # Extract occlusion flags for this object
            # is_occluded is flat: [T * N], indexed as [t * N + obj_idx]
            gt_occ_obj: List[bool] = [
                batch.is_occluded[t * N + obj_idx] for t in range(T)
            ]

            # ------------------------------------------------------------------
            # Step 6: Simulate prompted frames for this object
            # ------------------------------------------------------------------
            prompted_frames: List[Tuple[int, PromptInput]] = (
                self._simulate_video_prompts(
                    gt_masks_obj=gt_masks_obj,
                    gt_occ_obj=gt_occ_obj,
                    T=T,
                )
            )
            prompted_frame_dict: Dict[int, PromptInput] = {
                fi: pi for fi, pi in prompted_frames
            }

            # ------------------------------------------------------------------
            # Step 7: Reset memory bank for this object
            # ------------------------------------------------------------------
            self.model.reset_memory()

            # ------------------------------------------------------------------
            # Step 8: Sequential forward pass over T frames
            # ------------------------------------------------------------------
            obj_total_loss: Tensor = torch.zeros(
                1, device=self.device, dtype=torch.float32
            )
            obj_focal: float = 0.0
            obj_dice: float = 0.0
            obj_iou: float = 0.0
            obj_occ: float = 0.0
            frames_with_loss: int = 0

            # Track current prediction for correction click sampling
            current_pred_mask: Optional[Tensor] = None

            for t in range(T):
                frame_embed_t: Tensor = frame_embeds[t]  # [1, C, H/16, W/16]
                skip_features_t: List[Tensor] = skip_features_list[t]

                # Determine prompt for this frame
                prompt_t: Optional[PromptInput] = None
                is_prompted: bool = t in prompted_frame_dict

                if is_prompted:
                    if t == 0:
                        # Initial prompt: sampled before the loop
                        prompt_t = prompted_frame_dict[t]
                    else:
                        # Correction prompt: sample based on current prediction
                        gt_mask_t: Tensor = gt_masks_obj[t]  # [H, W]
                        if current_pred_mask is not None:
                            prompt_t = self.prompt_sampler.sample_correction_clicks(
                                gt_mask=gt_mask_t,
                                pred_mask=current_pred_mask.squeeze(),
                                num_clicks=3,
                                frame_idx=t,
                            )
                        else:
                            # No prediction yet — use initial prompt strategy
                            prompt_t = self.prompt_sampler.sample_initial_prompt(
                                gt_mask=gt_mask_t,
                                frame_idx=t,
                            )

                # ------------------------------------------------------------------
                # Forward pass for this frame
                # ------------------------------------------------------------------
                with autocast_ctx:
                    frame_output: SAM2FrameOutput = self.model.forward_video_frame(
                        frame_embed=frame_embed_t,
                        skip_features=skip_features_t,
                        prompts=prompt_t,
                        memory_bank=self.model.memory_bank,
                    )

                    # Update memory bank with current prediction
                    self.model._update_memory_bank(
                        frame_embed=frame_embed_t,
                        frame_output=frame_output,
                        memory_bank=self.model.memory_bank,
                        is_prompted=is_prompted,
                        frame_idx=t,
                    )

                    # Update current prediction for next correction click
                    selected_idx: int = frame_output.selected_mask_idx
                    current_pred_mask = frame_output.masks[
                        :, selected_idx, :, :
                    ]  # [1, H, W]

                    # ------------------------------------------------------------------
                    # Compute loss for this frame
                    # ------------------------------------------------------------------
                    gt_mask_t_for_loss: Tensor = gt_masks_obj[t].unsqueeze(0)
                    # [1, H, W]

                    # Resize predicted masks to match GT if needed
                    pred_masks_t: Tensor = frame_output.masks  # [1, N_masks, H, W]
                    if pred_masks_t.shape[-2:] != gt_mask_t_for_loss.shape[-2:]:
                        pred_masks_t = F.interpolate(
                            pred_masks_t,
                            size=gt_mask_t_for_loss.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )

                    # GT occlusion for this frame
                    gt_occ_t: Tensor = torch.tensor(
                        [1.0 if gt_occ_obj[t] else 0.0],
                        dtype=torch.float32,
                        device=self.device,
                    )

                    # GT mask is None if occluded (no mask supervision)
                    gt_mask_for_loss: Optional[Tensor] = (
                        None if gt_occ_obj[t] else gt_mask_t_for_loss
                    )

                    loss_result: Dict[str, Tensor] = self.losses.compute_total_loss(
                        pred_masks=pred_masks_t,
                        gt