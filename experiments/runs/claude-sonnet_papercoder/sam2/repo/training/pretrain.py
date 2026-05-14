```python
## training/pretrain.py
"""Pre-training pipeline for SAM 2 on SA-1B dataset (Stage 1).

This module implements the Pretrainer class responsible for Stage 1 pre-training
of SAM 2 on the SA-1B dataset. This stage trains the model on static images only,
establishing strong image segmentation capabilities before joint video/image
training in Stage 2.

Key design decisions:
    - Memory bank is empty during pre-training (image-as-single-frame-video)
    - 7 correction clicks per image (vs 8 in SAM, per Appendix D.2.1)
    - No no-prompt iterations (unlike SAM's 2 no-prompt iterations)
    - Multi-mask output on first click only (ambiguous single click)
    - Occlusion head exists but is not supervised (all SA-1B images are visible)
    - bfloat16 mixed precision throughout

Config references (config.yaml pretrain section):
    pretrain.steps: 90000
    pretrain.batch_size: 256
    pretrain.resolution: 1024
    pretrain.precision: "bfloat16"
    pretrain.optimizer.learning_rate: 4.0e-4
    pretrain.optimizer.weight_decay: 0.1
    pretrain.optimizer.gradient_clip_max: 0.1
    pretrain.scheduler.timescale: 1000
    pretrain.scheduler.warmup_iters: 1000
    pretrain.scheduler.cooldown_iters: 5000
    pretrain.layer_wise_decay.hiera_b_plus: 0.9
    pretrain.losses.focal_weight: 20
    pretrain.losses.dice_weight: 1
    pretrain.losses.iou_weight: 1
    pretrain.data.max_masks_per_image: 64
    pretrain.data.mask_area_filter: 0.90
    pretrain.interactive_clicks.num_correction_clicks: 7
    training.correction_click_random_prob: 0.10

Paper references:
    Appendix D.2.1: "The image encoder is initialized from MAE pre-trained Hiera."
    Appendix D.2.1: "we use 7 correction clicks (instead of 8 in SAM)"
    Appendix D.2.1: "we do not add such iterations during our training"
    Appendix D.2.1: "for multi-mask predictions (on the first click), we supervise
        the IoU predictions of all masks ... but only supervise the mask logits
        with the lowest segmentation loss"
    Table 12: Full hyperparameter table for pre-training.
"""

import logging
import os
import random
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import PromptInput
from datasets.prompt_sampler import PromptSampler
from models.memory_bank import MemoryBank
from models.sam2 import SAM2Model
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
    load_hiera_mae_weights,
    log_metrics,
    save_checkpoint,
    setup_logger,
    setup_tensorboard,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default hyperparameters from config.yaml pretrain section
# ---------------------------------------------------------------------------

_DEFAULT_TOTAL_STEPS: int = 90000
_DEFAULT_BATCH_SIZE: int = 256
_DEFAULT_RESOLUTION: int = 1024
_DEFAULT_PRECISION: str = "bfloat16"
_DEFAULT_BASE_LR: float = 4.0e-4
_DEFAULT_WEIGHT_DECAY: float = 0.1
_DEFAULT_GRAD_CLIP_MAX: float = 0.1
_DEFAULT_GRAD_CLIP_TYPE: str = "l2"
_DEFAULT_TIMESCALE: int = 1000
_DEFAULT_WARMUP_ITERS: int = 1000
_DEFAULT_COOLDOWN_ITERS: int = 5000
_DEFAULT_FOCAL_WEIGHT: float = 20.0
_DEFAULT_DICE_WEIGHT: float = 1.0
_DEFAULT_IOU_WEIGHT: float = 1.0
_DEFAULT_NUM_CORRECTION_CLICKS: int = 7
_DEFAULT_CORRECTION_CLICK_RANDOM_PROB: float = 0.10
_DEFAULT_MASK_THRESHOLD: float = 0.0
_DEFAULT_LOG_INTERVAL: int = 50
_DEFAULT_CHECKPOINT_INTERVAL: int = 5000
_DEFAULT_ENCODER_TYPE: str = "hiera_b_plus"


class Pretrainer:
    """Pre-trainer for SAM 2 Stage 1: SA-1B image pre-training.

    Manages the complete pre-training pipeline including:
        - MAE pre-trained Hiera weight initialization
        - AdamW optimizer with layer-wise LR decay on image encoder
        - Reciprocal sqrt LR schedule with warmup and cooldown
        - Interactive click simulation (7 correction clicks per image)
        - Multi-mask supervision on first click
        - bfloat16 mixed precision training
        - Checkpoint save/load for training resumption

    The memory bank is empty throughout pre-training — the model processes
    each image as a single-frame video with no temporal context, behaving
    identically to SAM on static images.

    Args:
        model: Initialized SAM2Model instance. The image encoder backbone
            will be loaded with MAE pre-trained Hiera weights if
            mae_checkpoint_path is provided.
        config: Pre-training configuration dict from config.yaml pretrain section.
            Expected keys: steps, batch_size, resolution, precision, optimizer.*,
            scheduler.*, layer_wise_decay.*, losses.*, data.*, interactive_clicks.*
        device: Target device string (e.g., "cuda:0", "cuda", "cpu").
        mae_checkpoint_path: Optional path to MAE pre-trained Hiera checkpoint.
            If provided, loads backbone weights before training begins.
        checkpoint_dir: Directory for saving training checkpoints.
            Defaults to "checkpoints/pretrain".
        log_dir: Directory for TensorBoard event files.
            Defaults to "logs/pretrain".
        log_interval: Log metrics every this many steps. Defaults to 50.
        checkpoint_interval: Save checkpoint every this many steps.
            Defaults to 5000.
        accumulation_steps: Gradient accumulation steps for effective batch size.
            Effective batch size = micro_batch_size × accumulation_steps.
            Defaults to 1 (no accumulation).

    Example:
        config = OmegaConf.to_container(cfg.pretrain, resolve=True)
        model = SAM2Model(SAM2Config.from_dict(cfg))
        pretrainer = Pretrainer(
            model=model,
            config=config,
            device="cuda:0",
            mae_checkpoint_path="/checkpoints/hiera_base_plus.pth",
        )
        pretrainer.pretrain(dataloader)
    """

    def __init__(
        self,
        model: SAM2Model,
        config: Dict[str, Any],
        device: str = "cuda",
        mae_checkpoint_path: Optional[str] = None,
        checkpoint_dir: str = "checkpoints/pretrain",
        log_dir: str = "logs/pretrain",
        log_interval: int = _DEFAULT_LOG_INTERVAL,
        checkpoint_interval: int = _DEFAULT_CHECKPOINT_INTERVAL,
        accumulation_steps: int = 1,
    ) -> None:
        self.model: SAM2Model = model
        self.config: Dict[str, Any] = config
        self.device: str = device
        self.checkpoint_dir: str = checkpoint_dir
        self.log_dir: str = log_dir
        self.log_interval: int = log_interval
        self.checkpoint_interval: int = checkpoint_interval
        self.accumulation_steps: int = max(1, accumulation_steps)

        # Move model to device
        self.model = self.model.to(device)

        # ------------------------------------------------------------------
        # Extract hyperparameters from config with defaults
        # ------------------------------------------------------------------
        optimizer_cfg: Dict[str, Any] = config.get("optimizer", {})
        scheduler_cfg: Dict[str, Any] = config.get("scheduler", {})
        losses_cfg: Dict[str, Any] = config.get("losses", {})
        layer_decay_cfg: Dict[str, Any] = config.get("layer_wise_decay", {})
        interactive_cfg: Dict[str, Any] = config.get("interactive_clicks", {})

        self.total_steps: int = int(config.get("steps", _DEFAULT_TOTAL_STEPS))
        self.precision: str = str(config.get("precision", _DEFAULT_PRECISION))
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
        self.timescale: int = int(
            scheduler_cfg.get("timescale", _DEFAULT_TIMESCALE)
        )
        self.warmup_iters: int = int(
            scheduler_cfg.get("warmup_iters", _DEFAULT_WARMUP_ITERS)
        )
        self.cooldown_iters: int = int(
            scheduler_cfg.get("cooldown_iters", _DEFAULT_COOLDOWN_ITERS)
        )
        self.focal_weight: float = float(
            losses_cfg.get("focal_weight", _DEFAULT_FOCAL_WEIGHT)
        )
        self.dice_weight: float = float(
            losses_cfg.get("dice_weight", _DEFAULT_DICE_WEIGHT)
        )
        self.iou_weight: float = float(
            losses_cfg.get("iou_weight", _DEFAULT_IOU_WEIGHT)
        )
        self.num_correction_clicks: int = int(
            interactive_cfg.get(
                "num_correction_clicks", _DEFAULT_NUM_CORRECTION_CLICKS
            )
        )
        self.correction_click_random_prob: float = float(
            config.get(
                "correction_click_random_prob",
                _DEFAULT_CORRECTION_CLICK_RANDOM_PROB,
            )
        )
        self.mask_threshold: float = float(
            config.get("mask_threshold", _DEFAULT_MASK_THRESHOLD)
        )

        # Encoder type for layer-wise decay lookup
        self.encoder_type: str = str(
            config.get("encoder_type", _DEFAULT_ENCODER_TYPE)
        )

        # ------------------------------------------------------------------
        # Step 1: Load MAE pre-trained Hiera weights into image encoder
        # ------------------------------------------------------------------
        if mae_checkpoint_path is not None and os.path.isfile(mae_checkpoint_path):
            logger.info(
                "Pretrainer: Loading MAE pre-trained Hiera weights from %s",
                mae_checkpoint_path,
            )
            load_hiera_mae_weights(
                model=self.model,
                hiera_checkpoint_path=mae_checkpoint_path,
                device=device,
            )
        elif mae_checkpoint_path is not None:
            logger.warning(
                "Pretrainer: MAE checkpoint path provided but file not found: %s. "
                "Proceeding with random initialization for image encoder.",
                mae_checkpoint_path,
            )
        else:
            logger.info(
                "Pretrainer: No MAE checkpoint provided. "
                "Image encoder backbone will use random initialization."
            )

        # ------------------------------------------------------------------
        # Step 2: Build AdamW optimizer with layer-wise LR decay
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
        # Step 3: Build reciprocal sqrt LR scheduler
        # ------------------------------------------------------------------
        self.scheduler: LambdaLR = build_scheduler(
            optimizer=self.optimizer,
            total_steps=self.total_steps,
            warmup_iters=self.warmup_iters,
            cooldown_iters=self.cooldown_iters,
            timescale=self.timescale,
            last_epoch=-1,
        )

        # ------------------------------------------------------------------
        # Step 4: Instantiate loss function
        # Occlusion loss weight is 0 during pre-training (no occlusion in SA-1B)
        # ------------------------------------------------------------------
        self.losses: SAM2Losses = SAM2Losses(
            focal_weight=self.focal_weight,
            dice_weight=self.dice_weight,
            iou_weight=self.iou_weight,
            occlusion_weight=0.0,  # Not supervised during pre-training
        )

        # ------------------------------------------------------------------
        # Step 5: Instantiate prompt sampler and click sampler
        # ------------------------------------------------------------------
        self.prompt_sampler: PromptSampler = PromptSampler(
            gt_mask_prob=float(
                config.get("prompt_probabilities", {}).get("gt_mask", 0.50)
            ),
            click_prob=float(
                config.get("prompt_probabilities", {}).get("positive_click", 0.25)
            ),
            box_prob=float(
                config.get("prompt_probabilities", {}).get("bounding_box", 0.25)
            ),
            correction_click_random_prob=self.correction_click_random_prob,
            mask_threshold=self.mask_threshold,
        )
        self.click_sampler: ClickSampler = ClickSampler()
        self.mask_utils: MaskUtils = MaskUtils()

        # ------------------------------------------------------------------
        # Step 6: GradScaler for mixed precision (bfloat16 typically doesn't
        # need scaling, but included for compatibility)
        # ------------------------------------------------------------------
        # bfloat16 doesn't require gradient scaling (no underflow risk)
        # We use a no-op scaler for bfloat16 and a real scaler for float16
        self._use_grad_scaler: bool = (self.precision == "float16")
        if self._use_grad_scaler:
            self.scaler: torch.cuda.amp.GradScaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None  # type: ignore[assignment]

        # ------------------------------------------------------------------
        # Step 7: TensorBoard writer (main process only)
        # ------------------------------------------------------------------
        self.writer: Optional[SummaryWriter] = None
        if is_main_process():
            ensure_dir(log_dir)
            ensure_dir(checkpoint_dir)
            self.writer = setup_tensorboard(log_dir=log_dir, rank=0)

        # Training state
        self._current_step: int = 0
        self._best_loss: float = float("inf")

        logger.info(
            "Pretrainer initialized: total_steps=%d, base_lr=%.2e, "
            "encoder_type=%s, num_correction_clicks=%d, precision=%s",
            self.total_steps,
            self.base_lr,
            self.encoder_type,
            self.num_correction_clicks,
            self.precision,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pretrain(self, dataloader: DataLoader) -> None:
        """Run the full pre-training loop on SA-1B.

        Iterates for total_steps steps, processing one batch per step.
        Handles checkpoint resumption, metric logging, and periodic saves.

        Args:
            dataloader: DataLoader yielding SA-1B batches. Each batch is a
                dict with keys: "image" [B, C, H, W], "masks" [B, N, H, W],
                "valid_mask" [B, N] bool, "num_masks" [B] int.
                The dataloader should be configured with batch_size matching
                config.pretrain.batch_size (256) or a micro-batch size for
                gradient accumulation.
        """
        self.model.train()

        # Resume from latest checkpoint if available
        latest_ckpt: str = os.path.join(self.checkpoint_dir, "latest.pth")
        if os.path.isfile(latest_ckpt):
            logger.info(
                "Pretrainer: Resuming from checkpoint %s", latest_ckpt
            )
            start_step: int = self.load_checkpoint(latest_ckpt)
            self._current_step = start_step
            logger.info(
                "Pretrainer: Resumed from step %d", start_step
            )
        else:
            start_step = 0
            self._current_step = 0

        # Create infinite dataloader iterator
        dataloader_iter: Iterator = iter(dataloader)

        # Advance iterator to resume position (skip already-processed batches)
        # For large datasets, this is handled by the DataLoader's sampler state
        # which is restored from the checkpoint's RNG state.

        logger.info(
            "Pretrainer: Starting pre-training from step %d to %d",
            start_step,
            self.total_steps,
        )

        # Accumulation state
        accumulated_loss: float = 0.0
        self.optimizer.zero_grad()

        for step in range(start_step, self.total_steps):
            self._current_step = step

            # Fetch next batch, cycling through the dataloader
            try:
                batch: Dict[str, Any] = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)

            # Move batch to device
            batch = self._move_batch_to_device(batch)

            # Skip batches with no valid masks
            if not self._batch_has_valid_masks(batch):
                logger.debug(
                    "Pretrainer: Skipping step %d — no valid masks in batch.",
                    step,
                )
                continue

            # ------------------------------------------------------------------
            # Forward pass + loss computation
            # ------------------------------------------------------------------
            try:
                loss_dict: Dict[str, float] = self._train_step(batch)
            except Exception as exc:
                logger.warning(
                    "Pretrainer: _train_step failed at step %d: %s. Skipping.",
                    step,
                    exc,
                )
                self.optimizer.zero_grad()
                continue

            # ------------------------------------------------------------------
            # Gradient accumulation
            # ------------------------------------------------------------------
            step_in_accumulation: int = (step - start_step) % self.accumulation_steps
            accumulated_loss += loss_dict.get("total", 0.0)

            if step_in_accumulation == self.accumulation_steps - 1:
                # Gradient clipping
                grad_norm: float = clip_gradients(
                    model=self.model,
                    clip_type=self.grad_clip_type,
                    clip_max=self.grad_clip_max,
                )

                # Optimizer step
                if self._use_grad_scaler and self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

                # Log accumulated loss
                loss_dict["grad_norm"] = grad_norm
                loss_dict["total"] = accumulated_loss / self.accumulation_steps
                accumulated_loss = 0.0

                # ------------------------------------------------------------------
                # Logging
                # ------------------------------------------------------------------
                if step % self.log_interval == 0 and is_main_process():
                    current_lr: float = self._get_current_lr()
                    loss_dict["lr"] = current_lr

                    log_metrics(
                        writer=self.writer,
                        metrics=loss_dict,
                        step=step,
                        prefix="pretrain",
                    )

                    logger.info(
                        "Pretrain step %d/%d | %s",
                        step,
                        self.total_steps,
                        format_metrics(loss_dict),
                    )

                # ------------------------------------------------------------------
                # Checkpoint saving
                # ------------------------------------------------------------------
                if step % self.checkpoint_interval == 0 and is_main_process():
                    is_best: bool = loss_dict.get("total", float("inf")) < self._best_loss
                    if is_best:
                        self._best_loss = loss_dict.get("total", float("inf"))

                    self.save_checkpoint(
                        path=self.checkpoint_dir,
                        epoch=step,
                    )
                    logger.info(
                        "Pretrainer: Saved checkpoint at step %d", step
                    )

        # Final checkpoint
        if is_main_process():
            self.save_checkpoint(path=self.checkpoint_dir, epoch=self.total_steps)
            logger.info(
                "Pretrainer: Pre-training complete. Final checkpoint saved."
            )

        if self.writer is not None:
            self.writer.close()

    def _train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Execute a single training step on one batch.

        Runs the full interactive click simulation pipeline:
            1. Encode image with Hiera + FPN (once per image)
            2. Apply memory attention (pass-through with empty bank)
            3. For each mask in the batch:
               a. Simulate 1 initial + 7 correction clicks
               b. Run mask decoder for each click iteration
               c. Accumulate losses

        Args:
            batch: Dict from SA1BDataset with keys:
                - "image": Tensor[B, C, H, W] float32, ImageNet-normalized
                - "masks": Tensor[B, N, H, W] float32 binary {0, 1}
                - "valid_mask": Tensor[B, N] bool — True for real masks
                - "num_masks": int or Tensor[B] — number of valid masks

        Returns:
            Dict[str, float] with keys: "total", "focal", "dice", "iou".
            All values are Python floats (detached from computation graph).
            The backward pass is called internally on the total loss tensor.
        """
        images: Tensor = batch["image"]  # [B, C, H, W]
        masks_padded: Tensor = batch["masks"]  # [B, N, H, W]
        valid_mask: Tensor = batch["valid_mask"]  # [B, N] bool

        B: int = images.shape[0]
        device: torch.device = images.device

        # Accumulate losses across all masks and click iterations
        total_focal: float = 0.0
        total_dice: float = 0.0
        total_iou: float = 0.0
        total_combined: float = 0.0
        num_processed: int = 0

        # ------------------------------------------------------------------
        # Step 1: Run image encoder once per batch (shared across all masks)
        # ------------------------------------------------------------------
        autocast_ctx = get_autocast_context(self.precision)

        with autocast_ctx:
            frame_embed, skip_features = self.model.forward_image(images)
            # frame_embed: [B, C, H/16, W/16]
            # skip_features: [stride4_feat, stride8_feat]

            # ------------------------------------------------------------------
            # Step 2: Apply memory attention with empty memory bank
            # With empty bank, this is effectively a pass-through (self-attention only)
            # Paper: "When applied to images, the memory is empty and the model
            #         behaves like SAM."
            # ------------------------------------------------------------------
            empty_memory_bank: MemoryBank = MemoryBank(
                max_recent_frames=self.model.config.num_recent_memories,
                memory_dim=self.model.config.memory_feature_dim,
                max_prompted_frames=2,
                object_pointer_dim=self.model.config.object_pointer_dim,
                num_object_pointer_tokens=self.model.config.num_object_pointer_tokens,
            )
            empty_memory_bank = empty_memory_bank.to(device)

            # Get memory bank output (empty tensors)
            from models.memory_bank import MemoryBankOutput
            memory_bank_output: MemoryBankOutput = (
                empty_memory_bank.get_memory_for_attention()
            )

            # Condition frame features on (empty) memory
            conditioned_embed: Tensor = self.model.memory_attention.forward(
                curr_frame_embed=frame_embed,
                memory_bank_output=memory_bank_output,
            )
            # conditioned_embed: [B, C, H/16, W/16]

            # Get image positional encoding
            image_pe: Tensor = self.model.prompt_encoder.get_dense_pe()
            # image_pe: [1, C, H/16, W/16]

        # ------------------------------------------------------------------
        # Step 3: Process each valid mask in the batch
        # ------------------------------------------------------------------
        # Iterate over each batch element and each valid mask within it
        for b_idx in range(B):
            # Find valid mask indices for this batch element
            valid_indices: List[int] = [
                n for n in range(masks_padded.shape[1])
                if valid_mask[b_idx, n].item()
            ]

            if not valid_indices:
                continue

            # Process each valid mask independently
            for mask_idx in valid_indices:
                gt_mask_single: Tensor = masks_padded[b_idx, mask_idx]
                # gt_mask_single: [H, W] float32 binary

                # Skip empty masks (all zeros)
                if float(gt_mask_single.sum().item()) == 0.0:
                    continue

                # Extract single-image features for this batch element
                # frame_embed_single: [1, C, H/16, W/16]
                frame_embed_single: Tensor = frame_embed[b_idx:b_idx + 1]
                conditioned_embed_single: Tensor = conditioned_embed[b_idx:b_idx + 1]
                skip_features_single: List[Tensor] = [
                    feat[b_idx:b_idx + 1] for feat in skip_features
                ]

                # gt_mask for loss: [1, H, W]
                gt_mask_for_loss: Tensor = gt_mask_single.unsqueeze(0)

                # ------------------------------------------------------------------
                # Step 4: Simulate interactive clicks for this mask
                # Returns list of PromptInputs (1 initial + 7 corrections)
                # ------------------------------------------------------------------
                prompts_list: List[PromptInput] = self._simulate_interactive_clicks_for_mask(
                    gt_mask=gt_mask_single,
                    conditioned_embed=conditioned_embed_single,
                    skip_features=skip_features_single,
                    image_pe=image_pe,
                    device=device,
                )

                # ------------------------------------------------------------------
                # Step 5: Forward pass for each click iteration
                # ------------------------------------------------------------------
                num_iterations: int = len(prompts_list)

                for iter_idx, prompt in enumerate(prompts_list):
                    is_first_click: bool = (iter_idx == 0)
                    # Multi-mask output only on first click (ambiguous single click)
                    # For box and mask prompts, also use single-mask output
                    multimask_output: bool = (
                        is_first_click
                        and prompt.has_clicks()
                        and not prompt.has_box()
                        and not prompt.has_mask()
                        and prompt.num_clicks() == 1
                    )

                    with autocast_ctx:
                        # Encode prompts
                        points_tuple: Optional[Tuple[Tensor, Tensor]] = None
                        if prompt.points is not None and prompt.point_labels is not None:
                            pts: Tensor = prompt.points.unsqueeze(0).to(device)
                            lbls: Tensor = prompt.point_labels.unsqueeze(0).to(device)
                            points_tuple = (pts, lbls)

                        boxes_input: Optional[Tensor] = None
                        if prompt.boxes is not None:
                            boxes_input = prompt.boxes.unsqueeze(0).to(device)

                        masks_input: Optional[Tensor] = None
                        if prompt.masks is not None:
                            masks_input = prompt.masks.unsqueeze(0).to(device)

                        sparse_embeddings, dense_embeddings = (
                            self.model.prompt_encoder.forward(
                                points=points_tuple,
                                boxes=boxes_input,
                                masks=masks_input,
                            )
                        )

                        # Run mask decoder
                        masks_out, iou_scores, occlusion_score, object_pointer = (
                            self.model.mask_decoder.forward(
                                image_embeddings=conditioned_embed_single,
                                image_pe=image_pe,
                                sparse_prompt_embeddings=sparse_embeddings,
                                dense_prompt_embeddings=dense_embeddings,
                                skip_features=skip_features_single,
                                multimask_output=multimask_output,
                            )
                        )
                        # masks_out: [1, N_masks, H, W]
                        # iou_scores: [1, N_masks]
                        # occlusion_score: [1, 1]

                        # Resize masks to match GT mask resolution if needed
                        if masks_out.shape[-2:] != gt_mask_for_loss.shape[-2:]:
                            masks_out = torch.nn.functional.interpolate(
                                masks_out,
                                size=gt_mask_for_loss.shape[-2:],
                                mode="bilinear",
                                align_corners=False,
                            )

                        # GT occlusion: all zeros for SA-1B (always visible)
                        gt_occ: Tensor = torch.zeros(
                            1, dtype=torch.float32, device=device
                        )

                        # Compute loss
                        loss_result: Dict[str, Tensor] = self.losses.compute_total_loss(
                            pred_masks=masks_out,
                            gt_masks=gt_mask_for_loss,
                            pred_iou=iou_scores,
                            pred_occ=occlusion_score,
                            gt_occ=gt_occ,
                        )

                    # Scale loss by accumulation steps and number of iterations
                    # to maintain consistent gradient magnitude
                    scale_factor: float = 1.0 / (
                        self.accumulation_steps * num_iterations
                    )
                    scaled_loss: Tensor = loss_result["total"] * scale_factor

                    # Backward pass
                    if self._use_grad_scaler and self.scaler is not None:
                        self.scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                    # Accumulate