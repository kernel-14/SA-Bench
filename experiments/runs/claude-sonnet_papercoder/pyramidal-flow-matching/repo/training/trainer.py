## Code: training/trainer.py

```python
## training/trainer.py
"""Main training loop for Pyramidal Flow Matching.

Implements the three-stage training procedure described in the paper
(Appendix B, Table 4):
    Stage 1: Image training (50k steps, lr=1e-4, beta2=0.999)
    Stage 2: Low-resolution video training (200k steps, lr=1e-4, beta2=0.95)
             Sub-stage 2a: 80k steps on 2-second videos
             Sub-stage 2b: 120k steps on 5-second videos
    Stage 3: High-resolution video training (50k steps, lr=5e-5, beta2=0.95)

Key features:
    - Accelerate-based distributed training on 128 A100 GPUs
    - bfloat16 mixed precision throughout
    - 12.5% image mixing during video training stages (Appendix B)
    - Stage-specific optimizer hyperparameters (Table 4)
    - Constant LR with linear warmup for all stages
    - Gradient clipping at 1.0 (Table 4)
    - Patch n' Pack batch construction via TokenPacker
    - Checkpoint save/load via utils/checkpointing.py

Usage:
    from training.trainer import Trainer

    trainer = Trainer(model=pyramid_flow_model, config=config)
    trainer.train_stage(stage=1)
    trainer.train_stage(stage=2)
    trainer.train_stage(stage=3)
"""

import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from data.dataset_loader import DatasetLoader
from models.pyramid_flow import PyramidFlowModel
from training.losses import FlowMatchingLoss
from training.schedules import (
    build_optimizer,
    build_optimizer_and_scheduler,
    get_constant_schedule_with_warmup,
)
from utils.checkpointing import load_checkpoint, save_checkpoint
from utils.distributed import barrier, is_main_process
from utils.logging import build_summary_writer, get_logger, log_metrics

## ---------------------------------------------------------------------------
## Module-level logger
## ---------------------------------------------------------------------------
logger = get_logger(__name__)

## ---------------------------------------------------------------------------
## Optional Accelerate import
## ---------------------------------------------------------------------------
_ACCELERATE_AVAILABLE: bool = False
try:
    from accelerate import Accelerator  # type: ignore[import]
    _ACCELERATE_AVAILABLE = True
except ImportError:
    logger.warning(
        "accelerate not available. Distributed training will be disabled. "
        "Install with: pip install accelerate==0.29.0"
    )


class Trainer:
    """Orchestrates the three-stage training procedure for Pyramidal Flow Matching.

    Manages the full training lifecycle: optimizer construction, learning rate
    scheduling, forward/backward passes, gradient clipping, checkpointing,
    and distributed training via Accelerate.

    Implements the paper's training protocol including:
    - Stage-specific hyperparameters from Table 4 (Appendix B)
    - 12.5% image mixing ratio during video training stages
    - Stage 2 sub-stage transition at 80k steps (2s → 5s videos)
    - Constant LR with linear warmup for all three stages
    - Gradient clipping at 1.0 for all stages

    Attributes:
        model: PyramidFlowModel wrapping VAE, MM-DiT, and text encoders.
        config: Full project configuration from configs/default.yaml.
        loss_fn: FlowMatchingLoss for computing flow matching and VAE losses.
        accelerator: Accelerate instance for distributed training and mixed precision.
        optimizer: AdamW optimizer (rebuilt at start of each stage).
        scheduler: LambdaLR constant-with-warmup scheduler (rebuilt per stage).
        global_step: Global training step counter across all stages.
        current_stage: Current training stage (1, 2, or 3).
        dataset_loader: DatasetLoader for building image and video dataloaders.
        writer: TensorBoard SummaryWriter (rank 0 only, or None).
        _stage_config_cache: Cached stage sub-config dict for the current stage.
    """

    def __init__(
        self,
        model: PyramidFlowModel,
        config: Dict[str, Any],
    ) -> None:
        """Initializes the Trainer.

        Sets up Accelerate for distributed training, initializes the loss
        function, and prepares the model. Optimizer and scheduler are built
        lazily at the start of each training stage to use stage-specific
        hyperparameters.

        Args:
            model: The PyramidFlowModel to train. Contains VAE, MM-DiT,
                and frozen text encoders.
            config: Project configuration dictionary from configs/default.yaml.
                All training hyperparameters are read from this config.
        """
        self.model: PyramidFlowModel = model
        self.config: Dict[str, Any] = config

        # ----------------------------------------------------------------
        # Parse top-level config sections
        # ----------------------------------------------------------------
        training_cfg: Dict[str, Any] = config.get("training", {})
        paths_cfg: Dict[str, Any] = config.get("paths", {})
        logging_cfg: Dict[str, Any] = config.get("logging", {})

        self._training_cfg: Dict[str, Any] = training_cfg
        self._paths_cfg: Dict[str, Any] = paths_cfg
        self._logging_cfg: Dict[str, Any] = logging_cfg

        # ----------------------------------------------------------------
        # Initialize Accelerate for distributed training + mixed precision
        # ----------------------------------------------------------------
        mixed_precision: str = str(training_cfg.get("mixed_precision", "bf16"))
        gradient_accumulation_steps: int = int(
            training_cfg.get("gradient_accumulation_steps", 1)
        )

        if _ACCELERATE_AVAILABLE:
            self.accelerator: Any = Accelerator(
                mixed_precision=mixed_precision,
                gradient_accumulation_steps=gradient_accumulation_steps,
                log_with=None,  # We handle logging manually
            )
            logger.info(
                "Accelerate initialized: mixed_precision=%s, "
                "gradient_accumulation_steps=%d, "
                "num_processes=%d",
                mixed_precision,
                gradient_accumulation_steps,
                self.accelerator.num_processes,
            )
        else:
            # Fallback: no distributed training
            self.accelerator = None
            logger.warning(
                "Accelerate not available. Running in single-process mode. "
                "Distributed training on 128 GPUs requires accelerate."
            )

        # ----------------------------------------------------------------
        # Prepare model with Accelerate (handles DDP wrapping)
        # ----------------------------------------------------------------
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
        else:
            # Move to CUDA if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()

        # ----------------------------------------------------------------
        # Initialize loss function
        # ----------------------------------------------------------------
        self.loss_fn: FlowMatchingLoss = FlowMatchingLoss(config)

        # ----------------------------------------------------------------
        # Initialize DatasetLoader
        # ----------------------------------------------------------------
        self.dataset_loader: DatasetLoader = DatasetLoader(config)

        # ----------------------------------------------------------------
        # Training state
        # ----------------------------------------------------------------
        self.global_step: int = 0
        self.current_stage: int = 1

        # Optimizer and scheduler are built lazily in train_stage
        self.optimizer: Optional[AdamW] = None
        self.scheduler: Optional[LambdaLR] = None

        # Cached stage sub-config for the currently active stage
        self._stage_config_cache: Dict[str, Any] = {}

        # ----------------------------------------------------------------
        # TensorBoard SummaryWriter (rank 0 only)
        # ----------------------------------------------------------------
        log_dir: str = str(paths_cfg.get("log_dir", "logs"))
        use_tensorboard: bool = bool(logging_cfg.get("use_tensorboard", True))
        self.writer: Optional[Any] = None
        if use_tensorboard:
            self.writer = build_summary_writer(log_dir=log_dir)

        # ----------------------------------------------------------------
        # Logging frequency and checkpoint frequency
        # ----------------------------------------------------------------
        self._log_every_steps: int = int(logging_cfg.get("log_every_steps", 100))

        logger.info(
            "Trainer initialized: global_step=%d, current_stage=%d, "
            "log_every_steps=%d",
            self.global_step,
            self.current_stage,
            self._log_every_steps,
        )

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _get_stage_config(self, stage: int) -> Dict[str, Any]:
        """Returns the stage-specific sub-config dict.

        Args:
            stage: Training stage integer (1, 2, or 3).

        Returns:
            Stage sub-config dict from config.training.stage{stage}.

        Raises:
            ValueError: If stage is not 1, 2, or 3.
        """
        if stage not in (1, 2, 3):
            raise ValueError(
                f"stage must be 1, 2, or 3, got stage={stage}. "
                f"Only three training stages are defined in the paper."
            )

        stage_key: str = f"stage{stage}"
        stage_cfg: Dict[str, Any] = dict(
            self._training_cfg.get(stage_key, {})
        )

        # Apply defaults from Table 4 (Appendix B) if keys are missing
        defaults: Dict[str, Any] = {
            "optimizer": "adamw",
            "beta1": 0.9,
            "beta2": 0.999 if stage == 1 else 0.95,
            "eps": 1.0e-6,
            "weight_decay": 1.0e-4,
            "learning_rate": 1.0e-4 if stage in (1, 2) else 5.0e-5,
            "lr_schedule": "constant_with_warmup",
            "warmup_steps": 1000,
            "total_steps": 50000 if stage in (1, 3) else 200000,
            "global_batch_size": 1536 if stage == 1 else (768 if stage == 2 else 384),
            "gradient_clip": 1.0,
            "dtype": "bfloat16",
            "checkpoint_every_steps": 5000 if stage in (1, 3) else 10000,
            "log_every_steps": 100,
        }

        for key, default_val in defaults.items():
            if key not in stage_cfg:
                stage_cfg[key] = default_val

        return stage_cfg

    def _get_device(self) -> torch.device:
        """Returns the current training device.

        Returns:
            torch.device for the current process.
        """
        if self.accelerator is not None:
            return self.accelerator.device
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _build_stage_optimizer_and_scheduler(
        self,
        stage: int,
    ) -> Tuple[AdamW, LambdaLR]:
        """Builds optimizer and scheduler for a specific training stage.

        Rebuilds from scratch at the start of each stage to apply
        stage-specific hyperparameters (betas, learning rate).

        Args:
            stage: Training stage integer (1, 2, or 3).

        Returns:
            Tuple (optimizer, scheduler) configured for the given stage.
        """
        stage_cfg: Dict[str, Any] = self._get_stage_config(stage)

        # Unwrap model for parameter access (handles DDP wrapping)
        if self.accelerator is not None:
            unwrapped_model: nn.Module = self.accelerator.unwrap_model(self.model)
        else:
            unwrapped_model = self.model

        optimizer, scheduler = build_optimizer_and_scheduler(
            model=unwrapped_model,
            stage_config=stage_cfg,
            schedule_type="constant_with_warmup",
        )

        logger.info(
            "Stage %d optimizer built: lr=%.2e, beta1=%.3f, beta2=%.3f, "
            "eps=%.2e, weight_decay=%.2e, warmup_steps=%d",
            stage,
            float(stage_cfg["learning_rate"]),
            float(stage_cfg["beta1"]),
            float(stage_cfg["beta2"]),
            float(stage_cfg["eps"]),
            float(stage_cfg["weight_decay"]),
            int(stage_cfg["warmup_steps"]),
        )

        return optimizer, scheduler

    def _prepare_optimizer_scheduler(
        self,
        optimizer: AdamW,
        scheduler: LambdaLR,
    ) -> Tuple[AdamW, LambdaLR]:
        """Prepares optimizer and scheduler with Accelerate.

        Args:
            optimizer: AdamW optimizer to prepare.
            scheduler: LambdaLR scheduler to prepare.

        Returns:
            Tuple (prepared_optimizer, prepared_scheduler).
        """
        if self.accelerator is not None:
            optimizer, scheduler = self.accelerator.prepare(optimizer, scheduler)
        return optimizer, scheduler

    def _prepare_dataloader(self, dataloader: DataLoader) -> DataLoader:
        """Prepares a dataloader with Accelerate for distributed training.

        Args:
            dataloader: DataLoader to prepare.

        Returns:
            Prepared DataLoader with distributed sampler and device placement.
        """
        if self.accelerator is not None:
            return self.accelerator.prepare(dataloader)
        return dataloader

    def _infinite_dataloader(
        self,
        dataloader: DataLoader,
    ) -> Iterator[Dict[str, Any]]:
        """Creates an infinite iterator over a dataloader.

        Restarts from the beginning when the dataloader is exhausted.
        This is necessary because training steps may exceed one epoch.

        Args:
            dataloader: The DataLoader to iterate over infinitely.

        Yields:
            Batch dicts from the dataloader, cycling indefinitely.
        """
        while True:
            for batch in dataloader:
                yield batch

    def _get_current_lr(self) -> float:
        """Returns the current learning rate from the scheduler.

        Returns:
            Current learning rate as a float. Returns 0.0 if no scheduler.
        """
        if self.scheduler is None:
            return 0.0
        try:
            last_lr: List[float] = self.scheduler.get_last_lr()
            return float(last_lr[0]) if last_lr else 0.0
        except Exception:
            return 0.0

    def _log_step_metrics(
        self,
        step_output: Dict[str, Any],
        stage: int,
        stage_step: int,
    ) -> None:
        """Logs training metrics for a single step.

        Args:
            step_output: Dict from train_step with 'loss', 'step', 'lr'.
            stage: Current training stage (1, 2, or 3).
            stage_step: Step count within the current stage.
        """
        metrics: Dict[str, Any] = {
            "train": {
                "loss": step_output.get("loss", 0.0),
                "lr": step_output.get("lr", 0.0),
                "stage": float(stage),
                "stage_step": float(stage_step),
            }
        }

        # Include pyramid stage if available
        if "pyramid_stage_id" in step_output:
            metrics["train"]["pyramid_stage_id"] = float(
                step_output["pyramid_stage_id"]
            )

        log_metrics(
            metrics=metrics,
            step=self.global_step,
            writer=self.writer,
        )

    # -----------------------------------------------------------------------
    # Core training methods
    # -----------------------------------------------------------------------

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Executes a single forward + backward training step.

        Implements the hot path: model forward pass, loss computation,
        backward pass, gradient clipping, and optimizer/scheduler step.

        Args:
            batch: Training batch dict from TokenPacker.collate_packed_batch.
                Expected keys:
                    - 'latents': Tensor or raw video/image data
                    - 'text_cond': dict with T5/CLIP embeddings
                    - 'frame_token_counts': list[int]
                    - 'attention_mask': Tensor
                    - 'stage_ids': Tensor (sampled pyramid stage per sample)
                    - 'captions': list[str] (if latents not pre-encoded)

        Returns:
            Dict with keys:
                - 'loss': float — scalar loss value for this step.
                - 'step': int — current global step.
                - 'lr': float — current learning rate.
                - 'pyramid_stage_id': int — pyramid stage sampled this step
                  (if available from model output).
        """
        # Retrieve gradient clip value from cached stage config
        grad_clip: float = float(
            self._stage_config_cache.get("gradient_clip", 1.0)
        )

        # ----------------------------------------------------------------
        # Forward pass
        # ----------------------------------------------------------------
        # Use Accelerate's context manager for gradient accumulation
        if self.accelerator is not None:
            context = self.accelerator.accumulate(self.model)
        else:
            import contextlib
            context = contextlib.nullcontext()

        with context:
            # Model forward pass: encodes text + video, samples pyramid stage,
            # computes coupled endpoints, runs transformer, returns velocities
            try:
                outputs: Dict[str, Any] = self.model.forward(batch)
            except Exception as exc:
                logger.error(
                    "Model forward pass failed at step %d: %s. "
                    "Skipping this batch.",
                    self.global_step,
                    exc,
                )
                return {
                    "loss": float("nan"),
                    "step": self.global_step,
                    "lr": self._get_current_lr(),
                }

            pred_velocity: Tensor = outputs["pred_velocity"]
            target_velocity: Tensor = outputs["target_velocity"]
            mask: Optional[Tensor] = outputs.get("mask", None)

            # ----------------------------------------------------------------
            # Loss computation
            # ----------------------------------------------------------------
            loss: Tensor = self.loss_fn.compute(
                pred_velocity=pred_velocity,
                target_velocity=target_velocity,
                mask=mask,
            )

            # ----------------------------------------------------------------
            # Backward pass
            # ----------------------------------------------------------------
            if self.accelerator is not None:
                self.accelerator.backward(loss)
            else:
                loss.backward()

            # ----------------------------------------------------------------
            # Gradient clipping (only when gradients are synchronized)
            # ----------------------------------------------------------------
            should_clip: bool = True
            if self.accelerator is not None:
                should_clip = self.accelerator.sync_gradients

            if should_clip:
                if self.accelerator is not None:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=grad_clip,
                    )
                else:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=grad_clip,
                    )

            # ----------------------------------------------------------------
            # Optimizer and scheduler step
            # ----------------------------------------------------------------
            if self.optimizer is not None:
                self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            if self.optimizer is not None:
                self.optimizer.zero_grad(set_to_none=True)

        # ----------------------------------------------------------------
        # Update global step counter
        # ----------------------------------------------------------------
        self.global_step += 1

        # ----------------------------------------------------------------
        # Build return dict
        # ----------------------------------------------------------------
        result: Dict[str, Any] = {
            "loss": loss.item() if not torch.isnan(loss) else float("nan"),
            "step": self.global_step,
            "lr": self._get_current_lr(),
        }

        # Include pyramid stage if model returned it
        if "pyramid_stage_id" in outputs:
            result["pyramid_stage_id"] = int(outputs["pyramid_stage_id"])

        return result

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Runs a validation loop to compute average loss.

        Sets the model to eval mode, runs forward passes without gradients,
        then restores training mode. Reduces the average loss across all
        distributed ranks.

        Args:
            dataloader: Validation DataLoader. Should be a finite iterator
                (not infinite) so the loop terminates naturally.

        Returns:
            Dict with key 'val_loss' containing the mean validation loss
            averaged across all batches and all distributed ranks.
        """
        logger.info("Running validation at step %d...", self.global_step)

        self.model.eval()
        total_loss: float = 0.0
        num_batches: int = 0

        with torch.no_grad():
            for batch in dataloader:
                try:
                    outputs: Dict[str, Any] = self.model.forward(batch)
                    loss: Tensor = self.loss_fn.compute(
                        pred_velocity=outputs["pred_velocity"],
                        target_velocity=outputs["target_velocity"],
                        mask=outputs.get("mask", None),
                    )
                    total_loss += loss.item()
                    num_batches += 1
                except Exception as exc:
                    logger.warning(
                        "Validation batch failed: %s. Skipping.", exc
                    )
                    continue

        self.model.train()

        # Compute average loss
        avg_loss: float = total_loss / max(num_batches, 1)

        # Reduce across distributed ranks
        device: torch.device = self._get_device()
        avg_loss_tensor: Tensor = torch.tensor(
            avg_loss, dtype=torch.float32, device=device
        )

        if self.accelerator is not None:
            avg_loss_tensor = self.accelerator.reduce(
                avg_loss_tensor, reduction="mean"
            )

        val_loss: float = avg_loss_tensor.item()

        logger.info(
            "Validation complete: val_loss=%.6f (num_batches=%d)",
            val_loss,
            num_batches,
        )

        return {"val_loss": val_loss}

    def _mix_image_video_batch(
        self,
        video_batch: Dict[str, Any],
        image_batch: Dict[str, Any],
        image_ratio: float = 0.125,
    ) -> Dict[str, Any]:
        """Mixes image samples into a video batch at the specified ratio.

        Implements the 12.5% image mixing described in the paper (Appendix B):
        "image data from stage 1 is also utilized at a proportion of 12.5%
        in each batch."

        Images are treated as single-frame videos (T=1), consistent with
        the paper's statement that "the first frame in a video acts as an
        image" and the model supports joint image/video training.

        The mixing is performed at the sequence level: image tokens are
        appended to the video token sequence, with the attention mask and
        frame_token_counts updated accordingly. Cross-sample attention is
        blocked by the causal mask (tokens from different samples cannot
        attend to each other).

        Args:
            video_batch: Batch dict from video DataLoader. Contains packed
                video sequences with keys: 'latents', 'text_cond',
                'frame_token_counts', 'attention_mask', 'stage_ids', etc.
            image_batch: Batch dict from image DataLoader. Same structure
                as video_batch but with single-frame sequences.
            image_ratio: Fraction of the batch to replace with image samples.
                Defaults to 0.125 (12.5%) as specified in the paper.

        Returns:
            Mixed batch dict with the same structure as the input batches,
            containing both video and image samples. The 'captions' list
            is extended with image captions. The 'frame_token_counts' list
            is extended with image frame token counts.
        """
        # ----------------------------------------------------------------
        # Determine how many image samples to include
        # ----------------------------------------------------------------
        # Count video samples from the batch
        video_captions: List[str] = video_batch.get("captions", [])
        num_video_samples: int = len(video_captions)

        if num_video_samples == 0:
            # Empty video batch: return image batch as-is
            return image_batch

        # Number of image samples to mix in
        num_image_samples: int = max(1, round(num_video_samples * image_ratio))

        # Get image captions (may be fewer than requested)
        image_captions: List[str] = image_batch.get("captions", [])
        actual_image_samples: int = min(num_image_samples, len(image_captions))

        if actual_image_samples == 0:
            # No image samples available: return video batch unchanged
            logger.debug(
                "No image samples available for mixing. "
                "Returning video batch unchanged."
            )
            return video_batch

        # ----------------------------------------------------------------
        # Merge batch dicts
        # ----------------------------------------------------------------
        # Strategy: concatenate along the sequence dimension for packed tensors.
        # For list fields (captions, frame_token_counts, stage_ids), extend.
        # For tensor fields (latents, attention_mask), concatenate along seq dim.

        mixed_batch: Dict[str, Any] = {}

        # ----------------------------------------------------------------
        # Merge 'captions' list
        # ----------------------------------------------------------------
        mixed_batch["captions"] = (
            video_captions + image_captions[:actual_image_samples]
        )

        # ----------------------------------------------------------------
        # Merge 'stage_ids'
        # ----------------------------------------------------------------
        video_stage_ids: Any = video_batch.get("stage_ids", [])
        image_stage_ids: Any = image_batch.get("stage_ids", [])

        if isinstance(video_stage_ids, Tensor) and isinstance(image_stage_ids, Tensor):
            mixed_batch["stage_ids"] = torch.cat(
                [video_stage_ids, image_stage_ids[:actual_image_samples]], dim=0
            )
        elif isinstance(video_stage_ids, list) and isinstance(image_stage_ids, list):
            mixed_batch["stage_ids"] = (
                video_stage_ids + image_stage_ids[:actual_image_samples]
            )
        else:
            mixed_batch["stage_ids"] = video_stage_ids

        # ----------------------------------------------------------------
        # Merge 'frame_token_counts'
        # ----------------------------------------------------------------
        video_ftc: List[Any] = video_batch.get("frame_token_counts", [])
        image_ftc: List[Any] = image_batch.get("frame_token_counts", [])

        if isinstance(video_ftc, list) and isinstance(image_ftc, list):
            mixed_batch["frame_token_counts"] = (
                video_ftc + image_ftc[:actual_image_samples]
            )
        else:
            mixed_batch["frame_token_counts"] = video_ftc

        # ----------------------------------------------------------------
        # Merge 'sample_token_counts' (for cross-sample attention blocking)
        # ----------------------------------------------------------------
        video_stc: List[Any] = video_batch.get("sample_token_counts", [])
        image_stc: List[Any] = image_batch.get("sample_token_counts", [])

        if isinstance(video_stc, list) and isinstance(image_stc, list):
            mixed_batch["sample_token_counts"] = (
                video_stc + image_stc[:actual_image_samples]
            )
        else:
            mixed_batch["sample_token_counts"] = video_stc

        # ----------------------------------------------------------------
        # Merge tensor fields: 'latents' and 'attention_mask'
        # ----------------------------------------------------------------
        for tensor_key in ("latents", "attention_mask", "padding_mask"):
            video_tensor: Optional[Tensor] = video_batch.get(tensor_key)
            image_tensor: Optional[Tensor] = image_batch.get(tensor_key)

            if video_tensor is None:
                mixed_batch[tensor_key] = image_tensor
                continue
            if image_tensor is None:
                mixed_batch[tensor_key] = video_tensor
                continue

            # Both tensors exist: concatenate along batch dimension (dim 0)
            # or sequence dimension depending on shape
            try:
                # Determine concatenation dimension
                # latents: [num_bins, max_seq_len, latent_channels] → dim 0
                # attention_mask: [num_bins, max_seq_len] → dim 0
                # Both have batch as dim 0
                image_slice: Tensor = image_tensor[:actual_image_samples]

                # Handle shape mismatch in sequence length dimension
                if video_tensor.dim() >= 2 and image_slice.dim() >= 2:
                    video_seq_len: int = video_tensor.shape[1]
                    image_seq_len: int = image_slice.shape[1]

                    if video_seq_len != image_seq_len:
                        # Pad the shorter one to match the longer
                        max_seq_len: int = max(video_seq_len, image_seq_len)

                        if video_seq_len < max_seq_len:
                            pad_size: int = max_seq_len - video_seq_len
                            # Pad along dim 1 (sequence dimension)
                            pad_shape: List[int] = list(video_tensor.shape)
                            pad_shape[1] = pad_size
                            video_pad: Tensor = torch.zeros(
                                pad_shape,
                                dtype=video_tensor.dtype,
                                device=video_tensor.device,
                            )
                            video_tensor = torch.cat(
                                [video_tensor, video_pad], dim=1
                            )

                        if image_seq_len < max_seq_len:
                            pad_size = max_seq_len - image_seq_len
                            pad_shape = list(image_slice.shape)
                            pad_shape[1] = pad_size
                            image_pad: Tensor = torch.zeros(
                                pad_shape,
                                dtype=image_slice.dtype,
                                device=image_slice.device,
                            )
                            image_slice = torch.cat(
                                [image_slice, image_pad], dim=1
                            )

                mixed_batch[tensor_key] = torch.cat(
                    [video_tensor, image_slice], dim=0
                )

            except Exception as exc:
                logger.warning(
                    "Failed to merge tensor '%s' during image/video mixing: %s. "
                    "Using video batch tensor only.",
                    tensor_key,
                    exc,
                )
                mixed_batch[tensor_key] = video_tensor

        # ----------------------------------------------------------------
        # Merge 'text_cond' dict
        # ----------------------------------------------------------------
        video_text_cond: Optional[Dict[str, Any]] = video_batch.get("text_cond")
        image_text_cond: Optional[Dict[str, Any]] = image_batch.get("text_cond")

        if video_text_cond is not None and image_text_cond is not None:
            merged_text_cond: Dict[str, Any] = {}
            for cond_key in video_text_cond:
                v_val: Any = video_text_cond.get(cond_key)
                i_val: Any = image_text_cond.get(cond_key)

                if isinstance(v_val, Tensor) and isinstance(i_val, Tensor):
                    try:
                        i_slice: Tensor = i_val[:actual_image_samples]
                        # Handle sequence length mismatch for T5 embeddings
                        if v_val.dim() >= 2 and i_slice.dim() >=