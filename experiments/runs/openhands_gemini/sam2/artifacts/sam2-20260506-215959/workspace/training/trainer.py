
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, List
import os
from tqdm import tqdm
import math

from model.sam2 import SAM2
from training.losses import SAM2Loss
from data.datasets import Sam2Dataset
from config import Config
from utils.misc import set_seed, get_logger, save_checkpoint, get_iou

class Trainer:
    def __init__(self, config: Config):
        self.config = config
        set_seed(self.config.SEED)
        self.logger = get_logger("SAM2_Trainer")

        self.model = SAM2(
            image_encoder_type=self.config.IMAGE_ENCODER_TYPE,
            image_encoder_out_chans=256, # Assuming this default for Hiera, adjust if needed
            prompt_encoder_embed_dim=self.config.PROMPT_ENCODER_EMBED_DIM,
            mask_decoder_num_heads=8, # Common for transformers, adjust if paper specifies
            mask_decoder_num_layers=2, # Common for SAM-like mask decoders
            mask_decoder_iou_head_depth=3, # Common for SAM-like, adjust if paper specifies
            mask_decoder_iou_head_hidden_dim=256, # Common for SAM-like, adjust if paper specifies
            memory_attention_num_heads=8, # Common for transformers
            memory_attention_num_layers=self.config.MEMORY_ATTENTION_LAYERS,
            memory_channels=self.config.MEMORY_CHANNELS,
            num_mask_tokens=4, # Config.MULTIPLE_MASKS_OUTPUT
            num_point_embeddings=self.config.CORRECTION_CLICKS + 2, # +2 for positive/negative base embeddings
            image_size=self.config.IMAGE_SIZE,
        ).to(self.config.DEVICE)
        
        # Set memory bank configs after model initialization
        self.model.set_memory_bank_configs(
            num_recent_frames=self.config.NUM_RECENT_FRAMES_MEMORY_BANK,
            num_prompted_frames=self.config.NUM_PROMPTED_FRAMES_MEMORY_BANK,
        )

        self.criterion = SAM2Loss(config).to(self.config.DEVICE)
        self.optimizer = self._configure_optimizer(self.model)
        self.scheduler = self._configure_scheduler(self.optimizer)

        self.train_dataset = Sam2Dataset(config, is_train=True)
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.BATCH_SIZE_FULL_TRAIN, # Or PRETRAIN_BATCH_SIZE
            shuffle=True,
            num_workers=os.cpu_count() // 2, # Use half cores for data loading
            collate_fn=self._collate_fn,
        )
        
        self.val_dataset = Sam2Dataset(config, is_train=False) # Assuming a validation set exists
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.config.BATCH_SIZE_FULL_TRAIN,
            shuffle=False,
            num_workers=os.cpu_count() // 2,
            collate_fn=self._collate_fn,
        )

        self.current_step = 0
        self.total_steps = self.config.PRETRAIN_STEPS + self.config.FULL_TRAINING_STEPS + self.config.FINE_TUNE_STEPS
        
        self.amp_scaler = torch.cuda.amp.GradScaler() if self.config.DEVICE == "cuda" else None


    def _configure_optimizer(self, model: nn.Module):
        # Implement AdamW with layer-wise decay (Section D.2.1)
        # This is a simplified version; proper layer decay requires specific layer groups.
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
            betas=(self.config.OPTIMIZER_MOMENTUM_BETA1, self.config.OPTIMIZER_MOMENTUM_BETA2),
        )
        return optimizer

    def _configure_scheduler(self, optimizer):
        # Implement reciprocal square-root schedule (Section D.2.1)
        # Simplified: a constant LR after warmup for now, or a simple cosine decay.
        # Paper references Zhai et al., 2022 for reciprocal square-root.
        
        # Placeholder scheduler
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, (step + 1) / self.config.WARMUP_ITERS)
        )
        return scheduler

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # This collate_fn needs to handle both image and video data,
        # and potentially pad sequences/batch items if they are not uniform.
        # For simplicity, let's assume batching of uniform items for now.
        # A more robust collate would pad images/masks to max H/W and handle varying prompt counts.

        # Separate image and video items for clarity if needed, or pad all to max length.
        
        # For now, let's try to stack directly, assuming all items in a batch are compatible.
        
        elem = batch[0]
        output = {}
        for key in elem:
            if key in ["images", "gt_masks"]:
                output[key] = torch.stack([d[key] for d in batch]) # (B, T, C, H, W) or (B, 1, C, H, W)
            elif key in ["points", "labels", "boxes", "input_masks"]:
                # These can be None or vary in size, so pad or handle carefully.
                # For simplicity, only stack if all are present and same shape.
                if all(d[key] is not None for d in batch):
                    # Need padding for points/labels/boxes if N_points/N_boxes vary
                    if key == "points": # (B, N, 2)
                        output[key] = torch.stack([d[key] for d in batch])
                    elif key == "labels": # (B, N)
                        output[key] = torch.stack([d[key] for d in batch])
                    elif key == "boxes": # (B, 4) or (B, N, 4)
                        output[key] = torch.stack([d[key] for d in batch])
                    elif key == "input_masks": # (B, 1, H, W)
                        output[key] = torch.stack([d[key] for d in batch])
                else:
                    output[key] = None # Or handle padding explicitly
            elif key in ["is_video"]:
                output[key] = [d[key] for d in batch] # Keep as list of bools
            else:
                output[key] = [d[key] for d in batch] # For metadata like video_id, original_size

        # After stacking, move to device
        for key in output:
            if isinstance(output[key], torch.Tensor):
                output[key] = output[key].to(self.config.DEVICE)

        return output


    def _train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()

        is_video = batch["is_video"][0] # Assuming batch is uniform in this aspect
        images = batch["images"]
        gt_masks = batch["gt_masks"]
        
        points = batch["points"]
        labels = batch["labels"]
        boxes = batch["boxes"]
        input_masks = batch["input_masks"] # Mask prompt for the first frame if any
        
        
        if is_video:
            # For video, iterate through frames (or a sampled subset)
            # Paper mentions: "sample sequences of 8 frames and randomly select up to 2 frames to prompt"
            # This is interactive and happens dynamically. For now, we simulate initial prompt on first frame
            # and then propagation.
            
            # The full interactive training loop would be here, with model making predictions
            # and then sampling corrective clicks. This is a simplified non-interactive pass.
            
            num_frames = images.shape[1] # (B, T, C, H, W)
            all_pred_masks = []
            all_pred_ious = []
            all_pred_objectness = []
            all_gt_ious = []
            all_gt_objectness = []

            for t in range(num_frames):
                # Reset memory bank for each video in the batch or handle per-video memory
                # This part is tricky for batch processing with separate memories.
                # For now, let's assume a single video per batch or memory is reset per batch.
                self.model.reset_memory_bank() # Reset memory for each video
                
                # Pass initial prompts only for the first frame or prompted frames
                if t == 0:
                    current_points = points
                    current_labels = labels
                    current_boxes = boxes
                    current_input_masks = input_masks
                    is_prompted = True
                else:
                    current_points = None
                    current_labels = None
                    current_boxes = None
                    current_input_masks = None
                    is_prompted = False # For simplicity, only first frame is initially prompted

                # Forward pass
                with torch.autocast(device_type=self.config.DEVICE, dtype=torch.bfloat16 if self.config.PRECISION == "bfloat16" else torch.float16, enabled=self.amp_scaler is not None):
                    pred_masks, pred_ious, pred_objectness = self.model(
                        video_frames=images,
                        current_frame_idx=t,
                        points=current_points,
                        labels=current_labels,
                        boxes=current_boxes,
                        masks=current_input_masks, # If mask prompt is used for first frame
                        is_prompted_frame=is_prompted,
                    )

                all_pred_masks.append(pred_masks)
                all_pred_ious.append(pred_ious)
                all_pred_objectness.append(pred_objectness)
                
                # Ground truth for current frame
                current_gt_mask = gt_masks[:, t, :, :, :] # (B, 1, H, W)
                current_gt_objectness = (current_gt_mask.sum(dim=(-1, -2)) > 0).float() # (B,)
                current_gt_iou = get_iou(pred_masks[:,0:1,:,:], current_gt_mask) # IoU for best predicted mask

                all_gt_ious.append(current_gt_iou)
                all_gt_objectness.append(current_gt_objectness)
            
            # Concatenate predictions and ground truths across time
            final_pred_masks = torch.cat(all_pred_masks, dim=0) # (B*T, N_masks, H, W)
            final_pred_ious = torch.cat(all_pred_ious, dim=0) # (B*T, N_masks)
            final_pred_objectness = torch.cat(all_pred_objectness, dim=0) # (B*T, 1)
            
            final_gt_masks = gt_masks.view(-1, 1, gt_masks.shape[-2], gt_masks.shape[-1]) # (B*T, 1, H, W)
            final_gt_ious = torch.cat(all_gt_ious, dim=0) # (B*T,)
            final_gt_objectness = torch.cat(all_gt_objectness, dim=0) # (B*T,)

        else: # Image
            # For image, images is (B, 1, C, H, W), gt_masks is (B, 1, 1, H, W)
            current_frame_idx = 0
            with torch.autocast(device_type=self.config.DEVICE, dtype=torch.bfloat16 if self.config.PRECISION == "bfloat16" else torch.float16, enabled=self.amp_scaler is not None):
                pred_masks, pred_ious, pred_objectness = self.model(
                    video_frames=images,
                    current_frame_idx=current_frame_idx,
                    points=points,
                    labels=labels,
                    boxes=boxes,
                    masks=input_masks,
                    is_prompted_frame=True,
                )
            final_pred_masks = pred_masks
            final_pred_ious = pred_ious
            final_pred_objectness = pred_objectness

            # For image, gt_masks is (B, 1, 1, H, W). Squeezing gives (B, 1, H, W).
            final_gt_masks = gt_masks.squeeze(1) 
            final_gt_objectness = (final_gt_masks.sum(dim=(-1, -2)) > 0).float() # (B,)
            final_gt_ious = get_iou(final_pred_masks[:,0:1,:,:], final_gt_masks) # IoU for best predicted mask

        total_loss, losses = self.criterion(
            final_pred_masks, final_gt_masks,
            final_pred_ious, final_gt_ious,
            final_pred_objectness, final_gt_objectness,
        )

        if self.amp_scaler is not None:
            self.amp_scaler.scale(total_loss).backward()
            self.amp_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIPPING_MAX_NORM)
            self.amp_scaler.step(self.optimizer)
            self.amp_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIPPING_MAX_NORM)
            self.optimizer.step()

        self.scheduler.step()
        self.current_step += 1

        return {k: v.item() for k, v in losses.items()}

    @torch.no_grad()
    def _validate_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        self.model.eval()

        is_video = batch["is_video"][0]
        images = batch["images"]
        gt_masks = batch["gt_masks"]
        
        points = batch["points"]
        labels = batch["labels"]
        boxes = batch["boxes"]
        input_masks = batch["input_masks"]

        
        if is_video:
            num_frames = images.shape[1]
            all_pred_masks = []
            all_pred_ious = []
            all_pred_objectness = []
            all_gt_ious = []
            all_gt_objectness = []

            for t in range(num_frames):
                self.model.reset_memory_bank() # Reset memory for each video
                
                if t == 0:
                    current_points = points
                    current_labels = labels
                    current_boxes = boxes
                    current_input_masks = input_masks
                    is_prompted = True
                else:
                    current_points = None
                    current_labels = None
                    current_boxes = None
                    current_input_masks = None
                    is_prompted = False

                with torch.autocast(device_type=self.config.DEVICE, dtype=torch.bfloat16 if self.config.PRECISION == "bfloat16" else torch.float16, enabled=self.amp_scaler is not None):
                    pred_masks, pred_ious, pred_objectness = self.model(
                        video_frames=images,
                        current_frame_idx=t,
                        points=current_points,
                        labels=current_labels,
                        boxes=current_boxes,
                        masks=current_input_masks,
                        is_prompted_frame=is_prompted,
                    )
                all_pred_masks.append(pred_masks)
                all_pred_ious.append(pred_ious)
                all_pred_objectness.append(pred_objectness)

                current_gt_mask = gt_masks[:, t, :, :, :]
                current_gt_objectness = (current_gt_mask.sum(dim=(-1, -2)) > 0).float()
                current_gt_iou = get_iou(pred_masks[:,0:1,:,:], current_gt_mask)

                all_gt_ious.append(current_gt_iou)
                all_gt_objectness.append(current_gt_objectness)
            
            final_pred_masks = torch.cat(all_pred_masks, dim=0)
            final_pred_ious = torch.cat(all_pred_ious, dim=0)
            final_pred_objectness = torch.cat(all_pred_objectness, dim=0)
            
            final_gt_masks = gt_masks.view(-1, 1, gt_masks.shape[-2], gt_masks.shape[-1])
            final_gt_ious = torch.cat(all_gt_ious, dim=0)
            final_gt_objectness = torch.cat(all_gt_objectness, dim=0)

        else: # Image
            current_frame_idx = 0
            with torch.autocast(device_type=self.config.DEVICE, dtype=torch.bfloat16 if self.config.PRECISION == "bfloat16" else torch.float16, enabled=self.amp_scaler is not None):
                pred_masks, pred_ious, pred_objectness = self.model(
                    video_frames=images,
                    current_frame_idx=current_frame_idx,
                    points=points,
                    labels=labels,
                    boxes=boxes,
                    masks=input_masks,
                    is_prompted_frame=True,
                )
            final_pred_masks = pred_masks
            final_pred_ious = pred_ious
            final_pred_objectness = pred_objectness

            final_gt_masks = gt_masks.squeeze(1) # (B, 1, H, W)
            final_gt_objectness = (final_gt_masks.sum(dim=(-1, -2)) > 0).float()
            final_gt_ious = get_iou(final_pred_masks[:,0:1,:,:], final_gt_masks)
        
        total_loss, losses = self.criterion(
            final_pred_masks, final_gt_masks,
            final_pred_ious, final_gt_ious,
            final_pred_objectness, final_gt_objectness,
        )

        return {k: v.item() for k, v in losses.items()}


    def train(self):
        self.logger.info("Starting training...")
        while self.current_step < self.total_steps:
            for batch in tqdm(self.train_dataloader, desc=f"Training Step {self.current_step}/{self.total_steps}"):
                if self.current_step >= self.total_steps:
                    break
                
                metrics = self._train_step(batch)
                
                if self.current_step % 100 == 0: # Log every 100 steps
                    self.logger.info(
                        f"Step [{self.current_step}/{self.total_steps}] "
                        f"Loss: {metrics['total_loss']:.4f}, "
                        f"Mask Loss: {metrics['mask_loss']:.4f}, "
                        f"IoU Loss: {metrics['iou_loss']:.4f}, "
                        f"Objectness Loss: {metrics['objectness_loss']:.4f}"
                    )
                
                if self.current_step % 1000 == 0: # Validate every 1000 steps
                    self.logger.info("Starting validation...")
                    val_metrics = self.validate()
                    self.logger.info(
                        f"Validation Step [{self.current_step}/{self.total_steps}] "
                        f"Loss: {val_metrics['total_loss']:.4f}, "
                        f"Mask Loss: {val_metrics['mask_loss']:.4f}, "
                        f"IoU Loss: {val_metrics['iou_loss']:.4f}, "
                        f"Objectness Loss: {val_metrics['objectness_loss']:.4f}"
                    )
                    # Save checkpoint
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        self.current_step,
                        val_metrics['total_loss'],
                        f"checkpoint_step_{self.current_step}.pth",
                    )
        self.logger.info("Training finished.")

    def validate(self) -> Dict[str, float]:
        self.model.eval()
        total_metrics = {
            "total_loss": 0.0,
            "mask_loss": 0.0,
            "iou_loss": 0.0,
            "objectness_loss": 0.0,
        }
        num_batches = 0
        for batch in tqdm(self.val_dataloader, desc="Validation"):
            metrics = self._validate_step(batch)
            for k, v in metrics.items():
                total_metrics[k] += v
            num_batches += 1
        
        for k in total_metrics:
            total_metrics[k] /= num_batches
        return total_metrics
