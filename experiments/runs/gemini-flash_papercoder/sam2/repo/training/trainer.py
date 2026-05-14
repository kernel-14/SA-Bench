```python
import os
import math
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from contextlib import nullcontext
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple, Union

# Import custom modules
from config import Config
from model.sam2_model import SAM2Model
from model.memory_modules import MemoryBank
from training.losses import Losses
from training.optimizer_scheduler import OptimizerScheduler
from utils.metrics import calculate_miou, calculate_j_and_f, calculate_g # Import for IoU ground truth calculation and eval
from data.prompt_simulator import PromptSimulator # Required for dynamic prompt generation in _train_step


class Trainer(object):
    """
    Orchestrates the training lifecycle of the SAM 2 model.
    Manages data loading, model updates, loss calculations, mixed precision,
    distributed training, logging, and checkpointing.
    """

    def __init__(
        self,
        model: SAM2Model,
        config: Config,
        train_loaders: Dict[str, DataLoader],
        val_loaders: Dict[str, DataLoader],
        device: torch.device,
    ):
        """
        Initializes the Trainer with model, configuration, data loaders, and device.

        Args:
            model (SAM2Model): The main SAM 2 model instance.
            config (Config): The global configuration object.
            train_loaders (Dict[str, DataLoader]): Dictionary of training DataLoaders,
                                                   keyed by dataset name (e.g., "SA-V", "SA-1B_subset").
            val_loaders (Dict[str, DataLoader]): Dictionary of validation DataLoaders.
            device (torch.device): The computational device (cuda or cpu).
        """
        self.model = model
        self.config = config
        self.train_loaders = train_loaders
        self.val_loaders = val_loaders
        self.device = device
        self.global_step: int = 0

        # Create output directories
        self.log_dir = os.path.join(self.config.paths.log_dir, self.config.model.name)
        self.checkpoint_dir = os.path.join(self.config.paths.checkpoint_dir, self.config.model.name)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Determine the current active training stage configuration (pretrain, full_train)
        # This will inform optimizer/scheduler and loss weights.
        if self.config.training.pretrain.enabled:
            self.current_training_stage_config_key = "training.pretrain"
        elif self.config.training.full_train.enabled:
            self.current_training_stage_config_key = "training.full_train"
        else:
            raise ValueError("No training stage is enabled in config. 'pretrain' or 'full_train' must be enabled.")
        
        # Get total steps for the current training stage
        if self.current_training_stage_config_key == "training.pretrain":
            total_steps = self.config.training.pretrain.steps
        else: # "training.full_train"
            # Calculate total steps based on max loader length and num_epochs, if steps is not explicitly defined.
            # Assuming number of iterations will be fixed or derived from dataset size.
            # If `num_epochs` is specified for full_train, use it. Else, fall back to a default large number.
            num_epochs = self.config.training.full_train.get('num_epochs', 100) # Default if not specified
            max_loader_len = self._get_max_loader_len(self.train_loaders)
            total_steps = num_epochs * max_loader_len
            if total_steps == 0:
                raise ValueError("Calculated total training steps is 0. Check train_loaders or num_epochs.")

        # Initialize Optimizer and Scheduler
        self.optimizer_scheduler = OptimizerScheduler(
            model=self.model,
            config=self.config,
            total_steps=total_steps,
            stage_config_key=self.current_training_stage_config_key,
        )
        self.optimizer = self.optimizer_scheduler.optimizer

        # Initialize Loss Functions
        self.losses_fn = Losses()
        losses_cfg = self.config.get(f"{self.current_training_stage_config_key}.losses")
        self.loss_weights = {
            'focal': losses_cfg.get('mask_focal_weight', 20),
            'dice': losses_cfg.get('mask_dice_weight', 1),
            'l1': losses_cfg.get('iou_l1_weight', 1),
            'ce': losses_cfg.get('occlusion_ce_weight', 1), # CE for occlusion head
        }
        
        # Mixed Precision Scaler
        self.use_amp = (
            self.config.get(f"{self.current_training_stage_config_key}.precision") == "bfloat16"
            and self.device.type == "cuda"
        )
        self.scaler = GradScaler() if self.use_amp else None
        
        # Distributed Training setup
        self.num_gpus = self.config.system.num_gpus
        if self.num_gpus > 1:
            print(f"Wrapping model in DistributedDataParallel for {self.num_gpus} GPUs.")
            self.model = DDP(self.model, device_ids=[self.device.index])

        # Prompt simulator for interactive training (for video data)
        self.prompt_simulator = PromptSimulator(self.config)
        
        # Store configuration for the active training stage directly for easier access
        self.active_train_cfg = self.config.get(self.current_training_stage_config_key)
        self.full_train_cfg = self.config.training.full_train # For video-specific params even during pretrain to avoid Nones
        self.pretrain_cfg = self.config.training.pretrain


    def _get_max_loader_len(self, loaders: Dict[str, DataLoader]) -> int:
        """
        Calculates the length of the longest DataLoader.
        Used to determine total training steps for epoch-based training.
        """
        max_len = 0
        for loader in loaders.values():
            if hasattr(loader, '__len__'):
                max_len = max(max_len, len(loader))
        return max_len if max_len > 0 else 1 # Avoid division by zero if loaders are empty

    def _log_metrics(self, metrics: Dict[str, Any], step: int, phase: str, iteration: Optional[int] = None) -> None:
        """
        Logs training/validation metrics to console. Can be extended to TensorBoard.
        """
        log_str = f"[{phase}] Step {step}/{self.optimizer_scheduler.total_steps}"
        if iteration is not None:
            log_str += f" (Iter {iteration})"
        log_str += f" | LR: {self.optimizer_scheduler.get_lr():.6f}"
        for k, v in metrics.items():
            log_str += f" | {k}: {v:.4f}"
        print(log_str)

        # TODO: Integrate with TensorBoard or W&B for richer logging
        # if self.config.logging.tensorboard_enabled:
        #     self.writer.add_scalar(f'{phase}/total_loss', metrics['total_loss'], step)
        #     ...

    def _save_checkpoint(self, step: int, phase: str = "train") -> None:
        """
        Saves the current model, optimizer, scheduler, and scaler states.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.config.model.name}_{phase}_step_{step:07d}.pth")
        
        model_state_dict = self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict()
        
        save_dict = {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "config": self.config._raw_data, # Save raw dict for re-initialization
        }
        if self.scaler:
            save_dict["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(save_dict, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Loads model, optimizer, scheduler, and scaler states from a checkpoint.
        """
        if not os.path.exists(checkpoint_path):
            print(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")
            return

        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        model_to_load = self.model.module if isinstance(self.model, DDP) else self.model
        model_to_load.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.global_step = checkpoint["global_step"]
        print(f"Resumed training from global step {self.global_step}")
        
        # Re-initialize optimizer_scheduler to reflect loaded global_step and correct LR
        # This will set the LR based on the schedule for the current global_step.
        self.optimizer_scheduler._current_step = self.global_step
        self.optimizer_scheduler.step() # Apply the LR schedule for the current step


    def _calculate_losses(
        self,
        pred_masks_multi_variant: torch.Tensor, # (B, num_mask_variants, H, W)
        pred_ious_multi_variant: torch.Tensor,  # (B, num_mask_variants)
        pred_occlusions_single_variant: torch.Tensor, # (B, 1) or (B,)
        gt_masks: torch.Tensor, # (B, 1, H, W) or (B, H, W)
        gt_occlusions: torch.Tensor, # (B,) boolean tensor
        is_training: bool,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculates and combines losses for a given batch of predictions.
        Handles multi-mask predictions and conditional supervision based on occlusion.

        Args:
            pred_masks_multi_variant (torch.Tensor): Raw logits (or probabilities for dice/focal) for multiple mask variants.
                                            Shape: (B, num_mask_variants, H, W).
            pred_ious_multi_variant (torch.Tensor): Predicted IoU scores for multiple mask variants.
                                            Shape: (B, num_mask_variants).
            pred_occlusions_single_variant (torch.Tensor): Predicted occlusion probability.
                                            Shape: (B, 1) or (B,).
            gt_masks (torch.Tensor): Ground truth binary masks. Shape: (B, 1, H, W) or (B, H, W).
            gt_occlusions (torch.Tensor): Ground truth occlusion status (boolean). Shape: (B,).
            is_training (bool): Flag indicating if in training mode (affects certain behaviors if needed).

        Returns:
            Dict[str, torch.Tensor]: A dictionary of calculated loss components.
        """
        batch_size = pred_masks_multi_variant.shape[0]
        num_mask_variants = pred_masks_multi_variant.shape[1]
        
        # Ensure gt_masks has a channel dimension for consistency (B, 1, H, W)
        if gt_masks.ndim == 3:
            gt_masks = gt_masks.unsqueeze(1)

        total_loss_focal = torch.tensor(0.0, device=self.device)
        total_loss_dice = torch.tensor(0.0, device=self.device)
        total_loss_l1 = torch.tensor(0.0, device=self.device)
        total_loss_ce_occlusion = torch.tensor(0.0, device=self.device)
        
        # Iterate over each item in the batch
        for i in range(batch_size):
            current_gt_mask = gt_masks[i] # (1, H, W)
            current_gt_occluded = gt_occlusions[i].item() # boolean scalar

            # Occlusion Loss (always calculate if head is enabled)
            # Ensure target is 0/1 for BCEWithLogits
            occlusion_target = torch.tensor([1.0 if current_gt_occluded else 0.0], device=self.device).float()
            total_loss_ce_occlusion += self.losses_fn.cross_entropy_loss(
                pred_occlusions_single_variant[i], occlusion_target, reduction='mean'
            )

            if current_gt_occluded:
                # If object is occluded, no mask or IoU loss
                continue

            # If not occluded, calculate mask and IoU losses
            pred_masks_frame_variants = pred_masks_multi_variant[i] # (num_mask_variants, H, W)
            pred_ious_frame_variants = pred_ious_multi_variant[i] # (num_mask_variants,)

            # Calculate segmentation loss for each mask variant
            seg_losses_per_variant = torch.zeros(num_mask_variants, device=self.device)
            for j in range(num_mask_variants):
                focal_loss = self.losses_fn.focal_loss(
                    pred_masks_frame_variants[j].unsqueeze(0), # (1, H, W)
                    current_gt_mask,
                    reduction='mean',
                )
                dice_loss = self.losses_fn.dice_loss(
                    pred_masks_frame_variants[j].unsqueeze(0),
                    current_gt_mask,
                    reduction='mean',
                )
                seg_losses_per_variant[j] = self.loss_weights['focal'] * focal_loss + self.loss_weights['dice'] * dice_loss

            # "Supervise the mask logits with the lowest segmentation loss"
            best_pred_idx = torch.argmin(seg_losses_per_variant)
            best_seg_loss = seg_losses_per_variant[best_pred_idx]
            
            # Add to total segmentation losses
            total_loss_focal += self.loss_weights['focal'] * self.losses_fn.focal_loss(
                pred_masks_frame_variants[best_pred_idx].unsqueeze(0), current_gt_mask, reduction='mean'
            )
            total_loss_dice += self.loss_weights['dice'] * self.losses_fn.dice_loss(
                pred_masks_frame_variants[best_pred_idx].unsqueeze(0), current_gt_mask, reduction='mean'
            )

            # "Supervise IoU predictions of all masks to encourage better learning of when a mask might be bad"
            # The paper says: "supervise the IoU predictions of all masks"
            # BUT: "supervise the mask logits with the lowest segmentation loss"
            # For IoU loss, it seems we use the best mask's IoU prediction vs. actual IoU.
            # Let's align with SAM's training which often uses the best mask for IoU supervision.

            # Calculate actual IoU (ground truth IoU) for the best predicted mask
            best_pred_mask_binary = (pred_masks_frame_variants[best_pred_idx] > 0.5).to(torch.bool).unsqueeze(0)
            actual_iou = calculate_miou(best_pred_mask_binary, current_gt_mask)
            
            total_loss_l1 += self.loss_weights['l1'] * self.losses_fn.l1_loss(
                pred_ious_frame_variants[best_pred_idx], torch.tensor([actual_iou], device=self.device), reduction='mean'
            )
        
        # Average losses over the batch
        total_loss_focal /= batch_size
        total_loss_dice /= batch_size
        total_loss_l1 /= batch_size
        total_loss_ce_occlusion /= batch_size

        total_loss = total_loss_focal + total_loss_dice + total_loss_l1 + total_loss_ce_occlusion
        
        return {
            'total_loss': total_loss,
            'mask_focal_loss': total_loss_focal,
            'mask_dice_loss': total_loss_dice,
            'iou_l1_loss': total_loss_l1,
            'occlusion_ce_loss': total_loss_ce_occlusion,
        }

    def _select_frames_for_interactive_training(self, sequence_length: int) -> List[int]:
        """
        Randomly selects frames to prompt for interactive training simulation.
        One frame must be the first frame (index 0). Up to max_prompted_frames_per_seq frames.
        """
        max_prompted_frames_per_seq = self.full_train_cfg.get('max_prompted_frames_per_seq', 2)
        
        if sequence_length == 0:
            return []

        # Ensure the first frame is always included if any frames are prompted
        num_frames_to_prompt = random.randint(1, min(max_prompted_frames_per_seq, sequence_length))
        
        if num_frames_to_prompt == 1:
            return [0] # Only prompt the first frame
        
        # If num_frames_to_prompt > 1, ensure 0 is in the list, and pick others randomly
        other_frames_indices = list(range(1, sequence_length)) # Exclude 0
        if len(other_frames_indices) < num_frames_to_prompt - 1:
            # Not enough other frames, just prompt all available frames up to num_frames_to_prompt
            prompted_frames = list(range(min(num_frames_to_prompt, sequence_length)))
        else:
            sampled_other_frames = random.sample(other_frames_indices, num_frames_to_prompt - 1)
            prompted_frames = sorted([0] + sampled_other_frames)
        
        return prompted_frames


    def _train_step(self, batch: Dict[str, Any], data_source_name: str) -> Dict[str, float]:
        """
        Performs a single training step for a given batch.
        Handles alternating between image and video data, and interactive prompt simulation.

        Args:
            batch (Dict[str, Any]): A dictionary containing the data for the current batch.
            data_source_name (str): The name of the dataset the batch came from (e.g., "SA-V", "SA-1B_subset").

        Returns:
            Dict[str, float]: A dictionary of scalar loss values for logging.
        """
        self.model.train()
        self.optimizer.zero_grad()

        is_video_data: bool = batch['meta'][0]['is_video'] # All items in batch are either video or image

        total_batch_losses = {
            'total_loss': torch.tensor(0.0, device=self.device),
            'mask_focal_loss': torch.tensor(0.0, device=self.device),
            'mask_dice_loss': torch.tensor(0.0, device=self.device),
            'iou_l1_loss': torch.tensor(0.0, device=self.device),
            'occlusion_ce_loss': torch.tensor(0.0, device=self.device),
        }
        
        # Assuming batch_size=1 for video data and potentially larger for image data
        # For simplicity in handling varying structures of `batch`, process each item (video/image) individually.
        # This means the DataLoader provides `batch` where items are already lists/tensors, with batch_size=1
        # for `DataLoader` created in `main.py`.
        # However, `SA1BDataset` returns `gt_masks` as `(N_sampled_masks, 1, H, W)`.
        # `video_frames` for image is `(C, H, W)`.
        # Let's assume the DataLoader `collate_fn` handles batching such that `batch['video_frames']`
        # is `List[Tensor(C,H,W)]` for video, and `Tensor(B,C,H,W)` for image.
        # And `gt_masks` is `List[List[Tensor(1,H,W)]]` for video (one list of masks per video in batch)
        # `Tensor(B, N_masks, 1, H, W)` for image.

        # Adapt batch structure to always be a list of individual items (videos/images)
        if is_video_data:
            # `batch['video_frames']` is List[List[Tensor(C,H,W)]] if batch_size > 1, or List[Tensor(C,H,W)] if batch_size = 1
            # `batch['gt_masks']` is List[List[Tensor(1,H,W)]] if batch_size > 1, or List[Tensor(1,H,W)] if batch_size = 1
            # So, for consistency, treat each as a batch of 1.
            video_frames_batch = batch['video_frames'][0] # List[Tensor(C,H,W)]
            gt_masks_batch = batch['gt_masks'][0]         # List[Tensor(1,H,W)]
            gt_occlusions_batch = batch['gt_occlusions'][0] # List[bool]
            meta_batch = batch['meta'][0]                   # Dict
            
            # Ensure lists are on device
            video_frames_on_device = [f.to(self.device) for f in video_frames_batch]
            gt_masks_on_device = [m.to(self.device) for m in gt_masks_batch]
            gt_occlusions_on_device = torch.tensor(gt_occlusions_batch, device=self.device, dtype=torch.bool)
            
            sequence_length = len(video_frames_on_device)
            memory_bank = MemoryBank(self.config).to(self.device)
            
            # For interactive training of video, we need to simulate the sequence of interactions
            # by processing frame by frame and dynamically generating prompts.
            frames_to_prompt_indices = self._select_frames_for_interactive_training(sequence_length)
            
            # Store temporary model predictions for corrective prompt generation.
            # Initialize with GT masks, so first corrective clicks are based on "perfect" prior knowledge.
            # The paper says: "sampled using the ground-truth masklet and model predictions during training."
            # The prompt_simulator for `generate_correction_prompts` takes a `pred_mask`.
            # This pred_mask must be an actual prediction from the model.
            
            # This implies a sequential forward pass within _train_step.
            # We need to accumulate gradients over this sequence.
            
            all_frame_losses: List[Dict[str, torch.Tensor]] = []
            current_model_pred_masks_for_prompt_gen = [torch.zeros_like(gt_m.squeeze(0), dtype=torch.float32) for gt_m in gt_masks_on_device] # Store probabilities
            
            for seq_frame_idx in range(sequence_length):
                current_frame_tensor = video_frames_on_device[seq_frame_idx]
                current_gt_mask = gt_masks_on_device[seq_frame_idx] # (1, H, W)
                current_gt_occluded = gt_occlusions_on_device[seq_frame_idx].item()

                prompts_for_current_frame: Dict[str, Any] = {}
                is_prompted_current_frame = False

                if seq_frame_idx in frames_to_prompt_indices:
                    is_prompted_current_frame = True
                    if seq_frame_idx == frames_to_prompt_indices[0]: # First prompted frame (initial prompt)
                        prompts_for_current_frame = self.prompt_simulator.generate_initial_prompts(
                            current_gt_mask.squeeze(0) # (H, W)
                        )
                    else: # Subsequent prompted frame (corrective click)
                        prev_pred_mask_for_prompt = (current_model_pred_masks_for_prompt_gen[seq_frame_idx] > 0.5).to(torch.bool) # Binary
                        prompts_for_current_frame = self.prompt_simulator.generate_correction_prompts(
                            current_gt_mask.squeeze(0), # (H, W)
                            prev_pred_mask_for_prompt,
                            num_clicks=self.full_train_cfg.get('max_prompted_frames_per_seq', 2), # Using this as num_clicks
                            is_training=True
                        )
                
                # Model forward for single frame
                with autocast(device_type=self.device.type, dtype=torch.bfloat16) if self.scaler else nullcontext():
                    # SAM2Model.forward expects Lists for images and prompts
                    pred_masks_f, pred_ious_f, pred_occlusions_f, memory_bank = self.model.forward(
                        images=[current_frame_tensor], # List of 1 image
                        prompts=[prompts_for_current_frame], # List of 1 prompt dict
                        memory_bank=memory_bank,
                        is_training=True,
                    )
                    # pred_masks_f: List[Tensor(1, num_mask_variants, H, W)]
                    # pred_ious_f: List[Tensor(1, num_mask_variants)]
                    # pred_occlusions_f: List[Tensor(1, 1)]

                # Calculate losses for this frame and add to list
                frame_losses = self._calculate_losses(