import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import PreTrainedTokenizer
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from accelerate import Accelerator
from typing import Dict, Any, Tuple, Optional, List
import json
import os

from config import Config
from model.navil import NaViLModel
from dataset.multimodal_dataset import MultimodalDataset
from dataset.collate_fn import CustomCollateFn
from utils import (
    setup_logging, get_learning_rate_scheduler, get_optimizer
)
from loguru import logger


class NaViLTrainer:
    """
    Orchestrates the multi-stage training process for the NaViL model,
    including data loading, optimization, scheduling, and checkpointing.
    It leverages Hugging Face's Accelerator for distributed training and mixed precision.
    """

    def __init__(self, model: NaViLModel, tokenizer: PreTrainedTokenizer, config: Config, accelerator: Accelerator):
        """
        Initializes the NaViLTrainer.

        Args:
            model: The NaViLModel instance to train.
            tokenizer: The pre-trained tokenizer for the LLM.
            config: The global configuration object, pre-loaded for a specific model variant.
            accelerator: Hugging Face Accelerator for distributed training and mixed precision.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.accelerator = accelerator
        self.device = accelerator.device

        # The config object is assumed to be loaded for a specific model variant,
        # so `self.config.training_stages` directly refers to that variant's stages.
        self.training_stages_config: Dict[str, Any] = self.config.training_stages

        self.optimizer: Optional[Optimizer] = None
        self.lr_scheduler: Optional[LRScheduler] = None

        self.train_dataloader: Optional[DataLoader] = None
        self.eval_dataloader: Optional[DataLoader] = None

        self.global_step = 0  # Total steps across all stages
        self.start_stage_idx = 0  # Index of the stage to start from (for resuming training)

        # Setup logging specific to trainer's Accelerator rank
        setup_logging(self.accelerator.local_rank)
        logger.info(f"Trainer initialized for rank {self.accelerator.local_rank}.")

        # Checkpoint directory
        self.checkpoint_dir: str = self.config.get("common.checkpoint_dir", "checkpoints")
        if self.accelerator.is_main_process:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            logger.info(f"Checkpoints and training state will be saved to: {self.checkpoint_dir}")

    def _initialize_optimizer_and_scheduler(self, trainable_params: List[nn.Parameter], stage_config: Dict[str, Any], current_step_in_stage: int = 0):
        """
        Initializes or re-initializes the optimizer and learning rate scheduler for a new stage.
        These are then prepared by the Accelerator for distributed training.

        Args:
            trainable_params: A list of model parameters that should be trained in this stage.
            stage_config: Configuration dictionary for the current training stage.
            current_step_in_stage: The current step within the stage (used for resuming scheduler state).
        """
        self.optimizer = get_optimizer(
            trainable_params=trainable_params,
            config=self.config,
            current_stage_config=stage_config,
        )

        total_steps_in_stage = stage_config.training_steps
        self.lr_scheduler = get_learning_rate_scheduler(
            optimizer=self.optimizer,
            config=self.config,
            current_stage_config=stage_config,
            total_steps=total_steps_in_stage,
            current_step_in_stage=current_step_in_stage,
        )
        if self.accelerator.is_main_process:
            logger.info(f"Optimizer and LR scheduler initialized for new stage. "
                        f"Schedule: {stage_config.lr_schedule}, Peak LR: {stage_config.peak_learning_rate}")

        # Prepare optimizer and scheduler for distributed training using Accelerator
        self.optimizer, self.lr_scheduler = self.accelerator.prepare(self.optimizer, self.lr_scheduler)

    def _prepare_dataloaders(self, stage_name: str, stage_config: Dict[str, Any]):
        """
        Prepares training and evaluation DataLoaders for the current stage.
        This involves creating MultimodalDataset and CustomCollateFn instances.

        Args:
            stage_name: The name of the current training stage (e.g., "stage_1_1").
            stage_config: Configuration dictionary for the current training stage.
        """
        data_paths: List[str] = self.config.data_paths[stage_name]

        # Handle global_batch_size, especially for NaViL-9B Stage 2 ambiguity
        global_batch_size: Optional[int] = stage_config.global_batch_size
        if global_batch_size is None:
            # Fallback for NaViL-9B Stage 2 batch size, defaulting to NaViL-2B's Stage 2 batch size.
            if hasattr(self.config, 'model_variants') and 'navil_2b' in self.config.model_variants:
                global_batch_size = self.config.model_variants.navil_2b.training_stages.stage_2.global_batch_size
                if self.accelerator.is_main_process:
                    logger.warning(f"Global batch size for {stage_name} was unspecified. "
                                   f"Defaulting to NaViL-2B's S2 batch size: {global_batch_size}")
            else:
                raise ValueError(f"Global batch size for {stage_name} is unspecified and no fallback for NaViL-2B S2 found in config.")
        
        if not isinstance(global_batch_size, int) or global_batch_size <= 0: # Safety check after fallback
            raise ValueError(f"Global batch size for {stage_name} is invalid: {global_batch_size}")

        # Calculate per-device batch size. `accelerate` handles distributing the global batch size.
        # The paper's `gradient_accumulation_steps: 1` implies the `global_batch_size`
        # is the total effective batch size, distributed across devices with no further accumulation per device.
        if global_batch_size % self.accelerator.num_processes != 0:
            if self.accelerator.is_main_process:
                logger.warning(f"Global batch size ({global_batch_size}) is not perfectly divisible by "
                               f"number of processes ({self.accelerator.num_processes}). "
                               f"Per-device batch size will be floor division: {global_batch_size // self.accelerator.num_processes}.")
        per_device_batch_size: int = global_batch_size // self.accelerator.num_processes
        
        if per_device_batch_size == 0:
            raise ValueError(f"Calculated per-device batch size is 0. Global batch size: {global_batch_size}, "
                             f"Num processes: {self.accelerator.num_processes}. "
                             "Please ensure global batch size is at least num_processes.")

        # Temporarily set `current_stage_name` on the config object for dataset/collate_fn
        # to access stage-specific parameters like VMP enablement.
        self.config.current_stage_name = stage_name 

        full_train_dataset = MultimodalDataset(
            data_paths=data_paths,
            tokenizer=self.tokenizer,
            config=self.config,
            stage=stage_name,
            is_train=True,
        )
        
        # Create a small held-out subset for validation loss calculation, as mentioned in the paper.
        val_fraction: float = self.config.get("common.validation_set_fraction", 0.001) # Default 0.1% for validation
        val_size: int = max(1, int(len(full_train_dataset) * val_fraction))
        train_size: int = len(full_train_dataset) - val_size
        
        if train_size <= 0 and self.accelerator.is_main_process:
            logger.error(f"Training subset is empty (size: {train_size}). Check dataset or validation_set_fraction.")
            raise ValueError("Training dataset is empty.")
        if val_size <= 0 and self.accelerator.is_main_process:
            logger.warning(f"Validation subset is empty (size: {val_size}). No validation loss will be calculated.")
            
        generator = torch.Generator().manual_seed(self.config.get("common.seed", 42))
        train_subset: Subset = Subset(full_train_dataset, [])
        eval_subset: Subset = Subset(full_train_dataset, [])

        if train_size > 0 and val_size > 0:
             train_subset, eval_subset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size], generator=generator)
        elif train_size > 0: # Only training data
             train_subset = full_train_dataset
        elif val_size > 0: # Only validation data (unlikely for training)
             eval_subset = full_train_dataset

        collate_fn = CustomCollateFn(self.tokenizer, self.config)

        self.train_dataloader = DataLoader(
            train_subset,
            batch_size=per_device_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.config.get("common.dataloader_num_workers", 4),
            pin_memory=True,
        )

        self.eval_dataloader = DataLoader(
            eval_subset,
            batch_size=per_device_batch_size,
            shuffle=False,  # No shuffle for evaluation
            collate_fn=collate_fn,
            num_workers=self.config.get("common.dataloader_num_workers", 4),
            pin_memory=True,
        )

        # Prepare dataloaders for distributed training using Accelerator
        self.train_dataloader, self.eval_dataloader = self.accelerator.prepare(
            self.train_dataloader, self.eval_dataloader
        )
        
        if self.accelerator.is_main_process:
            logger.info(f"Dataloaders prepared for {stage_name}. "
                        f"Train samples: {len(train_subset)}, Eval samples: {len(eval_subset)}. "
                        f"Per-device batch size: {per_device_batch_size}.")

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Performs a single training step (forward pass, backward pass, optimizer update).

        Args:
            batch: A dictionary containing batched tensors from the CustomCollateFn.

        Returns:
            The detached, mean-reduced loss for the current step.
        """
        self.model.train()  # Ensure model is in training mode
        
        # `accelerator.accumulate` handles gradient accumulation if needed,
        # based on `gradient_accumulation_steps` set in Accelerator config.
        with self.accelerator.accumulate(self.model):
            logits, loss = self.model(
                pixel_values=batch['images'],  # `images` from collate_fn are raw pixel values
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )

            self.accelerator.backward(loss)

            # Gradient clipping (optional, not explicitly detailed in paper, but good practice)
            # max_grad_norm = self.config.get("common.max_grad_norm", None)
            # if max_grad_norm is not None:
            #     # `accelerator.clip_grad_norm_` handles unwrapping the model
            #     self.accelerator.clip_grad_norm_(self.model.parameters(), max_grad_norm)

            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
        
        # Unscale the loss for logging if using mixed precision, then reduce across processes for aggregate loss
        return self.accelerator.reduce(loss.detach(), reduction="mean").item()

    def _evaluate_loss(self, dataloader: DataLoader) -> float:
        """
        Calculates the average validation loss over the evaluation dataloader.

        Args:
            dataloader: The DataLoader for the evaluation dataset.

        Returns:
            The average validation loss.
        """
        self.model.eval()  # Set model to evaluation mode
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                # Dataloader is already prepared by accelerator, so batch tensors are on device
                logits, loss = self.model(
                    pixel_values=batch['images'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                total_loss += self.accelerator.reduce(loss.detach(), reduction="mean").item()
                num_batches += 1

        self.model.train()  # Set model back to training mode
        return total_loss / num_batches if num_batches > 0 else 0.0

    def _run_stage(self, stage_name: str, stage_idx: int, resume_step_in_stage: int = 0):
        """
        Executes a single training stage (e.g., "stage_1_1", "stage_1_2", or "stage_2").

        Args:
            stage_name: The name of the current training stage.
            stage_idx: The numerical index of the current stage in the sequence.
            resume_step_in_stage: The step number within this stage to resume from.
        """
        if self.accelerator.is_main_process:
            logger.info(f"--- Starting training stage: {stage_name} (Index: {stage_idx}) ---")

        stage_config: Dict[str, Any] = self.training_stages_config[stage_name]
        total_steps_in_stage: int = stage_config.training_steps
        
        # 1. Set trainable parameters based on the stage (freeze/unfreeze logic)
        trainable_params: List[nn.Parameter] = self.model.get_trainable_params(stage_name)
        
        # 2. Re-initialize optimizer and scheduler for the current stage.
        # These will be prepared by the Accelerator during this call.
        self._initialize_optimizer_and_scheduler(trainable_params, stage_config, current_step_in_stage=resume_step_in_stage)
        
        # 3. Prepare/Re-prepare DataLoaders for the current stage.
        self._prepare_dataloaders(stage_name, stage_config)

        # Loop through steps in the current stage
        for step_in_stage in range(resume_step_in_stage, total_steps_in_stage):
            # Using `next(iter(self.train_dataloader))` to get a batch.
            # The dataloader is already prepared by Accelerator.
            batch: Dict[str, torch.Tensor] = next(iter(self.train_dataloader)) 
            
            current_loss: float = self._train_step(batch)
            self.global_step += 1  # Global step is cumulative across all stages

            if self.accelerator.is_main_process:
                # Log training progress at specified intervals
                if self.global_step % self.config.get("common.log_interval", 50) == 0:
                    current_lr: float = self.optimizer.param_groups[0]['lr']
                    logger.info(f"Stage: {stage_name} ({stage_idx+1}/{len(self.training_stages_config.keys())}) | "
                                f"Global Step: {self.global_step} | Stage Step: {step_in_stage+1}/{total_steps_in_stage} | "
                                f"Loss: {current_loss:.4f} | LR: {current_lr:.6f}")

                # Evaluate validation loss at specified intervals
                if (step_in_stage + 1) % self.config.get("common.eval_interval", 5000) == 0:
                    val_loss: float = self._evaluate_loss(self.eval_dataloader)
                    logger.info(f"Validation Loss after {self.global_step} global steps "
                                f"({step_in_stage+1} stage steps): {val_loss:.4f}")

                # Save Accelerator state at specified intervals
                if (step_in_stage + 1) % self.config.get("common.save_interval", 10000) == 0:
                    self.accelerator.save_state(output_dir=self.checkpoint_dir)
                    logger.info(f"Accelerator state saved at global step {self.global_step} to {self.checkpoint_dir}")
                    # Also save meta-information for robust resume (global step, next stage index, next step in current stage)
                    training_meta_state_path: str = os.path.join(self.checkpoint_dir, "training_meta_state.json")
                    with open(training_meta_state_path, "w") as f:
                        json.dump({"global_step": self.global_step, 
                                   "start_stage_idx": stage_idx, # Continue this stage
                                   "resume_step_in_stage": step_in_stage + 1}, f)

        if self.accelerator.is_main_process:
            logger.info(f"--- Stage {stage_name} completed ---")
            # Final evaluation at the end of the stage
            final_val_loss: float = self._evaluate_loss(self.eval_dataloader)
            logger.info(f"Final Validation Loss for {stage_name}: {final_val_loss:.4f}")
            
            # Save Accelerator state at the end of the stage
            self.accelerator.save_state(output_dir=self.checkpoint_dir)
            logger.info(f"Accelerator state saved at end of stage {stage_name} to {self.checkpoint_dir}")
            
            # Update and save meta-information to indicate readiness for the next stage
            training_meta_state_path = os.path.join(self.checkpoint_dir, "training_meta_state.json")
            with open(training_meta_state_path, "w") as f:
                json.dump({"global_step": self.global_step, 
                           "start_stage_idx": stage_idx + 1,  # Ready for the next stage
                           "resume_step_in_stage": 0}, f)     # Start from step 0 in the next stage

    def train(self):
        """
        Orchestrates the multi-stage training process for NaViL, handling resume logic
        and delegating to `_run_stage` for each individual stage.
        """
        if self.accelerator.is_main_process:
            logger.info("Starting NaViL training process.")

        # Load latest checkpoint if available using Accelerator's mechanism
        training_meta_state_path: str = os.path.join(self.checkpoint_dir, "training_meta_state.json")
        
        loaded_global_step: int = 0
        loaded_start_stage_idx: int = 0
        loaded_resume_step_in_stage: int = 0

        # Attempt to load training meta state for resume
        if os.path.exists(training_meta_state_path):
            try:
                with open(training_meta_state_path, "r") as f:
                    training_meta_state: Dict[str, int] = json.load(f)
                loaded_global_step = training_meta_state.get("global_step", 0)
                loaded_start_stage_idx = training_meta_state.get("start_stage_idx", 0)
                loaded_resume_step_in_stage = training_meta_state.get("resume_step_in_stage", 0)
                if self.accelerator.is_main_process:
                    logger.info(f"Loaded training meta state: Global Step: {loaded_global_step}, "
                                f"Start Stage Index: {loaded_start_stage_idx}, "
                                f"Resume Step in Current Stage: {loaded_resume_step_in_stage}.")
            except Exception as e:
                if self.accelerator.is_main_process:
                    logger.warning(f"Error loading training_meta_state.json: {e}. Starting from scratch.")
        
        self.global_step = loaded_global_step
        self.start_stage_idx = loaded_start_stage_idx
        
        # Prepare model for accelerator (optimizer and scheduler are prepared per stage in _initialize_optimizer_and_scheduler)
        self.model = self.accelerator.prepare(self.model)

        # Attempt to load full Accelerator state if resuming
        if loaded_global_step > 0:
            try:
                self.accelerator.load_state(input_dir=self.checkpoint_dir)
                if self.accelerator.is_main_process:
                    logger.info(f"Accelerator state loaded from {self.checkpoint_dir}. Resuming training.")
            except Exception as e:
                if self.accelerator.is_main_process:
                    logger.error(f"Error loading Accelerator state from {self.checkpoint_dir}: {e}. "
                                 "This can happen if the model architecture or Accelerator config changed. "
                                 "Proceeding with model parameters loaded (if any from `load_state_dict` in utils), "
                                 "but optimizer/scheduler might need re-initialization.")
                # If Accelerator state loading fails, reset to start from scratch for robust behavior
                self.global_step = 0
                self.start_stage_idx = 0
                loaded_resume_step_in_stage = 0
                if self.accelerator.is_main_process:
                    logger.info("Resetting to start from scratch due to Accelerator state loading failure.")
        else:
            if self.accelerator.is_main_process:
                logger.info("No Accelerator state found, starting training from scratch.")

        stages_list: List[str] = list(self.training_stages_config.keys())

        for idx, stage_name in enumerate(stages_list):
            if idx < self.start_stage_idx:
                if self.accelerator.is_main_process:
                    logger.info(f"Skipping already completed stage: {stage_name}")
                continue
            
            # For the stage we are resuming, use the loaded_resume_step_in_stage;
            # otherwise, start from step 0 for subsequent new stages.
            current_stage_resume_step: int = loaded_resume_step_in_stage if idx == loaded_start_stage_idx else 0
            
            self._run_stage(stage_name, idx, current_stage_resume_step)

            # After a stage, `loaded_resume_step_in_stage` should reset for subsequent stages
            # The `training_meta_state.json` is updated at the end of `_run_stage`
            # to point to the *next* stage or to resume at step 0 if current stage completed.
            loaded_resume_step_in_stage = 0 
            
        if self.accelerator.is_main_process:
            logger.info("NaViL training completed successfully across all stages.")

