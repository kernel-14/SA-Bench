"""
This module implements the Pretrainer class, which orchestrates the pretraining
phase of the OLMoE model. It manages the training loop, data loading,
optimization, logging, evaluation, and checkpointing, leveraging `accelerate`
for distributed and mixed-precision training.
"""

import os
import math
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs
from typing import Dict, Any, Tuple, Optional

# Local imports from the project structure
from config import Config
from model.olmoe_model import OLMoEModel
from data.pretraining_dataset import PretrainingDataset
from data.data_collator import OLMoEDataCollator
from training.loss_functions import LossCalculator
from training.optimizer_scheduler import OptimizerSchedulerFactory
from utils.logger import Logger
from evaluation.evaluator import Evaluator # Evaluator is required for running evaluations


class Pretrainer:
    """
    Manages the pretraining process for the OLMoE model.

    This class handles the main training loop, data loading and shuffling,
    optimizer and learning rate scheduler steps, gradient accumulation,
    mixed-precision training, distributed training using Hugging Face Accelerate,
    logging with Weights & Biases, periodic evaluation, and checkpointing.
    """

    def __init__(
        self,
        model: OLMoEModel,
        train_ds: PretrainingDataset,
        eval_ds_map: Dict[str, Dataset], # Map of evaluation dataset names to Dataset objects
        config: Config,
        tokenizer: PreTrainedTokenizer,
        logger: Logger,
        loss_calculator: LossCalculator,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    ):
        """
        Initializes the Pretrainer with model, data, and training configurations.

        Args:
            model: The OLMoEModel instance to be pretrained.
            train_ds: The PretrainingDataset for training data.
            eval_ds_map: A dictionary mapping evaluation dataset names to Dataset objects.
            config: The global configuration object.
            tokenizer: The PreTrainedTokenizer instance.
            logger: The Logger instance for experiment tracking.
            loss_calculator: The LossCalculator instance for computing losses.
            optimizer: The AdamW optimizer configured for pretraining.
            lr_scheduler: The learning rate scheduler configured for pretraining.
        """
        self.model = model
        self.train_ds = train_ds
        self.eval_ds_map = eval_ds_map
        self.config = config
        self.tokenizer = tokenizer
        self.logger = logger
        self.loss_calculator = loss_calculator
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # Accelerator initialization
        # DDP kwargs to enable gradient checkpointing if desired, or other DDP specific options
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True) # Find unused params for MoE possibly
        
        self.accelerator = Accelerator(
            mixed_precision=self.config.training.precision,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            log_with="wandb",
            project_dir=self.config.training.checkpoint_dir, # Project directory for accelerator states
            project_config=ProjectConfiguration(
                project_dir=self.config.training.checkpoint_dir,
                logging_dir=os.path.join(self.config.training.checkpoint_dir, "logs")
            ),
            kwargs_handlers=[ddp_kwargs],
        )
        # Set up accelerator logging to wandb
        self.accelerator.init_trackers(
            project_name=self.config.training.project_name,
            config=self.config.__dict__, # Pass the full config object
            init_kwargs={"wandb": {"name": self.config.training.run_name}}
        )

        # Data collator
        self.data_collator = OLMoEDataCollator(self.tokenizer, self.config.data.max_seq_len)

        # DataLoader for training
        self.train_dataloader = DataLoader(
            self.train_ds,
            batch_size=self.config.training.per_device_batch_size_samples,
            shuffle=True, # Will be reshuffled by train_ds.shuffle_data at epoch start
            collate_fn=self.data_collator,
            num_workers=os.cpu_count() or 0, # Use CPU cores or 0 if unknown
            pin_memory=True,
            drop_last=True, # Crucial for consistent batch sizes in distributed training
        )

        # DataLoaders for evaluation
        self.eval_dataloaders: Dict[str, DataLoader] = {}
        for name, ds in self.eval_ds_map.items():
            self.eval_dataloaders[name] = DataLoader(
                ds,
                batch_size=self.config.training.per_device_batch_size_samples, # Can be higher for eval
                shuffle=False,
                collate_fn=self.data_collator,
                num_workers=os.cpu_count() or 0,
                pin_memory=True,
                drop_last=False,
            )

        # Prepare all components for distributed training and mixed precision
        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            *prepared_eval_dataloaders,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_dataloader,
            *self.eval_dataloaders.values(),
        )
        # Re-map prepared eval dataloaders
        self.eval_dataloaders = {
            name: prepared_eval_dataloaders[i] for i, name in enumerate(self.eval_ds_map.keys())
        }

        # Calculate training parameters
        self.tokens_per_global_batch = (
            self.config.training.global_batch_size_samples * self.config.data.max_seq_len
        )
        if self.tokens_per_global_batch <= 0:
            raise ValueError(
                "Calculated tokens_per_global_batch is zero or negative. "
                "Check global_batch_size_samples and max_seq_len in config."
            )

        # Total training steps based on total tokens
        self.total_training_steps = math.ceil(
            self.config.training.total_tokens / self.tokens_per_global_batch
        )
        
        # Calculate annealing start step
        annealing_tokens_in_steps = math.ceil(
            self.config.training.annealing_tokens / self.tokens_per_global_batch
        )
        self.annealing_start_step = self.total_training_steps - annealing_tokens_in_steps
        # Ensure annealing_start_step is not negative or beyond total_training_steps
        self.annealing_start_step = max(0, min(self.annealing_start_step, self.total_training_steps))


        self.current_step = 0
        self.current_tokens_processed = 0
        self.current_epoch = 0

        # Total tokens in the dataset for epoch management
        # Note: self.train_ds must have `total_tokens_in_dataset` calculated after preprocessing
        if not hasattr(self.train_ds, 'total_tokens_in_dataset') or self.train_ds.total_tokens_in_dataset == 0:
            raise RuntimeError("PretrainingDataset must have 'total_tokens_in_dataset' attribute calculated.")
        self.tokens_per_full_epoch = self.train_ds.total_tokens_in_dataset


        # Log basic info on main process
        if self.accelerator.is_main_process:
            self.logger.watch_model(self.model)
            self.logger.log({"total_training_steps": self.total_training_steps}, step=0)
            print(f"Total training steps calculated: {self.total_training_steps}")
            print(f"Annealing phase starts at step: {self.annealing_start_step}")


    def train(self):
        """
        Executes the main pretraining loop.
        """
        # Training loop
        while self.current_step < self.total_training_steps:
            if self.accelerator.is_main_process:
                print(f"\n--- Epoch {self.current_epoch + 1} / {math.ceil(self.config.training.total_tokens / self.tokens_per_full_epoch)} ---")

            # Epoch management: reshuffle data at the beginning of each conceptual epoch
            # The paper states: "We shuffle all samples randomly at the beginning of each epoch"
            # and "During our annealing phase (...) we first reshuffle the entire dataset".
            # This logic supports shuffling at epoch boundaries for the full dataset.
            if self.current_tokens_processed >= (self.current_epoch + 1) * self.tokens_per_full_epoch:
                self.current_epoch += 1
                if self.accelerator.is_main_process:
                    print(f"Reshuffling dataset for new epoch {self.current_epoch + 1}")
                self.train_ds.shuffle_data()
                # Re-create DataLoader from reshuffled dataset (needs to be prepared again by accelerator)
                self.train_dataloader = self.accelerator.prepare(
                    DataLoader(
                        self.train_ds,
                        batch_size=self.config.training.per_device_batch_size_samples,
                        shuffle=True, # DataLoader shuffle is technically ignored if `shuffle_data` called outside.
                                      # But it's good practice to keep it.
                        collate_fn=self.data_collator,
                        num_workers=os.cpu_count() or 0,
                        pin_memory=True,
                        drop_last=True,
                    )
                )


            for batch_idx, batch in enumerate(self.train_dataloader):
                if self.current_step >= self.total_training_steps:
                    break # Stop if total steps reached within an epoch

                with self.accelerator.accumulate(self.model):
                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]
                    labels = batch["labels"]

                    # Forward pass
                    # model returns: logits, ce_loss, lbl_loss, rz_loss
                    logits, ce_loss, lbl_loss, rz_loss = self.model(
                        input_ids, attention_mask, labels
                    )

                    # Loss calculation
                    total_loss = self.loss_calculator.calculate_pretrain_loss(
                        ce_loss, lbl_loss, rz_loss
                    )

                    # Backward pass
                    self.accelerator.backward(total_loss)

                    # Optimizer step and scheduler step if gradients are synced
                    if self.accelerator.sync_gradients:
                        # Gradient clipping
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.training.gradient_clipping_norm,
                        )

                        self.optimizer.step()
                        self.lr_scheduler.step() # Handles warmup, cosine decay, and linear annealing

                        self.optimizer.zero_grad()
                    
                    # Update step counters
                    self.current_step += 1
                    self.current_tokens_processed += self.tokens_per_global_batch

                    # Logging training metrics
                    if (
                        self.current_step % self.config.training.log_interval == 0
                        and self.accelerator.is_main_process
                    ):
                        metrics = {
                            "train/total_loss": total_loss.item(),
                            "train/ce_loss": ce_loss.item(),
                            "train/lbl_loss": lbl_loss.item(),
                            "train/rz_loss": rz_loss.item(),
                            "train/learning_rate": self.lr_scheduler.get_last_lr()[0],
                            "train/tokens_processed": self.current_tokens_processed,
                        }
                        self.logger.log(metrics, step=self.current_step)
                        self.accelerator.print(
                            f"Step {self.current_step}/{self.total_training_steps} | "
                            f"Loss: {total_loss.item():.4f} | LR: {self.lr_scheduler.get_last_lr()[0]:.7f}"
                        )

                    # Evaluation and Checkpointing
                    if (
                        self.current_step % self.config.training.eval_interval == 0
                        and self.current_step > 0 # Ensure eval isn't run at step 0 if interval is small
                    ):
                        self._run_evaluation(self.current_step)
                        self.accelerator.wait_for_everyone() # Wait for all processes before saving
                        self._save_checkpoint(self.current_step)
                        self.accelerator.wait_for_everyone() # Wait for all processes after saving

            # End of DataLoader iteration. If total steps not reached, loop continues to next epoch.

        # Final evaluation and checkpoint after the main training loop concludes
        if self.accelerator.is_main_process:
            self.accelerator.print("\nPretraining complete. Running final evaluation and saving final checkpoint.")
            self._run_evaluation(self.current_step)
            self.accelerator.wait_for_everyone()
            self._save_checkpoint(self.current_step)
            self.accelerator.wait_for_everyone()

        self.accelerator.end_of_training() # Mark end of training for accelerate

    @torch.no_grad()
    def _run_evaluation(self, step: int) -> Dict[str, Any]:
        """
        Runs evaluation on all specified evaluation datasets.

        Args:
            step: The current training step at which evaluation is performed.

        Returns:
            A dictionary containing evaluation metrics.
        """
        # Only evaluate on the main process to avoid redundant computation
        if self.accelerator.is_main_process:
            self.accelerator.print(f"--- Running evaluation at step {step} ---")
            self.model.eval() # Set model to evaluation mode

            evaluator_instance = Evaluator(
                model=self.accelerator.unwrap_model(self.model), # Unwrap for direct evaluation
                tokenizer=self.tokenizer,
                config=self.config,
                logger=self.logger
            )
            
            # The evaluator.evaluate_pretraining_progress expects a dictionary of dataloaders
            eval_metrics = evaluator_instance.evaluate_pretraining_progress(self.eval_dataloaders)
            
            # Log metrics to wandb with "eval_pretrain" prefix
            prefixed_metrics = {f"eval_pretrain/{k}": v for k, v in eval_metrics.items()}
            self.logger.log(prefixed_metrics, step=step)
            self.accelerator.print(f"Evaluation results at step {step}: {prefixed_metrics}")
            
            self.model.train() # Set model back to training mode
            return eval_metrics
        
        return {} # Return empty dict for non-main processes


    def _save_checkpoint(self, step: int):
        """
        Saves the model, optimizer, and scheduler states for potential resumption,
        and also saves the unwrapped model in Hugging Face format.

        Args:
            step: The current training step for naming the checkpoint.
        """
        checkpoint_path = os.path.join(self.config.training.checkpoint_dir, f"step_{step}")
        
        # Save accelerator state (model, optimizer, scheduler, RNG state)
        # This allows full resumption of training, including FSDP state.
        self.accelerator.save_state(checkpoint_path)

        if self.accelerator.is_main_process:
            self.accelerator.print(f"Saving checkpoint at step {step} to {checkpoint_path}")
            # Save unwrapped model and tokenizer in Hugging Face format for easier inference/loading
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            hf_model_path = os.path.join(checkpoint_path, "hf_model")
            unwrapped_model.save_pretrained(hf_model_path)
            self.tokenizer.save_pretrained(hf_model_path) # Save tokenizer with the model
            
            self.logger.log({"checkpoint_saved_step": step}, step=step)
