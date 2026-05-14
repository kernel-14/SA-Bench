"""
This module implements the SFTTrainer class, which orchestrates the Supervised Fine-Tuning (SFT)
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
from tqdm import tqdm
from typing import Dict, Any, Tuple, Optional

# Local imports from the project structure
from config import Config
from model.olmoe_model import OLMoEModel
from data.adaptation_dataset import AdaptationDataset
from data.data_collator import OLMoEDataCollator
from training.loss_functions import LossCalculator
from training.optimizer_scheduler import OptimizerSchedulerFactory
from utils.logger import Logger
from evaluation.evaluator import Evaluator


class SFTTrainer:
    """
    Manages the Supervised Fine-Tuning (SFT) process for the OLMoE model.

    This class handles the main SFT training loop, data loading,
    optimizer and learning rate scheduler steps, gradient accumulation,
    mixed-precision training, distributed training using Hugging Face Accelerate,
    logging with Weights & Biases, periodic evaluation, and checkpointing.
    """

    def __init__(
        self,
        model: OLMoEModel,
        sft_ds: AdaptationDataset,
        eval_ds: Optional[AdaptationDataset],
        config: Config,
        tokenizer: PreTrainedTokenizer,
        logger: Logger,
        loss_calculator: LossCalculator,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    ):
        """
        Initializes the SFTTrainer with model, data, and training configurations.

        Args:
            model: The OLMoEModel instance to be fine-tuned.
            sft_ds: The AdaptationDataset for SFT training data.
            eval_ds: An optional AdaptationDataset for SFT evaluation data.
            config: The global configuration object.
            tokenizer: The PreTrainedTokenizer instance.
            logger: The Logger instance for experiment tracking.
            loss_calculator: The LossCalculator instance for computing losses.
            optimizer: The AdamW optimizer configured for SFT.
            lr_scheduler: The learning rate scheduler configured for SFT.
        """
        self.model = model
        self.sft_ds = sft_ds
        self.eval_ds = eval_ds
        self.config = config
        self.tokenizer = tokenizer
        self.logger = logger
        self.loss_calculator = loss_calculator
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # Accelerator initialization
        # Use SFT-specific gradient_accumulation_steps and num_gpus for batch consistency checks
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True) # Good for MoE
        self.accelerator = Accelerator(
            mixed_precision=self.config.training.precision,
            gradient_accumulation_steps=self.config.training.sft_gradient_accumulation_steps,
            log_with="wandb",
            project_dir=self.config.training.checkpoint_dir,
            project_config=ProjectConfiguration(
                project_dir=self.config.training.checkpoint_dir,
                logging_dir=os.path.join(self.config.training.checkpoint_dir, "logs")
            ),
            kwargs_handlers=[ddp_kwargs],
        )
        self.accelerator.init_trackers(
            project_name=self.config.training.project_name,
            config=self.config.__dict__,
            init_kwargs={"wandb": {"name": f"{self.config.training.run_name}_SFT"}}
        )

        # Data collator
        self.data_collator = OLMoEDataCollator(self.tokenizer, self.config.data.max_seq_len)

        # DataLoader for SFT training
        self.train_dataloader = DataLoader(
            self.sft_ds,
            batch_size=self.config.training.sft_per_device_batch_size_samples,
            shuffle=True, # Shuffle training data per epoch
            collate_fn=self.data_collator,
            num_workers=os.cpu_count() or 0,
            pin_memory=True,
            drop_last=True,
        )

        # DataLoader for SFT evaluation (if provided)
        self.eval_dataloader: Optional[DataLoader] = None
        if self.eval_ds:
            self.eval_dataloader = DataLoader(
                self.eval_ds,
                batch_size=self.config.training.sft_per_device_batch_size_samples, # Can use a larger batch size for eval if memory permits
                shuffle=False,
                collate_fn=self.data_collator,
                num_workers=os.cpu_count() or 0,
                pin_memory=True,
                drop_last=False,
            )

        # Prepare all components for distributed training and mixed precision
        self.model, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader
        )
        if self.eval_dataloader:
            self.eval_dataloader = self.accelerator.prepare(self.eval_dataloader)

        # Calculate total SFT training steps
        self.num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.accelerator.gradient_accumulation_steps)
        self.total_training_steps = self.config.training.sft_epochs * self.num_update_steps_per_epoch

        self.current_step = 0

        if self.accelerator.is_main_process:
            self.logger.watch_model(self.model)
            self.accelerator.print(f"Total SFT training steps: {self.total_training_steps}")
            self.accelerator.print(f"SFT epochs: {self.config.training.sft_epochs}")

    def train(self):
        """
        Executes the main Supervised Fine-Tuning (SFT) loop.
        """
        self.model.train() # Set model to training mode
        progress_bar = tqdm(
            range(self.total_training_steps),
            disable=not self.accelerator.is_main_process,
            desc="SFT Training",
        )

        for epoch in range(self.config.training.sft_epochs):
            if self.accelerator.is_main_process:
                self.accelerator.print(f"\n--- SFT Epoch {epoch + 1}/{self.config.training.sft_epochs} ---")

            # Ensure sampler for distributed training shuffles data each epoch
            if hasattr(self.train_dataloader, 'sampler') and hasattr(self.train_dataloader.sampler, 'set_epoch'):
                 self.train_dataloader.sampler.set_epoch(epoch)


            for batch_idx, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.model):
                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]
                    labels = batch["labels"]

                    # Forward pass
                    # For SFT, we only care about ce_loss. Auxiliary losses (lbl_loss, rz_loss) are returned but ignored.
                    logits, ce_loss, _, _ = self.model(
                        input_ids, attention_mask, labels
                    )

                    # Loss calculation: SFT only uses Cross-Entropy Loss
                    # Paper states: "do not use load balancing during adaptation" (§4.3)
                    total_loss = self.loss_calculator.calculate_sft_loss(ce_loss)

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
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()
                        self.current_step += 1
                        progress_bar.update(1)

                    # Logging training metrics
                    if (
                        self.current_step % self.config.training.log_interval == 0
                        and self.accelerator.is_main_process
                        and self.current_step > 0 # Avoid logging at step 0 if interval is small
                    ):
                        metrics = {
                            "sft/train_loss": total_loss.item(),
                            "sft/learning_rate": self.lr_scheduler.get_last_lr()[0],
                            "sft/global_step": self.current_step,
                        }
                        self.logger.log(metrics, step=self.current_step)
                        self.accelerator.print(
                            f"SFT Step {self.current_step}/{self.total_training_steps} | "
                            f"Loss: {total_loss.item():.4f} | LR: {self.lr_scheduler.get_last_lr()[0]:.7f}"
                        )

                    # Evaluation and Checkpointing
                    if (
                        self.eval_dataloader
                        and self.current_step % self.config.training.eval_interval == 0
                        and self.current_step > 0 # Avoid evaluation at step 0 if interval is small
                    ):
                        self.accelerator.wait_for_everyone() # Ensure all processes reach this point
                        self._run_evaluation(self.current_step)
                        self.accelerator.wait_for_everyone()
                        self._save_checkpoint(self.current_step, epoch)
                        self.accelerator.wait_for_everyone()
                    
                    if self.current_step >= self.total_training_steps:
                        break # Stop if total steps reached within an epoch loop

            # End of epoch, save checkpoint
            self.accelerator.wait_for_everyone()
            self._save_checkpoint(self.current_step, epoch, is_final_epoch=True)
            self.accelerator.wait_for_everyone()


        # Final actions after training completes
        self.accelerator.print("\nSFT training complete.")
        self.accelerator.end_of_training()
        progress_bar.close()

    @torch.no_grad()
    def _run_evaluation(self, step: int) -> Dict[str, Any]:
        """
        Runs evaluation on the SFT evaluation dataset if available.
        Calculates and logs the average Cross-Entropy loss on the evaluation set.

        Args:
            step: The current training step at which evaluation is performed.

        Returns:
            A dictionary containing evaluation metrics.
        """
        if not self.eval_dataloader:
            return {} # No evaluation dataset provided

        if self.accelerator.is_main_process:
            self.accelerator.print(f"--- Running SFT evaluation at step {step} ---")
            
        self.model.eval() # Set model to evaluation mode
        total_eval_loss = torch.tensor(0.0, device=self.accelerator.device)
        num_eval_batches = 0

        for eval_batch_idx, batch in enumerate(self.eval_dataloader):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            # Forward pass
            _, ce_loss, _, _ = self.model(input_ids, attention_mask, labels)
            
            # Accumulate loss
            total_eval_loss += ce_loss.detach()
            num_eval_batches += 1

        # Gather losses from all processes and average
        if num_eval_batches > 0:
            total_eval_loss = self.accelerator.reduce(total_eval_loss, reduction="sum")
            # Calculate total effective batches across all processes
            total_effective_batches = self.accelerator.reduce(torch.tensor(num_eval_batches, device=self.accelerator.device), reduction="sum")
            average_eval_loss = total_eval_loss / total_effective_batches
        else:
            average_eval_loss = torch.tensor(float('nan'), device=self.accelerator.device)


        metrics = {}
        if self.accelerator.is_main_process:
            metrics = {
                "sft/eval_loss": average_eval_loss.item(),
            }
            self.logger.log(metrics, step=step)
            self.accelerator.print(f"SFT Evaluation results at step {step}: {metrics}")

        self.model.train() # Set model back to training mode
        return metrics

    def _save_checkpoint(self, step: int, epoch: int, is_final_epoch: bool = False):
        """
        Saves the model, optimizer, and scheduler states for potential resumption,
        and also saves the unwrapped model in Hugging Face format.

        Args:
            step: The current training step for naming the checkpoint.
            epoch: The current epoch number.
            is_final_epoch: If True, indicates this is the final checkpoint of an epoch.
        """
        checkpoint_name = f"sft_step_{step}"
        if is_final_epoch:
            checkpoint_name = f"sft_epoch_{epoch+1}"
        
        checkpoint_path = os.path.join(self.config.training.checkpoint_dir, checkpoint_name)
        
        # Save accelerator state (model, optimizer, scheduler, RNG state)
        self.accelerator.save_state(checkpoint_path)

        if self.accelerator.is_main_process:
            self.accelerator.print(f"Saving SFT checkpoint at step {step} to {checkpoint_path}")
            # Save unwrapped model and tokenizer in Hugging Face format
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            hf_model_path = os.path.join(checkpoint_path, "hf_model")
            unwrapped_model.save_pretrained(hf_model_path)
            self.tokenizer.save_pretrained(hf_model_path) # Save tokenizer with the model
            
            self.logger.log({"sft/checkpoint_saved_step": step}, step=step)

