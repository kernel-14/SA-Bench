"""
This module implements the DPOTrainer class, which orchestrates the Direct Preference Optimization (DPO)
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


class DPOTrainer:
    """
    Manages the Direct Preference Optimization (DPO) process for the OLMoE model.

    This class handles the main DPO training loop, data loading,
    optimizer and learning rate scheduler steps, gradient accumulation,
    mixed-precision training, distributed training using Hugging Face Accelerate,
    logging with Weights & Biases, periodic evaluation, and checkpointing.
    """

    def __init__(
        self,
        model: OLMoEModel,
        dpo_ds: AdaptationDataset,
        eval_ds: Optional[AdaptationDataset],
        config: Config,
        tokenizer: PreTrainedTokenizer,
        logger: Logger,
        loss_calculator: LossCalculator,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    ):
        """
        Initializes the DPOTrainer with model, data, and training configurations.

        Args:
            model: The OLMoEModel instance to be fine-tuned.
            dpo_ds: The AdaptationDataset for DPO training data.
            eval_ds: An optional AdaptationDataset for DPO evaluation data.
            config: The global configuration object.
            tokenizer: The PreTrainedTokenizer instance.
            logger: The Logger instance for experiment tracking.
            loss_calculator: The LossCalculator instance for computing losses.
            optimizer: The AdamW optimizer configured for DPO.
            lr_scheduler: The learning rate scheduler configured for DPO.
        """
        self.model = model
        self.dpo_ds = dpo_ds
        self.eval_ds = eval_ds
        self.config = config
        self.tokenizer = tokenizer
        self.logger = logger
        self.loss_calculator = loss_calculator
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # Accelerator initialization
        # Use DPO-specific gradient_accumulation_steps for batch consistency checks
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True) # Good for MoE
        self.accelerator = Accelerator(
            mixed_precision=self.config.training.precision,
            gradient_accumulation_steps=self.config.training.dpo_gradient_accumulation_steps,
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
            init_kwargs={"wandb": {"name": f"{self.config.training.run_name}_DPO"}}
        )

        # Data collator
        # Note: OLMoEDataCollator for DPO should be initialized with is_dpo=True during its internal call logic
        # Here we initialize it once and its __call__ method handles DPO specific format
        self.data_collator = OLMoEDataCollator(self.tokenizer, self.config.data.max_seq_len)

        # DataLoader for DPO training
        self.train_dataloader = DataLoader(
            self.dpo_ds,
            batch_size=self.config.training.dpo_per_device_batch_size_samples,
            shuffle=True, # Shuffle training data per epoch
            collate_fn=self.data_collator,
            num_workers=os.cpu_count() or 0,
            pin_memory=True,
            drop_last=True,
        )

        # DataLoader for DPO evaluation (if provided)
        self.eval_dataloader: Optional[DataLoader] = None
        if self.eval_ds:
            # For evaluation, AdaptationDataset must be initialized with is_dpo=True
            if not self.eval_ds.is_dpo:
                raise ValueError("Evaluation dataset for DPO must be initialized with is_dpo=True.")
            self.eval_dataloader = DataLoader(
                self.eval_ds,
                batch_size=self.config.training.dpo_per_device_batch_size_samples, # Can use a larger batch size for eval if memory permits
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

        # Calculate total DPO training steps
        self.num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.accelerator.gradient_accumulation_steps)
        self.total_training_steps = self.config.training.dpo_epochs * self.num_update_steps_per_epoch

        self.current_step = 0

        if self.accelerator.is_main_process:
            self.logger.watch_model(self.model)
            self.accelerator.print(f"Total DPO training steps: {self.total_training_steps}")
            self.accelerator.print(f"DPO epochs: {self.config.training.dpo_epochs}")


    def _compute_dpo_loss(
        self,
        model_output_chosen: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        model_output_rejected: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Helper method to compute the DPO loss using the LossCalculator.

        Args:
            model_output_chosen: The output tuple from the model's forward pass for the chosen response.
                                 Expected: (logits, ce_loss, lbl, rz_loss).
            model_output_rejected: The output tuple from the model's forward pass for the rejected response.
                                   Expected: (logits, ce_loss, lbl, rz_loss).

        Returns:
            A scalar `torch.Tensor` representing the averaged DPO loss over the batch.
        """
        # Extract Cross-Entropy loss components. These represent -log(P(y|x))
        ce_loss_chosen = model_output_chosen[1]
        ce_loss_rejected = model_output_rejected[1]
        
        # Pass to LossCalculator for DPO loss computation
        dpo_loss = self.loss_calculator.calculate_dpo_loss(
            ce_loss_chosen=ce_loss_chosen,
            ce_loss_rejected=ce_loss_rejected,
            dpo_beta=self.config.training.dpo_beta,
        )
        return dpo_loss


    def train(self):
        """
        Executes the main Direct Preference Optimization (DPO) loop.
        """
        self.model.train() # Set model to training mode
        progress_bar = tqdm(
            range(self.total_training_steps),
            disable=not self.accelerator.is_main_process,
            desc="DPO Training",
        )

        for epoch in range(self.config.training.dpo_epochs):
            if self.accelerator.is_main_process:
                self.accelerator.print(f"\n--- DPO Epoch {epoch + 1}/{self.config.training.dpo_epochs} ---")

            # Ensure sampler for distributed training shuffles data each epoch
            if hasattr(self.train_dataloader, 'sampler') and hasattr(self.train_dataloader.sampler, 'set_epoch'):
                 self.train_dataloader.sampler.set_epoch(epoch)


            for batch_idx, batch in enumerate(self.train_dataloader):
                if self.current_step >= self.total_training_steps:
                    break # Stop if total steps reached within an epoch

                with self.accelerator.accumulate(self.model):
                    # Forward pass for Chosen response
                    model_output_chosen = self.model(
                        input_ids=batch["chosen_input_ids"],
                        attention_mask=batch["chosen_attention_mask"],
                        labels=batch["chosen_labels"], # Labels needed for CE loss calculation
                    )

                    # Forward pass for Rejected response
                    model_output_rejected = self.model(
                        input_ids=batch["rejected_input_ids"],
                        attention_mask=batch["rejected_attention_mask"],
                        labels=batch["rejected_labels"], # Labels needed for CE loss calculation
                    )

                    # Compute DPO loss
                    total_loss = self._compute_dpo_loss(model_output_chosen, model_output_rejected)

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
                            "dpo/train_loss": total_loss.item(),
                            "dpo/learning_rate": self.lr_scheduler.get_last_lr()[0],
                            "dpo/global_step": self.current_step,
                        }
                        self.logger.log(metrics, step=self.current_step)
                        self.accelerator.print(
                            f"DPO Step {self.current_step}/{self.total_training_steps} | "
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
        self.accelerator.print("\nDPO training complete. Saving final DPO model.")
        self.accelerator.end_of_training()
        progress_bar.close()

    @torch.no_grad()
    def _run_evaluation(self, step: int) -> Dict[str, Any]:
        """
        Runs evaluation on the DPO evaluation dataset if available.
        Calculates and logs the average DPO loss on the evaluation set.

        Args:
            step: The current training step at which evaluation is performed.

        Returns:
            A dictionary containing evaluation metrics.
        """
        if not self.eval_dataloader:
            return {} # No evaluation dataset provided

        if self.accelerator.is_main_process:
            self.accelerator.print(f"--- Running DPO evaluation at step {step} ---")
            
        self.model.eval() # Set model to evaluation mode
        total_eval_loss = torch.tensor(0.0, device=self.accelerator.device)
        num_eval_batches = 0

        for eval_batch_idx, batch in enumerate(self.eval_dataloader):
            # Forward pass for Chosen response
            model_output_chosen = self.model(
                input_ids=batch["chosen_input_ids"],
                attention_mask=batch["chosen_attention_mask"],
                labels=batch["chosen_labels"],
            )

            # Forward pass for Rejected response
            model_output_rejected = self.model(
                input_ids=batch["rejected_input_ids"],
                attention_mask=batch["rejected_attention_mask"],
                labels=batch["rejected_labels"],
            )
            
            # Compute DPO loss
            loss = self._compute_dpo_loss(model_output_chosen, model_output_rejected)
            
            total_eval_loss += loss.detach()
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
                "dpo/eval_loss": average_eval_loss.item(),
            }
            # The evaluator instance can be used to run full adaptation benchmarks if desired,
            # but for simplicity, the current design implies evaluating DPO loss here.
            # If the design were to call evaluator.evaluate_adaptation(dataloader) directly,
            # it would need to be passed a dictionary of dataloaders for specific tasks.
            # For now, we calculate eval loss directly.
            self.logger.log(metrics, step=step)
            self.accelerator.print(f"DPO Evaluation results at step {step}: {metrics}")

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
        checkpoint_name = f"dpo_step_{step}"
        if is_final_epoch:
            checkpoint_name = f"dpo_epoch_{epoch+1}"
        
        checkpoint_path = os.path.join(self.config.training.checkpoint_dir, checkpoint_name)
        
        # Save accelerator state (model, optimizer, scheduler, RNG state)
        self.accelerator.save_state(checkpoint_path)

        if self.accelerator.is_main_process:
            self.accelerator.print(f"Saving DPO checkpoint at step {step} to {checkpoint_path}")
            # Save unwrapped model and tokenizer in Hugging Face format
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            hf_model_path = os.path.join(checkpoint_path, "hf_model")
            unwrapped_model.save_pretrained(hf_model_path)
            self.tokenizer.save_pretrained(hf_model_path) # Save tokenizer with the model
            
            self.logger.log({"dpo/checkpoint_saved_step": step}, step=step)

