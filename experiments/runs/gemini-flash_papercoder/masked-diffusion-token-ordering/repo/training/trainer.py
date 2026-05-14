import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
import math
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

# Placeholder for Config and NoiseScheduler to avoid circular imports.
# In main.py, the actual Config and NoiseScheduler objects will be imported.
# For this file's standalone integrity and type hinting, placeholders are used.
class _ConfigPlaceholder:
    """
    A placeholder for the Config class to allow type hinting and access to its
    methods without creating a direct import dependency that might lead to
    circular imports in a larger project structure. This placeholder mimics
    the necessary 'get' method.
    """
    def get(self, key: str, default: Any = None) -> Any:
        """Retrieves a configuration value from the underlying config dictionary."""
        raise NotImplementedError("This is a placeholder for the Config object. "
                                  "Its 'get' method should not be called directly from here. "
                                  "Ensure the actual Config object is passed and used.")

# Re-assign for type hinting within this module.
# In the actual project, these would be:
# from config import Config
# from utils.noise_schedules import NoiseScheduler
Config = _ConfigPlaceholder

class _NoiseSchedulerPlaceholder:
    """
    A placeholder for the NoiseScheduler class for type hinting.
    """
    def __init__(self, schedule_type: str, num_steps: int) -> None:
        pass
    def get_alpha(self, t: float) -> float:
        raise NotImplementedError
    def get_alpha_prime(self, t: float) -> float:
        raise NotImplementedError
    def get_mask_prob(self, t: float) -> float:
        raise NotImplementedError

NoiseScheduler = _NoiseSchedulerPlaceholder


# Get logger instance. The logger is set up in utils/logger.py and retrieved here.
logger = logging.getLogger("MDM_Project_Logger")


class Trainer:
    """
    The Trainer class orchestrates the training loop for both Masked Diffusion Models (MDMs)
    and Autoregressive Models (ARMs, including pi-learners). It initializes the optimizer,
    learning rate scheduler, and handles data loading. It implements the loss functions,
    performs forward and backward passes, gradient updates, and integrates with the logger.
    """

    def __init__(
        self,
        config: Config,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        noise_scheduler: NoiseScheduler, # Explicitly passed, not initialized here
        logger_instance: Any # Renamed to avoid clash with imported 'logger'
    ) -> None:
        """
        Initializes the Trainer with the provided configuration, model, data loaders,
        noise scheduler, and logger instance.

        Args:
            config (Config): The global configuration object.
            model (nn.Module): The model to be trained (e.g., TransformerMDM, PiLearnerARM).
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the validation dataset.
            noise_scheduler (NoiseScheduler): An instance of the NoiseScheduler.
            logger_instance (Any): The logger instance (e.g., from utils.logger) for logging.
        """
        self.config: Config = config
        self.model: nn.Module = model
        self.train_loader: DataLoader = train_loader
        self.val_loader: DataLoader = val_loader
        self.noise_scheduler: NoiseScheduler = noise_scheduler
        self.logger: Any = logger_instance

        self.device: str = self.config.get('general.device', 'cpu')
        self.model.to(self.device)
        self.model_type: str = self.config.get('model.model_type')
        self.mask_token_id: int = self.config.get('data.mask_token_id', 0) # Default mask token ID is 0

        logger.info(f"Trainer initialized for model type: {self.model_type} on device: {self.device}")

        # Optimizer Initialization
        optimizer_config = self.config.get('training.optimizer')
        max_lr = self.config.get('training.lr_scheduler.max_lr')
        self.optimizer: optim.Optimizer = optim.AdamW(
            self.model.parameters(),
            lr=max_lr,
            betas=tuple(optimizer_config.get('betas', [0.9, 0.95])),
            weight_decay=optimizer_config.get('weight_decay', 0.1)
        )
        logger.info(f"Optimizer initialized: {self.optimizer}")

        # Learning Rate Scheduler Initialization
        lr_scheduler_config = self.config.get('training.lr_scheduler')
        warmup_steps = lr_scheduler_config.get('warmup_steps', 1000)
        
        # Calculate total_steps for the scheduler
        training_iterations = self.config.get('training.iterations', 0)
        training_epochs = self.config.get('training.epochs', 0)

        if training_iterations > 0:
            self.total_training_steps: int = training_iterations
            self.log_mode = "iterations"
        elif training_epochs > 0:
            self.total_training_steps = training_epochs * len(train_loader)
            self.log_mode = "epochs"
        else:
            raise ValueError("Either 'training.iterations' or 'training.epochs' must be positive in config.")

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_training_steps
        )
        logger.info(f"LR Scheduler initialized with total training steps: {self.total_training_steps}")
        logger.info(f"Warmup steps: {warmup_steps}, Max LR: {max_lr}, Min LR: {lr_scheduler_config.get('min_lr')}")

        self.gradient_accumulation_steps: int = self.config.get('training.gradient_accumulation_steps', 1)
        self.save_interval_epochs: int = self.config.get('training.save_interval_epochs', 50)
        self.eval_interval_epochs: int = self.config.get('training.eval_interval_epochs', 10)
        self.arm_teacher_forcing: bool = self.config.get('training.arm_teacher_forcing', False)


    def _compute_mdm_loss(
        self,
        x0: torch.Tensor,
        logits: torch.Tensor,
        masked_mask_bool_tensor: torch.Tensor,
        t: float
    ) -> torch.Tensor:
        """
        Calculates the Masked Diffusion Model (MDM) training loss.

        Args:
            x0 (torch.Tensor): The original (clean) input sequence batch.
                               Shape: (batch_size, sequence_length).
            logits (torch.Tensor): The model's predicted logits for each token.
                                   Shape: (batch_size, sequence_length, vocab_size).
            masked_mask_bool_tensor (torch.Tensor): A boolean tensor indicating
                                                    which tokens were masked in x_t.
                                                    Shape: (batch_size, sequence_length).
            t (float): The continuous time step at which masking was applied.

        Returns:
            torch.Tensor: The computed scalar loss for the batch.
        """
        # Loss is computed only for tokens that were masked.
        # We need the ground truth tokens (from x0) at these masked positions.
        target_tokens: torch.Tensor = x0[masked_mask_bool_tensor].to(self.device)
        
        # And the model's predictions (logits) for these masked positions.
        masked_logits: torch.Tensor = logits[masked_mask_bool_tensor].to(self.device)

        if target_tokens.numel() == 0: # If no tokens were masked in this batch, loss is 0
            return torch.tensor(0.0, device=self.device)

        # Compute cross-entropy loss for each masked token.
        # F.cross_entropy expects (N, C) and (N) for inputs and targets respectively.
        loss_per_token: torch.Tensor = F.cross_entropy(masked_logits, target_tokens, reduction='none')

        # Apply the weighting factor from the paper: alpha_t_prime / (1 - alpha_t)
        alpha_t: float = self.noise_scheduler.get_alpha(t)
        alpha_t_prime: float = self.noise_scheduler.get_alpha_prime(t)
        
        # Add a small epsilon for numerical stability if (1 - alpha_t) approaches zero
        weight_factor: float = alpha_t_prime / (1.0 - alpha_t + 1e-8)

        # The paper implies an expectation over x_t (which is handled by batching)
        # and a summation over masked tokens. A mean over masked tokens is typical
        # for batch-wise loss computation, then scaled by the weight factor.
        loss: torch.Tensor = loss_per_token.mean() * abs(weight_factor) # Use abs as alpha_t_prime is negative

        return loss

    def _compute_arm_loss(
        self,
        x_pi: torch.Tensor,
        logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the Autoregressive Model (ARM) training loss (negative log-likelihood).
        This method assumes `logits` are causal predictions for `x_pi`.

        Args:
            x_pi (torch.Tensor): The input sequence, possibly permuted, that was fed to the model.
                                 Shape: (batch_size, sequence_length).
            logits (torch.Tensor): The model's predicted causal logits. `logits[b, i, :]`
                                   predicts `x_pi[b, i+1]`. Shape: (batch_size, sequence_length, vocab_size).

        Returns:
            torch.Tensor: The computed scalar loss for the batch.
        """
        batch_size, sequence_length = x_pi.size()
        
        # Target tokens are the next tokens in the sequence (shifted by one position).
        # We predict x_pi[..., 1:] using logits[..., :-1, :].
        target_tokens: torch.Tensor = x_pi[:, 1:].contiguous().to(self.device)
        
        # Logits used for prediction (shifted to align with targets).
        prediction_logits: torch.Tensor = logits[:, :-1, :].contiguous().to(self.device)

        # Reshape for F.cross_entropy: (N, C) for predictions, (N) for targets.
        loss: torch.Tensor = F.cross_entropy(
            prediction_logits.view(-1, prediction_logits.size(-1)),
            target_tokens.view(-1),
            reduction='mean'
        )
        return loss

    def _apply_forward_masking(self, x0: torch.Tensor, t: float) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Applies the forward diffusion (masking) process to a batch of clean sequences `x0`
        at a given continuous time `t`.

        Args:
            x0 (torch.Tensor): The original (clean) input sequence batch.
                               Shape: (batch_size, sequence_length).
            t (float): The continuous time step for masking (sampled from U(0,1)).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, float]:
                - x_t (torch.Tensor): The partially masked sequence.
                                      Shape: (batch_size, sequence_length).
                - masked_mask_bool_tensor (torch.Tensor): A boolean tensor indicating
                                                        which positions were masked.
                                                        Shape: (batch_size, sequence_length).
                - t (float): The actual continuous time step used for masking.
        """
        # Ensure x0 is on the correct device
        x0 = x0.to(self.device)

        # Get the masking probability for the given time step t.
        # This is (1 - alpha_t).
        mask_prob: float = self.noise_scheduler.get_mask_prob(t)

        # Generate a random tensor of the same shape as x0, with values from U(0,1).
        random_values: torch.Tensor = torch.rand_like(x0, dtype=torch.float32, device=self.device)

        # Create a boolean mask: True where a token should be masked.
        masked_mask_bool_tensor: torch.Tensor = (random_values < mask_prob)

        # Create x_t by cloning x0 and applying the mask.
        x_t: torch.Tensor = x0.clone()
        x_t[masked_mask_bool_tensor] = self.mask_token_id

        return x_t, masked_mask_bool_tensor, t

    def train(self) -> None:
        """
        Executes the main training loop, iterating through data, computing losses,
        updating model parameters, and logging progress.
        """
        self.logger.info("Starting training...")
        
        global_step: int = 0
        current_epoch: int = 0
        train_iter = iter(self.train_loader)

        while global_step < self.total_training_steps:
            self.model.train() # Set model to training mode

            # Fetch batch, handling DataLoader exhaustion for iteration-based training
            try:
                batch: Dict[str, Any] = next(train_iter)
            except StopIteration:
                if self.log_mode == "iterations":
                    self.logger.info(f"Train DataLoader exhausted at step {global_step}. Re-initializing iterator.")
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)
                    current_epoch += 1 # Increment epoch count even in iteration mode for logging context
                else: # For epoch-based training, StopIteration marks end of epoch
                    current_epoch += 1
                    if current_epoch >= self.config.get('training.epochs', 0):
                        self.logger.info(f"Finished all {self.config.get('training.epochs')} epochs.")
                        break # Exit loop if all epochs are done
                    self.logger.info(f"Epoch {current_epoch} finished. Re-initializing DataLoader for next epoch.")
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter) # Get first batch of new epoch

            x0_batch: torch.Tensor = batch['input_ids'].to(self.device)
            # Labels might be needed for puzzle tasks, ensure they are on device too
            labels_batch: Optional[torch.Tensor] = batch.get('labels', None)
            if labels_batch is not None:
                labels_batch = labels_batch.to(self.device)

            loss: torch.Tensor

            if self.model_type == "mdm_transformer":
                # Sample continuous time t
                t: float = torch.rand(1).item() # Sample t from U(0,1)

                x_t, masked_mask_bool_tensor, actual_t = self._apply_forward_masking(x0_batch, t)
                logits: torch.Tensor = self.model(x_t)
                loss = self._compute_mdm_loss(x0_batch, logits, masked_mask_bool_tensor, actual_t)

            elif self.model_type == "arm_transformer":
                permutation_batch: Optional[List[List[int]]] = None
                if self.arm_teacher_forcing and 'permutation' in batch:
                    # Assumes `permutation` is a list of lists (batch_size, seq_len) or tensor.
                    # Convert to a list of lists if it's a tensor for type consistency with PiLearnerARM.
                    if isinstance(batch['permutation'], torch.Tensor):
                        permutation_batch = batch['permutation'].tolist()
                    else:
                        permutation_batch = batch['permutation']
                else:
                    # Default to identity permutation for standard left-to-right ARM
                    max_seq_len: int = self.config.get('data.max_sequence_length')
                    permutation_batch = [list(range(max_seq_len))] * x0_batch.size(0)
                
                # Apply permutation to x0_batch before passing to model.
                # PiLearnerARM expects an already permuted sequence.
                # Assuming permutation_batch is (batch_size, seq_len)
                permutation_tensor = torch.tensor(permutation_batch, device=self.device, dtype=torch.long)
                x_pi: torch.Tensor = torch.gather(x0_batch, 1, permutation_tensor)
                
                logits = self.model(x_pi)
                loss = self._compute_arm_loss(x_pi, logits) # Pass x_pi as target for causal loss

            else:
                raise ValueError(f"Unsupported model_type for training: {self.model_type}")

            # Gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            loss.backward()

            if (global_step + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping (optional, but good practice for large models)
                # max_grad_norm = self.config.get('training.max_grad_norm', 1.0)
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            global_step += 1

            # Logging
            if global_step % self.config.get('training.log_interval_steps', 10) == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                self.logger.info(
                    f"Step {global_step}/{self.total_training_steps} "
                    f"(Epoch {current_epoch+1 if self.log_mode=='epochs' else 'N/A'}): "
                    f"Loss: {loss.item():.4f}, LR: {current_lr:.6f}"
                )
                self.logger.wandb.log({
                    "train/loss": loss.item() * self.gradient_accumulation_steps, # Scale back for logging true loss
                    "train/learning_rate": current_lr,
                    "train/global_step": global_step,
                    "train/epoch": current_epoch
                }, step=global_step)

            # Validation and Checkpointing
            if (self.log_mode == "epochs" and (current_epoch + 1) % self.eval_interval_epochs == 0 and (global_step + 1) % self.gradient_accumulation_steps == 0) or \
               (self.log_mode == "iterations" and (global_step % self.config.get('training.eval_interval_steps', 1000) == 0) and (global_step + 1) % self.gradient_accumulation_steps == 0):
                self.logger.info(f"Evaluating model at step {global_step} (Epoch {current_epoch+1 if self.log_mode=='epochs' else 'N/A'})...")
                # Placeholder for evaluation logic. Actual evaluation would be done by Evaluator.
                # For now, we'll just log a placeholder validation loss.
                val_loss = self._run_validation_step()
                self.logger.info(f"Validation Loss: {val_loss:.4f}")
                self.logger.wandb.log({"val/loss": val_loss}, step=global_step)
                
                # Save model checkpoint
                if (self.log_mode == "epochs" and (current_epoch + 1) % self.save_interval_epochs == 0) or \
                   (self.log_mode == "iterations" and (global_step % self.config.get('training.save_interval_steps', 5000) == 0)):
                    self._save_checkpoint(global_step, current_epoch, val_loss)

        self.logger.info("Training complete.")
        self._save_checkpoint(global_step, current_epoch, final=True) # Save final model


    def _run_validation_step(self) -> float:
        """
        Runs a validation step to compute average loss on the validation set.

        Returns:
            float: The average validation loss.
        """
        self.model.eval() # Set model to evaluation mode
        total_val_loss: float = 0.0
        num_batches: int = 0

        with torch.no_grad(): # Disable gradient calculations
            for batch_idx, batch in enumerate(self.val_loader):
                x0_batch: torch.Tensor = batch['input_ids'].to(self.device)
                
                loss: torch.Tensor

                if self.model_type == "mdm_transformer":
                    # For validation, we can average over a few sampled `t` values or pick a representative one.
                    # Or simply apply a full random mask at a fixed t. Let's use a fixed t=0.5 for validation clarity.
                    t: float = 0.5 
                    x_t, masked_mask_bool_tensor, _ = self._apply_forward_masking(x0_batch, t)
                    logits: torch.Tensor = self.model(x_t)
                    loss = self._compute_mdm_loss(x0_batch, logits, masked_mask_bool_tensor, t)

                elif self.model_type == "arm_transformer":
                    permutation_batch: Optional[List[List[int]]] = None
                    # For validation, if teacher forcing is off, use identity. If on, `permutation` might be in batch.
                    if 'permutation' in batch: # If val_loader provides permutations
                         if isinstance(batch['permutation'], torch.Tensor):
                             permutation_batch = batch['permutation'].tolist()
                         else:
                             permutation_batch = batch['permutation']
                    else: # Default to identity if no specific permutation for validation
                        max_seq_len: int = self.config.get('data.max_sequence_length')
                        permutation_batch = [list(range(max_seq_len))] * x0_batch.size(0)

                    permutation_tensor = torch.tensor(permutation_batch, device=self.device, dtype=torch.long)
                    x_pi: torch.Tensor = torch.gather(x0_batch, 1, permutation_tensor)
                    logits = self.model(x_pi)
                    loss = self._compute_arm_loss(x_pi, logits)

                else:
                    raise ValueError(f"Unsupported model_type for validation: {self.model_type}")

                total_val_loss += loss.item()
                num_batches += 1

        if num_batches == 0:
            return 0.0 # Avoid division by zero if val_loader is empty
        return total_val_loss / num_batches

    def _save_checkpoint(self, global_step: int, epoch: int, val_loss: float, final: bool = False) -> None:
        """
        Saves the current model and optimizer state as a checkpoint.

        Args:
            global_step (int): The current global training step.
            epoch (int): The current training epoch.
            val_loss (float): The validation loss at the time of checkpointing.
            final (bool): If True, indicates this is the final model save.
        """
        output_dir: Path = Path(self.config.get('general.output_dir', 'outputs'))
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_name: str = f"{self.config.get('general.experiment_name')}_step_{global_step:07d}.pt"
        if final:
            checkpoint_name = f"{self.config.get('general.experiment_name')}_final.pt"

        checkpoint_path: Path = output_dir / checkpoint_name
        
        torch.save({
            'global_step': global_step,
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config.config_dict # Save full config for reproducibility
        }, checkpoint_path)
        self.logger.info(f"Model checkpoint saved to: {checkpoint_path}")

