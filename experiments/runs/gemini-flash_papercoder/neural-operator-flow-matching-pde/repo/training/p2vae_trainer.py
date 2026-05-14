## training/p2vae_trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import logging
import os
import math
from typing import Dict, Any, Tuple

# Local imports
from config import Config
from utils import logging_utils
from models.p2vae import P2VAEModel # Assuming P2VAEModel is defined in models/p2vae.py
from utils import metrics # Assuming metrics.py exists and contains l2_relative_error and vrmse

class P2VAETrainer:
    """
    Trainer class for the Pretrained Physics Variational Autoencoder (P2VAE).
    Manages the training loop, loss calculation, optimization, learning rate scheduling,
    validation, and checkpointing.
    """

    def __init__(self, model: P2VAEModel, train_loader: DataLoader, val_loader: DataLoader, config: Config, device: str):
        """
        Initializes the P2VAETrainer.

        Args:
            model (P2VAEModel): The P2VAE model instance.
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the validation dataset.
            config (Config): The configuration object for the experiment.
            device (str): The compute device ('cuda' or 'cpu').
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        self.global_step = 0
        self.best_val_loss = float('inf')

        # Setup logging
        self.logger, self.tb_writer = logging_utils.setup_logging(self.config, "p2vae_train")

        # Optimizer and Scheduler
        self.optimizer, self.scheduler = self._setup_optimizer_scheduler()

        # Mixed precision training setup
        # Check if device is cuda and global dtype is float16 for AMP
        self.use_amp = (self.device == 'cuda' and self.config.get('global.dtype', 'float32') == 'float16')
        self.scaler = GradScaler() if self.use_amp else None
        
        self.logger.info(f"P2VAETrainer initialized. Using device: {self.device}, AMP enabled: {self.use_amp}")

    def _setup_optimizer_scheduler(self) -> Tuple[optim.Optimizer, Any]:
        """
        Configures the optimizer and learning rate scheduler based on config settings.

        Returns:
            Tuple[optim.Optimizer, Any]: The configured optimizer and scheduler.
        """
        # Optimizer setup
        optim_config = self.config.get('p2vae_training.optimizer', {})
        optimizer_name = optim_config.get('name', 'AdamW')
        base_lr = optim_config.get('lr', 1e-4)
        betas = tuple(optim_config.get('betas', [0.9, 0.995]))
        weight_decay = optim_config.get('weight_decay', 1e-4)

        if optimizer_name == 'AdamW':
            optimizer = optim.AdamW(self.model.parameters(), lr=base_lr, betas=betas, weight_decay=weight_decay)
        else:
            self.logger.warning(f"Unsupported optimizer: {optimizer_name}. Using AdamW as default.")
            optimizer = optim.AdamW(self.model.parameters(), lr=base_lr, betas=betas, weight_decay=weight_decay)

        # Scheduler setup
        scheduler_config = self.config.get('p2vae_training.scheduler', {})
        warmup_steps_ratio = scheduler_config.get('warmup_steps_ratio', 0.1)
        
        num_training_steps = self.config.get('p2vae_training.num_training_steps', 100000)
        warmup_steps = int(num_training_steps * warmup_steps_ratio)

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            
            # Cosine annealing after warmup
            # Ensure progress is between 0 and 1 for cosine
            progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        # LambdaLR applies the lr_lambda function to multiply the initial LR
        scheduler = LambdaLR(optimizer, lr_lambda)
        
        self.logger.info(f"Optimizer: {optimizer_name}, Base LR: {base_lr}, Weight Decay: {weight_decay}")
        self.logger.info(f"Scheduler: CosineAnnealingLR with {warmup_steps} warmup steps out of {num_training_steps} total steps.")

        return optimizer, scheduler

    def _compute_loss(self, x_true: torch.Tensor, x_reco: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the L_VAE loss: 0.5 * ||x - x_hat||^2 + beta * KL(q(y|x) || p(y)).

        Args:
            x_true (torch.Tensor): The original input physical field.
            x_reco (torch.Tensor): The reconstructed physical field.
            mu (torch.Tensor): Mean of the latent distribution.
            log_var (torch.Tensor): Log-variance of the latent distribution.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - total_loss (torch.Tensor): The combined VAE loss.
                - recon_loss (torch.Tensor): The reconstruction (MSE) loss component.
                - kl_loss (torch.Tensor): The KL divergence loss component.
        """
        # Reconstruction Loss: 1/2 * ||x - x_hat||^2
        # F.mse_loss(reduction='mean') computes (1/N) * sum((x-x_hat)^2) where N is total elements.
        # The paper's loss formulation implies MSE where each element (or image) is a sample,
        # so F.mse_mse(reduction='mean') is suitable for averaging over batch and spatial dims.
        # The 0.5 factor is applied to this term as per paper equation.
        recon_loss = F.mse_loss(x_reco, x_true, reduction='mean')

        # KL Divergence Loss: -0.5 * (1 + log_var - mu^2 - exp(log_var))
        # We want to average this across the batch and latent dimensions.
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Apply KL weight (beta)
        kl_weight = self.config.get('p2vae_model.kl_weight', 1e-3)
        weighted_kl_loss = kl_weight * kl_loss

        total_loss = 0.5 * recon_loss + weighted_kl_loss
        return total_loss, recon_loss, kl_loss # Return components for logging

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Executes one training step for a given batch.

        Args:
            batch (Dict[str, torch.Tensor]): A dictionary containing input data,
                                            e.g., {'x_0': input_tensor}.

        Returns:
            float: The computed total loss for the current step.
        """
        x_true = batch['x_0'].to(self.device)
        
        # Determine if it's time to zero gradients based on accumulation steps
        grad_accum_steps = self.config.get('p2vae_training.gradient_accumulation_steps', 1)
        
        # If accumulating gradients, zero_grad only at the start of an accumulation cycle
        if self.global_step % grad_accum_steps == 0:
            self.optimizer.zero_grad()

        with autocast(enabled=self.use_amp, dtype=torch.float16 if self.use_amp else torch.float32):
            x_reco, mu, log_var, _ = self.model(x_true)
            total_loss, recon_loss, kl_loss = self._compute_loss(x_true, x_reco, mu, log_var)
            
            # Scale the loss for gradient accumulation (backward expects sum of losses over accumulation)
            scaled_total_loss = total_loss / grad_accum_steps

        if self.use_amp:
            self.scaler.scale(scaled_total_loss).backward()
        else:
            scaled_total_loss.backward()

        # Perform optimizer step and scheduler step if accumulation is complete
        if (self.global_step + 1) % grad_accum_steps == 0: # +1 because global_step is 0-indexed
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.scheduler.step()
        
        return total_loss.item() # Return unscaled loss for logging purposes

    @torch.no_grad()
    def _validate_epoch(self) -> Dict[str, float]:
        """
        Evaluates the model on the validation set.

        Returns:
            Dict[str, float]: A dictionary containing validation metrics.
        """
        self.model.eval() # Set model to evaluation mode
        
        val_losses_total = []
        val_losses_recon = []
        val_losses_kl = []
        val_l2res = []
        val_vrmses = []

        self.logger.info("Starting validation epoch...")
        for i, batch in enumerate(self.val_loader):
            x_true = batch['x_0'].to(self.device)

            with autocast(enabled=self.use_amp, dtype=torch.float16 if self.use_amp else torch.float32):
                x_reco, mu, log_var, _ = self.model(x_true)
                total_loss, recon_loss, kl_loss = self._compute_loss(x_true, x_reco, mu, log_var)
            
            val_losses_total.append(total_loss.item())
            val_losses_recon.append(recon_loss.item())
            val_losses_kl.append(kl_loss.item())
            
            # Calculate L2RE and VRMSE (metrics should handle tensors on device)
            # Ensure float32 for metric calculation to avoid potential numerical issues with small values
            l2re = metrics.l2_relative_error(x_reco.float(), x_true.float())
            vrmse = metrics.vrmse(x_reco.float(), x_true.float())
            val_l2res.append(l2re.item())
            val_vrmses.append(vrmse.item())

            if i == 0: # Log first batch reconstruction for visualization
                # For logging, typically move to CPU and ensure 3 channels for image visualization
                logging_utils.log_image(self.tb_writer, 'Validation/x_true_sample', x_true[0].cpu(), self.global_step)
                logging_utils.log_image(self.tb_writer, 'Validation/x_reco_sample', x_reco[0].cpu(), self.global_step)
        
        avg_total_loss = sum(val_losses_total) / len(val_losses_total)
        avg_recon_loss = sum(val_losses_recon) / len(val_losses_recon)
        avg_kl_loss = sum(val_losses_kl) / len(val_losses_kl)
        avg_l2re = sum(val_l2res) / len(val_l2res)
        avg_vrmse = sum(val_vrmses) / len(val_vrmses)

        self.logger.info(f"Validation results (step {self.global_step}): "
                         f"Loss: {avg_total_loss:.4f}, Recon Loss: {avg_recon_loss:.4f}, "
                         f"KL Loss: {avg_kl_loss:.4f}, L2RE: {avg_l2re:.4f}, VRMSE: {avg_vrmse:.4f}")

        logging_utils.log_scalar(self.tb_writer, 'Validation/Total_Loss', avg_total_loss, self.global_step)
        logging_utils.log_scalar(self.tb_writer, 'Validation/Reconstruction_Loss', avg_recon_loss, self.global_step)
        logging_utils.log_scalar(self.tb_writer, 'Validation/KL_Loss', avg_kl_loss, self.global_step)
        logging_utils.log_scalar(self.tb_writer, 'Validation/L2RE', avg_l2re, self.global_step)
        logging_utils.log_scalar(self.tb_writer, 'Validation/VRMSE', avg_vrmse, self.global_step)

        if avg_total_loss < self.best_val_loss:
            self.best_val_loss = avg_total_loss
            self._save_checkpoint(is_best=True) # Save checkpoint as best if validation loss improves
            self.logger.info(f"New best validation loss: {self.best_val_loss:.4f}. Saved best model checkpoint.")

        self.model.train() # Set model back to training mode
        return {
            'val_total_loss': avg_total_loss,
            'val_recon_loss': avg_recon_loss,
            'val_kl_loss': avg_kl_loss,
            'val_l2re': avg_l2re,
            'val_vrmse': avg_vrmse
        }

    def _save_checkpoint(self, is_best: bool = False) -> None:
        """
        Saves the current model, optimizer, and scheduler states.

        Args:
            is_best (bool, optional): If True, saves as 'best_model.pth'. Defaults to False.
        """
        checkpoint_dir = self.config.get('logging.checkpoint_dir', './checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_filename = f'p2vae_checkpoint_step_{self.global_step}.pth'
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
        state = {
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        if self.use_amp:
            state['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(state, checkpoint_path)
        self.logger.info(f"Saved P2VAE checkpoint at step {self.global_step} to {checkpoint_path}")

        if is_best:
            best_model_path = os.path.join(checkpoint_dir, 'p2vae_best_model.pth')
            torch.save(state, best_model_path)
            self.logger.info(f"Saved best P2VAE model to {best_model_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Loads model, optimizer, and scheduler states from a checkpoint.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"Checkpoint file not found at {checkpoint_path}. Starting training from scratch.")
            return

        self.logger.info(f"Loading P2VAE checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.logger.info(f"P2VAE checkpoint loaded successfully. Resuming from step {self.global_step}.")

    def train(self) -> P2VAEModel:
        """
        Orchestrates the entire P2VAE training process.

        Returns:
            P2VAEModel: The trained P2VAE model instance.
        """
        self.logger.info("Starting P2VAE training...")
        self.model.train() # Ensure model is in training mode

        num_training_steps = self.config.get('p2vae_training.num_training_steps', 100000)
        log_interval_steps = self.config.get('p2vae_training.log_interval_steps', 100)
        validate_interval_steps = self.config.get('p2vae_training.validate_interval_steps', 5000)
        save_interval_steps = self.config.get('p2vae_training.save_interval_steps', 10000)
        grad_accum_steps = self.config.get('p2vae_training.gradient_accumulation_steps', 1)

        # Create an iterator for the train_loader to loop indefinitely until num_training_steps is met
        train_iter = iter(self.train_loader)
        
        # Loop until the specified number of global training steps is reached
        while self.global_step < num_training_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                self.logger.info("End of training dataset epoch. Re-initializing train_loader iterator.")
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            current_loss = self._train_step(batch)

            # Logging after each effective step (after optimizer.step)
            # The global_step is incremented *after* an optimizer step if grad_accum_steps == 1
            # or after a full accumulation cycle.
            # We want to log for the step that just completed its optimization.
            current_log_step = self.global_step + 1 # Use +1 for 1-indexed display

            if current_log_step % log_interval_steps == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(f"Step {current_log_step}/{num_training_steps} | "
                                 f"LR: {current_lr:.6f} | Train Loss: {current_loss:.4f}")
                logging_utils.log_scalar(self.tb_writer, 'Training/Total_Loss', current_loss, self.global_step)
                logging_utils.log_scalar(self.tb_writer, 'Training/Learning_Rate', current_lr, self.global_step)

            # Validation
            if current_log_step % validate_interval_steps == 0:
                self._validate_epoch()

            # Checkpointing
            if current_log_step % save_interval_steps == 0:
                self._save_checkpoint(is_best=False)
            
            # Increment global_step counter.
            # This is crucial: self.global_step represents the number of *optimization steps* completed.
            # It's incremented here for the next iteration to reflect the current state.
            # The check `if (self.global_step + 1) % grad_accum_steps == 0` in _train_step
            # means self.global_step is incremented *before* that check is true for the first time.
            # This logic needs to ensure global_step counts optimizer steps.
            # If grad_accum_steps is 1, global_step increments every batch.
            # If grad_accum_steps > 1, global_step increments every N batches.
            # Let's align global_step to count "effective" training steps.
            if (self.global_step + 1) % grad_accum_steps == 0:
                self.global_step += 1 # Only increment once per optimizer step
            elif grad_accum_steps == 1:
                self.global_step += 1 # Always increment if no accumulation

        self.logger.info("P2VAE training finished.")
        logging_utils.close_writers(self.tb_writer)
        return self.model

