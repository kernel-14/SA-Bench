"""
trainer.py
Handles the lifecycle of model training for LoRA-SB. Implements methods for gradient computation during initialization
and fine-tuning of the R matrix. Adheres strictly to the design and configuration outlined in the project specification.
"""
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import logging
from typing import Tuple, Dict
from utils import compute_truncated_svd, set_device, normalize_gradients, aggregate_gradients


class Trainer:
    """
    Trainer: Manages fine-tuning of the LoRA_SB_Model. Includes gradient computation for initialization
    and optimization of the R matrix during training.
    """
    def __init__(self, model: torch.nn.Module, dataloaders: Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader], config: Dict) -> None:
        """
        Initialize the Trainer with model, dataloaders, and training configuration.

        Args:
            model (torch.nn.Module): Preconfigured LoRA_SB_Model instance with frozen B, A, and trainable R.
            dataloaders (Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]): 
                Tuple containing DataLoader for training and validation data.
            config (Dict): Configuration dictionary loaded from config.yaml.
        """
        self.model = model.to(set_device(config['hardware']['use_gpu']))
        self.train_loader, self.validation_loader = dataloaders
        self.config = config

        # Extract hyperparameters from config
        self.learning_rate = config['training']['learning_rate']
        self.batch_size = config['training']['batch_size']
        self.grad_accumulation_steps = config['training']['grad_accumulation_steps']
        self.epochs = config['training']['epochs']
        self.lr_scheduler_type = config['training'].get('scheduler', 'cosine')
        self.warmup_ratio = config['training']['warmup_ratio']
        self.device = set_device(config['hardware']['use_gpu'])

        # Optimizer for R matrix only
        self.optimizer = AdamW([param for name, param in model.named_parameters() if param.requires_grad], 
                               lr=self.learning_rate)

        # Learning rate scheduler
        num_training_steps = len(self.train_loader) * self.epochs
        self.scheduler = self._initialize_scheduler(self.optimizer, num_training_steps)

        # Logging setup
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [Trainer] %(message)s")
        self.logger = logging.getLogger('Trainer')

    def _initialize_scheduler(self, optimizer: torch.optim.Optimizer, total_steps: int) -> LambdaLR:
        """
        Set up the learning rate scheduler based on configuration.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer instance for scheduling.
            total_steps (int): Total number of training steps.

        Returns:
            LambdaLR: Configured learning rate scheduler.
        """
        if self.lr_scheduler_type == "cosine":
            scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 0.5 * (1 + torch.cos(torch.pi * step / total_steps)))
        elif self.lr_scheduler_type == "linear":
            scheduler = LambdaLR(optimizer, lr_lambda=lambda step: max(1 - step / total_steps, self.warmup_ratio))
        else:
            raise ValueError(f"Unsupported scheduler type: {self.lr_scheduler_type}")
        return scheduler

    def compute_gradients(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute averaged gradients over a subset of data and perform truncated SVD.

        Args:
            num_samples (int): Number of samples for gradient computation (subset size).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: B_init, A_init, R_init matrices from SVD.
        """
        self.logger.info(f"Computing averaged gradients using {num_samples} samples for initialization.")
        
        # Subset DataLoader for initialization
        small_loader = torch.utils.data.DataLoader(self.train_loader.dataset, batch_size=1, shuffle=True)
        subset_loader = [next(iter(small_loader)) for _ in range(num_samples)]  # Obtain num_samples batches
        
        # Aggregate gradients across subset
        averaged_gradients = aggregate_gradients(self.model, subset_loader, self.device)
        averaged_gradients = normalize_gradients(averaged_gradients, norm_type="frobenius")  # Normalize
        
        # Perform truncated SVD
        B_init, R_init, A_init = compute_truncated_svd(averaged_gradients, self.model.rank)

        self.logger.info("Initialization gradients computed and truncated SVD applied.")
        return B_init, A_init, R_init

    def train(self) -> None:
        """
        Fine-tune the R matrix of the model using the training dataset.

        Handles memory-efficient processing and logs progress at regular intervals.
        """
        self.logger.info("Starting fine-tuning.")

        for epoch in range(self.epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.epochs}")
            self.model.train()
            running_loss = 0.0
            gradient_accumulation_counter = 0

            for batch_idx, batch in enumerate(self.train_loader):
                # Move batch to the device
                batch = {key: value.to(self.device) for key, value in batch.items()}
                self.optimizer.zero_grad()

                # Forward pass and calculate loss
                outputs = self.model(batch)
                loss = outputs.loss  # Assuming model output includes precomputed loss
                loss.backward()  # Backward pass

                # Accumulate gradients
                gradient_accumulation_counter += 1
                running_loss += loss.item()

                if gradient_accumulation_counter % self.grad_accumulation_steps == 0:
                    # Perform optimizer step and update scheduler
                    self.optimizer.step()
                    self.scheduler.step()
                    gradient_accumulation_counter = 0

                if batch_idx % 100 == 0:  # Log every 100 batches
                    self.logger.info(f"Batch {batch_idx}/{len(self.train_loader)} - Loss: {loss.item():.4f}")

            avg_loss = running_loss / len(self.train_loader)
            self.logger.info(f"Epoch {epoch + 1} complete. Average loss: {avg_loss:.4f}")
            
            # Validation step (optional)
            if self.validation_loader:
                self.validate()

        self.logger.info("Training complete.")

    def validate(self) -> None:
        """
        Evaluate the model on the validation dataset (if provided).
        Logs validation loss and optionally additional metrics.
        """
        self.logger.info("Running validation.")
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.validation_loader:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                outputs = self.model(batch)
                loss = outputs.loss
                total_loss += loss.item()

        avg_loss = total_loss / len(self.validation_loader)
        self.logger.info(f"Validation complete. Average loss: {avg_loss:.4f}")

