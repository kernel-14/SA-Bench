## trainers/fno_trainer.py
import torch
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any, Tuple

# Import project-specific modules
from config import Config
from losses import Losses
from trainers.base_trainer import BaseTrainer
from models.fno import FNO  # Specific FNO model

# Third-party library for progress bar
from tqdm import tqdm


class FNOTrainer(BaseTrainer):
    """
    Concrete trainer for FNO models.
    This trainer computes only the data loss (L_u) for model optimization,
    as per the FNO's original configuration.
    """

    def __init(
        self,
        model: FNO,  # Expecting an FNO model specifically
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        config: Config,
        losses: Losses,
        device: str
    ) -> None:
        """
        Initializes the FNOTrainer.

        Args:
            model (FNO): The FNO model instance to be trained.
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the validation dataset.
            optimizer (torch.optim.Optimizer): PyTorch optimizer instance.
            scheduler (Optional[Any]): Optional learning rate scheduler.
            config (Config): Configuration object.
            losses (Losses): Instance of the Losses class.
            device (str): The computational device ('cuda' or 'cpu').
        """
        # Ensure the model is indeed an FNO instance
        if not isinstance(model, FNO):
            raise TypeError("FNOTrainer expects a model of type FNO.")
        
        super().__init__(model, train_loader, val_loader, optimizer, scheduler, config, losses, device)
        print("FNOTrainer initialized.")

    def _compute_batch_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the total loss for a given batch for FNO models.
        For FNO, this exclusively involves calculating the data loss (L_u).

        Args:
            batch (Dict[str, torch.Tensor]): A dictionary containing batch data.
                                             Expected keys: 'fno_input_encoder_data', 'fno_target_u'.

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: A tuple containing:
                - total_loss (torch.Tensor): The scalar total weighted loss for the batch.
                - loss_details (Dict[str, float]): A dictionary of individual loss values ('u_loss').
        """
        # Extract FNO input features and target true solution from the batch
        input_features = batch['fno_input_encoder_data'].to(self.device)
        u_true = batch['fno_target_u'].to(self.device)

        # Perform a forward pass through the FNO model
        u_pred = self.model(input_features)

        # Compute the data loss (L_u) using the Losses manager
        u_loss = self.losses.compute_u_loss(u_pred, u_true)

        # Combine losses. For FNO, only u_loss is considered,
        # and the `combine_losses` method will apply its configured weight (typically 1.0).
        total_loss = self.losses.combine_losses(u_loss=u_loss)

        # Prepare loss details for logging and tracking
        loss_details = {
            'u_loss': u_loss.item(),
            'total_loss': total_loss.item(),  # Include total_loss for easier tracking
        }

        return total_loss, loss_details

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Implements the training logic for a single epoch for FNO models.

        Args:
            epoch (int): The current epoch number.

        Returns:
            Dict[str, float]: A dictionary containing averaged training metrics for the epoch.
        """
        self.model.train()  # Set the model to training mode
        
        # Initialize running loss sums and batch counter
        running_u_loss = 0.0
        running_total_loss = 0.0
        num_batches = 0

        # Iterate over the training data using a progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for batch in pbar:
            self.optimizer.zero_grad()  # Clear previous gradients

            # Compute the batch loss
            total_loss, loss_details = self._compute_batch_loss(batch)
            
            total_loss.backward()  # Perform backpropagation
            self.optimizer.step()   # Update model parameters

            # Accumulate losses
            running_u_loss += loss_details['u_loss']
            running_total_loss += loss_details['total_loss']
            num_batches += 1

            # Update progress bar postfix with current batch losses
            pbar.set_postfix({
                'u_loss': f"{loss_details['u_loss']:.4f}",
                'total_loss': f"{loss_details['total_loss']:.4f}"
            })

        # Calculate average losses for the epoch
        avg_u_loss = running_u_loss / num_batches if num_batches > 0 else 0.0
        avg_total_loss = running_total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'u_loss': avg_u_loss,
            'total_loss': avg_total_loss,
        }

    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Implements the validation logic for a single epoch for FNO models.

        Args:
            epoch (int): The current epoch number.

        Returns:
            Dict[str, float]: A dictionary containing averaged validation metrics for the epoch.
        """
        self.model.eval()  # Set the model to evaluation mode
        
        # Initialize running loss sums and batch counter
        running_u_loss = 0.0
        running_total_loss = 0.0
        num_batches = 0

        # Disable gradient calculations during validation
        with torch.no_grad():
            # Iterate over the validation data using a progress bar
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
            for batch in pbar:
                # Compute the batch loss
                total_loss, loss_details = self._compute_batch_loss(batch)

                # Accumulate losses
                running_u_loss += loss_details['u_loss']
                running_total_loss += loss_details['total_loss']
                num_batches += 1

                # Update progress bar postfix with current batch losses
                pbar.set_postfix({
                    'u_loss': f"{loss_details['u_loss']:.4f}",
                    'total_loss': f"{loss_details['total_loss']:.4f}"
                })

        # Calculate average losses for the epoch
        avg_u_loss = running_u_loss / num_batches if num_batches > 0 else 0.0
        avg_total_loss = running_total_loss / num_batches if num_batches > 0 else 0.0

        return {
            'u_loss': avg_u_loss,
            'total_loss': avg_total_loss,
        }

