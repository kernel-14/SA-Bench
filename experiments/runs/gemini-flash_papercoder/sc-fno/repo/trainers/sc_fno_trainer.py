## trainers/sc_fno_trainer.py
import torch
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any, Tuple

# Import project-specific modules
from config import Config
from losses import Losses
from trainers.base_trainer import BaseTrainer
from models.sc_fno_base import SCFNOBase  # Specific SC-FNOBase model

# Third-party library for progress bar
from tqdm import tqdm


class SCFNOTrainer(BaseTrainer):
    """
    Concrete trainer for Sensitivity-Constrained FNO (SC-FNO) models.
    This trainer computes both data loss (L_u) and sensitivity loss (L_s)
    for model optimization, as described in the paper's Algorithm 2.
    """

    def __init__(
        self,
        model: SCFNOBase,  # Expecting an SCFNOBase model specifically
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        config: Config,
        losses: Losses,
        device: str
    ) -> None:
        """
        Initializes the SCFNOTrainer.

        Args:
            model (SCFNOBase): The SC-FNOBase model instance to be trained.
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the validation dataset.
            optimizer (torch.optim.Optimizer): PyTorch optimizer instance.
            scheduler (Optional[Any]): Optional learning rate scheduler.
            config (Config): Configuration object.
            losses (Losses): Instance of the Losses class.
            device (str): The computational device ('cuda' or 'cpu').
        """
        # Ensure the model is indeed an SCFNOBase instance
        if not isinstance(model, SCFNOBase):
            raise TypeError("SCFNOTrainer expects a model of type SCFNOBase.")
        
        super().__init__(model, train_loader, val_loader, optimizer, scheduler, config, losses, device)
        
        # Get sensitivity sampling percentage from config
        self.sensitivity_sampling_percentage: float = self.config.get(
            "dataset_generation.sensitivity_sampling_percentage", 0.1
        )
        print("SCFNOTrainer initialized.")

    def _compute_batch_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the total loss for a given batch for SC-FNO models.
        This includes data loss (L_u) and sensitivity loss (L_s).

        Args:
            batch (Dict[str, torch.Tensor]): A dictionary containing batch data.
                                             Expected keys: 'fno_input_encoder_data',
                                             'fno_params_for_ad', 'fno_target_u', 'fno_target_du_dp'.

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: A tuple containing:
                - total_loss (torch.Tensor): The scalar total weighted loss for the batch.
                - loss_details (Dict[str, float]): A dictionary of individual loss values ('u_loss', 's_loss').
        """
        # --- Data Preparation ---
        # input_features: (batch_size, *grid_dims, input_dim_features)
        input_features = batch['fno_input_encoder_data'].to(self.device)
        
        # p_params: (batch_size, param_dim)
        # It's crucial for these parameters to require gradients for L_s calculation
        p_params = batch['fno_params_for_ad'].to(self.device)
        p_params.requires_grad_(True) # Enable gradient tracking for parameters

        # u_true: (batch_size, *grid_dims_target, output_dim)
        u_true = batch['fno_target_u'].to(self.device)
        
        # du_true_dp: (batch_size, *grid_dims_target, output_dim, param_dim)
        du_true_dp = batch['fno_target_du_dp'].to(self.device)

        # --- Forward Pass ---
        # u_pred: (batch_size, *grid_dims_target, output_dim)
        u_pred = self.model(input_features, p_params)

        # --- Compute Data Loss (L_u) ---
        u_loss = self.losses.compute_u_loss(u_pred, u_true)

        # --- Compute Sensitivity Loss (L_s) ---
        # predicted Jacobian: du_pred_dp_full (batch_size, *grid_dims_target, output_dim, param_dim)
        # This calls the SCFNOBase's compute_jacobian which uses AD.
        du_pred_dp_full = self.model.compute_jacobian(u_pred, p_params)

        # Sensitivity Sampling for efficiency
        # Reshape u_pred to (batch_size, total_output_elements, output_dim)
        # E.g., if u_pred is (B, T, S_x, D_u), reshape to (B, T*S_x, D_u)
        batch_size = u_pred.shape[0]
        output_dim = u_pred.shape[-1]
        
        # Flatten the spatial-temporal dimensions
        num_output_elements_per_sample = u_pred.numel() // (batch_size * output_dim) # T*S_x (or T*S_x*S_y)
        
        # Reshape to (batch_size, total_output_elements_per_sample, output_dim, param_dim)
        du_pred_dp_flat = du_pred_dp_full.view(batch_size, num_output_elements_per_sample, output_dim, -1)
        du_true_dp_flat = du_true_dp.view(batch_size, num_output_elements_per_sample, output_dim, -1)
        
        # Sample a subset of points for the sensitivity loss
        num_total_spatial_temporal_points = num_output_elements_per_sample * output_dim
        num_sampled_points = int(num_total_spatial_temporal_points * self.sensitivity_sampling_percentage)
        
        # Ensure at least one point is sampled if possible
        if num_sampled_points == 0 and num_total_spatial_temporal_points > 0:
            num_sampled_points = 1
        elif num_sampled_points > num_total_spatial_temporal_points:
            num_sampled_points = num_total_spatial_temporal_points # Cap at max available
        
        if num_sampled_points > 0:
            # Flatten to (batch_size, total_output_elements_flat) and sample indices
            # where total_output_elements_flat = num_output_elements_per_sample * output_dim
            # The sampling is across the combined output elements for each batch sample.
            # Example: pick `num_sampled_points` random indices for (T*S_x*D_u) elements for each batch.
            
            # Create sampling indices for each batch item
            all_indices = torch.randint(0, num_total_spatial_temporal_points, (batch_size, num_sampled_points), device=self.device)

            # Reshape du_pred_dp_flat and du_true_dp_flat for vectorized sampling
            # Original: (batch_size, num_output_elements_per_sample, output_dim, param_dim)
            # Flatten output and output_dim: (batch_size, num_output_elements_per_sample * output_dim, param_dim)
            du_pred_dp_for_sampling = du_pred_dp_flat.view(batch_size, -1, du_pred_dp_flat.shape[-1])
            du_true_dp_for_sampling = du_true_dp_flat.view(batch_size, -1, du_true_dp_flat.shape[-1])

            # Use advanced indexing to select sampled points for each batch item
            # all_indices needs to be (batch_size, num_sampled_points, 1) for broadcasting
            all_indices_expanded = all_indices.unsqueeze(-1)
            
            du_pred_dp_sampled = torch.gather(du_pred_dp_for_sampling, 1, all_indices_expanded.expand(-1, -1, du_pred_dp_for_sampling.shape[-1]))
            du_true_dp_sampled = torch.gather(du_true_dp_for_sampling, 1, all_indices_expanded.expand(-1, -1, du_true_dp_for_sampling.shape[-1]))
            
            # Reshape back for L2 loss if needed, or directly pass flattened to compute_s_loss
            # L_s computation expects tensors of same shape for comparison.
            s_loss = self.losses.compute_s_loss(du_pred_dp_sampled, du_true_dp_sampled)
        else:
            # If no points are sampled (e.g., if num_total_spatial_temporal_points is 0),
            # the sensitivity loss is 0.
            s_loss = torch.tensor(0.0, device=self.device)


        # --- Combine Losses ---
        total_loss = self.losses.combine_losses(u_loss=u_loss, s_loss=s_loss)

        # --- Prepare Loss Details for Logging ---
        loss_details = {
            'u_loss': u_loss.item(),
            's_loss': s_loss.item(),
            'total_loss': total_loss.item(),
        }

        return total_loss, loss_details

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Implements the training logic for a single epoch for SC-FNO models.

        Args:
            epoch (int): The current epoch number.

        Returns:
            Dict[str, float]: A dictionary containing averaged training metrics for the epoch.
        """
        self.model.train()  # Set the model to training mode
        
        # Initialize running loss sums and batch counter
        running_u_loss = 0.0
        running_s_loss = 0.0
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
            running_s_loss += loss_details['s_loss']
            running_total_loss += loss_details['total_loss']
            num_batches += 1

            # Update progress bar postfix with current batch losses
            pbar.set_postfix({
                'u_loss': f"{loss_details['u_loss']:.4f}",
                's_loss': f"{loss_details['s_loss']:.4f}",
                'total_loss': f"{loss_details['total_loss']:.4f}"
            })

        # Calculate average losses for the epoch
        avg_u_loss = running_u_loss / num_batches if num_batches > 0 else 0.0
        avg_s_loss = running_s_loss / num_batches if num_batches > 0 else 0.0
        avg_total_loss = running_total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'u_loss': avg_u_loss,
            's_loss': avg_s_loss,
            'total_loss': avg_total_loss,
        }

    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Implements the validation logic for a single epoch for SC-FNO models.

        Args:
            epoch (int): The current epoch number.

        Returns:
            Dict[str, float]: A dictionary containing averaged validation metrics for the epoch.
        """
        self.model.eval()  # Set the model to evaluation mode
        
        # Initialize running loss sums and batch counter
        running_u_loss = 0.0
        running_s_loss = 0.0
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
                running_s_loss += loss_details['s_loss']
                running_total_loss += loss_details['total_loss']
                num_batches += 1

                # Update progress bar postfix with current batch losses
                pbar.set_postfix({
                    'u_loss': f"{loss_details['u_loss']:.4f}",
                    's_loss': f"{loss_details['s_loss']:.4f}",
                    'total_loss': f"{loss_details['total_loss']:.4f}"
                })

        # Calculate average losses for the epoch
        avg_u_loss = running_u_loss / num_batches if num_batches > 0 else 0.0
        avg_s_loss = running_s_loss / num_batches if num_batches > 0 else 0.0
        avg_total_loss = running_total_loss / num_batches if num_batches > 0 else 0.0

        return {
            'u_loss': avg_u_loss,
            's_loss': avg_s_loss,
            'total_loss': avg_total_loss,
        }

