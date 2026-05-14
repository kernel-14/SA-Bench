## losses.py

import torch
from torch import Tensor
import torch.nn.functional as F
import yaml
from typing import Optional


class LossFunctions:
    """
    LossFunctions class encapsulates all required losses for SC-FNO and related frameworks.
    Includes functions to compute:
    - Primary loss (`L_u`)
    - Sensitivity loss (`L_s`)
    - Total loss (`L_total`)
    - Optional equation loss (`L_Eq`) for SC-FNO-PINN.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initializes loss functions and loads configuration from YAML file.

        Args:
        - config_path (str): Path to the YAML configuration file.
        """
        # Load YAML configuration file
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Extract loss weights from the configuration
        loss_weights = config.get("loss_weights", {})
        self.primary_loss_weight = loss_weights.get("primary_loss", 1.0)
        self.sensitivity_loss_weight = loss_weights.get("sensitivity_loss", 1.0)
        self.equation_loss_weight = loss_weights.get("equation_loss", 0.0)

        # Numerical stability constant (if needed for edge cases)
        self.epsilon = 1e-8

    def compute_primary_loss(self, pred: Tensor, target: Tensor) -> float:
        """
        Computes the primary loss L_u between model predictions and true solution paths.

        Args:
        - pred (Tensor): Predicted solution paths `u_hat`.
        - target (Tensor): Ground-truth solution paths `u_true`.

        Returns:
        - float: Scalar value of the primary loss (Mean Squared Error).
        """
        primary_loss = F.mse_loss(pred, target)
        return max(primary_loss.item(), self.epsilon)  # Ensure numerical stability

    def compute_sensitivity_loss(self, predicted_jacobian: Tensor, true_jacobian: Tensor) -> float:
        """
        Computes the sensitivity loss L_s between predicted and true Jacobians.

        Args:
        - predicted_jacobian (Tensor): Jacobian of predicted solution paths ∂u_hat/∂p.
        - true_jacobian (Tensor): Jacobian of true solution paths ∂u_true/∂p.

        Returns:
        - float: Scalar value of the sensitivity loss (Mean Squared Error for Jacobians).
        """
        # Compute mean squared error between predicted and true Jacobians
        sensitivity_loss = F.mse_loss(predicted_jacobian, true_jacobian)
        return max(sensitivity_loss.item(), self.epsilon)  # Ensure numerical stability

    def compute_total_loss(
        self, 
        primary_loss: float, 
        sensitivity_loss: float, 
        equation_loss: Optional[float] = None
    ) -> float:
        """
        Computes the total loss L_total as a weighted sum of component losses.

        Args:
        - primary_loss (float): Loss term L_u comparing predicted and true solution paths.
        - sensitivity_loss (float): Loss term L_s comparing predicted and true Jacobians.
        - equation_loss (Optional[float]): Optional term L_Eq enforcing physics constraints (default: None).

        Returns:
        - float: Scalar value of the total loss L_total.
        """
        equation_loss = equation_loss or 0.0

        # Combine individual loss components using weights from config
        total_loss = (
            self.primary_loss_weight * primary_loss +
            self.sensitivity_loss_weight * sensitivity_loss +
            self.equation_loss_weight * equation_loss
        )
        return max(total_loss, self.epsilon)  # Ensure numerical stability

# Example usage within a training pipeline:
# loss_functions = LossFunctions(config_path="config/config.yaml")
# primary_loss = loss_functions.compute_primary_loss(predicted_u, true_u)
# sensitivity_loss = loss_functions.compute_sensitivity_loss(predicted_jacobian, true_jacobian)
# total_loss = loss_functions.compute_total_loss(primary_loss, sensitivity_loss)
