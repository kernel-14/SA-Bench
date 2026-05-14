## losses.py
import torch
from typing import Optional, Callable, Dict, Any

# Assuming Config is available from config.py
from config import Config
# Assuming PDESolver is available from data_generation.pde_solver
from data_generation.pde_solver import PDESolver
# Assuming compute_relative_l2 is available from utils.py
from utils import compute_relative_l2


class Losses:
    """
    Defines the loss components for FNO, SC-FNO, and SC-FNO-PINN models.
    It encapsulates methods for calculating data loss (L_u), sensitivity loss (L_s),
    and equation loss (L_Eq), along with their weighted combination.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the Losses instance with weighting factors from the configuration.

        Args:
            config (Config): The configuration object containing loss weighting factors.
        """
        self.u_loss_weight: float = config.get("training.loss_weights.u_loss", 1.0)
        self.s_loss_weight: float = config.get("training.loss_weights.s_loss", 0.1)
        self.eq_loss_weight: float = config.get("training.loss_weights.eq_loss", 0.01)

    def compute_u_loss(self, u_pred: torch.Tensor, u_true: torch.Tensor) -> torch.Tensor:
        """
        Calculates the data loss (L_u) as the Relative L2 error between
        predicted and true solutions.

        Args:
            u_pred (torch.Tensor): The predicted solution values from the model.
            u_true (torch.Tensor): The ground truth solution values.

        Returns:
            torch.Tensor: A scalar tensor representing the Relative L2 error.
        """
        # Ensure input tensors are float type for calculations
        u_pred_float = u_pred.to(torch.float32)
        u_true_float = u_true.to(torch.float32)
        
        # compute_relative_l2 returns a float, convert to tensor for consistency with other losses
        return torch.tensor(compute_relative_l2(u_pred_float, u_true_float), dtype=torch.float32, device=u_pred.device)

    def compute_s_loss(self, du_pred_dp: torch.Tensor, du_true_dp: torch.Tensor) -> torch.Tensor:
        """
        Calculates the sensitivity loss (L_s) as the Relative L2 error between
        predicted and true Jacobians (sensitivities).

        Args:
            du_pred_dp (torch.Tensor): The predicted Jacobian (sensitivity) from the model.
            du_true_dp (torch.Tensor): The ground truth Jacobian (sensitivity).

        Returns:
            torch.Tensor: A scalar tensor representing the Relative L2 error for Jacobians.
        """
        # Ensure input tensors are float type for calculations
        du_pred_dp_float = du_pred_dp.to(torch.float32)
        du_true_dp_float = du_true_dp.to(torch.float32)

        # compute_relative_l2 returns a float, convert to tensor
        return torch.tensor(compute_relative_l2(du_pred_dp_float, du_true_dp_float), dtype=torch.float32, device=du_pred_dp.device)

    def compute_eq_loss(
        self,
        u_pred: torch.Tensor,
        p_params: Dict[str, torch.Tensor],
        x_coords: torch.Tensor,
        t_coords: torch.Tensor,
        equation_fn: Callable[..., Any],
        pde_solver: PDESolver,
        equation_id: str
    ) -> torch.Tensor:
        """
        Calculates the equation loss (L_Eq), which quantifies how well the model's
        predictions satisfy the underlying governing differential equation (PDE residual).
        This method specifically computes the L_PDE part of L_Eq, delegating to PDESolver.

        Args:
            u_pred (torch.Tensor): The model's predicted solution at collocation points.
                                   Must have requires_grad=True for differentiation.
            p_params (Dict[str, torch.Tensor]): Dictionary of PDE parameters (batch of scalar or zoned).
            x_coords (torch.Tensor): Spatial coordinates of collocation points.
                                     Must have requires_grad=True.
            t_coords (torch.Tensor): Temporal coordinates of collocation points.
                                     Must have requires_grad=True.
            equation_fn (Callable): The Python function representing the PDE's right-hand side.
            pde_solver (PDESolver): An instance of the PDESolver to compute the PDE residual.
            equation_id (str): The identifier for the specific PDE (e.g., "PDE1", "PDE2").

        Returns:
            torch.Tensor: A scalar tensor representing the mean squared PDE residual.
        """
        # PDESolver.compute_pde_residual is designed to compute (residual**2).mean()
        # so this method directly returns its output.
        # Ensure u_pred, x_coords, t_coords have requires_grad=True before passing to pde_solver for PINN AD.
        # This is handled by the SCFNO_PINNTrainer calling this function.
        return pde_solver.compute_pde_residual(
            u_pred=u_pred,
            x_coords=x_coords,
            t_coords=t_coords,
            p_params=p_params,
            equation_fn=equation_fn,
            equation_id=equation_id
        )

    def combine_losses(
        self,
        u_loss: torch.Tensor,
        s_loss: Optional[torch.Tensor] = None,
        eq_loss: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Combines the individual loss components into a single total loss using
        their respective weighting factors.

        Args:
            u_loss (torch.Tensor): The calculated data loss (L_u).
            s_loss (Optional[torch.Tensor]): The calculated sensitivity loss (L_s), if applicable.
            eq_loss (Optional[torch.Tensor]): The calculated equation loss (L_Eq), if applicable.

        Returns:
            torch.Tensor: A scalar tensor representing the total weighted loss.
        """
        total_loss = self.u_loss_weight * u_loss

        if s_loss is not None:
            total_loss = total_loss + self.s_loss_weight * s_loss
        
        if eq_loss is not None:
            total_loss = total_loss + self.eq_loss_weight * eq_loss
            
        return total_loss

