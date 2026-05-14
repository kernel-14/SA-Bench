## models/sc_fno_pinn.py
import torch
import torch.nn as nn
from typing import Callable, Tuple, Any

# Assuming SCFNOBase is defined in models.sc_fno_base
from models.sc_fno_base import SCFNOBase 
# Assuming PDESolver is defined in data_generation.pde_solver
from data_generation.pde_solver import PDESolver


class SCFNO_PINN(nn.Module):
    """
    Implements the Sensitivity-Constrained FNO with Physics-Informed Neural Network (PINN) regularization.
    This class wraps an SCFNOBase instance and holds references to the PDESolver and the
    equation function, which are used by the trainer/loss module to compute the PDE residual loss.
    """
    def __init__(self, sc_fno_base: SCFNOBase, pde_solver: PDESolver, equation_fn: Callable[[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any], torch.Tensor]) -> None:
        """
        Initializes the SC-FNO-PINN model.

        Args:
            sc_fno_base (SCFNOBase): An instance of the SCFNOBase model. This is the core FNO
                                     backbone with sensitivity computation capabilities.
            pde_solver (PDESolver): An instance of the PDESolver. This is used externally
                                    to compute the PDE residual for the PINN loss component.
            equation_fn (Callable): The Python function representing the PDE's right-hand side or
                                    residual form. This function is specific to the PDE being solved
                                    and is passed to the PDESolver for residual calculation.
        """
        super().__init__()
        self.sc_fno_base: SCFNOBase = sc_fno_base
        self.pde_solver: PDESolver = pde_solver
        self.equation_fn: Callable[[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any], torch.Tensor] = equation_fn

    def forward(self, input_features: torch.Tensor, params_input: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the underlying SC-FNO model.

        Args:
            input_features (torch.Tensor): The input tensor for the FNO backbone, typically
                                           containing initial conditions, spatial/temporal coordinates.
                                           Shape: (batch_size, *grid_dims, input_dim).
            params_input (torch.Tensor): The tensor of physical parameters for which the FNO
                                         is making predictions. This tensor is also used for
                                         sensitivity computation.
                                         Shape: (batch_size, num_parameters).

        Returns:
            torch.Tensor: The predicted solution from the FNO model.
                          Shape: (batch_size, *grid_dims_target, output_dim).
        """
        # Delegate the forward pass directly to the wrapped SCFNOBase instance.
        u_pred: torch.Tensor = self.sc_fno_base.forward(input_features, params_input)
        return u_pred

    def compute_jacobian(self, u_pred: torch.Tensor, params_input: torch.Tensor) -> torch.Tensor:
        """
        Computes the Jacobian of the predicted solution with respect to the input parameters.

        Args:
            u_pred (torch.Tensor): The predicted solution from the FNO model, usually
                                   the output of `self.forward`.
            params_input (torch.Tensor): The input physical parameters for which the
                                         Jacobian is to be computed. This tensor must
                                         have `requires_grad=True` (or be part of a computation
                                         graph where gradients can be traced back to it).

        Returns:
            torch.Tensor: The computed Jacobian (sensitivities) ∂u_pred/∂params_input.
                          Shape: (batch_size, *grid_dims_target, output_dim, num_parameters).
        """
        # Delegate the Jacobian computation to the wrapped SCFNOBase instance.
        du_pred_dp: torch.Tensor = self.sc_fno_base.compute_jacobian(u_pred, params_input)
        return du_pred_dp

