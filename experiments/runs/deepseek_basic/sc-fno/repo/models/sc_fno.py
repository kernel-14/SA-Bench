"""
Sensitivity-Constrained FNO (SC-FNO) wrapper.

Extends FNO with:
- Sensitivity loss term L_s for Jacobian supervision
- Optional PINN equation loss L_eq
- Support for computing Jacobians via automatic differentiation

Training configurations:
1. FNO:          L = L_u
2. FNO-PINN:     L = L_u + L_eq
3. SC-FNO:       L = c1*L_u + c2*L_s
4. SC-FNO-PINN:  L = c1*L_u + c2*L_s + c3*L_eq
"""

import torch
import torch.nn as nn
from .fno import FNO


class SC_FNO(nn.Module):
    """
    Sensitivity-Constrained Fourier Neural Operator.
    
    Wraps an FNO model with sensitivity loss computation.
    Both FNO and SC-FNO share identical architectures; they differ
    only in their loss configurations.
    """

    def __init__(self, fno_model: FNO, loss_weights=None):
        """
        Args:
            fno_model: An FNO instance (or other neural operator)
            loss_weights: dict with keys 'c1' (L_u weight), 'c2' (L_s weight), 'c3' (L_eq weight)
        """
        super().__init__()
        self.model = fno_model
        if loss_weights is None:
            loss_weights = {'c1': 1.0, 'c2': 1.0, 'c3': 0.1}
        self.c1 = loss_weights.get('c1', 1.0)
        self.c2 = loss_weights.get('c2', 1.0)
        self.c3 = loss_weights.get('c3', 0.1)
        self.mse_loss = nn.MSELoss()

    def forward(self, x):
        """Forward pass through the underlying FNO model."""
        return self.model(x)

    def compute_jacobian(self, x, params_idx=None):
        """
        Compute Jacobian of model output with respect to input parameters.
        
        Uses automatic differentiation (torch.autograd) to compute ∂u/∂p.
        
        Args:
            x: Input tensor (batch, *grid_dims, input_dim)
            params_idx: Indices of parameter dimensions in the input tensor.
                        If None, computes Jacobian w.r.t. all input dimensions.
        
        Returns:
            Jacobian tensor of shape (batch, *grid_dims, output_dim, n_params)
        """
        x.requires_grad_(True)
        u_pred = self.model(x)
        
        if params_idx is not None:
            # Compute Jacobian only for specified parameter dimensions
            jac = []
            for idx in params_idx:
                grad_outputs = torch.ones_like(u_pred)
                grad = torch.autograd.grad(
                    outputs=u_pred, inputs=x,
                    grad_outputs=grad_outputs,
                    retain_graph=True, create_graph=True
                )[0]
                jac.append(grad[..., idx:idx+1])
            jacobian = torch.cat(jac, dim=-1)
        else:
            # Compute full Jacobian
            batch_shape = u_pred.shape
            n_elements = u_pred.numel()
            grad_outputs = torch.ones_like(u_pred)
            grad = torch.autograd.grad(
                outputs=u_pred, inputs=x,
                grad_outputs=grad_outputs,
                create_graph=True
            )[0]
            jacobian = grad
        
        return jacobian

    def compute_losses(self, batch, use_sensitivity=True, use_pinn=False):
        """
        Compute all relevant losses for a batch.
        
        Args:
            batch: dict with keys:
                - 'input': model input tensor
                - 'u_true': true solution values
                - 'jac_true': true Jacobian values (optional, for SC-FNO)
                - 'pde_residual_fn': function to compute PDE residual (optional, for PINN)
            use_sensitivity: if True, compute L_s
            use_pinn: if True, compute L_eq
        
        Returns:
            dict with keys: 'total', 'L_u', 'L_s', 'L_eq'
        """
        x = batch['input']
        u_true = batch['u_true']
        
        # Forward pass
        u_pred = self.model(x)
        
        # Data loss L_u
        L_u = self.mse_loss(u_pred, u_true)
        
        losses = {'L_u': L_u}
        total = self.c1 * L_u
        
        # Sensitivity loss L_s
        if use_sensitivity and 'jac_true' in batch:
            jac_pred = self.compute_jacobian(x, batch.get('params_idx'))
            L_s = self.mse_loss(jac_pred, batch['jac_true'])
            losses['L_s'] = L_s
            total = total + self.c2 * L_s
        
        # Equation loss L_eq (PINN-style)
        if use_pinn and 'pde_residual_fn' in batch:
            residual = batch['pde_residual_fn'](u_pred)
            L_eq = torch.mean(residual ** 2)
            losses['L_eq'] = L_eq
            total = total + self.c3 * L_eq
        
        losses['total'] = total
        return losses, u_pred

    def count_params(self):
        return self.model.count_params()
