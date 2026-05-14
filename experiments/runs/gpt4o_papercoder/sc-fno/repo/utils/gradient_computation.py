## utils/gradient_computation.py

# Import required libraries
import torch
from torch import Tensor
from typing import Callable

class GradientComputation:
    """
    GradientComputation class handles methods for calculating gradients of solution paths.
    Provides functionality for both Automatic Differentiation (AD) and Finite Difference (FD).
    """

    @staticmethod
    def compute_automatic_differentiation(func: Callable, inputs: Tensor) -> Tensor:
        """
        Computes gradients using PyTorch's automatic differentiation.

        Args:
        - func (Callable): Function that computes solutions (ODE/PDE solver or ML model).
        - inputs (Tensor): Input tensor containing parameters and inputs to the function.

        Returns:
        - Tensor: Jacobians (∂func/∂inputs) computed via autodiff.
        """
        # Ensure inputs require gradients
        inputs = inputs.clone().detach().requires_grad_(True)

        # Forward pass to compute outputs
        try:
            outputs = func(inputs)
            if not outputs.requires_grad:
                raise RuntimeError("Outputs are not differentiable. Check if 'func' is compatible with PyTorch's autodiff.")
        except Exception as e:
            raise RuntimeError(f"Error during forward pass for automatic differentiation: {str(e)}")

        # Compute gradients
        jacobians = []
        for i in range(outputs.shape[1]):  # Loop over output dimensions
            grad = torch.autograd.grad(outputs[:, i].sum(), inputs, retain_graph=True, allow_unused=True)[0]
            if grad is None:
                raise RuntimeError("Gradient computation failed. Ensure all input tensors can track gradients.")
            jacobians.append(grad.unsqueeze(-1))  # Append gradients for each dimension

        # Combine gradients into a single tensor
        return torch.cat(jacobians, dim=-1)  # Shape: [batch_size x input_dimensions x output_dimensions]

    @staticmethod
    def compute_finite_difference(func: Callable, inputs: Tensor, epsilon: float = 1e-5) -> Tensor:
        """
        Computes gradients using finite differences. Useful for non-differentiable models or validation.

        Args:
        - func (Callable): Function that computes solutions (ODE/PDE solver or ML model).
        - inputs (Tensor): Input tensor containing parameters and inputs to the function.
        - epsilon (float): Step size for finite difference calculations (default: 1e-5).

        Returns:
        - Tensor: Jacobians (∂func/∂inputs) via finite difference estimation.
        """
        # Validate epsilon
        if epsilon <= 0:
            raise ValueError("Epsilon must be a positive number for finite difference calculations.")

        # Prepare input shape and initialize gradient storage
        inputs = inputs.clone().detach()
        num_params = inputs.size(1)  # Dimension of parameters space
        outputs = func(inputs)  # Evaluate function to get base output shape
        if not isinstance(outputs, Tensor):
            raise ValueError("Function 'func' must output a PyTorch Tensor.")

        # Allocate memory for gradients
        jacobians = torch.zeros(inputs.shape[0], num_params, outputs.shape[1], dtype=torch.float32)

        # Compute finite differences for each parameter
        for param_idx in range(num_params):
            perturb = torch.zeros_like(inputs)
            perturb[:, param_idx] += epsilon  # Apply positive perturbation

            # Forward Pass: Evaluate function with perturbed inputs
            outputs_plus = func(inputs + perturb)
            outputs_minus = func(inputs - perturb)

            # Central difference calculation
            gradient = (outputs_plus - outputs_minus) / (2 * epsilon)
            jacobians[:, param_idx, :] = gradient

        return jacobians

# Example usage:
# Assume callable_func and input_tensor are defined externally
# gradient_ad = GradientComputation.compute_automatic_differentiation(callable_func, input_tensor)
# gradient_fd = GradientComputation.compute_finite_difference(callable_func, input_tensor, epsilon=1e-5)
