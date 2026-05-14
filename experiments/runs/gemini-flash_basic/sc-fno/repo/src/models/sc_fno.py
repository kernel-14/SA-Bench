import torch
import torch.nn as nn
import torch.nn.functional as F

from .fno import FNO  # Assuming fno.py is in the same directory

class SCFNO(nn.Module):
    """
    Conceptual implementation of the Sensitivity-Constrained Fourier Neural Operator (SC-FNO).
    
    This model integrates the base FNO architecture with the capability to compute
    gradients (sensitivities) of the output with respect to input parameters using
    automatic differentiation (AD). The overall architecture is identical to FNO,
    but the training process involves additional sensitivity loss.

    As per the paper, "The SC-FNO architecture processes parameters τ(p) alongside
    spatial coordinates and initial conditions through the lifting layer as function inputs."
    This implies that the FNO itself takes the parameters 'p' as part of its input.
    """
    def __init__(self, modes, width, base_input_dim, num_parameters):
        super(SCFNO, self).__init__()
        # The core FNO model. The input_dim here should account for initial conditions,
        # spatial coordinates, time (base_input_dim), and the parameters 'p' (num_parameters).
        self.total_fno_input_dim = base_input_dim + num_parameters
        self.fno = FNO(modes, width, self.total_fno_input_dim)
        self.base_input_dim = base_input_dim
        self.num_parameters = num_parameters
        
    def forward(self, input_data, parameters):
        """
        Forward pass of the SC-FNO model.
        
        Args:
            input_data (torch.Tensor): Tensor containing initial conditions, spatial coordinates, and time.
                                       Shape: (batch_size, sequence_length, base_input_dim).
                                       This should be differentiable if parameters are part of it.
            parameters (torch.Tensor): Tensor containing the parameters 'p'.
                                       Shape: (batch_size, num_parameters).
                                       This tensor *must* have requires_grad=True for sensitivity calculation.

        Returns:
            torch.Tensor: Predicted solution u(x, t).
            
        The paper states: "The SC-FNO architecture processes parameters τ(p) alongside
        spatial coordinates and initial conditions through the lifting layer as function inputs."
        This means we need to concatenate parameters 'p' with other inputs before passing to FNO.
        The parameters 'p' are repeated/reshaped to match spatial-temporal dimensions.
        """
        batch_size, seq_len, _ = input_data.shape
        
        # Reshape parameters to (batch_size, 1, num_parameters) and expand to (batch_size, seq_len, num_parameters)
        # to concatenate with input_data (initial condition, x, t)
        expanded_parameters = parameters.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Concatenate parameters with input_data
        combined_input = torch.cat([input_data, expanded_parameters], dim=-1)
        
        # Predict solution u
        u_pred = self.fno(combined_input)
        
        return u_pred

if __name__ == '__main__':
    # Example usage (conceptual)
    modes = 8
    width = 20
    base_input_dim = 3 # Example: u0, x, t
    num_parameters = 2 # Example: alpha, beta
    
    sc_fno_model = SCFNO(modes, width, base_input_dim, num_parameters)
    print(f"SC-FNO Model: {sc_fno_model}")

    # Simulate input_data (e.g., u0, x, t)
    batch_size = 2
    sequence_length = 100
    
    dummy_input_data = torch.randn(batch_size, sequence_length, base_input_dim)
    
    # Simulate parameters 'p'. Crucially, requires_grad=True for sensitivity calculation
    dummy_parameters = torch.randn(batch_size, num_parameters, requires_grad=True)

    # Forward pass
    u_pred = sc_fno_model(dummy_input_data, dummy_parameters)
    print(f"Predicted solution u_pred shape: {u_pred.shape}")

    # To demonstrate Jacobian computation: calculate sum of u_pred and then its gradient w.r.t. parameters
    # This simulates d(sum(u_pred))/d(parameters)
    sum_u_pred = u_pred.sum()
    
    # Compute gradients of sum_u_pred with respect to dummy_parameters
    # This gives us a conceptual predicted_jacobian for Ls
    predicted_jacobian_sum = torch.autograd.grad(sum_u_pred, dummy_parameters, create_graph=True)[0]
    print(f"Predicted Jacobian (for sum of u_pred) shape: {predicted_jacobian_sum.shape}") # Should be (batch_size, num_parameters)

    # For the actual Ls calculation, we need the Jacobian of *each output element*
    # with respect to *each parameter*. This is a full Jacobian.
    # torch.autograd.functional.jacobian is suitable for this, or manual iteration/vmap.
    
    # Example of computing full Jacobian (conceptual, might be memory intensive for large outputs)
    # This is a conceptual way to get the Jacobian needed for Ls, as described in the paper.
    # d(u_hat)/d(p) would be of shape (batch_size * sequence_length * output_dim, num_parameters)

    # To get a per-output Jacobian, we can iterate or use a functional approach if available.
    # For simplicity in this example, let's assume we need to compute gradients for each scalar output.
    
    # This is often handled by flattening the output and then computing the Jacobian
    # or by computing gradients of a scalar loss (like L_u) w.r.t. parameters.

    # Let's use a simplified approach for demonstration: compute gradient of a single output element.
    # In a real scenario, you would sum over randomly selected points as described in the paper (Section 2.4)
    # "we randomly select a subset of spatial-temporal points in each epoch ... where n < N and t < T."

    # This section is purely for showing *how* one would get a Jacobian, not the exact mechanism
    # described for the loss computation itself in the paper, which samples points.
    
    # predicted_jacobian_for_Ls will need to be of shape (M, num_parameters) where M is the number of sampled points.
    # If we consider one batch element and one output point (sampled), its gradient w.r.t parameters:
    # single_output_point = u_pred[0, 0, 0]
    # grad_single_output_point = torch.autograd.grad(single_output_point, dummy_parameters, retain_graph=True)[0]
    # print(f"Grad of single output point w.r.t. parameters: {grad_single_output_point.shape}")

    # The actual implementation in a training loop would sum up `L_s` over `M` sampled points.
