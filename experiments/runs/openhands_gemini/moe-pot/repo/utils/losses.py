
import torch
import torch.nn as nn
import torch.nn.functional as F

class L2RelativeError(nn.Module):
    """
    Computes the L2 Relative Error between prediction and ground truth.
    L2RE = ||x_pred - x_gt||_2 / ||x_gt||_2
    """
    def __init__(self):
        super().__init__()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prediction (torch.Tensor): The predicted tensor.
            target (torch.Tensor): The ground truth tensor.

        Returns:
            torch.Tensor: The L2 Relative Error.
        """
        # Ensure that the dimensions are handled correctly for the L2 norm
        # The norm is typically computed over all dimensions or specific ones.
        # Assuming the norm over the last few dimensions (e.g., spatial, channels)
        # and then averaging over batch.
        
        # Calculate L2 norm of the difference
        diff_norm = torch.norm(prediction - target, p=2, dim=(-1, -2, -3)) # Norm over H, W, C
        
        # Calculate L2 norm of the target
        target_norm = torch.norm(target, p=2, dim=(-1, -2, -3)) # Norm over H, W, C
        
        # Avoid division by zero
        l2_relative_error = diff_norm / (target_norm + 1e-6)
        
        return l2_relative_error.mean() # Return mean over batch dimension

if __name__ == '__main__':
    # Test L2RelativeError
    pred = torch.randn(10, 64, 64, 3) # Batch, H, W, C
    gt = torch.randn(10, 64, 64, 3)

    l2_error_metric = L2RelativeError()
    error = l2_error_metric(pred, gt)
    print(f"L2 Relative Error: {error.item()}")

    # Test with identical tensors
    error_zero = l2_error_metric(gt, gt)
    print(f"L2 Relative Error (identical): {error_zero.item()}")
    assert torch.isclose(error_zero, torch.tensor(0.0), atol=1e-5)

    # Test with a known difference
    pred_shifted = gt + 1.0 # Shifted by 1.0
    error_shifted = l2_error_metric(pred_shifted, gt)
    print(f"L2 Relative Error (shifted): {error_shifted.item()}")
    
    print("L2RelativeError tested successfully!")
