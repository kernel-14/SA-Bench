import torch
import torch.nn as nn

class SensitivityLoss(nn.Module):
    """
    Implements the Sensitivity Loss (Ls) as defined in the paper (Equation 70).

    Ls = (1/M) * sum(|| d(u_hat)/dp - d(u_true)/dp ||^2)

    where:
    - u_hat: Predicted solution from the FNO model.
    - u_true: True solution from the differentiable solver.
    - dp: Parameters with respect to which sensitivity is calculated.
    - M: Number of evaluation points.

    This loss term explicitly penalizes discrepancies between the predicted
    sensitivities (Jacobians) and the true sensitivities, encouraging the model
    to learn the correct dependency of the solution on input parameters.
    """
    def __init__(self):
        super(SensitivityLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='sum') # Sum for later division by M

    def forward(self, predicted_jacobian, true_jacobian):
        """
        Calculates the sensitivity loss.

        Args:
            predicted_jacobian (torch.Tensor): Jacobian of the predicted solution
                                               with respect to the parameters.
                                               Shape: (M, num_parameters)
            true_jacobian (torch.Tensor): True Jacobian of the solution
                                          with respect to the parameters.
                                          Shape: (M, num_parameters)

        Returns:
            torch.Tensor: The scalar sensitivity loss.
        """
        if predicted_jacobian.shape != true_jacobian.shape:
            raise ValueError("Predicted and true Jacobians must have the same shape.")

        # The paper defines Ls as (1/M) * sum(|| ... ||^2)
        # nn.MSELoss with reduction='sum' calculates sum((x_i - y_i)^2)
        # So we just need to divide by M (number of evaluation points)
        M = predicted_jacobian.shape[0]
        loss = self.mse_loss(predicted_jacobian, true_jacobian) / M
        return loss

if __name__ == '__main__':
    # Example usage (conceptual)
    sensitivity_loss_fn = SensitivityLoss()

    # Simulate predicted and true Jacobians
    # M = 100 evaluation points, num_parameters = 3
    M = 100
    num_parameters = 3
    predicted_jac = torch.randn(M, num_parameters, requires_grad=True)
    true_jac = torch.randn(M, num_parameters)

    loss = sensitivity_loss_fn(predicted_jac, true_jac)
    print(f"Sensitivity Loss: {loss.item()}")

    # Test with different shapes (should raise error)
    try:
        sensitivity_loss_fn(torch.randn(50, 2), torch.randn(100, 2))
    except ValueError as e:
        print(f"Error caught as expected: {e}")
