## evaluators/evaluator.py
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional

# Import project-specific modules
from config import Config
from utils import compute_r2, compute_relative_l2
from models.fno import FNO  # Base FNO model
from models.sc_fno_base import SCFNOBase  # SC-FNO model
from models.sc_fno_pinn import SCFNO_PINN  # SC-FNO-PINN model


class Evaluator:
    """
    Evaluates trained models on test data, computing R-squared and Relative L2 errors
    for both predicted solutions and, if applicable, predicted sensitivities.
    """

    def __init__(self, model: torch.nn.Module, test_loader: DataLoader, config: Config, device: str) -> None:
        """
        Initializes the Evaluator.

        Args:
            model (torch.nn.Module): The trained model to be evaluated. Can be FNO, SCFNOBase, or SCFNO_PINN.
            test_loader (DataLoader): DataLoader for the test dataset.
            config (Config): The configuration object for the experiment.
            device (str): The computational device ('cuda' or 'cpu').
        """
        self.model = model
        self.test_loader = test_loader
        self.config = config
        self.device = device
        
        self.model.eval()  # Set the model to evaluation mode
        self.model.to(self.device)

        # Determine if the model is an SC-FNO variant to decide whether to compute sensitivity metrics.
        # SCFNO_PINN wraps SCFNOBase, so checking for SCFNOBase is sufficient for sensitivity capability.
        self.is_sc_model = isinstance(self.model, (SCFNOBase, SCFNO_PINN))

        # Get equation_id to correctly handle the FNO input concatenation for vanilla FNO
        self.equation_id = self.config.get("experiment.equation_id")
        if self.equation_id is None:
            raise ValueError("Equation ID must be specified in config for evaluator.")
        
        print(f"Evaluator initialized for model type: {type(self.model).__name__} on device: {self.device}")

    def _prepare_fno_combined_input(self, input_base_features: torch.Tensor, params_input: torch.Tensor) -> torch.Tensor:
        """
        Prepares the combined input for a vanilla FNO model by concatenating
        input base features (u0, coords) with broadcasted parameters.
        This is a helper for vanilla FNO models that expect all inputs combined
        into a single tensor.

        Args:
            input_base_features (torch.Tensor): Features without parameters (u0, coords).
                                                Shape: (batch_size, D0, D1, ..., Dn-1, input_base_dim).
            params_input (torch.Tensor): The raw parameter tensor.
                                         Shape: (batch_size, num_parameters).

        Returns:
            torch.Tensor: Combined input tensor for the FNO backbone.
                          Shape: (batch_size, D0, D1, ..., Dn-1, input_base_dim + num_parameters).
        """
        # Determine the target shape for broadcasting parameters
        # The dimensions are (batch_size, *grid_dims, feature_dim).
        # We need to broadcast params_input (batch_size, num_params) to (batch_size, *grid_dims, num_params).
        
        target_grid_dims = input_base_features.shape[1:-1] # (D0, D1, ..., Dn-1)
        
        # Reshape params_input to (batch_size, 1, ..., 1, num_params)
        num_grid_dims = len(target_grid_dims)
        broadcast_shape = (params_input.shape[0], *([1] * num_grid_dims), params_input.shape[-1])
        
        broadcasted_params = params_input.view(broadcast_shape)
        
        # Expand to match the spatial-temporal dimensions of input_base_features
        # Example: if input_base_features is (B, T, S, F), target_grid_dims is (T, S).
        # broadcasted_params (B, 1, 1, P) -> (B, T, S, P)
        expanded_params = broadcasted_params.expand(input_base_features.shape[0], *target_grid_dims, params_input.shape[-1])
        
        # Concatenate along the last dimension (feature dimension)
        return torch.cat([input_base_features, expanded_params], dim=-1)

    def evaluate(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluates the model's performance on the test dataset.

        Returns:
            Dict[str, Dict[str, float]]: A dictionary containing evaluation metrics
                                         (R2, Relative L2) for 'u' (solution)
                                         and potentially 'du_dp' (sensitivities).
        """
        all_u_true: list[torch.Tensor] = []
        all_u_pred: list[torch.Tensor] = []
        all_du_dp_true: list[torch.Tensor] = []
        all_du_dp_pred: list[torch.Tensor] = []

        # Ensure model is in evaluation mode; gradients are handled conditionally below.
        self.model.eval()

        # Iterate over the test data. No `torch.no_grad()` at the top level
        # because SC models need gradients for Jacobian computation.
        for batch in self.test_loader:
            # Extract data from batch.
            # Assuming dataset_generator provides:
            #   - fno_input_base_features: (u0, coords only)
            #   - fno_params_for_ad: (raw parameters for AD)
            #   - fno_target_u: true solution
            #   - fno_target_du_dp: true Jacobian
            input_base_features = batch['fno_input_base_features'].to(self.device)
            params_input_raw = batch['fno_params_for_ad'].to(self.device)
            u_true = batch['fno_target_u'].to(self.device)
            du_dp_true = batch['fno_target_du_dp'].to(self.device)

            u_pred: torch.Tensor
            du_dp_pred: Optional[torch.Tensor] = None

            if self.is_sc_model:
                # For SC models, params_input needs to track gradients for Jacobian computation.
                params_input_grad = params_input_raw.clone().detach().requires_grad_(True)
                
                # Forward pass: model accepts base features and separate params_input_grad
                u_pred = self.model(input_base_features, params_input_grad)
                
                # Compute predicted Jacobian with respect to params_input_grad
                du_dp_pred = self.model.compute_jacobian(u_pred, params_input_grad)
                
                # Detach all outputs before storing
                u_pred = u_pred.detach()
                du_dp_pred = du_dp_pred.detach()

                all_du_dp_true.append(du_dp_true)
                all_du_dp_pred.append(du_dp_pred)
            else:
                # For vanilla FNO, parameters are expected to be embedded in the input features.
                # We construct this combined input here using the helper.
                combined_input_for_fno = self._prepare_fno_combined_input(input_base_features, params_input_raw)
                
                # Perform forward pass within no_grad context for vanilla FNO
                # as we don't need its internal gradients for Jacobian here.
                with torch.no_grad():
                    u_pred = self.model(combined_input_for_fno)
                
                # Detach u_pred before storing
                u_pred = u_pred.detach()
                # No sensitivity prediction for vanilla FNO

            all_u_true.append(u_true)
            all_u_pred.append(u_pred)

        # Aggregate all results by concatenating tensors along the batch dimension
        all_u_true_cat = torch.cat(all_u_true, dim=0)
        all_u_pred_cat = torch.cat(all_u_pred, dim=0)

        results: Dict[str, Dict[str, float]] = {}
        
        # Compute metrics for solution paths (u)
        results["u"] = {
            "R2": compute_r2(all_u_pred_cat, all_u_true_cat),
            "Relative L2": compute_relative_l2(all_u_pred_cat, all_u_true_cat)
        }

        # Compute metrics for sensitivities (du/dp) if applicable and data is present
        if self.is_sc_model and len(all_du_dp_true) > 0:
            all_du_dp_true_cat = torch.cat(all_du_dp_true, dim=0)
            all_du_dp_pred_cat = torch.cat(all_du_dp_pred, dim=0)
            
            results["du_dp"] = {
                "R2": compute_r2(all_du_dp_pred_cat, all_du_dp_true_cat),
                "Relative L2": compute_relative_l2(all_du_dp_pred_cat, all_du_dp_true_cat)
            }
        elif self.is_sc_model and len(all_du_dp_true) == 0:
            print("Warning: SC model detected but no sensitivity data was present in the test set or du/dp_true is empty. Skipping du/dp metrics.")

        return results

