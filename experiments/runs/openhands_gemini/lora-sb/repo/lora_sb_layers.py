
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRASBLayer(nn.Module):
    def __init__(self, original_weight: nn.Module, rank: int, scaling_factor: float = 1.0):
        super().__init__()
        self.original_weight = original_weight
        self.rank = rank
        self.scaling_factor = scaling_factor

        # W_0 is the original pre-trained weight, which remains fixed
        # For Linear layers, original_weight will be `weight` attribute
        # For Conv2d layers, original_weight will be `weight` attribute
        # It's important that original_weight.requires_grad remains False if it's part of W_0
        if hasattr(original_weight, 'weight'):
            self.original_weight_data = original_weight.weight.data
            self.in_features = original_weight.in_features if hasattr(original_weight, 'in_features') else original_weight.weight.shape[1]
            self.out_features = original_weight.out_features if hasattr(original_weight, 'out_features') else original_weight.weight.shape[0]
        else:
            raise ValueError("LoRASBLayer expects original_weight to have a 'weight' attribute.")
        
        # Initialize B, R, A here, but their values will be set by the initialization routine
        # B: m x r matrix
        # A: r x n matrix
        # R: r x r matrix
        self.B = nn.Parameter(torch.empty(self.out_features, self.rank), requires_grad=False)
        self.A = nn.Parameter(torch.empty(self.rank, self.in_features), requires_grad=False)
        self.R = nn.Parameter(torch.empty(self.rank, self.rank), requires_grad=True) # Only R is trainable

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize with zeros for B, A, R. Actual values will be set by the LoRA-SB initialization process
        nn.init.zeros_(self.B)
        nn.init.zeros_(self.A)
        nn.init.zeros_(self.R)

    def forward(self, x: torch.Tensor):
        # The full weight matrix is W = W_0 + s B R A
        # W_0 can be implicit in the original_weight module's forward pass
        # Or, if W_0 is extracted, it would be explicitly added.
        # Given the paper's formulation, it modifies the _effective_ weight.
        
        # Assuming original_weight is a Linear layer:
        # original_weight(x) = x @ W_0^T + bias
        # New behavior: x @ (W_0 + s B R A)^T + bias = x @ W_0^T + x @ (s B R A)^T + bias
        # So we add the low-rank update to the output of the original layer.

        # The paper describes this as a replacement of W, not an additive adapter
        # so we need to construct the full W and use it.
        # This implies that we take W_0 from original_weight and effectively replace its weight.
        # A more common PEFT approach would be to add the low-rank update to the output of W_0.
        # However, the paper implies W = W_0 + delta_W, where delta_W = s B R A.
        
        delta_W = self.scaling_factor * (self.B @ self.R @ self.A)
        
        # If original_weight is a nn.Linear layer, its .weight is transposed compared to math notation (in_features, out_features)
        # PyTorch Linear layer expects (out_features, in_features) for its weight matrix
        # So if original_weight.weight has shape (out_features, in_features), then B @ R @ A should have the same shape
        # B: (out_features, rank)
        # R: (rank, rank)
        # A: (rank, in_features)
        # B @ R @ A: (out_features, in_features)
        
        # Apply the update to the original weight and then perform the linear operation
        # This effectively replaces the original weight with W_0 + delta_W
        updated_weight = self.original_weight_data + delta_W
        
        if isinstance(self.original_weight, nn.Linear):
            return F.linear(x, updated_weight, self.original_weight.bias)
        elif isinstance(self.original_weight, nn.Conv2d):
            # This case might be more complex depending on Conv2d's parameters (stride, padding, dilation, groups)
            # For simplicity, assuming typical use where these parameters are fixed
            return F.conv2d(
                x, updated_weight, self.original_weight.bias,
                self.original_weight.stride, self.original_weight.padding,
                self.original_weight.dilation, self.original_weight.groups
            )
        else:
            raise NotImplementedError(f"LoRASBLayer does not support original_weight type: {type(self.original_weight)}")

