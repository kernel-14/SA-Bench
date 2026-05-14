import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRASBLayer(nn.Module):
    def __init__(self, original_linear_layer: nn.Linear, rank: int, scaling_factor: float = 1.0):
        super().__init__()
        self.in_features = original_linear_layer.in_features
        self.out_features = original_linear_layer.out_features
        self.rank = rank
        self.scaling_factor = scaling_factor

        # Store the original weight and bias as buffers, not as trainable parameters
        # W_0 is fixed. We only train R.
        self.register_buffer('W_0', original_linear_layer.weight.data.clone())
        if original_linear_layer.bias is not None:
            self.register_buffer('bias_0', original_linear_layer.bias.data.clone())
        else:
            self.register_buffer('bias_0', None)

        # B and A are fixed after initialization. R is trainable.
        # They will be initialized by the lora_sb_init.py module using SVD.
        # For now, initialize with zeros as placeholders. Their requires_grad will be set to False.
        # B: (out_features, rank)
        # A: (rank, in_features)
        # R: (rank, rank)
        self.B = nn.Parameter(torch.zeros(self.out_features, self.rank), requires_grad=False)
        self.A = nn.Parameter(torch.zeros(self.rank, self.in_features), requires_grad=False)
        self.R = nn.Parameter(torch.zeros(self.rank, self.rank)) # This is trainable

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize R to identity by default. The SVD-based initialization will override this.
        nn.init.eye_(self.R)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # W = W_0 + s * B @ R @ A
        delta_W = self.scaling_factor * (self.B @ self.R @ self.A)
        # The effective weight matrix used for the linear transformation
        effective_weight = self.W_0 + delta_W

        # Apply the linear transformation: x @ effective_weight.T + bias_0
        return F.linear(x, effective_weight, self.bias_0)

    def __repr__(self):
        return f"LoRASBLayer(in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, scaling_factor={self.scaling_factor})"


# Helper function to inject LoRA-SB layers into a model
def inject_lora_sb_layers(model: nn.Module, target_modules: list[str], rank: int, scaling_factor: float = 1.0):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(target_key in name for target_key in target_modules):
            print(f"Injecting LoRASBLayer into {name}")
            # Create a new LoRASBLayer instance
            lora_sb_layer = LoRASBLayer(module, rank, scaling_factor)

            # Replace the original linear layer with the LoRASBLayer
            # We need to find the parent module and replace the child
            sub_names = name.split('.')
            parent = model
            for sub_name in sub_names[:-1]:
                parent = getattr(parent, sub_name)
            setattr(parent, sub_names[-1], lora_sb_layer)
    return model

