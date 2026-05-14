import torch
import torch.nn as nn

class LoRA(nn.Module):
    def __init__(self, linear_layer, rank):
        """
        Wraps a given linear layer with LoRA low-rank adaptation.
        """
        super(LoRA, self).__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = rank

        # Original linear transformation
        self.linear_layer = linear_layer

        # LoRA low-rank parameterization
        self.B = nn.Parameter(torch.randn(self.out_features, rank))
        self.A = nn.Parameter(torch.randn(rank, self.in_features))
        self.R = nn.Parameter(torch.eye(rank))

        # Initialize LoRA parameters
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        """
        Forward pass with LoRA approximation.
        """
        lora_update = torch.matmul(torch.matmul(self.B, self.R), torch.matmul(self.A, x.T)).T
        return self.linear_layer(x) + lora_update

# Example Usage:
# original_layer = nn.Linear(in_features=768, out_features=768)
# lora_layer = LoRA(original_layer, rank=32)

