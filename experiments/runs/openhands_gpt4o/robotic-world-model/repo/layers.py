import torch
import torch.nn as nn

class CustomLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(CustomLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.linear(x))

# Example usage
if __name__ == "__main__":
    custom_layer = CustomLayer(input_dim=128, output_dim=64)
    x = torch.randn(32, 128)  # Batch size 32
    out = custom_layer(x)
    print("Custom Layer Output Shape:", out.shape)