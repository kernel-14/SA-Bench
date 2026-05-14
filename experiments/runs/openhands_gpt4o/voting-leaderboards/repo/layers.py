import torch
import torch.nn as nn

class CustomLinearLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(CustomLinearLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.linear(x))

class CustomDropoutLayer(nn.Module):
    def __init__(self, dropout_rate: float):
        super(CustomDropoutLayer, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.dropout(x)

# Example usage
if __name__ == "__main__":
    linear_layer = CustomLinearLayer(input_dim=256, output_dim=128)
    dropout_layer = CustomDropoutLayer(dropout_rate=0.5)

    sample_input = torch.randn(1, 256)
    output = linear_layer(sample_input)
    output = dropout_layer(output)
    print(f"Output shape: {output.shape}")