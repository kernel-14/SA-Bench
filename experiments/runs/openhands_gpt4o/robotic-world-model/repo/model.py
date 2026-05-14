import torch
import torch.nn as nn

class GRUWorldModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(GRUWorldModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc_mean = nn.Linear(hidden_dim, output_dim)
        self.fc_std = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        mean = self.fc_mean(out[:, -1, :])
        std = torch.exp(self.fc_std(out[:, -1, :]))  # Ensure std is positive
        return mean, std

class MLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(MLPHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Example instantiation
if __name__ == "__main__":
    input_dim = 128
    hidden_dim = 256
    output_dim = 64

    model = GRUWorldModel(input_dim, hidden_dim, output_dim)
    x = torch.randn(32, 10, input_dim)  # Batch size 32, sequence length 10
    mean, std = model(x)
    print("Mean:", mean.shape, "Std:", std.shape)