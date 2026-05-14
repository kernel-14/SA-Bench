import torch
import torch.nn as nn

class GRUBase(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(GRUBase, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        out, _ = self.gru(x)
        return out

class MLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(MLPHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Example usage
if __name__ == "__main__":
    gru_base = GRUBase(input_dim=128, hidden_dim=256)
    mlp_head = MLPHead(input_dim=256, hidden_dim=128, output_dim=64)

    x = torch.randn(32, 10, 128)  # Batch size 32, sequence length 10
    gru_out = gru_base(x)
    mlp_out = mlp_head(gru_out[:, -1, :])

    print("GRU Output Shape:", gru_out.shape)
    print("MLP Output Shape:", mlp_out.shape)