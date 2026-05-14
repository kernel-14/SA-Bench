import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import GRUWorldModel

# Hyperparameters
INPUT_DIM = 128
HIDDEN_DIM = 256
OUTPUT_DIM = 64
LEARNING_RATE = 1e-4
BATCH_SIZE = 1024
EPOCHS = 10

# Dummy Dataset
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, seq_len, input_dim):
        self.data = torch.randn(num_samples, seq_len, input_dim)
        self.labels = torch.randn(num_samples, OUTPUT_DIM)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Training Loop
def train():
    # Model, Loss, Optimizer
    model = GRUWorldModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # DataLoader
    dataset = DummyDataset(10000, 10, INPUT_DIM)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Training
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            mean, std = model(inputs)
            loss = criterion(mean, targets)  # Example loss on mean
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss/len(dataloader):.4f}")

if __name__ == "__main__":
    train()