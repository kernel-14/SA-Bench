import torch
import torch.optim as optim
import torch.nn as nn
from scripts.dataset_utils import get_cifar10
from models.consistency_model import ConsistencyModel
from configs.default_config import *

def train():
    # Initialize dataset
    train_loader, test_loader = get_cifar10(cifar10, batch_size)

    # Initialize model
    model = ConsistencyModel(input_channels=data_channels, resolution=resolution)
    model = model.cuda() if torch.cuda.is_available() else model

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, _ in train_loader:
            inputs = inputs.cuda() if torch.cuda.is_available() else inputs
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)  # Ideally replaced with GC or IC losses
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

if __name__ == "__main__":
    train()
