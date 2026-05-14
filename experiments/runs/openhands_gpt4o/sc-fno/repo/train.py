import torch
import torch.nn as nn
import torch.optim as optim
from model import SCFNO

def train_model(model, dataloader, epochs, lr):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for batch in dataloader:
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Example usage
# model = SCFNO(modes=8, width=20, layers=4)
# train_model(model, dataloader, epochs=100, lr=0.001)