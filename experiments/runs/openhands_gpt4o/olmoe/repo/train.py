# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import OLMoEDataset
from model import OLMoEModel
from config import Config

def train():
    # Load configuration
    config = Config()

    # Initialize dataset and dataloader
    dataset = OLMoEDataset(config.data_path)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Initialize model
    model = OLMoEModel(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_experts=config.num_experts,
        num_active_experts=config.num_active_experts
    )

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, eps=config.adam_epsilon)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(config.num_epochs):
        model.train()
        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, config.vocab_size), targets.view(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{config.num_epochs}, Loss: {loss.item()}")

if __name__ == "__main__":
    train()