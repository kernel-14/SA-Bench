# train.py

import torch
from torch.utils.data import DataLoader
from data import VideoDataset
from model import PyramidalFlowMatchingModel
from config import Config

def train():
    # Load configuration
    config = Config()

    # Initialize dataset and dataloader
    dataset = VideoDataset(config.dataset_path)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Initialize model
    model = PyramidalFlowMatchingModel(num_stages=config.num_stages, base_model=config.base_model)
    model = model.to(config.device)

    # Define optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.MSELoss()

    # Training loop
    for epoch in range(config.num_epochs):
        model.train()
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(config.device), targets.to(config.device)

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch+1}/{config.num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()