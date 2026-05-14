import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import ConsistencyModel, GeneratorAugmentedFlow
from data import get_dataset
from config import Config

def train():
    # Load configuration
    config = Config()

    # Initialize dataset and dataloader
    dataset = get_dataset(config.dataset_name, config.dataset_path)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Initialize model, optimizer, and loss function
    consistency_model = ConsistencyModel(config.input_dim, config.hidden_dim, config.output_dim)
    model = GeneratorAugmentedFlow(consistency_model)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()

    # Training loop
    model.train()
    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            x, sigma = batch
            optimizer.zero_grad()
            outputs = model(x, sigma)
            loss = loss_fn(outputs, x)  # Example loss computation
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{config.num_epochs}], Loss: {epoch_loss:.4f}")

if __name__ == "__main__":
    train()