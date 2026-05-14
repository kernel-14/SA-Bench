import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import PDEDataLoader
from model import MoEPOT
from config import Config

def train():
    # Load configuration
    config = Config()

    # Initialize model
    model = MoEPOT(
        attention_dim=config.attention_dim,
        mlp_dim=config.mlp_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        num_routed_experts=config.num_routed_experts,
        num_shared_experts=config.num_shared_experts,
        top_k=config.top_k
    ).to(config.device)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # Load data
    train_loader = PDEDataLoader(config.train_data_path, config.batch_size, shuffle=True)
    val_loader = PDEDataLoader(config.val_data_path, config.batch_size, shuffle=False)

    # Training loop
    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(config.device), targets.to(config.device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(config.device), targets.to(config.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch + 1}/{config.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

if __name__ == "__main__":
    train()