import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import CustomDataset
from model import GatedAttentionModel
from config import Config

def train():
    # Load configuration
    config = Config()

    # Initialize dataset and dataloader
    train_dataset = CustomDataset(config.train_data_path)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    # Initialize model, loss, and optimizer
    model = GatedAttentionModel(
        d_model=config.d_model,
        n_heads=config.n_heads,
        num_layers=config.num_layers,
        d_ff=config.d_ff,
        dropout=config.dropout
    ).to(config.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Training loop
    model.train()
    for epoch in range(config.num_epochs):
        total_loss = 0
        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(config.device), targets.to(config.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{config.num_epochs}, Loss: {total_loss / len(train_loader):.4f}")

if __name__ == "__main__":
    train()