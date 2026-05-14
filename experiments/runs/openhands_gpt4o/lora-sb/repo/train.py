import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import LoRASBModel
from data import get_dataset
from config import Config

def train():
    # Load configuration
    config = Config()

    # Load dataset
    train_dataset, val_dataset = get_dataset(config.dataset_name)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    # Initialize model
    base_model = nn.Linear(config.input_dim, config.output_dim)  # Example base model
    model = LoRASBModel(base_model, rank=config.rank, scaling_factor=config.scaling_factor)

    # Define optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{config.epochs}, Loss: {total_loss / len(train_loader)}")

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        print(f"Validation Accuracy: {100 * correct / total}%")

if __name__ == "__main__":
    train()