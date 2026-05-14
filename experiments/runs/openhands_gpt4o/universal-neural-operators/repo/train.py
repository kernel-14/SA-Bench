# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import PDEBenchDataset
from model import NeuralOperator

def train_model(config):
    # Load dataset
    train_dataset = PDEBenchDataset(config['data']['train_path'])
    val_dataset = PDEBenchDataset(config['data']['val_path'])
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    # Initialize model
    model = NeuralOperator(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        output_dim=config['model']['output_dim'],
        num_layers=config['model']['num_layers']
    )
    model = model.to(config['device'])

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])

    # Training loop
    for epoch in range(config['training']['epochs']):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(config['device']), targets.to(config['device'])

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(config['device']), targets.to(config['device'])
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{config['training']['epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

if __name__ == "__main__":
    from config import config
    train_model(config)