import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import NormalizedTransformer
from data import OpenWebTextDataset
from config import Config

def train():
    config = Config()

    # Initialize model, dataset, and optimizer
    model = NormalizedTransformer(
        d_model=config.d_model,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        n_layers=config.n_layers,
        vocab_size=config.vocab_size
    ).to(config.device)

    dataset = OpenWebTextDataset(config.dataset_path)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0

        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(config.device), targets.to(config.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, config.vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{config.epochs}, Loss: {total_loss / len(dataloader)}")

if __name__ == "__main__":
    train()