# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import MaskedDataset
from model import MaskedDiffusionModel
from config import Config

def train():
    config = Config()
    dataset = MaskedDataset(config.data_path)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = MaskedDiffusionModel(config.vocab_size, config.sequence_length, config.hidden_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    model.train()
    for epoch in range(config.epochs):
        total_loss = 0
        for batch in dataloader:
            x_t, t, target = batch
            optimizer.zero_grad()
            output = model.forward(x_t, t)
            loss = criterion(output.view(-1, config.vocab_size), target.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{config.epochs}, Loss: {total_loss / len(dataloader)}")

if __name__ == "__main__":
    train()