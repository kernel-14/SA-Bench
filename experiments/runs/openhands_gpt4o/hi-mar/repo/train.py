import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import ImageDataset
from model import HiMAR
from config import Config

def train():
    # Load configuration
    config = Config()

    # Initialize dataset and dataloader
    train_dataset = ImageDataset(config.train_data_path, config.image_size)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    # Initialize model, loss, and optimizer
    model = HiMAR(config.model)
    model = model.to(config.device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # Training loop
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            low_res_tokens, high_res_tokens, context_tokens = batch
            low_res_tokens = low_res_tokens.to(config.device)
            high_res_tokens = high_res_tokens.to(config.device)
            context_tokens = context_tokens.to(config.device)

            optimizer.zero_grad()

            low_res_output, high_res_output = model(low_res_tokens, high_res_tokens, context_tokens)

            loss = criterion(low_res_output, low_res_tokens) + criterion(high_res_output, high_res_tokens)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{config.epochs}], Loss: {epoch_loss/len(train_loader):.4f}")

if __name__ == "__main__":
    train()