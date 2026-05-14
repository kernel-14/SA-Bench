import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import DRC
from data import SokobanDataset
from config import Config

def train():
    config = Config()

    # Initialize dataset and dataloader
    train_dataset = SokobanDataset(config.train_data_path)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    # Initialize model, optimizer, and loss function
    model = DRC(
        input_channels=config.input_channels,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_internal_ticks=config.num_internal_ticks,
        action_dim=config.action_dim
    )
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()

    for epoch in range(config.num_epochs):
        epoch_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            policy_logits, value = model(inputs)
            loss = criterion(policy_logits, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{config.num_epochs}, Loss: {epoch_loss / len(train_loader)}")

if __name__ == "__main__":
    train()