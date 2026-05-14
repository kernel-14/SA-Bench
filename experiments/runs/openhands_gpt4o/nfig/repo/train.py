import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import FRVAE, TransformerAR
from data import ImageNetDataset
from config import Config

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        images = batch.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            images = batch.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize dataset and dataloaders
    train_dataset = ImageNetDataset(config.data_path, split='train')
    val_dataset = ImageNetDataset(config.data_path, split='val')
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # Initialize model, optimizer, and loss function
    model = FRVAE(config.encoder, config.decoder, config.frequency_masks, config.codebook_size, config.feature_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(config.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{config.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), config.model_save_path)

if __name__ == "__main__":
    main()