import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import NaViLModel
from data import MultimodalDataset
from config import Config

def train():
    # Load configuration
    config = Config()

    # Initialize dataset and dataloader
    train_dataset = MultimodalDataset(config.train_data_path)
    val_dataset = MultimodalDataset(config.val_data_path)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Initialize model, loss, and optimizer
    model = NaViLModel(config.visual_encoder_config, config.llm_config, config.moe_config)
    model = model.to(config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # Training loop
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        for images, texts, labels in train_loader:
            images, texts, labels = images.to(config.device), texts.to(config.device), labels.to(config.device)

            optimizer.zero_grad()
            outputs = model(images, texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, texts, labels in val_loader:
                images, texts, labels = images.to(config.device), texts.to(config.device), labels.to(config.device)
                outputs = model(images, texts)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        print(f"Validation Loss: {val_loss/len(val_loader):.4f}")

if __name__ == "__main__":
    train()