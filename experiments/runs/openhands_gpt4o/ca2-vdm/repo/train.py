import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Ca2VDM
from data import VideoDataset
from config import CONFIG

def train():
    # Load configuration
    config = CONFIG

    # Initialize model
    model = Ca2VDM(
        latent_dim=config['model']['latent_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout'],
        causal_attention=config['model']['causal_attention'],
        prefix_enhanced_attention=config['model']['prefix_enhanced_attention']
    ).to('cuda')

    # Define optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    criterion = nn.MSELoss()

    # Load dataset
    train_dataset = VideoDataset(split=config['dataset']['train_split'], resolution=config['dataset']['resolution'])
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['dataset']['num_workers'])

    # Training loop
    model.train()
    for step in range(config['training']['num_steps']):
        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to('cuda'), targets.to('cuda')

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Logging
        if step % config['logging']['save_interval'] == 0:
            print(f"Step {step}, Loss: {loss.item()}")
            torch.save(model.state_dict(), f"{config['logging']['checkpoint_dir']}/model_step_{step}.pth")

if __name__ == "__main__":
    train()