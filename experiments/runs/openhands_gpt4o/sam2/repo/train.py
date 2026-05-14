import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import SAM2
from data import SAVDataset
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize dataset and dataloader
dataset = SAVDataset(config['data']['path'])
dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)

# Initialize model
model = SAM2(
    hiera_config=config['model']['hiera_config'],
    num_layers=config['model']['num_layers'],
    embed_dim=config['model']['embed_dim']
)
model = model.to(config['training']['device'])

# Define loss functions
mask_loss_fn = nn.BCELoss()
iou_loss_fn = nn.L1Loss()

# Define optimizer
optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])

# Training loop
for epoch in range(config['training']['epochs']):
    model.train()
    epoch_loss = 0
    for batch in dataloader:
        frames, prompts, memory, ground_truth_masks = batch
        frames = frames.to(config['training']['device'])
        prompts = prompts.to(config['training']['device'])
        memory = memory.to(config['training']['device'])
        ground_truth_masks = ground_truth_masks.to(config['training']['device'])

        # Forward pass
        predicted_masks, updated_memory = model(frames, prompts, memory)

        # Compute losses
        mask_loss = mask_loss_fn(predicted_masks, ground_truth_masks)
        iou_loss = iou_loss_fn(predicted_masks, ground_truth_masks)
        loss = mask_loss + config['training']['iou_loss_weight'] * iou_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{config['training']['epochs']}, Loss: {epoch_loss / len(dataloader)}")

# Save the model
torch.save(model.state_dict(), config['training']['save_path'])