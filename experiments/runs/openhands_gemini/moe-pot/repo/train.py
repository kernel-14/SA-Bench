
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random

from config import config
from models import MoEPOT
from data import PDEDataset, BalancedBatchSampler
from utils import L2RelativeError

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_epoch(model, dataloader, optimizer, criterion_l2, balancer_loss_weight, device):
    model.train()
    total_loss = 0.0
    total_l2_loss = 0.0
    total_lb_loss = 0.0

    for u_input, u_target, mask, dataset_names in tqdm(dataloader, desc="Training"):
        u_input = u_input.to(device) # (B, T, H, W, C)
        u_target = u_target.to(device) # (B, H, W, C)
        mask = mask.to(device) # (B, H, W, 1)

        optimizer.zero_grad()

        # Model prediction: predict u_t from u_0...u_{t-1}
        # Output: (B, H, W, C), scalar lb_loss, list of expert_weights_flat
        prediction, lb_loss_per_block, expert_weights_list = model(u_input)
        
        # Calculate L2 loss
        # Apply mask to prediction and target if necessary (e.g., for irregular geometries)
        # The paper: sum_1^T ||G_w(u^<t + epsilon) - u^t||_2^2
        # Our current setup: predict u_t from T previous frames.
        # The `u_target` is effectively `u^t` (the next frame).
        # Mask is applied to select relevant parts of the loss.
        masked_prediction = prediction * mask
        masked_target = u_target * mask
        
        l2_loss = criterion_l2(masked_prediction, masked_target)
        
        # Combine L2 loss and Load Balancing loss
        # total_lb_loss = sum(lb_loss_per_block for each block)
        # The model returns `total_lb_loss` as the sum of all blocks.
        total_lb_loss_epoch = lb_loss_per_block * balancer_loss_weight
        
        loss = l2_loss + total_lb_loss_epoch
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_l2_loss += l2_loss.item()
        total_lb_loss += total_lb_loss_epoch.item() # Add the weighted loss

    return total_loss / len(dataloader), total_l2_loss / len(dataloader), total_lb_loss / len(dataloader)

def validate_one_epoch(model, dataloader, criterion_l2, device):
    model.eval()
    total_l2_error = 0.0
    with torch.no_grad():
        for u_input, u_target, mask, dataset_names in tqdm(dataloader, desc="Validation"):
            u_input = u_input.to(device)
            u_target = u_target.to(device)
            mask = mask.to(device)

            prediction, _, _ = model(u_input)
            
            masked_prediction = prediction * mask
            masked_target = u_target * mask

            l2_error = criterion_l2(masked_prediction, masked_target)
            total_l2_error += l2_error.item()
    return total_l2_error / len(dataloader)

def main():
    set_seed(config.seed)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dummy dataset paths (replace with actual paths when available)
    # These paths are used by PDEDataset to determine what datasets are "available".
    dummy_dataset_paths = {name: os.path.join(config.data.base_path, f"{name.lower().replace('-', '_').replace('.', '')}.npy") for name in config.data.dataset_names}
    
    # Create dummy data files if they don't exist, for testing purposes
    for path in dummy_dataset_paths.values():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            # Simulate different original resolutions and channels, but consistent enough for a dummy
            if "fno" in path:
                dummy_data = torch.randn(100, config.data.time_steps + 50, 64, 64, 1).numpy() # (N_samples, T_full, H, W, C)
            elif "cns" in path:
                dummy_data = torch.randn(100, config.data.time_steps + 40, 128, 128, 3).numpy()
            else:
                dummy_data = torch.randn(100, config.data.time_steps + 30, 96, 96, 2).numpy()
            dummy_data.tofile(path)
            print(f"Created dummy data file: {path}")

    # Determine max_channels dynamically from dummy_dataset_paths or config.
    # For now, let's hardcode based on CNS example having 3 channels.
    # In a real scenario, this would involve inspecting actual data files.
    max_channels_in_data = 3 # Max channels observed in dummy data

    train_dataset = PDEDataset(
        dataset_paths=dummy_dataset_paths,
        dataset_names=config.data.dataset_names,
        is_train=True,
        target_resolution=config.data.h_resolution,
        target_time_steps=config.data.time_steps,
        padding_value=config.data.padding_value,
        noise_epsilon=config.data.noise_epsilon,
        train_split_ratio=config.data.train_split_ratio,
        max_channels=max_channels_in_data
    )
    val_dataset = PDEDataset(
        dataset_paths=dummy_dataset_paths,
        dataset_names=config.data.dataset_names,
        is_train=False,
        target_resolution=config.data.h_resolution,
        target_time_steps=config.data.time_steps,
        padding_value=config.data.padding_value,
        noise_epsilon=0.0, # No noise for validation
        train_split_ratio=config.data.train_split_ratio,
        max_channels=max_channels_in_data
    )

    train_sampler = BalancedBatchSampler(train_dataset, config.data.balance_weights, config.data.batch_size)
    val_sampler = BalancedBatchSampler(val_dataset, config.data.balance_weights, config.data.batch_size)

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # Update model config with actual input/output channels
    config.model.in_channels = max_channels_in_data
    config.model.out_channels = max_channels_in_data # Predicting next frame, so same channels

    model = MoEPOT(
        patch_size=config.model.patch_size,
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        embed_dim=config.model.attention_dim,
        mlp_dim=config.model.mlp_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        num_routed_experts=config.model.num_routed_experts,
        num_shared_experts=config.model.num_shared_experts,
        top_k=config.model.top_k_experts,
        H=config.data.h_resolution,
        W=config.data.h_resolution,
        time_steps=config.data.time_steps
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
        betas=config.train.adam_betas
    )
    
    # Learning rate scheduler (One-cycle learning rate schedule)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.train.learning_rate,
        total_steps=config.train.epochs,
        pct_start=config.train.warmup_epochs / config.train.epochs,
        anneal_strategy='linear'
    )

    criterion_l2 = L2RelativeError()

    # Resume training if specified
    if config.checkpoint.resume and os.path.exists(config.checkpoint.resume_path):
        checkpoint = torch.load(config.checkpoint.resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0
        print("Starting new training run.")

    best_val_l2_error = float('inf')

    for epoch in range(start_epoch, config.train.epochs):
        train_loss, train_l2_loss, train_lb_loss = train_one_epoch(
            model, train_loader, optimizer, criterion_l2, config.train.balance_loss_weight, device
        )
        scheduler.step()

        print(f"Epoch {epoch+1}/{config.train.epochs} | "
              f"Train Loss: {train_loss:.6f} | L2 Loss: {train_l2_loss:.6f} | LB Loss: {train_lb_loss:.6f}")

        if (epoch + 1) % config.eval.eval_interval == 0:
            val_l2_error = validate_one_epoch(model, val_loader, criterion_l2, device)
            print(f"Validation L2 Relative Error: {val_l2_error:.6f}")

            # Save checkpoint
            if val_l2_error < best_val_l2_error:
                best_val_l2_error = val_l2_error
                checkpoint_path = os.path.join(config.checkpoint.save_dir, f"{config.run_name}_best.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'l2_error': best_val_l2_error,
                }, checkpoint_path)
                print(f"Saved best model checkpoint to {checkpoint_path} with L2 Error: {best_val_l2_error:.6f}")

            if (epoch + 1) % config.checkpoint.save_interval == 0:
                checkpoint_path = os.path.join(config.checkpoint.save_dir, f"{config.run_name}_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")

    print("Training finished.")

if __name__ == '__main__':
    main()
