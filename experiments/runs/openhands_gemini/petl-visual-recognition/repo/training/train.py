
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os

from models.vit import vit_base_patch16_224_in21k, clip_vit_base_patch16_224
from models.model import build_peft_model
from data.data import get_dataloader
from configs.config import get_config

def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast() if scaler else torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

def main():
    config = get_config()

    # Set device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # Load backbone model
    if 'clip' in config.backbone.lower():
        backbone_model = clip_vit_base_patch16_224(pretrained=config.pretrained, num_classes=config.num_classes)
    else:
        backbone_model = vit_base_patch16_224_in21k(pretrained=config.pretrained, num_classes=config.num_classes)
    
    backbone_model.to(device)

    # Apply PEFT method
    model = build_peft_model(config, backbone_model)
    model.to(device)

    # Setup data loaders
    train_dataloader = get_dataloader(config, is_train=True)
    val_dataloader = get_dataloader(config, is_train=False) # Use same for simplicity, but could be separate val set

    # Optimizer
    current_lr = config.lr
    current_weight_decay = config.weight_decay

    if is_many_shot:
        current_lr = config.many_shot_lr
    elif is_robustness_exp:
        current_lr = config.robustness_lr
        current_weight_decay = config.robustness_weight_decay

    if config.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), 
                                lr=current_lr, 
                                weight_decay=current_weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # LR Scheduler
    if config.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    else:
        scheduler = None # No scheduler or other scheduler types

    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.amp.is_available() else None

    best_val_accuracy = 0.0
    
    num_epochs = config.epochs
    is_many_shot = config.dataset in ['cifar100', 'resisc', 'clevr_distance'] # Datasets for many-shot
    is_robustness_exp = (config.dataset == 'imagenet_1k' and config.peft_method in ['full_finetune', 'lora', 'bitfit', 'layernorm', 'houl_adapter', 'adaptformer', 'repadapter', 'convpass', 'fact_tt', 'fact_tk']) # Based on robustness section 7 and Table 2

    if is_many_shot:
        num_epochs = config.many_shot_epochs
        print(f"Using many-shot training epochs: {num_epochs}")
    elif is_robustness_exp:
        num_epochs = config.robustness_epochs
        print(f"Using robustness training epochs: {num_epochs}")
    else: # VTAB-1K default
        num_epochs = config.epochs
        print(f"Using VTAB-1K training epochs: {num_epochs}")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss, train_acc = train_one_epoch(model, train_dataloader, optimizer, criterion, device, scaler)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        val_loss, val_acc = evaluate_model(model, val_dataloader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if scheduler:
            scheduler.step()

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            # Save best model
            # torch.save(model.state_dict(), os.path.join("checkpoints", f"{config.peft_method}_best_model.pth"))
            print(f"New best validation accuracy: {best_val_accuracy:.4f}. Model saved.")

    print(f"Training finished. Best validation accuracy: {best_val_accuracy:.4f}")

if __name__ == '__main__':
    main()

