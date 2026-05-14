"""Training and evaluation loops for PEFT experiments.

Supports:
- VTAB-1K low-shot training (100 epochs)
- Many-shot training (40 epochs)
- Robustness training (100-shot ImageNet)
- HP tuning with ArgParse/OmegaConf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import numpy as np
from tqdm import tqdm
import os
import copy
import math

from models.vit import VisionTransformer, vit_base_patch16_224
from models.peft_vit import build_peft_vit
from models.peft import (
    enable_bitfit, enable_layernorm_tuning, count_trainable_params
)


def build_model(num_classes, peft_method=None, peft_config=None, pretrained_path=None,
                backbone='in21k'):
    """Build ViT-B/16 with optional PEFT.

    Args:
        num_classes: number of downstream classes
        peft_method: None (linear probing), 'full', or PEFT method name
        peft_config: dict of PEFT hyperparameters
        pretrained_path: path to pretrained ViT weights
        backbone: 'in21k' or 'clip'

    Returns:
        model, list of trainable params
    """
    # Build base ViT
    base_vit = vit_base_patch16_224(
        num_classes=num_classes,
        drop_path_rate=peft_config.get('drop_path_rate', 0.1) if peft_config else 0.1,
    )

    if pretrained_path and os.path.exists(pretrained_path):
        state_dict = torch.load(pretrained_path, map_location='cpu')
        base_vit.load_state_dict(state_dict, strict=False)

    if peft_method is None:
        # Linear probing: freeze backbone, train only head
        for name, param in base_vit.named_parameters():
            if 'head' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        model = base_vit

    elif peft_method == 'full':
        # Full fine-tuning: all params trainable
        for param in base_vit.parameters():
            param.requires_grad = True
        model = base_vit

    else:
        # PEFT method
        model = build_peft_vit(peft_method, base_vit, peft_config)
        # Set head as trainable
        for param in model.head.parameters():
            param.requires_grad = True
        # Freeze backbone, then unfreeze PEFT-specific
        for name, param in model.named_parameters():
            if 'head' not in name:
                param.requires_grad = False

        if peft_method == 'bitfit':
            enable_bitfit(model)
        elif peft_method == 'layernorm':
            enable_layernorm_tuning(model)
        elif peft_method == 'difffit':
            enable_bitfit(model)
            enable_layernorm_tuning(model)
            # Enable DiffFit scales
            for name, param in model.named_parameters():
                if 'difffit_scales' in name:
                    param.requires_grad = True
        else:
            # Enable PEFT module parameters
            for name, param in model.named_parameters():
                if any(prefix in name for prefix in [
                    'vpt_shallow', 'vpt_deep', 'pfeif_adapter', 'houl_adapter',
                    'adaptformer', 'repadapter', 'convpass', 'ssf', 'lora',
                    'fact_tt', 'fact_tk'
                ]):
                    param.requires_grad = True

    trainable_params = count_trainable_params(model)
    print(f"Model: {peft_method or 'linear'}, Trainable params: {trainable_params:,} "
          f"({trainable_params / 86e6 * 100:.2f}% of ViT-B/16)")

    return model


def train_one_epoch(model, loader, optimizer, criterion, device, epoch,
                     warmup_scheduler=None, main_scheduler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f'Train Epoch {epoch}')
    for batch_idx, (images, targets) in enumerate(pbar):
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if warmup_scheduler is not None and main_scheduler is not None:
            warmup_scheduler.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix({
            'loss': f'{total_loss / (batch_idx + 1):.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    if main_scheduler is not None and warmup_scheduler is not None:
        main_scheduler.step()

    return total_loss / len(loader), 100. * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_outputs = []
    all_targets = []

    for images, targets in tqdm(loader, desc='Eval'):
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        all_outputs.append(outputs.cpu())
        all_targets.append(targets.cpu())

    acc = 100. * correct / total
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    return {
        'loss': total_loss / len(loader),
        'acc': acc,
        'outputs': all_outputs,
        'targets': all_targets,
    }


def train_vtab1k(model, train_loader, val_loader, test_loader=None,
                 epochs=100, lr=1e-3, weight_decay=1e-4, device='cuda',
                 warmup_epochs=5):
    """Train for VTAB-1K (100 epochs, AdamW, cosine decay)."""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    steps_per_epoch = len(train_loader)
    warmup_scheduler = LinearLR(
        optimizer, start_factor=1e-3, total_iters=warmup_epochs * steps_per_epoch
    )
    main_scheduler = CosineAnnealingLR(
        optimizer, T_max=(epochs - warmup_epochs) * steps_per_epoch
    )

    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch + 1, warmup_scheduler, main_scheduler
        )
        val_results = evaluate(model, val_loader, criterion, device)
        val_loss, val_acc = val_results['loss'], val_results['acc']

        print(f'Epoch {epoch + 1}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.2f}%, '
              f'Val Loss {val_loss:.4f}, Val Acc {val_acc:.2f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    if test_loader is not None:
        test_results = evaluate(model, test_loader, criterion, device)
        print(f'Test Acc: {test_results["acc"]:.2f}%')
        return model, best_val_acc, test_results['acc']

    return model, best_val_acc, None


def train_many_shot(model, train_loader, val_loader, test_loader=None,
                    epochs=40, lr=5e-4, weight_decay=1e-4, device='cuda',
                    warmup_epochs=2):
    """Train for many-shot datasets (40 epochs)."""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    steps_per_epoch = len(train_loader)
    warmup_scheduler = LinearLR(
        optimizer, start_factor=1e-3, total_iters=warmup_epochs * steps_per_epoch
    )
    main_scheduler = CosineAnnealingLR(
        optimizer, T_max=(epochs - warmup_epochs) * steps_per_epoch
    )

    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch + 1, warmup_scheduler, main_scheduler
        )
        val_results = evaluate(model, val_loader, criterion, device)
        val_loss, val_acc = val_results['loss'], val_results['acc']

        print(f'Epoch {epoch + 1}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.2f}%, '
              f'Val Loss {val_loss:.4f}, Val Acc {val_acc:.2f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    if test_loader is not None:
        test_results = evaluate(model, test_loader, criterion, device)
        print(f'Test Acc: {test_results["acc"]:.2f}%')
        return model, best_val_acc, test_results['acc']

    return model, best_val_acc, None


def train_robustness(model, train_loader, val_loader, shift_loaders=None,
                     epochs=10, lr=3e-5, weight_decay=5e-3, device='cuda',
                     warmup_epochs=1):
    """Train for robustness experiments (CLIP ViT, ImageNet 100-shot)."""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    steps_per_epoch = len(train_loader)
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs * steps_per_epoch
    )
    main_scheduler = CosineAnnealingLR(
        optimizer, T_max=(epochs - warmup_epochs) * steps_per_epoch
    )

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch + 1, warmup_scheduler, main_scheduler
        )
        val_results = evaluate(model, val_loader, criterion, device)
        val_loss, val_acc = val_results['loss'], val_results['acc']
        print(f'Epoch {epoch + 1}: Train Acc {train_acc:.2f}%, Val Acc {val_acc:.2f}%')

    # Evaluate on target and distribution shifts
    results = {}
    val_results = evaluate(model, val_loader, criterion, device)
    results['target'] = val_results['acc']

    if shift_loaders:
        shift_accs = {}
        for shift_name, shift_loader in shift_loaders.items():
            shift_results = evaluate(model, shift_loader, criterion, device)
            shift_accs[shift_name] = shift_results['acc']
            print(f'{shift_name}: {shift_results["acc"]:.2f}%')
        results['shifts'] = shift_accs
        results['avg_shift'] = np.mean(list(shift_accs.values()))

    return model, results
