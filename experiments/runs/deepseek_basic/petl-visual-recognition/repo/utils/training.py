"""Training and evaluation utilities for PETL experiments."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import copy
import itertools
import time


def train_epoch(model, dataloader, optimizer, criterion, device='cuda', 
                epoch=0, num_epochs=100):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, dataloader, criterion, device='cuda'):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    all_logits = []
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        all_preds.append(predicted.cpu())
        all_targets.append(targets.cpu())
        all_logits.append(outputs.cpu())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_logits = torch.cat(all_logits)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'predictions': all_preds,
        'targets': all_targets,
        'logits': all_logits,
    }


def train_model(model, train_loader, val_loader, test_loader,
                method_name='', dataset_name='',
                lr=1e-3, weight_decay=1e-4, num_epochs=100,
                device='cuda', save_path=None, verbose=True):
    """Full training pipeline with cosine decay scheduler.
    
    Following the paper:
    - AdamW optimizer
    - Cosine decay learning rate scheduler
    - Batch size 64
    """
    # Collect trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=weight_decay,
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )
    
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    best_model_state = None
    best_epoch = 0
    
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch=epoch, num_epochs=num_epochs
        )
        
        val_results = evaluate(model, val_loader, criterion, device)
        val_acc = val_results['accuracy']
        
        train_losses.append(train_loss)
        val_accuracies.append(val_acc)
        
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        
        if verbose and (epoch % 20 == 0 or epoch == num_epochs - 1):
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                  f"train_acc={train_acc:.2f}%, val_acc={val_acc:.2f}%")
    
    # Load best model and evaluate on test set
    model.load_state_dict(best_model_state)
    test_results = evaluate(model, test_loader, criterion, device)
    
    if verbose:
        print(f"\nBest epoch: {best_epoch}, val_acc: {best_val_acc:.2f}%")
        print(f"Test accuracy: {test_results['accuracy']:.2f}%")
    
    # Count trainable params
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    result = {
        'method': method_name,
        'dataset': dataset_name,
        'test_accuracy': test_results['accuracy'],
        'best_val_accuracy': best_val_acc,
        'best_epoch': best_epoch,
        'trainable_params': param_count,
        'trainable_params_millions': param_count / 1e6,
        'predictions': test_results['predictions'].numpy(),
        'targets': test_results['targets'].numpy(),
        'logits': test_results['logits'].numpy(),
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'lr': lr,
        'weight_decay': weight_decay,
    }
    
    if save_path:
        torch.save({
            'model_state_dict': model.state_dict(),
            'result': result,
        }, save_path)
    
    return result


def hyperparameter_search(build_model_fn, train_loader, val_loader, test_loader,
                          hparam_grid, method_name, dataset_name, num_classes,
                          device='cuda', num_epochs=100, verbose=True):
    """Perform grid search over hyperparameters.
    
    Args:
        build_model_fn: Function that builds a model given hparams
        train_loader, val_loader, test_loader: DataLoaders
        hparam_grid: Dict of lists of hyperparameter values to search
        method_name: Name of the PEFT method
        dataset_name: Name of the dataset
        num_classes: Number of classes
        device: Device to use
        num_epochs: Number of epochs per trial
        verbose: Print progress
    
    Returns:
        best_result, best_hparams, all_results
    """
    # Generate all combinations
    if not hparam_grid:
        # No hparams to search
        model = build_model_fn(num_classes=num_classes)
        model = model.to(device)
        return train_model(
            model, train_loader, val_loader, test_loader,
            method_name=method_name, dataset_name=dataset_name,
            num_epochs=num_epochs, device=device, verbose=verbose
        ), {}, []
    
    hparam_names = list(hparam_grid.keys())
    hparam_values = list(hparam_grid.values())
    
    all_combinations = list(itertools.product(*hparam_values))
    
    if verbose:
        print(f"Hyperparameter search for {method_name} on {dataset_name}: "
              f"{len(all_combinations)} combinations")
    
    all_results = []
    best_val_acc = 0.0
    best_result = None
    best_hparams = None
    
    for combo_idx, combo in enumerate(all_combinations):
        hparams = dict(zip(hparam_names, combo))
        
        if verbose:
            print(f"  Trial {combo_idx+1}/{len(all_combinations)}: {hparams}")
        
        model = build_model_fn(num_classes=num_classes, **hparams)
        model = model.to(device)
        
        result = train_model(
            model, train_loader, val_loader, test_loader,
            method_name=method_name, dataset_name=dataset_name,
            num_epochs=num_epochs, device=device, verbose=False
        )
        result['hparams'] = hparams
        all_results.append(result)
        
        if result['best_val_accuracy'] > best_val_acc:
            best_val_acc = result['best_val_accuracy']
            best_result = result
            best_hparams = hparams
    
    if verbose:
        print(f"Best val accuracy: {best_val_acc:.2f}% with {best_hparams}")
        print(f"Best test accuracy: {best_result['test_accuracy']:.2f}%")
    
    return best_result, best_hparams, all_results


def get_vtab_hparam_grid(method_name):
    """Get the VTAB-1K hyperparameter grid for a method (from Table 3)."""
    
    # Base grid for all methods
    base_grid = {
        'lr': [1e-3, 5e-3, 1e-2],
        'weight_decay': [1e-4, 5e-4, 1e-3],
        'drop_path_rate': [0.0, 0.1],
    }
    
    # Method-specific grids from Table 3
    method_grids = {
        'vpt_shallow': {'prompt_number': [5, 10, 50, 100, 200]},
        'vpt_deep': {'prompt_number': [5, 10, 50, 100]},
        'bitfit': {},
        'difffit': {},
        'layernorm': {},
        'ssf': {},
        'pfeiffer_adapter': {
            'adapter_scale': [0.01, 0.1, 1, 10],
            'adapter_bottleneck': [4, 8, 16, 32],
        },
        'houlsby_adapter': {
            'adapter_scale': [0.01, 0.1, 1, 10],
            'adapter_bottleneck': [4, 8, 16, 32],
        },
        'adaptformer': {
            'adapter_scale': [0.05, 0.1, 0.2],
            'adapter_bottleneck': [4, 16, 32],
        },
        'repadapter': {
            'adapter_scale': [0.1, 0.5, 1, 5, 10],
            'adapter_bottleneck': [8, 16, 32],
        },
        'convpass': {
            'adapter_scale': [0.01, 0.1, 1, 10, 100],
            'adapter_bottleneck': [8, 16],
            'xavier_init': [True, False],
        },
        'lora': {'lora_rank': [1, 8, 16, 32]},
        'fact_tt': {
            'fact_scale': [0.01, 0.1, 1, 10, 100],
            'fact_bottleneck': [8, 16, 32],
        },
        'fact_tk': {
            'fact_scale': [0.01, 0.1, 1, 10, 100],
            'fact_bottleneck': [16, 32, 64],
        },
    }
    
    if method_name in method_grids:
        grid = {**base_grid}
        # Don't include lr/wd in method-specific grid for now - those are tuned separately
        for k, v in method_grids[method_name].items():
            grid[k] = v
        return grid
    
    return base_grid
