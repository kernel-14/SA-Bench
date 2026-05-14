import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import yaml
import os

from models.vit import ViT
from models.peft_modules import get_peft_model
from data_utils.vtab_datasets import get_vtab_datasets
from data_utils.many_shot_datasets import get_many_shot_datasets
from data_utils.robustness_datasets import get_robustness_datasets
from training.evaluate import evaluate_model
from training.utils import get_device

def train_model(config):
    device = get_device()
    print(f"Using device: {device}")

    # Initialize model with a dummy num_classes, it will be reset per task
    model = ViT(model_name=config['model']['name'], num_classes=1000) 

    # Apply PEFT method or handle full_ft/linear_probing
    peft_method = config['peft']['method']
    if peft_method == "none": # This will default to linear probing if no other action is taken
        model.freeze_backbone() # Linear probing: freeze backbone, train only head
        print("Using Linear Probing (frozen backbone, trainable head).")
    elif peft_method == "full_ft":
        model.unfreeze_backbone() # Full Fine-tuning: unfreeze all parameters
        print("Using Full Fine-Tuning (all parameters trainable).")
    else:
        model.freeze_backbone() # Freeze backbone before applying PEFT modules
        model = get_peft_model(model, peft_method=peft_method, **config['peft']['kwargs'])
        print(f"Using PEFT method: {peft_method}.")
    
    model.to(device)

    # Get trainable parameters (PEFT modules + classification head or all for full_ft)
    trainable_params = model.get_trainable_parameters()
    # For VPT-Deep, the model is wrapped, so we need to ensure the head of the wrapped ViT is included
    if peft_method == "vpt_deep":
        trainable_params.extend(model.vit.head.parameters())
    
    optimizer = optim.AdamW(trainable_params, lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    # Load datasets based on scenario
    if config['scenario'] == "vtab_1k":
        data_loaders_map = get_vtab_datasets(
            image_size=config['data']['image_size'],
            batch_size=config['training']['batch_size'],
            num_workers=config['data']['num_workers'],
            seed=config['seed']
        )
    elif config['scenario'] == "many_shot":
        data_loaders_map = get_many_shot_datasets(
            image_size=config['data']['image_size'],
            batch_size=config['training']['batch_size'],
            num_workers=config['data']['num_workers'],
            seed=config['seed']
        )
    elif config['scenario'] == "robustness":
        data_loaders_map = get_robustness_datasets(
            image_size=config['data']['image_size'],
            batch_size=config['training']['batch_size'],
            num_workers=config['data']['num_workers'],
            seed=config['seed']
        )
    else:
        raise ValueError(f"Unknown scenario: {config['scenario']}")

    # Iterate over tasks (e.g., different VTAB-1K datasets, or a single many-shot/robustness task)
    results = {}
    for task_name, loaders in data_loaders_map.items():
        print(f"
Training and evaluating on task: {task_name}")
        
        # Reset classifier for the current task's number of classes
        model.reset_classifier(loaders['num_classes'])
        model.to(device) # Re-move to device after head reset

        # Re-initialize optimizer with potentially new trainable parameters after classifier reset
        trainable_params = model.get_trainable_parameters()
        if peft_method == "vpt_deep":
            trainable_params.extend(model.vit.head.parameters())
        optimizer = optim.AdamW(trainable_params, lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])

        best_val_accuracy = 0.0
        for epoch in range(config['training']['epochs']):
            model.train()
            running_loss = 0.0
            for inputs, labels in tqdm(loaders['train'], desc=f"Epoch {epoch+1}/{config['training']['epochs']} (Train)"):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(loaders['train'].dataset)
            print(f"Epoch {epoch+1} Train Loss: {epoch_loss:.4f}")

            # Evaluate on validation set (if available, e.g., VTAB-1K)
            if 'val' in loaders:
                val_loss, val_accuracy = evaluate_model(model, loaders['val'], device)
                print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    # Optionally save best model state here
            else: # If no explicit val set, use the test set for 'best' tracking in many-shot/robustness
                # This is a simplification; ideally, a separate dev set would be used
                val_loss, val_accuracy = evaluate_model(model, loaders['test'], device)
                print(f"Epoch {epoch+1} Test Loss: {val_loss:.4f}, Test Accuracy: {val_accuracy:.4f}")
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy

        # Final evaluation on the test set for the task
        test_loss, test_accuracy = evaluate_model(model, loaders['test'], device)
        print(f"Task {task_name} Final Test Loss: {test_loss:.4f}, Final Test Accuracy: {test_accuracy:.4f}")
        results[task_name] = {"test_accuracy": test_accuracy, "best_val_accuracy": best_val_accuracy}
    
    return results

if __name__ == '__main__':
    # Example configuration loading (replace with actual config management)
    # For simplicity, a basic config is defined here.
    # In a real setup, this would be loaded from a YAML file.
    config = {
        "scenario": "vtab_1k", # "vtab_1k", "many_shot", "robustness"
        "seed": 42,
        "model": {
            "name": "google/vit-base-patch16-224-in21k"
        },
        "peft": {
            "method": "lora", # Options: "none" (linear probing), "full_ft", "lora", "bitfit", "layernorm", "adapter", "vpt_deep"
            "kwargs": {"rank": 4, "lora_alpha": 1} # for lora
            # "kwargs": {"bottleneck_dim": 64} # for adapter
            # "kwargs": {"prompt_length": 10, "prompt_dropout": 0.0} # for vpt_deep
        },
        "dataset": {
            "num_classes": 100, # Placeholder, will be updated per task
        },
        "data": {
            "image_size": 224,
            "num_workers": 0 # Set to >0 for actual training
        },
        "training": {
            "epochs": 5,
            "batch_size": 32,
            "learning_rate": 5e-5,
            "weight_decay": 0.01
        }
    }

    # Dynamically adjust num_classes for the initial model if not handled by get_peft_model
    # This is handled per-task within the train_model loop, but for init of the base model.
    # For the `if __name__ == '__main__':` block, `num_classes` is a dummy.
    # The actual num_classes for each task is set during the loop.

    results = train_model(config)
    print("
--- Training Results ---")
    for task_name, res in results.items():
        print(f"Task: {task_name}, Test Accuracy: {res['test_accuracy']:.4f}, Best Val Accuracy: {res['best_val_accuracy']:.4f}")

