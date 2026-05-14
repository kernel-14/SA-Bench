"""
Training utilities for PEFT methods.
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np


class Trainer:
    """
    Trainer for PEFT methods.
    
    Uses AdamW optimizer with cosine decay learning rate scheduler,
    following the paper's experimental setup.
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        num_classes,
        lr=1e-3,
        weight_decay=1e-4,
        num_epochs=100,
        device='cuda',
        output_dir='./output',
        experiment_name='experiment',
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.device = device
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Setup optimizer and scheduler
        self.optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy
    
    def train(self, verbose=True):
        """Full training loop."""
        best_model_path = os.path.join(self.output_dir, f'{self.experiment_name}_best.pth')
        
        for epoch in range(self.num_epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            self.scheduler.step()
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, best_model_path)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.num_epochs}] '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% '
                      f'(Best: {self.best_val_acc:.2f}% @ epoch {self.best_epoch+1})')
        
        # Load best model
        checkpoint = torch.load(best_model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        return self.best_val_acc
    
    def save_checkpoint(self, path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
        }, path)
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'best_val_acc' in checkpoint:
            self.best_val_acc = checkpoint['best_val_acc']
