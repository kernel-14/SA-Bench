"""
Training utilities for neural operators.

Implements:
1. Standard training (from scratch)
2. Pre-training on multiple physics
3. Fine-tuning with frozen backbone (adapter-based)
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Callable, Any

from .metrics import compute_metrics


class Trainer:
    """
    Standard trainer for neural operators.
    
    Supports both training from scratch and fine-tuning with frozen backbone.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        loss_fn: Optional[Callable] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.loss_fn = loss_fn or nn.MSELoss()
        
        if optimizer is None:
            self.optimizer = optim.Adam(model.parameters(), lr=1e-3)
        else:
            self.optimizer = optimizer
        
        self.scheduler = scheduler
        self.history = {"train_loss": [], "val_loss": [], "val_mse": [], "val_nmae": []}

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        epoch_start = time.time()
        
        for batch in dataloader:
            if len(batch) == 2:
                inputs, targets = batch
            else:
                inputs, targets = batch[0], batch[1]
            
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        epoch_time = time.time() - epoch_start
        
        return {
            "loss": total_loss / n_batches,
            "epoch_time": epoch_time,
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate on a dataset."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        for batch in dataloader:
            if len(batch) == 2:
                inputs, targets = batch
            else:
                inputs, targets = batch[0], batch[1]
            
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            
            total_loss += loss.item()
            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())
        
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = compute_metrics(all_preds, all_targets)
        metrics["loss"] = total_loss / len(dataloader)
        
        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        n_epochs: int = 100,
        verbose: bool = True,
        save_path: Optional[str] = None,
    ) -> Dict[str, List]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            n_epochs: Number of epochs
            verbose: Print progress
            save_path: Path to save best model
        
        Returns:
            Training history
        """
        best_val_loss = float('inf')
        
        for epoch in range(n_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_metrics["loss"])
            
            # Validate
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_mse"].append(val_metrics["mse"])
                self.history["val_nmae"].append(val_metrics["nmae"])
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch {epoch+1}/{n_epochs} | "
                        f"Train Loss: {train_metrics['loss']:.6f} | "
                        f"Val MSE: {val_metrics['mse']:.2e} | "
                        f"Val NMAE: {val_metrics['nmae']*100:.4f}% | "
                        f"Time: {train_metrics['epoch_time']:.2f}s"
                    )
                
                # Save best model
                if save_path and val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    torch.save(self.model.state_dict(), save_path)
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch {epoch+1}/{n_epochs} | "
                        f"Train Loss: {train_metrics['loss']:.6f} | "
                        f"Time: {train_metrics['epoch_time']:.2f}s"
                    )
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    if val_loader is not None:
                        self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()
        
        return self.history


class MultiPhysicsTrainer:
    """
    Trainer for multi-physics pretraining.
    
    Handles simultaneous training on multiple PDE problems with different
    input function sets using the adapter-based approach.
    
    From the paper:
    "In the pre-training phase the entire parameters set (theta_P1, ..., theta_PN,
    theta_F, theta_L1, ..., theta_LN) is subject to optimization."
    """

    def __init__(
        self,
        models: List[nn.Module],
        shared_backbone: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        loss_fn: Optional[Callable] = None,
    ):
        """
        Args:
            models: List of full models (one per physics), sharing backbone
            shared_backbone: The shared backbone module
            optimizer: Optimizer (if None, uses Adam on all parameters)
            device: Device to use
            loss_fn: Loss function
        """
        self.models = [m.to(device) for m in models]
        self.shared_backbone = shared_backbone
        self.device = device
        self.loss_fn = loss_fn or nn.MSELoss()
        
        if optimizer is None:
            # Collect all parameters from all models
            all_params = []
            for model in models:
                all_params.extend(model.parameters())
            self.optimizer = optim.Adam(all_params, lr=1e-3)
        else:
            self.optimizer = optimizer
        
        self.history = {"train_loss": [], "val_losses": [[] for _ in models]}

    def train_epoch(self, dataloaders: List[DataLoader]) -> Dict[str, float]:
        """Train for one epoch on all physics simultaneously."""
        for model in self.models:
            model.train()
        
        total_loss = 0.0
        n_batches = 0
        epoch_start = time.time()
        
        # Interleave batches from different physics
        iterators = [iter(dl) for dl in dataloaders]
        active = list(range(len(dataloaders)))
        
        while active:
            for i in list(active):
                try:
                    batch = next(iterators[i])
                    if len(batch) == 2:
                        inputs, targets = batch
                    else:
                        inputs, targets = batch[0], batch[1]
                    
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    self.optimizer.zero_grad()
                    outputs = self.models[i](inputs)
                    loss = self.loss_fn(outputs, targets)
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(
                        self.models[i].parameters(), max_norm=1.0
                    )
                    
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    n_batches += 1
                except StopIteration:
                    active.remove(i)
        
        epoch_time = time.time() - epoch_start
        
        return {
            "loss": total_loss / max(n_batches, 1),
            "epoch_time": epoch_time,
        }

    def train(
        self,
        train_loaders: List[DataLoader],
        val_loaders: Optional[List[DataLoader]] = None,
        n_epochs: int = 100,
        verbose: bool = True,
        save_paths: Optional[List[str]] = None,
    ) -> Dict:
        """Full multi-physics training loop."""
        for epoch in range(n_epochs):
            train_metrics = self.train_epoch(train_loaders)
            self.history["train_loss"].append(train_metrics["loss"])
            
            if val_loaders is not None:
                val_metrics_list = []
                for i, (model, val_loader) in enumerate(zip(self.models, val_loaders)):
                    trainer = Trainer(model, device=self.device, loss_fn=self.loss_fn)
                    val_metrics = trainer.evaluate(val_loader)
                    val_metrics_list.append(val_metrics)
                    self.history["val_losses"][i].append(val_metrics["loss"])
                
                if verbose and (epoch + 1) % 10 == 0:
                    val_str = " | ".join([
                        f"Physics {i} NMAE: {m['nmae']*100:.4f}%"
                        for i, m in enumerate(val_metrics_list)
                    ])
                    print(
                        f"Epoch {epoch+1}/{n_epochs} | "
                        f"Train Loss: {train_metrics['loss']:.6f} | "
                        f"{val_str} | "
                        f"Time: {train_metrics['epoch_time']:.2f}s"
                    )
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch {epoch+1}/{n_epochs} | "
                        f"Train Loss: {train_metrics['loss']:.6f} | "
                        f"Time: {train_metrics['epoch_time']:.2f}s"
                    )
        
        return self.history
