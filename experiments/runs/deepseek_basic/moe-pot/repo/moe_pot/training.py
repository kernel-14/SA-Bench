"""
Training and fine-tuning utilities for MoE-POT.

Implements:
- Pre-training with auto-regressive denoising objective
- One-cycle learning rate schedule
- Balanced data sampling across heterogeneous PDE datasets
- Noise injection for training stability
- Freezing router during fine-tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, List, Tuple
import math
import numpy as np


class PDEDataset(Dataset):
    """
    Dataset for PDE spatiotemporal data.
    
    Handles data from multiple PDE sources with:
    - Spatial resolution standardization (H=128)
    - Channel padding to unify variable counts
    - Mask channel for irregular geometries
    """
    
    def __init__(
        self,
        data: torch.Tensor,  # [N, T_total, C, H, W]
        T: int = 10,
        dataset_id: int = 0,
    ):
        self.data = data
        self.T = T
        self.dataset_id = dataset_id
        self.total_timesteps = data.shape[1]
        
        # Number of possible (input, target) pairs
        self.num_samples = max(0, self.total_timesteps - T)
        
    def __len__(self):
        return self.num_samples * self.data.shape[0]
    
    def __getitem__(self, idx):
        sample_idx = idx // self.num_samples
        start_t = idx % self.num_samples
        
        # Input: T consecutive frames
        x = self.data[sample_idx, start_t:start_t + self.T]  # [T, C, H, W]
        
        # Target: next frame after the T input frames
        y = self.data[sample_idx, start_t + self.T]  # [C, H, W]
        
        return x, y, self.dataset_id


class MultiPDEDataset(Dataset):
    """
    Dataset combining multiple PDE datasets with balanced sampling.
    
    Implements the balanced sampling strategy from Appendix B.1:
    p_k = w_k / (K * |D_k| * Σ_k w_k)
    
    All datasets have weight w_k = 1 by default (as specified in B.3).
    """
    
    def __init__(
        self,
        datasets: List[torch.Tensor],
        T: int = 10,
        weights: Optional[List[float]] = None,
        dataset_names: Optional[List[str]] = None,
    ):
        self.T = T
        self.dataset_names = dataset_names or [f"dataset_{i}" for i in range(len(datasets))]
        
        if weights is None:
            weights = [1.0] * len(datasets)
        
        # Create individual datasets
        self.sub_datasets = []
        for i, data in enumerate(datasets):
            self.sub_datasets.append(PDEDataset(data, T=T, dataset_id=i))
        
        # Compute sampling probabilities for balanced sampling
        K = len(datasets)
        total_weight = sum(weights)
        self.sample_probs = []
        self.cumulative_samples = [0]
        
        for k, w in enumerate(weights):
            num_samples = self.sub_datasets[k].__len__()
            prob = w / (K * num_samples * total_weight) if num_samples > 0 else 0
            self.sample_probs.append(prob)
            self.cumulative_samples.append(self.cumulative_samples[-1] + num_samples)
        
        # Normalize probabilities
        total_prob = sum(self.sample_probs)
        if total_prob > 0:
            self.sample_probs = [p / total_prob for p in self.sample_probs]
        
        self.total_samples = sum(d.__len__() for d in self.sub_datasets)
        
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        # Random sampling based on balanced probabilities
        # In practice, this is done at the DataLoader level
        # Here we provide direct access
        
        # Map flat index to dataset and sample
        dataset_idx = 0
        for i, cum in enumerate(self.cumulative_samples[1:], 1):
            if idx < cum:
                dataset_idx = i - 1
                sample_idx = idx - self.cumulative_samples[dataset_idx]
                break
        
        x, y, did = self.sub_datasets[dataset_idx][sample_idx]
        return x, y, did


class BalancedBatchSampler:
    """
    Batch sampler that implements balanced sampling across datasets.
    
    As described in Appendix B.1:
    The probability of sampling from dataset k is:
    p_k = w_k / (K * |D_k| * Σ_k w_k)
    """
    
    def __init__(
        self,
        dataset: MultiPDEDataset,
        batch_size: int,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __iter__(self):
        # Sample dataset indices based on probabilities
        K = len(self.dataset.sub_datasets)
        batch = []
        
        for _ in range(len(self.dataset) // self.batch_size):
            # Choose dataset according to balanced probability
            dataset_idx = np.random.choice(K, p=self.dataset.sample_probs)
            
            # Sample from that dataset
            sub_dataset = self.dataset.sub_datasets[dataset_idx]
            sample_idx = np.random.randint(0, len(sub_dataset))
            
            x, y, did = sub_dataset[sample_idx]
            batch.append((x, y, did))
            
            if len(batch) == self.batch_size:
                # Collate
                xs = torch.stack([b[0] for b in batch])
                ys = torch.stack([b[1] for b in batch])
                dids = torch.tensor([b[2] for b in batch])
                yield xs, ys, dids
                batch = []
    
    def __len__(self):
        return len(self.dataset) // self.batch_size


class NoiseInjection:
    """
    Noise injection for auto-regressive denoising pre-training.
    
    As described in Section 2.2 and Appendix B.1:
    ε ~ N(0, ε·||u^{<t}||·I)
    
    The noise is only added during pre-training, not fine-tuning.
    """
    
    def __init__(self, epsilon: float = 0.01):
        self.epsilon = epsilon
        
    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise scaled by the norm of the input."""
        if self.epsilon <= 0:
            return x
        
        # Compute norm of input
        norm = torch.norm(x)
        
        # Scale noise
        noise_std = self.epsilon * norm
        
        # Generate and add noise
        noise = torch.randn_like(x) * noise_std
        return x + noise


class OneCycleLR:
    """
    One-cycle learning rate scheduler.
    
    As described in Appendix B.3:
    - Pre-training: 1000 epochs, first 200 warm-up
    - Fine-tuning: 200 epochs, first 40 warm-up
    - Downstream: 500 epochs, first 100 warm-up
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float,
        total_epochs: int,
        warmup_epochs: int,
        min_lr_ratio: float = 0.01,
    ):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = max_lr * min_lr_ratio
        
    def get_lr(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            # Linear warmup
            return self.max_lr * epoch / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
    
    def step(self, epoch: int):
        lr = self.get_lr(epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


class MoEPOTTrainer:
    """
    Trainer for MoE-POT covering pre-training, fine-tuning, and downstream tasks.
    
    Implements the training procedures described in Section 5 and Appendix B.3.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        betas: Tuple[float, float] = (0.9, 0.9),
        noise_epsilon: float = 0.01,
        load_balance_weight: float = 0.1,
    ):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.noise_epsilon = noise_epsilon
        self.load_balance_weight = load_balance_weight
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
        )
        
        self.noise_injector = NoiseInjection(epsilon=noise_epsilon)
        
    def train_step(
        self,
        x: torch.Tensor,  # [B, T, C, H, W]
        y: torch.Tensor,  # [B, C, H, W]
        add_noise: bool = True,
        use_load_balance: bool = True,
    ) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        
        # Move to device
        x = x.to(self.device)
        y = y.to(self.device)
        
        # Add noise during pre-training
        if add_noise:
            x = self.noise_injector.add_noise(x)
        
        # Forward pass
        pred = self.model(x)
        
        # Primary loss: L2 reconstruction loss
        # L = Σ_t ||G_w(u^{<t} + ε) - u^t||_2^2
        recon_loss = F.mse_loss(pred, y)
        
        # Load balancing loss
        if use_load_balance:
            balance_loss = self.model.get_load_balancing_loss(x)
            total_loss = recon_loss + self.load_balance_weight * balance_loss
        else:
            balance_loss = torch.tensor(0.0)
            total_loss = recon_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Compute L2RE for logging
        with torch.no_grad():
            l2re = self.model.compute_l2_relative_error(pred, y)
        
        return {
            'loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'balance_loss': balance_loss.item() if use_load_balance else 0.0,
            'l2re': l2re.item(),
        }
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """Evaluate model on a dataset."""
        self.model.eval()
        
        total_l2re = 0.0
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (x, y, _) in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
                
            x = x.to(self.device)
            y = y.to(self.device)
            
            pred = self.model(x)
            
            l2re = self.model.compute_l2_relative_error(pred, y)
            loss = F.mse_loss(pred, y)
            
            total_l2re += l2re.item()
            total_loss += loss.item()
            num_batches += 1
        
        return {
            'l2re': total_l2re / num_batches,
            'loss': total_loss / num_batches,
        }
    
    def pre_train(
        self,
        train_loader: DataLoader,
        num_epochs: int = 1000,
        warmup_epochs: int = 200,
        save_path: Optional[str] = None,
        log_interval: int = 10,
    ):
        """
        Pre-training loop as described in Appendix B.3.
        
        - 1000 epochs with One-cycle LR schedule
        - 200 warmup epochs
        - Learning rate 1e-3
        - Noise injection enabled
        - Load balancing enabled
        """
        scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            total_epochs=num_epochs,
            warmup_epochs=warmup_epochs,
        )
        
        for epoch in range(num_epochs):
            lr = scheduler.step(epoch)
            
            epoch_loss = 0.0
            epoch_l2re = 0.0
            num_batches = 0
            
            for x, y, _ in train_loader:
                metrics = self.train_step(x, y, add_noise=True, use_load_balance=True)
                epoch_loss += metrics['loss']
                epoch_l2re += metrics['l2re']
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            avg_l2re = epoch_l2re / num_batches
            
            if epoch % log_interval == 0:
                print(f"Epoch {epoch:4d}/{num_epochs} | LR: {lr:.6f} | "
                      f"Loss: {avg_loss:.6f} | L2RE: {avg_l2re:.6f}")
        
        if save_path:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': num_epochs,
            }, save_path)
    
    def fine_tune(
        self,
        train_loader: DataLoader,
        num_epochs: int = 200,
        warmup_epochs: int = 40,
        freeze_router: bool = True,
        save_path: Optional[str] = None,
        log_interval: int = 10,
    ):
        """
        Fine-tuning loop as described in Section 5.1 and Appendix B.3.
        
        Key differences from pre-training:
        - Router-gating network is frozen
        - No noise injection
        - 200 epochs with One-cycle LR
        - 40 warmup epochs
        """
        # Freeze router-gating network parameters
        if freeze_router:
            for block in self.model.blocks:
                for param in block.moe.router.parameters():
                    param.requires_grad = False
        
        scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            total_epochs=num_epochs,
            warmup_epochs=warmup_epochs,
        )
        
        for epoch in range(num_epochs):
            lr = scheduler.step(epoch)
            
            epoch_loss = 0.0
            epoch_l2re = 0.0
            num_batches = 0
            
            for x, y, _ in train_loader:
                # No noise injection during fine-tuning
                metrics = self.train_step(x, y, add_noise=False, use_load_balance=False)
                epoch_loss += metrics['loss']
                epoch_l2re += metrics['l2re']
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            avg_l2re = epoch_l2re / num_batches
            
            if epoch % log_interval == 0:
                print(f"Fine-tune Epoch {epoch:4d}/{num_epochs} | LR: {lr:.6f} | "
                      f"Loss: {avg_loss:.6f} | L2RE: {avg_l2re:.6f}")
        
        if save_path:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': num_epochs,
            }, save_path)
    
    def train_downstream(
        self,
        train_loader: DataLoader,
        num_epochs: int = 500,
        warmup_epochs: int = 100,
        save_path: Optional[str] = None,
        log_interval: int = 10,
    ):
        """
        Downstream task training as described in Section 5.2 and Appendix B.3.
        
        - 500 epochs with One-cycle LR
        - 100 warmup epochs
        - No noise injection
        """
        scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            total_epochs=num_epochs,
            warmup_epochs=warmup_epochs,
        )
        
        for epoch in range(num_epochs):
            lr = scheduler.step(epoch)
            
            epoch_loss = 0.0
            epoch_l2re = 0.0
            num_batches = 0
            
            for x, y, _ in train_loader:
                metrics = self.train_step(x, y, add_noise=False, use_load_balance=False)
                epoch_loss += metrics['loss']
                epoch_l2re += metrics['l2re']
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            avg_l2re = epoch_l2re / num_batches
            
            if epoch % log_interval == 0:
                print(f"Downstream Epoch {epoch:4d}/{num_epochs} | "
                      f"Loss: {avg_loss:.6f} | L2RE: {avg_l2re:.6f}")
        
        if save_path:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': num_epochs,
            }, save_path)
