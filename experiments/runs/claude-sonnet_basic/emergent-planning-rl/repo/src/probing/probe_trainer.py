"""
Probe training and evaluation utilities.

From the paper (Appendix D.1):
- All probes trained for 10 epochs
- AdamW optimizer
- Batch size: 16
- Learning rate: 0.001
- Weight decay: 0.001
- Implemented as convolutions

Training dataset: 3000 episodes from Boxoban unfiltered training set
Test dataset: 1000 episodes from Boxoban unfiltered validation set
5 unique initialization seeds per probe
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict, Optional
import os

from .linear_probe import LinearProbe, BaselineProbe, compute_macro_f1, compute_class_metrics


class ConceptDataset(Dataset):
    """
    Dataset for training linear probes.
    
    Each sample consists of:
    - Cell state activations at a specific layer
    - Concept labels for each grid square
    """
    
    def __init__(
        self,
        cell_states: List[np.ndarray],  # List of (hidden_channels, H, W) arrays
        labels: List[np.ndarray],        # List of (H, W) label arrays
    ):
        """
        Args:
            cell_states: List of cell state arrays
            labels: List of label arrays
        """
        self.cell_states = [torch.FloatTensor(cs) for cs in cell_states]
        self.labels = [torch.LongTensor(l) for l in labels]
    
    def __len__(self):
        return len(self.cell_states)
    
    def __getitem__(self, idx):
        return self.cell_states[idx], self.labels[idx]


class ObsDataset(Dataset):
    """Dataset for baseline probes using raw observations."""
    
    def __init__(
        self,
        observations: List[np.ndarray],  # List of (H, W, obs_channels) arrays
        labels: List[np.ndarray],         # List of (H, W) label arrays
    ):
        # Convert to channel-first format
        self.observations = [torch.FloatTensor(obs).permute(2, 0, 1) for obs in observations]
        self.labels = [torch.LongTensor(l) for l in labels]
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        return self.observations[idx], self.labels[idx]


def train_probe(
    probe: nn.Module,
    train_dataset: Dataset,
    num_epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    weight_decay: float = 0.001,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> List[float]:
    """
    Train a linear probe.
    
    Args:
        probe: LinearProbe or BaselineProbe to train
        train_dataset: Training dataset
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate for AdamW
        weight_decay: Weight decay for AdamW
        device: Device to train on
        verbose: Whether to print training progress
        
    Returns:
        List of training losses per epoch
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    probe = probe.to(device)
    probe.train()
    
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_inputs, batch_labels in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            logits = probe(batch_inputs)  # (B, num_classes, H, W)
            
            # Reshape for cross entropy: (B, num_classes, H, W) -> (B*H*W, num_classes)
            B, C, H, W = logits.shape
            logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)
            labels_flat = batch_labels.reshape(-1)
            
            loss = criterion(logits_flat, labels_flat)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        losses.append(avg_loss)
        
        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return losses


def evaluate_probe(
    probe: nn.Module,
    test_dataset: Dataset,
    batch_size: int = 16,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Evaluate a trained probe.
    
    Args:
        probe: Trained probe
        test_dataset: Test dataset
        batch_size: Batch size
        device: Device to evaluate on
        
    Returns:
        Dict with 'macro_f1' and per-class metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    probe = probe.to(device)
    probe.eval()
    
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_inputs, batch_labels in dataloader:
            batch_inputs = batch_inputs.to(device)
            
            logits = probe(batch_inputs)  # (B, num_classes, H, W)
            predictions = logits.argmax(dim=1)  # (B, H, W)
            
            all_predictions.append(predictions.cpu().numpy().flatten())
            all_labels.append(batch_labels.numpy().flatten())
    
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    
    macro_f1 = compute_macro_f1(all_predictions, all_labels)
    class_metrics = compute_class_metrics(all_predictions, all_labels)
    
    return {
        'macro_f1': macro_f1,
        'class_metrics': class_metrics,
        'predictions': all_predictions,
        'labels': all_labels,
    }


def train_and_evaluate_probes(
    cell_states_train: List[np.ndarray],
    labels_train: List[np.ndarray],
    cell_states_test: List[np.ndarray],
    labels_test: List[np.ndarray],
    hidden_channels: int = 32,
    num_classes: int = 5,
    probe_sizes: List[int] = [1, 3],
    num_seeds: int = 5,
    num_epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    weight_decay: float = 0.001,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> Dict:
    """
    Train and evaluate probes with multiple seeds.
    
    Args:
        cell_states_train: Training cell states
        labels_train: Training labels
        cell_states_test: Test cell states
        labels_test: Test labels
        hidden_channels: Number of hidden channels
        num_classes: Number of concept classes
        probe_sizes: List of probe sizes to evaluate
        num_seeds: Number of random seeds
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: Weight decay
        device: Device to use
        verbose: Whether to print progress
        
    Returns:
        Dict with results for each probe size
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset = ConceptDataset(cell_states_train, labels_train)
    test_dataset = ConceptDataset(cell_states_test, labels_test)
    
    results = {}
    
    for probe_size in probe_sizes:
        seed_results = []
        
        for seed in range(num_seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            probe = LinearProbe(
                hidden_channels=hidden_channels,
                num_classes=num_classes,
                probe_size=probe_size,
            )
            
            train_probe(
                probe, train_dataset,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                device=device,
                verbose=verbose,
            )
            
            eval_results = evaluate_probe(probe, test_dataset, batch_size=batch_size, device=device)
            seed_results.append(eval_results['macro_f1'])
            
            if verbose:
                print(f"Probe size {probe_size}x{probe_size}, Seed {seed}: Macro F1 = {eval_results['macro_f1']:.4f}")
        
        results[probe_size] = {
            'macro_f1_mean': np.mean(seed_results),
            'macro_f1_std': np.std(seed_results),
            'macro_f1_per_seed': seed_results,
        }
    
    return results
