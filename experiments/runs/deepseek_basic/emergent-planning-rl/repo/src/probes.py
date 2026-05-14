"""
Linear Probes for Concept-Based Interpretability.

Implements the probing methodology from the paper:
- 1x1 and 3x3 (also 5x5, 7x7) spatial probes
- Trained to predict concept classes C_A and C_B from cell state activations
- Logistic regression using AdamW optimizer
- Macro F1 evaluation metric

Concepts:
- C_A (Agent Approach Direction): For each square, encodes whether the agent will move
  onto that square in the future, and if so, from which direction.
  Classes: {UP, DOWN, LEFT, RIGHT, NEVER}
  
- C_B (Box Push Direction): For each square, encodes whether a box will be pushed off
  that square in the future, and if so, in which direction.
  Classes: {UP, DOWN, LEFT, RIGHT, NEVER}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import warnings


class ConceptClasses:
    """Class indices for planning-relevant concepts."""
    NEVER = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    
    NUM_CLASSES = 5
    
    @staticmethod
    def class_names():
        return ['NEVER', 'UP', 'DOWN', 'LEFT', 'RIGHT']
    
    @staticmethod
    def direction_to_class(dy: int, dx: int) -> int:
        """Convert a movement delta to concept class."""
        if dy == -1 and dx == 0:
            return ConceptClasses.UP
        elif dy == 1 and dx == 0:
            return ConceptClasses.DOWN
        elif dy == 0 and dx == -1:
            return ConceptClasses.LEFT
        elif dy == 0 and dx == 1:
            return ConceptClasses.RIGHT
        else:
            return ConceptClasses.NEVER


class LinearProbe(nn.Module):
    """
    A linear probe implemented as a convolution for spatially-localized predictions.
    
    For 1x1 probes: uses 1x1 convolution (each square independently)
    For KxK probes: uses KxK convolution with appropriate padding
    
    Each probe predicts one of 5 classes (NEVER, UP, DOWN, LEFT, RIGHT).
    
    Args:
        in_channels: Number of input channels (e.g., 32 for DRC cell state)
        kernel_size: Size of spatial context (1 for 1x1, 3 for 3x3, etc.)
        num_classes: Number of concept classes (default 5)
    """
    def __init__(
        self, 
        in_channels: int = 32, 
        kernel_size: int = 1, 
        num_classes: int = 5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.num_classes = num_classes
        
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=kernel_size, padding=padding)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Cell state activations (B, C, H, W)
        
        Returns:
            logits (B, num_classes, H, W)
        """
        return self.conv(x)


class GlobalProbe(nn.Module):
    """
    A global probe that receives the entire cell state as input.
    Used for probing global concepts like 'Action in n steps'.
    
    As described in Appendix D.5, these have ~10,240 parameters.
    """
    def __init__(self, in_features: int, num_classes: int = 5):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Flattened activations (B, in_features)
        
        Returns:
            logits (B, num_classes)
        """
        return self.linear(x)


def train_probe_pytorch(
    probe: nn.Module,
    train_activations: torch.Tensor,
    train_labels: torch.Tensor,
    test_activations: Optional[torch.Tensor] = None,
    test_labels: Optional[torch.Tensor] = None,
    num_epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    weight_decay: float = 0.001,
    device: str = 'cpu',
    verbose: bool = False,
) -> Dict:
    """
    Train a linear probe using PyTorch with AdamW optimizer.
    
    Follows the paper's training details (Appendix D.1):
    - 10 epochs
    - AdamW optimizer
    - Batch size 16
    - Learning rate 0.001
    - Weight decay 0.001
    
    Args:
        probe: LinearProbe or GlobalProbe module
        train_activations: Training activations
        train_labels: Training labels
        test_activations: Test activations (optional)
        test_labels: Test labels (optional)
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: Weight decay
        device: Torch device
    
    Returns:
        dict with training history and test metrics
    """
    probe = probe.to(device)
    train_activations = train_activations.to(device)
    train_labels = train_labels.to(device)
    
    if test_activations is not None:
        test_activations = test_activations.to(device)
        test_labels = test_labels.to(device)
    
    optimizer = torch.optim.AdamW(
        probe.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()
    
    n_samples = train_activations.shape[0]
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    
    for epoch in range(num_epochs):
        # Shuffle
        perm = torch.randperm(n_samples, device=device)
        
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            x_batch = train_activations[idx]
            y_batch = train_labels[idx]
            
            optimizer.zero_grad()
            logits = probe(x_batch)
            
            # Handle spatial probes: reshape to (B*H*W, num_classes)
            if logits.dim() == 4:
                B, C, H, W = logits.shape
                logits = logits.permute(0, 2, 3, 1).reshape(-1, C)
                y_batch = y_batch.reshape(-1)
            
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(idx)
            pred = logits.argmax(dim=-1)
            correct += (pred == y_batch).sum().item()
            total += y_batch.numel()
        
        avg_loss = epoch_loss / n_samples
        train_acc = correct / total
        
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        
        if test_activations is not None:
            with torch.no_grad():
                test_logits = probe(test_activations)
                if test_logits.dim() == 4:
                    B, C, H, W = test_logits.shape
                    test_logits = test_logits.permute(0, 2, 3, 1).reshape(-1, C)
                    t_labels = test_labels.reshape(-1)
                else:
                    t_labels = test_labels
                test_pred = test_logits.argmax(dim=-1)
                test_acc = (test_pred == t_labels).float().mean().item()
                history['test_acc'].append(test_acc)
        
        if verbose:
            msg = f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Acc: {train_acc:.4f}"
            if test_activations is not None:
                msg += f", Test Acc: {test_acc:.4f}"
            print(msg)
    
    # Final evaluation
    results = {'history': history}
    
    if test_activations is not None:
        with torch.no_grad():
            test_logits = probe(test_activations)
            if test_logits.dim() == 4:
                B, C, H, W = test_logits.shape
                test_logits = test_logits.permute(0, 2, 3, 1).reshape(-1, C)
                t_labels_flat = test_labels.reshape(-1)
            else:
                t_labels_flat = test_labels
            
            test_pred = test_logits.argmax(dim=-1).cpu().numpy()
            t_labels_np = t_labels_flat.cpu().numpy()
            
            # Macro F1
            results['macro_f1'] = f1_score(t_labels_np, test_pred, average='macro')
            
            # Per-class metrics
            results['per_class'] = {}
            for cls_idx in range(probe.num_classes):
                cls_name = ConceptClasses.class_names()[cls_idx]
                cls_true = (t_labels_np == cls_idx)
                cls_pred = (test_pred == cls_idx)
                
                tp = np.sum(cls_true & cls_pred)
                fp = np.sum(~cls_true & cls_pred)
                fn = np.sum(cls_true & ~cls_pred)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                results['per_class'][cls_name] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                }
            
            results['accuracy'] = (test_pred == t_labels_np).mean()
    
    return results


def get_probe_vectors(probe: LinearProbe) -> Dict[int, np.ndarray]:
    """
    Extract the class vectors w_k from a trained 1x1 probe.
    
    For a 1x1 probe (kernel_size=1), the convolution weight (C_out, C_in, 1, 1)
    gives us vectors w_k for each class k.
    
    Returns:
        dict mapping class index to vector w_k of shape (in_channels,)
    """
    if probe.kernel_size != 1:
        raise ValueError("Probe vectors only defined for 1x1 probes")
    
    weight = probe.conv.weight.data.cpu().numpy()  # (num_classes, in_channels, 1, 1)
    vectors = {}
    for k in range(probe.num_classes):
        vectors[k] = weight[k, :, 0, 0].copy()
    return vectors
