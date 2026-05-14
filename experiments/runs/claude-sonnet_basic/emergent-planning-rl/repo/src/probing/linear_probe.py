"""
Linear probe implementation for concept-based interpretability.

Linear probes are linear classifiers trained to predict concept classes
from network activations. They are used to determine if a network linearly
represents specific concepts.

From the paper:
- 1x1 probes: take as input just the activations at position (x,y)
  - 160 parameters (32 channels * 5 classes)
- 3x3 probes: take as input the 3x3 patch of activations around (x,y)
  - 1440 parameters (32 * 9 * 5)

Probes are trained using logistic regression with AdamW optimizer.
Training details (Appendix D.1):
- 10 epochs
- AdamW optimizer
- Batch size: 16
- Learning rate: 0.001
- Weight decay: 0.001
- Implemented as convolutions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict


NUM_CLASSES = 5  # NEVER, UP, DOWN, LEFT, RIGHT
GRID_SIZE = 8
HIDDEN_CHANNELS = 32


class LinearProbe(nn.Module):
    """
    Linear probe for predicting concept classes from cell state activations.
    
    Implemented as a convolution for efficiency (as described in paper).
    
    For 1x1 probes: Conv2d(hidden_channels, num_classes, kernel_size=1)
    For 3x3 probes: Conv2d(hidden_channels, num_classes, kernel_size=3, padding=1)
    """
    
    def __init__(
        self, 
        hidden_channels: int = HIDDEN_CHANNELS,
        num_classes: int = NUM_CLASSES,
        probe_size: int = 1,  # 1 for 1x1, 3 for 3x3
    ):
        """
        Args:
            hidden_channels: Number of input channels (from cell state)
            num_classes: Number of concept classes
            probe_size: Size of the probe (1 or 3)
        """
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.probe_size = probe_size
        
        padding = probe_size // 2
        self.conv = nn.Conv2d(
            hidden_channels, 
            num_classes, 
            kernel_size=probe_size, 
            padding=padding,
            bias=True,
        )
    
    def forward(self, cell_state: torch.Tensor) -> torch.Tensor:
        """
        Predict concept classes from cell state.
        
        Args:
            cell_state: Cell state tensor (B, hidden_channels, H, W)
            
        Returns:
            Logits tensor (B, num_classes, H, W)
        """
        return self.conv(cell_state)
    
    def predict(self, cell_state: torch.Tensor) -> torch.Tensor:
        """
        Predict concept class labels.
        
        Args:
            cell_state: Cell state tensor (B, hidden_channels, H, W)
            
        Returns:
            Predicted class labels (B, H, W)
        """
        logits = self.forward(cell_state)
        return logits.argmax(dim=1)
    
    def get_class_vectors(self) -> torch.Tensor:
        """
        Get the weight vectors for each class.
        
        For 1x1 probes, these are the w_k vectors used for interventions.
        
        Returns:
            Weight vectors (num_classes, hidden_channels) for 1x1 probes
        """
        if self.probe_size == 1:
            # Conv weight shape: (num_classes, hidden_channels, 1, 1)
            return self.conv.weight.squeeze(-1).squeeze(-1)  # (num_classes, hidden_channels)
        else:
            raise ValueError("Class vectors only defined for 1x1 probes")
    
    def get_class_vector(self, class_idx: int) -> torch.Tensor:
        """
        Get the weight vector for a specific class.
        
        This is the w_k vector used for interventions:
        g'_{x,y} <- g_{x,y} + w_k
        
        Args:
            class_idx: Index of the class
            
        Returns:
            Weight vector (hidden_channels,) for 1x1 probes
        """
        if self.probe_size == 1:
            return self.conv.weight[class_idx].squeeze(-1).squeeze(-1)
        else:
            raise ValueError("Class vectors only defined for 1x1 probes")


class BaselineProbe(nn.Module):
    """
    Baseline probe that receives the raw observation as input.
    
    Used to compare against cell state probes to assess whether
    probes' abilities are due to internal representations or
    the probes learning to predict concepts themselves.
    """
    
    def __init__(
        self,
        obs_channels: int = 7,
        num_classes: int = NUM_CLASSES,
        probe_size: int = 1,
    ):
        super().__init__()
        self.obs_channels = obs_channels
        self.num_classes = num_classes
        self.probe_size = probe_size
        
        padding = probe_size // 2
        self.conv = nn.Conv2d(
            obs_channels,
            num_classes,
            kernel_size=probe_size,
            padding=padding,
            bias=True,
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: Observation tensor (B, obs_channels, H, W) - channel first
            
        Returns:
            Logits (B, num_classes, H, W)
        """
        return self.conv(obs)
    
    def predict(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict class labels."""
        return self.forward(obs).argmax(dim=1)


def compute_macro_f1(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> float:
    """
    Compute macro F1 score.
    
    Macro F1 is used instead of accuracy due to class imbalance
    (many squares are labeled NEVER).
    
    Args:
        predictions: Predicted class labels (N,)
        labels: True class labels (N,)
        num_classes: Number of classes
        
    Returns:
        Macro F1 score
    """
    f1_scores = []
    
    for c in range(num_classes):
        tp = np.sum((predictions == c) & (labels == c))
        fp = np.sum((predictions == c) & (labels != c))
        fn = np.sum((predictions != c) & (labels == c))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        f1_scores.append(f1)
    
    return np.mean(f1_scores)


def compute_class_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> Dict[int, Dict[str, float]]:
    """
    Compute per-class precision, recall, and F1.
    
    Args:
        predictions: Predicted class labels (N,)
        labels: True class labels (N,)
        num_classes: Number of classes
        
    Returns:
        Dict mapping class index to {'precision', 'recall', 'f1'}
    """
    metrics = {}
    
    for c in range(num_classes):
        tp = np.sum((predictions == c) & (labels == c))
        fp = np.sum((predictions == c) & (labels != c))
        fn = np.sum((predictions != c) & (labels == c))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        metrics[c] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
    
    return metrics
