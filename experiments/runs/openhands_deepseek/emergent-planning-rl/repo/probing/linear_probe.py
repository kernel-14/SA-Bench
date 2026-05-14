"""
Linear probes for decoding concept representations from DRC agent cell states.

Probes are trained using logistic regression (cross-entropy loss) with the
AdamW optimizer, as described in Appendix D.1.

Probe variants:
- 1x1 probes: use only the cell state at (x, y) to predict concept at (x, y)
- 3x3 probes: use a 3x3 patch of cell state around (x, y)
- Larger probes: 5x5 and 7x7 for comparison

All probes are implemented as nn.Conv2d with appropriate kernel sizes.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple
from collections import defaultdict

from sklearn.metrics import f1_score, precision_score, recall_score


class LinearProbe(nn.Module):
    """
    Linear probe implemented as a Conv2d layer.

    For 1x1 probes: Conv2d(in_channels, num_classes, kernel_size=1)
    For KxK probes: Conv2d(in_channels, num_classes, kernel_size=K, padding=K//2)

    The output is (B, num_classes, H, W) where each spatial position predicts
    the concept class for the corresponding Sokoban square.
    """

    def __init__(
        self,
        in_channels: int = 32,
        num_classes: int = 5,
        kernel_size: int = 1,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels, num_classes,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: cell state (B, C, H, W)

        Returns:
            logits: (B, num_classes, H, W)
        """
        return self.conv(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return class predictions."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=1)


def train_probe(
    probe: LinearProbe,
    train_states: torch.Tensor,
    train_labels: torch.Tensor,
    test_states: Optional[torch.Tensor] = None,
    test_labels: Optional[torch.Tensor] = None,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    weight_decay: float = 0.001,
    device: str = "cpu",
    class_weights: Optional[torch.Tensor] = None,
) -> dict:
    """
    Train a linear probe and return evaluation metrics.

    Args:
        probe: LinearProbe module
        train_states: (N, C, H, W) cell state activations
        train_labels: (N, H, W) concept class labels
        test_states, test_labels: optional test data
        epochs, batch_size, lr, weight_decay: training hyperparameters
        device: torch device
        class_weights: optional class weights for imbalanced data

    Returns:
        metrics: dict of test metrics (macro F1, per-class F1, etc.)
    """
    probe = probe.to(device)
    train_states = train_states.to(device)
    train_labels = train_labels.to(device)

    dataset = TensorDataset(train_states, train_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    for epoch in range(epochs):
        probe.train()
        total_loss = 0.0
        for batch_states, batch_labels in dataloader:
            optimizer.zero_grad()
            logits = probe(batch_states)  # (B, num_classes, H, W)

            # Reshape for cross-entropy: (B*H*W, num_classes) vs (B*H*W)
            B, C, H, W = logits.shape
            logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)
            labels_flat = batch_labels.reshape(-1)

            loss = F.cross_entropy(logits_flat, labels_flat, weight=class_weights)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    # Evaluate
    probe.eval()
    metrics = {}

    if test_states is not None and test_labels is not None:
        test_states = test_states.to(device)
        with torch.no_grad():
            test_logits = probe(test_states)
            preds = torch.argmax(test_logits, dim=1)  # (N, H, W)

        preds_np = preds.cpu().numpy().flatten()
        labels_np = test_labels.cpu().numpy().flatten()

        # Mask out ignore indices? For now use all
        metrics["macro_f1"] = f1_score(labels_np, preds_np, average="macro")
        metrics["accuracy"] = (preds_np == labels_np).mean()

        # Per-class metrics
        for cls_idx in range(probe.num_classes):
            cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else str(cls_idx)
            cls_mask = labels_np == cls_idx
            if cls_mask.sum() > 0:
                y_true_cls = (labels_np == cls_idx).astype(int)
                y_pred_cls = (preds_np == cls_idx).astype(int)
                metrics[f"f1_{cls_name}"] = f1_score(y_true_cls, y_pred_cls)
                metrics[f"precision_{cls_name}"] = precision_score(y_true_cls, y_pred_cls)
                metrics[f"recall_{cls_name}"] = recall_score(y_true_cls, y_pred_cls)

    return metrics


CLASS_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "NEVER"]
