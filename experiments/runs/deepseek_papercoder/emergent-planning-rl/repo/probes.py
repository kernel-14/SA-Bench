## probes.py
"""
Linear probing and training pipeline for the interpretability experiments.

Provides:
- `LinearProbe`: a 1×1 or k×k convolutional linear classifier that maps internal
  cell‑state activations (or raw observations) to per‑spatial‑location concept class
  logits.
- `ProbeTrainer`: orchestrates multi‑seed training, evaluation (macro F1, per‑class
  metrics), and saving of probes for later visualisation and causal interventions.

All hyperparameters are taken from the `probing` section of `config.yaml`.
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# ------------------------------------------------------------------------------
# Project‑internal imports (avoids circular dependencies)
# ------------------------------------------------------------------------------
from utils import set_seed
from dataset import ProbeDataset


# ------------------------------------------------------------------------------
# 1.  LinearProbe
# ------------------------------------------------------------------------------
class LinearProbe(nn.Module):
    """
    Linear convolutional probe that predicts, for each spatial location, the
    logits of the five concept classes (UP, DOWN, LEFT, RIGHT, NEVER).

    Args:
        in_channels: Number of input channels (32 for cell states, 7 for observations).
        num_classes: Number of concept classes (5).
        kernel_size: Size of the convolution kernel (1 for 1×1, 3 for 3×3).
        bias: Whether to include a bias term. Default ``True``.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        kernel_size: int,
        bias: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        # Padding to preserve spatial dimensions (8×8 -> 8×8)
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            num_classes,
            kernel_size,
            padding=padding,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape ``(B, in_channels, H, W)``.
        Returns:
            logits: Tensor of shape ``(B, num_classes, H, W)``.
        """
        return self.conv(x)

    def get_class_vectors(self) -> torch.Tensor:
        """
        Return the weight vectors for each class.
        Only defined for 1×1 probes (kernel size = 1).

        Returns:
            Tensor of shape ``(num_classes, in_channels)`` where row ``k`` is the
            vector ``w_k`` used to compute the logit for class ``k`` at each location.
        """
        if self.kernel_size != 1:
            raise RuntimeError(
                "get_class_vectors() is only defined for 1×1 probes."
            )
        # conv.weight shape: (num_classes, in_channels, 1, 1)
        return self.conv.weight.data.squeeze(-1).squeeze(-1)


# ------------------------------------------------------------------------------
# 2.  Metric computation
# ------------------------------------------------------------------------------
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 5
) -> Dict:
    """
    Compute per‑class precision, recall, F1 and macro‑averaged F1.

    Args:
        y_true: Ground‑truth class indices (flat array, shape ``(N,)``).
        y_pred: Predicted class indices (flat array, shape ``(N,)``).
        num_classes: Total number of classes (default 5).

    Returns:
        dict with keys:
            - ``macro_f1`` (float)
            - ``per_class_precision`` (list of floats)
            - ``per_class_recall`` (list)
            - ``per_class_f1`` (list)
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)

    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []

    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        per_class_precision.append(precision)
        per_class_recall.append(recall)
        per_class_f1.append(f1)

    macro_f1 = float(np.mean(per_class_f1))

    return {
        "macro_f1": macro_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
    }


# ------------------------------------------------------------------------------
# 3.  ProbeTrainer
# ------------------------------------------------------------------------------
class ProbeTrainer:
    """
    Handles training and evaluation of linear probes for a specific layer and
    concept.  Supports multiple random seeds and multiple kernel sizes.

    Args:
        dataset: ``ProbeDataset`` instance that provides access to the stored
                 train/test splits via :meth:`load` and :meth:`get_dataloader`.
        config: Dictionary‑like object containing the ``probing`` section of
                ``config.yaml``.  Required keys: ``epochs``, ``learning_rate``,
                ``weight_decay``, ``batch_size``, ``probe_kernel_sizes``, ``seeds``.
        layer_idx: Index of the ConvLSTM layer (0, 1, or 2) whose cell state is
                   being probed.
        concept: Concept string, ``'C_A'`` or ``'C_B'``.
        input_channels: Number of input channels (32 for cell states,
                        7 for raw observation baseline).
        device: Torch device string; auto‑detected if ``None``.
    """

    def __init__(
        self,
        dataset: ProbeDataset,
        config: Union[Dict, object],
        layer_idx: int,
        concept: str,
        input_channels: int = 32,
        device: Optional[str] = None,
    ):
        self.dataset = dataset
        self.config = config
        self.layer_idx = layer_idx
        self.concept = concept
        self.input_channels = input_channels
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Extract hyperparameters (with defaults matching config.yaml)
        self.epochs = getattr(config, 'epochs', 10)
        self.learning_rate = getattr(config, 'learning_rate', 0.001)
        self.weight_decay = getattr(config, 'weight_decay', 0.001)
        self.batch_size = getattr(config, 'batch_size', 16)
        self.kernel_sizes = getattr(config, 'probe_kernel_sizes', [1, 3])
        self.seeds = getattr(config, 'seeds', 5)
        self.num_classes = getattr(config, 'num_classes', 5)

    # ------------------------------------------------------------------
    #  Data loading helpers
    # ------------------------------------------------------------------
    def _load_split(self, split: str) -> None:
        """Load the specified split into the dataset object."""
        self.dataset.load(split)

    def _get_dataloader(self, split: str, shuffle: bool) -> DataLoader:
        """
        Obtain a DataLoader for the given split after ensuring the data is loaded.
        """
        self._load_split(split)
        return self.dataset.get_dataloader(
            layer_idx=self.layer_idx,
            concept=self.concept,
            batch_size=self.batch_size,
            shuffle=shuffle,
        )

    # ------------------------------------------------------------------
    #  Single probe training + evaluation
    # ------------------------------------------------------------------
    def train_single_probe(
        self,
        kernel_size: int,
        seed: int
    ) -> Tuple[LinearProbe, Dict]:
        """
        Train a single probe with a fixed kernel size and random seed.

        Args:
            kernel_size: kernel size (1 or 3).
            seed: random seed for reproducibility.

        Returns:
            Trained ``LinearProbe`` and its evaluation metrics dictionary
            (as returned by :meth:`evaluate_probe`).
        """
        set_seed(seed)

        probe = LinearProbe(
            in_channels=self.input_channels,
            num_classes=self.num_classes,
            kernel_size=kernel_size,
            bias=True,
        )
        probe.to(self.device)

        optimizer = optim.AdamW(
            probe.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = nn.CrossEntropyLoss()

        train_loader = self._get_dataloader('train', shuffle=True)

        # Training loop
        probe.train()
        for epoch in range(self.epochs):
            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)          # (B, H, W)

                logits = probe(x)              # (B, num_classes, H, W)
                # Reshape for cross‑entropy loss
                B, C, H, W = logits.shape
                logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)
                y_flat = y.reshape(-1)

                loss = loss_fn(logits_flat, y_flat)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Evaluate on test split
        test_loader = self._get_dataloader('test', shuffle=False)
        metrics = self.evaluate_probe(probe, test_loader)
        return probe, metrics

    def evaluate_probe(
        self,
        probe: LinearProbe,
        dataloader: DataLoader
    ) -> Dict:
        """
        Evaluate a trained probe on a given DataLoader.

        Returns:
            Dictionary containing ``macro_f1`` and per‑class precision, recall, F1.
        """
        probe.eval()
        probe.to(self.device)

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                logits = probe(x)            # (B, num_classes, H, W)
                preds = logits.argmax(dim=1) # (B, H, W)
                all_preds.append(preds.cpu().numpy().reshape(-1))
                all_labels.append(y.numpy().reshape(-1))

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)

        return compute_metrics(y_true, y_pred, self.num_classes)

    # ------------------------------------------------------------------
    #  Full experiment: multiple kernels, multiple seeds
    # ------------------------------------------------------------------
    def train_all(self, save_dir: Optional[str] = None) -> Dict:
        """
        Run the complete probing experiment.

        For each kernel size in ``probe_kernel_sizes`` and each seed in ``seeds``,
        a probe is independently trained and evaluated.  Mean and standard deviation
        of macro F1 per kernel size are reported.

        Args:
            save_dir: If provided, each trained probe is saved as
                      ``probe_l<layer>_<concept>_k<kernel>_s<seed>.pt`` inside this
                      directory.

        Returns:
            Dictionary mapping ``kernel_size`` -> dict with keys:
                - ``mean_macro_f1`` (float)
                - ``std_macro_f1`` (float)
                - ``all_metrics`` (list of per‑seed metric dicts)
        """
        results = {}
        for k in self.kernel_sizes:
            seed_metrics = []
            probes_list = []
            for seed in range(self.seeds):
                probe, metrics = self.train_single_probe(k, seed)
                seed_metrics.append(metrics)
                probes_list.append(probe)

            # Aggregate across seeds
            macro_f1s = [m['macro_f1'] for m in seed_metrics]
            results[k] = {
                'mean_macro_f1': float(np.mean(macro_f1s)),
                'std_macro_f1': float(np.std(macro_f1s)),
                'all_metrics': seed_metrics,
            }

            # Save probes if directory is given
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                for seed, probe in enumerate(probes_list):
                    path = os.path.join(
                        save_dir,
                        f"probe_l{self.layer_idx}_{self.concept}_k{k}_s{seed}.pt"
                    )
                    self.save_model(probe, path)

        return results

    # ------------------------------------------------------------------
    #  Model persistence
    # ------------------------------------------------------------------
    def save_model(self, probe: LinearProbe, path: str) -> None:
        """Save the probe's state dictionary to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(probe.state_dict(), path)

    def load_model(
        self, path: str, kernel_size: int
    ) -> LinearProbe:
        """
        Load a probe from a checkpoint.  Architecture must match.

        Args:
            path: file path to the saved state dict.
            kernel_size: kernel size used for the probe (must be known).

        Returns:
            A ``LinearProbe`` with loaded weights, set to eval mode.
        """
        probe = LinearProbe(
            in_channels=self.input_channels,
            num_classes=self.num_classes,
            kernel_size=kernel_size,
            bias=True,
        )
        probe.load_state_dict(torch.load(path, map_location=self.device))
        probe.to(self.device)
        probe.eval()
        return probe


# ------------------------------------------------------------------------------
#  Optional main for quick testing (not used in production)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # This block is only for manual sanity checks.
    print("probes.py self-test not implemented.")
