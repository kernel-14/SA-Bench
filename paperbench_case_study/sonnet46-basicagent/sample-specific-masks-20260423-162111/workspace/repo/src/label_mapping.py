"""
Output label mapping functions for visual reprogramming.

Three methods:
1. Random Label Mapping (Rlm): randomly assigns source labels to target labels
2. Frequent Label Mapping (Flm): assigns based on prediction frequency
3. Iterative Label Mapping (Ilm): updates mapping each epoch (default in paper)
"""

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment


def random_label_mapping(num_source_classes, num_target_classes, seed=None):
    """
    Random Label Mapping (Rlm): randomly selects num_target_classes source labels
    and creates a random injective mapping from source subset to target labels.

    Args:
        num_source_classes: |Y^P| (e.g., 1000 for ImageNet)
        num_target_classes: |Y^T|
        seed: random seed

    Returns:
        mapping: dict {source_label -> target_label}
        source_subset: list of selected source labels
    """
    rng = np.random.RandomState(seed)
    source_subset = rng.choice(num_source_classes, num_target_classes, replace=False).tolist()
    target_labels = rng.permutation(num_target_classes).tolist()
    mapping = {src: tgt for src, tgt in zip(source_subset, target_labels)}
    return mapping, source_subset


def compute_frequency_distribution(model, f_in, data_loader, num_source_classes,
                                   num_target_classes, device):
    """
    Algorithm 2: Compute frequency distribution matrix d of shape
    [num_source_classes x num_target_classes].

    d[y_P, y_T] = count of times model predicted y_P when true label is y_T.

    Args:
        model: pre-trained model f_P (frozen)
        f_in: input transformation function (takes batch, returns transformed batch)
        data_loader: DataLoader for target training data
        num_source_classes: |Y^P|
        num_target_classes: |Y^T|
        device: torch device

    Returns:
        d: numpy array of shape [num_source_classes, num_target_classes]
    """
    d = np.zeros((num_source_classes, num_target_classes), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            transformed = f_in(images)
            outputs = model(transformed)
            preds = outputs.argmax(dim=1)
            for pred, label in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                d[pred, label] += 1
    return d


def frequent_label_mapping(d):
    """
    Algorithm 3: Frequent Label Mapping (Flm).
    Greedily assigns source labels to target labels based on frequency.

    Args:
        d: frequency distribution matrix [num_source_classes x num_target_classes]

    Returns:
        mapping: dict {source_label -> target_label}
        source_subset: list of selected source labels
    """
    num_source, num_target = d.shape
    d_copy = d.copy().astype(float)
    mapping = {}
    source_subset = []
    assigned_targets = set()

    while len(source_subset) < num_target:
        # Find maximum in d
        idx = np.unravel_index(np.argmax(d_copy), d_copy.shape)
        y_P, y_T = idx

        if d_copy[y_P, y_T] <= 0:
            break

        mapping[int(y_P)] = int(y_T)
        source_subset.append(int(y_P))
        assigned_targets.add(int(y_T))

        # Zero out row y_P and column y_T to enforce injectivity
        d_copy[y_P, :] = 0
        d_copy[:, y_T] = 0

    return mapping, source_subset


def iterative_label_mapping(model, f_in, data_loader, num_source_classes,
                             num_target_classes, device):
    """
    Algorithm 4: Iterative Label Mapping (Ilm).
    Computes frequency distribution and applies Flm for the current epoch.

    This is called at the beginning of each training epoch to update the mapping.

    Args:
        model: pre-trained model f_P (frozen)
        f_in: input transformation function
        data_loader: DataLoader for target training data
        num_source_classes: |Y^P|
        num_target_classes: |Y^T|
        device: torch device

    Returns:
        mapping: dict {source_label -> target_label}
        source_subset: list of selected source labels
    """
    d = compute_frequency_distribution(model, f_in, data_loader, num_source_classes,
                                       num_target_classes, device)
    return frequent_label_mapping(d)


class LabelMapper:
    """
    Wrapper class for label mapping that handles the conversion from
    source predictions to target labels.
    """
    def __init__(self, mapping, source_subset, num_source_classes):
        """
        Args:
            mapping: dict {source_label -> target_label}
            source_subset: list of source labels in the subset
            num_source_classes: total number of source classes
        """
        self.mapping = mapping
        self.source_subset = source_subset
        self.num_source_classes = num_source_classes

        # Build reverse mapping: target -> source
        self.reverse_mapping = {v: k for k, v in mapping.items()}

        # Build index tensor for fast lookup
        # Maps source class index to target class index (-1 if not in subset)
        self._build_lookup()

    def _build_lookup(self):
        """Build a lookup tensor for fast source->target mapping."""
        self.lookup = torch.full((self.num_source_classes,), -1, dtype=torch.long)
        for src, tgt in self.mapping.items():
            self.lookup[src] = tgt

    def map_predictions(self, source_logits):
        """
        Map source model logits to target class predictions.

        Args:
            source_logits: tensor of shape (B, num_source_classes)

        Returns:
            target_preds: tensor of shape (B,) with target class indices
        """
        # Only consider logits for source classes in the subset
        subset_indices = torch.tensor(self.source_subset, dtype=torch.long,
                                      device=source_logits.device)
        subset_logits = source_logits[:, subset_indices]  # (B, num_target_classes)

        # Get best source class within subset
        best_subset_idx = subset_logits.argmax(dim=1)  # (B,)
        best_source_class = subset_indices[best_subset_idx]  # (B,)

        # Map to target class
        lookup = self.lookup.to(source_logits.device)
        target_preds = lookup[best_source_class]
        return target_preds

    def get_target_logits(self, source_logits):
        """
        Extract logits for the source subset, ordered by target class.

        Args:
            source_logits: tensor of shape (B, num_source_classes)

        Returns:
            target_logits: tensor of shape (B, num_target_classes)
        """
        num_target = len(self.mapping)
        target_logits = torch.zeros(source_logits.shape[0], num_target,
                                    device=source_logits.device)
        for src, tgt in self.mapping.items():
            target_logits[:, tgt] = source_logits[:, src]
        return target_logits
