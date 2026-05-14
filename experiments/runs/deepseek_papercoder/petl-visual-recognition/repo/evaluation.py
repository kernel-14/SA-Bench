## evaluation.py

"""Evaluation utilities for reproducing the PEFT study.

Provides the `Evaluator` class to compute top‑1 accuracy on target
datasets, distribution‑shift accuracies, prediction similarities
between two models, and ensemble predictions via averaged logits.
All metrics follow the definitions in the paper.
"""

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Assumes PEFTModel is defined in model_builder.py as per the project design.
from model_builder import PEFTModel


class Evaluator:
    """
    Central evaluation class for a fine‑tuned PEFT model.

    It is initialised with the model and a primary test DataLoader.
    All methods operate in evaluation mode (no gradients) and report
    top‑1 accuracy as a percentage (0‑100).

    Attributes:
        model:        Trained PEFTModel.
        test_loader:  Default DataLoader used when no explicit loader is provided.
        device:       Torch device on which the model resides.
    """

    def __init__(self, model: PEFTModel, test_loader: DataLoader) -> None:
        """
        Args:
            model:        Instance of PEFTModel (already trained).
            test_loader:  DataLoader for the primary test set (e.g., VTAB test
                          or ImageNet validation). Loader should not shuffle.
        """
        self.model = model
        self.test_loader = test_loader

        # Automatically infer the device from the model's parameters.
        # The model is assumed to already be on the target device.
        self.device = next(model.parameters()).device

    # ------------------------------------------------------------------ #
    #  Internal helper
    # ------------------------------------------------------------------ #
    def _evaluate_accuracy(self, loader: DataLoader) -> float:
        """
        Compute top‑1 accuracy (in %) for a single DataLoader.

        Sets the model to evaluation mode and disables gradients.
        """
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                logits = self.model(images)
                preds = logits.argmax(dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        if total == 0:
            return 0.0
        return (correct / total) * 100.0

    # ------------------------------------------------------------------ #
    #  Public methods (matching the class diagram)
    # ------------------------------------------------------------------ #
    def evaluate_accuracy(self, loader: Optional[DataLoader] = None) -> float:
        """
        Return top‑1 accuracy (0‑100) on the given loader.

        If no loader is supplied, uses the internal `test_loader`.
        """
        if loader is None:
            loader = self.test_loader
        return self._evaluate_accuracy(loader)

    def evaluate_distribution_shift(
        self, shift_loaders: List[DataLoader]
    ) -> List[float]:
        """
        Evaluate the model on a list of distribution‑shift DataLoaders.

        Args:
            shift_loaders:  List of DataLoader objects, each corresponding to
                            a shift dataset (ImageNet‑V2, ‑R, ‑S, ‑A).

        Returns:
            A list of per‑dataloader accuracies (each a float in 0‑100).
        """
        return [self._evaluate_accuracy(loader) for loader in shift_loaders]

    def prediction_similarity(
        self,
        other_model: PEFTModel,
        loader: Optional[DataLoader] = None,
    ) -> float:
        """
        Compute the percentage of samples for which `self.model` and
        `other_model` make the same prediction.

        Args:
            other_model:  Another fine‑tuned PEFTModel (same dataset).
            loader:       DataLoader providing the evaluation data. Defaults
                          to `self.test_loader`.

        Returns:
            Similarity score (0‑100): percent of identical top‑1 predictions.
        """
        if loader is None:
            loader = self.test_loader

        self.model.eval()
        other_model.eval()

        total = 0
        matches = 0
        with torch.no_grad():
            for images, _ in loader:   # labels not needed for comparison
                images = images.to(self.device, non_blocking=True)
                out1 = self.model(images).argmax(dim=1)
                out2 = other_model(images).argmax(dim=1)

                matches += (out1 == out2).sum().item()
                total += images.size(0)

        if total == 0:
            return 0.0
        return (matches / total) * 100.0

    def ensemble_predict(
        self,
        models: List[PEFTModel],
        loader: Optional[DataLoader] = None,
    ) -> List[int]:
        """
        Generate ensemble predictions via averaged logits (as in Fig. 4 of the paper).

        For each sample, the logits of all models are averaged, and the
        class with the highest averaged logit is chosen.

        Args:
            models:  List of PEFTModel instances (all trained on the same dataset).
            loader:  DataLoader for the dataset. Defaults to `self.test_loader`.

        Returns:
            A list of predicted class IDs (length = total number of samples).
        """
        if loader is None:
            loader = self.test_loader

        for m in models:
            m.eval()

        all_preds = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(self.device, non_blocking=True)
                logits_list = [m(images) for m in models]

                # Average logits (could use sum, but mean is equivalent)
                avg_logits = sum(logits_list) / len(models)
                preds = avg_logits.argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)

        return all_preds
