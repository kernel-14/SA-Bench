## evaluation.py

import os
import torch
import json
import numpy as np
from torch.utils.data import DataLoader
from typing import List, Dict, Any
from utils import save_results, plot_metrics, set_random_seed


class Evaluation:
    """
    Class responsible for evaluating PEFT methods on various metrics and datasets.
    Implements Top-1 accuracy evaluation, prediction similarity analysis, and ensemble evaluation.
    """

    def __init__(self, model: torch.nn.Module, test_data: DataLoader):
        """
        Initialize the Evaluation class with the model and test dataset.

        Args:
            model (torch.nn.Module): Trained model with PEFT modules.
            test_data (DataLoader): PyTorch DataLoader for the test set.
        """
        self.model = model
        self.test_data = test_data
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def evaluate_top1_accuracy(self) -> Dict[str, float]:
        """
        Computes the Top-1 accuracy of the model on the test dataset.

        Returns:
            Dict[str, float]: A dictionary containing Top-1 accuracy.
        """
        self.model.eval()
        total = 0
        correct = 0

        with torch.no_grad():
            for inputs, labels in self.test_data:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predictions = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

        accuracy = correct / total * 100.0
        return {"top1_accuracy": accuracy}

    def compute_prediction_similarity(self, 
                                      peft_methods: List[torch.nn.Module], 
                                      dataset: DataLoader) -> np.ndarray:
        """
        Compute the prediction similarity between multiple PEFT methods.

        Args:
            peft_methods (List[torch.nn.Module]): List of PEFT methods as trained models.
            dataset (DataLoader): Test dataset for prediction similarity analysis.

        Returns:
            np.ndarray: A similarity matrix where each cell represents the overlap ratio of predictions.
        """
        # Collect predictions for each method across all batches
        predictions = {}
        
        for idx, method in enumerate(peft_methods):
            method.to(self.device).eval()
            method_preds = []

            with torch.no_grad():
                for inputs, _ in dataset:
                    inputs = inputs.to(self.device)
                    outputs = method(inputs)
                    _, batch_preds = torch.max(outputs, dim=1)
                    method_preds.append(batch_preds.cpu().numpy())

            predictions[f"method_{idx}"] = np.concatenate(method_preds)

        # Compute pairwise prediction similarities
        methods = list(predictions.keys())
        n_methods = len(methods)
        similarity_matrix = np.zeros((n_methods, n_methods))

        for i in range(n_methods):
            for j in range(n_methods):
                method_i_preds = predictions[methods[i]]
                method_j_preds = predictions[methods[j]]
                overlap = np.sum(method_i_preds == method_j_preds)
                similarity_matrix[i, j] = overlap / len(method_i_preds)

        return similarity_matrix

    def ensemble_evaluation(self, 
                            peft_methods: List[torch.nn.Module], 
                            dataset: DataLoader, 
                            strategy: str = "majority_voting") -> Dict[str, float]:
        """
        Evaluate ensemble performance of multiple PEFT methods.

        Args:
            peft_methods (List[torch.nn.Module]): List of trained PEFT methods.
            dataset (DataLoader): Dataset for evaluating ensemble predictions.
            strategy (str): Strategy for ensemble voting. Options are:
                - "majority_voting"
                - "logit_averaging"

        Returns:
            Dict[str, float]: Top-1 accuracy for ensemble predictions.
        """
        ensemble_predictions = []
        labels = []

        # Phase 1: Collect logits or predictions
        all_logits = []
        
        for method in peft_methods:
            method.to(self.device).eval()
            method_logits = []

            with torch.no_grad():
                for inputs, _ in dataset:
                    inputs = inputs.to(self.device)
                    logits = method(inputs)
                    method_logits.append(logits.cpu().numpy())

            all_logits.append(np.concatenate(method_logits))

        # Phase 2: Perform ensemble voting
        if strategy == "majority_voting":
            for batch_logits in zip(*all_logits):
                batch_preds = [np.argmax(logit, axis=1) for logit in batch_logits]
                batch_ensemble_preds = np.apply_along_axis(
                    lambda x: np.bincount(x).argmax(), axis=0, arr=np.array(batch_preds)
                )
                ensemble_predictions.append(batch_ensemble_preds)
        elif strategy == "logit_averaging":
            averaged_logits = np.mean(np.array(all_logits), axis=0)
            ensemble_predictions = np.argmax(averaged_logits, axis=1)
        else:
            raise ValueError(f"Unsupported ensemble strategy: {strategy}")

        # Phase 3: Compute Top-1 accuracy
        ensemble_predictions = np.concatenate(ensemble_predictions)
        with torch.no_grad():
            for _, batch_labels in dataset:
                labels.append(batch_labels.numpy())

        labels = np.concatenate(labels)
        accuracy = np.mean(ensemble_predictions == labels) * 100.0
        return {"ensemble_accuracy": accuracy}

    def save_and_plot_results(self, metrics: Dict[str, Any], save_path: str) -> None:
        """
        Save evaluation results as JSON and generate plots for metric visualizations.

        Args:
            metrics (Dict[str, Any]): Evaluation results including accuracy, similarity matrices, etc.
            save_path (str): Path to save the results and plots.
        """
        os.makedirs(save_path, exist_ok=True)

        # Save raw results in JSON format
        raw_results_path = os.path.join(save_path, "evaluation_results.json")
        save_results(metrics, raw_results_path)

        # Generate visualizations
        if "Prediction Similarity Matrix" in metrics:
            similarity_matrix = metrics["Prediction Similarity Matrix"]
            matrix_plot_path = os.path.join(save_path, "prediction_similarity_matrix.png")
            plot_metrics({"Prediction Similarity": similarity_matrix}, matrix_plot_path)

        if "Top-1 Accuracy" in metrics:
            accuracy_plot_path = os.path.join(save_path, "accuracy_plot.png")
            plot_metrics({"Accuracy": metrics["Top-1 Accuracy"]}, accuracy_plot_path)

        if "Ensemble Performance" in metrics:
            ensemble_plot_path = os.path.join(save_path, "ensemble_performance.png")
            plot_metrics({"Ensemble Performance": metrics["Ensemble Performance"]}, ensemble_plot_path)

    def analyze_prediction_diversity(self, 
                                     peft_methods: List[torch.nn.Module], 
                                     dataset: DataLoader, 
                                     confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Analyze diversity in predictions for high-confidence correct and low-confidence wrong cases.

        Args:
            peft_methods (List[torch.nn.Module]): List of trained PEFT models.
            dataset (DataLoader): Dataset for evaluating diversity.
            confidence_threshold (float): Threshold for separating high- and low-confidence predictions.

        Returns:
            Dict[str, Any]: A dictionary containing overlap statistics and diversity insights.
        """
        diversity_metrics = {"high_confidence_overlap": {}, "low_confidence_diversity": {}}

        # Store high-confidence correct predictions and low-confidence wrong predictions
        high_confidence_correct = []
        low_confidence_wrong = []

        for method in peft_methods:
            method.to(self.device).eval()
            high_correct, low_wrong = [], []

            with torch.no_grad():
                for inputs, labels in dataset:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = method(inputs)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidences, predictions = probabilities.max(dim=1)

                    # High-confidence correct
                    high_correct.append(
                        (predictions == labels).cpu().numpy() & (confidences >= confidence_threshold).cpu().numpy()
                    )
                    # Low-confidence wrong
                    low_wrong.append(
                        (predictions != labels).cpu().numpy() & (confidences < confidence_threshold).cpu().numpy()
                    )

            high_confidence_correct.append(np.concatenate(high_correct))
            low_confidence_wrong.append(np.concatenate(low_wrong))

        # Compute overlaps
        diversity_metrics["high_confidence_overlap"] = self._compute_overlap_statistics(high_confidence_correct)
        diversity_metrics["low_confidence_diversity"] = self._compute_overlap_statistics(low_confidence_wrong)
        return diversity_metrics

    def _compute_overlap_statistics(self, prediction_bins: List[np.ndarray]) -> Dict[str, float]:
        """
        Compute pairwise overlaps as a diversity measure.

        Args:
            prediction_bins (List[np.ndarray]): List of binary prediction outcomes (e.g., high-confidence).

        Returns:
            Dict[str, float]: Overlap statistics for a pairwise basis.
        """
        n_methods = len(prediction_bins)
        overlap_matrix = np.zeros((n_methods, n_methods))

        for i in range(n_methods):
            for j in range(i, n_methods):
                intersection = np.sum(prediction_bins[i] & prediction_bins[j])
                union = np.sum(prediction_bins[i] | prediction_bins[j])
                overlap_matrix[i, j] = intersection / union if union > 0 else 0.0
                overlap_matrix[j, i] = overlap_matrix[i, j]  # Symmetric matrix

        return {
            "average_overlap": np.mean(overlap_matrix),
            "pairwise_overlap": overlap_matrix.tolist(),
        }
