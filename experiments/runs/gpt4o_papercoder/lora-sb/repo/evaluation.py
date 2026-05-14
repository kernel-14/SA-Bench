"""
evaluation.py
Implements the Evaluator class responsible for assessing the performance of the LoRA-SB model.
The evaluator computes evaluation metrics such as accuracy and correlation coefficients for different tasks based on the dataset provided.
Adheres to the predefined design and configuration in config.yaml.
"""

import torch
from sklearn.metrics import matthews_corrcoef, mean_squared_error
from scipy.stats import pearsonr
from typing import List, Dict, Any
import logging


class Evaluator:
    """
    Evaluator: Computes metrics to assess the performance of LoRA-SB model across benchmarks.
    Handles arithmetic tasks, commonsense reasoning datasets, and GLUE benchmarks.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        evaluation_metrics: List[str] = ["accuracy"]
    ) -> None:
        """
        Initialize the Evaluator.

        Args:
            model (torch.nn.Module): Trained LoRA-SB model to be evaluated.
            test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
            evaluation_metrics (List[str]): List of metrics to compute (default: ["accuracy"]).
        """
        self.model = model
        self.test_loader = test_loader
        self.evaluation_metrics = evaluation_metrics

        # Configure logging for evaluation
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [Evaluator] %(message)s")
        self.logger = logging.getLogger('Evaluator')

        # Ensure model is in evaluation mode
        self.model.eval()

    def evaluate(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluate the model on the test dataset and compute specified metrics.

        Returns:
            Dict[str, Dict[str, float]]: Evaluation results by dataset and metric.
        """
        self.logger.info("Starting evaluation...")

        # Step 1: Run inference
        self.logger.info("Running inference on test dataset...")
        predictions, labels = self._run_inference(self.test_loader)

        # Step 2: Compute evaluation metrics
        results = {}
        for metric in self.evaluation_metrics:
            if metric == "accuracy":
                results["accuracy"] = self._compute_accuracy(predictions, labels)
            elif metric == "matthews_corr":
                results["matthews_corr"] = self._compute_matthews_corr(predictions, labels)
            elif metric == "pearson_corr":
                results["pearson_corr"] = self._compute_pearson_corr(predictions, labels)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
        
        self.logger.info("Evaluation completed successfully.")
        return results

    def _run_inference(self, data_loader: torch.utils.data.DataLoader) -> Tuple[List[Any], List[Any]]:
        """
        Perform inference on the test dataset and collect predictions and labels.

        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader containing the test dataset.

        Returns:
            Tuple[List[Any], List[Any]]: Predictions and corresponding labels from the test dataset.
        """
        device = next(self.model.parameters()).device
        all_predictions = []
        all_labels = []

        # Disable gradient computation for inference
        with torch.no_grad():
            for batch in data_loader:
                # Move batch data to the correct device (CPU/GPU)
                batch_inputs = {key: value.to(device) for key, value in batch.items() if key != 'labels'}
                batch_labels = batch.get('labels')  # Labels may not be part of all benchmarks
                if batch_labels is not None:
                    batch_labels = batch_labels.to(device)

                # Get model predictions from forward pass
                outputs = self.model(batch_inputs)
                predictions = torch.argmax(outputs, dim=-1) if outputs.ndim > 1 else outputs.squeeze()

                # Collect predictions and labels
                all_predictions.extend(predictions.cpu().numpy())
                if batch_labels is not None:
                    all_labels.extend(batch_labels.cpu().numpy())

        return all_predictions, all_labels

    def _compute_accuracy(self, predictions: List[Any], labels: List[Any]) -> float:
        """
        Compute accuracy by comparing predictions with labels.

        Args:
            predictions (List[Any]): List of predicted values or class labels.
            labels (List[Any]): List of ground-truth labels.

        Returns:
            float: Accuracy score.
        """
        if len(predictions) != len(labels):
            raise ValueError("Predictions and labels must have the same length.")

        correct_predictions = sum([1 for pred, label in zip(predictions, labels) if pred == label])
        accuracy = correct_predictions / len(labels)
        self.logger.info(f"Accuracy computed: {accuracy:.4f}")
        return accuracy

    def _compute_matthews_corr(self, predictions: List[Any], labels: List[Any]) -> float:
        """
        Compute Matthews correlation coefficient for classification tasks.

        Args:
            predictions (List[Any]): Predicted class labels.
            labels (List[Any]): Ground-truth class labels.

        Returns:
            float: Matthews correlation score.
        """
        if len(predictions) != len(labels):
            raise ValueError("Predictions and labels must have the same length.")
        
        mcc = matthews_corrcoef(labels, predictions)
        self.logger.info(f"Matthews correlation coefficient computed: {mcc:.4f}")
        return mcc

    def _compute_pearson_corr(self, predictions: List[Any], labels: List[Any]) -> float:
        """
        Compute Pearson correlation coefficient for regression or similarity tasks.

        Args:
            predictions (List[Any]): Predicted continuous values.
            labels (List[Any]): Ground-truth continuous values.

        Returns:
            float: Pearson correlation coefficient.
        """
        if len(predictions) != len(labels):
            raise ValueError("Predictions and labels must have the same length.")
        
        pearson_corr, _ = pearsonr(labels, predictions)
        self.logger.info(f"Pearson correlation coefficient computed: {pearson_corr:.4f}")
        return pearson_corr
