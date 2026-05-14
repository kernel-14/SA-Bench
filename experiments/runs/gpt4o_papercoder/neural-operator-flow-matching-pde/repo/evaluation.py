# evaluation.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader
import pandas as pd


class Evaluation:
    """Handles evaluation of trained models using metrics such as L2RE and VRMSE, 
    and visualizes predictions against ground truth data."""
    
    def __init__(self, model: torch.nn.Module, dataset: Dict[str, Dict[str, torch.Tensor]], metrics: List[str], config: Dict):
        """
        Initializes Evaluation class attributes.

        Args:
            model (torch.nn.Module): Trained model for evaluation.
            dataset (Dict): Dictionary containing train, validation, and test splits.
            metrics (List[str]): List of metrics to compute (e.g., ["L2RE", "VRMSE"]).
            config (Dict): Configuration dictionary loaded from config.yaml.
        """
        self.model = model
        self.dataset = dataset
        self.metrics = metrics
        self.device = torch.device("cuda" if torch.cuda.is_available() and config["hardware"]["devices"] == "gpu" else "cpu")
        self.save_path = config['evaluation']['visualization']['save_path']
        self.batch_size = config['training']['fmt']['batch_size']
        self.logger_dir = config['logging']['save_dir']
        self.model.to(self.device)
        os.makedirs(self.save_path, exist_ok=True)

    def evaluate(self) -> Dict[str, float]:
        """
        Runs the evaluation process over the test dataset, computes metrics, and saves results.

        Returns:
            Dict[str, float]: A dictionary containing computed metric values.
        """
        self.model.eval()
        test_loader = DataLoader(
            self.dataset['test']['inputs'], 
            batch_size=self.batch_size, 
            shuffle=False
        )
        results = {metric: 0.0 for metric in self.metrics}

        self._log("Starting evaluation.")

        with torch.no_grad():
            total_samples = 0
            for batch_idx, batch in enumerate(test_loader):
                inputs = batch.to(self.device)
                targets = self.dataset['test']['targets'][batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size].to(self.device)

                # Model prediction
                predictions = self.model(inputs)

                # Compute metrics for the batch
                batch_metrics = self._compute_batch_metrics(predictions, targets)

                # Update metrics based on current batch
                for metric, value in batch_metrics.items():
                    results[metric] += value * inputs.size(0)
                total_samples += inputs.size(0)

        # Normalize metrics over total samples
        for metric in results:
            results[metric] /= total_samples

        self._log(f"Final Evaluation Results: {results}")
        self._save_metrics(results, os.path.join(self.logger_dir, "evaluation_metrics.csv"))
        self._plot_sample_results(test_loader, self.save_path)

        return results

    def _compute_batch_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Computes individual metrics for a batch of predictions and targets.

        Args:
            predictions (torch.Tensor): Batch of predicted values.
            targets (torch.Tensor): Ground truth values for the batch.

        Returns:
            Dict[str, float]: Batch metrics such as L2RE and VRMSE.
        """
        metrics = {}
        for metric in self.metrics:
            if metric == "L2RE":
                metrics["L2RE"] = self._compute_l2re(predictions, targets)
            elif metric == "VRMSE":
                metrics["VRMSE"] = self._compute_vrmse(predictions, targets)
            else:
                raise ValueError(f"Unknown metric: {metric}")
        return metrics

    def _compute_l2re(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Computes the L2 Relative Error (L2RE).

        Args:
            predictions (torch.Tensor): Batch predictions.
            targets (torch.Tensor): Batch targets.

        Returns:
            float: L2 Relative Error value.
        """
        numerator = torch.norm(predictions - targets, p=2)
        denominator = torch.norm(targets, p=2)
        return (numerator / (denominator + 1e-8)).item()  # Avoid division by zero

    def _compute_vrmse(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Computes Variance-Normalized Root Mean Square Error (VRMSE).

        Args:
            predictions (torch.Tensor): Batch predictions.
            targets (torch.Tensor): Batch targets.

        Returns:
            float: VRMSE value.
        """
        residuals = predictions - targets
        variance = torch.var(targets, unbiased=False)
        vrmse = torch.sqrt(torch.mean(residuals ** 2 / (variance + 1e-8)))
        return vrmse.item()

    def _plot_sample_results(self, test_loader: DataLoader, save_path: str) -> None:
        """
        Plots example predictions vs targets and saves the figures to the specified directory.

        Args:
            test_loader (DataLoader): Test DataLoader instance.
            save_path (str): Path to save the visualizations.

        Raises:
            IOError: If plotting fails or files cannot be saved.
        """
        self._log("Generating sample result plots.")
        os.makedirs(save_path, exist_ok=True)

        with torch.no_grad():
            batch = next(iter(test_loader))
            inputs = batch.to(self.device)
            predictions = self.model(inputs)
            targets = batch.to(self.device)

        # Plot results for a small subset
        num_samples_to_plot = min(8, inputs.size(0))  # Choose a subset for visualization
        fig, axes = plt.subplots(num_samples_to_plot, 2, figsize=(10, 5 * num_samples_to_plot))

        for i in range(num_samples_to_plot):
            axes[i, 0].imshow(targets[i].cpu().numpy().transpose(1, 2, 0), cmap="viridis")
            axes[i, 0].title.set_text("Ground Truth")
            axes[i, 1].imshow(predictions[i].cpu().numpy().transpose(1, 2, 0), cmap="viridis")
            axes[i, 1].title.set_text("Prediction")

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "evaluation_results.png"))
        plt.close(fig)

    def _save_metrics(self, metrics: Dict[str, float], path: str) -> None:
        """
        Saves metrics to a CSV file.

        Args:
            metrics (Dict[str, float]): Dictionary of computed metric values.
            path (str): Destination file path for saving the metrics.

        Raises:
            IOError: If metrics cannot be written to the file.
        """
        try:
            pd.DataFrame([metrics]).to_csv(path, index=False)
            self._log(f"Metrics successfully saved to {path}")
        except Exception as e:
            raise IOError(f"Could not save metrics to {path}: {e}")

    def _log(self, message: str) -> None:
        """
        Logs a message during evaluation.

        Args:
            message (str): Message to log.
        """
        print(f"[Evaluation]: {message}")
