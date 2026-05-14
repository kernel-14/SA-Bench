## evaluation.py

"""
This module defines the Evaluation class for assessing model performance on test datasets.
Key functionalities include calculating evaluation metrics (e.g., NMAE), visualizing results, and logging key metrics.

Classes:
    - Evaluation: Handles evaluation workflows, metric computation, and result visualization.

Functions:
    - evaluate: Performs evaluation on the test dataset and computes metrics.
    - compute_nmae: Calculates the Normalized Mean Absolute Error (NMAE).
    - visualize_field: Generates and saves visualizations for individual PDE solution fields.
    - visualize_results: Automates the visualization of predictions and ground truth comparisons.
    - log_metrics: Logs evaluation metrics to both console and log files.
"""

import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import Dict, Tuple
from utils import log_metrics, plot_field


class Evaluation:
    """
    Evaluates a trained model on a given test dataset. Includes methods to compute metrics,
    visualize results, and log numeric and qualitative performance details.

    Attributes:
        model (torch.nn.Module): Trained neural operator model.
        test_data (torch.utils.data.Dataset): Test dataset for evaluation.
        config (dict): Configuration dictionary for evaluation and logging.
        device (torch.device): Computation device (CPU/GPU).
    """

    def __init__(self, model: torch.nn.Module, test_data: torch.utils.data.Dataset, config: Dict):
        """
        Initializes the evaluation process with the model, test dataset, and configuration.

        Args:
            model (torch.nn.Module): Trained model to be evaluated.
            test_data (torch.utils.data.Dataset): Dataset containing test samples.
            config (Dict): Configuration dictionary from 'config.yaml'.
        """
        self.model = model
        self.test_data = test_data
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Transfer model to the evaluation device
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        # DataLoader for batching test samples
        self.test_loader = DataLoader(test_data, batch_size=config["training"]["batch_size"], shuffle=False)

        # Metric and logging configuration
        self.metric_name = config["evaluation"]["metrics"]
        self.log_dir = config["logging"]["log_dir"]
        os.makedirs(self.log_dir, exist_ok=True)  # Ensure log directory exists

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model performance on the test dataset and compute metrics.

        Returns:
            Dict[str, float]: Dictionary containing evaluation metrics (e.g., NMAE).
        """
        print("Starting evaluation...")
        total_nmae = 0.0
        total_samples = 0

        with torch.no_grad():
            for inputs, ground_truth in self.test_loader:
                # Move data to the computation device
                inputs, ground_truth = inputs.to(self.device), ground_truth.to(self.device)

                # Model predictions
                predictions = self.model(inputs)

                # Compute batch NMAE
                batch_nmae = self.compute_nmae(predictions, ground_truth)
                total_nmae += batch_nmae * inputs.size(0)
                total_samples += inputs.size(0)

            # Final average NMAE across all test samples
            final_nmae = total_nmae / total_samples

        # Log evaluation results
        metrics = {self.metric_name: final_nmae}
        self.log_metrics(metrics)

        # Print results for user review
        print(f"Evaluation completed. {self.metric_name}: {final_nmae:.6f}")
        return metrics

    def compute_nmae(self, predictions: torch.Tensor, ground_truth: torch.Tensor) -> float:
        """
        Computes the Normalized Mean Absolute Error (NMAE) between predictions and ground truth.

        Args:
            predictions (torch.Tensor): Model predictions of shape (batch_size, ...).
            ground_truth (torch.Tensor): Ground truth values of shape (batch_size, ...).

        Returns:
            float: Computed NMAE metric.
        """
        # Element-wise absolute error
        absolute_error = torch.abs(predictions - ground_truth)

        # Normalization factor: range of ground truth values
        normalization = torch.max(ground_truth) - torch.min(ground_truth) + 1e-8

        # Compute mean normalized absolute error (batch-wise)
        nmae = torch.mean(absolute_error / normalization).item()
        return nmae

    def visualize_field(self, field: torch.Tensor, title: str, save_path: str) -> None:
        """
        Visualizes a single 2D PDE solution field.

        Args:
            field (torch.Tensor): Tensor representing the PDE solution (2D field).
            title (str): Title for the visualization plot.
            save_path (str): Path to save the visualized plot.
        """
        plot_field(field, title, save_path)

    def visualize_results(self, predictions: torch.Tensor, ground_truth: torch.Tensor) -> None:
        """
        Generates comparative visualizations for model predictions and ground truth.

        Args:
            predictions (torch.Tensor): Tensor containing model predictions.
            ground_truth (torch.Tensor): Tensor containing ground truth values.

        Saves:
            Visualizations in the logging directory.
        """
        num_samples_to_visualize = min(len(predictions), 5)  # Visualize up to 5 samples by default

        for i in range(num_samples_to_visualize):
            # Select the sample field
            pred_field = predictions[i].detach().cpu()  # Predictions
            gt_field = ground_truth[i].detach().cpu()  # Ground truth

            # Define plot file paths
            pred_save_path = os.path.join(self.log_dir, f"sample_{i}_prediction.png")
            gt_save_path = os.path.join(self.log_dir, f"sample_{i}_ground_truth.png")

            # Plot and save visualizations
            self.visualize_field(pred_field, f"Sample {i} - Prediction", pred_save_path)
            self.visualize_field(gt_field, f"Sample {i} - Ground Truth", gt_save_path)

        print("Visualizations generated for predictions and ground truth.")

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Logs evaluation metrics to the console and a log file.

        Args:
            metrics (Dict[str, float]): Dictionary of evaluation metrics (e.g., NMAE).
        """
        log_file_path = os.path.join(self.log_dir, "evaluation_metrics.yaml")
        log_metrics(metrics, log_file_path)
        print(f"Metrics logged to {log_file_path}.")
