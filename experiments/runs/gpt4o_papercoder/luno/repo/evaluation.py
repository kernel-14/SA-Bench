## evaluation.py

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any


class Evaluation:
    """
    Evaluation class for assessing predictive uncertainty and accuracy.
    Supports evaluation metrics (RMSE, NLL, χ²-statistic) and visualizations.
    """

    def __init__(self, model: Any, metrics: List[str] = ["rmse", "marginal_nll", "chi_squared"], visualizations: bool = True):
        """
        Initialize the Evaluation class.

        Args:
            model (Any): Trained neural operator model.
            metrics (List[str]): List of metrics to compute. Defaults to ["rmse", "marginal_nll", "chi_squared"].
            visualizations (bool): Whether to generate and save visualizations. Defaults to True.
        """
        self.model = model
        self.metrics_to_compute = metrics
        self.visualizations_enabled = visualizations

    def evaluate(self, data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate the model's predictions with selected metrics.

        Args:
            data (Dict[str, np.ndarray]): Dictionary containing:
                - "inputs": Input trajectories.
                - "truth": Ground truth outputs.
                - "predictions": Model predictions (mean and std).

        Returns:
            Dict[str, float]: Computed metrics, e.g., {"rmse": ..., "marginal_nll": ..., "chi_squared": ...}.
        """
        if "truth" not in data or "predictions" not in data:
            raise ValueError("Data must contain 'truth' and 'predictions' keys.")
        if not {"mean", "std"}.issubset(data["predictions"]):
            raise ValueError("Predictions must contain 'mean' and 'std' trajectories.")

        truth = data["truth"]
        predictions = data["predictions"]

        # Compute metrics
        results = self.compute_metrics(truth=truth, predictions=predictions)

        # Print results
        print("Evaluation Metrics:")
        for metric, value in results.items():
            print(f"  {metric}: {value:.6f}")

        return results

    def visualize_uncertainty(self, predictions: Dict[str, np.ndarray], truth: np.ndarray, save_filename: str = "uncertainty_visualization.png"):
        """
        Visualize predictive uncertainty using confidence intervals and GP samples.

        Args:
            predictions (Dict[str, np.ndarray]): Dictionary containing:
                - "mean": Mean predictions.
                - "std": Standard deviations (uncertainty).
                - (Optional) "samples": Samples from the GP posterior.
            truth (np.ndarray): Ground truth values for comparison.
            save_filename (str): Filename to save the visualization. Default is "uncertainty_visualization.png".
        """
        mean, std = predictions["mean"], predictions["std"]
        samples = predictions.get("samples", None)

        if len(mean.shape) > 2 or len(std.shape) > 2:
            raise ValueError("Only 1D or 2D data visualizations are supported.")

        # Create the time/space grid (assumes sequential order)
        grid = np.arange(mean.shape[1])

        # Plot mean and confidence intervals
        plt.figure(figsize=(10, 6))
        plt.plot(grid, truth[0], label="Ground Truth", color="black", linestyle="-", linewidth=2)
        plt.plot(grid, mean[0], label="Mean Prediction", color="blue", linestyle="--", linewidth=2)
        plt.fill_between(
            grid,
            mean[0] - 1.96 * std[0],  # Lower bound CI
            mean[0] + 1.96 * std[0],  # Upper bound CI
            color="blue",
            alpha=0.3,
            label="95% Confidence Interval",
        )

        # Plot samples if available
        if samples is not None:
            for i, sample in enumerate(samples[:5]):  # Visualize up to 5 samples
                plt.plot(grid, sample, alpha=0.5, label=f"Sample {i + 1}")

        plt.xlabel("Grid Index / Time Step")
        plt.ylabel("Prediction Value")
        plt.title("Predictive Uncertainty Visualization")
        plt.legend()
        plt.grid(True)

        if self.visualizations_enabled:
            plt.savefig(save_filename)
            print(f"Visualization saved to {save_filename}.")
        plt.show()

    def compute_metrics(self, truth: np.ndarray, predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute evaluation metrics (RMSE, NLL, χ²).

        Args:
            truth (np.ndarray): Ground truth values (batch_size, grid_dim).
            predictions (Dict[str, np.ndarray]): Dictionary containing:
                - "mean": Mean predictions (batch_size, grid_dim).
                - "std": Standard deviations (batch_size, grid_dim).

        Returns:
            Dict[str, float]: Computed metrics.
        """
        mean = predictions["mean"]
        std = predictions["std"]
        metrics = {}

        # Root Mean Squared Error (RMSE)
        if "rmse" in self.metrics_to_compute:
            rmse = np.sqrt(np.mean((truth - mean) ** 2))
            metrics["rmse"] = rmse

        # Marginal Negative Log-Likelihood (NLL)
        if "marginal_nll" in self.metrics_to_compute:
            nll = -np.mean(
                -0.5 * np.log(2 * np.pi * std**2 + 1e-10)
                - 0.5 * ((truth - mean) ** 2) / (std**2 + 1e-10)
            )
            metrics["marginal_nll"] = nll

        # χ²-statistic
        if "chi_squared" in self.metrics_to_compute:
            chi_squared = np.mean(((truth - mean) ** 2) / (std**2 + 1e-10))
            metrics["chi_squared"] = chi_squared

        return metrics
