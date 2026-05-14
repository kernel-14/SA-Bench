## evaluation.py

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any
from torch import Tensor

class Evaluation:
    """
    Evaluation class to compute metrics (e.g., total variation, KL divergence) and visualize convergence results.
    """

    def __init__(self, model: Any, dataset: Tuple[Tensor, Tensor], config: Dict[str, Any]) -> None:
        """
        Initialize the Evaluation class.

        Args:
            model (Any): DiffusionModel instance.
            dataset (Tuple[Tensor, Tensor]): Tuple containing true dataset (X_0, training/test samples) and sampled data.
            config (Dict[str, Any]): Configuration dictionary containing output settings and evaluation metrics.
        """
        self.model = model
        self.dataset = dataset  # Tuple(x_true, y_sampled)
        self.config = config

        # Output directory for saving plots and checkpoints
        self.output_dir = config["output"]["directory"]
        self.save_plots = config["output"]["save_plots"]

        # Metrics to be evaluated
        self.metrics = config["evaluation"]["metrics"]

        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def compute_total_variation(self, y_sampled: Tensor, x_true: Tensor) -> float:
        """
        Compute Total Variation (TV) Distance between the sampled distribution and the target distribution.

        Args:
            y_sampled (Tensor): Sampled points from the reverse process.
            x_true (Tensor): True samples from the target data distribution.

        Returns:
            float: Computed TV distance.
        """
        # Convert tensors to numpy arrays
        y_sampled_np = y_sampled.cpu().numpy()
        x_true_np = x_true.cpu().numpy()

        # Discretize into histograms
        sampled_hist, bins = np.histogram(y_sampled_np, bins=100, density=True)
        true_hist, _ = np.histogram(x_true_np, bins=bins, density=True)

        # Compute TV distance (0.5 * L1 norm of differences)
        tv_distance = 0.5 * np.sum(np.abs(sampled_hist - true_hist))

        return tv_distance

    def compute_kl_divergence(self, y_sampled: Tensor, x_true: Tensor) -> float:
        """
        Compute KL Divergence between the sampled distribution and the target distribution.

        Args:
            y_sampled (Tensor): Sampled points from the reverse process.
            x_true (Tensor): True samples from the target data distribution.

        Returns:
            float: Computed KL divergence.
        """
        # Convert tensors to numpy arrays
        y_sampled_np = y_sampled.cpu().numpy()
        x_true_np = x_true.cpu().numpy()

        # Discretize into histograms
        sampled_hist, bins = np.histogram(y_sampled_np, bins=100, density=True)
        true_hist, _ = np.histogram(x_true_np, bins=bins, density=True)

        # Numerical stability: Avoid log(0) errors
        epsilon = 1e-12
        sampled_hist = sampled_hist + epsilon
        true_hist = true_hist + epsilon

        # Compute KL divergence
        kl_div = np.sum(sampled_hist * np.log(sampled_hist / true_hist))

        return kl_div

    def plot_convergence_results(self, results: Dict[str, List[float]]) -> None:
        """
        Plot and save convergence metrics like TV Distance and KL Divergence.

        Args:
            results (Dict[str, List[float]]): Dictionary of metrics containing values across iterations.
                                              Example: {"total_variation": [...], "kl_divergence": [...]}.
        """
        plt.figure(figsize=(10, 6))
        for metric_name, metric_values in results.items():
            plt.plot(range(len(metric_values)), metric_values, label=metric_name)

        plt.xlabel("Iteration")
        plt.ylabel("Metric Value")
        plt.title("Convergence Metrics")
        plt.legend()
        plt.grid()

        if self.save_plots:
            plt.savefig(os.path.join(self.output_dir, "convergence_plot.png"))
            print("[INFO] Convergence plot saved to convergence_plot.png")
        else:
            plt.show()
        plt.close()

    def evaluate(self, y_sampled: Tensor, x_true: Tensor, num_iterations: int) -> Dict[str, Any]:
        """
        Perform evaluation to compute metrics and plot convergence.

        Args:
            y_sampled (Tensor): Sampled points from the reverse process.
            x_true (Tensor): True samples from the target data distribution.
            num_iterations (int): Number of iterations/samples over which metrics are computed.

        Returns:
            Dict[str, Any]: Dictionary with computed metric values.
        """
        results = {}

        # Compute total variation distance if enabled in configuration
        if "total_variation" in self.metrics:
            tv_distances = []
            for iteration in range(num_iterations):
                tv_distance = self.compute_total_variation(y_sampled, x_true)
                tv_distances.append(tv_distance)
            results["total_variation"] = tv_distances

        # Compute KL divergence if enabled in configuration
        if "kl_divergence" in self.metrics:
            kl_distances = []
            for iteration in range(num_iterations):
                kl_divergence = self.compute_kl_divergence(y_sampled, x_true)
                kl_distances.append(kl_divergence)
            results["kl_divergence"] = kl_distances

        # Plot convergence results
        self.plot_convergence_results(results)

        return results

    def save_results(self, results: Dict[str, Any]) -> None:
        """
        Save evaluation results (metrics and values) to a file.

        Args:
            results (Dict[str, Any]): Dictionary containing metric values.
        """
        results_file = os.path.join(self.output_dir, "evaluation_results.json")

        with open(results_file, "w") as f:
            import json
            json.dump(results, f, indent=4)

        print(f"[INFO] Evaluation results saved to {results_file}")
