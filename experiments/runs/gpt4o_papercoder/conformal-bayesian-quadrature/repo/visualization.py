# visualization.py

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

class Visualization:
    """
    Provides visualization utilities for generating intuitive plots of the results 
    from the evaluation of decision rules on synthetic and real datasets.
    """

    def __init__(self) -> None:
        """
        Initialize the Visualization class.
        No specific configurations are required at initialization.
        """
        pass

    def plot_loss_histogram(self, losses: np.ndarray, method_names: List[str]) -> None:
        """
        Plot histogram of loss values for each decision rule method.

        Args:
            losses (np.ndarray): 2D Array of loss values across trials. Shape: (num_trials, num_methods).
            method_names (List[str]): Names of the decision rule methods. Length must match number of methods.
        
        Returns:
            None
        """
        if losses.ndim != 2 or len(method_names) != losses.shape[1]:
            raise ValueError("Losses must be a 2D array with a matching number of method names.")

        plt.figure(figsize=(10, 6))
        num_methods = losses.shape[1]
        colors = plt.cm.tab10.colors  # Use default colormap for distinct method colors

        # Plot histograms for each method
        for i in range(num_methods):
            plt.hist(
                losses[:, i],
                bins=30,
                alpha=0.7,
                color=colors[i % len(colors)],
                label=method_names[i],
                density=True,
            )

        # Add labels, title, and legend
        plt.xlabel("Loss Value", fontsize=12)
        plt.ylabel("Frequency (Density)", fontsize=12)
        plt.title("Histogram of Losses Across Decision Rule Methods", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_risk_comparison(self, results: Dict[str, Dict[str, float]]) -> None:
        """
        Plot comparison of relative failure rates and average prediction set size for methods.

        Args:
            results (dict): Dictionary containing "failure_rate" and "average_set_size".
                Example:
                {
                    "failure_rate": {"Bayesian HPD": 0.03, "CRC": 0.21, "RCPS": 0.0},
                    "average_set_size": {"Bayesian HPD": 3.04, "CRC": 2.92, "RCPS": 3.57}
                }
        
        Returns:
            None
        """
        if not ("failure_rate" in results and "average_set_size" in results):
            raise ValueError("Results must contain 'failure_rate' and 'average_set_size' keys.")

        # Extract data
        methods = list(results["failure_rate"].keys())
        failure_rates = [results["failure_rate"][method] for method in methods]
        set_sizes = [results["average_set_size"][method] for method in methods]

        # Plot failure rates
        plt.figure(figsize=(12, 6))
        x_indices = np.arange(len(methods))

        plt.subplot(1, 2, 1)
        plt.bar(x_indices, failure_rates, color="skyblue", alpha=0.8)
        plt.xticks(x_indices, methods, rotation=15, fontsize=10)
        plt.ylabel("Failure Rate (Proportion)", fontsize=12)
        plt.title("Relative Failure Rate Comparison", fontsize=14)
        plt.grid(alpha=0.3, linestyle="--", axis="y")

        # Plot prediction set sizes
        plt.subplot(1, 2, 2)
        plt.bar(x_indices, set_sizes, color="salmon", alpha=0.8)
        plt.xticks(x_indices, methods, rotation=15, fontsize=10)
        plt.ylabel("Average Prediction Set Size", fontsize=12)
        plt.title("Prediction Set Size Comparison", fontsize=14)
        plt.grid(alpha=0.3, linestyle="--", axis="y")

        plt.tight_layout()
        plt.show()

    def plot_dirichlet_loss_density(self, samples: np.ndarray, alpha_values: List[float]) -> None:
        """
        Plot kernel density estimation (KDE) of posterior loss bounds (L^+) for various alpha values.

        Args:
            samples (np.ndarray): Array of posterior expected loss samples for all settings of `alpha`.
                                  Shape: (num_alpha_values, num_samples_per_alpha).
            alpha_values (List[float]): List of alpha thresholds corresponding to the rows in `samples`.
        
        Returns:
            None
        """
        if samples.ndim != 2 or samples.shape[0] != len(alpha_values):
            raise ValueError(
                "Samples must be a 2D array where rows match the number of alpha values."
            )

        plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(alpha_values)))  # Colormap for alpha values

        for i, alpha in enumerate(alpha_values):
            # Compute KDE for each alpha value
            density = np.histogram(samples[i], bins=100, density=True)
            bin_centers = (density[1][1:] + density[1][:-1]) / 2
            plt.plot(bin_centers, density[0], label=f"Alpha={alpha:.2f}", color=colors[i])

        # Add labels, legend, and title
        plt.xlabel("L^+ (Expected Loss)", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.title("Posterior Density of L^+ (Dirichlet Sampled)", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_calibration_vs_deployment(self, metrics: Dict[str, List[float]]) -> None:
        """
        Plot comparison of calibration and deployment failure rates across trials.

        Args:
            metrics (dict): Dictionary containing failure rates for calibration and deployment:
                - "calibration_failure_rate": List of failure rates during calibration.
                - "deployment_failure_rate": List of failure rates during deployment.

        Returns:
            None
        """
        if not ("calibration_failure_rate" in metrics and "deployment_failure_rate" in metrics):
            raise ValueError("Metrics must contain 'calibration_failure_rate' and 'deployment_failure_rate'.")

        calibration_rates = metrics["calibration_failure_rate"]
        deployment_rates = metrics["deployment_failure_rate"]

        if len(calibration_rates) != len(deployment_rates):
            raise ValueError("Calibration and deployment failure rates must have the same length.")

        plt.figure(figsize=(10, 6))
        trials = np.arange(len(calibration_rates))

        # Plot calibration and deployment failure rates
        plt.plot(trials, calibration_rates, label="Calibration", color="blue", linestyle="--", linewidth=2)
        plt.plot(trials, deployment_rates, label="Deployment", color="red", linestyle="-", linewidth=2)

        # Add labels, legend, and title
        plt.xlabel("Trial Index", fontsize=12)
        plt.ylabel("Failure Rate (Proportion)", fontsize=12)
        plt.title("Calibration vs Deployment Failure Rates", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
