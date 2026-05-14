```python
## evaluation.py

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from model import Model
import yaml


class Evaluation:
    """
    Evaluation class for SC-FNO and its variants.
    Responsible for assessing model performance through metrics and visualizations.
    """

    def __init__(self, model: Model, test_loader: DataLoader, config_path: str = "config/config.yaml"):
        """
        Initializes the evaluation pipeline.

        Args:
        - model (Model): Trained SC-FNO model instance.
        - test_loader (DataLoader): DataLoader for test data.
        - config_path (str): Path to the YAML configuration file.
        """
        self.model = model.eval()  # Set model to evaluation mode
        self.test_loader = test_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Load configuration
        with open(config_path, "r") as config_file:
            self.config = yaml.safe_load(config_file)

        # Load evaluation-specific settings
        self.output_dir = self.config.get("experiment", {}).get("output_dir", "results/")
        os.makedirs(self.output_dir, exist_ok=True)

    def evaluate_metrics(self) -> Dict[str, float]:
        """
        Computes quantitative metrics (R² and relative L² error) for solutions and sensitivities.

        Returns:
        - Dict[str, float]: Dictionary containing computed metrics:
          {
              "solution_r2": float,
              "solution_l2": float,
              "sensitivity_r2": float,
              "sensitivity_l2": float
          }
        """
        solution_r2_scores = []
        solution_l2_errors = []
        sensitivity_r2_scores = []
        sensitivity_l2_errors = []

        with torch.no_grad():
            for batch in self.test_loader:
                inputs, solutions_true, gradients_true = batch
                inputs, solutions_true, gradients_true = (
                    inputs.to(self.device),
                    solutions_true.to(self.device),
                    gradients_true.to(self.device),
                )

                # Model predictions
                solutions_pred = self.model.forward(inputs)
                sensitivities_pred = self.model.compute_sensitivities(inputs)

                # Compute R² and Relative L² for solutions
                solution_r2 = self._compute_r2(solutions_true, solutions_pred)
                solution_l2 = self._compute_relative_l2(solutions_true, solutions_pred)
                solution_r2_scores.append(solution_r2)
                solution_l2_errors.append(solution_l2)

                # Compute R² and Relative L² for sensitivities
                sensitivity_r2 = self._compute_r2(gradients_true, sensitivities_pred)
                sensitivity_l2 = self._compute_relative_l2(gradients_true, sensitivities_pred)
                sensitivity_r2_scores.append(sensitivity_r2)
                sensitivity_l2_errors.append(sensitivity_l2)

        # Aggregate results
        metrics = {
            "solution_r2": np.mean(solution_r2_scores),
            "solution_l2": np.mean(solution_l2_errors),
            "sensitivity_r2": np.mean(sensitivity_r2_scores),
            "sensitivity_l2": np.mean(sensitivity_l2_errors),
        }

        # Save the metrics for reproducibility
        self._save_metrics(metrics, os.path.join(self.output_dir, "evaluation_metrics.json"))

        return metrics

    def evaluate_generalization(self) -> Dict[str, float]:
        """
        Evaluates the model's generalization performance under extrapolated test conditions.

        Returns: 
        - Dict[str, float]: Metrics for original and perturbed condition ranges.
        """
        perturbation_ratio = self.config.get("experiment", {}).get("perturbation_ratio", 0.4)
        
        metrics_original = self.evaluate_metrics()
        perturbed_loader = self._perturb_test_loader(perturbation_ratio)
        self.test_loader = perturbed_loader  # Replace loader temporarily
        
        metrics_perturbed = self.evaluate_metrics()
        self.test_loader = self.test_loader  # Reset to the original loader
        
        generalization_metrics = {
            "solution_r2_delta-original/results":
        }