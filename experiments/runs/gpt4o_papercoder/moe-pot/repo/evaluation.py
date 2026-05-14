## evaluation.py

import torch
from typing import Dict, Tuple
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from utilities import log_metrics, inject_gaussian_noise
from moe_pot_model import MoEPOTModel


class Evaluation:
    """
    Evaluation class for assessing the MoE-POT model performance, error propagation, and interpretability analysis.
    """

    def __init__(self, model: MoEPOTModel, test_data: torch.utils.data.Dataset, config: Dict):
        """
        Initialize the evaluation utilities.
        
        Args:
            model (MoEPOTModel): Trained MoE-POT model instance.
            test_data (Dataset): Test dataset for evaluation.
            config (Dict): Configuration loaded from `config.yaml`.
        """
        self.model = model
        self.test_data = test_data
        self.config = config

        # Configuration settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
        self.model.to(self.device)
        self.batch_size = config["training"].get("batch_size", 20)
        self.rollout_steps = config["evaluation"].get("rollout_steps", 100)

        # Initialize DataLoader
        self.test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, pin_memory=True)

    def evaluate_l2re(self) -> Dict[str, float]:
        """
        Evaluate the model's L2 Relative Error (L2RE) on the test dataset.

        Returns:
            Dict[str, float]: Aggregated metrics, including mean, median, and percentile L2RE scores.
        """
        self.model.eval()
        total_l2re = []
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Forward pass to predict next timestep outputs
                predictions = self.model(inputs)
                
                # Compute L2RE per sample
                l2re_batch = torch.norm(predictions - targets, p=2, dim=-1) / torch.norm(targets, p=2, dim=-1)
                total_l2re.extend(l2re_batch.cpu().tolist())

        # Compute aggregate metrics
        mean_l2re = np.mean(total_l2re)
        median_l2re = np.median(total_l2re)
        percentile_95_l2re = np.percentile(total_l2re, 95)

        # Log results for debugging
        metrics = {
            "mean_l2re": mean_l2re,
            "median_l2re": median_l2re,
            "percentile_95_l2re": percentile_95_l2re,
        }
        print(metrics)
        return metrics

    def compute_rollout_error(self, initial_inputs: torch.Tensor, ground_truth: torch.Tensor) -> Dict[str, float]:
        """
        Compute the rollout error over multiple timesteps to analyze error propagation.

        Args:
            initial_inputs (Tensor): Initial input tensor [Batch, Time, Channels, Height, Width].
            ground_truth (Tensor): Ground truth tensor [Batch, TimeRollout, Channels, Height, Width].

        Returns:
            Dict[str, float]: Error progression and aggregate rollout metrics.
        """
        self.model.eval()
        rollout_errors = []
        inputs = initial_inputs.to(self.device)
        with torch.no_grad():
            for t in range(self.rollout_steps):
                # Predict next timestep
                predictions = self.model(inputs)

                # Compute L2RE for current timestep
                l2re = torch.norm(predictions - ground_truth[:, t], p=2) / torch.norm(ground_truth[:, t], p=2)
                rollout_errors.append(l2re.item())

                # Update inputs by appending predictions for next step
                inputs = torch.cat([inputs[:, 1:], predictions.unsqueeze(1)], dim=1)

        # Compute aggregate metrics
        mean_rollout_error = np.mean(rollout_errors)
        max_rollout_error = np.max(rollout_errors)

        # Log error progression
        metrics = {
            "mean_rollout_error": mean_rollout_error,
            "max_rollout_error": max_rollout_error,
        }
        print(f"Rollout Metrics: {metrics}")
        return metrics

    def analyze_router_gate_behavior(self) -> Dict[str, float]:
        """
        Analyze router-gating network behavior and evaluate its interpretability.

        Returns:
            Dict[str, float]: Dataset classification accuracy and expert usage patterns.
        """
        self.model.eval()
        dataset_accuracies = {}
        expert_usage = {}

        for inputs, targets in self.test_loader:
            inputs = inputs.to(self.device)

            # Forward pass to extract router gating activations
            with torch.no_grad():
                router_logits = self.model.router_gating_network(inputs.mean(dim=1))

            # Compute activations and aggregate across batches
            softmax_weights = torch.softmax(router_logits, dim=-1)
            top_k_activations = torch.topk(softmax_weights, self.config["architecture"].get("top_k", 4), dim=-1).values

            # Compute classification metrics and expert statistics
            dataset_accuracies = self._evaluate_classification_accuracy(softmax_weights)
            expert_usage = self._compute_expert_utilization_distribution(softmax_weights)

        # Log aggregated evaluation metrics
        print(f"Analysis Results - Dataset Accuracy: {dataset_accuracies}, Expert Usage: {expert_usage}")

        return {
            "dataset_classification_accuracy": np.mean(list(dataset_accuracies.values())),
            "expert_usage_distribution": expert_usage,
        }

    def _evaluate_classification_accuracy(self, softmax_weights: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate the classification accuracy of the router-gating network based on softmax weights.

        Args:
            softmax_weights (Tensor): Activation weights from router gating layers.

        Returns:
            Dict[str, float]: Classification accuracy per dataset.
        """
        dataset_accuracies = {}
        for dataset_name, average_weights in self._compute_dataset_average_weights().items():
            distances = torch.norm(softmax_weights - average_weights, p=2, dim=-1)
            pred_class = torch.argmin(distances, dim=-1)
            correct_classifications = torch.sum(pred_class == dataset_name).item()
            dataset_accuracy = correct_classifications / len(softmax_weights)
            dataset_accuracies[dataset_name] = dataset_accuracy
        return dataset_accuracies

    def _compute_dataset_average_weights(self) -> Dict[str, torch.Tensor]:
        """
        Precompute the average router-gating network weights for each dataset.

        Returns:
            Dict[str, Tensor]: Average activation weights for each dataset.
        """
        averages = {}
        for dataset_name, dataset_loader in self._load_datasets_for_analysis():
            with torch.no_grad():
                inputs, _ = next(iter(dataset_loader))
                router_logits = self.model.router_gating_network(inputs.mean(dim=1))
                softmax_weights = torch.softmax(router_logits, dim=-1)
                averages[dataset_name] = softmax_weights.mean(dim=0)
        return averages

    def _compute_expert_utilization_distribution(self, weights: torch.Tensor) -> Dict[str, float]:
        """
        Compute the distribution of expert utilization.

        Args:
            weights (Tensor): Router weights for expert selection.

        Returns:
            Dict[str, float]: Per-expert utilization statistics.
        """
        utilization = torch.sum(weights, dim=0).cpu().numpy()  # Sum over batches
        total_utilization = torch.sum(utilization).item()
        return {
            f"Expert_{i}": utilization[i] / total_utilization
            for i in range(weights.size(-1))
        }
