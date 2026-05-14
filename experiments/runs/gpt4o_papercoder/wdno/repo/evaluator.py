# evaluator.py
"""
Evaluator module for the Wavelet Diffusion Neural Operator (WDNO).
This class evaluates simulation and control performances via Mean Squared Error (MSE) and control objective metrics.
It also provides high-resolution reconstructed outputs and optional visualization of results.
"""

from typing import Dict, Any, List, Tuple
import torch
from torch import Tensor
from wavelet_transform import WaveletTransform
from utils import calculate_metrics


class Evaluator:
    """
    Evaluator class for prediction and control tasks on WDNO.
    Attributes:
        model: The trained WDNO model for inference.
        test_data: Wavelet-transformed test dataset containing initial conditions, parameters, etc.
        config: Evaluation configuration loaded from `config.yaml`.
    """

    def __init__(self, model, test_data: Dict[str, Any], config: Dict[str, Any]) -> None:
        """
        Initializes the evaluator with required resources including wavelet transform utilities.
        
        Args:
            model: The pre-trained WDNO model for simulation and control evaluation.
            test_data (Dict[str, Any]): Preprocessed wavelet-transformed test dataset.
            config (dict): Configuration dictionary.
        """
        self.model = model
        self.test_data = test_data
        self.config = config
        self.metrics = self.config["evaluation"]["metrics"]

        # Initialize wavelet transforms based on configuration
        self.wavelet_transform = WaveletTransform(
            wavelet_type=self.config["wavelet"]["basis_1d"], 
            mode=self.config["wavelet"]["mode_1d"]
        )

    def evaluate_simulation(self) -> Dict[str, float]:
        """
        Evaluates the model on simulation tasks by generating state trajectories and comparing with ground truth.

        Returns:
            dict: Dictionary containing MSE and any additional evaluation metrics.
        """
        mse_total = 0.0
        num_samples = 0

        print("Starting simulation evaluation...")
        with torch.no_grad():
            for sample in self.test_data["simulation"]:
                initial_condition = sample["initial_condition"]
                true_states = sample["true_states"]
                parameters = sample["parameters"]

                # Step 1: Generate predicted states
                predicted_wavelet_coeffs = self.model.forward(
                    initial_condition, {"parameters": parameters}
                )
                
                # Step 2: Inverse transform to recover physical domain results
                predicted_states = self.wavelet_transform.inverse_transform(predicted_wavelet_coeffs)

                # Step 3: Calculate MSE for the current sample
                mse = calculate_metrics(predicted_states, true_states)["mse"]
                mse_total += mse
                num_samples += 1

        avg_mse = mse_total / num_samples
        print(f"Simulation Evaluation Complete - Avg. MSE: {avg_mse:.6f}")
        return {"mse": avg_mse}

    def evaluate_control(self) -> Dict[str, float]:
        """
        Evaluates the model on control tasks by optimizing control signals and computing the control objective metric.

        Returns:
            dict: Dictionary containing control objective metrics and other relevant performance results.
        """
        control_objective_total = 0.0
        num_samples = 0

        print("Starting control evaluation...")
        with torch.no_grad():
            for sample in self.test_data["control"]:
                initial_condition = sample["initial_condition"]
                target_state = sample["target_state"]
                parameters = sample["parameters"]

                # Step 1: Initialize Gaussian noise for control sequences
                noise = torch.randn_like(initial_condition)

                # Step 2: Perform DDIM sampling with control-specific guidance
                optimized_wavelet_coeffs = self.model.sample(
                    time_steps=self.config["training"]["ddim_steps"],
                    noise=noise,
                    conditions={
                        "initial_state": initial_condition,
                        "parameters": parameters,
                        "gradient": self._calculate_control_gradient(initial_condition, target_state),
                    },
                )

                # Step 3: Inverse transform to recover optimized control sequence
                optimized_controls = self.wavelet_transform.inverse_transform(optimized_wavelet_coeffs)

                # Step 4: Evaluate the control objective metric
                control_objective = self._compute_control_objective(
                    optimized_controls, target_state, parameters
                )
                control_objective_total += control_objective
                num_samples += 1

        avg_control_objective = control_objective_total / num_samples
        print(f"Control Evaluation Complete - Avg. Objective: {avg_control_objective:.6f}")
        return {"control_objective": avg_control_objective}

    def visualize_results(self, predictions: Tensor, ground_truth: Tensor) -> None:
        """
        Optional visualization for simulation or control results.

        Args:
            predictions (Tensor): Generated output by the model (e.g., trajectories or controls).
            ground_truth (Tensor): Corresponding ground truth data for comparison.
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(predictions.cpu().numpy(), label="Predictions", linestyle="--")
        plt.plot(ground_truth.cpu().numpy(), label="Ground Truth", alpha=0.7)
        plt.title("Model Predictions vs Ground Truth")
        plt.xlabel("Time Steps")
        plt.ylabel("States")
        plt.legend()
        plt.grid(True)
        plt.show()

    def _calculate_control_gradient(self, initial_condition: Tensor, target_state: Tensor) -> Tensor:
        """
        Computes the gradient of the control objective with respect to the control wavelet coefficients.

        Args:
            initial_condition (Tensor): Initial state wavelet coefficients.
            target_state (Tensor): Target state wavelet coefficients.

        Returns:
            Tensor: Gradient of the control objective.
        """
        # Placeholder gradient calculation (would typically involve automatic differentiation)
        return -(initial_condition - target_state)

    def _compute_control_objective(self, controls: Tensor, target_state: Tensor, parameters: Dict[str, Any]) -> float:
        """
        Computes the control objective as defined in the paper based on deviation and energy cost.

        Args:
            controls (Tensor): Optimized control sequence generated by the model.
            target_state (Tensor): Target state to be achieved.
            parameters (Dict[str, Any]): Additional PDE parameters (e.g., physical constants).

        Returns:
            float: The computed control objective value.
        """
        # Energy cost weight from the configuration
        alpha = self.config["evaluation"].get("control_objective_weight", 1.0)

        # Compute deviation from target state
        deviation_term = torch.mean((controls - target_state) ** 2).item()

        # Compute control energy cost
        energy_term = torch.mean(controls ** 2).item()

        # Total control objective
        return alpha * deviation_term + energy_term
