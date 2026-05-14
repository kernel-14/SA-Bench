## dataset_loader.py

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, List, Any, Dict, Callable
from utils.differential_equations import PDE_REGISTRY
from utils.gradient_computation import GradientComputation
import yaml


class DatasetLoader:
    """
    DatasetLoader class responsible for loading synthetic datasets for ODEs/PDEs,
    precomputing gradients, and splitting data into train/validation/test subsets.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initializes the DatasetLoader with configurations from a YAML file.

        Args:
        - config_path (str): Path to the configuration YAML file.
        """
        # Load configuration
        with open(config_path, "r") as config_file:
            self.config = yaml.safe_load(config_file)

        # Extract relevant configurations
        self.gradient_method = self.config.get("data", {}).get("gradient_method", "AD")
        self.split_ratios = self.config.get("data", {}).get("train_val_test_split", [0.7, 0.15, 0.15])
        self.equation_configs = self.config.get("experiment", {}).get("equations", {})

        # Validate split ratio
        if sum(self.split_ratios) != 1.0:
            raise ValueError("Train/Validation/Test split ratios must sum to 1.")

    def load_data(self, equation_name: str) -> Tuple[torch.Tensor, torch.Tensor, None]:
        """
        Loads synthetic data for the specified equation.

        Args:
        - equation_name (str): Name of the equation (e.g., 'composite_harmonic_oscillator').

        Returns:
        - inputs (torch.Tensor): Model inputs (X).
        - solutions (torch.Tensor): Ground-truth solutions (Y).
        - gradients (None): Empty gradient placeholder before precomputation.
        """
        # Ensure the equation exists in configuration and registry
        if equation_name not in self.equation_configs:
            raise ValueError(f"Equation '{equation_name}' is not defined in config.yaml.")
        if equation_name not in PDE_REGISTRY:
            raise ValueError(f"Equation '{equation_name}' is not implemented in utils/differential_equations.py.")

        equation_config = self.equation_configs[equation_name]
        solver_function = PDE_REGISTRY[equation_name]

        # Extract parameter ranges and domains
        parameter_ranges = equation_config.get("parameters", {})
        time_domain = equation_config.get("time_domain", [0, 1])
        spatial_domain = equation_config.get("spatial_domain", [0, 1])
        time_steps = equation_config.get("time_steps", 100)
        space_steps = equation_config.get("space_steps", 100)

        # Generate time and space grids
        t = torch.linspace(time_domain[0], time_domain[1], time_steps)
        x = torch.linspace(spatial_domain[0], spatial_domain[1], space_steps) if "spatial_domain" in equation_config else None

        # Generate random parameter samples
        parameter_samples = self._generate_parameter_samples(parameter_ranges, 2000)  # Default: 2,000 samples

        # Solve the equation for each parameter combination
        solutions = []
        for params in parameter_samples:
            if x is not None:  # For PDEs
                u = solver_function(**params, x=x, t=t)
            else:  # For ODEs
                u = solver_function(**params, t=t)
            solutions.append(u.unsqueeze(0))  # Add batch dimension

        # Convert to tensors
        inputs = torch.tensor(parameter_samples, dtype=torch.float32)
        solutions = torch.cat(solutions, dim=0)

        return inputs, solutions, None  # Gradients will be computed later

    def _generate_parameter_samples(self, parameter_ranges: Dict[str, List[float]], num_samples: int) -> List[Dict[str, float]]:
        """
        Generates random samples of parameters based on uniform distributions specified in config.yaml.

        Args:
        - parameter_ranges (Dict[str, List[float]]): Parameter bounds as `[low, high]`.
        - num_samples (int): Number of random samples to generate.

        Returns:
        - List[Dict[str, float]]: Random parameter samples.
        """
        samples = []
        for _ in range(num_samples):
            sample = {param: np.random.uniform(low, high) for param, (low, high) in parameter_ranges.items()}
            samples.append(sample)
        return samples

    def precompute_gradients(self, method: str = None) -> Any:
        """
        Precomputes gradients using the specified method (AD or FD).

        Args:
        - method (str): Gradient computation method ('AD' or 'FD'). Default is the YAML config value.

        Returns:
        - gradients (torch.Tensor): Precomputed Jacobians (∂Y/∂P).
        """
        method = method or self.gradient_method

        # Validate gradient method
        if method not in ["AD", "FD"]:
            raise ValueError("Gradient computation method must be either 'AD' (Automatic Differentiation) or 'FD' (Finite Difference).")

        gradients = []
        for equation_name, equation_config in self.equation_configs.items():
            inputs, solutions, _ = self.load_data(equation_name)
            solver_function = PDE_REGISTRY[equation_name]

            if method == "AD":
                # Compute gradients using PyTorch automatic differentiation
                gradients.append(
                    GradientComputation.compute_automatic_differentiation(solver_function, inputs)
                )
            elif method == "FD":
                # Compute gradients using finite difference approximation
                gradients.append(
                    GradientComputation.compute_finite_difference(solver_function, inputs)
                )

        return gradients

    def split_data(self, percentage_train_val_test: List[float] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Splits data into train, validation, and test sets.

        Args:
        - percentage_train_val_test (List[float]): Ratios for splitting data. Defaults to the YAML value.

        Returns:
        - Tuple[DataLoader, DataLoader, DataLoader]: Train, Validation, Test sets.
        """
        percentage_train_val_test = percentage_train_val_test or self.split_ratios
        if sum(percentage_train_val_test) != 1.0:
            raise ValueError("Percentage ratios must sum to 1.0.")

        # Load data and shuffle
        all_data = []
        for equation_name in self.equation_configs.keys():
            inputs, solutions, gradients = self.load_data(equation_name)
            all_data.append((inputs, solutions))

        inputs, solutions = torch.cat([d[0] for d in all_data]), torch.cat([d[1] for d in all_data])
        dataset_size = len(inputs)
        indices = torch.randperm(dataset_size)

        # Split indices
        num_train = int(percentage_train_val_test[0] * dataset_size)
        num_val = int(percentage_train_val_test[1] * dataset_size)
        train_indices = indices[:num_train]
        val_indices = indices[num_train:num_train + num_val]
        test_indices = indices[num_train + num_val:]

        # Create DataLoaders
        train_set = DataLoader(TensorDataset(inputs[train_indices], solutions[train_indices]), batch_size=32, shuffle=True)
        val_set = DataLoader(TensorDataset(inputs[val_indices], solutions[val_indices]), batch_size=32, shuffle=False)
        test_set = DataLoader(TensorDataset(inputs[test_indices], solutions[test_indices]), batch_size=32, shuffle=False)

        return train_set, val_set, test_set
