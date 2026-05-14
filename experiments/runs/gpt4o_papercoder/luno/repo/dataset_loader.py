## dataset_loader.py

import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Any, Tuple

class DatasetLoader:
    """
    DatasetLoader class for loading and preparing datasets for low-data and OOD regimes.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DatasetLoader with provided configuration.

        Args:
            config (dict): Configuration dictionary, typically loaded from 'config.yaml'.
        """
        self.config = config
        self.low_data_config = config.get("data", {}).get("low_data", {})
        self.ood_config = config.get("data", {}).get("ood", {})

    def load_low_data(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Load and prepare the low-data regime dataset.

        Returns:
            dict: A dictionary containing training and test datasets.
        """
        # Retrieve config parameters
        num_train = self.low_data_config.get("trajectories", 25)
        num_test = self.low_data_config.get("test_trajectories", 250)
        grid_res = self.low_data_config.get("grid_resolution", 256)
        time_res = self.low_data_config.get("temporal_resolution", 59)

        # Generate trajectories using PDE solutions
        trajectories = self._generate_pde_trajectories(grid_res, time_res, num_train + num_test)

        # Split into training and testing
        train_data = trajectories[:num_train]
        test_data = trajectories[num_train:num_train + num_test]

        # Return training and test datasets
        return {
            "train": {
                "inputs": np.array([data["inputs"] for data in train_data]),
                "outputs": np.array([data["outputs"] for data in train_data])
            },
            "test": {
                "inputs": np.array([data["inputs"] for data in test_data]),
                "outputs": np.array([data["outputs"] for data in test_data])
            }
        }

    def load_ood_data(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Load and prepare OOD datasets with specified perturbations.

        Returns:
            dict: A dictionary containing different OOD datasets.
        """
        # Retrieve config parameters
        num_base_traj = self.ood_config.get("trajectories_base", 1000)
        grid_res = self.ood_config.get("grid_resolution", 100)
        time_res = self.ood_config.get("temporal_resolution", 59)
        perturbations = self.ood_config.get("perturbations", {})

        # Generate base trajectories
        base_data = self._generate_pde_trajectories(grid_res, time_res, num_base_traj)

        # Apply perturbations to create OOD datasets
        ood_datasets = {
            "Base": base_data,
            "Flip": self._add_ood_perturbations(base_data, perturbation_type="velocity_flip") 
                     if perturbations.get("velocity_flip", False) else base_data,
            "Pos": self._add_ood_perturbations(base_data, perturbation_type="heat_source")
                   if perturbations.get("heat_source_sink", False) else base_data,
            "Pos-Neg": self._add_ood_perturbations(base_data, perturbation_type="heat_sink"),
            "Pos-Neg-Flip": self._add_ood_perturbations(
                base_data, perturbation_type="combined"
            )
        }

        return ood_datasets

    def _generate_pde_trajectories(self, grid_res: int, time_res: int, num_trajectories: int) -> list:
        """
        Generate numerical solutions for PDEs to create trajectories.

        Args:
            grid_res (int): Spatial resolution of the grid.
            time_res (int): Number of time steps to sample.
            num_trajectories (int): Number of trajectories to generate.

        Returns:
            list: A list of trajectory dictionaries with 'inputs' and 'outputs'.
        """
        trajectories = []
        x = np.linspace(0, 1, grid_res)
        t = np.linspace(0, 1, time_res)

        for _ in range(num_trajectories):
            # Example: Solve Burgers' equation
            initial_condition = np.sin(2 * np.pi * x) + np.random.normal(0, 0.1, size=x.shape)

            def burgers_eq(_, u):
                return -u * np.gradient(u, edge_order=2) + 0.01 * np.gradient(np.gradient(u, edge_order=2))

            sol = solve_ivp(
                burgers_eq, [0, 1], initial_condition,
                t_eval=t, method="RK45", atol=1e-6, rtol=1e-6
            )

            trajectories.append({
                "inputs": sol.y[:, :-1],  # Trajectory minus the last step
                "outputs": sol.y[:, 1:]  # Shifted trajectory for prediction
            })

        return trajectories

    def _add_ood_perturbations(self, base_data: list, perturbation_type: str) -> list:
        """
        Add synthetic perturbations to base trajectories for OOD cases.

        Args:
            base_data (list): Base data trajectories.
            perturbation_type (str): Type of perturbation to apply.

        Returns:
            list: Perturbed dataset.
        """
        perturbed_trajectories = []

        for trajectory in base_data:
            inputs, outputs = trajectory["inputs"], trajectory["outputs"]

            if perturbation_type == "velocity_flip":
                # Flip velocity at the center of the grid
                perturbed_inputs = inputs * -1 if np.random.rand() > 0.5 else inputs

            elif perturbation_type == "heat_source":
                # Add random Gaussian heat sources
                heat_source = np.exp(-((inputs - 0.5) ** 2) / 0.02)
                perturbed_inputs = inputs + heat_source

            elif perturbation_type == "heat_sink":
                # Add Gaussian blobs for heat sink
                heat_sink = np.exp(-((inputs - 0.75) ** 2) / 0.02)
                perturbed_inputs = inputs - heat_sink

            elif perturbation_type == "combined":
                # Combine velocity flip and both heat sink/source
                vel_flip = inputs * -1 if np.random.rand() > 0.5 else inputs
                heat_source = np.exp(-((inputs - 0.5) ** 2) / 0.02)
                heat_sink = np.exp(-((inputs - 0.75) ** 2) / 0.02)
                perturbed_inputs = vel_flip + heat_source - heat_sink

            else:
                # No perturbations
                perturbed_inputs = inputs

            perturbed_trajectories.append({
                "inputs": perturbed_inputs,
                "outputs": outputs
            })

        return perturbed_trajectories
