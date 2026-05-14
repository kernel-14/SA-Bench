# dataset_loader.py

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, Optional, List


class DatasetLoader:
    """DatasetLoader handles loading, preprocessing, normalizing, and batching the dataset
    to be used for training the Robotic World Model (RWM) and policy optimization (MBPO-PPO)."""

    def __init__(self, config: dict, dataset_path: str):
        """
        Initialize the DatasetLoader instance.

        Args:
            config (dict): Configuration dictionary parsed from config.yaml.
            dataset_path (str): Path to the raw/preprocessed dataset.
        """
        self.config = config
        self.dataset_path = dataset_path
        self.history_horizon = config["training"]["history_horizon"]  # `M` from config.yaml
        self.forecast_horizon = config["training"]["forecast_horizon"]  # `N` from config.yaml
        self.batch_size = config["training"]["batch_size"]
        self.normalization_mode = "z-score"  # Default normalization type

    def load_data(self) -> Dict[str, DataLoader]:
        """
        Loads and preprocesses the dataset files, creating DataLoaders for training, validation, and testing.

        Returns:
            Dict[str, DataLoader]: Dictionary containing training, validation, and testing DataLoaders.
        """
        # Load raw dataset (example: train, validation, and test splits)
        raw_data = {}
        for split in ["train", "val", "test"]:
            file_path = os.path.join(self.dataset_path, f"{split}.npz")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Dataset file not found: {file_path}")
            raw_data[split] = np.load(file_path)

        # Normalize the data
        for split in raw_data:
            raw_data[split] = self._normalize_data(raw_data[split], mode=self.normalization_mode)

        # Generate sliding window batches (history and forecast horizons)
        data_loaders = {
            split: self._generate_batches(raw_data[split], self.history_horizon, self.forecast_horizon)
            for split in raw_data
        }

        return data_loaders

    def _normalize_data(self, dataset: dict, mode: str = "z-score") -> dict:
        """
        Normalize the loaded dataset based on the chosen mode.

        Args:
            dataset (dict): Dictionary containing 'observations', 'actions', and 'privileged' arrays.
            mode (str): Normalization mode, either "z-score" (default) or "min-max".

        Returns:
            dict: Dictionary containing normalized 'observations', 'actions', and 'privileged' arrays.
        """
        normalized_data = {}
        for key in ["observations", "actions", "privileged"]:
            raw_data = dataset[key]
            if mode == "z-score":
                mean = np.mean(raw_data, axis=0)
                std = np.std(raw_data, axis=0) + 1e-8  # Avoid division by zero
                normalized_data[key] = (raw_data - mean) / std
            elif mode == "min-max":
                min_val = np.min(raw_data, axis=0)
                max_val = np.max(raw_data, axis=0)
                normalized_data[key] = (raw_data - min_val) / (max_val - min_val + 1e-8)
            else:
                raise ValueError(f"Unsupported normalization mode: {mode}")
        return normalized_data

    def _generate_batches(self, data: dict, M: int, N: int) -> DataLoader:
        """
        Create PyTorch DataLoader with sliding window batches for history and forecast horizons.

        Args:
            data (dict): Dictionary containing preprocessed 'observations', 'actions', and 'privileged' data.
            M (int): History horizon.
            N (int): Forecast horizon.

        Returns:
            DataLoader: PyTorch DataLoader with historical context and forecast targets for training/validation/testing.
        """
        observations = data["observations"]
        actions = data["actions"]
        privileged_info = data["privileged"]

        num_samples = len(observations) - (M + N - 1)
        if num_samples <= 0:
            raise ValueError("Insufficient samples for the given history and forecast horizons.")

        # Prepare sliding windows
        history_obs = []
        history_actions = []
        forecast_obs = []
        forecast_privileged = []

        for i in range(num_samples):
            history_obs.append(observations[i : i + M])
            history_actions.append(actions[i : i + M])
            forecast_obs.append(observations[i + M : i + M + N])
            forecast_privileged.append(privileged_info[i + M : i + M + N])

        history_obs = np.array(history_obs)
        history_actions = np.array(history_actions)
        forecast_obs = np.array(forecast_obs)
        forecast_privileged = np.array(forecast_privileged)

        # Convert to PyTorch tensors
        dataset = TensorDataset(
            torch.tensor(history_obs, dtype=torch.float32),
            torch.tensor(history_actions, dtype=torch.float32),
            torch.tensor(forecast_obs, dtype=torch.float32),
            torch.tensor(forecast_privileged, dtype=torch.float32),
        )

        # Return DataLoader
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def apply_noise(self, data: np.ndarray, level: float = 0.1) -> np.ndarray:
        """
        Add Gaussian noise to the dataset for robustness evaluation.

        Args:
            data (np.ndarray): Data array to which noise is applied.
            level (float): Standard deviation of Gaussian noise to add.

        Returns:
            np.ndarray: Noisy dataset.
        """
        noise = np.random.normal(0, level, size=data.shape)
        return data + noise
