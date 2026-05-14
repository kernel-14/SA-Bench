# dataset_loader.py
import numpy as np
import torch
from typing import Tuple, Dict, Any


class DatasetLoader:
    """
    DatasetLoader handles loading and preprocessing of datasets. It is responsible for generating synthetic
    Gaussian data based on the configuration supplied, optionally preprocessing it, and returning
    PyTorch-compatible Tensors.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the dataset loader with the given configuration.

        Args:
            config (Dict[str, Any]): Loaded configuration dict that includes dataset specifications such as
                                     type, dimensions, and sample sizes.
        """
        self.config = config
        self.dataset_config = config.get("dataset", {})
        self.type = self.dataset_config.get("type", "Gaussian")
        self.dimensions = self.dataset_config.get("dimensions", 10)  # Default dimension
        self.training_samples = self.dataset_config.get("training_samples", 10000)
        self.testing_samples = self.dataset_config.get("testing_samples", 2000)

        # Internal validation
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """
        Validate the dataset configuration to ensure all required parameters are valid.

        Raises:
            ValueError: If essential dataset parameters are invalid.
        """
        if not isinstance(self.type, str) or self.type.lower() != "gaussian":
            raise ValueError(f"Unsupported dataset type: {self.type}. Only 'Gaussian' is currently supported.")

        if not isinstance(self.dimensions, int) or self.dimensions <= 0:
            raise ValueError("Invalid dimensions: Must be a positive integer.")

        if not isinstance(self.training_samples, int) or self.training_samples <= 0:
            raise ValueError("Invalid training_samples: Must be a positive integer.")

        if not isinstance(self.testing_samples, int) or self.testing_samples <= 0:
            raise ValueError("Invalid testing_samples: Must be a positive integer.")

    def _generate_gaussian_data(self, num_samples: int) -> np.ndarray:
        """
        Generate synthetic Gaussian data with a zero mean and identity covariance matrix.

        Args:
            num_samples (int): Number of samples to generate.

        Returns:
            np.ndarray: A numpy array of shape (num_samples, dimensions) with Gaussian-distributed values.
        """
        mean = np.zeros(self.dimensions)
        covariance = np.eye(self.dimensions)  # Identity covariance

        data = np.random.multivariate_normal(mean, covariance, size=num_samples)
        return data

    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess and normalize data to zero mean and unit variance for numeric stability.

        Args:
            data (np.ndarray): Input data array of shape (num_samples, dimensions).

        Returns:
            np.ndarray: Normalized data array.
        """
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std == 0] = 1  # Avoid division by zero
        normalized_data = (data - mean) / std
        return normalized_data

    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and preprocess the dataset. Based on the configuration, this method generates synthetic
        data, preprocesses it (if necessary), and converts it into PyTorch tensors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of PyTorch tensors for training and testing datasets.
        """
        if self.type.lower() == "gaussian":
            # Generate synthetic Gaussian data
            train_data = self._generate_gaussian_data(self.training_samples)
            test_data = self._generate_gaussian_data(self.testing_samples)

            # Normalize data
            train_data = self._normalize_data(train_data)
            test_data = self._normalize_data(test_data)

            # Convert data to PyTorch tensors
            train_tensor = torch.tensor(train_data, dtype=torch.float32)
            test_tensor = torch.tensor(test_data, dtype=torch.float32)

            return train_tensor, test_tensor
        else:
            raise ValueError(f"Unsupported dataset type: {self.type}")
