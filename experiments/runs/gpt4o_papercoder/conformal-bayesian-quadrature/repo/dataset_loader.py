# dataset_loader.py

import numpy as np
from typing import Tuple
from pathlib import Path

class DatasetLoader:
    """
    Handles loading datasets for experiments:
    - Synthetic datasets (binomial and heteroskedastic)
    - Real-world dataset (e.g., MS-COCO)
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize the DatasetLoader with configurations.
        Args:
            config (dict): Configuration dictionary loaded from `config.yaml`.
        """
        self.config = config

    def load_binomial_data(self, n: int, K: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create synthetic Binomial data with `n` samples and `K` trials.

        Args:
            n (int): Number of calibration samples.
            K (int): Number of trials per sample.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Arrays X (inputs) and Y (proportions).
        """
        # Generate a matrix V of size (n, K) with random values from Uniform(0, 1)
        rng = np.random.default_rng(seed=42)  # Fixed seed for reproducibility
        V = rng.uniform(0, 1, (n, K))

        # Calculate Y (proportion of successes per sample)
        Y = np.mean(V > 0.5, axis=1)

        # Generate X as simple indices (e.g., 1 to n)
        X = np.arange(1, n + 1)

        return X, Y

    def load_heteroskedastic_data(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create heterogeneous synthetic data (variance depends on input).
        
        Args:
            n (int): Number of calibration samples.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Arrays X (inputs) and Y (outputs).
        """
        # Generate inputs X uniformly from [0, 4]
        rng = np.random.default_rng(seed=42)  # Fixed seed for reproducibility
        X = rng.uniform(0, 4, n)

        # Generate outputs Y ~ N(0, X^2)
        Y = rng.normal(0, X**2)

        return X, Y

    def load_ms_coco(self, path: str, num_calibration: int, num_test: int
                     ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Load and split the MS-COCO dataset into calibration and test sets.

        Args:
            path (str): Path to the dataset file.
            num_calibration (int): Number of samples for calibration set.
            num_test (int): Number of samples for test set.

        Returns:
            Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
                (calibration_X, calibration_Y), (test_X, test_Y)
        """
        # Verify the dataset file exists
        dataset_path = Path(path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found at {path}")

        # Load the dataset (simulation: generate random data for example purposes)
        # Assuming dataset to be preprocessed into features (X) and multilabel outputs (Y)
        rng = np.random.default_rng(seed=42)  # Fixed seed for reproducibility
        feature_dim = 512  # Example feature dimension
        n_labels = 80      # Example number of labels (based on MS-COCO categories)

        # Total number of samples must be num_calibration + num_test
        total_samples = num_calibration + num_test
        X = rng.random((total_samples, feature_dim))  # Random feature vectors
        Y = rng.integers(0, 2, (total_samples, n_labels))  # Random binary labels

        # Split the dataset
        calibration_X, calibration_Y = X[:num_calibration], Y[:num_calibration]
        test_X, test_Y = X[num_calibration:num_calibration + num_test], Y[num_calibration:num_calibration + num_test]

        return (calibration_X, calibration_Y), (test_X, test_Y)
