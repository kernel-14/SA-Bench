# utilities.py

import os
import json
import yaml
import numpy as np
import torch
import datetime
from typing import Dict, Union, Optional


def load_yaml_config(file_path: str) -> Dict:
    """
    Load a YAML configuration file and return its contents as a Python dictionary.

    Args:
        file_path (str): Path to the configuration YAML file.

    Returns:
        dict: Parsed dictionary with keys as configuration settings.

    Raises:
        FileNotFoundError: If the file path is invalid or unreachable.
        yaml.YAMLError: If the YAML structure is invalid.
    """
    try:
        with open(file_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Config file not found at {file_path}. Ensure the path is correct.") from e
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {file_path}. Please check the syntax.") from e
    return config


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across NumPy and PyTorch operations.

    Args:
        seed (int): The seed value to initialize the random number generators.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU handling
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_metrics(metrics: Dict[str, Union[float, int]], log_path: str) -> None:
    """
    Log experiment metrics into a JSON-formatted log file. Metrics are appended
    to the log file, including a timestamp indicating when they were logged.

    Args:
        metrics (dict): Dictionary containing metric names and values (e.g., {"L2RE": 0.0056}).
        log_path (str): Path to the log file.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics["timestamp"] = timestamp

    # Create the directory for the log file, if it doesn't exist
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    try:
        # Append to the log file
        with open(log_path, "a") as log_file:
            json.dump(metrics, log_file)
            log_file.write("\n")  # Add a newline to separate entries
    except Exception as e:
        raise IOError(f"Failed to log metrics to file: {log_path}. Error: {e}")


def compute_balanced_sampling_weights(
    dataset_sizes: Dict[str, int],
    importance_weights: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Compute balanced sampling weights for pretraining datasets based on size
    and importance weights.

    Args:
        dataset_sizes (dict): Dictionary where keys are dataset names and values are sizes (e.g., {"dataset1": 1000, ...}).
        importance_weights (dict, optional): Dictionary of importance weights for each dataset. Defaults to uniform (1.0) for all datasets.

    Returns:
        dict: Dictionary of normalized sampling probabilities for each dataset.

    Raises:
        ValueError: If dataset sizes contain non-positive values.
    """
    if importance_weights is None:
        importance_weights = {name: 1.0 for name in dataset_sizes}

    # Ensure dataset sizes and weights are valid
    if not all(size > 0 for size in dataset_sizes.values()):
        raise ValueError("All dataset sizes must be positive integers.")
    
    # Compute probabilities based on dataset sizes and weights
    total_weight = sum(
        importance_weights[dataset] / max(size, 1) for dataset, size in dataset_sizes.items()
    )
    sampling_weights = {
        dataset: (importance_weights.get(dataset, 1.0) / size) / total_weight
        for dataset, size in dataset_sizes.items()
    }
    return sampling_weights


def inject_gaussian_noise(input_tensor: torch.Tensor, amplitude: float = 0.1) -> torch.Tensor:
    """
    Add Gaussian noise to an input tensor for augmentation. The noise is proportional
    to the magnitude of the input tensor.

    Args:
        input_tensor (torch.Tensor): Input tensor of shape [Batch, Channels, Height, Width, Time].
        amplitude (float): Noise amplitude factor (default: 0.1, as per config.yaml).

    Returns:
        torch.Tensor: Noisy tensor with the same shape as the input.

    Raises:
        ValueError: If the amplitude is negative or the input tensor is invalid.
    """
    if amplitude < 0:
        raise ValueError(f"Amplitude must be non-negative. Received: {amplitude}")

    noise_std = amplitude * input_tensor.norm(p=2, dim=-1, keepdim=True)
    noise = torch.randn_like(input_tensor) * noise_std
    return input_tensor + noise


# Optional Fourier-related utility functions if needed
def apply_fourier_transform(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Apply a Fourier Transform to the input tensor.

    Args:
        input_tensor (torch.Tensor): Input tensor to be transformed.

    Returns:
        torch.Tensor: Fourier-transformed tensor.
    """
    return torch.fft.fft2(input_tensor)


def apply_inverse_fourier_transform(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Apply an inverse Fourier Transform to the input tensor.

    Args:
        input_tensor (torch.Tensor): Input tensor in the Fourier domain.

    Returns:
        torch.Tensor: Inverse Fourier-transformed tensor.
    """
    return torch.fft.ifft2(input_tensor)
