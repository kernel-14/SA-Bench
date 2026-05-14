# utils.py

import os
import random
import logging
from typing import Tuple, Dict
import numpy as np
import torch
import yaml

def seed_everything(seed: int = 42) -> None:
    """Sets the random seeds across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_config(config: Dict, path: str) -> None:
    """Saves a configuration dictionary to a YAML file.
    
    Args:
        config (Dict): The configuration dictionary.
        path (str): The output path for the file.
        
    Raises:
        IOError: If the file cannot be written to the specified path.
    """
    try:
        with open(path, 'w') as file:
            yaml.safe_dump(config, file)
    except Exception as e:
        raise IOError(f"Could not save configuration to {path}: {e}")

def load_config(path: str) -> Dict:
    """Loads a configuration dictionary from a YAML file.
    
    Args:
        path (str): Path to the YAML file.
    
    Returns:
        Dict: The configuration dictionary.
        
    Raises:
        FileNotFoundError: If the file path does not exist.
        ValueError: If the YAML file is invalid or cannot be parsed.
    """
    try:
        with open(path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at path: {path}")
    except yaml.YAMLError as e:
        raise ValueError(f"YAML syntax error in file {path}: {e}")

def initialize_logger(save_dir: str) -> logging.Logger:
    """Initializes a logger for recording experiment logs to both console and file output.
    
    Args:
        save_dir (str): Directory where the log file will be saved.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger("experiment_logger")
    logger.setLevel(logging.INFO)

    # File handler
    log_file = os.path.join(save_dir, "experiment.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
    console_handler.setFormatter(console_formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Avoid duplicate logs when re-initializing
    logger.propagate = False
    return logger

def compute_mean_std(array: np.ndarray) -> Tuple[float, float]:
    """Computes the mean and standard deviation of a NumPy array.
    
    Args:
        array (np.ndarray): Input array.

    Returns:
        Tuple[float, float]: (mean, standard deviation)
    """
    mean = np.mean(array)
    std = np.std(array)
    return mean, std

def normalize_array(array: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Normalizes a NumPy array using the provided mean and standard deviation.
    
    Args:
        array (np.ndarray): Array to normalize.
        mean (float): Mean value.
        std (float): Standard deviation.
    
    Returns:
        np.ndarray: Normalized array.
    """
    return (array - mean) / (std + 1e-8)  # Adding epsilon to avoid division by zero.

def denormalize_array(array: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Denormalizes a NumPy array using the provided mean and standard deviation.
    
    Args:
        array (np.ndarray): Normalized array to denormalize.
        mean (float): Original mean used for normalization.
        std (float): Original standard deviation used for normalization.
    
    Returns:
        np.ndarray: Denormalized array.
    """
    return array * std + mean

def create_experiment_directory(base_dir: str, exp_name: str) -> str:
    """Creates an experiment directory for organizing logs, checkpoints, and outputs.
    
    Args:
        base_dir (str): The base directory for experiments.
        exp_name (str): Name of the specific experiment (e.g., 'P2VAE_training').
    
    Returns:
        str: Path to the created experiment directory.
    """
    exp_dir = os.path.join(base_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict[str, any]:
    """Flattens a nested dictionary for easier logging or visualization.
    
    Args:
        d (Dict): Dictionary to flatten.
        parent_key (str): Key prefix, used for recursion.
        sep (str): Separator used for concatenating keys.
    
    Returns:
        Dict[str, any]: Flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
