# utils.py

import yaml
import torch
import numpy as np
import random
import logging
from typing import Any, Dict


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads the configuration file from the provided path and returns it as a dictionary.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Parsed configuration settings.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the YAML file contains syntax errors.
    """
    try:
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
        return config
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Configuration file not found: {config_path}") from e
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {config_path}") from e


def set_seed(seed: int) -> None:
    """
    Sets random seeds for reproducible results across frameworks.

    Args:
        seed (int): Random seed value.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logging.info(f"Random seed set to: {seed}")


def log_results(results: Dict[str, Any], log_path: str) -> None:
    """
    Logs experiment results and metrics to a specified file.

    Args:
        results (dict): Experiment results and metrics to log.
        log_path (str): File path where logs should be stored.

    Returns:
        None

    Raises:
        IOError: If writing to the log file fails.
    """
    try:
        # Configure the logger
        logger = logging.getLogger("SCoRe")
        logger.setLevel(logging.INFO)
        
        # Create a file handler that appends to the log file
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)

        # Specify the log format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add handler to the logger if not already added
        if not logger.handlers:
            logger.addHandler(file_handler)

        # Log the results
        logger.info("----- Experiment Results -----")
        for key, value in results.items():
            logger.info(f"{key}: {value}")

        logger.info("Results successfully logged.")

    except IOError as e:
        raise IOError(f"Failed to write logs to {log_path}") from e

