"""
utils/constants.py

This module defines constants and utility functions for hyperparameters, MDP complexity parameters,
and reward variance configurations shared across modules.
"""

import yaml
import os
from typing import Dict, List

# Load configuration from config.yaml
CONFIG_PATH = "config.yaml"
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Configuration file {CONFIG_PATH} not found.")

with open(CONFIG_PATH, "r") as file:
    CONFIG = yaml.safe_load(file)

# Hyperparameters from config.yaml
DEFAULT_LEARNING_RATE: float = CONFIG.get("training", {}).get("learning_rate", 0.01)
DEFAULT_NUM_ITERATIONS: int = CONFIG.get("training", {}).get("num_iterations", 2000)
DEFAULT_PROJECTION_THRESHOLD: float = CONFIG.get("projection_threshold", 1e-6)

# State and action space sizes from config.yaml
STATE_SPACE_SIZES: List[int] = CONFIG.get("mdp", {}).get("state_space_sizes", [3, 9, 81])
ACTION_SPACE_SIZES: List[int] = CONFIG.get("mdp", {}).get("action_space_sizes", [3, 9, 81])

# Reward variance levels from config.yaml
REWARD_VARIANCE_LEVELS: List[str] = CONFIG.get("mdp", {}).get(
    "reward_variance_levels", ["no_variance", "low_variance", "high_variance", "max_variance"]
)

# Reward variance configuration
REWARD_VARIANCE_CONFIG: Dict[str, List[float]] = {
    "no_variance": [1],
    "low_variance": [0.875, 0.125],
    "high_variance": [0.75, 0.25],
    "max_variance": [0.5, 0.5],
}

# Complexity constants for MDPs
COMPLEXITY_CONSTANTS: Dict[int, Dict[str, float]] = {
    3: {"C_m": 6.0, "C_p": 1.73, "C_r": 1.73, "kappa_r": 2.0},  # Small state/action size
    9: {"C_m": 18.0, "C_p": 3.0, "C_r": 3.0, "kappa_r": 2.0},  # Medium state/action size
    81: {"C_m": 162.0, "C_p": 9.0, "C_r": 9.0, "kappa_r": 2.0},  # Large state/action size
}


def get_hyperparameters() -> Dict[str, float]:
    """
    Returns the training hyperparameters from the config file.
    """
    return {
        "learning_rate": DEFAULT_LEARNING_RATE,
        "num_iterations": DEFAULT_NUM_ITERATIONS,
        "projection_threshold": DEFAULT_PROJECTION_THRESHOLD,
    }


def get_state_action_sizes() -> List[Dict[str, int]]:
    """
    Provides predefined state and action space sizes for experiments.
    """
    return [{"state_size": s, "action_size": a} for s, a in zip(STATE_SPACE_SIZES, ACTION_SPACE_SIZES)]


def get_complexity_parameters(state_size: int, action_size: int) -> Dict[str, float]:
    """
    Returns MDP complexity parameters (C_m, C_p, C_r, kappa_r).
    Dynamically computes these constants from predefined values.
    
    Args:
        state_size (int): Number of states in the MDP.
        action_size (int): Number of actions in the MDP.
    
    Returns:
        dict: Dictionary containing complexity parameters.
    """
    if state_size in COMPLEXITY_CONSTANTS:
        constants = COMPLEXITY_CONSTANTS[state_size]
    else:
        # Default computation based on state and action sizes
        C_m = 2 * 10 * state_size / (1 - 0.99)  # Example geometric mixing constant
        C_p = (action_size ** 0.5)
        C_r = (action_size ** 0.5)
        kappa_r = 2.0  # Fixed value for reward variance
        constants = {"C_m": C_m, "C_p": C_p, "C_r": C_r, "kappa_r": kappa_r}

    return constants


def get_reward_variance_config(level: str) -> List[float]:
    """
    Maps reward variance levels to predefined configurations.
    
    Args:
        level (str): Reward variance level (e.g., "no_variance").
    
    Returns:
        list: List of weights for reward generation.
    """
    if level not in REWARD_VARIANCE_CONFIG:
        raise ValueError(f"Unknown reward variance level: {level}. Expected one of {REWARD_VARIANCE_LEVELS}.")
    return REWARD_VARIANCE_CONFIG[level]


# Example validation function
def validate_configurations() -> None:
    """
    Ensures that the configuration values are within acceptable ranges.
    Raises exceptions if validations fail.
    """
    if DEFAULT_LEARNING_RATE <= 0 or DEFAULT_LEARNING_RATE > 0.1:
        raise ValueError("Learning rate must be between 0 and 0.1.")

    if DEFAULT_NUM_ITERATIONS <= 0:
        raise ValueError("Number of iterations must be greater than 0.")

    for level in REWARD_VARIANCE_LEVELS:
        if level not in REWARD_VARIANCE_CONFIG:
            raise ValueError(f"Reward variance level '{level}' is missing in REWARD_VARIANCE_CONFIG.")

validate_configurations()  # Run configuration validation at import
