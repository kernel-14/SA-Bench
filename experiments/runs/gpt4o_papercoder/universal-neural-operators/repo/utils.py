## utils.py

"""
This module provides utility functions for setting deterministic behavior, logging metrics, and visualizing PDE solution fields.

Functions:
    - set_seed: Ensures reproducibility across libraries.
    - log_metrics: Records experimental results to a log file.
    - plot_field: Visualizes PDE solution fields in scalar or vector forms.
"""

# Required Imports
import os
import random
from typing import Optional, Dict
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
import yaml


def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility across multiple libraries.

    Args:
        seed (int): The random seed to use.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior on GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def log_metrics(metrics: Dict[str, float], log_path: str) -> None:
    """
    Record metrics to a log file for traceability of experiments.

    Args:
        metrics (dict): A dictionary of experimental metrics (e.g., NMAE, MSE).
        log_path (str): Path to the log file. Will append if the file exists.

    Returns:
        None
    """
    # Ensure the directory for logs exists
    log_dir = os.path.dirname(log_path)
    os.makedirs(log_dir, exist_ok=True)

    # Append metrics to the log file in YAML format for readability
    with open(log_path, "a") as log_file:
        log_data = {"Experiment": metrics}
        yaml.dump(log_data, log_file, default_flow_style=False, sort_keys=False)
        log_file.write("\n")  # Add spacing between entries


def plot_field(field: torch.Tensor, title: str, save_path: Optional[str] = None) -> None:
    """
    Visualize a scalar PDE solution field using matplotlib.

    Args:
        field (torch.Tensor): 2D tensor representing the field to be visualized.
        title (str): Title for the plot.
        save_path (Optional[str]): Path to save the plot. If None, displays the plot interactively.

    Returns:
        None
    """
    # Convert field to numpy (if it is a PyTorch tensor)
    if isinstance(field, torch.Tensor):
        field = field.detach().cpu().numpy()

    # Normalize the field using min-max scaling
    field_min, field_max = np.min(field), np.max(field)
    field_normalized = (field - field_min) / (field_max - field_min + 1e-8)

    # Plot the field using imshow with a color mapping
    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        field_normalized,
        cmap="viridis",
        interpolation="nearest",
        origin="lower",
        aspect="auto",
    )
    plt.colorbar(im, orientation="vertical", fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")

    # Save or display the plot
    if save_path:
        # Ensure the save directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    # Close the plot to avoid excessive memory use
    plt.close()

