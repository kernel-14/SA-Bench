# utils.py

import os
import random
import yaml
import numpy as np
import torch
from typing import Any, Union, Dict, Iterable
from torch import Tensor


def set_seed(seed: int) -> None:
    """Set the random seeds for reproducible outputs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[INFO] Seed set to {seed} for reproducibility.")


def save_config(config: Dict[str, Any], path: str) -> None:
    """Save configuration dictionary to a YAML file."""
    check_create_dir(os.path.dirname(path))
    with open(path, "w") as file:
        yaml.dump(config, file, default_flow_style=False)
    print(f"[INFO] Configuration saved successfully to {path}.")


def load_config(path: str) -> Union[Dict[str, Any], None]:
    """Load a YAML configuration file into a dictionary."""
    if not os.path.exists(path):
        print(f"[ERROR] Config file not found at {path}.")
        return None
    with open(path, "r") as file:
        try:
            config = yaml.safe_load(file)
            print(f"[INFO] Successfully loaded configuration from {path}.")
            return config
        except yaml.YAMLError as e:
            print(f"[ERROR] Failed to parse YAML file at {path}: {e}")
            return None


def check_create_dir(path: str) -> None:
    """Ensure the directory exists; if not, create it recursively."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print(f"[INFO] Directory created: {path}")


def format_duration(seconds: float) -> str:
    """Convert seconds to a human-readable HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"


def compute_grad_norm(parameters: Iterable[Tensor]) -> float:
    """Compute total gradient norm across model parameters."""
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def log_message(message: str, log_path: str = None, verbose: bool = True) -> None:
    """Log a message to console and optionally to a file."""
    if verbose:
        print(message)
    if log_path:
        check_create_dir(os.path.dirname(log_path))
        with open(log_path, "a") as log_file:
            log_file.write(message + "\n")


def tensor_to_image(tensor: Tensor, mode: str = "RGB") -> np.ndarray:
    """Convert a PyTorch tensor to a NumPy array for saving/visualization."""
    if mode not in ["RGB", "L"]:
        raise ValueError(f"Unsupported mode {mode}. Use 'RGB' or 'L'.")
    
    tensor = tensor.detach().cpu().permute(1, 2, 0).numpy()
    tensor = (tensor * 255).clip(0, 255).astype(np.uint8)
    
    if mode == "L" and tensor.ndim == 3:
        tensor = tensor[:, :, 0]  # Convert single-channel grayscale
    
    return tensor


def to_device(data: Any, device: str) -> Any:
    """Move tensors or data structures containing tensors to the specified device."""
    if isinstance(data, Tensor):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return [to_device(d, device) for d in data]
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    else:
        return data  # Return as-is for unsupported types
