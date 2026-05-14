import random
import numpy as np
import torch
import torch.nn as nn
from typing import Union, Dict, Any, Type

def set_seed(seed: int, deterministic_cudnn: bool = True) -> None:
    """
    Sets random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use for all random number generators.
        deterministic_cudnn (bool): If True, sets PyTorch CUDA backend to use
                                    deterministic algorithms, potentially affecting
                                    performance but ensuring bit-for-bit reproducibility.
    """
    if not isinstance(seed, int):
        raise TypeError(f"Expected 'seed' to be an integer, but got {type(seed)}.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # If not enforcing deterministic behavior, it's generally good to allow
        # cudnn.benchmark for better performance if reproducibility isn't strictly required
        # (e.g., during hyperparameter tuning where exact reproducibility isn't the priority).
        torch.backends.cudnn.benchmark = True
    # For CUDA 10.2 and later, potentially add:
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # torch.use_deterministic_algorithms(True) # Requires specific CUDA versions and might not be applicable everywhere

def get_device(preferred_device_str: str = "cuda") -> torch.device:
    """
    Dynamically determines and returns the most appropriate computing device (GPU or CPU).

    Args:
        preferred_device_str (str): The preferred device as a string, e.g., "cuda" or "cpu".
                                    Defaults to "cuda" if not specified.

    Returns:
        torch.device: The selected PyTorch device.
    """
    if not isinstance(preferred_device_str, str):
        raise TypeError(f"Expected 'preferred_device_str' to be a string, but got {type(preferred_device_str)}.")

    if preferred_device_str.lower() == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU (CUDA).")
    else:
        device = torch.device("cpu")
        if preferred_device_str.lower() == "cuda":
            print(f"Warning: CUDA requested but not available. Falling back to CPU.")
        else:
            print("Using CPU.")
    return device

def count_parameters(model: torch.nn.Module) -> int:
    """
    Counts the total number of trainable parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model instance.

    Returns:
        int: The total number of trainable parameters.
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Expected 'model' to be a torch.nn.Module, but got {type(model)}.")

    total_params = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            total_params += parameter.numel()
    return total_params

def normalize_data(data_tensor: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """
    Performs min-max normalization on a PyTorch tensor to the range [0, 1].

    Args:
        data_tensor (torch.Tensor): The input tensor to normalize.
        min_val (float): The minimum value of the original data range.
        max_val (float): The maximum value of the original data range.

    Returns:
        torch.Tensor: The normalized tensor.
    """
    if not isinstance(data_tensor, torch.Tensor):
        raise TypeError(f"Expected 'data_tensor' to be a torch.Tensor, but got {type(data_tensor)}.")
    if not isinstance(min_val, (int, float)):
        raise TypeError(f"Expected 'min_val' to be a float, but got {type(min_val)}.")
    if not isinstance(max_val, (int, float)):
        raise TypeError(f"Expected 'max_val' to be a float, but got {type(max_val)}.")

    if max_val == min_val:
        # Handle cases where the data is constant to avoid division by zero
        return torch.zeros_like(data_tensor, dtype=data_tensor.dtype)
    return (data_tensor - min_val) / (max_val - min_val)

def denormalize_data(normalized_tensor: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """
    Reverses the min-max normalization, restoring the tensor to its original data range.

    Args:
        normalized_tensor (torch.Tensor): The normalized input tensor.
        min_val (float): The minimum value of the original data range.
        max_val (float): The maximum value of the original data range.

    Returns:
        torch.Tensor: The denormalized tensor.
    """
    if not isinstance(normalized_tensor, torch.Tensor):
        raise TypeError(f"Expected 'normalized_tensor' to be a torch.Tensor, but got {type(normalized_tensor)}.")
    if not isinstance(min_val, (int, float)):
        raise TypeError(f"Expected 'min_val' to be a float, but got {type(min_val)}.")
    if not isinstance(max_val, (int, float)):
        raise TypeError(f"Expected 'max_val' to be a float, but got {type(max_val)}.")

    if max_val == min_val:
        # Handle cases where the original data was constant
        return torch.full_like(normalized_tensor, min_val, dtype=normalized_tensor.dtype)
    return normalized_tensor * (max_val - min_val) + min_val

def get_activation_fn(name: str) -> torch.nn.Module:
    """
    Returns a PyTorch activation function module based on its string name.

    Args:
        name (str): The name of the activation function (e.g., 'relu', 'gelu', 'tanh', 'sigmoid').

    Returns:
        torch.nn.Module: An instance of the corresponding PyTorch activation function module.

    Raises:
        ValueError: If an unsupported activation function name is provided.
    """
    if not isinstance(name, str):
        raise TypeError(f"Expected 'name' to be a string, but got {type(name)}.")

    activation_functions: Dict[str, Type[torch.nn.Module]] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "leakyrelu": nn.LeakyReLU,
        "elu": nn.ELU,
        "silu": nn.SiLU # Also known as Swish
    }
    
    name_lower = name.lower()
    if name_lower in activation_functions:
        return activation_functions[name_lower]()
    else:
        raise ValueError(
            f"Unsupported activation function: '{name}'. "
            f"Supported functions are: {list(activation_functions.keys())}"
        )

