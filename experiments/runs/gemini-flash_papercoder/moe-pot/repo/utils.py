## utils.py
import os
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.ndimage
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.optim import Optimizer

# Import Config for type hinting and access to config values
from config import Config


def set_seed(seed: int) -> None:
    """
    Sets the random seed for reproducibility across multiple libraries.

    Args:
        seed: The integer seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_l2re(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    Computes the L2 relative error between predicted and ground truth tensors.

    Args:
        pred: The tensor representing the model's prediction.
        gt: The tensor representing the ground truth.

    Returns:
        The L2 relative error as a float. Returns 0 if gt is effectively zero,
        to avoid division by zero.
    """
    # Ensure tensors are on CPU for numpy conversion or direct L2 norm
    pred = pred.to(torch.float32)
    gt = gt.to(torch.float32)

    diff_norm = torch.norm(pred - gt)
    gt_norm = torch.norm(gt)

    # Handle the case where ground truth is zero to avoid division by zero.
    # If gt is zero, and pred is also zero, error is 0. Otherwise, error is inf.
    # For practical purposes, a small epsilon helps numerical stability.
    if gt_norm < 1e-8:  # Check if gt_norm is effectively zero
        if diff_norm < 1e-8: # If prediction is also effectively zero
            return 0.0
        else: # If gt is zero but pred is not, error is very large
            return float('inf')
    
    return (diff_norm / gt_norm).item()


def inject_noise(u_seq: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    Injects Gaussian noise into the input sequences for pre-training.
    The noise is sampled from N(0, epsilon * ||u_seq|| * I), where I is identity.
    This means the standard deviation is scaled by the L2 norm of the input sequence.

    Args:
        u_seq: The input sequence (u^<t) to which noise should be added.
               Expected shape (batch_size, H, W, C) or similar.
        epsilon: The scaling factor for the noise standard deviation.

    Returns:
        The input sequence with injected noise.
    """
    if epsilon == 0.0:
        return u_seq
    
    # Calculate the L2 norm of the input sequence across all dimensions
    norm_u_seq = torch.norm(u_seq)
    
    # Compute the standard deviation for the noise
    std_dev = epsilon * norm_u_seq
    
    # Generate noise from a normal distribution with the calculated std_dev
    # and the same shape as u_seq.
    noise = torch.randn_like(u_seq) * std_dev
    
    # Add the noise to the input sequence
    u_seq_noisy = u_seq + noise
    
    return u_seq_noisy


def get_lr_scheduler(optimizer: Optimizer, config: Config, total_epochs: int, warmup_epochs: int) -> LRScheduler:
    """
    Instantiates a One-cycle learning rate scheduler.

    Args:
        optimizer: The optimizer whose learning rate will be scheduled.
        config: The global configuration object.
        total_epochs: The total number of epochs for the current training stage.
        warmup_epochs: The number of warm-up epochs for the current training stage.

    Returns:
        An instance of torch.optim.lr_scheduler.OneCycleLR.
    """
    if total_epochs <= 0:
        raise ValueError("total_epochs must be greater than 0 for LR scheduler.")
    if warmup_epochs < 0 or warmup_epochs >= total_epochs:
        raise ValueError("warmup_epochs must be non-negative and less than total_epochs.")

    max_lr = config.training.learning_rate
    pct_start = float(warmup_epochs) / total_epochs if total_epochs > 0 else 0.0

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_epochs,  # Stepped per epoch
        pct_start=pct_start,
        div_factor=25.0,  # Default from PyTorch documentation
        final_div_factor=1e4,  # Default from PyTorch documentation
        anneal_strategy='cos',
        cycle_momentum=False # Momentum not cycled for Adam, which is common.
    )
    return scheduler


def setup_distributed_training(rank: int, world_size: int, backend: str) -> None:
    """
    Initializes the distributed training environment.

    Args:
        rank: The unique identifier for the current process within the distributed group.
        world_size: The total number of processes participating in distributed training.
        backend: The distributed backend to use (e.g., 'nccl', 'gloo').
    """
    if world_size > 1:
        # Set environment variables for master address and port if not already set
        # These are typically set by the launcher (e.g., torch.distributed.launch)
        os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', 'localhost')
        os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', '12355') # Default port

        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        dist.barrier() # Synchronize all processes


def cleanup_distributed_training() -> None:
    """
    Cleans up the distributed training environment.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def save_checkpoint(model: nn.Module, optimizer: Optimizer, epoch: int, path: str, best_metric: float, is_best: bool = False) -> None:
    """
    Saves the current state of the model, optimizer, training epoch, and best performance metric to a file.

    Args:
        model: The model instance to save. If DDP-wrapped, its .module.state_dict() is saved.
        optimizer: The optimizer instance to save.
        epoch: The current training epoch.
        path: The base file path where the checkpoint will be saved.
        best_metric: The best performance metric achieved so far.
        is_best: A boolean flag indicating if this is the best model checkpoint.
    """
    # Get the model's state_dict, handling DDP wrap if present
    model_state_dict = model.module.state_dict() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.state_dict()

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': best_metric,
    }

    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save the checkpoint
    torch.save(checkpoint, path)

    if is_best:
        best_path = os.path.join(os.path.dirname(path), "model_best.pt")
        torch.save(checkpoint, best_path)
        print(f"--> Checkpoint saved as {path} and model_best.pt")
    else:
        print(f"--> Checkpoint saved as {path}")


def load_checkpoint(model: nn.Module, optimizer: Optimizer, path: str) -> Tuple[int, float]:
    """
    Loads the model and optimizer states from a previously saved checkpoint.

    Args:
        model: The model instance to load state into. If DDP-wrapped, its .module is used.
        optimizer: The optimizer instance to load state into.
        path: The full file path to the checkpoint file.

    Returns:
        A tuple containing the loaded epoch number and the best metric achieved at that checkpoint.

    Raises:
        FileNotFoundError: If the specified checkpoint path does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found at {path}")

    # Load the checkpoint dictionary, mapping to CPU first to be device-agnostic
    checkpoint = torch.load(path, map_location='cpu')

    # Get the target model (handle DDP wrap)
    target_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    
    target_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    best_metric = checkpoint.get('best_metric', float('inf')) # Default to inf for metrics where lower is better

    print(f"--> Checkpoint loaded from {path} (Epoch: {epoch}, Best Metric: {best_metric:.4f})")
    return epoch, best_metric


def get_activation(name: str) -> nn.Module:
    """
    Returns an activation function module based on its string name.

    Args:
        name: The string name of the desired activation function (e.g., 'GELU', 'ReLU').

    Returns:
        An instance of torch.nn.Module representing the activation function.

    Raises:
        ValueError: If an unsupported activation name is provided.
    """
    activation_map = {
        'ReLU': nn.ReLU,
        'GELU': nn.GELU,
        'SiLU': nn.SiLU,
        'LeakyReLU': nn.LeakyReLU,
        'Sigmoid': nn.Sigmoid,
        'Tanh': nn.Tanh,
        'Identity': nn.Identity # For cases where no activation is desired
    }
    
    if name not in activation_map:
        raise ValueError(f"Unsupported activation function: {name}. "
                         f"Supported: {list(activation_map.keys())}")
    
    return activation_map[name]()


def pad_and_mask_channels(data: np.ndarray, target_data_channels: int, pad_value: float = 1.0, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Standardizes the number of data channels for PDE data and optionally appends a mask channel.

    Args:
        data: Input PDE data, expected shape (..., H, W, C_current).
              The ellipsis indicates it could be (H, W, C_current) or (T, H, W, C_current), etc.
        target_data_channels: The desired number of data channels *excluding* any mask channel.
                              This will be the maximum 'C' observed across all datasets.
        pad_value: The value used to fill newly padded data channels.
        mask: An optional binary numpy array representing the geometric mask.
              Its shape should be (..., H, W). If provided, it will be reshaped
              and appended as an additional channel AFTER channel padding.

    Returns:
        A numpy array with standardized data channels and an optional appended mask channel.
    """
    current_channels = data.shape[-1]
    
    if current_channels < target_data_channels:
        # Calculate padding needed for the channel dimension
        padding_needed = target_data_channels - current_channels
        # Create a padding tuple for np.pad. Only the last dimension is padded.
        # Example: if data shape (T, H, W, C), padding is ( (0,0), (0,0), (0,0), (0, padding_needed) )
        pad_width = [(0, 0)] * (data.ndim - 1) + [(0, padding_needed)]
        processed_data = np.pad(data, pad_width, mode='constant', constant_values=pad_value)
    elif current_channels > target_data_channels:
        # Truncate if current channels exceed target (should ideally not happen if target is max)
        processed_data = data[..., :target_data_channels]
    else:
        processed_data = data
    
    # Append mask channel if provided
    if mask is not None:
        # Ensure mask has a channel dimension of 1
        # Example: mask (T, H, W) -> (T, H, W, 1)
        mask_reshaped = mask[..., np.newaxis]
        processed_data = np.concatenate((processed_data, mask_reshaped), axis=-1)
            
    return processed_data


def resize_spatial_resolution(data: np.ndarray, target_res: int, method: str) -> np.ndarray:
    """
    Resizes the spatial dimensions (H, W) of the input data to a uniform target resolution.

    Args:
        data: Input PDE data, expected shape (..., H_current, W_current, C).
        target_res: The desired uniform spatial resolution (e.g., 128).
        method: The interpolation method to use ('bicubic', 'bilinear').

    Returns:
        A numpy array with spatial dimensions resized to (target_res, target_res).

    Raises:
        ValueError: If an unsupported interpolation method is provided.
    """
    if data.ndim < 3:
        raise ValueError("Input data must have at least 3 dimensions (H, W, C).")

    # Current spatial dimensions are the 3rd and 2nd to last dimensions
    current_h, current_w = data.shape[-3], data.shape[-2]

    if current_h == target_res and current_w == target_res:
        return data # No resizing needed

    # Calculate zoom factors for H and W. Other dimensions have a zoom factor of 1.
    zoom_factors = [1.0] * data.ndim
    zoom_factors[-3] = float(target_res) / current_h # Zoom factor for H
    zoom_factors[-2] = float(target_res) / current_w # Zoom factor for W

    # Map method string to scipy.ndimage.zoom order parameter
    order_map = {
        'nearest': 0,
        'bilinear': 1,
        'biquadratic': 2,
        'bicubic': 3,
        'biquartic': 4,
        'biquintic': 5,
    }

    if method not in order_map:
        raise ValueError(f"Unsupported interpolation method: {method}. "
                         f"Supported: {list(order_map.keys())}")
    
    order = order_map[method]

    # Apply resizing using scipy.ndimage.zoom
    # Note: scipy.ndimage.zoom can handle multi-dimensional arrays,
    # apply different zoom factors to different axes.
    resized_data = scipy.ndimage.zoom(data, zoom_factors, order=order)
    
    return resized_data

