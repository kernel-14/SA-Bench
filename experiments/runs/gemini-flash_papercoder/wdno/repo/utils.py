import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple


def seed_everything(seed: int):
    """
    Seeds all random number generators for reproducibility.

    Args:
        seed: The integer seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str) -> torch.device:
    """
    Determines the appropriate computational device (CPU or CUDA).

    Args:
        device_str: A string indicating the preferred device ('cuda' or 'cpu').

    Returns:
        A torch.device object.
    """
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def normalize_data(data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Normalizes input data using element-wise mean and standard deviation.

    Args:
        data: The torch.Tensor to be normalized.
        mean: A torch.Tensor representing the mean of the training data.
              Must be broadcastable to the data's shape.
        std: A torch.Tensor representing the standard deviation of the training data.
             Must be broadcastable to the data's shape.
        epsilon: A small constant to prevent division by zero.

    Returns:
        The normalized torch.Tensor.
    """
    return (data - mean) / (std + epsilon)


def denormalize_data(normalized_data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Denormalizes data back to its original scale.

    Args:
        normalized_data: The torch.Tensor to be denormalized.
        mean: A torch.Tensor representing the mean used for normalization.
        std: A torch.Tensor representing the standard deviation used for normalization.
        epsilon: A small constant used during normalization.

    Returns:
        The denormalized torch.Tensor.
    """
    return normalized_data * (std + epsilon) + mean


def save_checkpoint(step: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler], filepath: str):
    """
    Saves the current state of training to a checkpoint file.

    Args:
        step: The current global training step.
        model: The torch.nn.Module whose state_dict needs to be saved.
        optimizer: The torch.optim.Optimizer whose state_dict needs to be saved.
        scheduler: An optional learning rate scheduler whose state_dict needs to be saved.
        filepath: The full path where the checkpoint file will be saved.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath} at step {step}")


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                    filepath: str, device: torch.device) -> int:
    """
    Loads a previously saved training state from a checkpoint file.

    Args:
        model: The torch.nn.Module into which the saved state will be loaded.
        optimizer: The torch.optim.Optimizer into which the saved state will be loaded.
        scheduler: An optional learning rate scheduler into which the saved state will be loaded.
        filepath: The path to the checkpoint file.
        device: The torch.device to map the loaded tensors to.

    Returns:
        The training step from which the checkpoint was saved.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found at: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print(f"Checkpoint loaded from {filepath}. Resuming from step {checkpoint['step']}")
    return checkpoint['step']


def calculate_metrics(predictions: torch.Tensor, targets: torch.Tensor,
                      metrics: List[str] = ['mse', 'l2_relative', 'mae', 'l_inf']) -> Dict[str, float]:
    """
    Computes a dictionary of specified evaluation metrics between predictions and ground truth.

    Args:
        predictions: The model's output torch.Tensor.
        targets: The ground truth torch.Tensor.
        metrics: A list of metric names to calculate. Supported: 'mse', 'l2_relative', 'mae', 'l_inf'.

    Returns:
        A dictionary where keys are metric names and values are their computed float values.
    """
    results = {}
    diff = predictions - targets

    if 'mse' in metrics:
        results['mse'] = torch.mean(diff.pow(2)).item()

    if 'mae' in metrics:
        results['mae'] = torch.mean(diff.abs()).item()

    if 'l_inf' in metrics:
        results['l_inf'] = torch.max(diff.abs()).item()

    if 'l2_relative' in metrics:
        # Calculate L2 norm across all dimensions except the batch dimension
        # This gives a per-sample L2 norm difference and target L2 norm
        dim_to_reduce = tuple(range(1, predictions.ndim))
        diff_norm = torch.linalg.norm(diff, dim=dim_to_reduce)
        target_norm = torch.linalg.norm(targets, dim=dim_to_reduce)
        
        # Avoid division by zero for target_norm, compute mean of relative errors
        # Handle cases where target_norm is 0 by making relative error 0 if diff_norm is also 0
        # or inf if diff_norm is non-zero
        relative_errors = torch.where(target_norm == 0,
                                      torch.where(diff_norm == 0, torch.tensor(0.0).to(targets.device), torch.tensor(float('inf')).to(targets.device)),
                                      diff_norm / target_norm)
        
        results['l2_relative'] = torch.mean(relative_errors).item()

    return results


def interpolate_to_finest_resolution(data: torch.Tensor, target_dims_shape: Union[Tuple[int], Tuple[int, int], Tuple[int, int, int]],
                                     interpolation_method: str = 'linear') -> torch.Tensor:
    """
    Upsamples data to a target resolution for super-resolution evaluation.

    Args:
        data: The input tensor, expected shape (B, C, D1, D2, ...) where (D1, D2, ...) are the dimensions to interpolate.
        target_dims_shape: A tuple indicating the desired shape for the dimensions to be interpolated
                           (e.g., (T_target, X_target) for 2D, or (T_target, H_target, W_target) for 3D).
        interpolation_method: String, either 'linear' or 'nearest'.

    Returns:
        The interpolated torch.Tensor.
    """
    num_dims_to_interpolate = len(target_dims_shape)

    if interpolation_method == 'nearest':
        mode = 'nearest'
        align_corners = None # align_corners not used for 'nearest' mode
    elif interpolation_method == 'linear':
        if num_dims_to_interpolate == 1:
            mode = 'linear'
        elif num_dims_to_interpolate == 2:
            mode = 'bilinear'
        elif num_dims_to_interpolate == 3:
            mode = 'trilinear'
        else:
            raise ValueError(f"Linear interpolation not supported for {num_dims_to_interpolate} dimensions. "
                             "Supported: 1, 2, 3.")
        align_corners = False # Common practice for feature maps, prevent extrapolation artifacts
    else:
        raise ValueError(f"Unsupported interpolation method: {interpolation_method}. Choose 'linear' or 'nearest'.")

    # F.interpolate expects the 'size' argument to correspond to the last N dimensions of the input tensor
    # If input is (B, C, D1, D2) and target_dims_shape is (D1_new, D2_new), then size=(D1_new, D2_new)
    interpolated_data = F.interpolate(data, size=target_dims_shape, mode=mode, align_corners=align_corners)

    return interpolated_data


if __name__ == "__main__":
    # --- Test seed_everything ---
    print("Testing seed_everything...")
    seed_everything(42)
    rand_num_py = random.random()
    rand_num_np = np.random.rand()
    rand_num_torch = torch.rand(1).item()
    print(f"Python random: {rand_num_py}")
    print(f"NumPy random: {rand_num_np}")
    print(f"PyTorch random: {rand_num_torch}")

    seed_everything(42)
    assert random.random() == rand_num_py
    assert np.random.rand() == rand_num_np
    assert torch.rand(1).item() == rand_num_torch
    print("seed_everything passed.")

    # --- Test get_device ---
    print("\nTesting get_device...")
    cpu_device = get_device('cpu')
    print(f"CPU device: {cpu_device}")
    assert cpu_device.type == 'cpu'
    if torch.cuda.is_available():
        cuda_device = get_device('cuda')
        print(f"CUDA device: {cuda_device}")
        assert cuda_device.type == 'cuda'
    print("get_device passed.")

    # --- Test normalize_data and denormalize_data ---
    print("\nTesting normalize_data and denormalize_data...")
    data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    mean = torch.tensor(3.0)
    std = torch.tensor(1.0)
    
    normalized = normalize_data(data, mean, std)
    print(f"Original data: {data}")
    print(f"Normalized data: {normalized}")
    assert torch.allclose(normalized, torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]))

    denormalized = denormalize_data(normalized, mean, std)
    print(f"Denormalized data: {denormalized}")
    assert torch.allclose(denormalized, data)
    print("normalize_data and denormalize_data passed.")

    # --- Test save_checkpoint and load_checkpoint ---
    print("\nTesting save_checkpoint and load_checkpoint...")
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 1)
        def forward(self, x):
            return self.linear(x)

    model = MockModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    
    initial_step = 10
    test_filepath = "./test_checkpoints/test_checkpoint.pt"
    
    save_checkpoint(initial_step, model, optimizer, scheduler, test_filepath)
    
    # Create new instances to load into
    new_model = MockModel()
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.01)
    new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=1, gamma=0.1)

    loaded_step = load_checkpoint(new_model, new_optimizer, new_scheduler, test_filepath, cpu_device)
    
    assert loaded_step == initial_step
    # Check model state dicts are identical
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1.data, p2.data)
    # Check optimizer state dicts are identical (this is complex, just compare keys for simplicity)
    assert optimizer.state_dict().keys() == new_optimizer.state_dict().keys()
    assert scheduler.state_dict().keys() == new_scheduler.state_dict().keys()

    os.remove(test_filepath)
    os.rmdir("./test_checkpoints")
    print("save_checkpoint and load_checkpoint passed.")

    # --- Test calculate_metrics ---
    print("\nTesting calculate_metrics...")
    predictions = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]], [[[5.0, 6.0], [7.0, 8.0]]]]) # (B=2, C=1, H=2, W=2)
    targets = torch.tensor([[[[1.1, 2.1], [2.9, 4.1]]], [[[5.2, 5.9], [7.1, 7.8]]]])

    metrics_results = calculate_metrics(predictions, targets)
    print(f"Metrics: {metrics_results}")
    
    expected_mse = torch.mean((predictions - targets).pow(2)).item()
    expected_mae = torch.mean((predictions - targets).abs()).item()
    expected_linf = torch.max((predictions - targets).abs()).item()
    
    diff_norm_sample1 = torch.linalg.norm(predictions[0,0] - targets[0,0]).item()
    target_norm_sample1 = torch.linalg.norm(targets[0,0]).item()
    rel_error_sample1 = diff_norm_sample1 / target_norm_sample1

    diff_norm_sample2 = torch.linalg.norm(predictions[1,0] - targets[1,0]).item()
    target_norm_sample2 = torch.linalg.norm(targets[1,0]).item()
    rel_error_sample2 = diff_norm_sample2 / target_norm_sample2

    expected_l2_relative = (rel_error_sample1 + rel_error_sample2) / 2
    
    assert abs(metrics_results['mse'] - expected_mse) < 1e-6
    assert abs(metrics_results['mae'] - expected_mae) < 1e-6
    assert abs(metrics_results['l_inf'] - expected_linf) < 1e-6
    assert abs(metrics_results['l2_relative'] - expected_l2_relative) < 1e-6
    
    print("calculate_metrics passed.")

    # --- Test interpolate_to_finest_resolution ---
    print("\nTesting interpolate_to_finest_resolution...")
    
    # 1D spatial problem data example: (B, C, T, X) -> (B, C, T_new, X_new)
    data_1d_spatial = torch.randn(2, 3, 10, 20) # Batch, Channels, Time, X-spatial
    target_shape_1d_spatial = (20, 40) # New Time, New X-spatial

    interp_linear_1d = interpolate_to_finest_resolution(data_1d_spatial, target_shape_1d_spatial, 'linear')
    assert interp_linear_1d.shape == (2, 3, 20, 40)
    print(f"Interpolated 1D spatial (linear) shape: {interp_linear_1d.shape}")

    interp_nearest_1d = interpolate_to_finest_resolution(data_1d_spatial, target_shape_1d_spatial, 'nearest')
    assert interp_nearest_1d.shape == (2, 3, 20, 40)
    print(f"Interpolated 1D spatial (nearest) shape: {interp_nearest_1d.shape}")

    # 2D spatial problem data example: (B, C, T, H, W) -> (B, C, T_new, H_new, W_new)
    data_2d_spatial = torch.randn(1, 2, 5, 8, 8) # Batch, Channels, Time, H-spatial, W-spatial
    target_shape_2d_spatial = (10, 16, 16) # New Time, New H-spatial, New W-spatial

    interp_linear_2d = interpolate_to_finest_resolution(data_2d_spatial, target_shape_2d_spatial, 'linear')
    assert interp_linear_2d.shape == (1, 2, 10, 16, 16)
    print(f"Interpolated 2D spatial (linear) shape: {interp_linear_2d.shape}")

    interp_nearest_2d = interpolate_to_finest_resolution(data_2d_spatial, target_shape_2d_spatial, 'nearest')
    assert interp_nearest_2d.shape == (1, 2, 10, 16, 16)
    print(f"Interpolated 2D spatial (nearest) shape: {interp_nearest_2d.shape}")
    
    # Test 1D interpolation (e.g. for a single spatial dimension)
    data_single_dim = torch.randn(4, 1, 30) # (B, C, X)
    target_shape_single_dim = (60,) # (X_new,)
    interp_linear_single = interpolate_to_finest_resolution(data_single_dim, target_shape_single_dim, 'linear')
    assert interp_linear_single.shape == (4, 1, 60)
    print(f"Interpolated 1D (linear) shape: {interp_linear_single.shape}")

    print("interpolate_to_finest_resolution passed.")
