## utils.py
"""
Utilities module for shared functionality across WDNO's components.
Includes metrics computation, checkpoint handling, and wavelet-related helper functions.
"""

import os
from typing import List, Dict, Any, Optional
import torch
from torch import Tensor
from wavelet_transform import WaveletTransform


def calculate_metrics(predictions: Tensor, ground_truth: Tensor) -> Dict[str, float]:
    """
    Compute evaluation metrics such as Mean Squared Error (MSE) and control objectives.

    Args:
        predictions (Tensor): Predicted spatial-temporal states or wavelet coefficients.
                              Shape: [batch_size, ..., spatial_dims].
        ground_truth (Tensor): Ground truth data corresponding to the predictions.
                              Shape: Same as `predictions`.

    Returns:
        Dict[str, float]: Dictionary containing calculated metrics. E.g., {"mse": value}.
    """
    if predictions.size() != ground_truth.size():
        raise ValueError(
            f"Shape mismatch between predictions {predictions.size()} and ground_truth {ground_truth.size()}"
        )

    # Mean Squared Error (MSE)
    mse = torch.mean((predictions - ground_truth) ** 2).item()

    # Additional metrics (like control-specific objectives) can be added here
    metrics = {"mse": mse}
    return metrics


def save_checkpoint(model: torch.nn.Module, file_path: str) -> None:
    """
    Save the model's state dictionary to a checkpoint file.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        file_path (str): Path where the checkpoint file will be stored.
                        (e.g., "checkpoints/model_epoch_10.pt").

    Raises:
        OSError: If unable to create the directory for saving the checkpoint.
    """
    # Ensure directory for checkpoint exists
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            raise OSError(f"Failed to create checkpoint directory: {directory}. Error: {e}")

    # Save the model's state dictionary
    torch.save(model.state_dict(), file_path)
    print(f"Checkpoint saved successfully to {file_path}")


def load_checkpoint(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Load a model's state dictionary from a checkpoint file.

    Args:
        file_path (str): Path to the checkpoint file (e.g., "checkpoints/model.pt").

    Returns:
        Dict[str, Any]: Loaded state dictionary if file exists, otherwise None.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {file_path}")
    
    # Load model state dictionary
    try:
        checkpoint = torch.load(file_path, map_location=torch.device("cpu"))
        print(f"Checkpoint loaded successfully from {file_path}")
        return checkpoint
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {file_path}. Error: {e}")


def generate_wavelet_coefficients(data: Tensor, wavelet_type: str, mode: str) -> List[Tensor]:
    """
    Decompose spatial-temporal data into wavelet coefficients.

    Args:
        data (Tensor): Input data tensor.
                       Shape:
                       - For 1D tasks: [batch_size, time_steps, spatial_points].
                       - For 2D tasks: [batch_size, time_steps, height, width].
        wavelet_type (str): Wavelet basis for decomposition (e.g., "bior2.4", "bior1.3").
        mode (str): Boundary mode for wavelet transform (e.g., "periodization", "zero").

    Returns:
        List[Tensor]: A list of tensors containing low-frequency and high-frequency coefficients.
    """
    transformer = WaveletTransform(wavelet_type=wavelet_type, mode=mode)
    return transformer.apply_transform(data)


def rebuild_from_wavelet_coefficients(coeffs: List[Tensor], wavelet_type: str, mode: str) -> Tensor:
    """
    Reconstruct original data from wavelet coefficients.

    Args:
        coeffs (List[Tensor]): Wavelet coefficients (low-frequency + high-frequency components).
        wavelet_type (str): Wavelet basis previously used for decomposition (e.g., "bior2.4", "bior1.3").
        mode (str): Boundary mode for wavelet transform (e.g., "periodization", "zero").

    Returns:
        Tensor: Reconstructed data tensor.
                Shape: Matches the shape of the input data during decomposition.
    """
    transformer = WaveletTransform(wavelet_type=wavelet_type, mode=mode)
    return transformer.inverse_transform(coeffs)


# Example for integration testing:
# ---------------------------------
# To use these utilities (e.g., during preprocessing or evaluation):
#
# >>> from utils import calculate_metrics, save_checkpoint, load_checkpoint, generate_wavelet_coefficients
# >>> import torch
# >>> sample_data = torch.randn(16, 80, 120)  # Example 1D data tensor
# >>>
# >>> # Example metrics calculation
# >>> predictions = sample_data + 0.1   # Adding slight noise
# >>> ground_truth = sample_data
# >>> metrics = calculate_metrics(predictions, ground_truth)
# >>> print(metrics)  # {"mse": 0.01} (example output)
# >>>
# >>> # Save model's state_dict
# >>> model = torch.nn.Linear(10, 5)
# >>> save_checkpoint(model, "checkpoints/example_model.pt")
# >>>
# >>> # Load state_dict to a model
# >>> restored_state_dict = load_checkpoint("checkpoints/example_model.pt")
# >>> model.load_state_dict(restored_state_dict)  # Restore model's weights
# >>>
# >>> # Perform wavelet decomposition
# >>> wavelet_coeffs = generate_wavelet_coefficients(sample_data, wavelet_type="bior2.4", mode="periodization")
# >>> reconstructed = rebuild_from_wavelet_coefficients(wavelet_coeffs, wavelet_type="bior2.4", mode="periodization")
# >>> print(reconstructed.shape)  # Should match sample_data shape
