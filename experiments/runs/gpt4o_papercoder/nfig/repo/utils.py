## utils.py

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from scipy.fft import fft2, ifft2


def apply_fft(image: Tensor) -> Tensor:
    """Performs a forward 2D FFT on spatial domain tensors.

    Args:
        image (Tensor): Input image tensor in spatial domain, 
                        shape (H, W, C) where H=height, W=width, C=channels.

    Returns:
        Tensor: Output tensor in frequency domain, 
                shape (H, W, C) containing complex numbers.
    """
    if not isinstance(image, Tensor):
        raise TypeError("Input 'image' must be a torch.Tensor")
    if image.dim() != 3:
        raise ValueError("Input tensor must have 3 dimensions (H, W, C)")
    
    # Apply FFT on each channel independently
    freq_components = torch.stack([torch.from_numpy(fft2(image[..., c].numpy())) for c in range(image.shape[-1])], dim=-1)
    return freq_components


def apply_inverse_fft(freq_components: Tensor) -> Tensor:
    """Performs an inverse 2D FFT on frequency domain tensors.

    Args:
        freq_components (Tensor): Input frequency-domain tensor, 
                                  shape (H, W, C) containing complex numbers.

    Returns:
        Tensor: Output tensor in spatial domain, 
                shape (H, W, C).
    """
    if not isinstance(freq_components, Tensor):
        raise TypeError("Input 'freq_components' must be a torch.Tensor")
    if freq_components.dim() != 3:
        raise ValueError("Input tensor must have 3 dimensions (H, W, C)")
    
    # Apply inverse FFT on each channel independently
    spatial_image = torch.stack(
        [torch.from_numpy(ifft2(freq_components[..., c].numpy()).real) for c in range(freq_components.shape[-1])], dim=-1
    )
    return spatial_image


def split_frequency_bands(features: Tensor, num_bands: int) -> list:
    """Divides input features into frequency bands using FFT masks.

    Args:
        features (Tensor): Latent feature representation, shape (H', W', C).
        num_bands (int): Number of frequency bands to divide the features into.

    Returns:
        list: List of tensors, each representing a frequency band, 
              shape of each tensor is (H', W', C).
    """
    if not isinstance(features, Tensor):
        raise TypeError("Input 'features' must be a torch.Tensor")
    if features.dim() != 3:
        raise ValueError("Input tensor must have 3 dimensions (H', W', C)")
    if num_bands <= 0:
        raise ValueError("Number of frequency bands 'num_bands' must be greater than 0")

    # Compute FFT for features
    freq_features = apply_fft(features)
    masks = generate_frequency_masks(freq_features, num_bands)
    frequency_bands = [
        apply_inverse_fft(freq_features * mask) for mask in masks
    ]
    return frequency_bands


def scale_tokens(tokens: Tensor, target_size: tuple) -> Tensor:
    """Scales tokens to the target resolution using interpolation.

    Args:
        tokens (Tensor): Input token tensor, shape (h, w, C).
        target_size (tuple): Target dimensions (h', w').

    Returns:
        Tensor: Rescaled token tensor, shape (h', w', C).
    """
    if not isinstance(tokens, Tensor):
        raise TypeError("Input 'tokens' must be a torch.Tensor")
    if tokens.dim() != 3:
        raise ValueError("Input tensor must have 3 dimensions (h, w, C)")
    if not isinstance(target_size, tuple) or len(target_size) != 2:
        raise ValueError("Target size must be a tuple with two elements (h', w')")
    
    scaled_tokens = F.interpolate(
        tokens.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False
    ).squeeze(0)
    return scaled_tokens


def generate_frequency_masks(features: Tensor, num_bands: int) -> list:
    """Generates frequency masks based on spectral energy distribution.

    Args:
        features (Tensor): Frequency-domain tensor, shape (H', W', C).
        num_bands (int): Number of frequency bands to divide into.

    Returns:
        list: List of tensors, where each tensor is a binary mask 
              corresponding to a frequency band, shape (H', W', C).
    """
    if not isinstance(features, Tensor):
        raise TypeError("Input 'features' must be a torch.Tensor")
    if features.dim() != 3:
        raise ValueError("Input tensor must have 3 dimensions (H', W', C)")
    if num_bands <= 0:
        raise ValueError("Number of frequency bands 'num_bands' must be greater than 0")

    # Compute the power spectrum for spectral energy distribution
    magnitude_spectrum = torch.sqrt((features.real ** 2) + (features.imag ** 2))
    total_energy = magnitude_spectrum.sum(dim=(0, 1))  # Compute energy per channel

    thresholds = torch.linspace(0, total_energy.max().item(), num_bands + 1)
    masks = []

    for i in range(num_bands):
        lower_bound = thresholds[i]
        upper_bound = thresholds[i + 1]
        mask = (magnitude_spectrum >= lower_bound) & (magnitude_spectrum <= upper_bound)
        masks.append(mask.float())

    return masks
