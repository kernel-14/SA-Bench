import os
import random
from typing import List, Tuple, Union

import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image


def get_frequency_masks(
    H_prime: int,
    W_prime: int,
    n_bands: int,
    token_dims: List[Tuple[int, int]],
    device: torch.device,
) -> List[torch.Tensor]:
    """
    Generates a list of 2D binary masks for frequency band selection in the Fourier domain.
    The masks are defined by radial cutoffs based on the distribution of tokens across bands.

    Args:
        H_prime: Height of the latent feature map in the frequency domain.
        W_prime: Width of the latent feature map in the frequency domain.
        n_bands: Total number of frequency bands.
        token_dims: A list of (h_i, w_i) tuples, representing the dimensions of the token grid
                    for each frequency band. The sum of h_i * w_i for all bands is used
                    to determine the relative size of each frequency band's spectrum.
        device: The PyTorch device (e.g., 'cuda', 'cpu') to create the tensors on.

    Returns:
        A list of `n_bands` torch.Tensor masks, each of shape (H_prime, W_prime).
        These masks are float32 tensors with values 0 or 1.
    """
    if not token_dims or len(token_dims) != n_bands:
        raise ValueError(
            f"token_dims must be a list of {n_bands} (h, w) tuples, but got {len(token_dims)}."
        )

    # Create coordinate grids for the frequency domain, centered at (0,0) after fftshift.
    # The `fftfreq` function typically generates frequencies for an unshifted spectrum.
    # To align with Figure 4 (low frequency at center), we conceptually generate
    # coordinates for a shifted spectrum.
    u_coords = torch.linspace(-H_prime / 2, H_prime / 2 - 1, H_prime, device=device)
    v_coords = torch.linspace(-W_prime / 2, W_prime / 2 - 1, W_prime, device=device)
    U, V = torch.meshgrid(u_coords, v_coords, indexing="ij")

    # Calculate radial frequency map: distance from the center (0,0)
    # This represents 'radial_frequency' f_r = sqrt(u^2 + v^2)
    radial_freq_map = torch.sqrt(U**2 + V**2)

    # Determine sigma_max: the maximum radial frequency in the grid.
    # This corresponds to the highest possible frequency component.
    sigma_max = radial_freq_map.max().item()
    if sigma_max == 0:  # Handle case for 1x1 feature map to avoid division by zero
        # If H_prime or W_prime is 1, radial_freq_map might be all zeros or very small.
        # In this edge case, return masks that are all ones, assuming all frequencies are 'low'.
        return [torch.ones((H_prime, W_prime), device=device, dtype=torch.float32)] * n_bands

    # Calculate sum of all token counts (h_i * w_i) across all bands
    sum_hi_wi = sum(h * w for h, w in token_dims)
    if sum_hi_wi == 0:
        raise ValueError("Sum of h_i * w_i for token_dims cannot be zero.")

    # Calculate sigma_i cutoffs based on the paper's formula:
    # sigma_i = sigma_{i-1} + (h_i * w_i / sum(h_j * w_j)) * sigma_max
    sigma_cutoffs = [0.0]  # sigma_0, the lower bound of the first frequency band
    for i in range(n_bands):
        current_h_i_w_i = token_dims[i][0] * token_dims[i][1]
        next_sigma = sigma_cutoffs[-1] + (current_h_i_w_i / sum_hi_wi) * sigma_max
        sigma_cutoffs.append(next_sigma)

    # Ensure the last cutoff exactly matches sigma_max due to potential floating-point inaccuracies
    sigma_cutoffs[-1] = sigma_max

    # Generate binary masks for each frequency band
    masks = []
    for i in range(n_bands):
        # The i-th band selects frequencies in the range [sigma_cutoffs[i], sigma_cutoffs[i+1])
        # For the last band, it should include sigma_max, so use <=
        if i == n_bands - 1:
            mask_i = (radial_freq_map >= sigma_cutoffs[i]) & (radial_freq_map <= sigma_cutoffs[i + 1])
        else:
            mask_i = (radial_freq_map >= sigma_cutoffs[i]) & (radial_freq_map < sigma_cutoffs[i + 1])
        masks.append(mask_i.float())

    return masks


def interpolate_feature_map(
    feature_map: torch.Tensor,
    target_size: Tuple[int, int],
    mode: str = "bilinear",
    align_corners: bool = False,
) -> torch.Tensor:
    """
    Resizes a feature map tensor using `torch.nn.functional.interpolate`.

    Args:
        feature_map: The input feature map tensor (B, C, H, W).
        target_size: The desired output size as (height, width).
        mode: The interpolation algorithm. Common options include 'nearest', 'bilinear', 'bicubic'.
              Default is 'bilinear'.
        align_corners: If True, the corner pixels of the input and output tensors are aligned.
                       This argument is only relevant for `mode` in ['linear', 'bilinear',
                       'bicubic', 'trilinear'] and should be set to False for consistency
                       with common image processing practices in deep learning unless
                       specific behavior is desired.

    Returns:
        The interpolated feature map tensor of shape (B, C, target_size[0], target_size[1]).
    """
    if feature_map.ndim != 4:
        raise ValueError(f"Input feature_map must be 4D (B, C, H, W), but got {feature_map.ndim}D.")
    if not isinstance(target_size, tuple) or len(target_size) != 2:
        raise ValueError("target_size must be a tuple of two integers (height, width).")
    if not all(isinstance(s, int) and s > 0 for s in target_size):
        raise ValueError("target_size dimensions must be positive integers.")
        
    return F.interpolate(feature_map, size=target_size, mode=mode, align_corners=align_corners)


def compute_lpips(
    img1: torch.Tensor, img2: torch.Tensor, lpips_model: lpips.LPIPS
) -> torch.Tensor:
    """
    Calculates the LPIPS (Learned Perceptual Image Patch Similarity) perceptual loss
    between two image tensors.

    Args:
        img1: First image tensor (B, C, H, W), expected to be in the range [-1, 1].
        img2: Second image tensor (B, C, H, W), expected to be in the range [-1, 1].
        lpips_model: An initialized LPIPS model instance (e.g., lpips.LPIPS(net='alex')).

    Returns:
        A scalar tensor representing the mean LPIPS distance across the batch.
    """
    if not isinstance(lpips_model, lpips.LPIPS):
        raise TypeError("lpips_model must be an instance of lpips.LPIPS.")
    if img1.shape != img2.shape:
        raise ValueError(f"Image tensors must have the same shape, but got {img1.shape} and {img2.shape}.")
    if img1.ndim != 4 or img2.ndim != 4:
        raise ValueError(f"Input images must be 4D tensors (B, C, H, W).")

    # Ensure images are on the same device as the LPIPS model
    img1 = img1.to(lpips_model.device)
    img2 = img2.to(lpips_model.device)

    # The LPIPS model expects inputs in the range [-1, 1].
    # It handles internal normalization to [0, 1] if needed.
    return lpips_model(img1, img2).mean()


def init_weights(m: nn.Module) -> None:
    """
    Initializes the weights of neural network modules using a standard scheme.
    - Kaiming uniform initialization for convolutional and linear layers.
    - Constant initialization for batch normalization and group normalization layers.

    Args:
        m: A PyTorch module (e.g., nn.Conv2d, nn.Linear, nn.BatchNorm2d).
    """
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.weight is not None:
            nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def straight_through_estimator(quantized_output: torch.Tensor, original_input: torch.Tensor) -> torch.Tensor:
    """
    Implements the Straight-Through Estimator (STE) for quantization operations.
    This technique allows gradients to pass through a non-differentiable quantization step.
    In the forward pass, it uses `quantized_output`. In the backward pass, it copies
    the gradients of `quantized_output` to `original_input`.

    Args:
        quantized_output: The output tensor from a quantization operation.
        original_input: The continuous input tensor that was fed into the quantization.

    Returns:
        A tensor whose forward pass value is `quantized_output`, but whose
        gradient computation for backpropagation uses `original_input`.
    """
    if quantized_output.shape != original_input.shape:
        raise ValueError("quantized_output and original_input must have the same shape.")
    return original_input + (quantized_output - original_input).detach()


def convert_image_to_tensor(image_path: str, image_size: int = 256) -> torch.Tensor:
    """
    Loads an image from a given path, resizes it, and converts it into a
    normalized PyTorch tensor suitable for model input.

    Args:
        image_path: The file system path to the image.
        image_size: The target height and width for resizing the image. Default is 256.

    Returns:
        A torch.Tensor of shape (3, image_size, image_size), with pixel values
        normalized to the range [-1, 1].
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")
    if not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("image_size must be a positive integer.")

    # Define standard image transformations: resize, convert to tensor, normalize to [-1, 1]
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # Converts to [0, 1] range, (C, H, W)
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Converts to [-1, 1]
        ]
    )

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image)
    return image_tensor


def set_seed(seed: int) -> None:
    """
    Sets the random seed for Python's `random` module, `numpy`, and `torch`
    (including CUDA operations if available) to ensure reproducibility.

    Args:
        seed: The integer seed value to use for all random number generators.
    """
    if not isinstance(seed, int):
        raise TypeError("Seed must be an integer.")
    if seed < 0:
        raise ValueError("Seed must be a non-negative integer.")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Ensure deterministic behavior in CuDNN, potentially at the cost of some performance
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

