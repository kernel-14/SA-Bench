"""
utils/helpers.py

Utility functions for the NFIG reproduction project.
Includes configuration loading, interpolation, frequency-domain operations,
token extraction, and batching helpers.

All functions are stateless and do not require project-specific imports
to avoid circular dependencies.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable
from omegaconf import OmegaConf


def load_config(path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file using OmegaConf.

    Args:
        path: Filesystem path to the config.yaml file.

    Returns:
        A nested dictionary containing all configuration keys.
    """
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def resize_tensor(
    tensor: torch.Tensor,
    target_h: int,
    target_w: int,
    mode: str = "bilinear"
) -> torch.Tensor:
    """
    Resize a 4D tensor (B, C, H, W) to (B, C, target_h, target_w) using bilinear
    or nearest interpolation.

    Args:
        tensor: Input feature map of shape (B, C, H, W).
        target_h: Desired height.
        target_w: Desired width.
        mode: Interpolation mode, either 'bilinear' or 'nearest'. Default 'bilinear'.

    Returns:
        Resized tensor of shape (B, C, target_h, target_w).
    """
    align_corners = False if mode == "bilinear" else None
    return F.interpolate(
        tensor,
        size=(target_h, target_w),
        mode=mode,
        align_corners=align_corners,
    )


# ----------------------------------------------------------------------
# Frequency-domain helpers
# ----------------------------------------------------------------------

def compute_radial_frequencies(H: int, W: int) -> torch.Tensor:
    """
    Compute a 2D grid of radial distances from the zero-frequency centre
    (after FFT shift). The DC component is at the centre.

    Args:
        H: Height of the spatial frequency grid.
        W: Width of the spatial frequency grid.

    Returns:
        A float tensor of shape (H, W) containing radial distances.
    """
    # Coordinates centred at zero: DC at (W//2, H//2) after fftshift.
    u = torch.arange(W) - W // 2
    v = torch.arange(H) - H // 2
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    radius = torch.sqrt(uu ** 2 + vv ** 2)
    return radius


def generate_frequency_masks(
    scale_sizes: List[int],
    H: int,
    W: int
) -> List[torch.Tensor]:
    """
    Generate binary masks for each frequency band according to the cumulative
    frequency division strategy described in the paper.

    The division ensures that the number of tokens per band (scale_sizes[i]**2)
    is proportional to the allocated frequency range. The final band includes
    all remaining frequencies.

    Args:
        scale_sizes: List of h_i (and w_i) dimensions for each band, e.g. [1,2,3,...16].
        H: Spatial height of the feature map (16 for ImageNet 256x256).
        W: Spatial width (16).

    Returns:
        A list of masks, each of shape (1, 1, H, W), broadcastable over
        batch and channel dimensions. The 'i'-th mask corresponds to the i-th
        frequency band.
    """
    # Total tokens across all bands (sum of h_i^2)
    total_tokens = sum(s * s for s in scale_sizes)
    # Radial distances over the H x W grid
    radius = compute_radial_frequencies(H, W)
    sigma_max = radius.max().item()

    masks = []
    sigma_prev = 0.0
    n_bands = len(scale_sizes)

    for i, s in enumerate(scale_sizes):
        tokens_this = s * s
        sigma_i = sigma_prev + (tokens_this / total_tokens) * sigma_max

        # Special handling for the last band: include all remaining frequencies
        # to avoid floating-point misses at the boundary.
        if i == n_bands - 1:
            mask = radius >= sigma_prev
        else:
            mask = (radius >= sigma_prev) & (radius < sigma_i)

        # Convert to float and unsqueeze to (1,1,H,W) for easy broadcasting.
        mask = mask.to(torch.float32).unsqueeze(0).unsqueeze(0)
        masks.append(mask)

        sigma_prev = sigma_i

    return masks


def fft2_shifted(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the 2D FFT of a real tensor and shift the zero-frequency component
    to the centre of the spectrum.

    Args:
        x: Real input tensor of shape (..., H, W).

    Returns:
        Complex tensor of the same shape with DC at the centre.
    """
    z = torch.fft.fft2(x)
    return torch.fft.fftshift(z)


def ifft2_real(
    x_fft_shifted: torch.Tensor,
    signal_sizes: Optional[Tuple[int, int]] = None
) -> torch.Tensor:
    """
    Inverse 2D FFT from a shifted complex spectrum back to the real spatial domain.

    Args:
        x_fft_shifted: Shifted complex tensor (..., H, W).
        signal_sizes: Optional output spatial size (H, W). If None, the input
            spatial dimensions are used.

    Returns:
        Real tensor of shape (..., H_out, W_out).
    """
    z = torch.fft.ifftshift(x_fft_shifted)
    x = torch.fft.ifft2(z, s=signal_sizes)
    return torch.real(x)


def apply_frequency_mask(
    f_fft_shifted: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Apply a binary mask in the frequency domain by element-wise multiplication.

    Args:
        f_fft_shifted: Complex tensor (..., H, W) with DC centred.
        mask: Real tensor broadcastable to f_fft_shifted, e.g. (1,1,H,W).

    Returns:
        Complex tensor of same shape as f_fft_shifted.
    """
    return f_fft_shifted * mask


def extract_frequency_component(
    f: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Extract the band-limited spatial component corresponding to a given frequency mask.

    Steps:
        1. FFT of the input feature map.
        2. Apply the mask in the frequency domain.
        3. Inverse FFT to obtain the spatial component.

    Args:
        f: Real feature map of shape (B, C, H, W).
        mask: A binary mask from generate_frequency_masks, shape (1,1,H,W).

    Returns:
        Real tensor of shape (B, C, H, W) containing only the frequencies
        selected by the mask.
    """
    B, C, H, W = f.shape
    f_fft = fft2_shifted(f)
    f_fft_masked = apply_frequency_mask(f_fft, mask)
    f_i = ifft2_real(f_fft_masked, signal_sizes=(H, W))
    return f_i


# ----------------------------------------------------------------------
# Token extraction, saving/loading, batching
# ----------------------------------------------------------------------

def extract_tokens(
    model: Any,  # expects FRVAE with tokenize() method, avoid circular imports
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> Tuple[List[Dict[int, torch.Tensor]], List[int]]:
    """
    Process all images through the trained FR‑VAE model and collect discrete token IDs
    for every frequency band.

    The model is expected to provide a method:
        tokenize(image: Tensor) -> List[Tensor]
    where each tensor in the list contains flattened token indices for that
    frequency band (shape (hi*wi,)).

    Args:
        model: FRVAE instance in eval mode.
        dataloader: DataLoader yielding (images, labels) batches.
        device: torch device on which to run the model.

    Returns:
        A tuple (all_tokens, all_labels):
            - all_tokens: list (length N) of dictionaries mapping scale index to
              token indices tensor (shape (hi*wi,)).
            - all_labels: list of integer class labels.
    """
    model.eval()
    all_tokens = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            # tokenize() returns a list of tensors, one per scale, each (B, hi*wi)
            batch_token_lists = model.tokenize(images)

            for b in range(images.size(0)):
                sample_tokens = {}
                for scale_idx, tok_batch in enumerate(batch_token_lists):
                    sample_tokens[scale_idx] = tok_batch[b].cpu().clone()
                all_tokens.append(sample_tokens)
                all_labels.append(labels[b].item())

    return all_tokens, all_labels


def save_tokens(
    tokens: List[Dict[int, torch.Tensor]],
    labels: List[int],
    output_file: str
) -> None:
    """
    Persist extracted tokens and labels to disk in a consolidated format.

    The format stores one tensor per scale stacked over all samples, plus a
    tensor of labels. This enables efficient slicing for training.

    Args:
        tokens: List of N dictionaries, each mapping scale_idx -> tensor of shape (hi*wi,).
        labels: List of N integer class labels.
        output_file: File path to store the data (usually .pt).
    """
    if len(tokens) == 0:
        raise ValueError("Cannot save empty token list.")
    n_samples = len(tokens)
    num_scales = len(tokens[0])  # number of frequency bands

    # Build a dictionary: key 'tokens_scale_<i>', value tensor of shape (N, hi*wi)
    data = {}
    for i in range(num_scales):
        stacked = torch.stack([sample[i] for sample in tokens], dim=0)
        data[f"tokens_scale_{i}"] = stacked

    # Store labels as a 1D tensor
    data["labels"] = torch.tensor(labels, dtype=torch.long)

    torch.save(data, output_file)


def load_tokens(file_path: str) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Load pre‑stacked token data and labels from a .pt file.

    Args:
        file_path: Path to the saved token file.

    Returns:
        A tuple (token_dict, labels):
            - token_dict: dictionary with keys like 'tokens_scale_<i>', values are
              tensors of shape (N, hi*wi).
            - labels: 1D tensor of shape (N,) containing class labels.
    """
    data = torch.load(file_path)
    labels = data.pop("labels")
    return data, labels


def collate_token_batch(
    batch: List[Tuple[Dict[int, torch.Tensor], int]]
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Custom collate function for a DataLoader that yields tokenized samples.

    Each sample is expected to be a tuple (token_dict, label), where token_dict
    maps scale index (int) to a 1D tensor of token indices (shape hi*wi,).
    The function stacks individual samples into batched tensors for each scale.

    Args:
        batch: List of tuples (token_dict, label).

    Returns:
        Tuple (scale_tensors, labels):
            - scale_tensors: list of tensors, one per scale, shape (B, hi*wi).
            - labels: tensor of shape (B,) containing class labels.
    """
    if len(batch) == 0:
        return [], torch.empty(0, dtype=torch.long)

    # Determine number of scales from the first sample
    first_tok_dict, _ = batch[0]
    num_scales = len(first_tok_dict)

    scale_tensors = []
    for i in range(num_scales):
        # Gather tokens for scale i from each sample
        tensors = [sample[0][i] for sample in batch]
        scale_tensors.append(torch.stack(tensors, dim=0))

    labels = torch.stack([torch.tensor(sample[1], dtype=torch.long) for sample in batch], dim=0)
    return scale_tensors, labels
