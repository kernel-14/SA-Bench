"""
utils.py
Utility functions required for preprocessing, initialization, and hardware configuration.
Adheres to the specified design and logic analysis, with references to config.yaml values.
"""
import torch
import numpy as np
import random
from typing import Tuple

def compute_truncated_svd(matrix: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform truncated Singular Value Decomposition (SVD) on the input matrix.

    Args:
        matrix (torch.Tensor): Input 2D tensor for decomposition.
        rank (int): Desired rank for the truncated SVD, must be <= min(matrix.shape).

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
            - U_selected (Top `rank` left singular vectors).
            - S_selected (Top `rank` singular values diagonal matrix).
            - V_selected (Top `rank` right singular vectors).
    """
    if not isinstance(matrix, torch.Tensor):
        raise TypeError("matrix must be a torch.Tensor.")
    if matrix.ndim != 2:
        raise ValueError("matrix must be a 2D tensor.")
    if rank <= 0 or rank > min(matrix.shape):
        raise ValueError("rank must be greater than 0 and less than or equal to the smallest dimension of matrix.")
    
    # Perform SVD with PyTorch's optimized backend
    U, S, Vt = torch.linalg.svd(matrix, full_matrices=False)  # U[m×m], S[m×n], Vt[n×n]
    
    # Truncate to required rank
    U_selected = U[:, :rank]  # Top `rank` columns of U
    S_selected = torch.diag(S[:rank])  # Top `rank` singular values as a diagonal matrix
    V_selected = Vt[:rank, :]  # Top `rank` rows of Vt

    return U_selected, S_selected, V_selected


def set_device(use_gpu: bool = True) -> torch.device:
    """
    Determine and configure whether GPU or CPU should be the computational device.

    Args:
        use_gpu (bool): Whether to prioritize GPU usage if available.

    Returns:
        torch.device: The selected computational device (either CUDA or CPU).
    """
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Device set to GPU: cuda:0")
    else:
        device = torch.device("cpu")
        print("Device set to CPU.")
    return device


def set_random_seed(seed: int = 42) -> None:
    """
    Set global random seed across PyTorch, NumPy, and Python for reproducibility.

    Args:
        seed (int): Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to: {seed}")


def normalize_gradients(gradients: torch.Tensor, norm_type: str = "frobenius") -> torch.Tensor:
    """
    Normalize gradients to ensure numerical stability.

    Args:
        gradients (torch.Tensor): Input gradient matrix to normalize.
        norm_type (str): Norm type for normalization ["frobenius", "l2"].

    Returns:
        torch.Tensor: Normalized gradients.
    """
    if not isinstance(gradients, torch.Tensor):
        raise TypeError("gradients must be a torch.Tensor.")
    
    if norm_type == "frobenius":
        norm_value = torch.norm(gradients, p="fro")
    elif norm_type == "l2":
        norm_value = torch.norm(gradients)
    else:
        raise ValueError("Invalid norm_type. Use 'frobenius' or 'l2'.")
    
    if norm_value == 0:
        return torch.zeros_like(gradients)
    
    normalized_gradients = gradients / norm_value
    return normalized_gradients


def aggregate_gradients(
    model: torch.nn.Module, 
    dataset_subset: torch.utils.data.DataLoader, 
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Aggregate gradients over a small subset of dataset for initialization.

    Args:
        model (torch.nn.Module): Pre-trained model loaded for gradient extraction.
        dataset_subset (torch.utils.data.DataLoader): Subset DataLoader to compute gradients.
        device (torch.device): Target device for computation (GPU/CPU).

    Returns:
        torch.Tensor: Averaged gradient matrix from the subset.
    """
    model.eval()  # Ensure evaluation mode for gradient extraction
    model.to(device)

    accumulated_gradients = None
    sample_count = 0

    for batch_idx, batch in enumerate(dataset_subset):
        # Move batch to the correct device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss  # Assumes the model computes loss for the subset
        
        # Backward pass to compute gradients
        loss.backward()

        # Aggregate gradients layer by layer
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if accumulated_gradients is None:
                    accumulated_gradients = param.grad.clone().detach()
                else:
                    accumulated_gradients += param.grad.clone().detach()
        
        sample_count += 1
    
    if accumulated_gradients is None:
        raise RuntimeError("Failed to accumulate gradients. Ensure model gradients are accessible.")

    # Compute averaged gradients
    averaged_gradients = accumulated_gradients / sample_count
    return averaged_gradients
