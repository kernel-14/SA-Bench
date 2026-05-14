import math
import torch
import torch.nn as nn
from typing import Tuple


class NormalizationUtils:
    """
    Utility class for normalization operations and Rotary Positional Embeddings (RoPE).
    """

    @staticmethod
    def norm(x: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the input tensor `x` to have unit L2 norm along its last dimension.
        Adds a small epsilon for numerical stability to prevent division by zero.

        Args:
            x: The input tensor to be normalized.

        Returns:
            The L2-normalized tensor.
        """
        # Calculate L2 norm along the last dimension, keeping the dimension for broadcasting
        norm_values = torch.norm(x, p=2, dim=-1, keepdim=True)
        # Add a small epsilon for numerical stability. Using torch.finfo for dtype-specific epsilon.
        epsilon = torch.finfo(x.dtype).eps
        norm_values = norm_values.clamp_min(epsilon)
        normalized_x = x / norm_values
        return normalized_x

    @staticmethod
    def normalize_embedding_dim(param: torch.nn.Parameter) -> None:
        """
        Normalizes a torch.nn.Parameter in-place along its embedding dimension (last dimension)
        to have unit L2 norm. This operation is performed without tracking gradients.

        Args:
            param: The torch.nn.Parameter to be normalized.
        """
        with torch.no_grad():
            # Apply the norm function to the underlying data of the parameter
            # This directly modifies the parameter's data without affecting the autograd graph
            param.data = NormalizationUtils.norm(param.data)

    @staticmethod
    def apply_rope(x: torch.Tensor, current_pos: int, rope_base: float = 10000.0) -> torch.Tensor:
        """
        Applies Rotary Positional Embeddings (RoPE) to the input tensor `x`.
        Assumes `x` has shape (batch_size, num_heads, seq_len, head_dim).

        Args:
            x: The input tensor (e.g., query or key vectors) to apply RoPE to.
               Expected shape: (batch_size, num_heads, seq_len, head_dim).
            current_pos: The starting position index for the current sequence.
                         Typically 0 for training, or a continuation point for inference.
            rope_base: The base value for the RoPE frequency calculation.

        Returns:
            The tensor with RoPE applied.
        """
        batch_size, num_heads, seq_len, head_dim = x.shape

        # Create position_ids for the current sequence segment
        # shape: (seq_len,)
        position_ids = torch.arange(current_pos, current_pos + seq_len, dtype=torch.float32, device=x.device)

        # Calculate inverse frequencies
        # shape: (head_dim // 2,)
        inv_freq = 1.0 / (rope_base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=x.device) / head_dim))

        # Compute theta (angles for rotation)
        # shape: (seq_len, head_dim // 2)
        theta = torch.outer(position_ids, inv_freq)

        # Reshape theta to (1, 1, seq_len, head_dim // 2) for broadcasting with x
        # x_first_half and x_second_half will have shape (batch_size, num_heads, seq_len, head_dim // 2)
        cos_theta = theta.cos().unsqueeze(0).unsqueeze(0)
        sin_theta = theta.sin().unsqueeze(0).unsqueeze(0)

        # Split x into two halves along the head_dim
        # x has shape (batch_size, num_heads, seq_len, head_dim)
        x_first_half = x[..., ::2]  # Elements at even indices
        x_second_half = x[..., 1::2] # Elements at odd indices

        # Apply the rotary transformation
        x_rotated_first = (x_first_half * cos_theta) - (x_second_half * sin_theta)
        x_rotated_second = (x_first_half * sin_theta) + (x_second_half * cos_theta)

        # Interleave the rotated halves to reconstruct the tensor
        x_rotated = torch.stack((x_rotated_first, x_rotated_second), dim=-1)
        x_rotated = x_rotated.flatten(-2, -1) # Flatten the last two dimensions (..., 2, head_dim//2) -> (..., head_dim)

        return x_rotated


class ScaledLearnableParameter(nn.Module):
    """
    A custom PyTorch Module encapsulating a learnable parameter that has a specific
    initialization strategy and a scaling mechanism applied during the forward pass,
    as described in Section 2.5 and 2.6 of the NGPT paper.

    The actual `torch.nn.Parameter` (`param_tensor`) is initialized to `s_scale`.
    During `get_effective_value()`, its value is adjusted using `s_init` and `s_scale`.
    """

    def __init__(self, size: Tuple[int, ...], s_init: float, s_scale: float, name: str = ""):
        """
        Initializes the ScaledLearnableParameter.

        Args:
            size: A tuple defining the shape of the learnable parameter tensor.
            s_init: The 's_a,init' value for this parameter (initial value in the effective formula).
            s_scale: The 's_a,scale' value for this parameter (initial value of the actual nn.Parameter
                     and denominator in the effective formula).
            name: An optional name for this parameter, useful for debugging.
        """
        super().__init__()
        if not isinstance(size, tuple):
            raise TypeError("Size must be a tuple of integers.")
        if not all(isinstance(dim, int) and dim > 0 for dim in size):
            raise ValueError("All dimensions in size must be positive integers.")
        if not isinstance(s_init, (int, float)):
            raise TypeError("s_init must be an int or float.")
        if not isinstance(s_scale, (int, float)):
            raise TypeError("s_scale must be an int or float.")
        if s_scale == 0:
            raise ValueError("s_scale cannot be zero to avoid division by zero.")

        self.s_init: float = s_init
        self.s_scale: float = s_scale
        self.name: str = name

        # The actual learnable parameter is initialized to s_scale
        self.param_tensor: nn.Parameter = nn.Parameter(torch.full(size, s_scale, dtype=torch.float32))

    def get_effective_value(self) -> torch.Tensor:
        """
        Computes and returns the effective value of the scaling factor or eigen learning rate
        to be used in the model's forward pass.

        The formula is: `effective_value = param_tensor * (s_init / s_scale)`
        where `param_tensor` is the actual learnable `nn.Parameter`.

        Returns:
            A torch.Tensor representing the effective scaled value.
        """
        # Ensure s_scale is not zero to prevent division by zero, though checked in __init__
        if self.s_scale == 0:
            # Fallback for safety, though it should be caught during init
            raise ValueError(f"s_scale for {self.name} is zero, cannot compute effective value.")
        return self.param_tensor * (self.s_init / self.s_scale)

    def __repr__(self):
        """
        Provides a string representation of the module.
        """
        return (f"{self.__class__.__name__}(name='{self.name}', size={tuple(self.param_tensor.shape)}, "
                f"s_init={self.s_init}, s_scale={self.s_scale}, "
                f"param_tensor_val={self.param_tensor.mean().item():.4f})")

