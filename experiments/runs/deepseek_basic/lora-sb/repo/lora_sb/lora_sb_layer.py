"""
LoRA-SB Layer Implementation.

Architecture: W = W0 + s * B @ R @ A

Where:
- W0: frozen pre-trained weight matrix (m × n)
- B: frozen matrix (m × r), initialized via truncated SVD of ΔW_avg
- A: frozen matrix (r × n), initialized via truncated SVD of ΔW_avg
- R: trainable matrix (r × r), initialized as S[:r,:r] / s
- s: scaling factor (can be set to 1 with orthonormal B and A)

During training, only R is updated. The gradient with respect to R is:
    g_R = s * B^T @ g @ A^T

where g is the gradient with respect to W.

With orthonormal B and A (B^T B = I, A A^T = I) and s = 1, the optimal
gradient approximation simplifies to g_R = g_LoRA-XS_R.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Union, Tuple


class LoRA_SB_Layer(nn.Module):
    """
    LoRA-SB layer that wraps a linear layer with the LoRA-XS architecture.

    This implements W = W0 + s * B @ R @ A where B and A are frozen after
    initialization and only R is trainable.

    The forward pass computes: y = x @ W0^T + s * ((x @ A^T) @ R^T) @ B^T

    Args:
        in_features: input dimension
        out_features: output dimension
        rank: rank of the low-rank decomposition (r)
        scaling: scaling factor s (default: 1.0; with orthonormal B, A, s=1 works)
        bias: whether the original layer has bias
        device: device to place parameters
        dtype: dtype for parameters
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        scaling: float = 1.0,
        bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = scaling

        # Initialize frozen B with orthonormal columns (initially random, will be set by init)
        B = torch.zeros(out_features, rank, device=device, dtype=dtype)
        nn.init.kaiming_uniform_(B, a=math.sqrt(5))
        B = self._orthonormalize(B)
        self.register_buffer("B", B)

        # Initialize trainable R
        R = torch.zeros(rank, rank, device=device, dtype=dtype)
        nn.init.zeros_(R)
        self.R = nn.Parameter(R)

        # Initialize frozen A with orthonormal rows (initially random, will be set by init)
        A = torch.zeros(rank, in_features, device=device, dtype=dtype)
        nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        A = self._orthonormalize_rows(A)
        self.register_buffer("A", A)

        # Original weight (frozen) - set later
        self.register_buffer("W0", None)

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
        else:
            self.bias = None

        self._initialized = False

    @staticmethod
    def _orthonormalize(matrix: torch.Tensor) -> torch.Tensor:
        """Make columns orthonormal using QR decomposition."""
        if matrix.size(0) >= matrix.size(1):
            Q, _ = torch.linalg.qr(matrix)
            return Q
        return matrix

    @staticmethod
    def _orthonormalize_rows(matrix: torch.Tensor) -> torch.Tensor:
        """Make rows orthonormal."""
        if matrix.size(1) >= matrix.size(0):
            Q, _ = torch.linalg.qr(matrix.T)
            return Q.T
        return matrix

    @classmethod
    def from_pretrained_linear(
        cls,
        linear_layer: nn.Linear,
        rank: int,
        scaling: float = 1.0,
    ) -> "LoRA_SB_Layer":
        """
        Create a LoRA-SB layer from a pre-trained linear layer.

        The original weight is stored as W0 (frozen).
        """
        lora_sb = cls(
            in_features=linear_layer.in_features,
            out_features=linear_layer.out_features,
            rank=rank,
            scaling=scaling,
            bias=linear_layer.bias is not None,
            device=linear_layer.weight.device,
            dtype=linear_layer.weight.dtype,
        )
        lora_sb.W0 = nn.Parameter(linear_layer.weight.data.clone(), requires_grad=False)
        if linear_layer.bias is not None:
            lora_sb.bias = nn.Parameter(linear_layer.bias.data.clone(), requires_grad=False)
        return lora_sb

    def initialize_ba(
        self,
        B_init: torch.Tensor,
        R_init: torch.Tensor,
        A_init: torch.Tensor,
    ):
        """
        Initialize B, R, A with pre-computed values from truncated SVD.

        Args:
            B_init: (out_features, rank) tensor - U[:, :r] from SVD
            R_init: (rank, rank) tensor - S[:r, :r] / s from SVD
            A_init: (rank, in_features) tensor - V[:r, :] from SVD
        """
        assert B_init.shape == (self.out_features, self.rank), \
            f"B_init shape {B_init.shape} != ({self.out_features}, {self.rank})"
        assert R_init.shape == (self.rank, self.rank), \
            f"R_init shape {R_init.shape} != ({self.rank}, {self.rank})"
        assert A_init.shape == (self.rank, self.in_features), \
            f"A_init shape {A_init.shape} != ({self.rank}, {self.in_features})"

        self.B.copy_(B_init.to(dtype=self.B.dtype, device=self.B.device))
        self.R.data.copy_(R_init.to(dtype=self.R.dtype, device=self.R.device))
        self.A.copy_(A_init.to(dtype=self.A.dtype, device=self.A.device))
        self._initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: y = x @ W0^T + s * x @ A^T @ R^T @ B^T

        For an input x of shape (..., in_features):
        - Original: x @ W0^T -> (..., out_features)
        - LoRA-SB term: s * x @ A^T @ R^T @ B^T

        Computational path:
        1. x @ A^T = x @ A^T where A is (r, n) stored as weight → (..., r)
        2. intermediate @ R^T where R is (r, r) stored as weight → (..., r)
        3. intermediate2 @ B^T where B is (m, r) stored as weight → (..., m)
        """
        if self.W0 is None:
            raise ValueError("W0 not set. Call from_pretrained_linear or set W0 manually.")

        # Original output
        result = F.linear(x, self.W0, self.bias)

        # LoRA-SB term: s * ((x @ A^T) @ R^T) @ B^T
        # F.linear(input, weight) = input @ weight^T
        # A has shape (r, n): F.linear(x, A) = x @ A^T = (..., n) @ (n, r) = (..., r)
        lora_intermediate = F.linear(x, self.A)  # (..., r)

        # R has shape (r, r): F.linear(lora_intermediate, R) = (...) @ R^T = (..., r) @ (r, r) = (..., r)
        lora_intermediate = F.linear(lora_intermediate, self.R)  # (..., r)

        # B has shape (m, r): F.linear(lora_intermediate, B) = (...) @ B^T = (..., r) @ (r, m) = (..., m)
        lora_out = F.linear(lora_intermediate, self.B)  # (..., m)

        result = result + self.scaling * lora_out
        return result

    @property
    def delta_w(self) -> torch.Tensor:
        """Return the learned update matrix: s * B @ R @ A"""
        return self.scaling * (self.B @ self.R @ self.A)

    def merge(self) -> nn.Linear:
        """
        Merge LoRA-SB weights into the original weight and return a plain nn.Linear.

        W_merged = W0 + s * B @ R @ A
        """
        merged_weight = self.W0 + self.delta_w
        merged_layer = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
        )
        merged_layer.weight.data.copy_(merged_weight)
        if self.bias is not None:
            merged_layer.bias.data.copy_(self.bias)
        return merged_layer

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, scaling={self.scaling}"
        )


def apply_lora_sb(
    model: nn.Module,
    rank: int,
    scaling: float = 1.0,
    target_modules: Optional[list] = None,
) -> nn.Module:
    """
    Apply LoRA-SB to a model by replacing target linear layers.

    Args:
        model: The pre-trained model to adapt.
        rank: Rank of the low-rank decomposition.
        scaling: Scaling factor (default: 1.0).
        target_modules: List of module name patterns to target.
                       If None, targets all nn.Linear layers.

    Returns:
        The model with LoRA-SB layers applied (modified in-place).
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "out_proj",
                         "fc1", "fc2", "dense", "query", "key", "value"]

    def _is_target(name: str) -> bool:
        return any(t in name for t in target_modules)

    def _replace_module(parent: nn.Module, child_name: str, child: nn.Module):
        if isinstance(child, nn.Linear) and _is_target(child_name):
            lora_sb_layer = LoRA_SB_Layer.from_pretrained_linear(
                child, rank=rank, scaling=scaling
            )
            setattr(parent, child_name, lora_sb_layer)

    for name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            _replace_module(module, child_name, child)

    return model
