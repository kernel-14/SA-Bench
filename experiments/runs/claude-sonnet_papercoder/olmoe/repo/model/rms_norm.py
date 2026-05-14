## model/rms_norm.py
"""Root Mean Square Layer Normalization (RMSNorm) for OLMoE.

Implements RMSNorm as described in:
  Zhang & Sennrich (2019). "Root Mean Square Layer Normalization."
  https://arxiv.org/abs/1910.07467

Used in three places within the OLMoE architecture:
  1. Pre-attention norm (attn_norm) in each OLMoEBlock
  2. Pre-MoE norm (ffn_norm) in each OLMoEBlock
  3. QK-Norm on query and key projections in OLMoEAttention (dim=head_dim=128)
  4. Final norm before the LM head in OLMoEModel

Key design decision from the paper (Section 4.2.3):
  The learnable weight parameter IS subject to weight decay (weight_decay=0.1),
  unlike standard practice which excludes normalization parameters from decay.
  This is enforced in training/optimizer.py by not creating a no-decay group.

Configuration values (from config.yaml):
  model.rms_norm_eps: 1.0e-05
  model.hidden_dim: 2048  (for block norms and final norm)
  model.num_heads: 16     (head_dim = 2048 // 16 = 128, for QK-Norm)
"""

import torch
import torch.nn as nn
from torch import Tensor


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Normalizes the input tensor along its last dimension using the RMS
    (Root Mean Square) statistic, then scales by a learnable weight.

    Unlike LayerNorm, RMSNorm:
      - Does NOT subtract the mean (no centering)
      - Does NOT have a bias parameter (only a learnable scale)
      - Is faster while achieving similar training stability benefits

    The paper (Section 4.2.3) reports that RMSNorm reduces gradient norm
    spikes compared to the non-parametric LayerNorm used in OLMo, at the
    cost of ~15% throughput reduction.

    Mathematical formulation:
        RMSNorm(x) = x / RMS(x) * weight
        where RMS(x) = sqrt(mean(x², dim=-1, keepdim=True) + eps)

    Attributes:
        normalized_shape: The size of the last dimension being normalized.
        eps: Small constant for numerical stability (default: 1e-5 from config).
        weight: Learnable scale parameter of shape (normalized_shape,),
                initialized to ones. Subject to weight decay (non-standard,
                per paper Section 4.2.3).

    Example usage:
        # Block pre-norm: dim = hidden_dim = 2048
        attn_norm = RMSNorm(dim=2048, eps=1e-5)
        x = torch.randn(2, 4096, 2048)  # (batch, seq, hidden)
        x_normed = attn_norm(x)         # (2, 4096, 2048)

        # QK-Norm: dim = head_dim = hidden_dim // num_heads = 128
        q_norm = RMSNorm(dim=128, eps=1e-5)
        q = torch.randn(2, 16, 4096, 128)  # (batch, heads, seq, head_dim)
        q_normed = q_norm(q)               # (2, 16, 4096, 128)
    """

    def __init__(self, dim: int, eps: float = 1.0e-05) -> None:
        """Initialize RMSNorm.

        Args:
            dim: Size of the last dimension to normalize over. Varies by usage:
                - Block norms (attn_norm, ffn_norm): dim = hidden_dim = 2048
                  (from config.yaml: model.hidden_dim)
                - QK-Norm (q_norm, k_norm): dim = head_dim = 128
                  (from config.yaml: hidden_dim // num_heads = 2048 // 16)
                - Final model norm: dim = hidden_dim = 2048
            eps: Small constant added to the denominator for numerical stability.
                Default matches config.yaml: model.rms_norm_eps = 1.0e-05
                and Table 10 of the paper.
        """
        super().__init__()

        if dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim}.")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}.")

        self.normalized_shape: int = dim
        self.eps: float = eps

        # Learnable scale parameter, initialized to ones (identity at init).
        # NOTE: This parameter IS included in weight decay (weight_decay=0.1)
        # per paper Section 4.2.3, unlike standard practice. The optimizer
        # in training/optimizer.py must NOT exclude this from weight decay.
        self.weight: nn.Parameter = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Apply RMSNorm to the input tensor.

        Normalizes along the last dimension using the RMS statistic, then
        scales by the learnable weight. Computation is performed in float32
        for numerical stability, then cast back to the input dtype.

        Args:
            x: Input tensor of shape (..., dim) where dim matches
               self.normalized_shape. Works for any number of leading
               dimensions (e.g., (B, T, D) or (B, H, T, D)).

        Returns:
            Normalized and scaled tensor of the same shape and dtype as x.

        Shape examples:
            - Block norm: (batch, seq_len, hidden_dim) -> same shape
            - QK-Norm: (batch, num_heads, seq_len, head_dim) -> same shape
        """
        # Store original dtype to restore after float32 computation.
        # This is critical for BF16 training stability: BF16 has limited
        # precision that can cause NaN/Inf in the variance computation.
        input_dtype: torch.dtype = x.dtype

        # Cast to float32 for numerically stable variance computation.
        # This follows the standard practice in LLaMA, Gemma, and other
        # production RMSNorm implementations.
        x_float: Tensor = x.float()

        # Compute the mean of squares along the last dimension.
        # Shape: (..., 1) with keepdim=True for broadcasting.
        variance: Tensor = x_float.pow(2).mean(dim=-1, keepdim=True)

        # Normalize using reciprocal square root (rsqrt is more efficient
        # than 1 / sqrt and numerically equivalent).
        # Shape: same as x_float (..., dim)
        x_normalized: Tensor = x_float * torch.rsqrt(variance + self.eps)

        # Apply learnable scale weight.
        # self.weight has shape (dim,) and broadcasts over leading dimensions.
        # Cast weight to float32 for the multiplication, then cast result
        # back to the original input dtype.
        output: Tensor = (x_normalized * self.weight.float()).to(input_dtype)

        return output

    def extra_repr(self) -> str:
        """Return extra representation string for printing the module.

        Returns:
            String showing normalized_shape and eps values.
        """
        return f"normalized_shape={self.normalized_shape}, eps={self.eps}"
