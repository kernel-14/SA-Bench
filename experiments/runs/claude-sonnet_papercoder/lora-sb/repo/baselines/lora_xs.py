## baselines/lora_xs.py
"""LoRA-XS baseline linear layer module.

Implements LoRAXSLinear, the baseline counterpart to LoRASBLinear. This module
implements the LoRA-XS architecture from Bałazy et al. (2024) with the same
W = W₀ + s·B·R·A parameterization but with PiSSA-inspired initialization on
the pre-trained weight matrix W₀ rather than on the first fine-tuning step
approximation ΔW_avg.

The key distinction from LoRA-SB:
    - LoRA-XS: B, A initialized from SVD(W₀) — pre-trained weight subspace
    - LoRA-SB: B, A initialized from SVD(ΔW_avg) — task-relevant subspace

Both methods share the same architecture (r² trainable parameters per layer),
but LoRA-XS's initialization "fails to capture the specific subspaces relevant
to the FT task" (paper Section 2.2), which is the core motivation for LoRA-SB.

Scaling factor:
    s = alpha / rank
    - LLM experiments: alpha = rank → s = 1.0 (config.yaml: lora_xs.alpha_equals_rank: true)
    - RoBERTa experiments: alpha = 16 (fixed, config.yaml: roberta_glue.lora_xs.alpha: 16)

This scaling requires manual tuning, unlike LoRA-SB where s=1.0 is provably
optimal (Theorem 5).

References:
    Paper Section 2.1: LoRA-XS architecture W = W₀ + s*B*R*A
    Paper Section 2.2: Limitations of LoRA-XS (motivation for LoRA-SB)
    Paper Tables 1-3: LoRA-XS baseline results
    config.yaml: baselines.lora_xs, lora_xs.alpha_equals_rank, svd_niter: 4
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class LoRAXSLinear(nn.Module):
    """LoRA-XS linear layer: W = W₀ + s * B @ R @ A.

    Implements the LoRA-XS parameterization with PiSSA-style initialization
    on the pre-trained weight matrix W₀. Only lora_R is trainable; lora_B
    and lora_A are frozen after initialization via truncated SVD of W₀.

    The base weight W₀ is stored as a non-trainable buffer. The scaling
    factor s = alpha / rank requires tuning (unlike LoRA-SB where s=1.0
    is provably optimal).

    Attributes:
        in_features: Input feature dimension n.
        out_features: Output feature dimension m.
        rank: LoRA rank r. Trainable parameter count = r².
        alpha: Alpha hyperparameter for scaling. For LLMs: alpha=rank → s=1.0.
            For RoBERTa: alpha=16 (fixed). Sourced from config.yaml.
        scaling: Computed scaling factor s = alpha / rank. Applied to the
            low-rank update path only.
        lora_B: Frozen low-rank matrix B, shape (out_features, rank).
            After initialization: left singular vectors of W₀ (orthonormal columns).
        lora_A: Frozen low-rank matrix A, shape (rank, in_features).
            After initialization: right singular vectors of W₀ (orthonormal rows).
        lora_R: Trainable low-rank matrix R, shape (rank, rank).
            Initialized to zeros. THE ONLY TRAINABLE PARAMETER.
        _svd_niter: Number of power iterations for torch.svd_lowrank.

    Example:
        >>> # Create from an existing nn.Linear layer
        >>> original_linear = nn.Linear(4096, 4096)
        >>> layer = LoRAXSLinear(
        ...     in_features=4096,
        ...     out_features=4096,
        ...     rank=32,
        ...     alpha=32.0,  # alpha=rank for LLM experiments
        ...     bias=True,
        ... )
        >>> # Copy pre-trained weight and initialize B, A via SVD of W₀
        >>> layer.weight.data.copy_(original_linear.weight.data)
        >>> layer._init_pissa_style(original_linear.weight.data)
        >>> x = torch.randn(2, 512, 4096)
        >>> out = layer(x)  # shape: (2, 512, 4096)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        bias: bool = True,
        svd_niter: int = 4,
    ) -> None:
        """Initialize LoRAXSLinear with PiSSA-style initialization on W₀.

        Creates the module structure with zero-initialized LoRA matrices.
        The actual SVD-based initialization of lora_B and lora_A is performed
        by _init_pissa_style(), which must be called after copying the
        pre-trained weight into self.weight.

        Args:
            in_features: Size of each input sample (n in paper notation).
                Corresponds to nn.Linear.in_features of the replaced layer.
            out_features: Size of each output sample (m in paper notation).
                Corresponds to nn.Linear.out_features of the replaced layer.
            rank: LoRA rank r. Must be positive and < min(in_features, out_features).
                Trainable parameter count for this layer = rank².
                Paper uses r ∈ {32, 64, 96} for LLMs and r ∈ {8, 16, 24}
                for RoBERTa (config.yaml: defaults.rank).
            alpha: Alpha hyperparameter for scaling. The scaling factor is
                computed as s = alpha / rank. For LLM experiments where
                alpha_equals_rank=True: alpha=rank → s=1.0. For RoBERTa:
                alpha=16 (fixed, config.yaml: roberta_glue.lora_xs.alpha: 16).
                This is the key difference from LoRA-SB where s=1.0 always.
            bias: If True, a frozen bias parameter is created matching the
                original nn.Linear. If False, self.bias is None. Defaults to True.
            svd_niter: Number of power iterations for torch.svd_lowrank.
                Controls accuracy-speed tradeoff of randomized SVD.
                Sourced from config.yaml: lora_sb.svd_niter: 4. Defaults to 4.

        Raises:
            ValueError: If rank <= 0, alpha <= 0, or rank >= min(in_features, out_features).
        """
        super().__init__()

        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if alpha <= 0.0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if rank >= min(in_features, out_features):
            raise ValueError(
                f"rank ({rank}) must be < min(in_features, out_features) "
                f"= min({in_features}, {out_features}) = {min(in_features, out_features)}"
            )
        if svd_niter <= 0:
            raise ValueError(f"svd_niter must be positive, got {svd_niter}")

        # -----------------------------------------------------------------------
        # Store dimensions and hyperparameters
        # -----------------------------------------------------------------------
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.rank: int = rank
        self.alpha: float = alpha
        # Scaling factor s = alpha / rank.
        # For LLM experiments (alpha=rank): s=1.0.
        # For RoBERTa (alpha=16, rank=8): s=2.0.
        # This requires manual tuning, unlike LoRA-SB where s=1.0 is provably optimal.
        self.scaling: float = alpha / rank
        self._svd_niter: int = svd_niter

        # -----------------------------------------------------------------------
        # Frozen base weight W₀: shape (out_features, in_features)
        # Stored as a non-trainable parameter (requires_grad=False).
        # Follows PyTorch nn.Linear convention: weight shape is (out, in).
        # Initialized to zeros; ModelBuilder copies pre-trained values here
        # before calling _init_pissa_style().
        # -----------------------------------------------------------------------
        self.weight: nn.Parameter = nn.Parameter(
            torch.zeros(out_features, in_features),
            requires_grad=False,
        )

        # -----------------------------------------------------------------------
        # Optional frozen bias: shape (out_features,)
        # Bias is part of the pre-trained model and is not adapted by LoRA-XS.
        # requires_grad=False: never updated during training.
        # -----------------------------------------------------------------------
        if bias:
            self.bias: Optional[nn.Parameter] = nn.Parameter(
                torch.zeros(out_features),
                requires_grad=False,
            )
        else:
            self.bias = None

        # -----------------------------------------------------------------------
        # Frozen low-rank matrix B: shape (out_features, rank) = (m, r)
        # After _init_pissa_style: B = U[:, :r] from SVD of W₀.
        # Orthonormal columns: B^T B = I_r (from SVD properties).
        # requires_grad=False: frozen throughout training.
        # Initialized to zeros; _init_pissa_style overwrites with SVD result.
        # -----------------------------------------------------------------------
        self.lora_B: nn.Parameter = nn.Parameter(
            torch.zeros(out_features, rank),
            requires_grad=False,
        )

        # -----------------------------------------------------------------------
        # Frozen low-rank matrix A: shape (rank, in_features) = (r, n)
        # After _init_pissa_style: A = Vt[:r, :] from SVD of W₀.
        # Orthonormal rows: A A^T = I_r (from SVD properties).
        # requires_grad=False: frozen throughout training.
        # Initialized to zeros; _init_pissa_style overwrites with SVD result.
        # -----------------------------------------------------------------------
        self.lora_A: nn.Parameter = nn.Parameter(
            torch.zeros(rank, in_features),
            requires_grad=False,
        )

        # -----------------------------------------------------------------------
        # Trainable low-rank matrix R: shape (rank, rank) = (r, r)
        # Initialized to zeros: ensures model starts from W₀ (no adaptation).
        # With R=0: s*B@0@A = 0, so initial output equals W₀·x.
        # requires_grad=True: THE ONLY TRAINABLE PARAMETER in LoRA-XS.
        # The AdamW optimizer in Trainer._setup_optimizer() targets only this.
        # -----------------------------------------------------------------------
        self.lora_R: nn.Parameter = nn.Parameter(
            torch.zeros(rank, rank),
            requires_grad=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Compute the LoRA-XS forward pass.

        Computes: output = F.linear(x, W₀, bias) + s * x @ A^T @ R^T @ B^T

        The low-rank path is computed as a sequence of small matrix multiplications
        through the rank bottleneck, avoiding materialization of the full (m×n)
        delta weight matrix. This is more memory-efficient than computing
        W₀ + s*B@R@A as a single (m×n) matrix.

        Computation order for the low-rank path:
            x: (..., n)
            x @ A^T: (..., r)      — project to rank-r subspace
            (...) @ R^T: (..., r)  — apply trainable transformation
            (...) @ B^T: (..., m)  — project back to output space

        Args:
            x: Input tensor of shape (..., in_features). Supports arbitrary
                batch dimensions (e.g., (batch, seq_len, in_features) for
                transformer hidden states).

        Returns:
            Output tensor of shape (..., out_features).
        """
        # Base weight path: W₀ · x^T + bias
        # F.linear handles arbitrary batch dimensions correctly.
        base_output: Tensor = F.linear(x, self.weight, self.bias)

        # Low-rank path: s * x @ A^T @ R^T @ B^T
        # Computed through the rank bottleneck to avoid (m×n) intermediate.
        # Step 1: x @ A^T — shape (..., rank)
        # lora_A has shape (rank, in_features), so A^T has shape (in_features, rank)
        xa: Tensor = x @ self.lora_A.T

        # Step 2: (x @ A^T) @ R^T — shape (..., rank)
        # lora_R has shape (rank, rank), so R^T has shape (rank, rank)
        xar: Tensor = xa @ self.lora_R.T

        # Step 3: (x @ A^T @ R^T) @ B^T — shape (..., out_features)
        # lora_B has shape (out_features, rank), so B^T has shape (rank, out_features)
        lora_output: Tensor = xar @ self.lora_B.T

        # Apply scaling factor s = alpha / rank
        return base_output + self.scaling * lora_output

    def _init_pissa_style(self, weight: Tensor) -> None:
        """Initialize B and A via truncated SVD of the pre-trained weight W₀.

        Implements the PiSSA-inspired initialization used by LoRA-XS (ref 2,
        inspired by PiSSA ref 30). Computes the top-r singular vectors of W₀
        and uses them to define the frozen low-rank subspace.

        Initialization:
            U, S, V = truncated_SVD(W₀, rank=r)
            B_init = U[:, :r]      # shape (m, r) — left singular vectors
            A_init = Vt[:r, :]     # shape (r, n) — right singular vectors (transposed)
            R_init = zeros(r, r)   # trainable matrix starts at zero

        The singular values S are discarded (unlike LoRA-SB where they are
        absorbed into R_init = diag(S)). R starts at zero, so the model
        begins from W₀ and learns the adaptation magnitude from scratch.

        This initialization captures the most significant subspaces of the
        pre-trained weight, but as the paper notes (Section 2.2), "this
        initialization is not aligned well with FT because it fails to capture
        the specific subspaces relevant to the FT task."

        Args:
            weight: Pre-trained weight tensor W₀, shape (out_features, in_features).
                May be in bfloat16 (per config.yaml: hardware.precision: bfloat16).
                SVD is computed in float32 for numerical stability, with results
                cast back to the original dtype.

        Note:
            Uses torch.svd_lowrank for memory efficiency on large matrices
            (e.g., 4096×4096 in 7B models). Consistent with config.yaml:
            lora_sb.svd_niter: 4 and the paper's Appendix F.

        Note:
            After this call, lora_B.requires_grad and lora_A.requires_grad
            are explicitly set to False to enforce the LoRA-XS constraint.
            Only lora_R.requires_grad remains True.
        """
        if weight.dim() != 2:
            raise ValueError(
                f"weight must be a 2D matrix, got shape {weight.shape}"
            )

        m, n = weight.shape
        original_dtype: torch.dtype = weight.dtype
        original_device: torch.device = weight.device

        # -----------------------------------------------------------------------
        # Clamp rank to valid range for svd_lowrank.
        # rank must satisfy 0 < rank < min(m, n).
        # -----------------------------------------------------------------------
        effective_rank: int = min(self.rank, min(m, n) - 1)
        if effective_rank <= 0:
            effective_rank = 1
        if effective_rank < self.rank:
            logger.warning(
                "Requested rank %d exceeds matrix dimensions (%d, %d) for W₀. "
                "Clamping to effective_rank=%d.",
                self.rank, m, n, effective_rank,
            )

        # -----------------------------------------------------------------------
        # Cast to float32 for numerical stability.
        # bfloat16 has only 7 mantissa bits, which can cause significant
        # numerical errors in the randomized SVD power iterations.
        # -----------------------------------------------------------------------
        weight_f32: Tensor = weight.to(dtype=torch.float32)

        # -----------------------------------------------------------------------
        # Check for degenerate weight matrix.
        # -----------------------------------------------------------------------
        weight_norm: float = weight_f32.norm().item()
        if weight_norm < 1e-8:
            logger.warning(
                "Pre-trained weight W₀ has near-zero norm (%.2e) for shape (%d, %d). "
                "SVD-based initialization may produce poor results.",
                weight_norm, m, n,
            )

        # -----------------------------------------------------------------------
        # Compute truncated SVD via torch.svd_lowrank.
        # Returns U (m × q), S (q,), V (n × q) — note V not Vt.
        # niter=self._svd_niter from config.yaml: lora_sb.svd_niter: 4.
        # -----------------------------------------------------------------------
        try:
            U: Tensor
            S: Tensor
            V: Tensor
            U, S, V = torch.svd_lowrank(
                weight_f32,
                q=effective_rank,
                niter=self._svd_niter,
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"torch.svd_lowrank failed for W₀ of shape ({m}, {n}) "
                f"with rank={effective_rank}, niter={self._svd_niter}. "
                f"Original error: {e}"
            ) from e

        # Validate output shapes
        assert U.shape == (m, effective_rank), (
            f"Expected U shape ({m}, {effective_rank}), got {U.shape}"
        )
        assert S.shape == (effective_rank,), (
            f"Expected S shape ({effective_rank},), got {S.shape}"
        )
        assert V.shape == (n, effective_rank), (
            f"Expected V shape ({n}, {effective_rank}), got {V.shape}"
        )

        # -----------------------------------------------------------------------
        # Set B_init = U[:, :r] — left singular vectors of W₀.
        # Shape: (out_features, rank) = (m, r).
        # Orthonormal columns: U^T U = I_r by SVD construction.
        # -----------------------------------------------------------------------
        B_init: Tensor = U.to(dtype=original_dtype, device=original_device)
        self.lora_B.data.copy_(B_init)

        # -----------------------------------------------------------------------
        # Set A_init = V[:, :r].T = Vt[:r, :] — right singular vectors transposed.
        # V from torch.svd_lowrank has shape (n, rank).
        # Transposing gives Vt of shape (rank, n) = (rank, in_features).
        # Orthonormal rows: Vt @ Vt^T = I_r by SVD construction.
        # -----------------------------------------------------------------------
        A_init: Tensor = V.T.to(dtype=original_dtype, device=original_device)
        self.lora_A.data.copy_(A_init)

        # -----------------------------------------------------------------------
        # Set R_init = zeros(rank, rank).
        # R starts at zero so the model begins from W₀ (no initial adaptation).
        # Unlike LoRA-SB where R_init = diag(S), LoRA-XS discards singular values.
        # -----------------------------------------------------------------------
        self.lora_R.data.zero_()

        # -----------------------------------------------------------------------
        # Explicitly enforce frozen status of B and A.
        # This is critical: if B or A accidentally receive gradients, the
        # LoRA-XS constraint is violated and the comparison with LoRA-SB
        # becomes invalid.
        # -----------------------------------------------------------------------
        self.lora_B.requires_grad_(False)
        self.lora_A.requires_grad_(False)
        self.lora_R.requires_grad_(True)

        logger.debug(
            "LoRA-XS PiSSA-style initialization complete for W₀ shape (%d, %d): "
            "B_init %s, A_init %s, R_init zeros %s | "
            "top singular value: %.4f, bottom singular value: %.4f | "
            "scaling: %.4f (alpha=%.1f, rank=%d)",
            m, n,
            tuple(self.lora_B.shape),
            tuple(self.lora_A.shape),
            tuple(self.lora_R.shape),
            S[0].item(),
            S[-1].item(),
            self.scaling,
            self.alpha,
            self.rank,
        )

    def extra_repr(self) -> str:
        """Return a human-readable string for print(model) output.

        Provides key architectural information for each LoRAXSLinear layer,
        including the trainable parameter count (rank²) and scaling factor.

        Returns:
            A string summarizing the layer's configuration, e.g.:
            "in_features=4096, out_features=4096, rank=32, alpha=32.0,
             scaling=1.0, bias=True, trainable_params=1024"
        """
        trainable_params: int = self.rank * self.rank
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"rank={self.rank}, "
            f"alpha={self.alpha}, "
            f"scaling={self.scaling:.4f}, "
            f"bias={self.bias is not None}, "
            f"trainable_params={trainable_params}"
        )
