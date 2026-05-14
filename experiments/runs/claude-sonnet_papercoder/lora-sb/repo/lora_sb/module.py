## lora_sb/module.py
"""Core LoRA-SB linear layer module.

Implements LoRASBLinear, the fundamental building block of LoRA-SB. This
module replaces a standard nn.Linear layer with the parameterization:

    W = W_0 + s * B @ R @ A

where:
    - W_0 ∈ R^{m×n}: frozen pre-trained weight
    - B ∈ R^{m×r}: frozen after initialization (orthonormal columns, B^T B = I)
    - A ∈ R^{r×n}: frozen after initialization (orthonormal rows, A A^T = I)
    - R ∈ R^{r×r}: the ONLY trainable matrix
    - s = 1.0: scaling factor (Theorem 5: scaling-factor independent when
      B^T B = A A^T = I)

The orthonormality of B and A is enforced by LoRASBInitializer (via truncated
SVD of ΔW_avg) and stored here. This guarantees:
    1. Optimal gradient approximation: g^R = g^R_{LoRA-XS} (Theorem 3 simplification)
    2. Scaling-factor independence (Theorem 5)
    3. Guaranteed loss reduction ΔL ≤ 0 (Theorem 4)

Parameter count per layer: r² (vs r(m+n) for LoRA), enabling 27–90x fewer
trainable parameters than standard LoRA at comparable performance.

References:
    Paper Section 2.1: LoRA-XS architecture W = W_0 + s*B*R*A
    Paper Section 2.6: LoRA-SB initialization and properties
    Paper Figure 1: Architecture diagram
    config.yaml: lora_sb.scaling: 1.0
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LoRASBLinear(nn.Module):
    """LoRA-SB linear layer: W = W_0 + s * B @ R @ A.

    Replaces a standard nn.Linear layer with the LoRA-SB parameterization.
    Only lora_R is trainable; lora_B and lora_A are frozen after initialization
    by LoRASBInitializer. The base weight W_0 is also frozen.

    This module is constructed with zero-initialized B, A, R. The actual
    orthonormal initialization values (from truncated SVD of ΔW_avg) are
    written into lora_B.data, lora_A.data, and lora_R.data by
    ModelBuilder.build_lora_sb() after LoRASBInitializer.initialize() runs.

    Attributes:
        in_features: Input feature dimension n.
        out_features: Output feature dimension m.
        rank: LoRA rank r. Trainable parameter count = r².
        scaling: Scaling factor s. Always 1.0 for LoRA-SB (Theorem 5).
        weight: Frozen pre-trained weight W_0, shape (out_features, in_features).
        bias: Frozen pre-trained bias, shape (out_features,), or None.
        lora_B: Frozen low-rank matrix B, shape (out_features, rank).
            After initialization: orthonormal columns, B^T B = I_r.
        lora_A: Frozen low-rank matrix A, shape (rank, in_features).
            After initialization: orthonormal rows, A A^T = I_r.
        lora_R: Trainable low-rank matrix R, shape (rank, rank).
            After initialization: R = diag(S) where S are singular values of ΔW_avg.
        _merged: Internal flag tracking whether merge_weights() has been called.

    Example:
        >>> layer = LoRASBLinear(in_features=4096, out_features=4096, rank=32)
        >>> x = torch.randn(2, 512, 4096)
        >>> out = layer(x)  # shape: (2, 512, 4096)
        >>> layer.merge_weights()   # fold delta into weight for inference
        >>> out_merged = layer(x)   # same result, no extra matmuls
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        scaling: float = 1.0,
        bias: bool = True,
    ) -> None:
        """Initialize LoRASBLinear with zero-filled LoRA matrices.

        All LoRA matrices (lora_B, lora_A, lora_R) are initialized to zeros
        here. ModelBuilder.build_lora_sb() overwrites them with the SVD-derived
        orthonormal initialization before training begins.

        The base weight and bias are also initialized to zeros; ModelBuilder
        copies the actual pre-trained values into weight.data and bias.data
        when replacing the original nn.Linear layer.

        Args:
            in_features: Size of each input sample (n in the paper notation).
            out_features: Size of each output sample (m in the paper notation).
            rank: LoRA rank r. Must be positive and ≤ min(in_features, out_features).
                Trainable parameter count for this layer = rank².
                Paper uses r ∈ {32, 64, 96} for LLMs and r ∈ {8, 16, 24} for
                RoBERTa (config.yaml: defaults.rank).
            scaling: Scaling factor s in W = W_0 + s*B@R@A. For LoRA-SB this
                is always 1.0 per Theorem 5 (scaling-factor independence when
                B^T B = A A^T = I). Sourced from config.yaml: lora_sb.scaling.
                Defaults to 1.0.
            bias: If True, a frozen bias parameter is created. If False,
                self.bias is None. Should match the bias setting of the
                original nn.Linear layer being replaced. Defaults to True.

        Raises:
            ValueError: If rank <= 0 or rank > min(in_features, out_features).
        """
        super().__init__()

        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if rank > min(in_features, out_features):
            raise ValueError(
                f"rank ({rank}) must be ≤ min(in_features, out_features) "
                f"= min({in_features}, {out_features}) = {min(in_features, out_features)}"
            )

        # -----------------------------------------------------------------------
        # Store dimensions and hyperparameters
        # -----------------------------------------------------------------------
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.rank: int = rank
        # scaling is stored as a plain float (not nn.Parameter) since it is
        # a fixed hyperparameter, not a learnable value. For LoRA-SB, s=1.0
        # always (Theorem 5). Stored as float for use in forward() arithmetic.
        self.scaling: float = scaling

        # -----------------------------------------------------------------------
        # Frozen base weight W_0: shape (out_features, in_features)
        # Follows PyTorch nn.Linear convention: weight shape is (out, in).
        # requires_grad=False: never updated during training.
        # Initialized to zeros; ModelBuilder copies pre-trained values here.
        # Using nn.Parameter (not register_buffer) so it appears in state_dict()
        # under a consistent name for checkpoint save/load.
        # -----------------------------------------------------------------------
        self.weight: nn.Parameter = nn.Parameter(
            torch.zeros(out_features, in_features),
            requires_grad=False,
        )

        # -----------------------------------------------------------------------
        # Optional frozen bias: shape (out_features,)
        # Bias is part of the pre-trained model and is not adapted by LoRA-SB.
        # requires_grad=False: never updated during training.
        # -----------------------------------------------------------------------
        if bias:
            self.bias: nn.Parameter | None = nn.Parameter(
                torch.zeros(out_features),
                requires_grad=False,
            )
        else:
            # Register as None so F.linear(x, w, None) works correctly.
            self.bias = None

        # -----------------------------------------------------------------------
        # Frozen low-rank matrix B: shape (out_features, rank) = (m, r)
        # After LoRASBInitializer: B = U[:, :r] from SVD of ΔW_avg.
        # Orthonormal columns: B^T B = I_r (guaranteed by SVD).
        # requires_grad=False: frozen throughout training (key LoRA-SB property).
        # Initialized to zeros; ModelBuilder writes orthonormal values here.
        # -----------------------------------------------------------------------
        self.lora_B: nn.Parameter = nn.Parameter(
            torch.zeros(out_features, rank),
            requires_grad=False,
        )

        # -----------------------------------------------------------------------
        # Frozen low-rank matrix A: shape (rank, in_features) = (r, n)
        # After LoRASBInitializer: A = Vt[:r, :] from SVD of ΔW_avg.
        # Orthonormal rows: A A^T = I_r (guaranteed by SVD).
        # requires_grad=False: frozen throughout training (key LoRA-SB property).
        # Initialized to zeros; ModelBuilder writes orthonormal values here.
        # -----------------------------------------------------------------------
        self.lora_A: nn.Parameter = nn.Parameter(
            torch.zeros(rank, in_features),
            requires_grad=False,
        )

        # -----------------------------------------------------------------------
        # Trainable low-rank matrix R: shape (rank, rank) = (r, r)
        # After LoRASBInitializer: R = diag(S[:r]) where S are singular values
        # of ΔW_avg. With s=1.0, R_init = diag(S) directly.
        # requires_grad=True: THE ONLY TRAINABLE PARAMETER in LoRA-SB.
        # Initialized to zeros; ModelBuilder writes diag(S) values here.
        # The AdamW optimizer in Trainer._setup_optimizer() targets only this.
        # -----------------------------------------------------------------------
        self.lora_R: nn.Parameter = nn.Parameter(
            torch.zeros(rank, rank),
            requires_grad=True,
        )

        # -----------------------------------------------------------------------
        # Internal state flag for weight merging
        # Tracks whether merge_weights() has been called to prevent double-adding
        # the delta in forward() and to make unmerge_weights() safe.
        # -----------------------------------------------------------------------
        self._merged: bool = False

    def forward(self, x: Tensor) -> Tensor:
        """Compute the LoRA-SB forward pass: output = x @ (W_0 + s*B@R@A)^T + bias.

        When weights are not merged (training mode), computes the full
        W = W_0 + s*B@R@A on every forward pass. When merged (inference mode
        after merge_weights()), uses the pre-computed effective weight directly.

        The computation order B @ R @ A is chosen to minimize FLOPs:
            - B @ R: (m, r) @ (r, r) = (m, r), cost O(m*r²)
            - (B@R) @ A: (m, r) @ (r, n) = (m, n), cost O(m*r*n)
        Total: O(m*r*(r+n)) vs O(m*n) for full weight. Since r << min(m,n),
        this is much cheaper than storing a full (m,n) delta.

        Args:
            x: Input tensor of shape (..., in_features). Supports arbitrary
                batch dimensions (e.g., (batch, seq_len, in_features) for
                transformer hidden states).

        Returns:
            Output tensor of shape (..., out_features).
        """
        if self._merged:
            # Weight already contains W_0 + s*B@R@A from merge_weights().
            # Direct linear transform with no extra computation.
            return F.linear(x, self.weight, self.bias)

        # Compute low-rank update: s * B @ R @ A
        # Shape trace: (m, r) @ (r, r) @ (r, n) = (m, n)
        # With scaling=1.0 (LoRA-SB default), this is just B @ R @ A.
        delta_w: Tensor = self.get_delta_w()

        # Effective weight: W_0 + s*B@R@A, shape (out_features, in_features)
        # Using + (not +=) to avoid in-place modification of self.weight.data,
        # which would corrupt the frozen pre-trained weight.
        effective_weight: Tensor = self.weight + delta_w

        # Apply linear transformation: x @ effective_weight^T + bias
        # F.linear handles arbitrary batch dimensions correctly.
        return F.linear(x, effective_weight, self.bias)

    def get_delta_w(self) -> Tensor:
        """Compute the low-rank weight update s * B @ R @ A.

        Returns the current low-rank update matrix that is added to the
        frozen pre-trained weight W_0. This is the core LoRA-SB update:

            ΔW = s * B @ R @ A

        where B and A are orthonormal (B^T B = A A^T = I) and R is the
        trainable matrix. With s=1.0 (LoRA-SB default), this simplifies to
        B @ R @ A.

        This method is used by:
        - forward(): to compute the effective weight during training
        - merge_weights(): to fold the delta into self.weight
        - unmerge_weights(): to subtract the delta from self.weight

        Returns:
            Delta weight tensor of shape (out_features, in_features),
            i.e., (m, n) in paper notation. Gradients flow through lora_R
            only (lora_B and lora_A have requires_grad=False).
        """
        # B @ R: (out_features, rank) @ (rank, rank) = (out_features, rank)
        br: Tensor = self.lora_B @ self.lora_R
        # (B @ R) @ A: (out_features, rank) @ (rank, in_features) = (out_features, in_features)
        bra: Tensor = br @ self.lora_A
        # Apply scaling factor s (1.0 for LoRA-SB, may differ for ablations)
        return self.scaling * bra

    def merge_weights(self) -> None:
        """Fold the low-rank update into the base weight for inference efficiency.

        After merging, the forward pass becomes a single F.linear call with
        no extra matrix multiplications, matching the inference cost of the
        original pre-trained model. This is the recommended mode for deployment.

        The merge operation is:
            self.weight.data += s * B @ R @ A

        After merging, forward() uses self.weight directly (which now contains
        W_0 + s*B@R@A) without recomputing the delta.

        This method is idempotent: calling it multiple times has no additional
        effect after the first call (guarded by self._merged flag).

        Note:
            merge_weights() should only be called after training is complete.
            If called during training and then unmerge_weights() is called,
            correctness is only guaranteed if lora_R has not changed between
            the merge and unmerge calls.

        Note:
            Uses .data assignment to bypass autograd tracking, since we are
            modifying the frozen weight tensor in-place.
        """
        if self._merged:
            # Already merged; no-op to prevent double-adding the delta.
            return

        # Compute delta with no_grad since we're doing an in-place data update.
        with torch.no_grad():
            delta_w: Tensor = self.get_delta_w()
            # In-place addition to self.weight.data (bypasses autograd).
            self.weight.data += delta_w

        self._merged = True

    def unmerge_weights(self) -> None:
        """Reverse merge_weights() to restore the separate W_0 and B@R@A representation.

        Subtracts the current low-rank update from self.weight.data, restoring
        the original pre-trained weight W_0. After unmerging, forward() will
        again compute the delta on every pass.

        This is useful for:
        - Resuming training after a temporary merge for evaluation
        - Ablation studies comparing merged vs unmerged inference
        - Saving the model in the unmerged format for flexibility

        This method is idempotent: calling it when not merged has no effect.

        Warning:
            Correctness requires that lora_R has not changed since merge_weights()
            was called. If R was updated between merge and unmerge, the subtracted
            delta will differ from what was added, corrupting self.weight.
        """
        if not self._merged:
            # Not merged; no-op.
            return

        with torch.no_grad():
            delta_w: Tensor = self.get_delta_w()
            # In-place subtraction to restore W_0.
            self.weight.data -= delta_w

        self._merged = False

    def extra_repr(self) -> str:
        """Return a human-readable string for print(model) output.

        Provides key architectural information for each LoRASBLinear layer,
        including the trainable parameter count (rank²) which is the primary
        efficiency metric reported in Tables 1–3 of the paper.

        Returns:
            A string summarizing the layer's configuration, e.g.:
            "in_features=4096, out_features=4096, rank=32, scaling=1.0,
             bias=True, trainable_params=1024, merged=False"
        """
        trainable_params: int = self.rank * self.rank
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"rank={self.rank}, "
            f"scaling={self.scaling}, "
            f"bias={self.bias is not None}, "
            f"trainable_params={trainable_params}, "
            f"merged={self._merged}"
        )
