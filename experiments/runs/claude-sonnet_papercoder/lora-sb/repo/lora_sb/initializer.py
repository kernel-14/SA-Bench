## lora_sb/initializer.py
"""LoRA-SB initialization via truncated SVD of the gradient approximation.

This module implements LoRASBInitializer, which takes the averaged gradient
approximation ΔW_avg (computed by GradientEstimator) and produces the three
initialization matrices (B_init, R_init, A_init) that define the frozen
low-rank subspace for LoRA-SB training.

The mathematical foundation is the truncated SVD of ΔW_avg:

    ΔW_avg ≈ U[:, :r] @ diag(S[:r]) @ Vt[:r, :]

which simultaneously delivers:
    1. Optimal rank-r approximation (Eckart-Young theorem, Section 2.6)
    2. Orthonormality: B^T B = I_r and A A^T = I_r (from SVD properties)
    3. Scaling independence: s=1.0 is provably optimal (Theorem 5)

With these properties, the optimal gradient simplification from Theorem 3:
    g^R = (1/s²)(B^T B)^{-1} g^R_{LoRA-XS} (A A^T)^{-1}
reduces to:
    g^R = g^R_{LoRA-XS}
since (B^T B)^{-1} = (A A^T)^{-1} = I_r and s=1.0.

References:
    Paper Section 2.4: Initialization using update approximation
    Paper Section 2.5: Scaling factor independence (Theorem 5)
    Paper Section 2.6: LoRA-SB silver bullet (Theorems 3, 4, 5, 6)
    Paper Appendix F: torch.svd_lowrank usage
    config.yaml: lora_sb.scaling: 1.0, lora_sb.svd_niter: 4,
                 lora_sb.ortho_atol: 1.0e-5
"""

from __future__ import annotations

import logging
import warnings
from typing import Tuple

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class LoRASBInitializer:
    """Computes B_init, R_init, A_init from ΔW_avg via truncated SVD.

    This class is the mathematical core of LoRA-SB. For each weight matrix
    in the model, it takes the averaged gradient approximation ΔW_avg and
    produces three matrices that initialize the LoRASBLinear module:

        B_init = U[:, :r]          shape (m, r), orthonormal columns
        A_init = Vt[:r, :]         shape (r, n), orthonormal rows
        R_init = diag(S[:r])       shape (r, r), diagonal singular values

    These satisfy:
        - B_init @ R_init @ A_init ≈ ΔW_avg  (optimal rank-r approx)
        - B_init^T @ B_init = I_r             (orthonormal columns)
        - A_init @ A_init^T = I_r             (orthonormal rows)

    Attributes:
        rank: Target LoRA rank r. Sourced from config.rank.
        scaling: Scaling factor s. Always 1.0 for LoRA-SB (Theorem 5).
            Sourced from config.yaml lora_sb.scaling.
        device: Compute device for tensor operations and identity matrices.
        svd_niter: Number of power iterations for torch.svd_lowrank.
            Sourced from config.yaml lora_sb.svd_niter (default: 4).
        ortho_atol: Absolute tolerance for orthonormality verification.
            Sourced from config.yaml lora_sb.ortho_atol (default: 1e-5).

    Example:
        >>> initializer = LoRASBInitializer(rank=32, scaling=1.0, device=device)
        >>> delta_w = torch.randn(4096, 4096)  # ΔW_avg for one layer
        >>> B_init, R_init, A_init = initializer.initialize(delta_w)
        >>> B_init.shape  # (4096, 32)
        >>> R_init.shape  # (32, 32)
        >>> A_init.shape  # (32, 4096)
        >>> initializer.verify_orthonormality(B_init, A_init)  # True
    """

    def __init__(
        self,
        rank: int,
        scaling: float = 1.0,
        device: torch.device = torch.device("cpu"),
        svd_niter: int = 4,
        ortho_atol: float = 1e-5,
    ) -> None:
        """Initialize the LoRASBInitializer with rank and scaling configuration.

        No computation happens here. All SVD computation is deferred to
        initialize() calls, one per weight matrix.

        Args:
            rank: Target LoRA rank r. Must be positive. Sourced from
                config.rank. Values used in paper: {32, 64, 96} for LLMs
                (Mistral-7B, Gemma-2 9B, Llama-3.2 3B) and {8, 16, 24}
                for RoBERTa-large (config.yaml: defaults.rank).
            scaling: Scaling factor s in W = W_0 + s*B@R@A. For LoRA-SB
                this is always 1.0 per Theorem 5 (scaling-factor independence
                when B^T B = A A^T = I). Sourced from config.yaml:
                lora_sb.scaling: 1.0. Defaults to 1.0.
            device: Compute device for tensor operations. Used to place
                identity matrices on the correct device in
                verify_orthonormality(). Sourced from utils.seed_utils.get_device().
                Defaults to CPU.
            svd_niter: Number of power iterations for torch.svd_lowrank.
                Controls the accuracy-speed tradeoff of the randomized SVD
                algorithm. Higher values give more accurate singular vectors
                at the cost of more computation. Sourced from config.yaml:
                lora_sb.svd_niter: 4. The paper reports initialization takes
                less than one second per model with this setting (Appendix F).
                Defaults to 4.
            ortho_atol: Absolute tolerance for orthonormality verification
                in verify_orthonormality(). Sourced from config.yaml:
                lora_sb.ortho_atol: 1.0e-5. Appropriate for float32 tensors.
                Defaults to 1e-5.

        Raises:
            ValueError: If rank <= 0 or scaling <= 0.
        """
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if scaling <= 0.0:
            raise ValueError(f"scaling must be positive, got {scaling}")
        if svd_niter <= 0:
            raise ValueError(f"svd_niter must be positive, got {svd_niter}")
        if ortho_atol <= 0.0:
            raise ValueError(f"ortho_atol must be positive, got {ortho_atol}")

        self.rank: int = rank
        self.scaling: float = scaling
        self.device: torch.device = device
        self.svd_niter: int = svd_niter
        self.ortho_atol: float = ortho_atol

    def initialize(self, delta_w: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute (B_init, R_init, A_init) from ΔW_avg via truncated SVD.

        Implements the LoRA-SB initialization from Section 2.6 of the paper:

            U, S, Vt = SVD(ΔW_avg)
            B_init = U[:, :r]          # orthonormal columns
            A_init = Vt[:r, :]         # orthonormal rows
            R_init = (1/s) * diag(S[:r])  # with s=1.0: R_init = diag(S)

        By the Eckart-Young theorem, B_init @ R_init @ A_init is the optimal
        rank-r approximation of ΔW_avg in Frobenius norm.

        The orthonormality of B_init and A_init (B^T B = A A^T = I_r) is
        guaranteed by the SVD decomposition and enables:
        - Theorem 3 simplification: g^R = g^R_{LoRA-XS} (no matrix inversions)
        - Theorem 5: scaling-factor independence (s=1.0 is optimal)
        - Theorem 4: guaranteed loss reduction ΔL ≤ 0

        Args:
            delta_w: Averaged gradient approximation ΔW_avg for one weight
                matrix, shape (m, n) = (out_features, in_features). Computed
                by GradientEstimator as -sign(Σ_i ∇_W L(W_0, x_i)).
                May be in bfloat16 (model precision) or float32.
                May be on CPU or GPU.

        Returns:
            A tuple (B_init, R_init, A_init) where:
                - B_init: shape (m, r) = (out_features, rank), orthonormal
                    columns. Assigned to LoRASBLinear.lora_B (frozen).
                - R_init: shape (r, r) = (rank, rank), diagonal matrix with
                    top-r singular values of ΔW_avg. Assigned to
                    LoRASBLinear.lora_R (trainable).
                - A_init: shape (r, n) = (rank, in_features), orthonormal
                    rows. Assigned to LoRASBLinear.lora_A (frozen).
            All tensors are on self.device and in the same dtype as the
            input delta_w (after internal float32 casting for SVD stability).

        Raises:
            ValueError: If delta_w has fewer than 2 dimensions.
            RuntimeError: If SVD computation fails (e.g., NaN/Inf in delta_w).

        Note:
            SVD is computed in float32 for numerical stability even if
            delta_w is in bfloat16. Results are cast back to delta_w's
            original dtype before returning.
        """
        if delta_w.dim() != 2:
            raise ValueError(
                f"delta_w must be a 2D matrix, got shape {delta_w.shape}"
            )

        m, n = delta_w.shape
        original_dtype: torch.dtype = delta_w.dtype
        original_device: torch.device = delta_w.device

        # -----------------------------------------------------------------------
        # Guard: clamp rank to valid range.
        # rank must be < min(m, n) for torch.svd_lowrank to work correctly.
        # This can happen for small GLUE tasks with RoBERTa's smaller layers.
        # -----------------------------------------------------------------------
        effective_rank: int = min(self.rank, min(m, n) - 1)
        if effective_rank <= 0:
            # Degenerate case: matrix is too small for any low-rank approximation.
            effective_rank = 1
        if effective_rank < self.rank:
            logger.warning(
                "Requested rank %d exceeds matrix dimensions (%d, %d). "
                "Clamping to effective_rank=%d.",
                self.rank, m, n, effective_rank,
            )

        # -----------------------------------------------------------------------
        # Check for degenerate (zero or near-zero) delta_w.
        # This corresponds to the "Kaiming initialization" failure case in
        # Table 4 of the paper (accuracy = 0.00) — when the gradient signal
        # is absent, the initialization captures no task-relevant subspace.
        # -----------------------------------------------------------------------
        delta_w_norm: float = delta_w.float().norm().item()
        if delta_w_norm < 1e-8:
            logger.warning(
                "delta_w has near-zero norm (%.2e) for matrix of shape (%d, %d). "
                "The initialization will capture no task-relevant subspace. "
                "This may result in poor performance (cf. Table 4 in the paper).",
                delta_w_norm, m, n,
            )

        # -----------------------------------------------------------------------
        # Cast to float32 for numerical stability.
        # bfloat16 has only 7 mantissa bits (vs 23 for float32), which can
        # cause significant numerical errors in the randomized SVD power
        # iterations. We always compute SVD in float32 and cast back.
        # -----------------------------------------------------------------------
        delta_w_f32: Tensor = delta_w.to(dtype=torch.float32, device=self.device)

        # -----------------------------------------------------------------------
        # Compute truncated SVD via torch.svd_lowrank.
        # Returns U (m × q), S (q,), V (n × q) — note V not Vt.
        # -----------------------------------------------------------------------
        U, S, V = self._truncated_svd(delta_w_f32, effective_rank)

        # -----------------------------------------------------------------------
        # Construct initialization matrices (Section 2.6, Equations 7-9):
        #
        #   B_init = U[:, :r]          shape (m, r)
        #   A_init = V[:, :r].T        shape (r, n)  [Vt in paper notation]
        #   R_init = (1/s) * diag(S)   shape (r, r)
        #
        # With s=1.0 (LoRA-SB default per Theorem 5): R_init = diag(S).
        # -----------------------------------------------------------------------

        # B_init: left singular vectors, shape (m, effective_rank)
        # Orthonormal columns: U^T U = I_{effective_rank} by SVD construction.
        B_init: Tensor = U  # shape (m, effective_rank)

        # A_init: right singular vectors transposed, shape (effective_rank, n)
        # V from torch.svd_lowrank has shape (n, effective_rank).
        # Transposing gives Vt of shape (effective_rank, n).
        # Orthonormal rows: Vt @ Vt^T = I_{effective_rank} by SVD construction.
        A_init: Tensor = V.T  # shape (effective_rank, n)

        # R_init: diagonal matrix of singular values, shape (effective_rank, effective_rank)
        # With scaling=1.0: R_init = diag(S) directly.
        # With scaling != 1.0: R_init = (1/s) * diag(S) so that
        #   s * B @ R @ A = s * U @ (1/s)*diag(S) @ Vt = U @ diag(S) @ Vt = ΔW_avg.
        R_init: Tensor = torch.diag(S / self.scaling)  # shape (effective_rank, effective_rank)

        # -----------------------------------------------------------------------
        # Verify orthonormality (sanity check).
        # Should always pass given correct SVD, but numerical issues with very
        # small singular values or near-rank-deficient matrices could cause
        # failures. Log a warning rather than raising to allow training to proceed.
        # -----------------------------------------------------------------------
        is_orthonormal: bool = self.verify_orthonormality(B_init, A_init)
        if not is_orthonormal:
            logger.warning(
                "Orthonormality check failed for matrix of shape (%d, %d) "
                "with rank=%d. The theoretical guarantees of Theorems 3, 4, 5 "
                "may not hold. Consider reducing rank or checking delta_w quality.",
                m, n, effective_rank,
            )

        # -----------------------------------------------------------------------
        # Cast results back to the original dtype (typically bfloat16).
        # Move to self.device (may already be there, but ensures consistency).
        # -----------------------------------------------------------------------
        B_init = B_init.to(dtype=original_dtype, device=self.device)
        R_init = R_init.to(dtype=original_dtype, device=self.device)
        A_init = A_init.to(dtype=original_dtype, device=self.device)

        logger.debug(
            "Initialized LoRA-SB matrices: B_init %s, R_init %s, A_init %s | "
            "top singular value: %.4f, bottom singular value: %.4f",
            tuple(B_init.shape), tuple(R_init.shape), tuple(A_init.shape),
            S[0].item(), S[-1].item(),
        )

        return B_init, R_init, A_init

    def _truncated_svd(
        self,
        matrix: Tensor,
        rank: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute memory-efficient truncated SVD using torch.svd_lowrank.

        Uses the randomized SVD algorithm from torch.svd_lowrank, which only
        computes the top-rank singular values and vectors. This is significantly
        more memory-efficient than full SVD (torch.linalg.svd) for large weight
        matrices (e.g., 4096×4096 in 7B models).

        The paper explicitly uses this function (Appendix F):
        "we directly compute the truncated SVD using optimized PyTorch
        libraries (torch.svd_lowrank)."

        torch.svd_lowrank API:
            U, S, V = torch.svd_lowrank(A, q=rank, niter=niter)
            - q: number of singular values/vectors to compute
            - niter: number of power iterations (config.yaml: lora_sb.svd_niter: 4)
            - Returns: U (m × q), S (q,), V (n × q)
            - NOTE: returns V, not Vt. Caller must transpose V to get Vt.

        Args:
            matrix: Input matrix of shape (m, n) in float32. Must be 2D.
                Should be on self.device for GPU acceleration.
            rank: Number of singular values/vectors to compute. Must satisfy
                0 < rank < min(m, n).

        Returns:
            A tuple (U, S, V) where:
                - U: Left singular vectors, shape (m, rank), float32.
                    Orthonormal columns: U^T U = I_rank.
                - S: Singular values, shape (rank,), float32.
                    Sorted in descending order (largest first).
                - V: Right singular vectors, shape (n, rank), float32.
                    Orthonormal columns: V^T V = I_rank.
                    NOTE: This is V, not Vt. Caller must transpose to get Vt.

        Raises:
            RuntimeError: If torch.svd_lowrank fails (e.g., NaN/Inf in matrix,
                or rank >= min(m, n)).
        """
        # Validate inputs
        if matrix.dim() != 2:
            raise ValueError(
                f"matrix must be 2D, got shape {matrix.shape}"
            )
        m, n = matrix.shape
        if rank <= 0 or rank >= min(m, n):
            raise ValueError(
                f"rank must satisfy 0 < rank < min(m, n) = min({m}, {n}) = {min(m, n)}, "
                f"got rank={rank}"
            )

        # Check for NaN/Inf which would cause SVD to fail silently or produce garbage
        if torch.isnan(matrix).any() or torch.isinf(matrix).any():
            raise RuntimeError(
                f"matrix contains NaN or Inf values for shape ({m}, {n}). "
                "Cannot compute SVD. Check gradient estimation for numerical issues."
            )

        try:
            # torch.svd_lowrank returns (U, S, V) where V has shape (n, rank).
            # niter=self.svd_niter controls accuracy (config.yaml: lora_sb.svd_niter: 4).
            # Higher niter → more accurate singular vectors, more computation.
            # The paper validates niter=4 as sufficient (Appendix F: <1 second per model).
            U: Tensor
            S: Tensor
            V: Tensor
            U, S, V = torch.svd_lowrank(matrix, q=rank, niter=self.svd_niter)
        except RuntimeError as e:
            raise RuntimeError(
                f"torch.svd_lowrank failed for matrix of shape ({m}, {n}) "
                f"with rank={rank}, niter={self.svd_niter}. "
                f"Original error: {e}"
            ) from e

        # Validate output shapes
        assert U.shape == (m, rank), (
            f"Expected U shape ({m}, {rank}), got {U.shape}"
        )
        assert S.shape == (rank,), (
            f"Expected S shape ({rank},), got {S.shape}"
        )
        assert V.shape == (n, rank), (
            f"Expected V shape ({n}, {rank}), got {V.shape}"
        )

        # Singular values should be non-negative and sorted descending.
        # torch.svd_lowrank guarantees this, but log a warning if violated.
        if (S < 0).any():
            logger.warning(
                "Negative singular values detected in SVD output for matrix "
                "shape (%d, %d). This should not happen with torch.svd_lowrank. "
                "Taking absolute values.",
                m, n,
            )
            S = S.abs()

        # Return (U, S, V) — note V not Vt. initialize() handles the transpose.
        return U, S, V

    def verify_orthonormality(self, B: Tensor, A: Tensor) -> bool:
        """Verify that B and A satisfy the orthonormality conditions.

        Checks that:
            B^T B ≈ I_r  (B has orthonormal columns)
            A A^T ≈ I_r  (A has orthonormal rows)

        These conditions are required for the theoretical guarantees of
        LoRA-SB (Theorems 3, 4, 5). They should always hold after correct
        truncated SVD, but numerical issues can cause small violations.

        The tolerance is sourced from config.yaml: lora_sb.ortho_atol: 1.0e-5.
        This is appropriate for float32 tensors. For bfloat16 inputs, the
        check is performed after casting to float32 for accuracy.

        Args:
            B: Left singular vectors matrix, shape (m, rank). Should have
                orthonormal columns: B^T B = I_rank.
            A: Right singular vectors matrix (transposed), shape (rank, n).
                Should have orthonormal rows: A A^T = I_rank.

        Returns:
            True if both orthonormality conditions hold within self.ortho_atol
            tolerance, False otherwise.

        Note:
            This method does not raise on failure — it returns False and logs
            a warning. The caller (initialize()) decides whether to proceed.
        """
        # Cast to float32 for accurate comparison (bfloat16 has limited precision)
        B_f32: Tensor = B.to(dtype=torch.float32, device=self.device)
        A_f32: Tensor = A.to(dtype=torch.float32, device=self.device)

        rank: int = B_f32.shape[1]

        # Verify B^T B ≈ I_rank
        # BtB shape: (rank, rank)
        BtB: Tensor = B_f32.T @ B_f32
        eye_rank: Tensor = torch.eye(rank, dtype=torch.float32, device=self.device)

        b_orthonormal: bool = bool(
            torch.allclose(BtB, eye_rank, atol=self.ortho_atol)
        )

        if not b_orthonormal:
            # Compute max deviation for diagnostic logging
            max_dev_b: float = (BtB - eye_rank).abs().max().item()
            logger.debug(
                "B^T B ≠ I_%d: max deviation = %.2e (tolerance = %.2e). "
                "B shape: %s",
                rank, max_dev_b, self.ortho_atol, tuple(B_f32.shape),
            )

        # Verify A A^T ≈ I_rank
        # AAt shape: (rank, rank)
        AAt: Tensor = A_f32 @ A_f32.T

        a_orthonormal: bool = bool(
            torch.allclose(AAt, eye_rank, atol=self.ortho_atol)
        )

        if not a_orthonormal:
            max_dev_a: float = (AAt - eye_rank).abs().max().item()
            logger.debug(
                "A A^T ≠ I_%d: max deviation = %.2e (tolerance = %.2e). "
                "A shape: %s",
                rank, max_dev_a, self.ortho_atol, tuple(A_f32.shape),
            )

        return b_orthonormal and a_orthonormal
