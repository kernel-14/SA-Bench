## baselines/lora_pro.py
"""LoRA-Pro baseline linear layer module.

Implements LoRAProLinear, the LoRA-Pro baseline from Wang et al. (2024, ref 46).
LoRA-Pro improves standard LoRA by applying a closed-form gradient correction
at each optimizer step, aligning the low-rank gradient more closely with the
full fine-tuning gradient.

The architecture is standard LoRA (W = W₀ + s·B·A) with both B and A trainable,
augmented by per-step gradient corrections:

    g^A_optimal = (1/s²) * (B^T B + ε·I)^{-1} @ g^A_{LoRA}
    g^B_optimal = (1/s²) * g^B_{LoRA} @ (A A^T + ε·I)^{-1}

These corrections are applied via backward hooks registered on lora_A and lora_B,
so the optimizer sees corrected gradients automatically without any changes to
the training loop.

Key differences from LoRA-SB:
    - Both B and A are trainable (r*(m+n) params vs r² for LoRA-SB)
    - Scaling requires tuning (alpha/rank, not provably 1.0)
    - Gradient correction is expensive (2 matrix inversions per layer per step)
    - No orthonormality guarantee (B, A drift during training)
    - Standard LoRA initialization (Gaussian B, zero A)

References:
    Paper Section 3 (Table 1-3): LoRA-Pro baseline results
    LoRA-Pro paper (ref 46): Wang et al., "LoRA-Pro: Are Low-Rank Adapters
        Properly Optimized?" arXiv:2407.18242
    config.yaml: baselines.lora_pro.alpha_equals_rank: true
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# Small regularization constant added to B^T B and A A^T before inversion
# to prevent numerical instability when matrices are near-singular.
# This is especially important early in training when A is near-zero
# (standard LoRA initializes A to zeros, making A A^T = 0).
_REGULARIZATION_EPS: float = 1e-6


class LoRAProLinear(nn.Module):
    """LoRA-Pro linear layer: W = W₀ + s * B @ A with optimal gradient correction.

    Implements standard LoRA architecture (B ∈ R^{m×r}, A ∈ R^{r×n}, both
    trainable) augmented with per-step closed-form gradient corrections that
    minimize the discrepancy between the low-rank equivalent gradient and the
    full fine-tuning gradient.

    The gradient correction is applied automatically via backward hooks on
    lora_A and lora_B. The optimizer (AdamW) sees the corrected gradients
    without any changes to the training loop.

    Initialization follows standard LoRA convention:
        - lora_B: Kaiming uniform initialization (non-zero, provides gradient signal)
        - lora_A: Zero initialization (ensures B @ A = 0 at start, no initial perturbation)

    Attributes:
        in_features: Input feature dimension n.
        out_features: Output feature dimension m.
        rank: LoRA rank r. Trainable parameter count = r*(m+n).
        scaling: Scaling factor s = alpha / rank. Requires tuning (unlike LoRA-SB).
        weight: Frozen pre-trained weight W₀, shape (out_features, in_features).
        bias_param: Frozen pre-trained bias, shape (out_features,), or None.
        lora_B: Trainable low-rank matrix B, shape (out_features, rank).
            Initialized with Kaiming uniform. Updated by AdamW with corrected gradients.
        lora_A: Trainable low-rank matrix A, shape (rank, in_features).
            Initialized to zeros. Updated by AdamW with corrected gradients.
        _hook_A: Handle for the backward hook on lora_A. None until
            register_grad_hooks() is called.
        _hook_B: Handle for the backward hook on lora_B. None until
            register_grad_hooks() is called.
        _reg_eps: Regularization epsilon for matrix inversions. Prevents
            numerical instability when B^T B or A A^T is near-singular.

    Example:
        >>> layer = LoRAProLinear(
        ...     in_features=4096, out_features=4096, rank=32, scaling=1.0
        ... )
        >>> layer.weight.data.copy_(pretrained_weight)
        >>> layer.register_grad_hooks()  # activate gradient correction
        >>> x = torch.randn(2, 512, 4096)
        >>> out = layer(x)  # forward pass
        >>> loss = out.sum()
        >>> loss.backward()  # hooks apply correction to lora_A.grad and lora_B.grad
        >>> optimizer.step()  # optimizer sees corrected gradients
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        scaling: float,
        bias: bool = True,
        reg_eps: float = _REGULARIZATION_EPS,
    ) -> None:
        """Initialize LoRAProLinear with standard LoRA initialization.

        Creates the module structure with Kaiming-uniform-initialized lora_B
        and zero-initialized lora_A (standard LoRA convention). The gradient
        correction hooks are NOT registered here — call register_grad_hooks()
        after construction to activate the LoRA-Pro correction.

        Args:
            in_features: Size of each input sample (n in paper notation).
                Corresponds to nn.Linear.in_features of the replaced layer.
            out_features: Size of each output sample (m in paper notation).
                Corresponds to nn.Linear.out_features of the replaced layer.
            rank: LoRA rank r. Must be positive and ≤ min(in_features, out_features).
                Trainable parameter count for this layer = rank * (in_features + out_features).
                Paper uses r=32 for all LoRA-Pro comparisons (Tables 1-3).
            scaling: Scaling factor s = alpha / rank. For LoRA-Pro with
                alpha_equals_rank=True (config.yaml: baselines.lora_pro):
                scaling = rank / rank = 1.0. Passed in by ModelBuilder as
                config.alpha / config.rank. Unlike LoRA-SB, this value is not
                provably optimal and may require tuning.
            bias: If True, a frozen bias parameter is created matching the
                original nn.Linear. If False, self.bias_param is None.
                Defaults to True.
            reg_eps: Regularization epsilon added to B^T B and A A^T before
                matrix inversion to prevent numerical instability. Defaults to
                1e-6, which is safe for float32 and bfloat16 computations.

        Raises:
            ValueError: If rank <= 0, scaling <= 0, or reg_eps < 0.
        """
        super().__init__()

        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if rank > min(in_features, out_features):
            raise ValueError(
                f"rank ({rank}) must be ≤ min(in_features, out_features) "
                f"= min({in_features}, {out_features}) = {min(in_features, out_features)}"
            )
        if scaling <= 0.0:
            raise ValueError(f"scaling must be positive, got {scaling}")
        if reg_eps < 0.0:
            raise ValueError(f"reg_eps must be non-negative, got {reg_eps}")

        # -----------------------------------------------------------------------
        # Store dimensions and hyperparameters
        # -----------------------------------------------------------------------
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.rank: int = rank
        # scaling = alpha / rank. For LoRA-Pro with alpha=rank: scaling=1.0.
        # This is a plain float (not nn.Parameter) — a fixed hyperparameter.
        self.scaling: float = scaling
        # Regularization epsilon for matrix inversions in gradient correction.
        self._reg_eps: float = reg_eps

        # -----------------------------------------------------------------------
        # Frozen base weight W₀: shape (out_features, in_features)
        # Follows PyTorch nn.Linear convention: weight shape is (out, in).
        # requires_grad=False: never updated during training.
        # Initialized to zeros; ModelBuilder copies pre-trained values here.
        # -----------------------------------------------------------------------
        self.weight: nn.Parameter = nn.Parameter(
            torch.zeros(out_features, in_features),
            requires_grad=False,
        )

        # -----------------------------------------------------------------------
        # Optional frozen bias: shape (out_features,)
        # Bias is part of the pre-trained model and is not adapted by LoRA-Pro.
        # requires_grad=False: never updated during training.
        # Named bias_param (not bias) to avoid conflict with nn.Module.bias
        # attribute convention while still appearing in state_dict().
        # -----------------------------------------------------------------------
        if bias:
            self.bias_param: Optional[nn.Parameter] = nn.Parameter(
                torch.zeros(out_features),
                requires_grad=False,
            )
        else:
            self.bias_param = None

        # -----------------------------------------------------------------------
        # Trainable low-rank matrix B: shape (out_features, rank) = (m, r)
        # Standard LoRA initialization: Kaiming uniform (non-zero).
        # This provides a non-zero gradient signal from the first step.
        # requires_grad=True: updated by AdamW with corrected gradients.
        # -----------------------------------------------------------------------
        self.lora_B: nn.Parameter = nn.Parameter(
            torch.empty(out_features, rank),
            requires_grad=True,
        )
        # Kaiming uniform initialization: standard for LoRA B matrix.
        # Uses fan_in mode with a=math.sqrt(5) (PyTorch default for Linear).
        nn.init.kaiming_uniform_(self.lora_B, a=5 ** 0.5)

        # -----------------------------------------------------------------------
        # Trainable low-rank matrix A: shape (rank, in_features) = (r, n)
        # Standard LoRA initialization: zeros.
        # Ensures B @ A = 0 at initialization → no perturbation to W₀.
        # requires_grad=True: updated by AdamW with corrected gradients.
        # -----------------------------------------------------------------------
        self.lora_A: nn.Parameter = nn.Parameter(
            torch.zeros(rank, in_features),
            requires_grad=True,
        )

        # -----------------------------------------------------------------------
        # Gradient hook handles.
        # Initialized to None; populated by register_grad_hooks().
        # Stored as instance attributes so they can be removed by
        # remove_grad_hooks() before model saving or inference.
        # -----------------------------------------------------------------------
        self._hook_A: Optional[torch.utils.hooks.RemovableHook] = None
        self._hook_B: Optional[torch.utils.hooks.RemovableHook] = None

    def forward(self, x: Tensor) -> Tensor:
        """Compute the LoRA-Pro forward pass: output = x @ (W₀ + s·B·A)^T + bias.

        Standard LoRA forward pass. The gradient correction is applied
        automatically during the backward pass via registered hooks — no
        special logic is needed here.

        The low-rank update is computed as:
            delta_W = scaling * lora_B @ lora_A   shape: (out_features, in_features)
            effective_W = weight + delta_W         shape: (out_features, in_features)
            output = F.linear(x, effective_W, bias_param)

        Args:
            x: Input tensor of shape (..., in_features). Supports arbitrary
                batch dimensions (e.g., (batch, seq_len, in_features) for
                transformer hidden states).

        Returns:
            Output tensor of shape (..., out_features).
        """
        # Compute low-rank update: s * B @ A, shape (out_features, in_features)
        delta_w: Tensor = self.scaling * (self.lora_B @ self.lora_A)

        # Effective weight: W₀ + s·B·A
        # Using + (not +=) to avoid in-place modification of self.weight.data.
        effective_weight: Tensor = self.weight + delta_w

        # Apply linear transformation with optional bias.
        # F.linear handles arbitrary batch dimensions correctly.
        return F.linear(x, effective_weight, self.bias_param)

    def compute_optimal_gradient(self, grad_input: Tensor, is_grad_A: bool) -> Tensor:
        """Apply the LoRA-Pro closed-form gradient correction.

        Implements the optimal gradient correction from LoRA-Pro (ref 46),
        which minimizes the Frobenius norm discrepancy between the low-rank
        equivalent gradient and the full fine-tuning gradient.

        For gradient w.r.t. A (is_grad_A=True):
            g^A_optimal = (1/s²) * (B^T B + ε·I)^{-1} @ g^A_{LoRA}

        For gradient w.r.t. B (is_grad_A=False):
            g^B_optimal = (1/s²) * g^B_{LoRA} @ (A A^T + ε·I)^{-1}

        The regularization term ε·I (self._reg_eps * I) prevents numerical
        instability when B^T B or A A^T is near-singular. This is especially
        important early in training when A is near-zero (A A^T ≈ 0).

        All intermediate computations use .detach() on B and A to avoid
        creating a second-order computation graph. The correction modifies
        the gradient, not the forward pass.

        Args:
            grad_input: Raw gradient tensor from the backward pass.
                - If is_grad_A=True: shape (rank, in_features) = (r, n).
                  This is ∂L/∂A from standard backprop.
                - If is_grad_A=False: shape (out_features, rank) = (m, r).
                  This is ∂L/∂B from standard backprop.
            is_grad_A: If True, applies the correction for grad_A using
                (B^T B)^{-1}. If False, applies the correction for grad_B
                using (A A^T)^{-1}.

        Returns:
            Corrected gradient tensor of the same shape as grad_input.
            The optimizer will use this corrected gradient instead of the
            raw backprop gradient.

        Note:
            Computations are performed in float32 for numerical stability
            (matrix inversion is sensitive to precision), then cast back to
            the input gradient's original dtype.
        """
        original_dtype: torch.dtype = grad_input.dtype
        original_device: torch.device = grad_input.device

        # Cast to float32 for stable matrix inversion.
        grad_f32: Tensor = grad_input.to(dtype=torch.float32)

        if is_grad_A:
            # -----------------------------------------------------------------------
            # Correction for grad_A:
            #   g^A_optimal = (1/s²) * (B^T B + ε·I)^{-1} @ g^A_{LoRA}
            #
            # B has shape (out_features, rank) = (m, r).
            # B^T B has shape (rank, rank) = (r, r).
            # (B^T B + ε·I)^{-1} has shape (rank, rank) = (r, r).
            # g^A_{LoRA} has shape (rank, in_features) = (r, n).
            # Result: (r, r) @ (r, n) = (r, n) ✓
            # -----------------------------------------------------------------------
            B_detached: Tensor = self.lora_B.detach().to(dtype=torch.float32)

            # B^T B: (rank, rank)
            BtB: Tensor = B_detached.T @ B_detached

            # Add regularization: (B^T B + ε·I)
            eye_r: Tensor = torch.eye(
                self.rank, dtype=torch.float32, device=original_device
            )
            BtB_reg: Tensor = BtB + self._reg_eps * eye_r

            # Compute (B^T B + ε·I)^{-1} using torch.linalg.inv.
            # For r×r matrices (r ≤ 96), this is fast and numerically stable
            # with the regularization term.
            try:
                BtB_inv: Tensor = torch.linalg.inv(BtB_reg)
            except RuntimeError as e:
                logger.warning(
                    "Matrix inversion failed for B^T B (rank=%d): %s. "
                    "Falling back to identity (no correction for this step).",
                    self.rank, str(e),
                )
                # Fallback: return unmodified gradient scaled by 1/s²
                corrected: Tensor = (1.0 / (self.scaling ** 2)) * grad_f32
                return corrected.to(dtype=original_dtype)

            # Apply correction: (1/s²) * (B^T B + ε·I)^{-1} @ g^A
            # Shape: (r, r) @ (r, n) = (r, n)
            corrected_f32: Tensor = (1.0 / (self.scaling ** 2)) * (BtB_inv @ grad_f32)

        else:
            # -----------------------------------------------------------------------
            # Correction for grad_B:
            #   g^B_optimal = (1/s²) * g^B_{LoRA} @ (A A^T + ε·I)^{-1}
            #
            # A has shape (rank, in_features) = (r, n).
            # A A^T has shape (rank, rank) = (r, r).
            # (A A^T + ε·I)^{-1} has shape (rank, rank) = (r, r).
            # g^B_{LoRA} has shape (out_features, rank) = (m, r).
            # Result: (m, r) @ (r, r) = (m, r) ✓
            # -----------------------------------------------------------------------
            A_detached: Tensor = self.lora_A.detach().to(dtype=torch.float32)

            # A A^T: (rank, rank)
            AAt: Tensor = A_detached @ A_detached.T

            # Add regularization: (A A^T + ε·I)
            eye_r_b: Tensor = torch.eye(
                self.rank, dtype=torch.float32, device=original_device
            )
            AAt_reg: Tensor = AAt + self._reg_eps * eye_r_b

            # Compute (A A^T + ε·I)^{-1}
            try:
                AAt_inv: Tensor = torch.linalg.inv(AAt_reg)
            except RuntimeError as e:
                logger.warning(
                    "Matrix inversion failed for A A^T (rank=%d): %s. "
                    "Falling back to identity (no correction for this step).",
                    self.rank, str(e),
                )
                corrected = (1.0 / (self.scaling ** 2)) * grad_f32
                return corrected.to(dtype=original_dtype)

            # Apply correction: (1/s²) * g^B @ (A A^T + ε·I)^{-1}
            # Shape: (m, r) @ (r, r) = (m, r)
            corrected_f32 = (1.0 / (self.scaling ** 2)) * (grad_f32 @ AAt_inv)

        # Cast corrected gradient back to original dtype (bfloat16 for LLMs).
        return corrected_f32.to(dtype=original_dtype)

    def register_grad_hooks(self) -> None:
        """Register backward hooks to apply gradient correction automatically.

        Registers two hooks:
            1. A hook on lora_A that applies the (B^T B)^{-1} correction.
            2. A hook on lora_B that applies the (A A^T)^{-1} correction.

        After this call, every backward pass will automatically produce
        corrected gradients for lora_A and lora_B. The AdamW optimizer in
        Trainer._setup_optimizer() will use these corrected gradients without
        any changes to the training loop.

        This method is idempotent: calling it multiple times removes existing
        hooks before registering new ones, preventing duplicate corrections.

        Note:
            Hooks must be registered AFTER the module is moved to the target
            device (e.g., after model.to(device) or model.cuda()). The hooks
            capture self by reference, so they always use the current state
            of lora_B and lora_A at the time of the backward pass.

        Note:
            The hooks return the corrected gradient tensor (not None), which
            replaces the raw gradient in the parameter's .grad attribute.
            This is the "gradient modification" hook pattern, distinct from
            the "observational" pattern used in GradientEstimator.
        """
        # Remove any existing hooks to prevent duplicate corrections.
        self.remove_grad_hooks()

        # -----------------------------------------------------------------------
        # Hook for lora_A: applies (B^T B + ε·I)^{-1} correction.
        # grad shape: (rank, in_features) = (r, n).
        # Returns corrected gradient of the same shape.
        # -----------------------------------------------------------------------
        def hook_A(grad: Tensor) -> Tensor:
            """Gradient hook for lora_A: applies (B^T B)^{-1} correction."""
            return self.compute_optimal_gradient(grad, is_grad_A=True)

        # -----------------------------------------------------------------------
        # Hook for lora_B: applies (A A^T + ε·I)^{-1} correction.
        # grad shape: (out_features, rank) = (m, r).
        # Returns corrected gradient of the same shape.
        # -----------------------------------------------------------------------
        def hook_B(grad: Tensor) -> Tensor:
            """Gradient hook for lora_B: applies (A A^T)^{-1} correction."""
            return self.compute_optimal_gradient(grad, is_grad_A=False)

        # Register hooks on the parameter tensors.
        # register_hook() returns a RemovableHook handle stored for later cleanup.
        self._hook_A = self.lora_A.register_hook(hook_A)
        self._hook_B = self.lora_B.register_hook(hook_B)

        logger.debug(
            "Registered LoRA-Pro gradient correction hooks for layer "
            "(out=%d, in=%d, rank=%d, scaling=%.4f, reg_eps=%.2e).",
            self.out_features, self.in_features, self.rank,
            self.scaling, self._reg_eps,
        )

    def remove_grad_hooks(self) -> None:
        """Remove registered gradient correction hooks.

        Removes the backward hooks on lora_A and lora_B that apply the
        LoRA-Pro gradient correction. After this call, the optimizer will
        use raw (uncorrected) gradients.

        This should be called:
        - Before model.save_pretrained() or torch.save() to avoid serializing
          hook state (hooks are not serializable).
        - When switching to inference mode (no backward pass needed).
        - Before register_grad_hooks() to prevent duplicate hooks.

        This method is safe to call even if no hooks are registered (no-op).
        """
        if self._hook_A is not None:
            self._hook_A.remove()
            self._hook_A = None
            logger.debug("Removed gradient correction hook from lora_A.")

        if self._hook_B is not None:
            self._hook_B.remove()
            self._hook_B = None
            logger.debug("Removed gradient correction hook from lora_B.")

    def extra_repr(self) -> str:
        """Return a human-readable string for print(model) output.

        Provides key architectural information for each LoRAProLinear layer,
        including the trainable parameter count (r*(m+n)) and whether gradient
        correction hooks are currently active.

        Returns:
            A string summarizing the layer's configuration, e.g.:
            "in_features=4096, out_features=4096, rank=32, scaling=1.0,
             bias=True, trainable_params=262144, hooks_active=True"
        """
        trainable_params: int = self.rank * (self.in_features + self.out_features)
        hooks_active: bool = (self._hook_A is not None) and (self._hook_B is not None)
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"rank={self.rank}, "
            f"scaling={self.scaling:.4f}, "
            f"bias={self.bias_param is not None}, "
            f"trainable_params={trainable_params}, "
            f"hooks_active={hooks_active}, "
            f"reg_eps={self._reg_eps:.2e}"
        )
