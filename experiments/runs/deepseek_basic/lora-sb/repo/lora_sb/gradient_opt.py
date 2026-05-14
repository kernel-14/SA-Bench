"""
Gradient Optimization for LoRA-SB.

Theorem 3 provides the closed-form optimal gradient for R:
    g^R = (1/s²) (B^T B)^{-1} g_LoRA-XS^R (A A^T)^{-1}

With orthonormal B and A (B^T B = I, A A^T = I) and s = 1:
    g^R = g_LoRA-XS^R

This module implements the optimal gradient approximation for LoRA-SB.
"""

import torch
import torch.nn as nn
from typing import Optional
from .lora_sb_layer import LoRA_SB_Layer


def compute_optimal_gradient(
    module: LoRA_SB_Layer,
    grad_w: torch.Tensor,
    use_orthonormal_shortcut: bool = True,
) -> torch.Tensor:
    """
    Compute the optimal gradient for R given the gradient w.r.t. W.

    Full formula:
        g^R_opt = (1/s²) (B^T B)^{-1} g_LoRA-XS^R (A A^T)^{-1}
    where:
        g_LoRA-XS^R = s * B^T @ g @ A^T

    With orthonormal B, A and s=1:
        g^R_opt = g_LoRA-XS^R

    Args:
        module: LoRA-SB layer.
        grad_w: Gradient of loss w.r.t. the output weight (same as g in paper).
        use_orthonormal_shortcut: If True, uses the simplified formula assuming
                                  orthonormal B and A. If False, computes full inverse.

    Returns:
        Optimal gradient for R.
    """
    s = module.scaling
    B = module.B
    A = module.A

    # Compute g_LoRA-XS^R = s * B^T @ g @ A^T
    g_r_xs = s * (B.T @ grad_w @ A.T)  # (r, r)

    if use_orthonormal_shortcut:
        # With orthonormal B, A and s = 1: g^R_opt = g_r_xs
        # Even with s != 1, the formula simplifies
        return g_r_xs / (s ** 2)

    # Full formula: (1/s²) (B^T B)^{-1} g_r_xs (A A^T)^{-1}
    # Compute (B^T B)^{-1}
    BtB = B.T @ B  # (r, r)
    BtB_inv = torch.linalg.inv(BtB)  # (r, r)

    # Compute (A A^T)^{-1}
    AAt = A @ A.T  # (r, r)
    AAt_inv = torch.linalg.inv(AAt)  # (r, r)

    g_r_opt = (1.0 / (s ** 2)) * BtB_inv @ g_r_xs @ AAt_inv

    return g_r_opt


class LoRASBOptimizerWrapper:
    """
    Wrapper that applies optimal gradient transformation to LoRA-SB layers.

    This can be used to modify gradient updates during training to use the
    optimal gradient approximation from Theorem 3.

    Usage:
        wrapper = LoRASBOptimizerWrapper(model)
        # In training loop:
        loss.backward()
        wrapper.apply_optimal_gradients()
        optimizer.step()
    """

    def __init__(self, model: nn.Module, use_shortcut: bool = True):
        self.model = model
        self.use_shortcut = use_shortcut
        self.lora_sb_modules = []

        for module in model.modules():
            if isinstance(module, LoRA_SB_Layer):
                self.lora_sb_modules.append(module)

    def apply_optimal_gradients(self):
        """
        Apply optimal gradient transformation to all LoRA-SB layers.

        For each LoRA-SB layer, transforms the gradient of R to the
        optimal gradient using the formula from Theorem 3.
        """
        for module in self.lora_sb_modules:
            if module.R.grad is not None:
                # The gradient accumulated on R is g_LoRA-XS^R
                # We need to transform it to g^R_opt
                if self.use_shortcut:
                    # With orthonormal B, A and s=1: g^R_opt = g_LoRA-XS^R
                    # No transformation needed!
                    pass
                else:
                    # Apply full transformation
                    s = module.scaling
                    B = module.B
                    A = module.A

                    g_r_xs = module.R.grad  # current gradient

                    BtB = B.T @ B
                    BtB_inv = torch.linalg.inv(BtB)
                    AAt = A @ A.T
                    AAt_inv = torch.linalg.inv(AAt)

                    g_r_opt = (1.0 / (s ** 2)) * BtB_inv @ g_r_xs @ AAt_inv
                    module.R.grad.copy_(g_r_opt)


def verify_orthonormality(model: nn.Module, tolerance: float = 1e-4):
    """
    Verify that B and A matrices in LoRA-SB layers are orthonormal.

    Args:
        model: Model with LoRA-SB layers.
        tolerance: Tolerance for deviation from identity.

    Returns:
        bool: True if all layers pass, False otherwise.
    """
    all_pass = True
    for name, module in model.named_modules():
        if isinstance(module, LoRA_SB_Layer):
            B = module.B
            A = module.A

            # Check B^T B ≈ I
            BtB = B.T @ B
            I_r = torch.eye(module.rank, device=B.device, dtype=B.dtype)
            btB_error = (BtB - I_r).abs().max().item()

            # Check A A^T ≈ I
            AAt = A @ A.T
            aAt_error = (AAt - I_r).abs().max().item()

            if btB_error > tolerance or aAt_error > tolerance:
                all_pass = False
                break

    return all_pass


def compute_equivalent_gradient(module: LoRA_SB_Layer) -> torch.Tensor:
    """
    Compute the equivalent gradient for W given the gradient of R.

    Definition: \tilde{g} = s * B @ g^R @ A

    This is the virtual low-rank gradient of W that results from updating R.

    Args:
        module: LoRA-SB layer.

    Returns:
        Equivalent gradient tensor of shape (m, n).
    """
    if module.R.grad is None:
        raise ValueError("R has no gradient. Call backward() first.")

    s = module.scaling
    B = module.B
    A = module.A
    g_R = module.R.grad

    return s * B @ g_R @ A
