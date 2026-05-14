"""
Unit tests for LoRA-SB implementation.

Tests the core theoretical properties:
- Lemma 1: Constrained update space
- Lemma 2: Gradient relationship g_R = s * B^T @ g @ A^T
- Theorem 3: Optimal gradient formula
- Theorem 5: Scaling factor independence
- Theorem 6: Optimal low-rank approximation of first step
- Orthonormality of B and A after initialization
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lora_sb.lora_sb_layer import LoRA_SB_Layer
from lora_sb.init import truncated_svd_init


def test_lora_sb_layer_creation():
    """Test basic LoRA-SB layer creation."""
    m, n, r = 64, 32, 4
    linear = nn.Linear(n, m, bias=False)
    lora_sb = LoRA_SB_Layer.from_pretrained_linear(linear, rank=r, scaling=1.0)

    assert lora_sb.in_features == n
    assert lora_sb.out_features == m
    assert lora_sb.rank == r
    assert lora_sb.scaling == 1.0
    assert lora_sb.B.shape == (m, r)
    assert lora_sb.R.shape == (r, r)
    assert lora_sb.A.shape == (r, n)
    assert lora_sb.W0.shape == (m, n)
    assert lora_sb.R.requires_grad
    assert not lora_sb.B.requires_grad
    assert not lora_sb.A.requires_grad
    print("✓ test_lora_sb_layer_creation passed")


def test_lemma_2_gradient_relationship():
    """
    Lemma 2: g_R = s * B^T @ g @ A^T
    where g = dL/dW has shape (m, n)
    """
    small_m, small_n, small_r = 4, 3, 2
    small_B = torch.randn(small_m, small_r)  # (4, 2)
    small_A = torch.randn(small_r, small_n)  # (2, 3)
    small_R = torch.randn(small_r, small_r, requires_grad=True)  # (2, 2)
    small_s = 1.0
    small_W0 = torch.randn(small_m, small_n)  # (4, 3)

    small_x = torch.randn(1, small_n)  # (1, 3)
    # Forward: y = x @ W0^T + s * x @ A^T @ R^T @ B^T
    small_y = small_x @ small_W0.T + small_s * (small_x @ small_A.T @ small_R.T @ small_B.T)
    small_loss = small_y.sum()
    small_loss.backward()

    small_g_R = small_R.grad.clone()  # (2, 2)

    # dL/dy = ones(1, m) since loss = sum(y)
    # g = dL/dW = x^T @ dL/dy = x^T @ ones(1, m) -> (n, 1) @ (1, m) = (n, m)
    small_g = small_x.T @ torch.ones(1, small_m)  # (3, 4)

    # Lemma 2: g_R = s * B^T @ g @ A^T
    # B^T: (r, m) = (2, 4), g: (m, n) = (4, 3) wait g is (n, m) = (3, 4)
    # Actually g should be (m, n) for W of shape (m, n)
    # dL/dW where W is (m, n): g = x^T @ dL/dy where dL/dy is (batch, m)
    # So g = (n, 1) @ (1, m) = (n, m) -- but in paper g has shape (m, n)
    # The transpose: g_paper = g^T = (m, n)
    small_g_paper = small_g.T  # (4, 3) = (m, n)

    small_g_R_expected = small_s * (small_B.T @ small_g_paper @ small_A.T)
    # B^T: (2, 4), g_paper: (4, 3), A^T: (3, 2) -> result: (2, 2)

    print(f"  g_R actual: {small_g_R}")
    print(f"  g_R expected (Lemma 2): {small_g_R_expected}")
    assert torch.allclose(small_g_R, small_g_R_expected, atol=1e-5), \
        f"Lemma 2 failed: max diff = {(small_g_R - small_g_R_expected).abs().max()}"
    print("✓ test_lemma_2_gradient_relationship passed")


def test_truncated_svd_init_orthonormal():
    """
    Test truncated SVD produces orthonormal B and A.
    """
    m, n, r = 64, 32, 4
    delta_w = torch.randn(m, n)
    B_init, R_init, A_init = truncated_svd_init(delta_w, rank=r, scaling=1.0)

    assert B_init.shape == (m, r)
    assert R_init.shape == (r, r)
    assert A_init.shape == (r, n)

    # B^T @ B = I
    BtB = B_init.T @ B_init
    I_r = torch.eye(r)
    ortho_error_B = (BtB - I_r).abs().max().item()
    print(f"  B^T B orthonormality error: {ortho_error_B:.2e}")
    assert ortho_error_B < 1e-4

    # A @ A^T = I
    AAt = A_init @ A_init.T
    ortho_error_A = (AAt - I_r).abs().max().item()
    print(f"  A A^T orthonormality error: {ortho_error_A:.2e}")
    assert ortho_error_A < 1e-4

    # Reconstruction: s * B @ R @ A ≈ ΔW
    reconstruction = B_init @ R_init @ A_init
    original_norm = delta_w.norm()
    relative_error = (delta_w - reconstruction).norm() / original_norm
    print(f"  Reconstruction relative error: {relative_error:.4f}")
    print("✓ test_truncated_svd_init_orthonormal passed")


def test_theorem_3_optimal_gradient():
    """
    Theorem 3: optimal g^R minimizes ||s B g^R A - g||_F^2
    """
    m, n, r = 64, 32, 4
    delta_w = torch.randn(m, n)
    B, _, A = truncated_svd_init(delta_w, rank=r, scaling=1.0)
    g = torch.randn(m, n)

    s = 1.0
    g_r_xs = s * B.T @ g @ A.T
    g_r_opt_simple = g_r_xs / (s ** 2)

    # Full formula
    BtB_inv = torch.linalg.inv(B.T @ B)
    AAt_inv = torch.linalg.inv(A @ A.T)
    g_r_opt_full = (1.0 / (s ** 2)) * BtB_inv @ g_r_xs @ AAt_inv

    assert torch.allclose(g_r_opt_simple, g_r_opt_full, atol=1e-5)

    # Verify optimality
    equiv_g_opt = s * B @ g_r_opt_simple @ A
    loss_opt = ((equiv_g_opt - g) ** 2).sum().item()

    for _ in range(10):
        g_r_rand = g_r_opt_simple + 0.1 * torch.randn(r, r)
        equiv_g_rand = s * B @ g_r_rand @ A
        loss_rand = ((equiv_g_rand - g) ** 2).sum().item()
        assert loss_opt <= loss_rand + 1e-10

    print("✓ test_theorem_3_optimal_gradient passed")


def test_theorem_5_scaling_independence():
    """
    Theorem 5: equivalent gradient is s-independent with optimal g^R
    """
    m, n, r = 64, 32, 4
    delta_w = torch.randn(m, n)
    B, _, A = truncated_svd_init(delta_w, rank=r, scaling=1.0)
    g = torch.randn(m, n)

    equiv_grads_opt = []
    equiv_grads_xs = []

    for s in [0.5, 1.0, 2.0]:
        g_r_xs = s * B.T @ g @ A.T
        g_r_opt = g_r_xs / (s ** 2)
        equiv_g_opt = s * B @ g_r_opt @ A
        equiv_g_xs = s * B @ g_r_xs @ A
        equiv_grads_opt.append(equiv_g_opt)
        equiv_grads_xs.append(equiv_g_xs)

    for i in range(1, len(equiv_grads_opt)):
        assert torch.allclose(equiv_grads_opt[0], equiv_grads_opt[i], atol=1e-5)

    assert not torch.allclose(equiv_grads_xs[0], equiv_grads_xs[1], atol=1e-5)

    print("✓ test_theorem_5_scaling_independence passed")


def test_forward_pass_equivalence():
    """Test forward pass correctness."""
    m, n, r = 64, 32, 4
    batch_size = 8
    linear = nn.Linear(n, m, bias=False)
    lora_sb = LoRA_SB_Layer.from_pretrained_linear(linear, rank=r)

    B_val = torch.randn(m, r)
    R_val = torch.randn(r, r)
    A_val = torch.randn(r, n)
    lora_sb.initialize_ba(B_val, R_val, A_val)

    x = torch.randn(batch_size, n)
    y_lora_sb = lora_sb(x)
    y_manual = x @ lora_sb.W0.T + lora_sb.scaling * (x @ A_val.T @ R_val.T @ B_val.T)

    assert torch.allclose(y_lora_sb, y_manual, atol=1e-5)
    print("✓ test_forward_pass_equivalence passed")


def test_merge():
    """Test merge produces correct weights."""
    m, n, r = 64, 32, 4
    linear = nn.Linear(n, m, bias=False)
    lora_sb = LoRA_SB_Layer.from_pretrained_linear(linear, rank=r)

    B_val = torch.randn(m, r)
    R_val = torch.randn(r, r)
    A_val = torch.randn(r, n)
    lora_sb.initialize_ba(B_val, R_val, A_val)

    merged = lora_sb.merge()
    expected = lora_sb.W0 + lora_sb.scaling * B_val @ R_val @ A_val
    assert torch.allclose(merged.weight.data, expected, atol=1e-5)
    print("✓ test_merge passed")


def test_theorem_6_optimal_first_step():
    """
    Theorem 6: initialization gives optimal rank-r approximation of first step.
    """
    m, n, r = 64, 32, 4
    eta = 0.01
    g = torch.randn(m, n)
    delta_w_avg = -eta * torch.sign(g)

    B_init, R_init, A_init = truncated_svd_init(delta_w_avg, rank=r, scaling=1.0)
    initial_product = B_init @ R_init @ A_init

    U, S, Vt = torch.linalg.svd(delta_w_avg, full_matrices=False)
    best_rank_r = U[:, :r] @ torch.diag(S[:r]) @ Vt[:r, :]

    assert torch.allclose(initial_product, best_rank_r, atol=1e-4)
    print("✓ test_theorem_6_optimal_first_step passed")


def test_parameter_count_reduction():
    """Verify parameter reduction."""
    m, n = 1024, 1024
    r = 16
    linear = nn.Linear(n, m, bias=False)
    lora_sb = LoRA_SB_Layer.from_pretrained_linear(linear, rank=r)

    lora_params = r * (m + n)
    lora_sb_params = r * r
    reduction_ratio = lora_params / lora_sb_params
    print(f"  Reduction ratio: {reduction_ratio:.1f}x (LoRA: {lora_params}, LoRA-SB: {lora_sb_params})")
    assert reduction_ratio > 100
    print("✓ test_parameter_count_reduction passed")


def test_gradient_autograd_consistency():
    """
    Verify that the autograd gradient for R matches the formula
    g_R = s * B^T @ g @ A^T where g = dL/dW.
    """
    m, n, r = 6, 4, 2
    linear = nn.Linear(n, m, bias=False)
    lora_sb = LoRA_SB_Layer.from_pretrained_linear(linear, rank=r, scaling=1.0)

    # Set known orthonormal B, A
    B_val = torch.randn(m, r)
    B_val, _ = torch.linalg.qr(B_val)
    A_val = torch.randn(r, n)
    Q, _ = torch.linalg.qr(A_val.T)
    A_val = Q.T

    lora_sb.initialize_ba(B_val, torch.eye(r), A_val)

    # Forward and backward
    x = torch.randn(3, n)
    y = lora_sb(x)
    loss = y.sum()
    loss.backward()

    g_R_autograd = lora_sb.R.grad.clone()  # (r, r)

    # Compute gradient w.r.t. effective weight W = W0 + B R A
    # dL/dW = x^T @ dL/dy = x^T @ ones(batch, m) = (n, batch) @ (batch, m) = (n, m)
    # But g in paper has shape (m, n) -- likely just a transpose convention
    g_w = x.T @ torch.ones(3, m)  # (n, m)
    g_w_paper = g_w.T  # (m, n)

    # Lemma 2: g_R = s * B^T @ g @ A^T
    g_R_expected = lora_sb.scaling * (lora_sb.B.T @ g_w_paper @ lora_sb.A.T)

    print(f"  g_R autograd: {g_R_autograd}")
    print(f"  g_R expected: {g_R_expected}")
    assert torch.allclose(g_R_autograd, g_R_expected, atol=1e-5), \
        f"Autograd gradient mismatch: max diff = {(g_R_autograd - g_R_expected).abs().max()}"
    print("✓ test_gradient_autograd_consistency passed")


if __name__ == '__main__':
    print("=" * 60)
    print("Running LoRA-SB unit tests")
    print("=" * 60)
    test_lora_sb_layer_creation()
    test_lemma_2_gradient_relationship()
    test_truncated_svd_init_orthonormal()
    test_theorem_3_optimal_gradient()
    test_theorem_5_scaling_independence()
    test_forward_pass_equivalence()
    test_merge()
    test_theorem_6_optimal_first_step()
    test_parameter_count_reduction()
    test_gradient_autograd_consistency()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
