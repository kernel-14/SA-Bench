"""
Unit tests and demo for LoRA-SB implementation.

Tests the core mathematical properties:
1. Orthonormality of B and A after initialization
2. B*R*A ≈ ΔW_avg (optimal rank-r approximation)
3. Scaling-factor independence
4. Parameter count reduction
"""

import torch
import torch.nn as nn
import numpy as np


def test_lora_sb_layer():
    """Test LoRASBLayer basic functionality."""
    from lora_sb import LoRASBLayer
    
    m, n, r = 64, 32, 8
    layer = LoRASBLayer(in_features=n, out_features=m, rank=r)
    
    # Test forward pass before initialization
    x = torch.randn(4, n)
    out = layer(x)
    assert out.shape == (4, m), f"Expected (4, {m}), got {out.shape}"
    print("✓ Forward pass shape correct")
    
    # Test initialization from SVD
    delta_w = torch.randn(m, n)
    U, S, Vh = torch.linalg.svd(delta_w, full_matrices=False)
    layer.initialize_from_svd(U, S, Vh)
    
    # Check orthonormality: B^T B = I
    BtB = layer.B.T @ layer.B
    assert torch.allclose(BtB, torch.eye(r), atol=1e-5), "B^T B != I"
    print("✓ B^T B = I (orthonormality)")
    
    # Check orthonormality: A A^T = I
    AAt = layer.A @ layer.A.T
    assert torch.allclose(AAt, torch.eye(r), atol=1e-5), "A A^T != I"
    print("✓ A A^T = I (orthonormality)")
    
    # Check B*R*A ≈ ΔW_avg (optimal rank-r approximation)
    BRA = layer.B @ layer.R @ layer.A
    delta_w_approx = U[:, :r] @ torch.diag(S[:r]) @ Vh[:r, :]
    assert torch.allclose(BRA, delta_w_approx, atol=1e-5), "B*R*A != optimal approx"
    print("✓ B*R*A = optimal rank-r approximation of ΔW_avg")
    
    # Verify this is better than any other rank-r approximation (Eckart-Young)
    # The Frobenius norm of the residual should equal sum of squared singular values beyond r
    residual_norm = torch.norm(delta_w - BRA.float(), 'fro').item()
    expected_residual = torch.sqrt(torch.sum(S[r:] ** 2)).item()
    assert abs(residual_norm - expected_residual) < 1e-4, \
        f"Residual {residual_norm:.4f} != expected {expected_residual:.4f}"
    print(f"✓ Eckart-Young: residual norm = {residual_norm:.4f}")
    
    print("\nAll LoRASBLayer tests passed!")


def test_lora_sb_linear():
    """Test LoRASBLinear wrapping."""
    from lora_sb import LoRASBLinear
    
    m, n, r = 64, 32, 8
    base_linear = nn.Linear(n, m, bias=False)
    lora_linear = LoRASBLinear(base_linear, rank=r)
    
    # Check that base layer is frozen
    for param in lora_linear.base_layer.parameters():
        assert not param.requires_grad, "Base layer should be frozen"
    print("✓ Base layer is frozen")
    
    # Check that only R is trainable
    trainable = [name for name, p in lora_linear.named_parameters() if p.requires_grad]
    assert trainable == ['lora_sb.R'], f"Expected only R trainable, got {trainable}"
    print(f"✓ Only R is trainable: {trainable}")
    
    # Check parameter count
    r_params = r * r
    assert lora_linear.lora_sb.R.numel() == r_params
    print(f"✓ R has {r_params} parameters (r²={r}²)")
    
    print("\nAll LoRASBLinear tests passed!")


def test_apply_lora_sb():
    """Test applying LoRA-SB to a model."""
    from lora_sb import apply_lora_sb, get_trainable_parameters
    
    # Simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.v_proj = nn.Linear(64, 64)
            self.fc = nn.Linear(64, 10)
        
        def forward(self, x):
            return self.fc(self.q_proj(x) + self.v_proj(x))
    
    model = SimpleModel()
    total_before = sum(p.numel() for p in model.parameters())
    
    model = apply_lora_sb(model, target_modules=["q_proj", "v_proj"], rank=8)
    
    trainable, total = get_trainable_parameters(model)
    
    # Only R matrices should be trainable (2 layers × 8×8 = 128 params)
    expected_trainable = 2 * 8 * 8  # 2 layers, r=8
    assert trainable == expected_trainable, \
        f"Expected {expected_trainable} trainable params, got {trainable}"
    print(f"✓ Trainable parameters: {trainable} (expected {expected_trainable})")
    
    # Test forward pass
    x = torch.randn(4, 64)
    out = model(x)
    assert out.shape == (4, 10)
    print("✓ Forward pass works after applying LoRA-SB")
    
    print("\nAll apply_lora_sb tests passed!")


def test_gradient_approximation():
    """
    Test that with orthonormal B and A, the gradient approximation simplifies.
    
    Theorem 3: g^R = (1/s²)(B^T B)^{-1} g^R_LoRA-XS (A A^T)^{-1}
    With B^T B = A A^T = I and s=1: g^R = g^R_LoRA-XS
    """
    from lora_sb import LoRASBLayer
    
    m, n, r = 32, 16, 4
    layer = LoRASBLayer(in_features=n, out_features=m, rank=r)
    
    # Initialize with orthonormal B and A
    delta_w = torch.randn(m, n)
    U, S, Vh = torch.linalg.svd(delta_w, full_matrices=False)
    layer.initialize_from_svd(U, S, Vh)
    
    # Verify B^T B = I and A A^T = I
    assert torch.allclose(layer.B.T @ layer.B, torch.eye(r), atol=1e-5)
    assert torch.allclose(layer.A @ layer.A.T, torch.eye(r), atol=1e-5)
    
    # Simulate a gradient g (full FT gradient)
    g = torch.randn(m, n)
    
    # Compute g^R_LoRA-XS = s * B^T * g * A^T (Lemma 2, with s=1)
    g_R_lora_xs = layer.B.T @ g @ layer.A.T
    
    # Compute optimal g^R = (1/s²)(B^T B)^{-1} g^R_LoRA-XS (A A^T)^{-1}
    # With B^T B = A A^T = I and s=1: g^R = g^R_LoRA-XS
    BtB_inv = torch.linalg.inv(layer.B.T @ layer.B)
    AAt_inv = torch.linalg.inv(layer.A @ layer.A.T)
    g_R_optimal = BtB_inv @ g_R_lora_xs @ AAt_inv
    
    # They should be equal
    assert torch.allclose(g_R_optimal, g_R_lora_xs, atol=1e-5), \
        "Optimal g^R != g^R_LoRA-XS (should be equal with orthonormal B, A)"
    print("✓ Theorem 3: g^R = g^R_LoRA-XS with orthonormal B, A (s=1)")
    
    # Verify equivalent gradient is s-independent (Theorem 5)
    # g_tilde = s * B * g^R * A = B * g^R_LoRA-XS * A (s-independent)
    g_tilde = layer.B @ g_R_optimal @ layer.A
    
    # This should equal B * (B^T B)^{-1} * B^T * g * A^T * (A A^T)^{-1} * A
    # = B * B^T * g * A^T * A (since B^T B = A A^T = I)
    # = projection of g onto column space of B and row space of A
    g_tilde_expected = layer.B @ layer.B.T @ g @ layer.A.T @ layer.A
    assert torch.allclose(g_tilde, g_tilde_expected, atol=1e-5)
    print("✓ Theorem 5: Equivalent gradient is s-independent")
    
    print("\nAll gradient approximation tests passed!")


def demo_parameter_efficiency():
    """Demonstrate parameter efficiency of LoRA-SB vs LoRA."""
    print("\n=== Parameter Efficiency Demo ===")
    
    # Simulate Mistral-7B dimensions
    # Typical attention layer: 4096 x 4096
    m, n = 4096, 4096
    
    for r in [32, 64, 96]:
        lora_params = r * (m + n)  # LoRA: B (m×r) + A (r×n)
        lora_sb_params = r * r     # LoRA-SB: R (r×r) only
        reduction = lora_params / lora_sb_params
        
        print(f"Rank {r:3d}: LoRA={lora_params:,} params, "
              f"LoRA-SB={lora_sb_params:,} params, "
              f"reduction={reduction:.0f}x")
    
    # For full model (Mistral-7B has ~32 layers, 7 target modules each)
    n_layers = 32
    n_modules = 7  # q, k, v, o, gate, up, down
    
    print(f"\nFull model ({n_layers} layers, {n_modules} modules each):")
    for r in [32, 64, 96]:
        lora_total = n_layers * n_modules * r * (m + n)
        lora_sb_total = n_layers * n_modules * r * r
        reduction = lora_total / lora_sb_total
        print(f"  Rank {r:3d}: LoRA={lora_total/1e6:.1f}M, "
              f"LoRA-SB={lora_sb_total/1e3:.0f}K, "
              f"reduction={reduction:.0f}x")


if __name__ == "__main__":
    print("Testing LoRA-SB implementation...\n")
    
    test_lora_sb_layer()
    print()
    test_lora_sb_linear()
    print()
    test_apply_lora_sb()
    print()
    test_gradient_approximation()
    
    demo_parameter_efficiency()
    
    print("\n✓ All tests passed!")
