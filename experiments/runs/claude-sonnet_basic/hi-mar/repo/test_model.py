"""
Simple test to verify Hi-MAR model can be instantiated and forward pass works.
"""

import torch
from models.hi_mar import HiMAR_B, HiMAR_L, HiMAR_H


def test_hi_mar_b():
    """Test Hi-MAR-B model instantiation and forward pass."""
    print("Testing Hi-MAR-B...")

    model = HiMAR_B(
        img_size=256,
        low_res_img_size=128,
        patch_size=16,
        in_channels=16,
        num_classes=1000,
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {num_params / 1e6:.1f}M (expected ~244M)")

    # Test forward pass with small batch
    B = 2
    N_high = (256 // 16) ** 2  # 256 tokens
    N_low = (128 // 16) ** 2   # 64 tokens
    C = 16

    high_res_tokens = torch.randn(B, N_high, C)
    low_res_tokens = torch.randn(B, N_low, C)
    class_labels = torch.randint(0, 1000, (B,))

    loss, loss_dict = model(high_res_tokens, low_res_tokens, class_labels)
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Loss phase1: {loss_dict['loss_phase1']:.4f}")
    print(f"  Loss phase2: {loss_dict['loss_phase2']:.4f}")
    print("  Hi-MAR-B test passed!")


def test_hi_mar_generate():
    """Test Hi-MAR generation."""
    print("Testing Hi-MAR generation...")

    model = HiMAR_B(
        img_size=256,
        low_res_img_size=128,
        patch_size=16,
        in_channels=16,
        num_classes=1000,
    )
    model.eval()

    B = 2
    class_labels = torch.randint(0, 1000, (B,))

    with torch.no_grad():
        tokens = model.generate(
            class_labels,
            num_steps_phase1=2,  # Use few steps for testing
            num_steps_phase2=1,
            cfg_scale=1.5,
        )

    N_high = (256 // 16) ** 2
    assert tokens.shape == (B, N_high, 16), f"Expected shape {(B, N_high, 16)}, got {tokens.shape}"
    print(f"  Generated tokens shape: {tokens.shape}")
    print("  Hi-MAR generation test passed!")


if __name__ == '__main__':
    test_hi_mar_b()
    test_hi_mar_generate()
    print("\nAll tests passed!")
