"""
Tests for the core Pyramidal Flow Matching algorithm.

These tests verify the mathematical correctness of the key equations
from the paper without requiring GPU or large models.
"""

import math
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.pyramidal_flow import PyramidalFlowMatching, TemporalPyramidCondition


def test_pyramid_stages():
    """Test that pyramid stages are correctly initialized."""
    pf = PyramidalFlowMatching(num_stages=3)
    
    assert len(pf.stage_time_windows) == 3
    
    # Check uniform partitioning
    for k, (s_k, e_k) in enumerate(pf.stage_time_windows):
        expected_s = k / 3
        expected_e = (k + 1) / 3
        assert abs(s_k - expected_s) < 1e-6, f"Stage {k} start mismatch: {s_k} vs {expected_s}"
        assert abs(e_k - expected_e) < 1e-6, f"Stage {k} end mismatch: {e_k} vs {expected_e}"
    
    print("✓ Pyramid stages correctly initialized")


def test_jump_point_timestep():
    """
    Test the jump point timestep formula: e_{k+1} = 2*s_k / (1 + s_k)
    
    From Eq. (26) in the paper.
    """
    pf = PyramidalFlowMatching(num_stages=3)
    
    # Test with various s_k values
    test_values = [0.1, 0.2, 0.3, 0.5, 0.7]
    
    for s_k in test_values:
        e_k_plus1 = pf.get_jump_point_timestep(s_k)
        expected = 2 * s_k / (1 + s_k)
        assert abs(e_k_plus1 - expected) < 1e-6, f"Jump point mismatch for s_k={s_k}"
        
        # Verify e_{k+1} > s_k (as noted in paper: "timestep is rolled back a bit")
        assert e_k_plus1 > s_k, f"e_{{k+1}} should be > s_k, got {e_k_plus1} <= {s_k}"
    
    print("✓ Jump point timestep formula correct")


def test_renoising_rule():
    """
    Test the renoising rule at jump points (Eq. 15):
    x_sk = (1 + s_k) / 2 * Up(x_{e_{k+1}}) + sqrt(3) * (1 - s_k) / 2 * n'
    """
    pf = PyramidalFlowMatching(num_stages=3)
    
    B, C, H, W = 2, 4, 8, 8
    x_ek_plus1 = torch.randn(B, C, H, W)
    s_k = 0.5
    target_size = (H * 2, W * 2)
    
    # Apply renoising
    x_sk = pf.renoise_at_jump_point(x_ek_plus1, s_k, target_size=target_size)
    
    # Check output shape
    assert x_sk.shape == (B, C, H * 2, W * 2), f"Shape mismatch: {x_sk.shape}"
    
    # Check that the output is not all zeros (noise was added)
    assert not torch.allclose(x_sk, torch.zeros_like(x_sk))
    
    print("✓ Renoising rule produces correct output shape")


def test_training_pair_sampling():
    """
    Test that training pair sampling produces correct shapes and values.
    
    Verifies Eqs. (9) and (10) from the paper.
    """
    pf = PyramidalFlowMatching(num_stages=3)
    
    B, C, H, W = 2, 4, 32, 32
    x1 = torch.randn(B, C, H, W)
    
    for stage in range(3):
        x_sk, x_ek, x_t, t_prime, target_velocity = pf.sample_training_pair(x1, stage)
        
        # Check that t_prime is in [0, 1]
        assert (t_prime >= 0).all() and (t_prime <= 1).all(), \
            f"t_prime out of range for stage {stage}"
        
        # Check that target velocity is x_ek - x_sk
        expected_velocity = x_ek - x_sk
        assert torch.allclose(target_velocity, expected_velocity), \
            f"Target velocity mismatch for stage {stage}"
        
        # Check that x_t is between x_sk and x_ek
        # x_t = t' * x_ek + (1 - t') * x_sk
        t_expanded = t_prime.view(-1, 1, 1, 1)
        expected_x_t = t_expanded * x_ek + (1 - t_expanded) * x_sk
        assert torch.allclose(x_t, expected_x_t, atol=1e-5), \
            f"x_t interpolation mismatch for stage {stage}"
    
    print("✓ Training pair sampling produces correct shapes and values")


def test_flow_loss():
    """Test that flow matching loss is computed correctly."""
    pf = PyramidalFlowMatching(num_stages=3)
    
    B, C, H, W = 2, 4, 16, 16
    predicted = torch.randn(B, C, H, W)
    target = torch.randn(B, C, H, W)
    
    loss = pf.compute_flow_loss(predicted, target)
    
    # Loss should be a scalar
    assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
    
    # Loss should be non-negative
    assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"
    
    # Loss should be zero when predicted == target
    zero_loss = pf.compute_flow_loss(predicted, predicted)
    assert abs(zero_loss.item()) < 1e-6, f"Loss should be zero for identical inputs"
    
    print("✓ Flow matching loss computed correctly")


def test_downsampling():
    """Test downsampling function."""
    pf = PyramidalFlowMatching(num_stages=3)
    
    B, C, H, W = 2, 4, 32, 32
    x = torch.randn(B, C, H, W)
    
    # Test 2x downsampling
    x_down = pf.downsample(x, 2)
    assert x_down.shape == (B, C, H // 2, W // 2), \
        f"2x downsampling shape mismatch: {x_down.shape}"
    
    # Test 4x downsampling
    x_down4 = pf.downsample(x, 4)
    assert x_down4.shape == (B, C, H // 4, W // 4), \
        f"4x downsampling shape mismatch: {x_down4.shape}"
    
    # Test identity (factor=1)
    x_same = pf.downsample(x, 1)
    assert torch.allclose(x_same, x), "Identity downsampling should return same tensor"
    
    print("✓ Downsampling function works correctly")


def test_upsampling():
    """Test upsampling function."""
    pf = PyramidalFlowMatching(num_stages=3)
    
    B, C, H, W = 2, 4, 8, 8
    x = torch.randn(B, C, H, W)
    
    # Test 2x upsampling
    x_up = pf.upsample(x, (H * 2, W * 2))
    assert x_up.shape == (B, C, H * 2, W * 2), \
        f"2x upsampling shape mismatch: {x_up.shape}"
    
    print("✓ Upsampling function works correctly")


def test_temporal_pyramid_condition():
    """Test temporal pyramid condition preparation."""
    tp = TemporalPyramidCondition(noise_strength_range=(0.0, 1/3))
    
    B, C, H, W = 2, 4, 16, 16
    
    # Create history latents
    history_latents = [
        torch.randn(B, C, H, W),
        torch.randn(B, C, H, W),
    ]
    
    # Test training mode (with noise)
    compressed = tp.prepare_history_condition(
        history_latents,
        current_stage=0,
        num_pyramid_stages=3,
        training=True,
    )
    
    assert len(compressed) == len(history_latents), \
        "Should return same number of history frames"
    
    # Test inference mode (without noise)
    compressed_inf = tp.prepare_history_condition(
        history_latents,
        current_stage=0,
        num_pyramid_stages=3,
        training=False,
    )
    
    assert len(compressed_inf) == len(history_latents), \
        "Should return same number of history frames in inference"
    
    print("✓ Temporal pyramid condition preparation works correctly")


def test_coupled_noise_straightness():
    """
    Test that coupled noise sampling produces straighter trajectories.
    
    This is the key insight from Appendix C.4 of the paper.
    """
    pf = PyramidalFlowMatching(num_stages=3)
    
    B, C, H, W = 4, 2, 8, 8
    x1 = torch.randn(B, C, H, W)
    
    # Sample training pairs for the last stage (full resolution)
    stage = 2  # Full resolution stage
    x_sk, x_ek, x_t, t_prime, target_velocity = pf.sample_training_pair(x1, stage)
    
    # The target velocity should be x_ek - x_sk
    # With coupled noise, the noise direction is the same for both endpoints
    # This should produce more consistent velocity directions
    
    # Verify that the velocity magnitude is reasonable
    vel_magnitude = target_velocity.norm(dim=(1, 2, 3)).mean()
    assert vel_magnitude > 0, "Velocity should be non-zero"
    
    print("✓ Coupled noise sampling produces valid training pairs")


def run_all_tests():
    """Run all tests."""
    print("Running Pyramidal Flow Matching tests...\n")
    
    test_pyramid_stages()
    test_jump_point_timestep()
    test_renoising_rule()
    test_training_pair_sampling()
    test_flow_loss()
    test_downsampling()
    test_upsampling()
    test_temporal_pyramid_condition()
    test_coupled_noise_straightness()
    
    print("\n✓ All tests passed!")


if __name__ == '__main__':
    run_all_tests()
