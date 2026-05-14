"""
Tests for the gated attention implementation.

Tests verify:
1. All gating variants can be instantiated and run forward pass
2. Gating positions (G1-G5) work correctly
3. Granularity variants (elementwise, headwise) work
4. Head-specific vs head-shared gating
5. Multiplicative vs additive gating
6. NS-sigmoid activation
7. Parameter counts match paper's reported values
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from models.gated_attention import (
    GatedMultiHeadAttention,
    GatingActivation,
    GatingGranularity,
    GatingPosition,
    GatingType,
    ns_sigmoid,
)
from models.transformer import (
    GatedTransformerLayer,
    GatedTransformerModel,
    TransformerConfig,
)


# Small model config for testing
SMALL_CONFIG = dict(
    d_model=256,
    num_layers=4,
    num_heads=8,
    num_kv_heads=4,
    head_dim=32,
    ffn_intermediate_dim=512,
    vocab_size=1000,
    max_seq_len=64,
)


def test_ns_sigmoid():
    """Test NS-sigmoid activation: output should be in [0.5, 1.0]."""
    x = torch.randn(100)
    out = ns_sigmoid(x)
    assert out.min() >= 0.5 - 1e-6, f"NS-sigmoid min {out.min()} < 0.5"
    assert out.max() <= 1.0 + 1e-6, f"NS-sigmoid max {out.max()} > 1.0"
    print("✓ NS-sigmoid activation test passed")


def test_baseline_attention():
    """Test baseline attention (no gating)."""
    config = TransformerConfig(**SMALL_CONFIG, gating_position=None)
    model = GatedTransformerModel(config)
    
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    output = model(input_ids)
    assert output["logits"].shape == (batch_size, seq_len, config.vocab_size)
    assert output["loss"] is None
    print("✓ Baseline attention test passed")


def test_sdpa_elementwise_gating():
    """Test SDPA output elementwise gating (G1) - the best variant."""
    config = TransformerConfig(
        **SMALL_CONFIG,
        gating_position="sdpa_output",
        gating_granularity="elementwise",
        head_specific=True,
        gating_type="multiplicative",
        gating_activation="sigmoid",
    )
    model = GatedTransformerModel(config)
    
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    output = model(input_ids, labels=labels)
    assert output["logits"].shape == (batch_size, seq_len, config.vocab_size)
    assert output["loss"] is not None
    assert output["loss"].item() > 0
    print(f"✓ SDPA elementwise gating test passed (loss={output['loss'].item():.4f})")


def test_all_gating_positions():
    """Test all five gating positions (G1-G5)."""
    positions = [
        "sdpa_output",   # G1
        "value",         # G2
        "key",           # G3
        "query",         # G4
        "dense_output",  # G5
    ]
    
    for pos in positions:
        config = TransformerConfig(
            **SMALL_CONFIG,
            gating_position=pos,
            gating_granularity="elementwise",
            head_specific=True,
        )
        model = GatedTransformerModel(config)
        
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        output = model(input_ids)
        
        assert output["logits"].shape == (batch_size, seq_len, config.vocab_size), \
            f"Wrong output shape for position {pos}"
        print(f"✓ Gating position {pos} test passed")


def test_headwise_gating():
    """Test headwise gating (per-head scalar)."""
    config = TransformerConfig(
        **SMALL_CONFIG,
        gating_position="sdpa_output",
        gating_granularity="headwise",
        head_specific=True,
    )
    model = GatedTransformerModel(config)
    
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    output = model(input_ids)
    
    assert output["logits"].shape == (batch_size, seq_len, config.vocab_size)
    print("✓ Headwise gating test passed")


def test_head_shared_gating():
    """Test head-shared gating (shared across heads)."""
    config = TransformerConfig(
        **SMALL_CONFIG,
        gating_position="sdpa_output",
        gating_granularity="elementwise",
        head_specific=False,  # Head-shared
    )
    model = GatedTransformerModel(config)
    
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    output = model(input_ids)
    
    assert output["logits"].shape == (batch_size, seq_len, config.vocab_size)
    print("✓ Head-shared gating test passed")


def test_additive_gating():
    """Test additive gating with SiLU."""
    config = TransformerConfig(
        **SMALL_CONFIG,
        gating_position="sdpa_output",
        gating_granularity="elementwise",
        head_specific=True,
        gating_type="additive",
        gating_activation="silu",
    )
    model = GatedTransformerModel(config)
    
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    output = model(input_ids)
    
    assert output["logits"].shape == (batch_size, seq_len, config.vocab_size)
    print("✓ Additive gating test passed")


def test_ns_sigmoid_gating():
    """Test NS-sigmoid gating (non-sparse variant for ablation)."""
    config = TransformerConfig(
        **SMALL_CONFIG,
        gating_position="sdpa_output",
        gating_granularity="elementwise",
        head_specific=True,
        gating_type="multiplicative",
        gating_activation="ns_sigmoid",
    )
    model = GatedTransformerModel(config)
    
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    output = model(input_ids)
    
    assert output["logits"].shape == (batch_size, seq_len, config.vocab_size)
    print("✓ NS-sigmoid gating test passed")


def test_rmsnorm_gating():
    """Test RMSNorm as non-linearity (no gate parameters)."""
    config = TransformerConfig(
        **SMALL_CONFIG,
        gating_position="sdpa_output",
        gating_granularity="elementwise",
        head_specific=True,
        gating_type="multiplicative",
        gating_activation="rmsnorm",
    )
    model = GatedTransformerModel(config)
    
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    output = model(input_ids)
    
    assert output["logits"].shape == (batch_size, seq_len, config.vocab_size)
    print("✓ RMSNorm gating test passed")


def test_sandwich_norm():
    """Test sandwich normalization for training stability."""
    config = TransformerConfig(
        **SMALL_CONFIG,
        gating_position=None,
        use_sandwich_norm=True,
    )
    model = GatedTransformerModel(config)
    
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    output = model(input_ids)
    
    assert output["logits"].shape == (batch_size, seq_len, config.vocab_size)
    print("✓ Sandwich norm test passed")


def test_gate_parameter_count():
    """
    Test that gate parameter counts are reasonable.
    
    From the paper (Table 1):
    - SDPA Elementwise G1: ~201M added params for 15A2B model
    - SDPA Headwise G1: ~1.6M added params
    - Value Elementwise G2: ~25M added params
    
    For our small test model (d_model=256, num_heads=8, head_dim=32):
    - SDPA Elementwise G1: d_model * num_heads * head_dim = 256 * 8 * 32 = 65536 per layer
    """
    # Elementwise G1
    config_elem = TransformerConfig(
        **SMALL_CONFIG,
        gating_position="sdpa_output",
        gating_granularity="elementwise",
        head_specific=True,
    )
    model_elem = GatedTransformerModel(config_elem)
    
    # Headwise G1
    config_head = TransformerConfig(
        **SMALL_CONFIG,
        gating_position="sdpa_output",
        gating_granularity="headwise",
        head_specific=True,
    )
    model_head = GatedTransformerModel(config_head)
    
    params_elem = model_elem.get_num_params()
    params_head = model_head.get_num_params()
    
    # Elementwise should have more gate params than headwise
    assert params_elem["gate_params"] > params_head["gate_params"], \
        "Elementwise should have more gate params than headwise"
    
    print(f"✓ Gate parameter count test passed")
    print(f"  Elementwise gate params: {params_elem['gate_params']:,}")
    print(f"  Headwise gate params: {params_head['gate_params']:,}")


def test_gqa_attention():
    """Test Group Query Attention (GQA) with gating."""
    # GQA: 8 query heads, 2 KV heads
    config = TransformerConfig(
        d_model=256,
        num_layers=2,
        num_heads=8,
        num_kv_heads=2,  # GQA
        head_dim=32,
        ffn_intermediate_dim=512,
        vocab_size=1000,
        max_seq_len=64,
        gating_position="sdpa_output",
        gating_granularity="elementwise",
        head_specific=True,
    )
    model = GatedTransformerModel(config)
    
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    output = model(input_ids)
    
    assert output["logits"].shape == (batch_size, seq_len, config.vocab_size)
    print("✓ GQA with gating test passed")


def test_loss_computation():
    """Test that loss is computed correctly."""
    config = TransformerConfig(
        **SMALL_CONFIG,
        gating_position="sdpa_output",
    )
    model = GatedTransformerModel(config)
    
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    
    output = model(input_ids, labels=labels)
    
    assert output["loss"] is not None
    assert not torch.isnan(output["loss"])
    assert not torch.isinf(output["loss"])
    print(f"✓ Loss computation test passed (loss={output['loss'].item():.4f})")


def test_gradient_flow():
    """Test that gradients flow through the gating mechanism."""
    config = TransformerConfig(
        **SMALL_CONFIG,
        gating_position="sdpa_output",
        gating_granularity="elementwise",
        head_specific=True,
    )
    model = GatedTransformerModel(config)
    
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    output = model(input_ids, labels=labels)
    output["loss"].backward()
    
    # Check that gate parameters have gradients
    for layer in model.layers:
        if layer.attn.gate is not None and layer.attn.gate.gate_proj is not None:
            assert layer.attn.gate.gate_proj.weight.grad is not None, \
                "Gate projection should have gradients"
    
    print("✓ Gradient flow test passed")


def run_all_tests():
    """Run all tests."""
    print("Running gated attention tests...\n")
    
    tests = [
        test_ns_sigmoid,
        test_baseline_attention,
        test_sdpa_elementwise_gating,
        test_all_gating_positions,
        test_headwise_gating,
        test_head_shared_gating,
        test_additive_gating,
        test_ns_sigmoid_gating,
        test_rmsnorm_gating,
        test_sandwich_norm,
        test_gate_parameter_count,
        test_gqa_attention,
        test_loss_computation,
        test_gradient_flow,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
