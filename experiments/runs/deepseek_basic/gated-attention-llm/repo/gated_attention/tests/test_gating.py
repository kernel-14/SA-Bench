"""
Tests for the gated attention module.

Validates the correctness of all gating variants described in the paper.
"""

import torch

from gated_attention.modules.gating import (
    GatedAttention,
    GatedAttentionConfig,
    GatingPosition,
    GatingGranularity,
    GatingMode,
    GatingScope,
    ActivationType,
    create_gated_attention,
)


def test_gating_positions():
    """Test all five gating positions (G1-G5)."""
    batch, seq_len, d_model = 2, 16, 128
    x = torch.randn(batch, seq_len, d_model)

    for pos in GatingPosition:
        config = GatedAttentionConfig(
            position=pos,
            d_model=d_model,
            num_heads=4,
            num_kv_heads=2,
            head_dim=32,
            max_seq_len=seq_len,
        )
        attn = GatedAttention(config)
        output, attn_weights, _ = attn(x)
        assert output.shape == (batch, seq_len, d_model), f"Failed for position {pos}"


def test_gating_granularity():
    """Test headwise vs elementwise gating."""
    batch, seq_len, d_model = 2, 16, 128
    x = torch.randn(batch, seq_len, d_model)

    for granularity in [GatingGranularity.HEADWISE, GatingGranularity.ELEMENTWISE]:
        config = GatedAttentionConfig(
            position=GatingPosition.G1_SDPA_OUTPUT,
            granularity=granularity,
            d_model=d_model,
            num_heads=4,
            num_kv_heads=2,
            head_dim=32,
            max_seq_len=seq_len,
        )
        attn = GatedAttention(config)
        output, _, _ = attn(x)
        assert output.shape == (batch, seq_len, d_model)


def test_gating_modes():
    """Test multiplicative vs additive gating."""
    batch, seq_len, d_model = 2, 16, 128
    x = torch.randn(batch, seq_len, d_model)

    for mode in [GatingMode.MULTIPLICATIVE, GatingMode.ADDITIVE]:
        activation = ActivationType.SILU if mode == GatingMode.ADDITIVE else ActivationType.SIGMOID
        config = GatedAttentionConfig(
            position=GatingPosition.G1_SDPA_OUTPUT,
            mode=mode,
            activation=activation,
            d_model=d_model,
            num_heads=4,
            num_kv_heads=2,
            head_dim=32,
            max_seq_len=seq_len,
        )
        attn = GatedAttention(config)
        output, _, _ = attn(x)
        assert output.shape == (batch, seq_len, d_model)


def test_head_specific_vs_shared():
    """Test head-specific vs head-shared gating."""
    batch, seq_len, d_model = 2, 16, 128
    x = torch.randn(batch, seq_len, d_model)

    for scope in [GatingScope.HEAD_SPECIFIC, GatingScope.HEAD_SHARED]:
        config = GatedAttentionConfig(
            position=GatingPosition.G1_SDPA_OUTPUT,
            scope=scope,
            d_model=d_model,
            num_heads=4,
            num_kv_heads=2,
            head_dim=32,
            max_seq_len=seq_len,
        )
        attn = GatedAttention(config)
        output, _, _ = attn(x)
        assert output.shape == (batch, seq_len, d_model)


def test_activation_functions():
    """Test all activation functions: sigmoid, SiLU, NS-sigmoid, identity."""
    batch, seq_len, d_model = 2, 16, 128
    x = torch.randn(batch, seq_len, d_model)

    for act in [ActivationType.SIGMOID, ActivationType.SILU,
                ActivationType.NS_SIGMOID, ActivationType.IDENTITY]:
        config = GatedAttentionConfig(
            position=GatingPosition.G1_SDPA_OUTPUT,
            activation=act,
            d_model=d_model,
            num_heads=4,
            num_kv_heads=2,
            head_dim=32,
            max_seq_len=seq_len,
        )
        attn = GatedAttention(config)
        output, _, _ = attn(x)
        assert output.shape == (batch, seq_len, d_model)


def test_factory_function():
    """Test the create_gated_attention factory function."""
    attn = create_gated_attention(
        d_model=128,
        num_heads=4,
        num_kv_heads=2,
        head_dim=32,
        position="g1",
        granularity="elementwise",
        mode="multiplicative",
        scope="head_specific",
        activation="sigmoid",
    )
    x = torch.randn(2, 16, 128)
    output, _, _ = attn(x)
    assert output.shape == (2, 16, 128)


def test_all_paper_variants():
    """Test all paper variants from Table 1."""
    batch, seq_len, d_model = 2, 16, 2048

    variants = [
        ("g1", "elementwise", "multiplicative", "head_specific", "sigmoid"),
        ("g1", "headwise", "multiplicative", "head_specific", "sigmoid"),
        ("g1", "elementwise", "multiplicative", "head_shared", "sigmoid"),
        ("g1", "elementwise", "additive", "head_specific", "silu"),
        ("g1", "elementwise", "multiplicative", "head_specific", "silu"),
        ("g2", "elementwise", "multiplicative", "head_specific", "sigmoid"),
        ("g2", "headwise", "multiplicative", "head_specific", "sigmoid"),
        ("g3", "elementwise", "multiplicative", "head_specific", "sigmoid"),
        ("g4", "elementwise", "multiplicative", "head_specific", "sigmoid"),
        ("g5", "elementwise", "multiplicative", "head_specific", "sigmoid"),
        ("g1", "elementwise", "multiplicative", "head_specific", "ns_sigmoid"),
    ]

    x = torch.randn(batch, seq_len, d_model)

    for pos, gran, mode, scope, act in variants:
        attn = create_gated_attention(
            d_model=d_model,
            num_heads=32,
            num_kv_heads=4,
            head_dim=128,
            position=pos,
            granularity=gran,
            mode=mode,
            scope=scope,
            activation=act,
            max_seq_len=seq_len,
        )
        output, _, _ = attn(x)
        assert output.shape == (batch, seq_len, d_model), \
            f"Failed for variant: {pos}, {gran}, {mode}, {scope}, {act}"


def test_attention_sink_computation():
    """Test attention sink ratio computation."""
    from gated_attention.analysis.attention_analysis import compute_attention_sink_ratio

    attn = torch.zeros(2, 4, 8, 8)
    attn[:, :, :, 0] = 0.5
    attn[:, :, :, 1:] = 0.5 / 7

    sink_ratio = compute_attention_sink_ratio(attn)
    assert abs(sink_ratio - 0.5) < 0.01, f"Expected 0.5, got {sink_ratio}"


def test_gate_score_statistics():
    """Test gate score statistics computation."""
    from gated_attention.analysis.attention_analysis import compute_gate_score_statistics

    gate_scores = torch.sigmoid(torch.randn(100) - 2.0)
    stats = compute_gate_score_statistics(gate_scores)

    assert "mean" in stats
    assert "std" in stats
    assert "sparsity_0_5" in stats
    assert 0 <= stats["mean"] <= 1.0


if __name__ == "__main__":
    test_gating_positions()
    test_gating_granularity()
    test_gating_modes()
    test_head_specific_vs_shared()
    test_activation_functions()
    test_factory_function()
    test_all_paper_variants()
    test_attention_sink_computation()
    test_gate_score_statistics()
    print("All tests passed!")
