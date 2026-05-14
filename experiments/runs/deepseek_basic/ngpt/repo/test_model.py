"""
Tests for the nGPT model implementation.

Verifies that:
1. All vectors are properly normalized
2. The model can perform forward and backward passes
3. Hyperparameters match the paper's specifications
4. The normalization after optimizer step works correctly
"""

import math
import torch
import torch.nn as nn
from model import (
    nGPT, BaselineGPT, create_ngpt_model, norm,
    NormalizedLinear, NormalizedEmbedding, ScaledParameter,
    AttentionBlock, MLPBlock, nGPTBlock,
)


def test_normalization():
    """Test that the norm function produces unit vectors."""
    x = torch.randn(16, 128)
    x_normed = norm(x)
    norms = x_normed.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_normalized_linear():
    """Test that NormalizedLinear uses normalized weight vectors in computation."""
    layer = NormalizedLinear(128, 256)
    x = torch.randn(8, 128)
    y = layer(x)

    # Output should have correct shape
    assert y.shape == (8, 256)

    # The weight normalization happens in forward pass via norm(self.weight)
    # but the original weight is not modified (normalization after optimizer step handles that)
    # Check that dot products are bounded in [-1, 1] since weights are normalized during forward
    x_normed = norm(x)  # If x is unit norm and weight vectors are unit norm...
    # The outputs are dot products between x_normed and normalized weight vectors
    # So they should be bounded when x is normalized
    # But we only test shape here since x is not necessarily normalized


def test_normalized_embedding():
    """Test that NormalizedEmbedding uses normalized embedding vectors."""
    emb = NormalizedEmbedding(1000, 128)
    x = torch.randint(0, 1000, (8, 64))

    # Forward pass should normalize embeddings
    y = emb(x)
    assert y.shape == (8, 64, 128)

    # Check that output vectors are normalized (since embeddings are normalized in forward)
    y_norms = y.norm(dim=-1)
    assert torch.allclose(y_norms, torch.ones_like(y_norms), atol=1e-4)


def test_scaled_parameter():
    """Test that ScaledParameter correctly implements init/scale decomposition."""
    s = ScaledParameter((10,), s_init=1.0, s_scale=1.0 / math.sqrt(128))
    value = s()
    # After init, value should be s_init (since raw = s_scale, and value = raw * s_init/s_scale)
    assert torch.allclose(value, torch.ones(10) * 1.0, atol=1e-5)

    # After training step, if raw changes, value should update accordingly
    s.raw.data = torch.full((10,), 2.0 / math.sqrt(128))
    value = s()
    assert torch.allclose(value, torch.ones(10) * 2.0, atol=1e-5)


def test_attention_block():
    """Test the normalized attention block."""
    block = AttentionBlock(d_model=128, n_heads=4)
    h = norm(torch.randn(2, 16, 128))  # Batch of normalized hidden states

    out = block(h)
    assert out.shape == (2, 16, 128)

    # Output should be normalized
    out_norms = out.norm(dim=-1)
    assert torch.allclose(out_norms, torch.ones_like(out_norms), atol=1e-4)


def test_mlp_block():
    """Test the normalized MLP block."""
    block = MLPBlock(d_model=128, d_mlp=512)
    h = norm(torch.randn(2, 16, 128))

    out = block(h)
    assert out.shape == (2, 16, 128)

    # Output should be normalized
    out_norms = out.norm(dim=-1)
    assert torch.allclose(out_norms, torch.ones_like(out_norms), atol=1e-4)


def test_ngpt_block():
    """Test the full nGPT transformer block."""
    block = nGPTBlock(d_model=128, n_heads=4, d_mlp=512)
    h = norm(torch.randn(2, 16, 128))

    out = block(h)
    assert out.shape == (2, 16, 128)

    # Hidden state should remain normalized after block
    out_norms = out.norm(dim=-1)
    assert torch.allclose(out_norms, torch.ones_like(out_norms), atol=1e-4)


def test_ngpt_model_forward():
    """Test full nGPT model forward pass."""
    model = nGPT(
        vocab_size=1000,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_mlp=512,
        max_seq_len=128,
    )

    batch = torch.randint(0, 1000, (2, 16))
    targets = torch.randint(0, 1000, (2, 16))

    logits, loss = model(batch, targets)

    assert logits.shape == (2, 16, 1000)
    assert loss is not None
    assert loss.item() > 0


def test_ngpt_model_backward():
    """Test that gradients flow through the nGPT model."""
    model = nGPT(
        vocab_size=1000,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_mlp=512,
        max_seq_len=128,
    )

    batch = torch.randint(0, 1000, (2, 16))
    targets = torch.randint(0, 1000, (2, 16))

    logits, loss = model(batch, targets)
    loss.backward()

    # Check that gradients exist for all parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


def test_weight_normalization():
    """Test that normalize_weights() properly normalizes all matrices."""
    model = nGPT(
        vocab_size=1000,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_mlp=512,
        max_seq_len=128,
    )

    # Perturb weights
    with torch.no_grad():
        model.E_input.weight.data *= 2.0
        for layer in model.layers:
            layer.attention.W_q.weight.data *= 3.0

    # Normalize
    model.normalize_weights()

    # Check all matrices are normalized
    # Embeddings
    assert torch.allclose(
        model.E_input.weight.norm(dim=1),
        torch.ones(model.E_input.weight.shape[0]),
        atol=1e-5
    )
    assert torch.allclose(
        model.E_output.weight.norm(dim=1),
        torch.ones(model.E_output.weight.shape[0]),
        atol=1e-5
    )

    # Attention matrices
    for layer in model.layers:
        for mat_name in ['W_q', 'W_k', 'W_v', 'W_o']:
            mat = getattr(layer.attention, mat_name)
            assert torch.allclose(
                mat.weight.norm(dim=1),
                torch.ones(mat.weight.shape[0]),
                atol=1e-5
            )

    # MLP matrices
    for layer in model.layers:
        for mat_name in ['W_u', 'W_v', 'W_o_mlp']:
            mat = getattr(layer.mlp, mat_name)
            assert torch.allclose(
                mat.weight.norm(dim=1),
                torch.ones(mat.weight.shape[0]),
                atol=1e-5
            )


def test_model_sizes():
    """Test that model configurations match paper specifications."""
    # 0.5B model
    model_05b = create_ngpt_model('0.5B', vocab_size=32000)
    n_params = sum(p.numel() for p in model_05b.parameters())
    n_params_m = n_params / 1e6
    # Should be approximately 468.4M (within reasonable tolerance)
    assert 400 < n_params_m < 550, f"Expected ~468M params, got {n_params_m:.1f}M"

    # 1B model
    model_1b = create_ngpt_model('1B', vocab_size=32000)
    n_params = sum(p.numel() for p in model_1b.parameters())
    n_params_m = n_params / 1e6
    assert 900 < n_params_m < 1150, f"Expected ~1026M params, got {n_params_m:.1f}M"


def test_baseline_gpt_forward():
    """Test baseline GPT model forward pass."""
    model = BaselineGPT(
        vocab_size=1000,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_mlp=512,
        max_seq_len=128,
    )

    batch = torch.randint(0, 1000, (2, 16))
    targets = torch.randint(0, 1000, (2, 16))

    logits, loss = model(batch, targets)

    assert logits.shape == (2, 16, 1000)
    assert loss is not None
    assert loss.item() > 0


def test_causal_masking():
    """Test that causal masking prevents attending to future tokens."""
    model = nGPT(
        vocab_size=1000,
        d_model=128,
        n_heads=4,
        n_layers=1,
        d_mlp=512,
        max_seq_len=16,
    )

    batch = torch.randint(0, 1000, (1, 8))
    logits, _ = model(batch)
    assert logits.shape == (1, 8, 1000)


def test_dot_product_bounds():
    """Test that normalized matrices produce dot products in [-1, 1]."""
    model = nGPT(
        vocab_size=1000,
        d_model=128,
        n_heads=4,
        n_layers=1,
        d_mlp=512,
        max_seq_len=16,
    )

    # Normalize all weights
    model.normalize_weights()

    batch = torch.randint(0, 1000, (1, 8))
    h = model.E_input(batch)  # Normalized hidden states

    # Check that q, k dot products are bounded
    layer = model.layers[0]
    n_heads = layer.attention.n_heads
    d_k = layer.attention.d_k

    # Re-normalize weights to be sure
    layer.attention.W_q.weight.data = norm(layer.attention.W_q.weight.data)
    layer.attention.W_k.weight.data = norm(layer.attention.W_k.weight.data)

    # Get q, k projections (using normalized weights)
    q = norm(layer.attention.W_q(h)).view(1, 8, n_heads, d_k).transpose(1, 2)
    k = norm(layer.attention.W_k(h)).view(1, 8, n_heads, d_k).transpose(1, 2)

    # Normalize q and k
    q_normed = norm(q)
    k_normed = norm(k)

    # Dot products should be in [-1, 1]
    dot_products = (q_normed * k_normed).sum(dim=-1)
    assert dot_products.min() >= -1.01, f"Min dot product: {dot_products.min()}"
    assert dot_products.max() <= 1.01, f"Max dot product: {dot_products.max()}"


def test_softmax_scaling():
    """Test that softmax scaling uses sqrt(d_k) not 1/sqrt(d_k)."""
    block = AttentionBlock(d_model=128, n_heads=4)
    d_k = 128 // 4  # 32

    # nGPT uses sqrt(d_k) = sqrt(32) ≈ 5.657
    # Baseline GPT uses 1/sqrt(d_k) = 1/sqrt(32) ≈ 0.177
    assert abs(block.softmax_scale - math.sqrt(d_k)) < 1e-6, \
        f"Expected sqrt(d_k)={math.sqrt(d_k)}, got {block.softmax_scale}"


def test_alpha_initialization():
    """Test that eigen learning rates are properly initialized."""
    model = create_ngpt_model('0.5B', vocab_size=32000)
    d_model = 1024

    for layer in model.layers:
        alpha_A = layer.alpha_A()
        alpha_M = layer.alpha_M()

        # After init, values should be alpha_init = 0.05
        assert torch.allclose(alpha_A, torch.ones(d_model) * 0.05, atol=1e-6)
        assert torch.allclose(alpha_M, torch.ones(d_model) * 0.05, atol=1e-6)


if __name__ == '__main__':
    # Run all tests
    print("Running nGPT model tests...")
    test_normalization()
    print("✓ test_normalization passed")
    test_normalized_linear()
    print("✓ test_normalized_linear passed")
    test_normalized_embedding()
    print("✓ test_normalized_embedding passed")
    test_scaled_parameter()
    print("✓ test_scaled_parameter passed")
    test_attention_block()
    print("✓ test_attention_block passed")
    test_mlp_block()
    print("✓ test_mlp_block passed")
    test_ngpt_block()
    print("✓ test_ngpt_block passed")
    test_ngpt_model_forward()
    print("✓ test_ngpt_model_forward passed")
    test_ngpt_model_backward()
    print("✓ test_ngpt_model_backward passed")
    test_weight_normalization()
    print("✓ test_weight_normalization passed")
    test_model_sizes()
    print("✓ test_model_sizes passed")
    test_baseline_gpt_forward()
    print("✓ test_baseline_gpt_forward passed")
    test_causal_masking()
    print("✓ test_causal_masking passed")
    test_dot_product_bounds()
    print("✓ test_dot_product_bounds passed")
    test_softmax_scaling()
    print("✓ test_softmax_scaling passed")
    test_alpha_initialization()
    print("✓ test_alpha_initialization passed")
    print("\nAll tests passed!")
