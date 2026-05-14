"""
Unit tests for Ca2-VDM model components.

Tests:
  - CausalTemporalAttention: Verify causal masking and KV-cache
  - PrefixEnhancedSpatialAttention: Verify prefix enhancement
  - KVCacheQueue: Verify queue operations
  - TemporalPositionalEmbedding: Verify Cyclic-TPE
  - Ca2VDMTransformer: Verify forward pass shapes
  - Ca2VDM: Verify training loss computation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from ca2_vdm.models.attention import CausalTemporalAttention, PrefixEnhancedSpatialAttention
from ca2_vdm.models.kv_cache import KVCacheQueue, SpatialKVCache
from ca2_vdm.models.positional_embedding import TemporalPositionalEmbedding, SpatialPositionalEmbedding
from ca2_vdm.models.transformer import Ca2VDMTransformer, Ca2VDMBlock
from ca2_vdm.models.diffusion import Ca2VDM


def test_causal_temporal_attention_shape():
    """Test output shape matches input."""
    dim = 64
    num_heads = 4
    B, L, C = 2, 8, dim
    attn = CausalTemporalAttention(dim, num_heads)
    x = torch.randn(B, L, C)
    out, kv = attn(x)
    assert out.shape == (B, L, C), f"Expected {(B, L, C)}, got {out.shape}"
    assert kv is None


def test_causal_temporal_attention_return_kv():
    """Test KV return for cache writing."""
    dim = 64
    num_heads = 4
    B, L, C = 2, 8, dim
    attn = CausalTemporalAttention(dim, num_heads)
    x = torch.randn(B, L, C)
    out, kv = attn(x, return_kv=True)
    assert kv is not None
    K, V = kv
    assert K.shape == (B, L, C)
    assert V.shape == (B, L, C)


def test_causal_temporal_attention_kv_cache():
    """Test inference with KV-cache."""
    dim = 64
    num_heads = 4
    B, L_cache, L_q = 2, 4, 3
    C = dim
    attn = CausalTemporalAttention(dim, num_heads)
    K_cache = torch.randn(B, L_cache, C)
    V_cache = torch.randn(B, L_cache, C)
    x = torch.randn(B, L_q, C)
    out, _ = attn(x, kv_cache=(K_cache, V_cache))
    assert out.shape == (B, L_q, C)


def test_causal_mask():
    """Test that causal mask prevents future attention."""
    dim = 64
    num_heads = 4
    B, L = 1, 4
    attn = CausalTemporalAttention(dim, num_heads)
    x = torch.zeros(B, L, dim)
    x[:, -1, :] = 100.0
    out1, _ = attn(x)
    x2 = x.clone()
    x2[:, -1, :] = -100.0
    out2, _ = attn(x2)
    assert torch.allclose(out1[:, 0, :], out2[:, 0, :], atol=1e-5), \
        "Causal mask failed: first frame is affected by last frame"


def test_prefix_enhanced_spatial_attention():
    """Test prefix-enhanced spatial attention."""
    dim = 64
    num_heads = 4
    prefix_len = 3
    B, HW = 4, 16
    attn = PrefixEnhancedSpatialAttention(dim, num_heads, prefix_len)
    x = torch.randn(B, HW, dim)
    out, _ = attn(x)
    assert out.shape == (B, HW, dim)


def test_prefix_enhanced_spatial_attention_with_prefix():
    """Test with prefix frames."""
    dim = 64
    num_heads = 4
    prefix_len = 3
    B, HW = 4, 16
    attn = PrefixEnhancedSpatialAttention(dim, num_heads, prefix_len)
    x = torch.randn(B, HW, dim)
    prefix = torch.randn(prefix_len, HW, dim)
    out, kv = attn(x, prefix_frames=prefix, return_kv=True)
    assert out.shape == (B, HW, dim)
    assert kv is not None


def test_kv_cache_queue_basic():
    """Test KVCacheQueue basic operations."""
    max_frames = 8
    chunk_size = 4
    num_layers = 2
    cache = KVCacheQueue(max_frames, chunk_size, num_layers)
    assert cache.num_cached_frames == 0
    assert cache.get_cache(0) is None

    B, C = 2, 32
    k1 = torch.randn(B, chunk_size, C)
    v1 = torch.randn(B, chunk_size, C)
    cache.update_all_layers([(k1, v1), (k1, v1)])
    assert cache.num_cached_frames == chunk_size

    K, V = cache.get_cache(0)
    assert K.shape == (B, chunk_size, C)


def test_kv_cache_queue_dequeue():
    """Test dequeue when full."""
    max_frames = 8
    chunk_size = 4
    num_layers = 1
    cache = KVCacheQueue(max_frames, chunk_size, num_layers)
    B, C = 2, 32
    for i in range(2):
        k = torch.randn(B, chunk_size, C)
        v = torch.randn(B, chunk_size, C)
        cache.update_all_layers([(k, v)])
    assert cache.num_cached_frames == max_frames

    k_new = torch.randn(B, chunk_size, C)
    v_new = torch.randn(B, chunk_size, C)
    cache.update_all_layers([(k_new, v_new)])
    assert cache.num_cached_frames == max_frames


def test_tpe_cyclic_wrapping():
    """Test Cyclic-TPE wrapping."""
    dim = 64
    max_len = 65
    tpe = TemporalPositionalEmbedding(dim, max_len)
    idx0 = torch.tensor([0])
    idx_max = torch.tensor([max_len])
    emb0 = tpe(idx0)
    emb_max = tpe(idx_max)
    assert torch.allclose(emb0, emb_max), "Cyclic wrapping failed"


def test_transformer_forward_training():
    """Test transformer forward pass during training."""
    model = Ca2VDMTransformer(
        in_channels=4, out_channels=8, patch_size=2,
        hidden_size=64, depth=2, num_heads=4,
        context_dim=None, prefix_len=2,
        max_seq_len=17, max_height=8, max_width=8,
    )
    B, L, C, H, W = 2, 9, 4, 16, 16
    z = torch.randn(B, L, C, H, W)
    t = torch.zeros(B, L, dtype=torch.long)
    t[:, 1:] = 500
    out, _ = model(z, t, prefix_len=1)
    assert out.shape == (B, 8, 8, H, W), f"Expected (B, 8, 8, H, W), got {out.shape}"


def test_transformer_forward_inference():
    """Test transformer forward pass during inference."""
    model = Ca2VDMTransformer(
        in_channels=4, out_channels=8, patch_size=2,
        hidden_size=64, depth=2, num_heads=4,
        context_dim=None, prefix_len=2,
        max_seq_len=17, max_height=8, max_width=8,
    )
    B, l, C, H, W = 2, 8, 4, 16, 16
    z = torch.randn(B, l, C, H, W)
    t = torch.full((B,), 500, dtype=torch.long)
    out, _ = model(z, t, prefix_len=0)
    assert out.shape == (B, l, 8, H, W)


def test_ca2vdm_training_loss():
    """Test Ca2VDM training loss."""
    transformer = Ca2VDMTransformer(
        in_channels=4, out_channels=8, patch_size=2,
        hidden_size=64, depth=2, num_heads=4,
        context_dim=None, prefix_len=2,
        max_seq_len=17, max_height=8, max_width=8,
    )
    model = Ca2VDM(transformer=transformer, T=100, chunk_size=8, max_prefix_len=9)
    B, L, C, H, W = 2, 17, 4, 16, 16
    z0 = torch.randn(B, L, C, H, W)
    loss_dict = model.training_loss(z0)
    assert "loss" in loss_dict
    assert "loss_simple" in loss_dict
    assert loss_dict["loss"].item() > 0


if __name__ == "__main__":
    print("Running Ca2-VDM tests...")
    test_causal_temporal_attention_shape()
    print("✓ test_causal_temporal_attention_shape")
    test_causal_temporal_attention_return_kv()
    print("✓ test_causal_temporal_attention_return_kv")
    test_causal_temporal_attention_kv_cache()
    print("✓ test_causal_temporal_attention_kv_cache")
    test_causal_mask()
    print("✓ test_causal_mask")
    test_prefix_enhanced_spatial_attention()
    print("✓ test_prefix_enhanced_spatial_attention")
    test_prefix_enhanced_spatial_attention_with_prefix()
    print("✓ test_prefix_enhanced_spatial_attention_with_prefix")
    test_kv_cache_queue_basic()
    print("✓ test_kv_cache_queue_basic")
    test_kv_cache_queue_dequeue()
    print("✓ test_kv_cache_queue_dequeue")
    test_tpe_cyclic_wrapping()
    print("✓ test_tpe_cyclic_wrapping")
    test_transformer_forward_training()
    print("✓ test_transformer_forward_training")
    test_transformer_forward_inference()
    print("✓ test_transformer_forward_inference")
    test_ca2vdm_training_loss()
    print("✓ test_ca2vdm_training_loss")
    print("\nAll tests passed!")
