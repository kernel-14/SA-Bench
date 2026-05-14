"""
Tests for Ca2-VDM model components.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from ca2_vdm.attention import CausalTemporalAttention, PrefixEnhancedSpatialAttention
from ca2_vdm.cache import TemporalKVCacheQueue, KVCacheManager
from ca2_vdm.tpe import CyclicTPE
from ca2_vdm.model import Ca2VDM
from ca2_vdm.diffusion import DiffusionProcess


def test_causal_attention_mask():
    attn = CausalTemporalAttention(dim=64, num_heads=4)
    x = torch.randn(2, 8, 16, 64)
    out, kv = attn(x)
    assert out.shape == x.shape

    cache_k = torch.randn(2*16, 4, 4, 16)
    cache_v = torch.randn(2*16, 4, 4, 16)
    out2, kv2 = attn(x, (cache_k, cache_v))
    assert out2.shape == x.shape


def test_prefix_enhanced_spatial_attention():
    attn = PrefixEnhancedSpatialAttention(dim=64, num_heads=4, prefix_len=3)
    h = torch.randn(2, 6, 16, 64)
    P = 3
    out, spat_kv = attn(h, P=P)
    assert out.shape == h.shape


def test_kv_cache_queue():
    cache = TemporalKVCacheQueue(max_length=10)
    k = torch.randn(8, 4, 3, 16)
    v = torch.randn(8, 4, 3, 16)
    cache.enqueue(k, v)
    assert len(cache) == 3

    k2 = torch.randn(8, 4, 5, 16)
    v2 = torch.randn(8, 4, 5, 16)
    cache.enqueue(k2, v2)
    assert len(cache) == 8

    kv_result = cache.get_kv()
    assert kv_result is not None
    assert kv_result[0].shape == (8, 4, 8, 16)

    cache.dequeue(4)
    assert len(cache) == 4


def test_kv_cache_manager():
    manager = KVCacheManager(num_layers=4, P_max=25, P_prime=3)
    k = torch.randn(8, 4, 3, 16)
    v = torch.randn(8, 4, 3, 16)
    for layer_idx in range(4):
        manager.update_temporal(layer_idx, k, v, 3)
    for layer_idx in range(4):
        kv = manager.get_temporal_kv(layer_idx)
        assert kv is not None
        assert kv[0].shape[-2] == 3


def test_cyclic_tpe():
    tpe = CyclicTPE(dim=64, L_train=33, P_max=25, l=8)
    tpe_seq = tpe(8, cyclic_offset=0)
    assert tpe_seq.shape == (1, 8, 1, 64)
    tpe_seq_offset = tpe(8, cyclic_offset=5)
    assert tpe_seq_offset.shape == (1, 8, 1, 64)
    tpe_wrap = tpe(8, cyclic_offset=33)
    assert torch.allclose(tpe_seq, tpe_wrap)


def test_partial_noising():
    diffusion = DiffusionProcess(num_timesteps=1000)
    z_0 = torch.randn(2, 4, 8, 32, 32)
    P = 3
    t = torch.randint(0, 1000, (2,))
    z_t, noise, mask = diffusion.q_sample_partial(z_0, t, P)
    assert z_t.shape == z_0.shape
    assert noise.shape == z_0.shape
    assert torch.allclose(z_t[:, :, :P, :, :], z_0[:, :, :P, :, :])
    assert mask[:, :, :P, :, :].sum() == 0
    assert mask[:, :, P:, :, :].sum() > 0


def test_model_forward():
    model = Ca2VDM(
        in_channels=4, H=8, W=8, dim=256, num_heads=4, num_layers=4,
        l=4, P_max=13, L_train=17, prefix_len=2,
        use_text_cond=True, text_dim=512,
    )
    B, C, L, H, W = 2, 4, 8, 8, 8
    z = torch.randn(B, C, L, H, W)
    t = torch.randint(0, 1000, (B,))
    P = 2
    text_emb = torch.randn(B, 77, 512)
    result = model(z, t, P=P, text_emb=text_emb)
    output = result['output']
    assert output.shape[0] == B
    assert output.shape[1] == C * 2


def test_kv_cache_inference():
    """Test the full KV-cache inference flow."""
    model = Ca2VDM(
        in_channels=4, H=8, W=8, dim=256, num_heads=4, num_layers=4,
        l=4, P_max=13, L_train=17, prefix_len=2,
        use_text_cond=False,
    )
    manager = KVCacheManager(num_layers=4, P_max=13, P_prime=2)
    
    # Cache writing: compute KVs for clean frames
    B, C, L, H, W = 1, 4, 4, 8, 8
    z_clean = torch.randn(B, C, L, H, W)
    t_zero = torch.zeros(B, dtype=torch.long)
    
    result = model(z_clean, t_zero, P=0, kv_cache_manager=manager, cache_write=True)
    
    # Manually update the cache manager with returned caches
    temporal_caches = result['temporal_caches']
    spatial_caches = result['spatial_caches']
    
    for layer_idx in range(len(temporal_caches)):
        if temporal_caches[layer_idx] is not None:
            k, v = temporal_caches[layer_idx]
            manager.update_temporal(layer_idx, k, v, L)
        if spatial_caches[layer_idx] is not None:
            k, v = spatial_caches[layer_idx]
            manager.update_spatial(layer_idx, k, v)
    
    # Now check caches are populated
    for i in range(4):
        kv = manager.get_temporal_kv(i)
        assert kv is not None, f"Temporal cache for layer {i} is None"
    
    # Denoising: use cache for denoising new frames
    z_noisy = torch.randn(B, C, L, H, W)
    t_noisy = torch.full((B,), 500, dtype=torch.long)
    
    result2 = model(z_noisy, t_noisy, P=4, kv_cache_manager=manager, cache_write=False)
    assert result2['output'].shape[0] == B


def test_loss_computation():
    model = Ca2VDM(
        in_channels=4, H=8, W=8, dim=256, num_heads=4, num_layers=4,
        l=4, P_max=13, L_train=17, prefix_len=2,
        use_text_cond=False,
    )
    diffusion = DiffusionProcess(num_timesteps=1000)
    z_0 = torch.randn(2, 4, 8, 8, 8)
    P = 2
    t = torch.randint(0, 1000, (2,))
    loss_dict = diffusion.compute_loss(model, z_0, P, t)
    assert 'loss' in loss_dict
    assert loss_dict['loss'].item() > 0


if __name__ == '__main__':
    print("Running Ca2-VDM tests...")
    test_causal_attention_mask(); print("✓ Causal attention mask")
    test_prefix_enhanced_spatial_attention(); print("✓ Prefix-enhanced spatial attention")
    test_kv_cache_queue(); print("✓ KV-cache queue")
    test_kv_cache_manager(); print("✓ KV-cache manager")
    test_cyclic_tpe(); print("✓ Cyclic TPE")
    test_partial_noising(); print("✓ Partial noising")
    test_model_forward(); print("✓ Model forward pass")
    test_kv_cache_inference(); print("✓ KV-cache inference")
    test_loss_computation(); print("✓ Loss computation")
    print("All tests passed!")
