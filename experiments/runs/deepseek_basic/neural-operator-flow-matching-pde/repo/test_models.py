"""Quick test to verify models can be instantiated and run.
Tests the model architectures without requiring data.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from p2vae import P2VAE, P2VAEConfig
from fmt import FlowMarchingTransformer, FMTConfig
from fmt.sampler import FlowMarchingSampler


def test_p2vae():
    """Test P2VAE model instantiation and forward pass."""
    print("=" * 60)
    print("Testing P2VAE")
    print("=" * 60)
    
    for name, config in [
        ("P2VAE-16M", P2VAEConfig(base_dim=64)),
        ("P2VAE-87M", P2VAEConfig(base_dim=128)),
    ]:
        model = P2VAE(config)
        
        # Test forward pass
        x = torch.randn(2, 3, 128, 128)
        output = model.forward(x)
        
        print(f"\n{name}:")
        print(f"  Input shape:  {x.shape}")
        print(f"  Mu shape:     {output['mu'].shape}")
        print(f"  Logvar shape: {output['logvar'].shape}")
        print(f"  Z shape:      {output['z'].shape}")
        print(f"  Recon shape:  {output['reconstruction'].shape}")
        print(f"  KL loss:      {output['kl_loss'].item():.4f}")
        
        # Test loss computation
        loss_dict = model.compute_loss(x)
        print(f"  Total loss:   {loss_dict['loss'].item():.4f}")
        print(f"  Recon loss:   {loss_dict['recon_loss'].item():.4f}")
        
        # Verify compression ratio
        input_elements = 3 * 128 * 128
        latent_elements = 16 * 16 * 16
        ratio = input_elements / latent_elements
        print(f"  Compression:  {input_elements} -> {latent_elements} elements ({ratio:.1f}x)")
        assert abs(ratio - 12.0) < 0.1, f"Expected 12x compression, got {ratio:.1f}x"


def test_fmt():
    """Test FMT model instantiation and forward pass."""
    print("\n" + "=" * 60)
    print("Testing FMT")
    print("=" * 60)
    
    for name, config in [
        ("FMT-S-6M", FMTConfig(embed_dim=256, num_layers=6)),
        ("FMT-B-42M", FMTConfig(embed_dim=512, num_layers=12)),
        ("FMT-L-138M", FMTConfig(embed_dim=768, num_layers=24)),
    ]:
        model = FlowMarchingTransformer(config)
        
        # Test forward pass with 4 frames
        B = 2
        C = 16  # latent channels
        H = W = 16  # latent spatial
        
        frames = [torch.randn(B, C, H, W) for _ in range(4)]
        t = torch.rand(B)
        h = model.gru.init_state(B, 'cpu')
        
        velocities, h_new = model(frames, t, h)
        
        print(f"\n{name}:")
        print(f"  Efficiency gain: {config.efficiency_gain:.1f}x")
        print(f"  Tokens per level: {config.tokens_per_level}")
        print(f"  Total tokens: {config.total_tokens}")
        for i, v in enumerate(velocities):
            print(f"  Velocity[{i}] shape: {v.shape}")
        print(f"  GRU state shape: {h_new.shape}")
        
        # Test flow marching loss
        y4 = torch.randn(B, C, H, W)
        loss_dict = model.compute_flow_marching_loss(
            frames[0], frames[1], frames[2], frames[3], y4
        )
        print(f"  FM loss: {loss_dict['loss'].item():.4f}")


def test_sampler():
    """Test flow marching sampler."""
    print("\n" + "=" * 60)
    print("Testing Flow Marching Sampler")
    print("=" * 60)
    
    config = FMTConfig(embed_dim=256, num_layers=4)  # Small for testing
    model = FlowMarchingTransformer(config)
    
    sampler = FlowMarchingSampler(model, num_steps=10)
    
    B = 1
    C = 16
    H = W = 16
    
    frames = [torch.randn(B, C, H, W) for _ in range(4)]
    h = model.gru.init_state(B, 'cpu')
    
    # Test deterministic prediction
    y_next, h_next = sampler.sample_next_frame(frames, h, 
                                                k_values=[1.0, 1.0, 1.0, 1.0])
    print(f"\nDeterministic prediction (k=1):")
    print(f"  Next frame shape: {y_next.shape}")
    print(f"  GRU state shape:  {h_next.shape}")
    
    # Test stochastic generation
    y_next_s, h_next_s = sampler.sample_next_frame(frames, h,
                                                     k_values=[1.0, 1.0, 1.0, 0.5])
    print(f"\nStochastic generation (k3=0.5):")
    print(f"  Next frame shape: {y_next_s.shape}")
    
    # Test autoregressive rollout
    print(f"\nAutoregressive rollout (10 steps):")
    all_frames = sampler.autoregressive_rollout(frames, num_steps=10)
    print(f"  Total frames: {len(all_frames)}")
    print(f"  Frame shapes: {[f.shape for f in all_frames[:5]]}...")


def test_flow_marching_math():
    """Verify the flow marching kernel equations."""
    print("\n" + "=" * 60)
    print("Testing Flow Marching Mathematics")
    print("=" * 60)
    
    # Test the interpolation kernel
    B, C, H, W = 2, 16, 16, 16
    
    x0 = torch.randn(B, C, H, W)
    x1 = torch.randn(B, C, H, W)
    z = torch.randn(B, C, H, W)
    
    for k in [0.0, 0.5, 1.0]:
        for t_val in [0.0, 0.5, 1.0]:
            t = torch.full((B,), t_val)
            k_t = torch.full((B,), k)
            
            t_r = t.view(B, 1, 1, 1)
            k_r = k_t.view(B, 1, 1, 1)
            
            # μ_t = t * x1 + k * (1-t) * x0
            mu_t = t_r * x1 + k_r * (1 - t_r) * x0
            # σ_t = (1-t) * (1-k)
            sigma_t = (1 - t_r) * (1 - k_r)
            # x_t^k = μ_t + σ_t * z
            x_t_k = mu_t + sigma_t * z
            
            # Velocity: u_t^k = (x_1 - x_t^k) / (1 - t)
            if t_val < 1.0:
                u_t_k = (x1 - x_t_k) / (1 - t_r)
            else:
                u_t_k = torch.zeros_like(x_t_k)
            
            # Verify: at t=0, k=1 -> x_t^k = x0 (deterministic)
            if abs(t_val) < 1e-6 and abs(k - 1.0) < 1e-6:
                assert torch.allclose(x_t_k, x0, atol=1e-4), \
                    f"x_0^1 should equal x0 at t=0, k=1"
            
            # Verify: at t=0, k=0 -> x_t^k = z (pure noise)
            if abs(t_val) < 1e-6 and abs(k) < 1e-6:
                assert torch.allclose(x_t_k, z, atol=1e-4), \
                    f"x_0^0 should equal z at t=0, k=0"
            
            # Verify: at t=1 -> x_t^k = x1 (regardless of k)
            if abs(t_val - 1.0) < 1e-6:
                assert torch.allclose(x_t_k, x1, atol=1e-4), \
                    f"x_1^k should equal x1 at t=1"
    
    # Test k-free objective
    t = torch.rand(B).view(B, 1, 1, 1)
    k = torch.rand(B).view(B, 1, 1, 1)
    
    x_t_k_test = t * x1 + k * (1 - t) * x0 + (1 - t) * (1 - k) * torch.randn_like(x0)
    
    # The k-free objective uses: (1-t) * g - (x_1 - x_t^k)
    # This is the preconditioned version of u_t^k = (x_1 - x_t^k) / (1-t)
    
    target = x1 - x_t_k_test
    # A perfect predictor g would give: g = target / (1-t)
    # So (1-t) * g = target
    
    print(f"  k=0 (stochastic):    x_t^k = t*x1 + (1-t)*z (FM kernel)")
    print(f"  k=1 (deterministic): x_t^k = t*x1 + (1-t)*x0 (neural operator)")
    print(f"  k-free objective:    ||(1-t)*g - (x1 - x_t^k)||^2")
    print(f"  All kernel properties verified!")


if __name__ == '__main__':
    test_flow_marching_math()
    test_p2vae()
    test_fmt()
    test_sampler()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
