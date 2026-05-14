"""
Quick test to verify NaViL model can be instantiated and forward pass works.

Usage:
    python tools/test_model.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import math


def test_visual_encoder():
    """Test visual encoder instantiation and forward pass."""
    print("Testing Visual Encoder...")
    from navil.visual_encoder import VisualEncoder, VisualEncoderConfig

    # Small config for testing
    config = VisualEncoderConfig(
        depth=3,
        width=256,
        num_heads=4,
        patch_size=16,
        pixel_shuffle_factor=2,
        llm_hidden_size=512,
    )
    encoder = VisualEncoder(config)
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Visual encoder params: {n_params:,}")

    # Test forward pass
    batch_size = 2
    img = torch.randn(batch_size, 3, 64, 64)  # 64x64 image
    tokens, grid_size = encoder(img)
    print(f"  Input: {img.shape}")
    print(f"  Output tokens: {tokens.shape}")
    print(f"  Grid size: {grid_size}")
    assert tokens.shape == (batch_size, grid_size[0] * grid_size[1], 512)
    print("  PASSED")


def test_moe():
    """Test MoE instantiation and forward pass."""
    print("Testing Modality-specific MoE...")
    from navil.moe import ModalitySpecificMoE, MMoEConfig, VISUAL_MODALITY, LINGUISTIC_MODALITY

    config = MMoEConfig(
        hidden_size=512,
        num_heads=8,
        head_dim=64,
        intermediate_size=1024,
        num_modalities=2,
    )
    moe = ModalitySpecificMoE(config)
    n_params = sum(p.numel() for p in moe.parameters())
    print(f"  MoE params: {n_params:,}")

    # Test forward pass
    batch_size = 2
    seq_len = 20
    x = torch.randn(batch_size, seq_len, 512)
    # Mix of visual (0) and linguistic (1) tokens
    modality_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    modality_ids[:, 10:] = 1  # Second half is linguistic

    out, pkv = moe(x, modality_ids=modality_ids)
    print(f"  Input: {x.shape}")
    print(f"  Output: {out.shape}")
    assert out.shape == x.shape
    print("  PASSED")


def test_navil_model():
    """Test NaViL model instantiation."""
    print("Testing NaViL Model (small config)...")
    from navil.model import NaViLModel, NaViLConfig

    # Very small config for testing
    config = NaViLConfig(
        llm_hidden_size=512,
        llm_num_layers=2,
        llm_num_heads=8,
        llm_num_kv_heads=4,
        llm_head_dim=64,
        llm_intermediate_size=1024,
        llm_vocab_size=1000,
        visual_encoder_depth=2,
        visual_encoder_width=256,
        visual_encoder_num_heads=4,
        visual_encoder_patch_size=16,
        use_moe=True,
        use_multiscale=False,
    )
    model = NaViLModel(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  NaViL model params: {n_params:,}")

    # Test forward pass (text only)
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    labels = input_ids.clone()

    outputs = model(input_ids=input_ids, labels=labels)
    print(f"  Loss: {outputs['loss'].item():.4f}")
    print(f"  Logits shape: {outputs['logits'].shape}")
    assert outputs['logits'].shape == (batch_size, seq_len, 1000)
    print("  PASSED")


def test_scaling_analysis():
    """Test scaling analysis utilities."""
    print("Testing Scaling Analysis...")
    from navil.scaling_analysis import (
        ScalingAnalyzer,
        ScalingExperimentResult,
        VisualEncoderArchitectureAnalyzer,
    )

    # Test parameter count formula
    analyzer = VisualEncoderArchitectureAnalyzer()
    for d, w in [(3, 4096), (6, 2880), (12, 2048), (24, 1472), (48, 1024)]:
        n = analyzer.compute_param_count(d, w)
        print(f"  d={d:3d}, w={w:4d}: N ≈ {n/1e6:.0f}M")

    # Test optimal encoder size finding
    scaling_analyzer = ScalingAnalyzer()
    results = []
    import numpy as np
    np.random.seed(42)
    for llm_m in [500, 1800, 7000]:
        for enc_m in [75, 150, 300, 600, 1200, 2400]:
            base_loss = 3.0 - 0.3 * math.log(llm_m / 500)
            enc_gain = 0.2 * (1 - math.exp(-enc_m / (llm_m * 0.3)))
            loss = base_loss - enc_gain + np.random.normal(0, 0.005)
            results.append(ScalingExperimentResult(
                llm_size_m=llm_m,
                encoder_size_m=enc_m,
                validation_loss=max(1.5, loss),
                num_training_samples=100_000_000,
            ))

    optimal = scaling_analyzer.compute_optimal_encoder_sizes(results, 100_000_000)
    print(f"  Optimal encoder sizes: {optimal}")

    if len(optimal) >= 2:
        llm_sizes = sorted(optimal.keys())
        enc_sizes = [optimal[s] for s in llm_sizes]
        alpha, beta = scaling_analyzer.fit_optimal_encoder_scaling(llm_sizes, enc_sizes)
        print(f"  Scaling fit: alpha={alpha:.3f}, beta={beta:.3f}")
        print(f"  (optimal_enc ∝ llm^{alpha:.3f})")

    print("  PASSED")


def test_multiscale():
    """Test multi-scale image processing."""
    print("Testing Multi-scale Packing...")
    from navil.data import MultiScaleImageProcessor
    from PIL import Image
    import numpy as np

    processor = MultiScaleImageProcessor(
        tau=0.5 * math.sqrt(2),
        min_area=32 * 32,
        patch_size=16,
    )

    # Create a test image
    img = Image.fromarray(np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8))
    scales = processor.process(img)
    print(f"  Input size: {img.size}")
    print(f"  Number of scales: {len(scales)}")
    for i, t in enumerate(scales):
        print(f"  Scale {i}: {t.shape}")

    print("  PASSED")


if __name__ == "__main__":
    print("=" * 50)
    print("NaViL Model Tests")
    print("=" * 50)

    try:
        test_visual_encoder()
    except Exception as e:
        print(f"  FAILED: {e}")

    try:
        test_moe()
    except Exception as e:
        print(f"  FAILED: {e}")

    try:
        test_navil_model()
    except Exception as e:
        print(f"  FAILED: {e}")

    try:
        test_scaling_analysis()
    except Exception as e:
        print(f"  FAILED: {e}")

    try:
        test_multiscale()
    except Exception as e:
        print(f"  FAILED: {e}")

    print("\nAll tests complete!")
