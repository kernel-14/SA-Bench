"""
Test script to verify the NFIG implementation.
Tests the full pipeline: FR-VAE tokenization + NFIG generation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from tokenizer.fr_vae import FRVAE, FrequencyDecomposer, FrequencyComposer
from models.nfig_transformer import NFIGTransformer, nfig_310m, build_causal_mask


def test_frequency_decomposer():
    """Test the frequency decomposer."""
    print("Testing FrequencyDecomposer...")
    scale_factors = [1, 2, 3, 4]
    decomposer = FrequencyDecomposer(n_bands=4, scale_factors=scale_factors)
    
    f = torch.randn(2, 32, 16, 16)
    components = decomposer(f)
    
    assert len(components) == 4, f"Expected 4 components, got {len(components)}"
    for comp in components:
        assert comp.shape == f.shape, f"Component shape mismatch: {comp.shape} vs {f.shape}"
    
    # Check that components sum to approximately the original
    f_reconstructed = sum(components)
    assert torch.allclose(f, f_reconstructed, atol=1e-4), "Decomposition is not lossless"
    
    print("  FrequencyDecomposer: PASSED")


def test_frequency_composer():
    """Test the frequency composer."""
    print("Testing FrequencyComposer...")
    composer = FrequencyComposer()
    
    components = [
        torch.randn(2, 32, 4, 4),
        torch.randn(2, 32, 8, 8),
        torch.randn(2, 32, 16, 16),
    ]
    
    result = composer(components, 16, 16)
    assert result.shape == (2, 32, 16, 16), f"Expected (2, 32, 16, 16), got {result.shape}"
    
    print("  FrequencyComposer: PASSED")


def test_fr_vae():
    """Test the FR-VAE."""
    print("Testing FR-VAE...")
    model = FRVAE(
        in_channels=3,
        latent_dim=32,
        base_channels=32,
        channel_mult=(1, 2),
        n_res_blocks=1,
        codebook_size=64,
        scale_factors=[1, 2, 3, 4],
    )
    
    # Check token counts
    token_counts = model.get_token_counts()
    assert token_counts == [1, 4, 9, 16], f"Expected [1,4,9,16], got {token_counts}"
    assert model.total_tokens() == 30, f"Expected 30, got {model.total_tokens()}"
    
    # Test forward pass
    x = torch.randn(2, 3, 64, 64)
    x_hat, vq_loss, f, f_hat = model(x)
    
    assert x_hat.shape == x.shape, f"Output shape mismatch: {x_hat.shape}"
    assert vq_loss.item() >= 0, "VQ loss should be non-negative"
    assert f.shape == f_hat.shape, "Feature map shapes should match"
    
    # Test encode/decode
    all_indices, _ = model.encode(x)
    assert len(all_indices) == 4, f"Expected 4 index tensors, got {len(all_indices)}"
    assert all_indices[0].shape == (2, 1), f"Band 0 indices shape: {all_indices[0].shape}"
    assert all_indices[3].shape == (2, 16), f"Band 3 indices shape: {all_indices[3].shape}"
    
    x_decoded = model.decode(all_indices, f.shape[2], f.shape[3])
    assert x_decoded.shape == x.shape, f"Decoded shape mismatch: {x_decoded.shape}"
    
    print("  FR-VAE: PASSED")


def test_causal_mask():
    """Test the block-wise causal attention mask."""
    print("Testing causal mask...")
    token_counts = [1, 4, 9]
    mask = build_causal_mask(token_counts, torch.device("cpu"))
    
    total = sum(token_counts)
    assert mask.shape == (total, total), f"Mask shape: {mask.shape}"
    
    # Band 0 (token 0) can only attend to itself
    assert not mask[0, 0], "Token 0 should attend to itself"
    assert mask[0, 1], "Token 0 should NOT attend to band 1"
    
    # Band 1 (tokens 1-4) can attend to band 0 and itself
    assert not mask[1, 0], "Band 1 should attend to band 0"
    assert not mask[1, 1], "Band 1 should attend to itself"
    assert mask[1, 5], "Band 1 should NOT attend to band 2"
    
    # Band 2 (tokens 5-13) can attend to bands 0, 1, and itself
    assert not mask[5, 0], "Band 2 should attend to band 0"
    assert not mask[5, 1], "Band 2 should attend to band 1"
    assert not mask[5, 5], "Band 2 should attend to itself"
    
    print("  Causal mask: PASSED")


def test_nfig_transformer():
    """Test the NFIG Transformer."""
    print("Testing NFIGTransformer...")
    model = NFIGTransformer(
        codebook_size=64,
        token_counts=[1, 4, 9, 16],
        n_classes=10,
        embed_dim=64,
        depth=2,
        n_heads=4,
        mlp_ratio=2.0,
    )
    
    B = 2
    token_sequences = [
        torch.randint(0, 64, (B, 1)),
        torch.randint(0, 64, (B, 4)),
        torch.randint(0, 64, (B, 9)),
        torch.randint(0, 64, (B, 16)),
    ]
    class_labels = torch.randint(0, 10, (B,))
    
    # Test forward pass
    logits = model(token_sequences, class_labels)
    assert logits.shape == (B, 30, 64), f"Logits shape: {logits.shape}"
    
    # Test generation
    model.eval()
    with torch.no_grad():
        generated = model.generate_fast(
            class_labels=torch.randint(0, 10, (B,)),
            cfg_scale=1.5,
            top_k=10,
        )
    
    assert len(generated) == 4, f"Expected 4 bands, got {len(generated)}"
    assert generated[0].shape == (B, 1), f"Band 0 shape: {generated[0].shape}"
    assert generated[3].shape == (B, 16), f"Band 3 shape: {generated[3].shape}"
    
    print("  NFIGTransformer: PASSED")


def test_full_pipeline():
    """Test the full FR-VAE + NFIG pipeline."""
    print("Testing full pipeline...")
    
    scale_factors = [1, 2, 3, 4]
    token_counts = [s*s for s in scale_factors]
    
    # Create tokenizer
    tokenizer = FRVAE(
        in_channels=3,
        latent_dim=32,
        base_channels=32,
        channel_mult=(1, 2),
        n_res_blocks=1,
        codebook_size=64,
        scale_factors=scale_factors,
    )
    
    # Create generator
    generator = NFIGTransformer(
        codebook_size=64,
        token_counts=token_counts,
        n_classes=10,
        embed_dim=64,
        depth=2,
        n_heads=4,
        mlp_ratio=2.0,
    )
    
    # Tokenize an image
    x = torch.randn(2, 3, 64, 64)
    all_indices, _ = tokenizer.encode(x)
    
    # Train step: predict tokens
    generator.train()
    logits = generator(all_indices, torch.randint(0, 10, (2,)))
    target = torch.cat(all_indices, dim=1)
    loss = F.cross_entropy(logits.reshape(-1, 64), target.reshape(-1))
    loss.backward()
    
    # Generate step
    generator.eval()
    with torch.no_grad():
        generated_tokens = generator.generate_fast(
            class_labels=torch.randint(0, 10, (2,)),
            cfg_scale=1.5,
            top_k=10,
        )
    
    # Decode generated tokens
    f = tokenizer.encoder(x)
    x_generated = tokenizer.decode(generated_tokens, f.shape[2], f.shape[3])
    assert x_generated.shape == x.shape, f"Generated shape: {x_generated.shape}"
    
    print("  Full pipeline: PASSED")


if __name__ == "__main__":
    print("Running NFIG tests...")
    test_frequency_decomposer()
    test_frequency_composer()
    test_fr_vae()
    test_causal_mask()
    test_nfig_transformer()
    test_full_pipeline()
    print("\nAll tests PASSED!")
