"""
Unit tests for NFIG model components.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest

from nfig.frequency_utils import (
    FrequencyGuidedDecomposer,
    FrequencyGuidedComposer,
    compute_frequency_band_boundaries,
    compute_frequency_keep_score,
    create_frequency_mask,
)
from nfig.fr_vae import FRVAE, VectorQuantizer
from nfig.nfig_transformer import NFIGTransformer


class TestFrequencyUtils:
    """Test frequency decomposition and composition utilities."""
    
    def test_band_boundaries(self):
        scales = [(1, 1), (2, 2), (3, 3)]
        boundaries = compute_frequency_band_boundaries(scales)
        assert len(boundaries) == 3
        assert boundaries[0][0] == 0.0
        # Check that the last boundary ends at 1.0
        assert abs(boundaries[-1][1] - 1.0) < 1e-6
    
    def test_frequency_mask_creation(self):
        mask = create_frequency_mask(16, 16, 256, 0.0, 0.5)
        assert mask.shape == (1, 16, 16, 1)
    
    def test_decomposer_composer(self):
        scales = [(1, 1), (2, 2), (4, 4)]
        decomposer = FrequencyGuidedDecomposer(scales, latent_dim=64)
        composer = FrequencyGuidedComposer()
        
        f = torch.randn(2, 64, 16, 16)
        components = decomposer(f)
        assert len(components) == 3
        
        # All components should have same spatial size as input
        for comp in components:
            assert comp.shape == f.shape
        
        # Recombine
        f_recon = composer(components, 16, 16)
        assert f_recon.shape == f.shape
    
    def test_frequency_keep_score(self):
        img = torch.randn(2, 3, 64, 64)
        # Same image should have perfect score
        psd_error, fks, band_scores = compute_frequency_keep_score(img, img)
        assert fks > 0.99  # Near perfect


class TestVectorQuantizer:
    """Test vector quantization."""
    
    def test_quantization(self):
        vq = VectorQuantizer(codebook_size=512, codebook_dim=64)
        
        v = torch.randn(2, 64, 8, 8)
        v_q, tokens, loss = vq(v)
        
        assert v_q.shape == v.shape
        assert tokens.shape == (2, 8, 8)
        assert loss.item() >= 0
    
    def test_quantize_indices(self):
        vq = VectorQuantizer(codebook_size=512, codebook_dim=64)
        
        tokens = torch.randint(0, 512, (2, 8, 8))
        v_q = vq.quantize_indices(tokens)
        
        assert v_q.shape == (2, 64, 8, 8)


class TestFRVAE:
    """Test FR-VAE tokenizer."""
    
    def test_model_creation(self):
        scales = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5),
                  (6, 6), (8, 8), (10, 10), (13, 13), (16, 16)]
        model = FRVAE(scales=scales, codebook_size=4096, codebook_dim=256, 
                      latent_dim=256, image_size=256, use_dino_disc=False)
        
        assert model.get_total_tokens() == 680
        assert model.n_scales == 10
    
    def test_forward_pass(self):
        scales = [(1, 1), (2, 2), (4, 4)]
        model = FRVAE(scales=scales, codebook_size=512, codebook_dim=64,
                      latent_dim=64, image_size=64, use_dino_disc=False)
        
        x = torch.randn(2, 3, 64, 64)
        x_recon, token_list, vq_loss = model(x)
        
        assert x_recon.shape == x.shape
        assert len(token_list) == 3
        assert token_list[0].shape == (2, 1, 1)
        assert token_list[1].shape == (2, 2, 2)
        assert token_list[2].shape == (2, 4, 4)
    
    def test_encode_decode(self):
        scales = [(1, 1), (2, 2), (4, 4)]
        model = FRVAE(scales=scales, codebook_size=512, codebook_dim=64,
                      latent_dim=64, image_size=64, use_dino_disc=False)
        model.eval()
        
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            token_list, _, _ = model.encode(x)
            x_recon = model.decode_from_tokens(token_list)
        
        assert x_recon.shape == x.shape
    
    def test_token_flattening(self):
        scales = [(1, 1), (2, 2), (4, 4)]
        model = FRVAE(scales=scales, codebook_size=512, codebook_dim=64,
                      latent_dim=64, image_size=64, use_dino_disc=False)
        
        # Create dummy token list
        token_list = [
            torch.randint(0, 512, (2, 1, 1)),
            torch.randint(0, 512, (2, 2, 2)),
            torch.randint(0, 512, (2, 4, 4)),
        ]
        
        flat = model.get_token_sequence(token_list)
        assert flat.shape == (2, 21)  # 1 + 4 + 16 = 21
        
        unflattened = model.unflatten_tokens(flat)
        assert len(unflattened) == 3
        for a, b in zip(unflattened, token_list):
            assert a.shape == b.shape


class TestNFIGTransformer:
    """Test NFIG autoregressive transformer."""
    
    def test_model_creation(self):
        scales = [(1, 1), (2, 2)]
        model = NFIGTransformer(scales=scales, codebook_size=512, dim=256,
                                depth=2, num_heads=4, num_classes=100)
        
        assert model.n_scales == 2
        assert model.total_tokens == 5  # 1 + 4
    
    def test_training_forward(self):
        scales = [(1, 1), (2, 2)]
        model = NFIGTransformer(scales=scales, codebook_size=512, dim=256,
                                depth=2, num_heads=4, num_classes=100)
        
        B = 2
        token_seqs = [
            torch.randint(0, 512, (B, 1)),
            torch.randint(0, 512, (B, 4)),
        ]
        class_ids = torch.randint(0, 100, (B,))
        
        logits, loss = model(token_seqs, class_ids)
        assert logits.shape == (B, 5, 512)
        assert loss.item() > 0
    
    def test_generation(self):
        scales = [(1, 1), (2, 2)]
        model = NFIGTransformer(scales=scales, codebook_size=512, dim=256,
                                depth=2, num_heads=4, num_classes=100)
        model.eval()
        
        class_ids = torch.tensor([5, 10])
        with torch.no_grad():
            tokens = model.generate(class_ids, top_k=50, cfg_scale=1.0)
        
        assert len(tokens) == 2
        assert tokens[0].shape == (2, 1)
        assert tokens[1].shape == (2, 4)
    
    def test_cfg_generation(self):
        scales = [(1, 1), (2, 2)]
        model = NFIGTransformer(scales=scales, codebook_size=512, dim=256,
                                depth=2, num_heads=4, num_classes=100)
        model.eval()
        
        class_ids = torch.tensor([5])
        with torch.no_grad():
            tokens = model.generate(class_ids, top_k=50, cfg_scale=3.0)
        
        assert len(tokens) == 2


if __name__ == '__main__':
    # Run tests manually
    test_freq = TestFrequencyUtils()
    test_freq.test_band_boundaries()
    test_freq.test_frequency_mask_creation()
    test_freq.test_decomposer_composer()
    print("Frequency utils: OK")
    
    test_vq = TestVectorQuantizer()
    test_vq.test_quantization()
    test_vq.test_quantize_indices()
    print("Vector Quantizer: OK")
    
    test_frvae = TestFRVAE()
    test_frvae.test_model_creation()
    test_frvae.test_forward_pass()
    test_frvae.test_encode_decode()
    test_frvae.test_token_flattening()
    print("FR-VAE: OK")
    
    test_nfig = TestNFIGTransformer()
    test_nfig.test_model_creation()
    test_nfig.test_training_forward()
    test_nfig.test_generation()
    test_nfig.test_cfg_generation()
    print("NFIG Transformer: OK")
    
    print("\nAll tests passed!")
