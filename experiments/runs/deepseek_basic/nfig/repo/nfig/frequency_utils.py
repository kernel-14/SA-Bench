"""
Frequency utilities for NFIG: Frequency-guided Decomposer and Composer.

This module implements the core frequency decomposition using Fast Fourier Transform (FFT),
as described in Section 3.1.1 of the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_frequency_mask(h, w, c, sigma_low, sigma_high, device='cpu'):
    """
    Create a frequency mask M_i for selecting a specific frequency band.
    
    The mask is a radial binary mask in the frequency domain, selecting
    frequencies in the range [sigma_low, sigma_high).
    
    Args:
        h, w: height and width of the feature map
        c: number of channels (mask is broadcast over channels)
        sigma_low: lower bound of the frequency band (normalized)
        sigma_high: upper bound of the frequency band (normalized)
        device: torch device
    
    Returns:
        mask: boolean mask of shape (1, h, w, 1) or (c, h, w)
    """
    # Create frequency grid
    u = torch.arange(h, device=device).float()
    v = torch.arange(w, device=device).float()
    
    # Center the frequencies
    u = u - h / 2
    v = v - w / 2
    
    # Normalize to [0, 1] range (sigma_max = 1.0)
    u = u / (h / 2)
    v = v / (w / 2)
    
    # Compute radial frequency
    u_grid, v_grid = torch.meshgrid(u, v, indexing='ij')
    radius = torch.sqrt(u_grid ** 2 + v_grid ** 2)
    
    # Create band-pass mask
    mask = (radius >= sigma_low) & (radius < sigma_high)
    
    # Shape: (1, h, w, 1) for broadcasting over channels and batch
    mask = mask.unsqueeze(0).unsqueeze(-1)  # (1, h, w, 1)
    
    return mask


def compute_frequency_band_boundaries(scales, sigma_max=1.0):
    """
    Compute frequency band boundaries based on the resolution of each scale.
    
    Following Equation in Section 3.2:
        sigma_i = sigma_{i-1} + (h_i * w_i) / (sum_j h_j * w_j) * sigma_max
    
    Args:
        scales: list of (h_i, w_i) tuples representing the resolution at each scale
        sigma_max: maximum frequency (default 1.0)
    
    Returns:
        boundaries: list of (sigma_low, sigma_high) for each frequency band
    """
    total_tokens = sum(h * w for h, w in scales)
    boundaries = []
    cum_sigma = 0.0
    
    for h, w in scales:
        sigma_low = cum_sigma
        sigma_high = cum_sigma + (h * w) / total_tokens * sigma_max
        boundaries.append((sigma_low, sigma_high))
        cum_sigma = sigma_high
    
    return boundaries


class FrequencyGuidedDecomposer(nn.Module):
    """
    Frequency-guided Decomposer: Decomposes feature maps into frequency components
    using Fast Fourier Transform (FFT).
    
    Given a feature map f, applies FFT, masks with M_i, and applies inverse FFT
    to obtain frequency component f_hat_i.
    
    f_hat_i = F^{-1}(F(f) ⊙ M_i)
    
    Args:
        scales: list of (h_i, w_i) for each frequency band
        latent_dim: number of channels in the feature map
        sigma_max: maximum normalized frequency
    """
    
    def __init__(self, scales, latent_dim, sigma_max=1.0):
        super().__init__()
        self.scales = scales
        self.latent_dim = latent_dim
        self.boundaries = compute_frequency_band_boundaries(scales, sigma_max)
    
    def forward(self, f):
        """
        Args:
            f: feature map of shape (B, C, H', W')
        
        Returns:
            components: list of tensors, each (B, C, H', W')
        """
        B, C, H, W = f.shape
        device = f.device
        
        # Apply 2D FFT (real-valued)
        f_fft = torch.fft.fft2(f, norm='ortho')
        f_fft_shifted = torch.fft.fftshift(f_fft)
        
        components = []
        
        for idx, (h_scale, w_scale) in enumerate(self.scales):
            sigma_low, sigma_high = self.boundaries[idx]
            mask = create_frequency_mask(H, W, C, sigma_low, sigma_high, device=device)
            # mask shape: (1, H, W, 1), need to broadcast to (1, C, H, W)
            mask = mask.permute(0, 3, 1, 2)  # (1, 1, H, W)
            
            # Apply mask in frequency domain
            f_band_fft = f_fft_shifted * mask
            
            # Inverse FFT
            f_band_fft_unshifted = torch.fft.ifftshift(f_band_fft)
            f_band = torch.fft.ifft2(f_band_fft_unshifted, norm='ortho').real
            
            components.append(f_band)
        
        return components


class FrequencyGuidedComposer(nn.Module):
    """
    Frequency-guided Composer: Combines frequency components back into a single
    feature map by interpolating to uniform size and summing.
    
    f_tilde = sum_i T(f_hat_i, H', W')
    
    where T(·, H', W') is an interpolation function to resize to original feature map size.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, components, target_h, target_w):
        """
        Args:
            components: list of tensors at potentially different resolutions
            target_h, target_w: target spatial dimensions
        
        Returns:
            f_tilde: combined feature map of shape (B, C, target_h, target_w)
        """
        # Sum all components interpolated to target size
        combined = 0
        for comp in components:
            if comp.shape[-2:] != (target_h, target_w):
                comp = F.interpolate(comp, size=(target_h, target_w), 
                                     mode='bilinear', align_corners=False)
            combined = combined + comp
        
        return combined


def compute_frequency_keep_score(original_img, reconstructed_img):
    """
    Compute Frequency Keep Score (FKS) between original and reconstructed images.
    
    As described in Appendix B.2, FKS is weighted similarity across
    High/Mid/Low frequency bands with weights (0.15, 0.28, 0.57).
    
    Args:
        original_img: (B, C, H, W)
        reconstructed_img: (B, C, H, W)
    
    Returns:
        fks: Frequency Keep Score
        band_scores: dict with Low, Middle, High band scores
    """
    def compute_psd(img):
        fft = torch.fft.fft2(img, norm='ortho')
        fft_shifted = torch.fft.fftshift(fft)
        psd = torch.abs(fft_shifted) ** 2
        return psd
    
    psd_orig = compute_psd(original_img)
    psd_recon = compute_psd(reconstructed_img)
    
    # PSD error
    psd_error = torch.mean(torch.abs(psd_orig - psd_recon)).item()
    
    H, W = original_img.shape[-2:]
    
    # Create radial frequency mask
    u = torch.arange(H).float()
    v = torch.arange(W).float()
    u = (u - H/2) / (H/2)
    v = (v - W/2) / (W/2)
    u_grid, v_grid = torch.meshgrid(u, v, indexing='ij')
    radius = torch.sqrt(u_grid**2 + v_grid**2)
    
    # Define bands: Low [0, 0.15), Mid [0.15, 0.6), High [0.6, 1.0]
    low_mask = (radius < 0.15).float().to(original_img.device)
    mid_mask = ((radius >= 0.15) & (radius < 0.6)).float().to(original_img.device)
    high_mask = (radius >= 0.6).float().to(original_img.device)
    
    # Compute band-wise similarity (cosine similarity in frequency domain)
    def band_similarity(psd1, psd2, mask):
        psd1_masked = psd1 * mask.unsqueeze(0).unsqueeze(0)
        psd2_masked = psd2 * mask.unsqueeze(0).unsqueeze(0)
        sim = F.cosine_similarity(
            psd1_masked.flatten(1), psd2_masked.flatten(1), dim=1
        ).mean().item()
        return sim
    
    low_score = band_similarity(psd_orig, psd_recon, low_mask)
    mid_score = band_similarity(psd_orig, psd_recon, mid_mask)
    high_score = band_similarity(psd_orig, psd_recon, high_mask)
    
    # Weighted combination
    weights = {'Low': 0.57, 'Middle': 0.28, 'High': 0.15}
    fks = weights['Low'] * low_score + weights['Middle'] * mid_score + weights['High'] * high_score
    
    return psd_error, fks, {'Low': low_score, 'Middle': mid_score, 'High': high_score}
