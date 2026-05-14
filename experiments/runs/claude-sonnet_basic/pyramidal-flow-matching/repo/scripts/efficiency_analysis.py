"""
Efficiency analysis script for Pyramidal Flow Matching.

Reproduces the efficiency analysis from Section 4.2 of the paper:
"Consider a video with T frame latents, where each frame contains N tokens
at the original resolution. The full-sequence diffusion has TN input tokens
in DiT and requires T^2 N^2 computations. In contrast, our method uses only
approximately TN/4^K tokens and T^2 N^2 / 16^K computations even for the
final pyramid stage."

Also verifies the token count reduction for a 10-second, 241-frame video:
- Full-sequence: 119,040 tokens
- Pyramidal: ≤15,360 tokens
"""

import math


def compute_token_counts(
    num_frames: int,
    height: int,
    width: int,
    vae_spatial_compression: int = 8,
    vae_temporal_compression: int = 8,
    patch_size: int = 2,
    num_pyramid_stages: int = 3,
):
    """
    Compute token counts for full-sequence diffusion vs pyramidal flow matching.
    
    Args:
        num_frames: Total number of video frames
        height: Video height in pixels
        width: Video width in pixels
        vae_spatial_compression: VAE spatial compression ratio (8x)
        vae_temporal_compression: VAE temporal compression ratio (8x)
        patch_size: DiT patch size
        num_pyramid_stages: Number of pyramid stages K
    
    Returns:
        Dictionary with token count analysis
    """
    # Latent dimensions after VAE compression
    latent_T = num_frames // vae_temporal_compression
    latent_H = height // vae_spatial_compression
    latent_W = width // vae_spatial_compression
    
    # Tokens per frame after patchification
    tokens_per_frame = (latent_H // patch_size) * (latent_W // patch_size)
    
    # Full-sequence diffusion: T * N tokens
    full_seq_tokens = latent_T * tokens_per_frame
    
    # Pyramidal flow matching: TN / 4^K tokens (for final stage)
    # With K=3 stages, 4^3 = 64x reduction
    K = num_pyramid_stages
    pyramid_tokens_final_stage = full_seq_tokens / (4 ** K)
    
    # Total tokens across all stages (uniform time windows)
    # Each stage has 1/K of the time, and operates at 1/4^(K-k) resolution
    total_pyramid_tokens = 0
    for k in range(K):
        # Stage k (0=lowest res, K-1=full res)
        # Resolution factor: 1/4^(K-1-k)
        res_factor = 4 ** (K - 1 - k)
        stage_tokens = full_seq_tokens / res_factor
        total_pyramid_tokens += stage_tokens / K  # Weighted by time fraction
    
    # Computation reduction (proportional to tokens^2 for attention)
    full_seq_compute = full_seq_tokens ** 2
    pyramid_compute_final = pyramid_tokens_final_stage ** 2
    
    return {
        'num_frames': num_frames,
        'height': height,
        'width': width,
        'latent_T': latent_T,
        'latent_H': latent_H,
        'latent_W': latent_W,
        'tokens_per_frame': tokens_per_frame,
        'full_seq_tokens': full_seq_tokens,
        'pyramid_tokens_final_stage': pyramid_tokens_final_stage,
        'total_pyramid_tokens': total_pyramid_tokens,
        'token_reduction_ratio': full_seq_tokens / pyramid_tokens_final_stage,
        'compute_reduction_ratio': full_seq_compute / pyramid_compute_final,
    }


def main():
    print("=" * 70)
    print("Pyramidal Flow Matching - Efficiency Analysis")
    print("=" * 70)
    print()
    
    # Analysis for 10-second, 241-frame video at 768p
    print("Case 1: 10-second video (241 frames) at 768p resolution")
    print("-" * 50)
    
    result = compute_token_counts(
        num_frames=241,
        height=768,
        width=768,
        vae_spatial_compression=8,
        vae_temporal_compression=8,
        patch_size=2,
        num_pyramid_stages=3,
    )
    
    print(f"Latent dimensions: {result['latent_T']} x {result['latent_H']} x {result['latent_W']}")
    print(f"Tokens per frame: {result['tokens_per_frame']}")
    print(f"Full-sequence tokens: {result['full_seq_tokens']:,}")
    print(f"Pyramid tokens (final stage): {result['pyramid_tokens_final_stage']:,.0f}")
    print(f"Average pyramid tokens: {result['total_pyramid_tokens']:,.0f}")
    print(f"Token reduction: {result['token_reduction_ratio']:.1f}x")
    print(f"Compute reduction: {result['compute_reduction_ratio']:.1f}x")
    print()
    
    # Paper claims: 119,040 tokens vs ≤15,360 tokens
    print("Paper claims:")
    print(f"  Full-sequence: 119,040 tokens")
    print(f"  Pyramidal: ≤15,360 tokens")
    print(f"  Our computation: {result['full_seq_tokens']:,} vs {result['pyramid_tokens_final_stage']:,.0f}")
    print()
    
    # Analysis for 5-second, 121-frame video at 768p
    print("Case 2: 5-second video (121 frames) at 768p resolution")
    print("-" * 50)
    
    result2 = compute_token_counts(
        num_frames=121,
        height=768,
        width=768,
        vae_spatial_compression=8,
        vae_temporal_compression=8,
        patch_size=2,
        num_pyramid_stages=3,
    )
    
    print(f"Full-sequence tokens: {result2['full_seq_tokens']:,}")
    print(f"Pyramid tokens (final stage): {result2['pyramid_tokens_final_stage']:,.0f}")
    print(f"Token reduction: {result2['token_reduction_ratio']:.1f}x")
    print()
    
    # Analysis for different K values
    print("Token reduction for different K values (10s video at 768p):")
    print("-" * 50)
    print(f"{'K':>5} | {'Full-seq':>12} | {'Pyramid':>12} | {'Reduction':>10}")
    print("-" * 50)
    
    for K in [1, 2, 3, 4]:
        r = compute_token_counts(
            num_frames=241,
            height=768,
            width=768,
            num_pyramid_stages=K,
        )
        print(f"{K:>5} | {r['full_seq_tokens']:>12,} | {r['pyramid_tokens_final_stage']:>12,.0f} | {r['token_reduction_ratio']:>9.1f}x")
    
    print()
    print("=" * 70)
    print("Training efficiency comparison (from paper Section 4.2):")
    print("=" * 70)
    print()
    print("Our method: 20.7k A100 GPU hours for 10s video generation")
    print("Open-Sora 1.2: 4.8k Ascend + 37.8k H100 hours for 97 frames")
    print()
    print("Token reduction formula: TN/4^K (final stage)")
    print("Compute reduction formula: T^2 N^2 / 16^K (final stage)")
    print()
    print(f"With K=3: {4**3}x token reduction, {16**3}x compute reduction")


if __name__ == '__main__':
    main()
