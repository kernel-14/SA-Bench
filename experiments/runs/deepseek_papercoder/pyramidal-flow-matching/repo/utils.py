## utils.py
"""
Utility functions for Pyramidal Flow Matching.

Contains spatial transformation helpers (downsample/upsample), corrective renoising for pyramidal stage jumps,
attention mask generation for packed batch causal attention, and patchify/unpatchify operations.

All functions are pure PyTorch, device/dtype agnostic, and follow the notations in the paper.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import math


def downsample(x: torch.Tensor, factor: int) -> torch.Tensor:
    """
    Bilinear spatial downsampling by an integer factor (applied only to H and W).

    Args:
        x: Input tensor of shape (B, C, H, W).
        factor: Downsampling factor (power of two).

    Returns:
        Downsampled tensor of shape (B, C, H // factor, W // factor).
    """
    assert factor >= 1 and (factor & (factor - 1)) == 0, "factor must be a power of two"
    if factor == 1:
        return x
    return F.interpolate(x, scale_factor=1.0 / factor, mode='bilinear', align_corners=False)


def nearest_upsample(x: torch.Tensor) -> torch.Tensor:
    """
    Nearest-neighbour spatial upsampling by a factor of 2.
    This is the Up(·) operation used at pyramidal jump points to match the covariance derivation.

    Args:
        x: Input tensor of shape (B, C, H, W).

    Returns:
        Upsampled tensor of shape (B, C, 2H, 2W).
    """
    return F.interpolate(x, scale_factor=2.0, mode='nearest')


def corrective_renoise(x: torch.Tensor, s_k: float) -> torch.Tensor:
    """
    Applies the corrective rescaling and renoising at a jump point from stage k+1 to stage k.
    Implements Eq. (15):  hat{x}_{s_k} = (1+s_k)/2 * Up(hat{x}_{e_{k+1}}) + sqrt(3)*(1-s_k)/2 * n'
    where n' has blockwise covariance with off-diagonal gamma = -1/3 and is independent of x.

    Args:
        x: Upsampled endpoint of stage k+1, shape (B, C, H, W).
        s_k: Starting timestep of the finer stage k (from schedule).

    Returns:
        Corrected noisy latent hat{x}_{s_k} of the same shape as x.
    """
    alpha = (1.0 + s_k) / 2.0
    beta = math.sqrt(3.0) * (1.0 - s_k) / 2.0

    # Generate noise with blockwise covariance Σ' where each 2x2 block has
    # diagonal=1 and off-diagonal=gamma=-1/3.
    # We construct it by taking standard Gaussian noise, subtracting the block mean,
    # and scaling to achieve the desired variance.
    B, C, H, W = x.shape
    assert H % 2 == 0 and W % 2 == 0, "H and W must be even for nearest-neighbour upsampling blocks"

    noise_std = torch.randn(B, C, H, W, dtype=x.dtype, device=x.device)

    # Rearrange into blocks: shape (B, C, 4, H/2, W/2)
    # pixel_unshuffle groups 2x2 spatial blocks into channel dimension
    noise_blocks = F.pixel_unshuffle(noise_std, downscale_factor=2)  # (B, C*4, H/2, W/2)
    noise_blocks = noise_blocks.reshape(B, C, 4, H // 2, W // 2)  # separate the 4 components

    # Subtract block mean to make each 4-vector orthogonal to [1,1,1,1]
    mean = noise_blocks.mean(dim=2, keepdim=True)  # (B, C, 1, H/2, W/2)
    centered = noise_blocks - mean  # now each group has mean zero

    # The desired covariance is: Σ' has variance 1 on diagonal, gamma=-1/3 off-diagonal.
    # For a vector v with [1,1,1,1] removed (mean subtracted), the covariance of the projection
    # onto the orthogonal subspace is (I - (1/4)J)Σ'(I - (1/4)J) = ? We need exact scaling.
    # Actually, the goal is to produce n' with block covariance Σ'.
    # A straightforward construction: generate a standard Gaussian vector of length 4, then
    # apply a matrix L such that LL^T = Σ'. For Σ' with 1 on diag and -1/3 off-diag,
    # one can use the Cholesky decomposition but L is not unique. Since we have mean-subtracted
    # vectors, we can use the fact that the distribution of the projection is simpler.
    # Alternative method: take centered (mean subtracted) standard Gaussian, scale to achieve
    # the correct variance in the subspace orthogonal to [1,1,1,1]. The covariance of the
    # mean-subtracted standard Gaussian in that subspace is (I - (1/4)J) which has eigenvalues:
    # 0 (for [1,1,1,1]), and 1 (multiplicity 3) for the orthogonal complement.
    # To get Σ' which also has eigenvalue 0 for [1,1,1,1] and (1 - (-1/3)) = 4/3 for the other 3 directions,
    # we need to scale the centered noise by sqrt(4/3).
    variance_in_ortho = 1.0  # because standard Gaussian after mean subtraction has variance 3/4? Let's check.
    # For a standard Gaussian vector v ~ N(0,I), the mean-subtracted vector w = v - mean(v) has covariance:
    # Cov(w) = I - (1/4) J. Its eigenvalues: 1 (multiplicity 3) for vectors orthogonal to all-ones, and 0 for all-ones.
    # The total variance along any orthogonal direction is 1. But we want noise with Σ' which has variance
    # 1 on diagonal but -1/3 off-diagonal. For a vector in the orthogonal subspace, the effect of Σ' is to
    # multiply any such vector by (1 - gamma) = 4/3? Actually, if x is orthogonal to all-ones, then Σ' x = (1 - gamma) x = (4/3) x.
    # Thus the variance in the orthogonal subspace should be scaled by sqrt(4/3) to match Σ'.
    # So we multiply centered by sqrt(4/3).
    scale = math.sqrt(4.0 / 3.0)
    corrected_noise_blocks = centered * scale

    # Reshape back to (B, C, H, W) using pixel_shuffle
    corrected_noise = F.pixel_shuffle(corrected_noise_blocks.reshape(B, C * 4, H // 2, W // 2), upscale_factor=2)

    return alpha * x + beta * corrected_noise


def compute_causal_mask(batch_cfg: Dict) -> torch.Tensor:
    """
    Constructs a block-diagonal causal attention mask for a packed batch.
    Within each sequence, tokens from frame t can only attend to tokens with frame index <= t.
    Between different sequences, all attention is masked to -inf (no cross-sequence attention).

    Expected batch_cfg format:
        {
            "sequences": [
                {"num_tokens": int, "num_frames": int},
                ...
            ],
            "patches_per_frame": int  # must be consistent across all sequences (or omitted, inferred)
        }

    Args:
        batch_cfg: Dict describing the composition of the packed batch.

    Returns:
        A float tensor of shape (1, 1, total_tokens, total_tokens) with 0.0 for allowed attention
        and -inf for disallowed. Suitable for adding to attention logits.
    """
    sequences: List[Dict] = batch_cfg["sequences"]
    patches_per_frame = batch_cfg.get("patches_per_frame", None)

    total_tokens = sum(seq["num_tokens"] for seq in sequences)
    mask = torch.full((total_tokens, total_tokens), -float('inf'))

    start = 0
    for seq in sequences:
        seq_len = seq["num_tokens"]
        num_frames = seq["num_frames"]
        end = start + seq_len

        # If only one frame (or no temporal dimension), full bidirectional within the block
        if num_frames == 1:
            mask[start:end, start:end] = 0.0
        else:
            if patches_per_frame is None:
                # Infer patches_per_frame from the first video sequence if not provided
                # All sequences must have the same patches_per_frame for this to work uniformly.
                # We'll compute from the current sequence and store it.
                p_frame = seq_len // num_frames
                if seq_len % num_frames != 0:
                    raise ValueError(f"num_tokens ({seq_len}) not divisible by num_frames ({num_frames})")
                # Use this for all sequences if none provided; this is a heuristic
                # but the caller should provide it for robustness.
                patches_per_frame = p_frame
            else:
                p_frame = patches_per_frame

            # Compute frame boundaries within this sequence block
            # Indices relative to the global tensor
            for i in range(start, end):
                frame_q = (i - start) // p_frame
                for j in range(start, end):
                    frame_k = (j - start) // p_frame
                    if frame_k <= frame_q:
                        mask[i, j] = 0.0

        start = end

    # Add batch and heads dimensions for typical attention masks
    return mask.unsqueeze(0).unsqueeze(0)  # shape (1, 1, tot, tot)


def patchify(latent: torch.Tensor, patch_size: Tuple[int, int]) -> torch.Tensor:
    """
    Converts a 5D video latent tensor into a flat sequence of patches, preserving time order.
    Each spatial patch (p_h, p_w) is flattened, and tokens are ordered frame-wise.

    Args:
        latent: Tensor of shape (B, C, T, H, W).
        patch_size: Tuple (p_h, p_w) for spatial patch size.

    Returns:
        Tensor of shape (B, T * (H/p_h) * (W/p_w), C * p_h * p_w)
    """
    p_h, p_w = patch_size
    B, C, T, H, W = latent.shape
    assert H % p_h == 0 and W % p_w == 0, f"Latent spatial dims {H}x{W} not divisible by patch size {patch_size}"

    # Using einops style rearrangement
    # Use view/reshape approach for clarity without extra dependency
    patches = latent.reshape(B, C, T, H // p_h, p_h, W // p_w, p_w)
    patches = patches.permute(0, 2, 3, 5, 1, 4, 6).contiguous()  # (B, T, H', W', C, p_h, p_w)
    patches = patches.reshape(B, T * (H // p_h) * (W // p_w), C * p_h * p_w)
    return patches


def unpatchify(patches: torch.Tensor, latent_shape: Tuple[int, int, int, int], patch_size: Tuple[int, int]) -> torch.Tensor:
    """
    Inverse of patchify: reconstructs the 5D latent tensor from patch tokens.

    Args:
        patches: Tensor of shape (B, total_tokens, C * p_h * p_w).
        latent_shape: Target latent shape (C, T, H, W) (without batch dim).
        patch_size: Spatial patch size (p_h, p_w).

    Returns:
        Tensor of shape (B, C, T, H, W).
    """
    p_h, p_w = patch_size
    C, T, H, W = latent_shape
    assert H % p_h == 0 and W % p_w == 0, "Target latent spatial dims must be divisible by patch size"
    h_patches = H // p_h
    w_patches = W // p_w
    total_tokens_expected = T * h_patches * w_patches
    B = patches.shape[0]
    assert patches.shape[1] == total_tokens_expected, (
        f"Token count mismatch: got {patches.shape[1]} but expected {total_tokens_expected}"
    )
    patches = patches.reshape(B, T, h_patches, w_patches, C, p_h, p_w)
    patches = patches.permute(0, 4, 1, 2, 5, 3, 6).contiguous()  # (B, C, T, h_patches, p_h, w_patches, p_w)
    patches = patches.reshape(B, C, T, H, W)
    return patches
