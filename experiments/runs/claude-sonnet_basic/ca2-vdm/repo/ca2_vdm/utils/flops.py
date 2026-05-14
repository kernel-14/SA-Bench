"""
FLOPs counting utilities for Ca2-VDM efficiency analysis.

Implements the FLOPs analysis from Section 4.3 (Figure 8):
  - Counts FLOPs for temporal, spatial, and visual-text attention layers
  - Compares Ca2-VDM vs OS-Ext (bidirectional baseline) as P_max grows
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn


def attention_flops(
    seq_len_q: int,
    seq_len_kv: int,
    dim: int,
    num_heads: int,
    batch_size: int = 1,
) -> int:
    """
    Compute FLOPs for a single attention operation.

    FLOPs = 2 * B * H * L_q * L_kv * D_h (for QK^T and AV)
          + 2 * B * L_q * D * D (for Q, K, V projections)
          + 2 * B * L_q * D * D (for output projection)

    Args:
        seq_len_q: Query sequence length.
        seq_len_kv: Key/value sequence length.
        dim: Feature dimension.
        num_heads: Number of attention heads.
        batch_size: Batch size.

    Returns:
        Total FLOPs (integer).
    """
    head_dim = dim // num_heads

    # QK^T: (B, H, L_q, D_h) x (B, H, D_h, L_kv) -> (B, H, L_q, L_kv)
    qk_flops = 2 * batch_size * num_heads * seq_len_q * seq_len_kv * head_dim

    # AV: (B, H, L_q, L_kv) x (B, H, L_kv, D_h) -> (B, H, L_q, D_h)
    av_flops = 2 * batch_size * num_heads * seq_len_q * seq_len_kv * head_dim

    # Q, K, V projections: 3 * (B, L, D) x (D, D)
    proj_flops = 3 * 2 * batch_size * seq_len_q * dim * dim

    # Output projection: (B, L, D) x (D, D)
    out_flops = 2 * batch_size * seq_len_q * dim * dim

    return qk_flops + av_flops + proj_flops + out_flops


def count_flops(
    model_config: Dict,
    p_k: int,
    chunk_size: int,
    use_kv_cache: bool = True,
    resolution: int = 256,
    patch_size: int = 2,
) -> Dict[str, int]:
    """
    Count FLOPs for one denoising step of Ca2-VDM or baseline.

    Counts FLOPs for:
      - Temporal attention
      - Spatial attention
      - Visual-text cross attention

    Args:
        model_config: Dict with model hyperparameters.
        p_k: Number of clean prefix frames at current AR step.
        chunk_size: l, number of frames in denoising target.
        use_kv_cache: If True, count Ca2-VDM FLOPs; else count baseline FLOPs.
        resolution: Video resolution.
        patch_size: Spatial patch size.

    Returns:
        Dict with FLOPs for each attention type.
    """
    hidden_size = model_config.get("hidden_size", 1152)
    num_heads = model_config.get("num_heads", 16)
    depth = model_config.get("depth", 28)
    context_dim = model_config.get("context_dim", 4096)
    context_len = model_config.get("context_len", 120)  # T5 max length
    prefix_len = model_config.get("prefix_len", 3)  # P'

    # Spatial dimensions after VAE (8x downsampling) and patch embedding
    latent_h = resolution // 8
    latent_w = resolution // 8
    h_p = latent_h // patch_size
    w_p = latent_w // patch_size
    hw = h_p * w_p  # Number of spatial tokens per frame

    # Batch size = 1 for FLOPs counting
    B = 1

    flops = {
        "temporal_attention": 0,
        "spatial_attention": 0,
        "cross_attention": 0,
        "total": 0,
    }

    for _ in range(depth):
        # --- Temporal Attention ---
        if use_kv_cache:
            # Ca2-VDM: Only process denoising target (l frames)
            # KV-cache provides p_k frames as context
            # Query: l frames, KV: p_k + l frames
            # Batch dimension: B * hw (spatial grid as batch)
            temporal_flops = attention_flops(
                seq_len_q=chunk_size,
                seq_len_kv=p_k + chunk_size,
                dim=hidden_size,
                num_heads=num_heads,
                batch_size=B * hw,
            )
        else:
            # Baseline (OS-Ext): Process all p_k + l frames together
            temporal_flops = attention_flops(
                seq_len_q=p_k + chunk_size,
                seq_len_kv=p_k + chunk_size,
                dim=hidden_size,
                num_heads=num_heads,
                batch_size=B * hw,
            )
        flops["temporal_attention"] += temporal_flops

        # --- Spatial Attention ---
        if use_kv_cache:
            # Ca2-VDM: Only process denoising target (l frames)
            # Prefix enhancement: each frame attends to P' prefix frames + itself
            # Batch dimension: B * l (frames as batch)
            spatial_flops = attention_flops(
                seq_len_q=hw,
                seq_len_kv=(prefix_len + 1) * hw,
                dim=hidden_size,
                num_heads=num_heads,
                batch_size=B * chunk_size,
            )
        else:
            # Baseline: Process all p_k + l frames
            spatial_flops = attention_flops(
                seq_len_q=hw,
                seq_len_kv=(prefix_len + 1) * hw,
                dim=hidden_size,
                num_heads=num_heads,
                batch_size=B * (p_k + chunk_size),
            )
        flops["spatial_attention"] += spatial_flops

        # --- Visual-Text Cross Attention ---
        if context_dim is not None:
            if use_kv_cache:
                # Ca2-VDM: Only process denoising target
                cross_flops = attention_flops(
                    seq_len_q=chunk_size * hw,
                    seq_len_kv=context_len,
                    dim=hidden_size,
                    num_heads=num_heads,
                    batch_size=B,
                )
            else:
                # Baseline: Process all frames
                cross_flops = attention_flops(
                    seq_len_q=(p_k + chunk_size) * hw,
                    seq_len_kv=context_len,
                    dim=hidden_size,
                    num_heads=num_heads,
                    batch_size=B,
                )
            flops["cross_attention"] += cross_flops

    flops["total"] = sum(v for k, v in flops.items() if k != "total")
    return flops


def compare_flops(
    model_config: Dict,
    chunk_size: int,
    max_prefix_len: int,
    num_ar_steps: int = 7,
    resolution: int = 256,
    patch_size: int = 2,
) -> Dict[str, list]:
    """
    Compare FLOPs between Ca2-VDM and OS-Ext across AR steps.

    Reproduces Figure 8 from the paper.

    Args:
        model_config: Model hyperparameters.
        chunk_size: l, frames per AR step.
        max_prefix_len: P_max.
        num_ar_steps: Number of AR steps to simulate.
        resolution: Video resolution.
        patch_size: Spatial patch size.

    Returns:
        Dict with FLOPs lists for Ca2-VDM and OS-Ext.
    """
    results = {
        "ca2_vdm": {"temporal": [], "spatial": [], "cross": [], "total": []},
        "os_ext": {"temporal": [], "spatial": [], "cross": [], "total": []},
        "ar_steps": list(range(1, num_ar_steps + 1)),
    }

    for step in range(1, num_ar_steps + 1):
        # p_k grows with each AR step, capped at max_prefix_len
        p_k = min((step - 1) * chunk_size + 1, max_prefix_len)

        # Ca2-VDM FLOPs
        ca2_flops = count_flops(
            model_config, p_k, chunk_size,
            use_kv_cache=True,
            resolution=resolution,
            patch_size=patch_size,
        )
        results["ca2_vdm"]["temporal"].append(ca2_flops["temporal_attention"])
        results["ca2_vdm"]["spatial"].append(ca2_flops["spatial_attention"])
        results["ca2_vdm"]["cross"].append(ca2_flops["cross_attention"])
        results["ca2_vdm"]["total"].append(ca2_flops["total"])

        # OS-Ext FLOPs (bidirectional baseline with extendable condition)
        os_flops = count_flops(
            model_config, p_k, chunk_size,
            use_kv_cache=False,
            resolution=resolution,
            patch_size=patch_size,
        )
        results["os_ext"]["temporal"].append(os_flops["temporal_attention"])
        results["os_ext"]["spatial"].append(os_flops["spatial_attention"])
        results["os_ext"]["cross"].append(os_flops["cross_attention"])
        results["os_ext"]["total"].append(os_flops["total"])

    return results
