"""
FLOPs and memory analysis for Ca2-VDM vs baselines.

Reproduces the analysis in Figure 8 and Table 6 of the paper.
Compares computational costs of:
- Ca2-VDM with KV-cache
- OS-Ext (bidirectional, extendable condition)
- OS-Fix (bidirectional, fixed condition)
"""

import torch
import numpy as np
from typing import Dict, Tuple


def compute_attention_flops(
    L: int,          # number of target frames
    P_k: int,        # number of conditional frames  
    S: int,          # spatial resolution (H*W)
    D: int,          # hidden dimension
    n_heads: int,    # number of attention heads
    num_layers: int, # number of transformer layers
    P_prime: int = 3,# prefix length for spatial attn
    T_text: int = 77,# text token length
    use_text_cond: bool = True,
    use_kv_cache: bool = False,
) -> Dict[str, float]:
    """
    Compute FLOPs for a single denoising step.
    
    Returns FLOPs in GFLOPs (10^9 operations).
    """
    d = D // n_heads  # head dimension
    
    # Helper: FLOPs for matmul (M,N) @ (N,K) = 2*M*N*K
    # Softmax is ~5 operations per element
    
    results = {}
    
    if use_kv_cache:
        # Ca2-VDM with KV-cache
        # Temporal attention: Q @ K^T where Q is (L*S, d) and K is ((P_k+L)*S, d)
        # But only for L frames
        temp_qkv = 3 * L * S * D * D  # QKV projection for L frames
        temp_attn = 2 * L * S * (P_k + L) * D  # Q@K^T
        temp_softmax = 5 * L * S * (P_k + L)
        temp_proj = L * S * D * D
        temp_ffn = 2 * L * S * D * D * 3  # approximate for FFN
        
        flops_temp = temp_qkv + temp_attn + temp_softmax + temp_proj + temp_ffn
        
        # Spatial attention: only on L frames, prefix-enhanced
        spat_q = L * S * D * D
        spat_kv = L * (P_prime + 1) * S * D * D * 2  # K and V
        spat_attn = 2 * L * S * (P_prime + 1) * S * D
        spat_softmax = 5 * L * S * (P_prime + 1) * S
        spat_proj = L * S * D * D
        
        flops_spat = spat_q + spat_kv + spat_attn + spat_softmax + spat_proj
        
        # Cross attention
        if use_text_cond:
            cross_q = L * S * D * D
            cross_kv = T_text * D * D * 2
            cross_attn = 2 * L * S * T_text * D
            cross_softmax = 5 * L * S * T_text
            cross_proj = L * S * D * D
            flops_cross = cross_q + cross_kv + cross_attn + cross_softmax + cross_proj
        else:
            flops_cross = 0
        
    else:
        # Bidirectional (OS-Ext or OS-Fix): all L+P_k frames processed together
        total_L = L + P_k
        
        # Temporal attention: full bidirectional
        temp_qkv = 3 * total_L * S * D * D
        temp_attn = 2 * total_L * S * total_L * D
        temp_softmax = 5 * total_L * S * total_L
        temp_proj = total_L * S * D * D
        temp_ffn = 2 * total_L * S * D * D * 3
        
        flops_temp = temp_qkv + temp_attn + temp_softmax + temp_proj + temp_ffn
        
        # Spatial attention: same for all frames
        spat_q = total_L * S * D * D
        spat_kv = total_L * S * D * D * 2
        spat_attn = 2 * total_L * S * S * D
        spat_softmax = 5 * total_L * S * S
        spat_proj = total_L * S * D * D
        
        flops_spat = spat_q + spat_kv + spat_attn + spat_softmax + spat_proj
        
        # Cross attention
        if use_text_cond:
            cross_q = total_L * S * D * D
            cross_kv = T_text * D * D * 2
            cross_attn = 2 * total_L * S * T_text * D
            cross_softmax = 5 * total_L * S * T_text
            cross_proj = total_L * S * D * D
            flops_cross = cross_q + cross_kv + cross_attn + cross_softmax + cross_proj
        else:
            flops_cross = 0
    
    # Multiply by number of layers
    total_flops = (flops_temp + flops_spat + flops_cross) * num_layers
    
    # Convert to GFLOPs
    results['temporal_attn'] = flops_temp * num_layers / 1e9
    results['spatial_attn'] = flops_spat * num_layers / 1e9
    results['cross_attn'] = flops_cross * num_layers / 1e9
    results['total'] = total_flops / 1e9
    
    return results


def analyze_ar_step_flops(
    l: int = 8,
    P_max: int = 25,
    num_ar_steps: int = 7,
    H: int = 32,
    W: int = 32,
    D: int = 1152,
    n_heads: int = 16,
    num_layers: int = 28,
    P_prime: int = 3,
    use_text_cond: bool = True,
):
    """
    Analyze FLOPs across autoregressive steps.
    
    Reproduces Figure 8 analysis.
    """
    S = H * W
    T_text = 77
    
    results = {
        'ar_step': [],
        'ca2_vdm_temp': [],
        'ca2_vdm_spat': [],
        'ca2_vdm_cross': [],
        'ca2_vdm_total': [],
        'os_ext_temp': [],
        'os_ext_spat': [],
        'os_ext_cross': [],
        'os_ext_total': [],
    }
    
    for k in range(1, num_ar_steps + 1):
        P_k = min(1 + (k - 1) * l, P_max)
        L = l
        
        # Ca2-VDM with KV-cache
        ca2_flops = compute_attention_flops(
            L=L, P_k=P_k, S=S, D=D, n_heads=n_heads,
            num_layers=num_layers, P_prime=P_prime,
            T_text=T_text, use_text_cond=use_text_cond,
            use_kv_cache=True,
        )
        
        # OS-Ext (bidirectional)
        os_flops = compute_attention_flops(
            L=L, P_k=P_k, S=S, D=D, n_heads=n_heads,
            num_layers=num_layers, P_prime=P_prime,
            T_text=T_text, use_text_cond=use_text_cond,
            use_kv_cache=False,
        )
        
        results['ar_step'].append(k)
        results['ca2_vdm_temp'].append(ca2_flops['temporal_attn'])
        results['ca2_vdm_spat'].append(ca2_flops['spatial_attn'])
        results['ca2_vdm_cross'].append(ca2_flops['cross_attn'])
        results['ca2_vdm_total'].append(ca2_flops['total'])
        results['os_ext_temp'].append(os_flops['temporal_attn'])
        results['os_ext_spat'].append(os_flops['spatial_attn'])
        results['os_ext_cross'].append(os_flops['cross_attn'])
        results['os_ext_total'].append(os_flops['total'])
    
    return results


def analyze_memory(
    l: int = 8,
    P_max: int = 25,
    T: int = 50,  # number of denoising steps
    H: int = 32,
    W: int = 32,
    C: int = 4,
    D: int = 1152,
    n_heads: int = 16,
    num_layers: int = 28,
    P_prime: int = 3,
    use_prefix_enhancement: bool = True,
):
    """
    Analyze GPU memory usage for KV-cache.
    
    Reproduces Table 6 analysis.
    """
    S = H * W
    d = D // n_heads
    
    # KV-cache size per layer (for Ca2-VDM)
    # Temporal cache: (B*S, nH, P_max, d) * 2 (K and V) * num_layers
    # Shared across all T, so only 1 copy
    temporal_cache_bytes = 1 * num_layers * 2 * 1 * S * n_heads * P_max * d * 4  # float32 = 4 bytes
    
    # Spatial cache (if prefix enhancement): (B, P_prime, S, D) * 2 * num_layers
    if use_prefix_enhancement:
        spatial_cache_bytes = 1 * num_layers * 2 * 1 * P_prime * S * D * 4
    else:
        spatial_cache_bytes = 0
    
    total_cache_bytes = temporal_cache_bytes + spatial_cache_bytes
    
    # For Live2diff: cache per denoising step
    # Temporal cache: (T, num_layers, B, S, nH, P_max, d) * 2
    live2diff_cache_bytes = T * num_layers * 2 * 1 * S * n_heads * P_max * d * 4
    
    return {
        'ca2_vdm_cache_gb': total_cache_bytes / 1e9,
        'ca2_vdm_temporal_gb': temporal_cache_bytes / 1e9,
        'ca2_vdm_spatial_gb': spatial_cache_bytes / 1e9,
        'live2diff_cache_gb': live2diff_cache_bytes / 1e9,
    }


if __name__ == '__main__':
    print("=" * 60)
    print("Ca2-VDM FLOPs and Memory Analysis")
    print("=" * 60)
    
    # FLOPs analysis (matching Figure 8)
    print("\nFLOPs per AR step (matching Figure 8):")
    print("Parameters: l=8, P_max=25, H=W=32, D=1152, nH=16, layers=28")
    
    flops_results = analyze_ar_step_flops(
        l=8, P_max=25, num_ar_steps=7,
        H=32, W=32, D=1152, n_heads=16, num_layers=28,
        P_prime=3, use_text_cond=True,
    )
    
    for i in range(7):
        print(f"\nAR step {i+1} (P_k = {min(1+i*8, 25)}):")
        print(f"  Ca2-VDM: {flops_results['ca2_vdm_total'][i]:.2f} GFLOPs")
        print(f"  OS-Ext:  {flops_results['os_ext_total'][i]:.2f} GFLOPs")
        print(f"  Speedup: {flops_results['os_ext_total'][i]/flops_results['ca2_vdm_total'][i]:.2f}x")
    
    # Memory analysis (matching Table 6)
    print("\n\nGPU Memory Analysis (matching Table 6):")
    print("Parameters: l=8, P_max=25, T=50, H=W=32, D=1152")
    
    mem_results = analyze_memory(
        l=8, P_max=25, T=50, H=32, W=32,
        C=4, D=1152, n_heads=16, num_layers=28,
        P_prime=3, use_prefix_enhancement=True,
    )
    
    print(f"Ca2-VDM w/PE total cache: {mem_results['ca2_vdm_cache_gb']:.2f} GB")
    print(f"  Temporal: {mem_results['ca2_vdm_temporal_gb']:.2f} GB")
    print(f"  Spatial:  {mem_results['ca2_vdm_spatial_gb']:.2f} GB")
    print(f"Live2diff (T=50): {mem_results['live2diff_cache_gb']:.2f} GB")
    print(f"Memory saving: {mem_results['live2diff_cache_gb']/mem_results['ca2_vdm_cache_gb']:.1f}x")
    
    # Also compute without PE
    mem_no_pe = analyze_memory(
        l=8, P_max=25, T=50, H=32, W=32,
        C=4, D=1152, n_heads=16, num_layers=28,
        P_prime=3, use_prefix_enhancement=False,
    )
    print(f"\nCa2-VDM w/o PE cache: {mem_no_pe['ca2_vdm_cache_gb']:.2f} GB")
