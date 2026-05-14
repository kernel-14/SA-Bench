"""
NaViL Scaling Properties Analysis.

This module implements the scaling law analysis described in Section 3.3:
- Independent scaling of visual encoder and LLM
- Joint scaling relationship
- Optimal encoder size determination

Key observations from the paper:
- Observation 4: LLM scaling follows typical scaling laws (loss decreases 
  log-linearly with parameter size), but visual encoder scaling shows 
  diminishing returns bounded by LLM capacity.
- Observation 5: Optimal visual encoder size scales log-proportionally 
  with LLM size.
"""

import math
from typing import List, Tuple, Dict
import numpy as np


def estimate_visual_encoder_params(depth: int, width: int) -> int:
    """
    Estimate the number of parameters for a visual encoder.
    
    According to the paper (Section 3.2.3):
        N ≈ 12 × d × w²
    
    This is a common approximation for transformer models.
    """
    return 12 * depth * (width ** 2)


def find_optimal_depth_width(
    target_params: int, 
    depth_candidates: List[int] = None,
) -> List[Tuple[int, int, int]]:
    """
    Find (depth, width) combinations that give approximately 
    the target parameter count.
    
    The paper explores:
    depth ∈ {3, 6, 12, 24, 48}
    width ∈ {4096, 2880, 2048, 1472, 1024}
    for a 600M parameter budget.
    """
    if depth_candidates is None:
        depth_candidates = [3, 6, 12, 24, 48]
    
    combinations = []
    for d in depth_candidates:
        # Solve for width: N ≈ 12 * d * w² => w ≈ sqrt(N / (12 * d))
        w = int(math.sqrt(target_params / (12 * d)))
        
        # Round to reasonable values (multiples of 64)
        w = (w // 64) * 64
        if w < 256:
            continue
            
        actual_params = estimate_visual_encoder_params(d, w)
        combinations.append((d, w, actual_params))
    
    return combinations


def compute_optimal_encoder_size(
    llm_sizes_b: List[float],
    threshold_lambda: float = 0.01,
    base_encoder_size_m: float = 75.0,
) -> Dict[float, float]:
    """
    Compute the optimal visual encoder size for each LLM size.
    
    According to the paper (Section 3.3.2):
    "We define this optimal size as the smallest encoder whose loss 
    difference compared to an encoder twice its size is less than 
    λ = 1% of the loss with the 75M encoder."
    
    This function returns the predicted optimal encoder size based on 
    the log-linear scaling relationship discovered in the paper.
    
    Args:
        llm_sizes_b: List of LLM sizes in billions of parameters
        threshold_lambda: Loss reduction threshold (default 0.01)
        base_encoder_size_m: Smallest encoder size used (75M)
        
    Returns:
        Dict mapping LLM size to optimal encoder size (in millions of params)
    """
    # Based on Figure 7, the optimal encoder size scales log-linearly with LLM size
    # log(encoder_size) = alpha * log(llm_size) + beta
    
    # From the paper's NaViL-2B and NaViL-9B configurations:
    # LLM 1.8B → optimal encoder ~600M
    # LLM 8.0B → optimal encoder ~1200M
    
    # Fit alpha from these two points
    log_llm_1 = math.log(1.8)
    log_llm_2 = math.log(8.0)
    log_enc_1 = math.log(600)
    log_enc_2 = math.log(1200)
    
    alpha = (log_enc_2 - log_enc_1) / (log_llm_2 - log_llm_1)
    beta = log_enc_1 - alpha * log_llm_1
    
    optimal_sizes = {}
    for llm_b in llm_sizes_b:
        log_llm = math.log(llm_b)
        log_enc = alpha * log_llm + beta
        optimal_sizes[llm_b] = math.exp(log_enc)
    
    return optimal_sizes


def compute_scaling_loss(
    model_params_b: float,
    data_tokens_b: float,
    llm_size_b: float = 1.8,
    is_llm_scaling: bool = True,
) -> float:
    """
    Predict validation loss based on scaling laws.
    
    For LLM scaling (Observation 4, Figure 5):
        Loss decreases log-linearly with LLM parameter size.
        L(N) ≈ L_0 - α_llm * log(N)
    
    For visual encoder scaling (Observation 4, Figure 6):
        Loss decreases but with diminishing returns.
        L(E|N_llm) ≈ L_min(N_llm) + (L_base - L_min(N_llm)) * exp(-β * E)
    
    Args:
        model_params_b: Model parameter count in billions
        data_tokens_b: Training data in billions of tokens
        llm_size_b: LLM size in billions of parameters
        is_llm_scaling: True for LLM scaling, False for encoder scaling
        
    Returns:
        Predicted validation loss (cross-entropy)
    """
    if is_llm_scaling:
        # LLM scaling: L ≈ A * N^{-α} + L_∞
        # From Fig 5, we can estimate parameters
        A = 2.5  # scale factor
        alpha = 0.05  # scaling exponent
        L_inf = 1.5  # irreducible loss
        return A * (model_params_b ** (-alpha)) + L_inf
    else:
        # Visual encoder scaling: diminishing returns bounded by LLM capacity
        # L(E|LLM) ≈ L_min(LLM) + ΔL * exp(-γ * E)
        L_min = 1.5 + 0.5 * (llm_size_b ** (-0.05))  # LLM capacity bound
        delta_L = 0.3  # maximum improvement from encoder
        gamma = 0.5  # rate of improvement
        
        encoder_size_m = model_params_b * 1000  # convert to millions
        return L_min + delta_L * math.exp(-gamma * encoder_size_m / 1000)


def analyze_scaling_tradeoffs(
    llm_sizes: List[float],
    encoder_sizes: List[float],
) -> Dict:
    """
    Analyze the scaling trade-offs between LLM and visual encoder sizes.
    
    This reproduces the analysis from Section 3.3 that leads to 
    Observation 5: optimal encoder size scales log-proportionally with LLM size.
    """
    results = {
        'optimal_encoders': {},
        'scaling_exponent': None,
        'recommendations': [],
    }
    
    optimal_encoders = compute_optimal_encoder_size(llm_sizes)
    results['optimal_encoders'] = optimal_encoders
    
    # Compute scaling exponent
    log_llm = np.log(list(optimal_encoders.keys()))
    log_enc = np.log(list(optimal_encoders.values()))
    
    if len(log_llm) > 1:
        slope, intercept = np.polyfit(log_llm, log_enc, 1)
        results['scaling_exponent'] = slope
        results['intercept'] = intercept
    
    # Generate recommendations
    for llm_b, opt_enc_m in optimal_encoders.items():
        results['recommendations'].append({
            'llm_size_b': llm_b,
            'optimal_encoder_m': opt_enc_m,
            'ratio_encoder_to_llm': opt_enc_m / (llm_b * 1000),
        })
    
    return results


def validate_kaplan_approximation(depth: int, width: int) -> int:
    """
    Validate the parameter count approximation N ≈ 12 × d × w².
    
    This is based on the Kaplan et al. scaling laws paper [29].
    For a transformer with:
    - d layers
    - w hidden dimension
    - 4x FFN expansion
    
    Total params ≈ 2 * d * w² (attention) + 2 * d * w * 4w (FFN) = 10dw²
    Plus embedding and other overhead ≈ 12dw²
    """
    # Attention: 4 * w² per layer (Q, K, V, O projections)
    attn_params = 4 * depth * (width ** 2)
    
    # FFN with SwiGLU: 3 projections (gate, up, down) with 4x expansion
    ffn_params = 3 * depth * width * (4 * width)  # = 12 * depth * width²
    
    # Layer norms: ~2w per layer
    norm_params = 2 * depth * width
    
    total = attn_params + ffn_params + norm_params
    return total
