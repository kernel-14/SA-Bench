"""
SLERP (Spherical Linear Interpolation) implementation for nGPT.

While the paper uses LERP (Linear Interpolation) as an approximation to SLERP,
they note that SLERP is the proper way to recombine on the hypersphere.
This module provides both, along with the Riemannian optimization variant
described in Appendix A.4.
"""

import torch
import math


def slerp(a: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Spherical Linear Interpolation (SLERP) by Shoemake (1985).

    Computes interpolation along the geodesic (shortest path) on a hypersphere.

    SLERP(a, b; α) = sin((1-α)θ)/sin(θ) * a + sin(αθ)/sin(θ) * b

    where θ = arccos(a·b) is the angle between points a and b,
    and α ∈ [0,1] is the interpolation parameter.

    Args:
        a: First point on hypersphere, shape (..., d_model), unit norm
        b: Second point on hypersphere, shape (..., d_model), unit norm
        alpha: Interpolation parameter in [0, 1]
               alpha=0 returns a, alpha=1 returns b

    Returns:
        Interpolated point on the hypersphere
    """
    # Compute cosine of angle
    cos_theta = (a * b).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)

    # Handle small angles (nearly parallel vectors)
    sin_theta = torch.sin(theta)
    small_angle_mask = (sin_theta < 1e-6)

    # For small angles, use LERP
    weight_a = torch.where(
        small_angle_mask,
        1.0 - alpha,
        torch.sin((1.0 - alpha) * theta) / sin_theta
    )
    weight_b = torch.where(
        small_angle_mask,
        alpha,
        torch.sin(alpha * theta) / sin_theta
    )

    return weight_a * a + weight_b * b


def lerp(a: torch.Tensor, b: torch.Tensor, alpha) -> torch.Tensor:
    """
    Linear Interpolation (LERP).

    LERP(a, b; α) = (1-α)a + αb = a + α(b - a)

    The paper finds that LERP approximates SLERP well in practice and
    uses LERP as the default update rule.

    Args:
        a: First point, shape (..., d_model)
        b: Second point, shape (..., d_model)
        alpha: Interpolation parameter (can be vector of per-dimension values)

    Returns:
        Interpolated point
    """
    return a + alpha * (b - a)


def normalizer_retraction(x: torch.Tensor) -> torch.Tensor:
    """
    Normalization as retraction step in Riemannian optimization.
    Maps a point back to the hypersphere manifold.

    Args:
        x: Point in Euclidean space

    Returns:
        Point on the hypersphere (unit norm)
    """
    return x / (x.norm(dim=-1, keepdim=True) + 1e-12)


def riemannian_update(
    h: torch.Tensor,
    h_block: torch.Tensor,
    alpha: torch.Tensor,
    use_projection: bool = False,
) -> torch.Tensor:
    """
    Riemannian optimization update for nGPT.

    The paper describes two variants:

    1. Simple LERP (default, used in main experiments):
       h ← Norm(h + α * (h_block - h))

    2. Riemannian with projection (Appendix A.4):
       h ← Norm(h - B * (h(h^T h_block) - h_block))

    Where the projection of gradient g = h - h_block onto the tangent space:
       g_proj = h(h^T h_block) - h_block

    Args:
        h: Current hidden state, shape (..., d_model), unit norm
        h_block: Output of attention/MLP block, unit norm
        alpha: Eigen learning rates, shape (..., d_model)
        use_projection: Whether to use tangent space projection

    Returns:
        Updated hidden state on hypersphere
    """
    if use_projection:
        # Riemannian projection onto tangent space
        # g = h - h_block is the Euclidean gradient
        # g_proj = h(h^T h_block) - h_block
        dot_product = (h * h_block).sum(dim=-1, keepdim=True)
        g_proj = h * dot_product - h_block  # Projected gradient
        # Variable-metric step
        h_new = h - alpha * g_proj
    else:
        # Simple LERP-based update
        h_new = lerp(h, h_block, alpha)

    # Retraction: map back to hypersphere
    return normalizer_retraction(h_new)


def nGPT_layer_update(
    h: torch.Tensor,
    h_A: torch.Tensor,
    h_M: torch.Tensor,
    alpha_A: torch.Tensor,
    alpha_M: torch.Tensor,
    use_slerp: bool = False,
    use_projection: bool = False,
) -> torch.Tensor:
    """
    Complete nGPT layer update combining attention and MLP blocks.

    Implements equations 10 and 11 from the paper:
    h ← Norm(h + α_A * (h_A - h))
    h ← Norm(h + α_M * (h_M - h))

    Args:
        h: Input hidden state, unit norm
        h_A: Normalized attention output
        h_M: Normalized MLP output
        alpha_A: Eigen learning rates for attention
        alpha_M: Eigen learning rates for MLP
        use_slerp: Use SLERP instead of LERP (experimental)
        use_projection: Use Riemannian projection (experimental)

    Returns:
        Updated hidden state
    """
    if use_slerp:
        # SLERP variant (equation 6)
        h = slerp(h, h_A, alpha_A.unsqueeze(0).unsqueeze(0))
        h = slerp(h, h_M, alpha_M.unsqueeze(0).unsqueeze(0))
    else:
        # LERP variant (default, equations 10 and 11)
        h = riemannian_update(h, h_A, alpha_A, use_projection)
        h = riemannian_update(h, h_M, alpha_M, use_projection)

    return h
