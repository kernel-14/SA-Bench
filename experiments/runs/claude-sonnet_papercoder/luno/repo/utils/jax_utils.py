## utils/jax_utils.py
"""JAX utility functions for the LUNO reproduction.

This module provides shared JAX-specific primitives used across the codebase:
  - Parameter pytree flattening/unflattening (for GGN and weight sampling)
  - Tree-based dot product (for Lanczos inner products)
  - Numerically stable log (for NLL computation)
  - Woodbury matrix identity solver (core of LUNO-LA posterior covariance)

This file has zero dependencies on other project files and must be imported
before any module that performs uncertainty quantification.

All functions are jax.jit-compatible unless otherwise noted.
"""

from __future__ import annotations

import functools
from typing import Callable, Tuple, Union

import jax
import jax.numpy as jnp
from jax import flatten_util


def flatten_params(
    params: dict,
) -> Tuple[jnp.ndarray, Callable[[jnp.ndarray], dict]]:
    """Flatten a Flax parameter pytree into a 1-D array.

    Uses ``jax.flatten_util.ravel_pytree`` which guarantees a deterministic
    leaf traversal order for a fixed pytree structure.  The returned
    ``unflatten_fn`` is a closure that captures the structure and can be
    stored for later use.

    Args:
        params: A nested dict (or any JAX pytree) of parameter arrays.

    Returns:
        A tuple ``(flat, unflatten_fn)`` where

        * ``flat`` is a 1-D ``jnp.ndarray`` containing all parameter values
          concatenated in depth-first order.
        * ``unflatten_fn`` is a callable ``flat -> params`` that reconstructs
          the original pytree from a flat array of the same length.

    Example::

        flat, unflatten = flatten_params(params)
        # ... modify flat ...
        restored = unflatten(flat)
    """
    flat: jnp.ndarray
    unflatten_fn: Callable[[jnp.ndarray], dict]
    flat, unflatten_fn = flatten_util.ravel_pytree(params)
    return flat, unflatten_fn


def unflatten_params(
    flat: jnp.ndarray,
    unflatten_fn: Callable[[jnp.ndarray], dict],
) -> dict:
    """Restore a flat 1-D array to the original parameter pytree.

    This is a thin wrapper around the ``unflatten_fn`` returned by
    :func:`flatten_params`, provided for API consistency and to make the
    intent explicit at call sites.

    Args:
        flat: A 1-D ``jnp.ndarray`` of the same length as the flattened
            parameter vector produced by :func:`flatten_params`.
        unflatten_fn: The callable returned by :func:`flatten_params` for
            the same pytree structure.

    Returns:
        The reconstructed parameter pytree (nested dict of arrays).

    Example::

        flat, unflatten = flatten_params(params)
        new_params = unflatten_params(flat + delta, unflatten)
    """
    return unflatten_fn(flat)


def tree_matvec(tree_v: dict, tree_u: dict) -> jnp.ndarray:
    """Compute the dot product between two parameter pytrees.

    Both pytrees must have identical structure and leaf shapes.  The result
    is the sum of element-wise products across all leaves, equivalent to
    ``jnp.dot(flatten(tree_v), flatten(tree_u))`` but without allocating
    two large flat arrays.

    This function is used in the Lanczos algorithm for computing inner
    products between parameter-space vectors.

    Args:
        tree_v: First parameter pytree.
        tree_u: Second parameter pytree with the same structure as
            ``tree_v``.

    Returns:
        A scalar ``jnp.ndarray`` equal to the dot product of the two
        flattened parameter vectors.

    Example::

        dot = tree_matvec(grad_a, grad_b)
    """
    # Element-wise products across all leaves
    products = jax.tree_util.tree_map(jnp.multiply, tree_v, tree_u)

    # Sum all leaves into a single scalar
    leaves = jax.tree_util.tree_leaves(products)
    result: jnp.ndarray = functools.reduce(
        lambda acc, x: acc + jnp.sum(x),
        leaves,
        jnp.zeros((), dtype=jnp.float32),
    )
    return result


def safe_log(x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Numerically stable element-wise natural logarithm.

    Clamps the argument to ``[eps, inf)`` before taking the log, preventing
    ``log(0) = -inf`` and the resulting NaN propagation in NLL computations.

    Uses ``jnp.maximum`` (not ``x + eps``) so that values well above ``eps``
    are unaffected.

    Args:
        x: Input array of any shape.  Values below ``eps`` are clamped.
        eps: Numerical floor applied before taking the log.  Default
            ``1e-8`` is appropriate for float32 arithmetic.

    Returns:
        ``log(max(x, eps))`` with the same shape and dtype as ``x``.

    Example::

        nll = 0.5 * safe_log(2.0 * jnp.pi * sigma_sq) + residual_sq / (2.0 * sigma_sq)
    """
    return jnp.log(jnp.maximum(x, eps))


def woodbury_solve(
    eigvecs: jnp.ndarray,
    eigvals: jnp.ndarray,
    prior_prec: float,
    n_data: int,
    v: jnp.ndarray,
) -> jnp.ndarray:
    """Apply the inverse of the LUNO-LA posterior precision matrix to a vector.

    Computes ``(n_data * G_approx + prior_prec * I)^{-1} * v`` efficiently
    using the Woodbury matrix identity, where the GGN approximation is

    .. math::

        G_{\\text{approx}} = V_{\\text{raw}} \\, \\text{diag}(\\lambda) \\, V_{\\text{raw}}^\\top

    with ``eigvecs = V_raw`` (shape ``[p, rank]``) and
    ``eigvals = lambda`` (shape ``[rank]``).

    The full precision matrix is therefore

    .. math::

        P = n \\cdot V_{\\text{raw}} \\, \\text{diag}(\\lambda) \\, V_{\\text{raw}}^\\top
            + \\sigma \\cdot I

    where :math:`\\sigma` = ``prior_prec`` and :math:`n` = ``n_data``.

    Defining the effective factor matrix
    :math:`V_{\\text{eff}} = V_{\\text{raw}} \\, \\text{diag}(\\sqrt{\\lambda})`,
    the Woodbury identity gives

    .. math::

        P^{-1} v = \\frac{1}{\\sigma} v
                   - \\frac{1}{\\sigma^2} V_{\\text{eff}}
                     \\left(n I + \\frac{1}{\\sigma} V_{\\text{eff}}^\\top V_{\\text{eff}}\\right)^{-1}
                     V_{\\text{eff}}^\\top v

    The inner system is ``rank × rank`` (500 × 500), making this feasible
    even when ``p`` is in the millions.

    Args:
        eigvecs: Eigenvectors of the GGN, shape ``[p, rank]``.  Columns are
            unit-norm eigenvectors.
        eigvals: Corresponding eigenvalues, shape ``[rank]``.  Non-negative;
            small negative values from floating-point errors are clipped to
            zero internally.
        prior_prec: Prior precision scalar :math:`\\sigma > 0`.  Acts as the
            isotropic regulariser.
        n_data: Number of data points :math:`n` used to scale the GGN.
        v: Vector(s) to multiply.  Shape ``[p]`` for a single vector or
            ``[p, k]`` for a batch of ``k`` vectors.  Both cases are handled
            transparently.

    Returns:
        ``P^{-1} * v`` with the same shape as ``v``.

    Raises:
        ValueError: If ``prior_prec <= 0``.

    Notes:
        * ``eigvals`` are clipped to ``[0, inf)`` before taking the square
          root to guard against small negative values from Lanczos.
        * The ``rank × rank`` inner matrix is recomputed on every call.
          Callers that invoke this function repeatedly with the same
          ``eigvecs``, ``eigvals``, ``prior_prec``, and ``n_data`` should
          precompute and cache the inner matrix externally (e.g., in
          ``LaplaceApprox.set_prior_prec``).
        * This function is ``jax.jit``-compatible.  ``prior_prec`` and
          ``n_data`` are treated as traced values; recompilation occurs only
          when the *shapes* of the inputs change.

    Example::

        # Compute marginal variance at a single output point
        # J has shape [out_dim, p_last]
        # sigma_sq[i] = J[i] @ Sigma @ J[i]
        Sigma_J = woodbury_solve(eigvecs, eigvals, prior_prec, n_data, J.T)
        # Sigma_J has shape [p_last, out_dim]
        marginal_var = jnp.sum(J * Sigma_J.T, axis=-1)  # shape [out_dim]
    """
    if prior_prec <= 0:
        raise ValueError(
            f"prior_prec must be positive, got {prior_prec}"
        )

    rank: int = eigvecs.shape[1]

    # ------------------------------------------------------------------
    # Step 0: Scale eigenvectors by sqrt(eigenvalues) to form V_eff.
    # Clip eigenvalues to [0, inf) to guard against small negative values
    # produced by the Lanczos algorithm due to floating-point arithmetic.
    # V_eff has shape [p, rank].
    # ------------------------------------------------------------------
    sqrt_eigvals: jnp.ndarray = jnp.sqrt(jnp.maximum(eigvals, 0.0))  # [rank]
    v_eff: jnp.ndarray = eigvecs * sqrt_eigvals[jnp.newaxis, :]  # [p, rank]

    # ------------------------------------------------------------------
    # Step 1: Compute V_eff^T * v.
    # For v of shape [p]:    result is [rank].
    # For v of shape [p, k]: result is [rank, k].
    # ------------------------------------------------------------------
    vt_v: jnp.ndarray = v_eff.T @ v  # [rank] or [rank, k]

    # ------------------------------------------------------------------
    # Step 2: Form the rank × rank inner matrix.
    # M_inner = n_data * I + (1 / prior_prec) * V_eff^T * V_eff
    # This matrix is symmetric positive definite.
    # ------------------------------------------------------------------
    vtv: jnp.ndarray = v_eff.T @ v_eff  # [rank, rank]
    m_inner: jnp.ndarray = (
        n_data * jnp.eye(rank, dtype=v_eff.dtype)
        + (1.0 / prior_prec) * vtv
    )  # [rank, rank]

    # ------------------------------------------------------------------
    # Step 3: Solve M_inner * z = V_eff^T * v.
    # jnp.linalg.solve handles batched right-hand sides:
    #   shape [rank] -> z shape [rank]
    #   shape [rank, k] -> z shape [rank, k]
    # ------------------------------------------------------------------
    z: jnp.ndarray = jnp.linalg.solve(m_inner, vt_v)  # [rank] or [rank, k]

    # ------------------------------------------------------------------
    # Step 4: Apply the Woodbury formula.
    # result = (1/sigma) * v - (1/sigma^2) * V_eff * z
    # ------------------------------------------------------------------
    result: jnp.ndarray = (
        (1.0 / prior_prec) * v
        - (1.0 / (prior_prec ** 2)) * (v_eff @ z)
    )  # same shape as v

    return result


def woodbury_marginal_variance(
    eigvecs: jnp.ndarray,
    eigvals: jnp.ndarray,
    prior_prec: float,
    n_data: int,
    jacobian: jnp.ndarray,
) -> jnp.ndarray:
    """Compute marginal variances ``diag(J * Sigma * J^T)`` via Woodbury.

    For each row ``j_i`` of the Jacobian matrix ``J`` (shape ``[m, p]``),
    computes the scalar ``j_i^T * Sigma * j_i`` where
    ``Sigma = (n_data * G_approx + prior_prec * I)^{-1}``.

    This is equivalent to calling :func:`woodbury_solve` with ``v = J^T``
    (shape ``[p, m]``) and then taking the row-wise dot product with ``J``,
    but is expressed explicitly for clarity.

    Args:
        eigvecs: Eigenvectors of the GGN, shape ``[p, rank]``.
        eigvals: Eigenvalues of the GGN, shape ``[rank]``.
        prior_prec: Prior precision scalar :math:`\\sigma > 0`.
        n_data: Number of data points used to scale the GGN.
        jacobian: Jacobian matrix, shape ``[m, p]``, where ``m`` is the
            number of output dimensions (e.g., spatial points × output
            channels).

    Returns:
        Marginal variances, shape ``[m]``.  Entry ``i`` equals
        ``jacobian[i] @ Sigma @ jacobian[i]``.

    Example::

        # jacobian has shape [n_spatial * d_out, p_last]
        var = woodbury_marginal_variance(eigvecs, eigvals, prior_prec, n_data, jacobian)
        std = jnp.sqrt(jnp.maximum(var, 0.0))
    """
    # Sigma * J^T has shape [p, m]
    sigma_jt: jnp.ndarray = woodbury_solve(
        eigvecs=eigvecs,
        eigvals=eigvals,
        prior_prec=prior_prec,
        n_data=n_data,
        v=jacobian.T,  # [p, m]
    )  # [p, m]

    # diag(J * Sigma * J^T) = sum_p J[i, p] * (Sigma * J^T)[p, i]
    # = einsum('mp,pm->m', jacobian, sigma_jt)
    marginal_var: jnp.ndarray = jnp.einsum("mp,pm->m", jacobian, sigma_jt)
    return marginal_var


def woodbury_sample(
    eigvecs: jnp.ndarray,
    eigvals: jnp.ndarray,
    prior_prec: float,
    n_data: int,
    mean: jnp.ndarray,
    key: jax.Array,
    n_samples: int = 1,
) -> jnp.ndarray:
    """Draw samples from the LUNO-LA posterior weight distribution.

    Samples from ``N(mean, Sigma)`` where
    ``Sigma = (n_data * G_approx + prior_prec * I)^{-1}`` using the
    reparameterisation trick:

    .. math::

        w = \\mu + \\Sigma^{1/2} \\varepsilon, \\quad \\varepsilon \\sim N(0, I)

    The square root ``Sigma^{1/2}`` is computed via the low-rank structure:

    .. math::

        \\Sigma^{1/2} \\varepsilon
        = \\frac{1}{\\sqrt{\\sigma}} \\varepsilon
          - \\frac{1}{\\sigma} V_{\\text{eff}}
            \\left(n I + \\frac{1}{\\sigma} V_{\\text{eff}}^\\top V_{\\text{eff}}\\right)^{-1/2}
            V_{\\text{eff}}^\\top \\varepsilon

    In practice we use the Cholesky decomposition of the small inner matrix
    to obtain the square root, then apply the Woodbury-like formula.

    Args:
        eigvecs: Eigenvectors of the GGN, shape ``[p, rank]``.
        eigvals: Eigenvalues of the GGN, shape ``[rank]``.
        prior_prec: Prior precision scalar :math:`\\sigma > 0`.
        n_data: Number of data points used to scale the GGN.
        mean: MAP weight vector, shape ``[p]``.
        key: JAX PRNG key for sampling.
        n_samples: Number of samples to draw.  Default 1.

    Returns:
        Weight samples, shape ``[n_samples, p]``.

    Notes:
        The sampling uses the exact posterior covariance (within the
        low-rank approximation), not an approximation.  The Cholesky
        decomposition of the ``rank × rank`` inner matrix is used for
        numerical stability.
    """
    p: int = eigvecs.shape[0]
    rank: int = eigvecs.shape[1]

    # Scale eigenvectors: V_eff = eigvecs * diag(sqrt(eigvals))
    sqrt_eigvals: jnp.ndarray = jnp.sqrt(jnp.maximum(eigvals, 0.0))  # [rank]
    v_eff: jnp.ndarray = eigvecs * sqrt_eigvals[jnp.newaxis, :]  # [p, rank]

    # Inner matrix: M = n_data * I + (1/prior_prec) * V_eff^T * V_eff
    vtv: jnp.ndarray = v_eff.T @ v_eff  # [rank, rank]
    m_inner: jnp.ndarray = (
        n_data * jnp.eye(rank, dtype=v_eff.dtype)
        + (1.0 / prior_prec) * vtv
    )  # [rank, rank]

    # Cholesky of M_inner for stable square root computation
    # M_inner = L * L^T  =>  M_inner^{-1/2} involves L^{-T}
    chol_m: jnp.ndarray = jnp.linalg.cholesky(m_inner)  # [rank, rank], lower triangular

    # Draw standard normal samples: eps has shape [n_samples, p]
    key_eps: jax.Array
    key_eps, _ = jax.random.split(key)
    eps: jnp.ndarray = jax.random.normal(key_eps, shape=(n_samples, p))  # [n_samples, p]

    # Compute Sigma^{1/2} * eps using the low-rank structure.
    # Sigma = (1/sigma)*I - (1/sigma^2)*V_eff * M_inner^{-1} * V_eff^T
    # Sigma^{1/2} eps ≈ (1/sqrt(sigma))*eps - correction term
    #
    # We use the exact formula via eigendecomposition of Sigma:
    # Sigma = (1/sigma)*I - (1/sigma^2)*V_eff * M_inner^{-1} * V_eff^T
    # This is a rank-r update of a scaled identity.
    # Sigma^{1/2} can be computed as:
    #   Sigma^{1/2} = (1/sqrt(sigma))*I + V_eff * D * V_eff^T
    # where D is chosen so that Sigma^{1/2} * Sigma^{1/2} = Sigma.
    #
    # For sampling, it's simpler to use the reparameterisation directly:
    # w = mu + Sigma^{1/2} * eps
    # We compute this as: apply woodbury_solve to eps^T, then take sqrt.
    # Actually, the cleanest approach for sampling is:
    #
    # w = mu + (1/sqrt(sigma)) * eps_perp + V_eff * alpha
    # where eps_perp is the component of eps orthogonal to V_eff's column space,
    # and alpha is chosen to match the covariance in the V_eff subspace.
    #
    # Concretely, using the low-rank structure of Sigma:
    # Sigma = U * diag(d) * U^T  (eigendecomposition)
    # where U = [V_eff_normalized, V_perp] and d contains the eigenvalues.
    # The eigenvalues of Sigma in the V_eff subspace are the eigenvalues of
    # (1/sigma)*I - (1/sigma^2)*V_eff*M_inner^{-1}*V_eff^T restricted to that subspace.
    #
    # Simpler: use the fact that for a matrix of the form A = c*I + V*B*V^T,
    # we can sample as: x = sqrt(c)*eps + V * (sqrt(B + c*I) - sqrt(c)*I) * V^T * eps / ||V^T*eps||
    # This is complex. Instead, use the direct Cholesky approach on the small system.
    #
    # Most practical approach: sample in the low-rank subspace + isotropic complement.
    # Sigma = (1/sigma)*I - (1/sigma^2)*V_eff*M_inner^{-1}*V_eff^T
    # = (1/sigma)*(I - (1/sigma)*V_eff*M_inner^{-1}*V_eff^T)
    #
    # Eigenvalues of Sigma:
    # - In the column space of V_eff: eigenvalues of (1/sigma)*I - (1/sigma^2)*V_eff*M_inner^{-1}*V_eff^T
    #   restricted to that subspace.
    # - In the orthogonal complement: 1/sigma (isotropic).
    #
    # For the column space of V_eff (rank-dimensional):
    # The matrix restricted to this subspace is (1/sigma)*I_rank - (1/sigma^2)*diag(lambda_eff)
    # where lambda_eff are eigenvalues of V_eff^T*V_eff (which are the squared singular values of V_eff).
    # Actually V_eff^T*V_eff has eigenvalues = eigvals (since V_eff = eigvecs*diag(sqrt(eigvals))
    # and eigvecs are orthonormal, so V_eff^T*V_eff = diag(eigvals)).
    #
    # So in the eigenvector basis of V_eff^T*V_eff = diag(eigvals):
    # Sigma restricted to V_eff subspace has eigenvalues:
    # d_i = 1/sigma - (1/sigma^2) * eigvals[i] / (n_data + eigvals[i]/sigma)
    #      = 1/sigma * (1 - eigvals[i]/(sigma*n_data + eigvals[i]))
    #      = n_data / (sigma*n_data + eigvals[i])
    #      = 1 / (n_data + sigma/eigvals[i]) ... wait let me redo this.
    #
    # M_inner = n_data*I + (1/sigma)*V_eff^T*V_eff = n_data*I + (1/sigma)*diag(eigvals)
    # M_inner^{-1} = diag(1/(n_data + eigvals[i]/sigma))
    #
    # Sigma restricted to V_eff subspace (in eigvecs basis):
    # d_i = 1/sigma - (1/sigma^2) * eigvals[i] * (1/(n_data + eigvals[i]/sigma))
    #      = 1/sigma - eigvals[i] / (sigma^2 * n_data + sigma*eigvals[i])
    #      = 1/sigma - eigvals[i] / (sigma*(sigma*n_data + eigvals[i]))
    #      = (sigma*n_data + eigvals[i] - eigvals[i]) / (sigma*(sigma*n_data + eigvals[i]))
    #      = n_data / (sigma*n_data + eigvals[i])
    #      = 1 / (sigma + eigvals[i]/n_data)  ... hmm
    #
    # Actually: d_i = 1/(n_data * eigvals[i] + sigma)  [this is the standard Laplace result]
    # Let me verify: P = n*G + sigma*I, G = V_raw*diag(lam)*V_raw^T
    # P restricted to eigvec[i] direction: n*lam[i] + sigma
    # So Sigma = P^{-1} restricted to eigvec[i]: 1/(n*lam[i] + sigma)
    # In the orthogonal complement: 1/sigma
    #
    # So the sampling is:
    # w = mu + sum_i sqrt(1/(n*lam[i]+sigma)) * z_i * eigvecs[:,i]
    #       + (1/sqrt(sigma)) * eps_perp
    # where z_i ~ N(0,1) and eps_perp is the component of a N(0,I) vector
    # orthogonal to all eigvecs.

    # Eigenvalues of Sigma in the eigvec subspace
    sigma_eigvals: jnp.ndarray = 1.0 / (
        n_data * jnp.maximum(eigvals, 0.0) + prior_prec
    )  # [rank]

    # Draw samples for the low-rank subspace: z has shape [n_samples, rank]
    key_z: jax.Array
    key_z, key_perp = jax.random.split(key)
    z: jnp.ndarray = jax.random.normal(key_z, shape=(n_samples, rank))  # [n_samples, rank]

    # Low-rank component: eigvecs * diag(sqrt(sigma_eigvals)) * z^T
    # shape: [p, rank] * [rank] * [rank, n_samples] -> [p, n_samples]
    low_rank_component: jnp.ndarray = (
        eigvecs * jnp.sqrt(sigma_eigvals)[jnp.newaxis, :]
    ) @ z.T  # [p, n_samples]

    # Isotropic component: (1/sqrt(prior_prec)) * eps_perp
    # eps_perp = eps - eigvecs * eigvecs^T * eps (project out the eigvec subspace)
    eps_full: jnp.ndarray = jax.random.normal(
        key_perp, shape=(n_samples, p)
    )  # [n_samples, p]
    # Project eps onto eigvec subspace and subtract
    proj: jnp.ndarray = eps_full @ eigvecs  # [n_samples, rank]
    eps_perp: jnp.ndarray = eps_full - proj @ eigvecs.T  # [n_samples, p]
    isotropic_component: jnp.ndarray = (
        (1.0 / jnp.sqrt(prior_prec)) * eps_perp
    )  # [n_samples, p]

    # Combine: samples shape [n_samples, p]
    samples: jnp.ndarray = (
        mean[jnp.newaxis, :]
        + low_rank_component.T
        + isotropic_component
    )  # [n_samples, p]

    return samples
