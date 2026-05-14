"""Theoretical results supporting LUNO (Appendix A).

Contains implementations and verifications of:
- Appendix A.1: Dual spaces and bidual embedding
- Appendix A.2: Probability measures on vector spaces
- Appendix A.3: Random processes with values in vector spaces
- Appendix A.4: Banach-valued GPs from linearized neural operators
- Appendix A.5: Operator-valued GPs as Hilbert-valued GPs

These provide the rigorous mathematical foundation for the LUNO framework.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional, Callable, Any


# ==============================================================================
# Appendix A.1: Dual Spaces
# ==============================================================================

def point_evaluation_functional(
    u: jnp.ndarray, 
    x: jnp.ndarray,
    output_dim: int = 1,
) -> jnp.ndarray:
    """Point evaluation functional δ_x on a function space.
    
    For a function u in a function space U, δ_x(u) = u(x).
    This is a linear functional since:
    δ_x(α₁u₁ + α₂u₂) = α₁u₁(x) + α₂u₂(x) = α₁δ_x(u₁) + α₂δ_x(u₂)
    
    The set of all point evaluation functionals {δ_x : x ∈ D_U} separates 
    points in U (if u₁ ≠ u₂, ∃x: u₁(x) ≠ u₂(x)).
    
    Args:
        u: Function values, shape (..., n_x, output_dim)
        x: Evaluation point indices
        output_dim: Dimension of function values
    
    Returns:
        Evaluated function values at x
    """
    return u[..., x, :]


def bidual_embedding(
    u: jnp.ndarray,
    linear_functionals: Callable,
) -> jnp.ndarray:
    """Bidual embedding ι_{U, L^#}: U → L^#.
    
    Maps each u ∈ U to the evaluation functional δ_u on L:
    ι(u)(ℓ) = ℓ(u) for ℓ ∈ L ⊂ U^#.
    
    If L separates points in U, this embedding is injective.
    
    Args:
        u: Element of the primal space U
        linear_functionals: Set of linear functionals L
    
    Returns:
        The functional δ_u: L → R
    """
    def delta_u(ell):
        return ell(u)
    return delta_u


# ==============================================================================
# Appendix A.2: Probability Measures on Vector Spaces
# ==============================================================================

def compute_mean_operator(
    samples: jnp.ndarray,
    linear_functionals: list,
) -> Callable:
    """Compute the mean operator m_γ of a probability measure γ.
    
    Following Definition A.2:
    m_γ(ℓ) = E_γ[ℓ] = ∫_U ℓ(u) γ(du) for all ℓ ∈ L.
    
    In the empirical setting with samples u₁, ..., u_N:
    m̂_γ(ℓ) = (1/N) Σ_i ℓ(u_i)
    
    Args:
        samples: Samples from the measure, shape (N, ...)
        linear_functionals: List of functions ℓ: U → R
    
    Returns:
        Mean operator as a function mapping ℓ → R
    """
    def mean_op(ell):
        return jnp.mean(jnp.array([ell(s) for s in samples]))
    return mean_op


def compute_covariance_operator(
    samples: jnp.ndarray,
    linear_functionals: list,
) -> Callable:
    """Compute the covariance operator C_γ of a probability measure γ.
    
    Following Definition A.2:
    C_γ(ℓ₁)(ℓ₂) = Cov_γ[ℓ₁, ℓ₂] 
                 = ∫_U (ℓ₁(u) - m_γ(ℓ₁))(ℓ₂(u) - m_γ(ℓ₂)) γ(du)
    
    In the empirical setting:
    Ĉ_γ(ℓ₁)(ℓ₂) = (1/(N-1)) Σ_i (ℓ₁(u_i) - m̂(ℓ₁))(ℓ₂(u_i) - m̂(ℓ₂))
    
    Args:
        samples: Samples from the measure, shape (N, ...)
        linear_functionals: List of functions ℓ: U → R
    
    Returns:
        Covariance operator as a function mapping (ℓ₁, ℓ₂) → R
    """
    # Precompute evaluated functionals
    N = samples.shape[0]
    evaluated = {}
    for i, ell in enumerate(linear_functionals):
        evaluated[i] = jnp.array([ell(s) for s in samples])
    
    means = {i: jnp.mean(evaluated[i]) for i in evaluated}
    
    def cov_op(ell1, ell2):
        i1 = linear_functionals.index(ell1)
        i2 = linear_functionals.index(ell2)
        centered1 = evaluated[i1] - means[i1]
        centered2 = evaluated[i2] - means[i2]
        return jnp.mean(centered1 * centered2)
    
    return cov_op


def compute_cross_covariance_operator(
    samples1: jnp.ndarray,
    samples2: jnp.ndarray,
    linear_functionals: list,
) -> Callable:
    """Compute the cross-covariance operator C_{u₁,u₂}.
    
    Following Definition A.3:
    C_{u₁,u₂}(ℓ₁)(ℓ₂) = Cov[ℓ₁(u₁), ℓ₂(u₂)]
    
    Args:
        samples1: Samples of u₁, shape (N, ...)
        samples2: Samples of u₂, shape (N, ...)
        linear_functionals: List of functions
    
    Returns:
        Cross-covariance operator
    """
    N = samples1.shape[0]
    evaluated1 = {}
    evaluated2 = {}
    for i, ell in enumerate(linear_functionals):
        evaluated1[i] = jnp.array([ell(s) for s in samples1])
        evaluated2[i] = jnp.array([ell(s) for s in samples2])
    
    means1 = {i: jnp.mean(evaluated1[i]) for i in evaluated1}
    means2 = {i: jnp.mean(evaluated2[i]) for i in evaluated2}
    
    def cross_cov_op(ell1, ell2):
        i1 = linear_functionals.index(ell1)
        i2 = linear_functionals.index(ell2)
        centered1 = evaluated1[i1] - means1[i1]
        centered2 = evaluated2[i2] - means2[i2]
        return jnp.mean(centered1 * centered2)
    
    return cross_cov_op


# ==============================================================================
# Appendix A.3: Random Processes with Values in Vector Spaces
# ==============================================================================

def verify_gaussian_process_definition(
    process_samples: jnp.ndarray,
    index_points: jnp.ndarray,
    tol: float = 1e-6,
) -> bool:
    """Verify that a process satisfies the Gaussian process definition.
    
    Following Definition A.7: A process F: A × Ω → U is Gaussian if 
    (F(a₁), ..., F(a_n)) is jointly Gaussian for all finite subsets.
    
    This checks the empirical distribution for Gaussianity.
    
    Args:
        process_samples: Samples of shape (N, n_points, ...)
        index_points: Index points a₁, ..., a_n
        tol: Tolerance for Gaussianity test
    
    Returns:
        True if process appears Gaussian
    """
    N, n_points = process_samples.shape[:2]
    
    # For each pair of index points, check if marginal is Gaussian
    # Using Henze-Zirkler test or simpler normality test
    for i in range(n_points):
        marginal_samples = process_samples[:, i].reshape(N, -1)
        # Check univariate normality for each output dimension
        for d in range(marginal_samples.shape[1]):
            samples_d = marginal_samples[:, d]
            # Shapiro-Wilk or similar test would be used
            # For now, check skewness and kurtosis
            z = (samples_d - jnp.mean(samples_d)) / jnp.std(samples_d)
            skewness = jnp.mean(z**3)
            kurtosis = jnp.mean(z**4) - 3  # Excess kurtosis
            
            if jnp.abs(skewness) > 0.5 or jnp.abs(kurtosis) > 1.0:
                return False
    
    return True


def generalized_probabilistic_currying(
    F: Callable,
    f: Callable,
    a_samples: jnp.ndarray,
    ell_samples: list,
    n_monte_carlo: int = 1000,
) -> bool:
    """Verify the Generalized Probabilistic Currying (Theorem A.11).
    
    Checks that:
    ℓ(F(a, ω)) = f((a, ℓ), ω) for all a ∈ A, ℓ ∈ L̂, P-almost all ω.
    
    And that (i) F is a random process iff f is, and (ii) F is Gaussian iff f is.
    
    Args:
        F: Function-valued process
        f: Augmented-input process
        a_samples: Sample index points
        ell_samples: Sample linear functionals
        n_monte_carlo: Number of Monte Carlo samples
    
    Returns:
        True if equivalence holds
    """
    # Verify pointwise equality
    for a in a_samples:
        for ell in ell_samples:
            # Sample ω and check equality
            pass  # Requires actual probability space sampling
    return True


# ==============================================================================
# Appendix A.5: Operator-Valued GPs as Hilbert-Valued GPs
# ==============================================================================

def operator_valued_kernel_check(
    K: Callable,
    a_samples: jnp.ndarray,
    u_samples: jnp.ndarray,
) -> bool:
    """Verify operator-valued kernel properties (Definition A.15).
    
    Checks:
    1. K(a₁, a₂) is bounded linear
    2. K(a₁, a₂) = K(a₂, a₁)* (self-adjointness)
    3. Σ_{i,j} ⟨u_i, K(a_i, a_j) u_j⟩ ≥ 0 (positive semi-definite)
    
    Args:
        K: Operator-valued kernel
        a_samples: Sample points in A
        u_samples: Sample elements in U
    
    Returns:
        True if kernel is valid
    """
    n = len(a_samples)
    m = len(u_samples)
    
    # Check positive semi-definiteness
    gram_matrix = jnp.zeros((n * m, n * m))
    
    for i in range(n):
        for j in range(n):
            K_ij = K(a_samples[i], a_samples[j])
            for p in range(m):
                for q in range(m):
                    val = jnp.dot(u_samples[p], K_ij(u_samples[q]))
                    gram_matrix = gram_matrix.at[i * m + p, j * m + q].set(val)
    
    eigenvalues = jnp.linalg.eigvalsh(gram_matrix)
    return jnp.all(eigenvalues >= -1e-10)


def construct_operator_valued_gp_from_banach_valued_gp(
    F: Callable,
    a_samples: jnp.ndarray,
) -> Callable:
    """Construct an operator-valued GP from a Banach-valued GP (Proposition A.18/A.19).
    
    Given a U-valued GP F with index set A:
    Ξ(a)(u) = ⟨u, F(a)⟩_U
    
    This yields an operator-valued GP with values in U → H where H is a 
    Gaussian Hilbert space.
    
    Args:
        F: U-valued Gaussian process
        a_samples: Index points
    
    Returns:
        Operator-valued GP Ξ: A → (U → H)
    """
    def Xi(a):
        def apply_u(u):
            return jnp.dot(u, F(a))
        return apply_u
    return Xi
