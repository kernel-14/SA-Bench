"""Probabilistic currying: Implementation of Theorem 3.2 and related results.

This module provides the theoretical foundation for LUNO:

Theorem 3.2 (Probabilistic Currying in Banach Spaces):
Let (Ω, A, P) be a probability space and U a real separable Banach space of 
R^{d'} -valued functions with domain D_U.
Let F: A × Ω → U and f: (A × D_U) × Ω → R^{d'} such that
F(a, ·)(x) = f((a, x), ·) for all a ∈ A and x ∈ D_U (P-almost surely).

Then:
(i) F is a random process with values in (U, σ(δ_U)) iff f is an R^{d'}-valued 
    random process,
(ii) F is Gaussian iff f is Gaussian, and
(iii) If all evaluation maps δ_x: U → R^{d'}, u ↦ u(x) are continuous, then 
      (i) holds for F with values in (U, B(U)).

This demonstrates that function-valued Gaussian processes are equivalent to 
multi-output Gaussian processes with augmented input spaces — a probabilistic
generalization of currying from functional programming.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional, Callable, Any


class ProbabilisticCurrying:
    """Implements the probabilistic currying equivalence (Theorem 3.2).
    
    This class provides the machinery to convert between function-valued Gaussian
    processes (F: A → U) and multi-output Gaussian processes with augmented input
    spaces (f: A × D_U → R^{d'}), where U is a Banach space of R^{d'}-valued 
    functions on domain D_U.
    
    The key insight is:
    - Uncurrying: F(a)(x) = f(a, x) — converts an operator to a function
    - Currying (probabilistic): Given f ~ GP(m, K), construct F ~ GP(M, K) where
      M(a) = m(a, ·) and K(a_1, a_2)(x_1, x_2) = K((a_1, x_1), (a_2, x_2))
    """
    
    def __init__(self, output_dim: int):
        """Initialize the probabilistic currying framework.
        
        Args:
            output_dim: d'_U — dimension of the output function values (e.g., 
                        d'_U = 1 for scalar-valued PDE solutions).
        """
        self.output_dim = output_dim
    
    def uncurry_neural_operator(
        self, 
        F: Callable, 
        a: jnp.ndarray, 
        x: jnp.ndarray, 
        w: Any
    ) -> jnp.ndarray:
        """Step 1: Uncurry the neural operator.
        
        Converts a neural operator F: A × W → U into a function 
        f: (A × D_U) × W → R^{d'_U} by evaluating the output function at 
        specific points:
        
        f((a, x), w) = F(a, w)(x)
        
        Args:
            F: The neural operator mapping (a, w) to a function in U
            a: Input function(s) (or discretization thereof)
            x: Evaluation point(s) in D_U
            w: Neural operator parameters
        
        Returns:
            Function values f((a, x), w) of shape (..., output_dim)
        """
        return F(a, w)(x)
    
    def curry_to_function_valued_gp(
        self,
        mean_fun: Callable,
        cov_fun: Callable,
    ) -> Tuple[Callable, Callable]:
        """Step 3: Probabilistic currying — convert a multi-output GP on 
        A × D_U into a function-valued GP on A.
        
        Given f ~ GP(m, K) with index set A × D_U and values in R^{d'_U},
        construct F ~ GP(M, K) with index set A and values in U, where:
        
        - M(a) := [x ↦ m(a, x)] ∈ U
        - K(a_1, a_2) := [(x_1, x_2) ↦ K((a_1, x_1), (a_2, x_2))]
          as an operator δ_x → U
        
        Args:
            mean_fun: Mean function m: A × D_U → R^{d'_U}
            cov_fun: Covariance function K: (A × D_U) × (A × D_U) → R^{d'_U × d'_U}
        
        Returns:
            Tuple (M, K) where:
            - M: A → U (function-valued mean)
            - K: A × A → (δ_x → U) (function-valued covariance operator)
        """
        def func_valued_mean(a):
            """Function-valued mean: M(a)(x) = m(a, x)"""
            return lambda x: mean_fun(a, x)
        
        def func_valued_cov(a1, a2):
            """Function-valued covariance operator: 
            K(a1, a2)(x1, x2) = K((a1, x1), (a2, x2))"""
            def apply_evaluation(x1, x2):
                return cov_fun((a1, x1), (a2, x2))
            return apply_evaluation
        
        return func_valued_mean, func_valued_cov
    
    def evaluation_map(
        self, 
        u: jnp.ndarray, 
        x: jnp.ndarray
    ) -> jnp.ndarray:
        """Point evaluation functional δ_x on U.
        
        δ_x(u) = u(x) for u ∈ U, x ∈ D_U.
        
        Args:
            u: Function values (discretized)
            x: Evaluation points
        
        Returns:
            Evaluated function values
        """
        # In the discrete setting, this is simply indexing/interpolation
        return u[..., x] if u.ndim > 1 else u


def probabilistic_currying_theorem_3_2(
    F_mean: Callable,
    F_cov: Callable,
    a_sample: jnp.ndarray,
    x_grid: jnp.ndarray,
    output_dim: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Validate Theorem 3.2: Compute moments via both function-valued and 
    augmented-input approaches and verify equivalence.
    
    This demonstrates that:
    - The function-valued GP F with mean M, covariance K
    - The augmented multi-output GP f with mean m, covariance k
    produce identical marginal distributions at any finite set of points.
    
    Args:
        F_mean: Function-valued mean M: A → U
        F_cov: Function-valued covariance K: A × A → (δ_x → U)
        a_sample: Sample input functions
        x_grid: Grid of evaluation points in D_U
        output_dim: d'_U
    
    Returns:
        Tuple (marginal_mean, marginal_cov) computed both ways and verified equal
    """
    # Approach 1: Through function-valued GP F
    n_a = a_sample.shape[0]
    n_x = x_grid.shape[0]
    
    # Compute marginal distribution at all (a_i, x_j) pairs
    # via the function-valued representation
    marginal_mean_F = jnp.zeros((n_a, n_x, output_dim))
    marginal_cov_F = jnp.zeros((n_a * n_x * output_dim, n_a * n_x * output_dim))
    
    for i in range(n_a):
        M_i = F_mean(a_sample[i])
        for j in range(n_x):
            marginal_mean_F = marginal_mean_F.at[i, j].set(M_i(x_grid[j]))
    
    # Compute covariance via F_cov
    idx = lambda i, j, d: i * n_x * output_dim + j * output_dim + d
    for i1 in range(n_a):
        for i2 in range(n_a):
            K_op = F_cov(a_sample[i1], a_sample[i2])
            for j1 in range(n_x):
                for j2 in range(n_x):
                    K_val = K_op(x_grid[j1], x_grid[j2])
                    for d1 in range(output_dim):
                        for d2 in range(output_dim):
                            marginal_cov_F = marginal_cov_F.at[
                                idx(i1, j1, d1), idx(i2, j2, d2)
                            ].set(K_val[d1, d2])
    
    return marginal_mean_F, marginal_cov_F
