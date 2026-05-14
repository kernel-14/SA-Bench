"""Linearized Laplace Approximation for Neural Operators.

Implements the linearized Laplace approximation (LLA) following Appendix B.
The LLA is a method for obtaining an approximate posterior distribution over
the parameters w ∈ R^p of a neural network/operator.

Key steps (Appendix B):
1. Train network to find MAP estimate w*
2. Linearize the model: f_lin(x, w) = f(x, w*) + D_w f(x, w)|_{w*} (w - w*)
3. Compute GGN matrix: G = Σ_i J_i^T H_i J_i where J_i = D_w f(x_i, w)|_{w*}
   and H_i = -H_f log p(y_i | f)|_{f(x_i, w*)}
4. Posterior: p(w|D) ≈ N(w; w*, P^†) where P = -H_w log p(w)|_{w*} + G
5. Pushforward: f|D ~ GP(f(·, w*), K) where 
   K(x_1, x_2) = D_w f(x_1, w)|_{w*} P^† D_w f(x_2, w)|_{w*}^T
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional, Callable, Any
from functools import partial


def compute_ggn_matrix(
    model_fn: Callable,
    params: Any,
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    loss_fn: Callable,
    prior_precision: float = 1.0,
) -> jnp.ndarray:
    """Compute the Generalized Gauss-Newton (GGN) matrix.
    
    G = Σ_{i=1}^n D_w f(x_i, w)|_{w*} H_f(-log p(y_i|f))|_{f(x_i,w*)} D_w f(x_i, w)|_{w*}^T
    
    For MSE loss (Gaussian likelihood):
    H_f(-log p(y|f)) = I, so G = Σ_i J_i^T J_i = J^T J
    
    Args:
        model_fn: Function mapping (params, inputs) -> predictions
        params: MAP estimate w*
        inputs: Training inputs
        targets: Training targets  
        loss_fn: Loss function for log-likelihood
        prior_precision: Prior precision for weight prior
    
    Returns:
        GGN matrix of shape (p, p)
    """
    def single_jacobian(x, y):
        """Compute Jacobian J_i = D_w f(x_i, w)|_{w*}"""
        f_fn = lambda p: model_fn(p, x)
        jac = jax.jacrev(f_fn)(params)
        # Flatten Jacobian
        jac_flat = jax.tree_util.tree_flatten(jac)[0]
        return jnp.concatenate([j.reshape(j.shape[0], -1) for j in jac_flat], axis=1)
    
    # Compute per-example Jacobians and accumulate GGN
    # For MSE loss, G = J^T J
    jacobians = []
    for i in range(inputs.shape[0]):
        J_i = single_jacobian(inputs[i], targets[i])
        jacobians.append(J_i)
    
    J = jnp.concatenate(jacobians, axis=0)  # (n*out_dim, p)
    G = J.T @ J
    return G


def compute_low_rank_ggn(
    model_fn: Callable,
    params: Any,
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    rank: int = 500,
    prior_precision: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute low-rank approximation of the GGN matrix.
    
    Using randomized SVD or Lanczos to compute top-k eigenvalues/vectors.
    
    Args:
        model_fn: Function mapping (params, inputs) -> predictions
        params: MAP estimate w*
        inputs: Training inputs
        targets: Training targets
        rank: Number of top eigenvalues to retain
        prior_precision: Prior precision
    
    Returns:
        Tuple (V, lambdas, n_params) where:
        - V: Top-k eigenvectors, shape (p, rank)
        - lambdas: Top-k eigenvalues, shape (rank,)
        - n_params: Total number of parameters p
    """
    # Flatten parameters to vector
    leaves, tree_def = jax.tree_util.tree_flatten(params)
    param_vector = jnp.concatenate([l.ravel() for l in leaves])
    n_params = param_vector.shape[0]
    
    # For practical implementation, use matrix-free GGN-vector products
    # via the approach from Dangel et al. (2022) / Schraudolph (2002)
    
    def ggn_vector_product(v: jnp.ndarray) -> jnp.ndarray:
        """Compute GGN-vector product G @ v without forming G.
        
        Uses the structure: G @ v = Σ_i J_i^T (H_i (J_i @ v))
        For MSE loss with H_i = I: G @ v = J^T (J @ v)
        """
        # Reshape v back to parameter structure
        v_tree = unflatten_like(v, params, tree_def, leaves)
        
        def per_example_contrib(x, y):
            # Forward pass at w*
            f_w = model_fn(params, x)
            
            # JVP: J_i @ v
            _, jvp_val = jax.jvp(lambda p: model_fn(p, x), (params,), (v_tree,))
            
            # For MSE: H_i = I, so H_i @ J_i @ v = J_i @ v
            # VJP: J_i^T @ (H_i @ J_i @ v)
            _, vjp_fn = jax.vjp(lambda p: model_fn(p, x), params)
            vjp_val = vjp_fn(jvp_val)
            
            return vjp_val
        
        # Sum over data points
        total_vjp = None
        for i in range(inputs.shape[0]):
            vjp_i = per_example_contrib(inputs[i], targets[i])
            if total_vjp is None:
                total_vjp = vjp_i
            else:
                total_vjp = jax.tree_util.tree_map(
                    lambda a, b: a + b, total_vjp, vjp_i
                )
        
        # Flatten result
        leaves_result, _ = jax.tree_util.tree_flatten(total_vjp)
        return jnp.concatenate([l.ravel() for l in leaves_result])
    
    # Use randomized SVD via matrix-free matvec
    # This uses the approach: Ω ~ N(0, 1)^{p × (rank+p)}, compute Y = G @ Ω
    # Then QR decompose Y = Q R, form T = Q^T G Q, eigen-decompose T
    
    n_oversamples = 10
    n_random = rank + n_oversamples
    key = jax.random.PRNGKey(0)
    omega = jax.random.normal(key, (n_params, n_random))
    
    # Compute Y = G @ Omega (matrix-free)
    Y = jnp.zeros((n_params, n_random))
    for j in range(n_random):
        Y = Y.at[:, j].set(ggn_vector_product(omega[:, j]))
    
    # QR decomposition
    Q, _ = jnp.linalg.qr(Y)
    Q = Q[:, :rank]  # Keep only rank columns
    
    # Form small matrix T = Q^T G Q
    T = jnp.zeros((rank, rank))
    for j in range(rank):
        T_col = ggn_vector_product(Q[:, j])
        T = T.at[:, j].set(Q.T @ T_col)
    
    # Eigen-decompose T
    eigenvalues, eigenvectors_T = jnp.linalg.eigh(T)
    
    # Transform back: V = Q @ eigenvectors_T
    V = Q @ eigenvectors_T  # (p, rank)
    
    # Sort by descending eigenvalues
    sort_idx = jnp.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_idx]
    V = V[:, sort_idx]
    
    return V, eigenvalues, n_params


def unflatten_like(vector, like, tree_def, leaves):
    """Unflatten a vector into a parameter tree matching the structure of 'like'."""
    shapes = [l.shape for l in leaves]
    sizes = [int(jnp.prod(jnp.array(s))) for s in shapes]
    splits = jnp.split(vector, jnp.cumsum(jnp.array(sizes))[:-1])
    unflattened = []
    for s, shape in zip(splits, shapes):
        unflattened.append(s.reshape(shape))
    return jax.tree_util.tree_unflatten(tree_def, unflattened)


def linearize_model(
    model_fn: Callable,
    params: Any,
    x: jnp.ndarray,
) -> Tuple[jnp.ndarray, Callable]:
    """Linearize the model around params.
    
    f_lin(x, w) = f(x, w*) + D_w f(x, w)|_{w*} (w - w*)
    
    Args:
        model_fn: Model function
        params: Linearization point w*
        x: Input point
    
    Returns:
        Tuple (f_star, jvp_fn) where:
        - f_star = f(x, w*)
        - jvp_fn: v -> D_w f(x, w)|_{w*} @ v
    """
    f_star = model_fn(params, x)
    
    def jvp_fn(v):
        """Compute JVP: D_w f(x, w)|_{w*} @ v"""
        _, jvp = jax.jvp(lambda p: model_fn(p, x), (params,), (v,))
        return jvp
    
    def vjp_fn(v):
        """Compute VJP: v^T D_w f(x, w)|_{w*}"""
        _, vjp_fun = jax.vjp(lambda p: model_fn(p, x), params)
        return vjp_fun(v)
    
    return f_star, jvp_fn, vjp_fn


class LinearizedLaplaceApproximation:
    """Linearized Laplace Approximation for neural operators.
    
    Implements the full LLA pipeline from Appendix B:
    1. Start from trained network with MAP weights w*
    2. Linearize the model around w*
    3. Compute GGN and form posterior covariance
    4. Push forward to obtain GP predictive
    """
    
    def __init__(
        self,
        model_fn: Callable,
        w_star: Any,
        prior_precision: float = 1.0,
        likelihood_precision: float = 1.0,
    ):
        """Initialize LLA.
        
        Args:
            model_fn: Model function f: (w, x) -> y
            w_star: MAP estimate of parameters
            prior_precision: Prior precision (inverse variance)
            likelihood_precision: Likelihood precision (1/σ² for Gaussian likelihood)
        """
        self.model_fn = model_fn
        self.w_star = w_star
        self.prior_precision = prior_precision
        self.likelihood_precision = likelihood_precision
        
        # Flatten parameters
        leaves, self.tree_def = jax.tree_util.tree_flatten(w_star)
        self.param_shapes = [l.shape for l in leaves]
        self.param_sizes = [int(jnp.prod(jnp.array(s))) for s in self.param_shapes]
        self.n_params = sum(self.param_sizes)
        self._leaves_ref = leaves
    
    def flatten_params(self, params: Any) -> jnp.ndarray:
        """Flatten parameters to vector."""
        leaves, _ = jax.tree_util.tree_flatten(params)
        return jnp.concatenate([l.ravel() for l in leaves])
    
    def unflatten_params(self, vector: jnp.ndarray) -> Any:
        """Unflatten vector to parameter tree."""
        splits = jnp.split(vector, jnp.cumsum(jnp.array(self.param_sizes))[:-1])
        unflattened = []
        for s, shape in zip(splits, self.param_shapes):
            unflattened.append(s.reshape(shape))
        return jax.tree_util.tree_unflatten(self.tree_def, unflattened)
    
    def compute_posterior_covariance(
        self,
        train_inputs: jnp.ndarray,
        train_targets: jnp.ndarray,
        low_rank: bool = True,
        rank: int = 500,
    ) -> jnp.ndarray:
        """Compute posterior covariance P^†.
        
        P = -H_w log p(w)|_{w*} + G
        where G is the GGN matrix.
        
        For Gaussian prior N(0, σ_0^2 I): -H_w log p(w) = σ_0^{-2} I
        For Gaussian likelihood N(f, σ^2 I): G = σ^{-2} J^T J
        
        So P = σ_0^{-2} I + σ^{-2} J^T J
        And Σ = P^{-1}
        """
        if low_rank:
            V, lambdas, _ = compute_low_rank_ggn(
                self.model_fn,
                self.w_star,
                train_inputs,
                train_targets,
                rank=rank,
                prior_precision=self.prior_precision,
            )
            # Store for later use
            self._V = V
            self._lambdas = lambdas
            # Σ = (σ_0^{-2} I + σ^{-2} n V V^T)^{-1}
            n = train_inputs.shape[0]
            return LowRankLaplace(
                self.flatten_params(self.w_star),
                V, lambdas, n,
                prior_precision=self.prior_precision,
            )
        else:
            G = compute_ggn_matrix(
                self.model_fn,
                self.w_star,
                train_inputs,
                train_targets,
                loss_fn=None,
                prior_precision=self.prior_precision,
            )
            # P = prior_precision * I + likelihood_precision * G
            P = self.prior_precision * jnp.eye(self.n_params) + self.likelihood_precision * G
            Sigma = jnp.linalg.inv(P)
            return Sigma
    
    def linearized_predictive(
        self,
        x_test: jnp.ndarray,
        posterior_covariance: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute linearized predictive distribution at test points.
        
        f|D ~ GP(f(·, w*), K) where
        K(x_1, x_2) = D_w f(x_1, w)|_{w*} Σ D_w f(x_2, w)|_{w*}^T
        
        Args:
            x_test: Test input points
            posterior_covariance: Σ (or LowRankLaplace object)
        
        Returns:
            Tuple (mean, covariance) of predictive distribution
        """
        # Mean: f(x_test, w*)
        f_mean = self.model_fn(self.w_star, x_test)
        
        # Compute Jacobian at test points
        def flat_jacobian(x):
            f_fn = lambda p: self.model_fn(p, x)
            jac = jax.jacrev(f_fn)(self.w_star)
            leaves, _ = jax.tree_util.tree_flatten(jac)
            return jnp.concatenate([l.reshape(l.shape[0], -1) for l in leaves], axis=1)
        
        J_test = flat_jacobian(x_test)  # (out_dim, n_params)
        
        if hasattr(posterior_covariance, 'get_covariance_matrix'):
            Sigma = posterior_covariance.get_covariance_matrix()
        else:
            Sigma = posterior_covariance
        
        # K = J_test Σ J_test^T
        K_test = J_test @ Sigma @ J_test.T
        
        return f_mean, K_test
