"""
Weight-space uncertainty representations for LUNO.

Implements:
  - IsotropicGaussian: w ~ N(w*, σ²I)
  - LowRankLaplace: w ~ N(w*, (n*VV^T + σ^{-2}I)^{-1}) via GGN low-rank approximation

The GGN is computed as G = Σ_i J_i^T J_i (for MSE loss, H_f = I).
Low-rank approximation: G ≈ VV^T where V ∈ R^{p × r} via randomized SVD / Lanczos.

Reference: Dangel et al. (2022) ViViT; Immer et al. (2021) LLA.
"""

from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


class IsotropicGaussian(NamedTuple):
    """
    Isotropic Gaussian weight-space uncertainty: w ~ N(w*, σ²I).

    Attributes:
        mean: flat parameter vector w* of shape (p,)
        sigma2: scalar variance σ²
    """
    mean: jax.Array
    sigma2: float


class LowRankLaplace(NamedTuple):
    """
    Low-rank Laplace approximation of the posterior.

    The GGN is approximated as G ≈ VV^T where V ∈ R^{p × r}.
    The posterior covariance is Σ = (n * VV^T + σ^{-2} I)^{-1}.

    Via Woodbury identity:
      Σ = σ² I - σ⁴ V (n^{-1} I + σ² V^T V)^{-1} V^T

    Attributes:
        mean: flat parameter vector w* of shape (p,)
        V: low-rank factor of GGN, shape (p, rank)
        n_data: number of data points used to compute GGN
        sigma2_prior: prior variance σ² (calibrated)
    """
    mean: jax.Array
    V: jax.Array
    n_data: int
    sigma2_prior: float


def get_flat_params(model: nnx.Module) -> jax.Array:
    """Extract all parameters as a flat vector."""
    graphdef, state = nnx.split(model)
    leaves = jax.tree_util.tree_leaves(state)
    return jnp.concatenate([x.ravel() for x in leaves])


def set_flat_params(model: nnx.Module, flat_params: jax.Array) -> nnx.Module:
    """Set model parameters from a flat vector."""
    graphdef, state = nnx.split(model)
    leaves, treedef = jax.tree_util.tree_flatten(state)
    shapes = [x.shape for x in leaves]
    sizes = [int(np.prod(s)) for s in shapes]
    splits = np.cumsum(sizes[:-1])
    new_leaves_flat = jnp.split(flat_params, splits)
    new_leaves = [v.reshape(s) for v, s in zip(new_leaves_flat, shapes)]
    new_state = jax.tree_util.tree_unflatten(treedef, new_leaves)
    return nnx.merge(graphdef, new_state)


def model_fn_flat(
    model: nnx.Module,
    flat_params: jax.Array,
    a: jax.Array,
) -> jax.Array:
    """
    Evaluate model with given flat parameter vector.

    Args:
        model: FNO model (used for structure only)
        flat_params: (p,) flat parameter vector
        a: input function discretization
    Returns:
        output of shape matching model output
    """
    model_with_params = set_flat_params(model, flat_params)
    return model_with_params(a)


def compute_jacobian_vector_product(
    model: nnx.Module,
    flat_params: jax.Array,
    a: jax.Array,
    v: jax.Array,
) -> jax.Array:
    """
    Compute J(a) @ v where J is the Jacobian of the model output w.r.t. parameters.

    Uses forward-mode AD (jvp) for efficiency.

    Args:
        model: FNO model
        flat_params: (p,) MAP parameters
        a: input function, shape (batch, ...)
        v: tangent vector, shape (p,)
    Returns:
        J @ v, shape matching model output
    """
    def f(params):
        return model_fn_flat(model, params, a)

    _, jvp_out = jax.jvp(f, (flat_params,), (v,))
    return jvp_out


def compute_vector_jacobian_product(
    model: nnx.Module,
    flat_params: jax.Array,
    a: jax.Array,
    v: jax.Array,
) -> jax.Array:
    """
    Compute J(a)^T @ v (vector-Jacobian product).

    Uses reverse-mode AD (vjp) for efficiency.

    Args:
        model: FNO model
        flat_params: (p,) MAP parameters
        a: input function
        v: cotangent vector, shape matching model output
    Returns:
        J^T @ v, shape (p,)
    """
    def f(params):
        return model_fn_flat(model, params, a)

    _, vjp_fn = jax.vjp(f, flat_params)
    return vjp_fn(v)[0]


def compute_ggn_low_rank(
    model: nnx.Module,
    flat_params: jax.Array,
    data_loader,
    rank: int = 500,
    n_data: Optional[int] = None,
    key: Optional[jax.Array] = None,
) -> Tuple[jax.Array, int]:
    """
    Compute low-rank approximation of the GGN matrix.

    For MSE loss, the GGN is G = Σ_i J_i^T J_i where J_i is the Jacobian
    of the model output w.r.t. parameters at input a^(i).

    We compute the top-r eigenvectors of G using a randomized approach:
    1. Form the matrix B = [J_1, J_2, ..., J_n] ∈ R^{p × (n * d_out)}
    2. Compute SVD of B to get V (top-r left singular vectors)

    In practice, we use the Lanczos algorithm via matrix-vector products
    with G: G @ v = Σ_i J_i^T (J_i @ v).

    Args:
        model: FNO model
        flat_params: (p,) MAP parameters
        data_loader: iterable of (a, u) batches
        rank: number of eigenvectors to compute
        n_data: max number of data points to use
        key: JAX random key
    Returns:
        V: (p, rank) low-rank factor
        n_used: number of data points used
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    p = flat_params.shape[0]

    # Collect all Jacobians via VJPs
    # For each data point, J_i^T @ e_j for all output dimensions
    # This is memory-intensive; we use a streaming approach

    # GGN-vector product: G @ v = Σ_i J_i^T (J_i @ v)
    def ggn_matvec(v: jax.Array, data_list) -> jax.Array:
        result = jnp.zeros_like(v)
        for a_batch, _ in data_list:
            # J_i @ v: forward pass tangent
            def f(params):
                return model_fn_flat(model, params, a_batch)

            _, jvp_out = jax.jvp(f, (flat_params,), (v,))
            # J_i^T @ (J_i @ v): backward pass
            _, vjp_fn = jax.vjp(f, flat_params)
            result = result + vjp_fn(jvp_out)[0]
        return result

    # Collect data
    data_list = []
    n_used = 0
    for batch in data_loader:
        data_list.append(batch)
        n_used += batch[0].shape[0]
        if n_data is not None and n_used >= n_data:
            break

    # Randomized Lanczos / power iteration for top-r eigenvectors
    # We use a randomized SVD approach on the stacked Jacobian matrix
    # For efficiency, we use the Lanczos algorithm

    # Initialize with random vectors
    key, subkey = jax.random.split(key)
    Q = jax.random.normal(subkey, (p, rank))
    Q, _ = jnp.linalg.qr(Q)  # Orthonormalize

    # Power iteration to find top eigenvectors of G
    n_iter = 10
    for _ in range(n_iter):
        Z = jax.vmap(lambda v: ggn_matvec(v, data_list), in_axes=1, out_axes=1)(Q)
        Q, _ = jnp.linalg.qr(Z)

    # Compute Rayleigh quotients
    GQ = jax.vmap(lambda v: ggn_matvec(v, data_list), in_axes=1, out_axes=1)(Q)
    eigenvalues = jnp.einsum("pi,pi->i", Q, GQ)

    # Sort by eigenvalue magnitude
    idx = jnp.argsort(-eigenvalues)
    Q = Q[:, idx]
    eigenvalues = eigenvalues[idx]

    # V = Q * sqrt(eigenvalues) so that VV^T ≈ G
    sqrt_eigs = jnp.sqrt(jnp.maximum(eigenvalues, 0.0))
    V = Q * sqrt_eigs[None, :]  # (p, rank)

    return V, n_used


def compute_ggn_low_rank_streaming(
    model: nnx.Module,
    flat_params: jax.Array,
    data_loader,
    rank: int = 500,
    n_data: Optional[int] = None,
    key: Optional[jax.Array] = None,
) -> Tuple[jax.Array, int]:
    """
    Compute low-rank GGN approximation using randomized SVD on stacked Jacobians.

    More memory-efficient than full Lanczos for moderate p and n.
    Computes B = [J_1^T, ..., J_n^T] ∈ R^{p × (n*d_out)} and takes SVD.

    For large p, uses the sketch-and-solve approach:
    1. Draw random sketch S ∈ R^{(n*d_out) × r}
    2. Compute Y = B @ S = Σ_i J_i^T (J_i @ S_i)
    3. Orthogonalize Y and compute B^T Y to get small matrix
    4. SVD of small matrix gives approximate top singular vectors

    Args:
        model: FNO model
        flat_params: (p,) MAP parameters
        data_loader: iterable of (a, u) batches
        rank: approximation rank
        n_data: max data points
        key: JAX random key
    Returns:
        V: (p, rank) such that VV^T ≈ G
        n_used: number of data points used
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    p = flat_params.shape[0]

    # Collect data
    data_list = []
    n_used = 0
    for batch in data_loader:
        data_list.append(batch)
        n_used += batch[0].shape[0]
        if n_data is not None and n_used >= n_data:
            break

    # Determine output dimension from first batch
    first_a = data_list[0][0]
    with jax.disable_jit():
        first_out = model_fn_flat(model, flat_params, first_a)
    d_out_total = int(np.prod(first_out.shape[1:]))  # per-sample output dim

    # Randomized SVD: sketch the Jacobian matrix
    # Y = Σ_i J_i^T @ S_i where S_i ∈ R^{d_out × r}
    oversampling = min(rank + 10, rank * 2)
    key, subkey = jax.random.split(key)

    Y = jnp.zeros((p, oversampling))

    for a_batch, _ in data_list:
        batch_size = a_batch.shape[0]
        key, subkey = jax.random.split(key)
        # Random sketch for this batch
        S = jax.random.normal(subkey, (batch_size * d_out_total, oversampling))

        def f(params):
            out = model_fn_flat(model, params, a_batch)
            return out.reshape(-1)  # flatten batch and output dims

        # J^T @ S columns: for each column s of S, compute J^T @ s
        def vjp_col(s):
            _, vjp_fn = jax.vjp(f, flat_params)
            return vjp_fn(s)[0]

        # Compute Y += J^T @ S
        Y_batch = jax.vmap(vjp_col, in_axes=1, out_axes=1)(S)
        Y = Y + Y_batch

    # Orthogonalize Y
    Q, _ = jnp.linalg.qr(Y)  # (p, oversampling)

    # Compute B^T Q = Σ_i J_i^T (J_i @ Q)
    BtQ = jnp.zeros((p, oversampling))
    for a_batch, _ in data_list:
        def f(params):
            out = model_fn_flat(model, params, a_batch)
            return out.reshape(-1)

        # For each column q of Q, compute J^T (J @ q)
        def ggn_col(q):
            _, jvp_out = jax.jvp(f, (flat_params,), (q,))
            _, vjp_fn = jax.vjp(f, flat_params)
            return vjp_fn(jvp_out)[0]

        BtQ_batch = jax.vmap(ggn_col, in_axes=1, out_axes=1)(Q)
        BtQ = BtQ + BtQ_batch

    # Small matrix eigendecomposition: Q^T (B^T B) Q = Q^T BtQ
    M = Q.T @ BtQ  # (oversampling, oversampling)
    eigenvalues, eigenvectors = jnp.linalg.eigh(M)

    # Take top-rank eigenvectors
    idx = jnp.argsort(-eigenvalues)[:rank]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Map back to full parameter space
    V_small = Q @ eigenvectors  # (p, rank)
    sqrt_eigs = jnp.sqrt(jnp.maximum(eigenvalues, 0.0))
    V = V_small * sqrt_eigs[None, :]  # (p, rank)

    return V, n_used


def posterior_covariance_matvec(
    weight_uncertainty,
    v: jax.Array,
) -> jax.Array:
    """
    Compute Σ @ v for the posterior covariance.

    For IsotropicGaussian: Σ @ v = σ² v
    For LowRankLaplace: Σ = (n VV^T + σ^{-2} I)^{-1}
      Via Woodbury: Σ @ v = σ² v - σ⁴ V (n^{-1} I + σ² V^T V)^{-1} V^T v

    Args:
        weight_uncertainty: IsotropicGaussian or LowRankLaplace
        v: (p,) vector
    Returns:
        Σ @ v, shape (p,)
    """
    if isinstance(weight_uncertainty, IsotropicGaussian):
        return weight_uncertainty.sigma2 * v

    elif isinstance(weight_uncertainty, LowRankLaplace):
        V = weight_uncertainty.V  # (p, rank)
        n = weight_uncertainty.n_data
        sigma2 = weight_uncertainty.sigma2_prior

        # Woodbury identity: Σ = σ² I - σ⁴ V (n^{-1} I + σ² V^T V)^{-1} V^T
        VtV = V.T @ V  # (rank, rank)
        M = jnp.eye(V.shape[1]) / n + sigma2 * VtV  # (rank, rank)
        Vt_v = V.T @ v  # (rank,)
        M_inv_Vt_v = jnp.linalg.solve(M, Vt_v)  # (rank,)
        return sigma2 * v - sigma2**2 * (V @ M_inv_Vt_v)

    else:
        raise ValueError(f"Unknown weight uncertainty type: {type(weight_uncertainty)}")


def sample_weights(
    weight_uncertainty,
    n_samples: int,
    key: jax.Array,
) -> jax.Array:
    """
    Draw samples from the weight-space distribution.

    For IsotropicGaussian: w ~ N(w*, σ²I)
    For LowRankLaplace: w ~ N(w*, Σ) using the Woodbury structure

    Args:
        weight_uncertainty: IsotropicGaussian or LowRankLaplace
        n_samples: number of samples
        key: JAX random key
    Returns:
        samples: (n_samples, p)
    """
    mean = weight_uncertainty.mean
    p = mean.shape[0]

    if isinstance(weight_uncertainty, IsotropicGaussian):
        key, subkey = jax.random.split(key)
        eps = jax.random.normal(subkey, (n_samples, p))
        return mean[None, :] + jnp.sqrt(weight_uncertainty.sigma2) * eps

    elif isinstance(weight_uncertainty, LowRankLaplace):
        V = weight_uncertainty.V  # (p, rank)
        n = weight_uncertainty.n_data
        sigma2 = weight_uncertainty.sigma2_prior
        rank = V.shape[1]

        # Sample from N(0, Σ) using the Woodbury structure
        # Σ = σ² I - σ⁴ V M^{-1} V^T where M = n^{-1} I + σ² V^T V
        # Cholesky of Σ: use the formula for low-rank updates
        # Σ = σ² (I - σ² V M^{-1} V^T)
        # = σ² (I - U U^T) where U = σ V M^{-1/2}

        VtV = V.T @ V  # (rank, rank)
        M = jnp.eye(rank) / n + sigma2 * VtV  # (rank, rank)
        L_M = jnp.linalg.cholesky(M)  # (rank, rank)

        # Sample: w = w* + σ ε - σ² V M^{-1} V^T ε
        # where ε ~ N(0, I_p)
        key, k1, k2 = jax.random.split(key, 3)
        eps = jax.random.normal(k1, (n_samples, p))  # (n_samples, p)

        # Correction term: σ² V M^{-1} V^T ε
        Vt_eps = eps @ V  # (n_samples, rank)
        M_inv_Vt_eps = jax.vmap(lambda x: jnp.linalg.solve(M, x))(Vt_eps)  # (n_samples, rank)
        correction = sigma2 * (M_inv_Vt_eps @ V.T)  # (n_samples, p)

        samples = mean[None, :] + jnp.sqrt(sigma2) * eps - sigma2 * correction
        return samples

    else:
        raise ValueError(f"Unknown weight uncertainty type: {type(weight_uncertainty)}")
