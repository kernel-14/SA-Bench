"""
LUNO linearization: pushes Gaussian weight-space uncertainty through the neural operator
via model linearization to produce a function-valued Gaussian process.

Steps (Section 3.2 of the paper):
  Step 1: Uncurry F(a,w)(x) -> f((a,x), w)
  Step 2: Linearize f around MAP weights μ:
            f_lin((a,x), w) = f((a,x), μ) + D_w f((a,x), w)|_μ (w - μ)
  Step 3: Probabilistic currying gives F(a) ~ GP(m_a, K_a) with
            m_a(x) = F(a, w*)(x)
            K_a(x1, x2) = J(a,x1) Σ J(a,x2)^T

For last-layer LUNO (Appendix C.1):
  F(a)(x) = q̃(m_{z^{L-1}}(x)) + Dq̃(m_{z^{L-1}}(x)) · (z^{L-1}(x) - m_{z^{L-1}}(x))
  K_a(x1, x2) = Dq̃(m_{z^{L-1}}(x1)) K_{z^{L-1}}(x1, x2) Dq̃(m_{z^{L-1}}(x2))^T

where z^{L-1} is the pre-activation of the last Fourier block, which is linear in w_{L-1}.
"""

from typing import Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from luno.weight_uncertainty import (
    IsotropicGaussian,
    LowRankLaplace,
    get_flat_params,
    model_fn_flat,
    posterior_covariance_matvec,
    set_flat_params,
)


def luno_predictive_mean(
    model: nnx.Module,
    a: jax.Array,
) -> jax.Array:
    """
    Predictive mean: m_a(x) = F(a, w*)(x).

    This is just the standard forward pass of the trained model.

    Args:
        model: trained FNO with MAP weights
        a: input function discretization
    Returns:
        mean prediction, same shape as model output
    """
    return model(a)


def luno_predictive_variance_diagonal(
    model: nnx.Module,
    weight_uncertainty: Union[IsotropicGaussian, LowRankLaplace],
    a: jax.Array,
) -> jax.Array:
    """
    Compute pointwise predictive variance: diag(K_a(x, x)) for all x.

    Var[F(a)(x)] = J(a,x) Σ J(a,x)^T

    For a discretized output of shape (batch, n_x, d_out), computes variance
    at each spatial point x independently.

    Uses the identity: Var = ||Σ^{1/2} J^T||_F^2 = Tr(J Σ J^T)

    For efficiency, we compute this as:
      Var[f_i] = e_i^T J Σ J^T e_i = (J^T e_i)^T Σ (J^T e_i)

    where e_i is the i-th standard basis vector in output space.

    Args:
        model: trained FNO
        weight_uncertainty: Gaussian weight-space belief
        a: input function, shape (1, n_x, d_in) or (1, n_x, n_y, d_in)
    Returns:
        variance: shape matching model output (1, n_x, d_out) or (1, n_x, n_y, d_out)
    """
    flat_params = get_flat_params(model)

    def f(params):
        return model_fn_flat(model, params, a)

    # Get output shape
    out = f(flat_params)
    out_shape = out.shape  # (1, n_x, d_out) or (1, n_x, n_y, d_out)
    out_flat = out.reshape(-1)  # (n_x * d_out,)
    n_out = out_flat.shape[0]

    # Compute variance for each output dimension
    # Var[f_i] = (J^T e_i)^T Σ (J^T e_i)
    def var_single_output(i):
        e_i = jnp.zeros(n_out).at[i].set(1.0)
        e_i_shaped = e_i.reshape(out_shape)
        # J^T e_i via VJP
        _, vjp_fn = jax.vjp(f, flat_params)
        jt_ei = vjp_fn(e_i_shaped)[0]  # (p,)
        # Σ (J^T e_i)
        sigma_jt_ei = posterior_covariance_matvec(weight_uncertainty, jt_ei)
        # (J^T e_i)^T Σ (J^T e_i)
        return jnp.dot(jt_ei, sigma_jt_ei)

    variances = jax.vmap(var_single_output)(jnp.arange(n_out))
    return variances.reshape(out_shape)


def luno_predictive_variance_diagonal_batched(
    model: nnx.Module,
    weight_uncertainty: Union[IsotropicGaussian, LowRankLaplace],
    a: jax.Array,
    chunk_size: int = 64,
) -> jax.Array:
    """
    Memory-efficient version of luno_predictive_variance_diagonal.

    Processes output dimensions in chunks to avoid OOM.

    Args:
        model: trained FNO
        weight_uncertainty: Gaussian weight-space belief
        a: input function
        chunk_size: number of output dimensions to process at once
    Returns:
        variance: shape matching model output
    """
    flat_params = get_flat_params(model)

    def f(params):
        return model_fn_flat(model, params, a)

    out = f(flat_params)
    out_shape = out.shape
    n_out = int(np.prod(out_shape))

    variances = []
    for start in range(0, n_out, chunk_size):
        end = min(start + chunk_size, n_out)
        chunk_vars = []
        for i in range(start, end):
            e_i = jnp.zeros(n_out).at[i].set(1.0)
            e_i_shaped = e_i.reshape(out_shape)
            _, vjp_fn = jax.vjp(f, flat_params)
            jt_ei = vjp_fn(e_i_shaped)[0]
            sigma_jt_ei = posterior_covariance_matvec(weight_uncertainty, jt_ei)
            var_i = jnp.dot(jt_ei, sigma_jt_ei)
            chunk_vars.append(var_i)
        variances.extend(chunk_vars)

    return jnp.array(variances).reshape(out_shape)


def luno_predictive_covariance(
    model: nnx.Module,
    weight_uncertainty: Union[IsotropicGaussian, LowRankLaplace],
    a: jax.Array,
) -> jax.Array:
    """
    Compute full predictive covariance matrix K_a.

    K_a[i, j] = J_i Σ J_j^T where J_i = D_w f((a, x_i), w)|_{w*}

    For output of shape (1, n_x, d_out), returns covariance of shape
    (n_x * d_out, n_x * d_out).

    Args:
        model: trained FNO
        weight_uncertainty: Gaussian weight-space belief
        a: input function, shape (1, ...)
    Returns:
        cov: (n_out, n_out) covariance matrix
    """
    flat_params = get_flat_params(model)

    def f(params):
        return model_fn_flat(model, params, a)

    out = f(flat_params)
    out_shape = out.shape
    n_out = int(np.prod(out_shape))

    # Compute J^T e_i for all i: this gives the Jacobian rows
    def get_jt_ei(i):
        e_i = jnp.zeros(n_out).at[i].set(1.0)
        e_i_shaped = e_i.reshape(out_shape)
        _, vjp_fn = jax.vjp(f, flat_params)
        return vjp_fn(e_i_shaped)[0]  # (p,)

    # Stack all J^T e_i: shape (n_out, p) = J
    J = jax.vmap(get_jt_ei)(jnp.arange(n_out))  # (n_out, p)

    # Compute J Σ J^T
    # For IsotropicGaussian: J Σ J^T = σ² J J^T
    # For LowRankLaplace: J Σ J^T = σ² J J^T - σ⁴ (J V) M^{-1} (J V)^T
    if isinstance(weight_uncertainty, IsotropicGaussian):
        cov = weight_uncertainty.sigma2 * (J @ J.T)

    elif isinstance(weight_uncertainty, LowRankLaplace):
        V = weight_uncertainty.V  # (p, rank)
        n = weight_uncertainty.n_data
        sigma2 = weight_uncertainty.sigma2_prior

        JV = J @ V  # (n_out, rank)
        VtV = V.T @ V  # (rank, rank)
        M = jnp.eye(V.shape[1]) / n + sigma2 * VtV  # (rank, rank)

        cov = sigma2 * (J @ J.T) - sigma2**2 * (JV @ jnp.linalg.solve(M, JV.T))

    return cov


def luno_last_layer_variance(
    model,
    weight_uncertainty: Union[IsotropicGaussian, LowRankLaplace],
    a: jax.Array,
    last_layer_params_mask: Optional[jax.Array] = None,
) -> jax.Array:
    """
    Efficient last-layer LUNO variance computation (Appendix C.1).

    Restricts weight-space uncertainty to the last Fourier block parameters.
    Exploits the linear structure of z^{L-1} in w_{L-1}.

    K_a(x1, x2) = Dq̃(m_{z^{L-1}}(x1)) K_{z^{L-1}}(x1, x2) Dq̃(m_{z^{L-1}}(x2))^T

    where K_{z^{L-1}}(x1, x2) = Φ(x1) Σ_{w_{L-1}} Φ(x2)^T
    and Φ(x) is the feature matrix of z^{L-1} at x.

    For the diagonal variance:
    Var[F(a)(x)] = ||Dq̃(m_{z^{L-1}}(x)) Σ_{w_{L-1}}^{1/2} Φ(x)^T||_F^2

    Args:
        model: trained FNO (FNO1d or FNO2d)
        weight_uncertainty: weight-space belief (restricted to last layer)
        a: input function
        last_layer_params_mask: boolean mask for last-layer parameters
    Returns:
        variance: shape matching model output
    """
    flat_params = get_flat_params(model)

    # Get v^{L-1}: hidden state before last Fourier layer
    v_Lm1 = model.get_last_layer_input(a)  # (batch, n_x, d_v) or (batch, n_x, n_y, d_v)

    # Get last-layer parameters only
    # The last Fourier layer parameters are w_{L-1} = (Re(R^{L-1}), Im(R^{L-1}), W^{L-1})
    # We need to identify which parameters belong to the last Fourier layer

    # Forward from last layer input to get mean prediction
    mean_out = model.forward_from_last_layer_input(v_Lm1)

    # Compute variance using the full Jacobian restricted to last-layer params
    # We use the mask to zero out gradients for non-last-layer parameters
    def f_last_layer(params):
        if last_layer_params_mask is not None:
            # Zero out non-last-layer parameters
            effective_params = jnp.where(last_layer_params_mask, params, flat_params)
        else:
            effective_params = params
        return model_fn_flat(model, effective_params, a)

    out_shape = mean_out.shape
    n_out = int(np.prod(out_shape))

    variances = []
    for i in range(n_out):
        e_i = jnp.zeros(n_out).at[i].set(1.0)
        e_i_shaped = e_i.reshape(out_shape)
        _, vjp_fn = jax.vjp(f_last_layer, flat_params)
        jt_ei = vjp_fn(e_i_shaped)[0]
        if last_layer_params_mask is not None:
            jt_ei = jt_ei * last_layer_params_mask
        sigma_jt_ei = posterior_covariance_matvec(weight_uncertainty, jt_ei)
        var_i = jnp.dot(jt_ei, sigma_jt_ei)
        variances.append(var_i)

    return jnp.array(variances).reshape(out_shape)


def luno_predictive_std(
    model: nnx.Module,
    weight_uncertainty: Union[IsotropicGaussian, LowRankLaplace],
    a: jax.Array,
    chunk_size: int = 256,
) -> jax.Array:
    """
    Compute pointwise predictive standard deviation.

    Args:
        model: trained FNO
        weight_uncertainty: Gaussian weight-space belief
        a: input function
        chunk_size: chunk size for batched variance computation
    Returns:
        std: shape matching model output
    """
    var = luno_predictive_variance_diagonal_batched(
        model, weight_uncertainty, a, chunk_size=chunk_size
    )
    return jnp.sqrt(jnp.maximum(var, 0.0))


def luno_predictive_covariance_low_rank(
    model: nnx.Module,
    weight_uncertainty: Union[IsotropicGaussian, LowRankLaplace],
    a: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Compute low-rank factorization of the predictive covariance.

    Returns eigendecomposition: K_a = U Λ U^T

    Useful for:
    - Drawing functional samples: f ~ m_a + U Λ^{1/2} z, z ~ N(0, I)
    - Computing top eigenfunctions for visualization

    Args:
        model: trained FNO
        weight_uncertainty: Gaussian weight-space belief
        a: input function
    Returns:
        eigenvalues: (n_out,) sorted descending
        eigenvectors: (n_out, n_out) columns are eigenvectors
        mean: (n_out,) mean prediction
    """
    flat_params = get_flat_params(model)

    def f(params):
        return model_fn_flat(model, params, a)

    out = f(flat_params)
    out_shape = out.shape
    n_out = int(np.prod(out_shape))
    mean = out.reshape(-1)

    # Compute full covariance
    cov = luno_predictive_covariance(model, weight_uncertainty, a)

    # Eigendecomposition
    eigenvalues, eigenvectors = jnp.linalg.eigh(cov)
    # Sort descending
    idx = jnp.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors, mean


def luno_sample_functions(
    model: nnx.Module,
    weight_uncertainty: Union[IsotropicGaussian, LowRankLaplace],
    a: jax.Array,
    n_samples: int,
    key: jax.Array,
) -> jax.Array:
    """
    Draw functional samples from the LUNO predictive distribution.

    Samples f ~ GP(m_a, K_a) using the parametric representation:
      f(x) = m_a(x) + J(a,x) Σ^{1/2} z,  z ~ N(0, I_p)

    This is equivalent to sampling w ~ N(w*, Σ) and computing
    f_lin((a,x), w) = f((a,x), w*) + J(a,x)(w - w*)

    Args:
        model: trained FNO
        weight_uncertainty: Gaussian weight-space belief
        a: input function
        n_samples: number of functional samples
        key: JAX random key
    Returns:
        samples: (n_samples, *output_shape)
    """
    from luno.weight_uncertainty import sample_weights

    flat_params = get_flat_params(model)
    mean_out = model(a)
    out_shape = mean_out.shape

    # Sample weight perturbations
    w_samples = sample_weights(weight_uncertainty, n_samples, key)  # (n_samples, p)
    delta_w = w_samples - weight_uncertainty.mean[None, :]  # (n_samples, p)

    def f(params):
        return model_fn_flat(model, params, a)

    # For each sample, compute J @ delta_w_i via JVP
    def sample_fn(dw):
        _, jvp_out = jax.jvp(f, (flat_params,), (dw,))
        return mean_out + jvp_out

    samples = jax.vmap(sample_fn)(delta_w)  # (n_samples, *out_shape)
    return samples
