"""
utils.py
=========
Low‑level numerical utilities for the LUNO reproduction project.

All functions are designed to be pure, composable, and compatible with JAX’s
functional transformations (``jit``, ``vmap``, ``grad``).  Where possible,
computations are returned on the CPU to facilitate inter‑op with non‑JAX code.
"""

from __future__ import annotations

import logging
import pickle
from typing import (
    Any,
    Callable,
    Iterator,
    Optional,
    Tuple,
    Union,
)

import jax
import jax.numpy as jnp
import numpy as np
from jax import flatten_util

logger = logging.getLogger(__name__)


# ============================================================================
# Mesh grid
# ============================================================================


def mesh_grid(
    spatial_res: Union[int, Tuple[int, ...]],
    domain_size: Union[float, Tuple[float, ...]],
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Create coordinate grids on a periodic domain.

    Parameters
    ----------
    spatial_res : int or tuple of int
        Number of grid points in each dimension.  A single ``int`` yields a 1D
        domain; a tuple yields a multi‑dimensional meshgrid.
    domain_size : float or tuple of float
        Length of the domain in each dimension.  Must match the dimensionality
        implied by ``spatial_res``.

    Returns
    -------
    grid : jnp.ndarray or tuple of jnp.ndarray
        1D array of shape ``(spatial_res,)`` when ``spatial_res`` is an ``int``,
        otherwise a tuple ``(xx, yy, ...)`` of 2D arrays each of shape
        ``spatial_res``.
    """
    if isinstance(spatial_res, int):
        res = spatial_res
        L = domain_size if isinstance(domain_size, (int, float)) else domain_size[0]
        return jnp.linspace(0.0, L, res, endpoint=False, dtype=jnp.float32)

    # Multi‑dimensional
    res = spatial_res
    Lx = domain_size[0] if isinstance(domain_size, (list, tuple)) else domain_size
    grids_1d = [jnp.linspace(0.0, L, n, endpoint=False, dtype=jnp.float32) for L, n in zip(domain_size, res)]
    mg = jnp.meshgrid(*grids_1d, indexing="ij")
    return mg if len(mg) > 1 else mg[0]


# ============================================================================
# FFT utilities
# ============================================================================


def rfft(x: jnp.ndarray, spatial_axis: Union[int, Tuple[int, ...]]) -> jnp.ndarray:
    """
    Real‑to‑complex Fourier transform along the given spatial axes.

    Parameters
    ----------
    x : jnp.ndarray
        Input array of shape ``(..., *spatial_dims, ...)``.
    spatial_axis : int or tuple of int
        Axis (or axes) over which to compute the RFFT.

    Returns
    -------
    jnp.ndarray
        Complex tensor with transformed spatial axes.  For a 1D transform the
        Fourier axis has length ``n//2 + 1``; for higher dimensions the last
        transformed axis is truncated appropriately.
    """
    if isinstance(spatial_axis, int):
        spatial_axis = (spatial_axis,)
    return jnp.fft.rfftn(x, axes=spatial_axis)


def irfft(
    x: jnp.ndarray,
    spatial_axis: Union[int, Tuple[int, ...]],
    s: Union[int, Tuple[int, ...]],
) -> jnp.ndarray:
    """
    Inverse real FFT, producing a real array with prescribed spatial size(s).

    Parameters
    ----------
    x : jnp.ndarray
        Complex input of length ``n//2 + 1`` (or its multi‑dimensional
        equivalent).
    spatial_axis : int or tuple of int
        Axes corresponding to the Fourier representation.
    s : int or tuple of int
        Full spatial size(s) of the output real array.

    Returns
    -------
    jnp.ndarray
        Real array with shape restored to ``s`` along each Fourier axis.
    """
    if isinstance(spatial_axis, int):
        spatial_axis = (spatial_axis,)
    return jnp.fft.irfftn(x, s=s, axes=spatial_axis)


def truncate_fft(
    x: jnp.ndarray,
    modes: int,
    spatial_axis: Union[int, Tuple[int, ...]],
) -> jnp.ndarray:
    """
    Keep only the lowest ``modes`` Fourier modes along each spatial axis.

    Assumes that the input has been transformed by :func:`rfft` and that the
    Fourier representation lies along the given axes.

    Parameters
    ----------
    x : jnp.ndarray
        Complex array with Fourier axes as produced by :func:`rfft`.
    modes : int
        Number of low‑frequency modes to retain in each direction.
    spatial_axis : int or tuple of int
        The Fourier axis / axes.

    Returns
    -------
    jnp.ndarray
        Array with the same overall shape except that the size along each
        Fourier axis is truncated to ``modes``.
    """
    if isinstance(spatial_axis, int):
        spatial_axis = (spatial_axis,)
    ndim = len(spatial_axis)
    # For 1D we simply slice the first `modes` entries.
    # For 2D we keep the upper‑left block of size (modes, modes).
    slicing = [slice(None)] * x.ndim
    for i, ax in enumerate(spatial_axis):
        if ndim == 1:
            slicing[ax] = slice(0, modes)
        else:
            # The last Fourier axis is already halved; we keep `modes` along it.
            slicing[ax] = slice(0, modes)
    return x[tuple(slicing)]


# ============================================================================
# Random keys
# ============================================================================


def split_key(key: jax.random.PRNGKey) -> Tuple[jax.random.PRNGKey, jax.random.PRNGKey]:
    """
    Split a JAX PRNG key into two independent keys.

    Parameters
    ----------
    key : jax.random.PRNGKey
        The key to split.

    Returns
    -------
    tuple
        Two new keys.
    """
    return jax.random.split(key)


# ============================================================================
# Checkpointing (CPU‑safe serialisation)
# ============================================================================


def save_pytree(pytree: Any, path: str) -> None:
    """
    Save a PyTree to disk using pickle after moving arrays to CPU.

    Parameters
    ----------
    pytree : Any
        The PyTree to persist (e.g., Flax ``params``, optimizer state).
    path : str
        Destination file path.
    """
    cpu_pytree = jax.device_get(pytree)
    with open(path, "wb") as f:
        pickle.dump(cpu_pytree, f)


def load_pytree(path: str) -> Any:
    """
    Load a PyTree saved with :func:`save_pytree`.

    Parameters
    ----------
    path : str
        File path to load from.

    Returns
    -------
    Any
        The restored PyTree.  Arrays will be on the CPU; the caller may need
        to transfer them to the desired device.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================================
# Parameter flatten / unflatten
# ============================================================================


def flatten_params(params: Any) -> Tuple[jnp.ndarray, Callable]:
    """
    Flatten a nested parameter PyTree into a 1D array.

    Parameters
    ----------
    params : Any
        A PyTree of JAX arrays (e.g., Flax model parameters).

    Returns
    -------
    flat_params : jnp.ndarray
        1D array containing all parameters concatenated.
    unflatten_fn : callable
        A function that reconstructs the original PyTree from a flat array.
    """
    return flatten_util.ravel_pytree(params)


def unflatten_params(flat_p: jnp.ndarray, unflatten_fn: Callable) -> Any:
    """
    Rebuild a PyTree from its flat representation.

    Parameters
    ----------
    flat_p : jnp.ndarray
        Flat parameter vector.
    unflatten_fn : callable
        The function returned by :func:`flatten_params`.

    Returns
    -------
    Any
        The reconstructed PyTree.
    """
    return unflatten_fn(flat_p)


# ============================================================================
# Batch iteration
# ============================================================================


def batch_iterator(
    data: Union[np.ndarray, Tuple[np.ndarray, ...]],
    batch_size: int,
    shuffle: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Iterator[Union[np.ndarray, Tuple[np.ndarray, ...]]]:
    """
    Yield consecutive (optionally shuffled) batches from a dataset.

    Parameters
    ----------
    data : ndarray or tuple of ndarray
        Either a single array (inputs) or a tuple ``(inputs, targets)``.
        All arrays must have the same first‑axis length.
    batch_size : int
        Number of samples per batch.
    shuffle : bool, optional
        If ``True``, shuffle the data in unison before yielding.
    rng : numpy.random.Generator, optional
        Required when ``shuffle=True``; used for shuffling.

    Yields
    ------
    batch : ndarray or tuple of ndarray
        A slice (or tuple of slices) of the data.
    """
    if isinstance(data, tuple):
        n_samples = data[0].shape[0]
        arrays = list(data)
    else:
        n_samples = data.shape[0]
        arrays = [data]

    if shuffle:
        if rng is None:
            raise ValueError("Must provide an `rng` when `shuffle=True`.")
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        arrays = [arr[indices] for arr in arrays]

    start = 0
    while start < n_samples:
        end = min(start + batch_size, n_samples)
        batch = tuple(arr[start:end] for arr in arrays)
        yield batch if len(batch) > 1 else batch[0]
        start = end


# ============================================================================
# Jacobian computations (for LUNO)
# ============================================================================


def compute_jacobian_norms(
    apply_fn: Callable,
    params: Any,
    inputs: jnp.ndarray,
    chunk_size: int = 64,
) -> jnp.ndarray:
    """
    Compute the *squared* L₂ norm of the Jacobian of each scalar output
    with respect to ``params``.

    This implements the efficient batched computation of
    ``diag(J @ Jᵗ)`` where ``J`` is the Jacobian of the model outputs.

    Parameters
    ----------
    apply_fn : callable
        Function ``apply_fn(params, inputs)`` that returns the model output
        of shape ``(batch, *spatial_dims, out_channels)``.
    params : Any
        Model parameters (a PyTree).
    inputs : jnp.ndarray
        Input batch of shape ``(batch, *spatial_dims, in_channels)``.
    chunk_size : int, optional
        Number of scalar outputs processed at once.  Larger values are faster
        but require more memory.

    Returns
    -------
    sq_norms : jnp.ndarray
        Array of shape ``(batch, *spatial_dims, out_channels)`` containing
        the squared Euclidean norm of the gradient for each output element.
        In LUNO‑Iso this is directly multiplied by ``sigma^2`` to obtain the
        marginal predictive variance.
    """
    outputs = apply_fn(params, inputs)
    # Flatten all outputs except the batch dimension? Actually we want per‑output scalar.
    # We can flatten spatially and channel dimensions.
    orig_shape = outputs.shape
    flat_outputs = outputs.reshape(-1)  # total M = batch * spatial * channels
    total_outs = flat_outputs.shape[0]

    # Build a function that returns the k‑th output scalar
    def output_scalar(k: int, p: Any, x: jnp.ndarray) -> jnp.ndarray:
        out = apply_fn(p, x)
        return out.ravel()[k]

    # Vectorised gradient wrt params (argnum=1) over chunks of `k`
    grad_single = jax.grad(output_scalar, argnums=1)

    sq_norms_list = []
    for start in range(0, total_outs, chunk_size):
        chunk_indices = jnp.arange(start, min(start + chunk_size, total_outs))
        # vmap over the first axis (chunk indices)
        chunk_grads = jax.vmap(grad_single, in_axes=(0, None, None))(
            chunk_indices, params, inputs
        )
        # chunk_grads is a PyTree with an extra leading dimension `chunk`.
        # Compute squared norm per element: sum over all parameter leaves.
        sq_norms = sum(
            jnp.sum(jnp.square(leaf), axis=tuple(range(1, leaf.ndim)))
            for leaf in jax.tree_util.tree_leaves(chunk_grads)
        )
        sq_norms_list.append(sq_norms)

    sq_norms = jnp.concatenate(sq_norms_list)
    return sq_norms.reshape(orig_shape)


def compute_jacobian_V(
    apply_fn: Callable,
    params: Any,
    inputs: jnp.ndarray,
    V: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute ``J @ V`` for a given set of right‑hand side vectors ``V``.

    ``J`` is the Jacobian of the (uncurried) model outputs with respect to the
    flattened parameters.  The result is a set of vector‑valued Jacobian‑vector
    products, one per column of ``V``.

    Parameters
    ----------
    apply_fn : callable
        Function ``apply_fn(params, inputs)`` returning outputs of shape
        ``(batch, *spatial_dims, out_channels)``.
    params : Any
        Model parameters (PyTree).
    inputs : jnp.ndarray
        Input batch.
    V : jnp.ndarray
        Array of shape ``(p, r)``, where ``p`` is the total number of parameters
        and ``r`` is the number of columns (e.g., eigenvectors of the GGN).

    Returns
    -------
    JV : jnp.ndarray
        Array of shape ``(r, *outputs.shape)``.  The slice ``JV[i]`` is
        ``J @ V[:, i]`` reshaped to the output grid.
    """
    flat_params, unflatten_fn = flatten_params(params)

    def f_flat(p_flat: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        return apply_fn(unflatten_fn(p_flat), x).ravel()

    # f_flat returns a flat array of length total_outputs.
    # We need one JVP per column of V; V is (p, r).  Transpose to (r, p) and vmap.
    V = V.T  # shape (r, p)

    def jvp_single(v: jnp.ndarray) -> jnp.ndarray:
        # v: shape (p,)
        out, jvp_val = jax.jvp(lambda p: f_flat(p, inputs), (flat_params,), (v,))
        return jvp_val  # shape (total_outputs,)

    # vmap over the first axis of V (which is now (r, p))
    JV_flat = jax.vmap(jvp_single)(V)  # (r, total_outputs)

    # Reshape to (r,) + output shape
    dummy_output = apply_fn(params, inputs)
    JV = JV_flat.reshape((JV_flat.shape[0],) + dummy_output.shape)
    return JV


# ============================================================================
# Low‑rank GGN approximation via randomised SVD
# ============================================================================


def compute_ggn_top_eigenvectors(
    f_flat_apply: Callable,
    flat_params: jnp.ndarray,
    input_batches: list[jnp.ndarray],
    rank: int,
    rng_key: jax.random.PRNGKey,
    oversamp: int = 10,
) -> jnp.ndarray:
    """
    Approximate the top‑`rank` eigenvectors of the Generalised Gauss‑Newton
    matrix ``Jᵗ J`` using randomised SVD.

    The GGN is formed from a collection of mini‑batches; the function
    ``f_flat_apply(flat_params, x)`` returns the *flattened* model outputs on
    ``x``.

    Parameters
    ----------
    f_flat_apply : callable
        Function ``(flat_params, x) -> flat_outputs``.
    flat_params : jnp.ndarray
        Flattened parameter vector (length ``p``).
    input_batches : list of jnp.ndarray
        Mini‑batches over which the GGN is accumulated.
    rank : int
        Number of eigenvectors to return (column dimension of ``V``).
    rng_key : jax.random.PRNGKey
        JAX PRNG key.
    oversamp : int, optional
        Extra columns in the random test matrix for stability (default 10).

    Returns
    -------
    V : jnp.ndarray
        Matrix of shape ``(p, rank)`` containing the top eigenvectors as
        columns, each of unit Euclidean norm.
    """
    p = flat_params.size

    # ---- build GGN‑vector product ------------------------------------------
    def ggn_vp(v: jnp.ndarray) -> jnp.ndarray:
        """
        Compute H_GGN @ v = sum_x J_xᵗ (J_x v).
        v: shape (p,)
        returns: shape (p,)
        """
        total = jnp.zeros_like(v)
        for x in input_batches:
            # J_v: Jacobian‑vector product for this batch
            _, j_v = jax.jvp(lambda p: f_flat_apply(p, x).ravel(), (flat_params,), (v,))
            # VJP: J_xᵗ (J_x v)
            _, vjp_fn = jax.vjp(lambda p: f_flat_apply(p, x).ravel(), flat_params)
            (contrib,) = vjp_fn(j_v)
            total = total + contrib
        return total

    # JIT‑compile for speed (the loop over batches is traced as a constant)
    ggn_vp_jit = jax.jit(ggn_vp)

    # ---- Randomised SVD (Algorithm 5.1 of Halko et al. 2011) ----------------
    n_oversamp = rank + oversamp
    key1, key2 = jax.random.split(rng_key)

    # 1. Draw random test matrix Omega (p, n_oversamp) from standard normal
    Omega = jax.random.normal(key1, (p, n_oversamp), dtype=jnp.float32)

    # 2. Y = H_GGN @ Omega.  vmap over columns of Omega.
    Y = jax.vmap(ggn_vp_jit, in_axes=1, out_axes=1)(Omega)  # (p, n_oversamp)

    # 3. Economy SVD of Y (or QR for stability)
    Q, _ = jnp.linalg.qr(Y)  # Q (p, n_oversamp), orthonormal columns

    # 4. Form B = Qᵗ H_GGN Q
    #    Compute H_GGN Q column‑wise via vmap
    HQ = jax.vmap(ggn_vp_jit, in_axes=1, out_axes=1)(Q)  # (p, n_oversamp)
    B = Q.T @ HQ  # (n_oversamp, n_oversamp)
    B = 0.5 * (B + B.T)  # symmetrise

    # 5. Eigen‑decompose B (real symmetric)
    eigvals, eigvecs = jnp.linalg.eigh(B)  # ascending order

    # 6. Take top `rank` eigenvectors (largest eigenvalues)
    idx = jnp.argsort(eigvals)[::-1][:rank]
    S = eigvecs[:, idx]  # (n_oversamp, rank)

    # 7. Approximate eigenvectors V = Q @ S
    V = Q @ S  # (p, rank)

    # Normalise columns (should already be, but for safety)
    norms = jnp.linalg.norm(V, axis=0, keepdims=True)
    V = V / jnp.where(norms > 0, norms, 1.0)

    return V
