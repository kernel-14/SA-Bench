"""
uncertainty.py
===============
Linearised and sample‑based uncertainty quantification methods for the
Fourier Neural Operator (FNO) as described in the LUNO paper.

All methods inherit from the abstract ``UQMethod`` and implement
``fit(calib_ds)`` and ``predict(inputs)``.  They produce a predictive
mean and a marginal (per‑grid‑point) variance.

The module is designed to be self‑contained; Jacobian computations and
the low‑rank GGN eigen‑decomposition are implemented directly here using
JAX building blocks so that the file can be used without modifying other
parts of the project.
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import flatten_util

from config import (
    CalibrationConfig,
    EnsembleConfig,
    InputPerturbConfig,
    LaplaceConfig,
    PushForwardConfig,
    UQConfig,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Parameter handling utilities
# =============================================================================

def _infer_num_blocks(params: Dict) -> int:
    """
    Return the number of Fourier blocks from the keys of a Flax parameter dict.

    The naming convention is ``FourierBlock_{i}``.
    """
    max_idx = -1
    for k in params.keys():
        if k.startswith("FourierBlock_"):
            try:
                idx = int(k.split("_")[-1])
            except ValueError:
                continue
            max_idx = max(max_idx, idx)
    if max_idx == -1:
        raise ValueError("No FourierBlock found in parameter keys.")
    return max_idx + 1


def split_params(params: Dict, num_blocks: int) -> Tuple[Dict, Dict]:
    """
    Split Flax FNO parameters into *base* (everything except the last
    Fourier block and the projection) and *last* (last Fourier block +
    projection).

    Raises ``KeyError`` if the expected keys are missing.
    """
    base = {}
    last = {}
    last_block_name = f"FourierBlock_{num_blocks - 1}"
    for key, value in params.items():
        if key in (last_block_name, "projection"):
            last[key] = value
        else:
            base[key] = value
    if last_block_name not in last or "projection" not in last:
        raise KeyError(
            f"Last Fourier block '{last_block_name}' and/or "
            f"'projection' not found in params."
        )
    return base, last


def merge_params(base: Dict, last: Dict) -> Dict:
    """Return a merged parameter dictionary; last overwrites base on conflict."""
    return {**base, **last}


def flatten_params_pytree(pytree: Any) -> Tuple[jnp.ndarray, Callable]:
    """Wrapper around ``flatten_util.ravel_pytree`` for clarity."""
    return flatten_util.ravel_pytree(pytree)


# =============================================================================
# Jacobian helpers (last‑layer only)
# =============================================================================

def _forward_and_jacobian(
    apply_fn: Callable[[Any, jnp.ndarray], jnp.ndarray],
    base_params: Dict,
    unflatten_fn: Callable,
    last_params_flat: jnp.ndarray,
    x: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Evaluate the model and compute the Jacobian of the (flattened) output
    with respect to the flattened last‑layer parameters.

    Parameters
    ----------
    apply_fn : callable
        Full model ``apply_fn(params, x)``.
    base_params : dict
        Frozen base parameters.
    unflatten_fn : callable
        Reconstructs last‑layer pytree from a 1‑D vector.
    last_params_flat : jnp.ndarray
        Flattened MAP of the last‑layer parameters, shape ``(p_last,)``.
    x : jnp.ndarray
        Single input example of shape ``(1, T, *spatial, C)`` (batch dim = 1).

    Returns
    -------
    J : jnp.ndarray
        Jacobian matrix of shape ``(N_out, p_last)``.
    f_val : jnp.ndarray
        Flat output vector of shape ``(N_out,)``.
    """
    def forward(flat_p: jnp.ndarray) -> jnp.ndarray:
        last_tree = unflatten_fn(flat_p)
        all_params = merge_params(base_params, last_tree)
        out = apply_fn(all_params, x)
        return out.ravel()  # flatten spatial + channels; batch size is 1
    J = jax.jacrev(forward)(last_params_flat)
    f_val = forward(last_params_flat)
    return J, f_val


def _compute_jacobian_norms_and_u(
    J: jnp.ndarray,
    U: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], jnp.ndarray]:
    """
    Extract squared L₂ norm per output point and (optionally) the projection
    ``u = J @ U``.

    Parameters
    ----------
    J : jnp.ndarray
        Jacobian of shape ``(N_out, p_last)``.
    U : jnp.ndarray, optional
        Pre‑computed eigenvector matrix of shape ``(p_last, rank)``.

    Returns
    -------
    norms : jnp.ndarray
        Shape ``(N_out,)``, row‑wise squared Euclidean norm.
    u : jnp.ndarray or None
        If ``U`` is provided, shape ``(N_out, rank)``; else ``None``.
    N_out : int
        Number of output points (for convenience).
    """
    norms = jnp.sum(J**2, axis=1)
    u = J @ U if U is not None else None
    return norms, u


# =============================================================================
# Low‑rank GGN eigen‑decomposition
# =============================================================================

def _ggn_vp(
    apply_fn: Callable[[Any, jnp.ndarray], jnp.ndarray],
    base_params: Dict,
    unflatten_last: Callable,
    last_params_flat: jnp.ndarray,
    batches: List[jnp.ndarray],
    v: jnp.ndarray,
) -> jnp.ndarray:
    """
    Matrix‑vector product ``H_GGN @ v`` for the last‑layer GGN.

    ``H = Σ_x J_xᵗ J_x`` where ``J_x`` is the Jacobian w.r.t. flat last‑layer
    parameters at the MAP value ``last_params_flat``.

    Parameters
    ----------
    apply_fn : callable
        Full model apply.
    base_params : dict
        Frozen base parameters.
    unflatten_last : callable
        Rebuilds last‑layer pytree from a flat vector.
    last_params_flat : jnp.ndarray
        MAP value of last‑layer parameters (``p_last,``).
    batches : list of jnp.ndarray
        Mini‑batches of input data (each of shape ``(B, T, *spatial, C)``).
    v : jnp.ndarray
        Input vector, shape ``(p_last,)``.

    Returns
    -------
    Hv : jnp.ndarray
        Result of shape ``(p_last,)``.
    """
    total = jnp.zeros_like(last_params_flat)
    for x in batches:
        # J_v = J_x @ v
        def flat_output(flat_p):
            last_tree = unflatten_last(flat_p)
            all_params = merge_params(base_params, last_tree)
            return apply_fn(all_params, x).ravel()

        _, jvp_val = jax.jvp(flat_output, (last_params_flat,), (v,))
        # J_xᵗ (J_x v)
        _, vjp_fn = jax.vjp(flat_output, last_params_flat)
        (contrib,) = vjp_fn(jvp_val)
        total = total + contrib
    return total


def compute_ggn_top_eigenvectors(
    apply_fn: Callable[[Any, jnp.ndarray], jnp.ndarray],
    base_params: Dict,
    last_params_map: Dict,
    train_pairs: Tuple[np.ndarray, np.ndarray],
    rank: int,
    seed: int = 42,
    oversamp: int = 10,
) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
    """
    Approximate the top ``rank`` eigenvectors of the last‑layer GGN
    using randomised SVD.

    Parameters
    ----------
    apply_fn : callable
        Full model forward pass.
    base_params : dict
        Fixed base parameters.
    last_params_map : dict
        MAP of the last‑layer parameters (pytree).
    train_pairs : tuple of ndarray
        ``(train_x, train_y)``.  Only ``train_x`` is used; ``y`` is
        ignored because the GGN uses only the model Jacobian.
    rank : int
        Number of eigenvectors to return.
    seed : int, optional
        Random seed for the test matrix.
    oversamp : int, optional
        Extra columns for stability (default 10).

    Returns
    -------
    U : jnp.ndarray
        Matrix of shape ``(p_last, rank)`` containing orthonormal
        eigenvectors as columns.
    S : jnp.ndarray
        Vector of eigenvalues, length ``rank``, in descending order.
    n_data : int
        Total number of training input examples used.
    """
    flat_last, unflatten_fn = flatten_params_pytree(last_params_map)
    train_x = train_pairs[0]
    # Split into manageable chunks for GGN products (batch ~ 64)
    batch_size = 64
    n_samples = train_x.shape[0]
    x_batches = [jnp.asarray(train_x[i:i + batch_size]) for i in range(0, n_samples, batch_size)]

    p = flat_last.size
    n_oversamp = rank + oversamp
    rng_key = jax.random.PRNGKey(seed)

    # Matrix‑free product wrapper
    ggn_vp = jax.tree_util.Partial(
        _ggn_vp, apply_fn, base_params, unflatten_fn, flat_last, x_batches
    )
    ggn_vp_jit = jax.jit(ggn_vp)

    # 1. Random test matrix Omega (p, n_oversamp)
    omega = jax.random.normal(rng_key, (p, n_oversamp), dtype=jnp.float32)

    # 2. Y = H @ Omega (column‑wise vmap)
    Y = jax.vmap(ggn_vp_jit, in_axes=1, out_axes=1)(omega)  # (p, n_oversamp)

    # 3. Economy QR (or SVD) for orthonormal basis Q
    Q, _ = jnp.linalg.qr(Y)  # (p, n_oversamp)

    # 4. B = Qᵗ H Q
    HQ = jax.vmap(ggn_vp_jit, in_axes=1, out_axes=1)(Q)  # (p, n_oversamp)
    B = Q.T @ HQ  # (n_oversamp, n_oversamp)
    B = 0.5 * (B + B.T)  # symmetrise

    # 5. Eigen‑decompose B (real symmetric)
    eigvals, eigvecs = jnp.linalg.eigh(B)  # ascending order

    # 6. Take top `rank` eigenvectors of B
    idx = jnp.argsort(eigvals)[::-1][:rank]
    S_small = eigvals[idx]  # (rank,)
    V_small = eigvecs[:, idx]  # (n_oversamp, rank)

    # 7. Approximate eigenvectors U = Q @ V_small
    U = Q @ V_small  # (p, rank)

    # Normalise columns (they should already be orthonormal, but for safety)
    norms_U = jnp.linalg.norm(U, axis=0, keepdims=True)
    U = U / jnp.where(norms_U > 0, norms_U, 1.0)

    return U, S_small, n_samples


# =============================================================================
# Negative log‑likelihood (Gaussian marginal)
# =============================================================================

def _gaussian_nll(
    y: jnp.ndarray, mean: jnp.ndarray, var: jnp.ndarray
) -> jnp.ndarray:
    """Per‑element Gaussian negative log‑likelihood (natural log)."""
    # var assumed > 0; add small epsilon for safety?
    var_safe = jnp.clip(var, 1e-12)
    return 0.5 * (jnp.log(2 * jnp.pi * var_safe) + (y - mean) ** 2 / var_safe)


# =============================================================================
# Abstract Base Class
# =============================================================================

class UQMethod(abc.ABC):
    """
    Abstract uncertainty‑quantification method for an FNO.

    Subclasses must implement ``fit(calib_ds)`` and ``predict(inputs)``.

    Parameters
    ----------
    apply_fn : callable
        The full FNO forward function ``apply_fn(params, x)``.
    params : dict
        Trained (MAP) parameters of the model.
    config : UQConfig
        Container with all hyper‑parameters.
    seed : int, optional
        Seed for random number generation (used by sample methods).
    """

    def __init__(
        self,
        apply_fn: Callable[[Any, jnp.ndarray], jnp.ndarray],
        params: Dict,
        config: UQConfig,
        seed: int = 42,
    ) -> None:
        self.apply_fn = apply_fn
        self.full_params = params
        self.config = config
        self.seed = seed

        # ---- split parameters -----------------------------------------------
        self.num_blocks = _infer_num_blocks(params)
        self.base_params, self.last_params_map = split_params(params, self.num_blocks)

        # Flattened MAP of last‑layer parameters
        self.last_params_vec, self.unflatten_last = flatten_params_pytree(
            self.last_params_map
        )
        self.p_last = self.last_params_vec.size
        logger.info(
            "Last‑layer parameter dimension: %d", self.p_last
        )

        # Convenience: forward pass that only requires flat last‑layer vector
        def _apply_last_flat(flat_p: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
            last_tree = self.unflatten_last(flat_p)
            all_params = merge_params(self.base_params, last_tree)
            return self.apply_fn(all_params, x)

        self._apply_last_flat = _apply_last_flat

    @abc.abstractmethod
    def fit(
        self,
        calib_ds: Tuple[np.ndarray, np.ndarray],
        train_ds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        """Calibrate hyper‑parameters on ``calib_ds``."""

    @abc.abstractmethod
    def predict(
        self, inputs: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Return (mean, variance) for the given `inputs`.

        Both tensors have the same shape as the model output
        (typically ``(H, W, 1)`` or ``(spatial,)``).
        """

    # ------------------------------------------------------------------
    # Common internal helpers
    # ------------------------------------------------------------------
    def _forward_full(self, x: jnp.ndarray) -> jnp.ndarray:
        """Full model prediction using the stored MAP parameters."""
        return self.apply_fn(self.full_params, x)

    def _jacobian_and_stats(
        self, x: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Compute Jacobian (w.r.t. last layer) and extract squared norms.

        Returns ``(J, norms, None)``.
        """
        J, _ = _forward_and_jacobian(
            self.apply_fn,
            self.base_params,
            self.unflatten_last,
            self.last_params_vec,
            x,
        )
        norms, _ = _compute_jacobian_norms_and_u(J)
        return J, norms, None


# =============================================================================
# LUNO --- Isotropic
# =============================================================================

class LUNOIsotropic(UQMethod):
    """
    Linearised uncertainty with isotropic Gaussian weight belief
    ``w_last ~ N(w_map, σ² I)``.

    Calibration selects ``σ²`` by minimising the marginal NLL on
    the validation set.
    """

    def __init__(
        self,
        apply_fn: Callable,
        params: Dict,
        config: UQConfig,
        seed: int = 42,
        sigma2: Optional[float] = None,
    ) -> None:
        super().__init__(apply_fn, params, config, seed)
        self.sigma2: Optional[float] = sigma2  # set during fit

    # ------------------------------------------------------------------
    def fit(
        self,
        calib_ds: Tuple[np.ndarray, np.ndarray],
        train_ds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        """Grid search over σ² on the calibration set."""
        del train_ds  # not used
        calib_x, calib_y = calib_ds

        # Pre‑compute norms for each calibration sample
        norms_list = []
        y_flat_list = []
        mean_flat_list = []
        for i in range(len(calib_x)):
            x_batch = jnp.asarray(calib_x[i : i + 1])  # keep batch dim
            y_batch = jnp.asarray(calib_y[i : i + 1])
            J, f_val = _forward_and_jacobian(
                self.apply_fn,
                self.base_params,
                self.unflatten_last,
                self.last_params_vec,
                x_batch,
            )
            norms, _ = _compute_jacobian_norms_and_u(J)
            norms_list.append(norms)  # (N_out,)
            mean_flat_list.append(f_val)
            y_flat_list.append(y_batch.ravel())

        norms_all = jnp.concatenate(norms_list)
        mean_all = jnp.concatenate(mean_flat_list)
        y_all = jnp.concatenate(y_flat_list)

        # Grid search
        calib_cfg: CalibrationConfig = self.config.calibration
        sigma2_candidates = np.logspace(
            calib_cfg.sigma_range[0],
            calib_cfg.sigma_range[1],
            calib_cfg.grid_size,
        )

        best_nll = float("inf")
        best_sigma2 = 1.0
        for s2 in sigma2_candidates:
            var = s2 * norms_all
            nll = jnp.mean(_gaussian_nll(y_all, mean_all, var))
            if nll < best_nll:
                best_nll = nll
                best_sigma2 = float(s2)

        self.sigma2 = best_sigma2
        logger.info("LUNO‑Iso calibrated σ² = %.6e", self.sigma2)

    # ------------------------------------------------------------------
    def predict(
        self, inputs: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if self.sigma2 is None:
            raise RuntimeError("Must call fit() before predict().")
        x_batch = jnp.atleast_1d(inputs)  # ensure batch dim
        J, f_val = _forward_and_jacobian(
            self.apply_fn,
            self.base_params,
            self.unflatten_last,
            self.last_params_vec,
            x_batch,
        )
        norms, _ = _compute_jacobian_norms_and_u(J)
        var = self.sigma2 * norms

        # Reshape to original output shape
        # output shape: (1, *spatial, 1)  -> mean has shape (1, ...)
        dummy = self._forward_full(x_batch)
        out_shape = dummy.shape[1:]  # drop batch dim
        var = var.reshape(out_shape)
        mean = f_val.reshape(out_shape)
        return mean, var


# =============================================================================
# LUNO --- Laplace
# =============================================================================

class LUNOLaplace(UQMethod):
    """
    Linearised uncertainty using a low‑rank Laplace posterior covariance
    ``Σ = (H + τ I)⁻¹``, with ``H ≈ U diag(S) Uᵗ`` and ``τ = 1 / σ²``.
    """

    def __init__(
        self,
        apply_fn: Callable,
        params: Dict,
        config: UQConfig,
        seed: int = 42,
        tau: Optional[float] = None,
    ) -> None:
        super().__init__(apply_fn, params, config, seed)
        self.tau: Optional[float] = tau
        self.U: Optional[jnp.ndarray] = None   # (p_last, rank)
        self.S: Optional[jnp.ndarray] = None   # (rank,)
        self.M_inv: Optional[jnp.ndarray] = None  # (rank, rank)

    # ------------------------------------------------------------------
    def fit(
        self,
        calib_ds: Tuple[np.ndarray, np.ndarray],
        train_ds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        if train_ds is None:
            raise ValueError("train_ds must be provided for Laplace GGN computation.")
        laplace_cfg: LaplaceConfig = self.config.laplace

        # 1. Compute GGN top eigenvectors
        U, S, n_data = compute_ggn_top_eigenvectors(
            self.apply_fn,
            self.base_params,
            self.last_params_map,
            train_ds,
            laplace_cfg.rank,
            seed=self.seed,
        )
        self.U = U  # (p_last, rank)
        self.S = S  # (rank,)

        # 2. Pre‑compute norms and u = J @ U for each calibration sample
        calib_x, calib_y = calib_ds
        norms_list = []
        u_list = []
        y_flat_list = []
        mean_flat_list = []

        for i in range(len(calib_x)):
            x_batch = jnp.asarray(calib_x[i : i + 1])
            y_batch = jnp.asarray(calib_y[i : i + 1])
            J, f_val = _forward_and_jacobian(
                self.apply_fn,
                self.base_params,
                self.unflatten_last,
                self.last_params_vec,
                x_batch,
            )
            norms, u = _compute_jacobian_norms_and_u(J, self.U)
            norms_list.append(norms)
            u_list.append(u)
            mean_flat_list.append(f_val)
            y_flat_list.append(y_batch.ravel())

        norms_all = jnp.concatenate(norms_list)
        u_all = jnp.concatenate(u_list, axis=0) if u_list[0] is not None else None
        mean_all = jnp.concatenate(mean_flat_list)
        y_all = jnp.concatenate(y_flat_list)

        # 3. Grid search over τ = 1/σ²
        calib_cfg: CalibrationConfig = self.config.calibration
        sigma2_candidates = np.logspace(
            calib_cfg.sigma_range[0], calib_cfg.sigma_range[1], calib_cfg.grid_size
        )
        tau_candidates = 1.0 / sigma2_candidates

        best_nll = float("inf")
        best_tau = tau_candidates[0]
        best_M_inv = None
        r = S.shape[0]

        for tau in tau_candidates:
            # M = diag(S) + tau * I
            M = jnp.diag(S) + tau * jnp.eye(r)
            # Cholesky for stability (S positive, tau>0)
            L = jnp.linalg.cholesky(M)
            M_inv = jax.scipy.linalg.solve_triangular(
                L, jnp.eye(r), lower=True
            )
            M_inv = jax.scipy.linalg.solve_triangular(
                L.T, M_inv, lower=False
            )  # M^{-1}

            # variance: (1/tau)*norms - (1/tau^2) * sum (u * (u M_inv))
            var = (1.0 / tau) * norms_all - (1.0 / (tau * tau)) * jnp.sum(
                u_all * (u_all @ M_inv), axis=1
            )
            nll = jnp.mean(_gaussian_nll(y_all, mean_all, var))
            if nll < best_nll:
                best_nll = nll
                best_tau = tau
                best_M_inv = M_inv

        self.tau = best_tau
        self.M_inv = best_M_inv
        logger.info(
            "LUNO‑LA calibrated τ = %.6e (σ² = %.6e)", self.tau, 1.0 / self.tau
        )

    # ------------------------------------------------------------------
    def predict(
        self, inputs: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if self.tau is None or self.U is None or self.M_inv is None:
            raise RuntimeError("Must call fit() before predict().")
        x_batch = jnp.atleast_1d(inputs)
        J, f_val = _forward_and_jacobian(
            self.apply_fn,
            self.base_params,
            self.unflatten_last,
            self.last_params_vec,
            x_batch,
        )
        norms, u = _compute_jacobian_norms_and_u(J, self.U)

        var = (1.0 / self.tau) * norms - (1.0 / (self.tau ** 2)) * jnp.sum(
            u * (u @ self.M_inv), axis=1
        )

        dummy = self._forward_full(x_batch)
        out_shape = dummy.shape[1:]  # drop batch
        mean = f_val.reshape(out_shape)
        var = var.reshape(out_shape)
        return mean, var


# =============================================================================
# Sample‑Based Methods
# =============================================================================

class SampleIsotropic(UQMethod):
    """Uncertainty via Monte‑Carlo sampling from isotropic Gaussian weight belief."""

    def __init__(
        self,
        apply_fn: Callable,
        params: Dict,
        config: UQConfig,
        seed: int = 42,
    ) -> None:
        super().__init__(apply_fn, params, config, seed)
        self.sigma2: Optional[float] = None
        self.num_samples = config.num_samples

    # ------------------------------------------------------------------
    def fit(
        self,
        calib_ds: Tuple[np.ndarray, np.ndarray],
        train_ds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        del train_ds
        # Use a linearised helper to select σ² cheaply
        luno_iso = LUNOIsotropic(self.apply_fn, self.full_params, self.config)
        luno_iso.fit(calib_ds)
        self.sigma2 = luno_iso.sigma2
        logger.info("Sample‑Iso inherited σ² = %.6e from LUNO‑Iso", self.sigma2)

    # ------------------------------------------------------------------
    def predict(
        self, inputs: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if self.sigma2 is None:
            raise RuntimeError("Must call fit() before predict().")
        key = jax.random.PRNGKey(self.seed)
        x_batch = jnp.atleast_1d(inputs)
        # Generate weight samples
        map_flat = self.last_params_vec
        noise = jax.random.normal(key, (self.num_samples, self.p_last))
        samples_flat = map_flat + jnp.sqrt(self.sigma2) * noise

        # Vectorised forward pass
        apply_batch = jax.vmap(self._apply_last_flat, in_axes=(0, None))
        outs = apply_batch(samples_flat, x_batch)  # (S, 1, *spatial, 1)
        outs = outs.squeeze(1)  # (S, *spatial, 1)
        mean = jnp.mean(outs, axis=0)
        var = jnp.var(outs, axis=0)
        return mean, var


class SampleLaplace(UQMethod):
    """Uncertainty via Monte‑Carlo sampling from Laplace posterior."""

    def __init__(
        self,
        apply_fn: Callable,
        params: Dict,
        config: UQConfig,
        seed: int = 42,
    ) -> None:
        super().__init__(apply_fn, params, config, seed)
        self.tau: Optional[float] = None
        self.U: Optional[jnp.ndarray] = None
        self.S: Optional[jnp.ndarray] = None
        self.num_samples = config.num_samples

    # ------------------------------------------------------------------
    def fit(
        self,
        calib_ds: Tuple[np.ndarray, np.ndarray],
        train_ds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        if train_ds is None:
            raise ValueError("train_ds required for Laplace GGN.")
        # 1. Compute GGN eigensystem
        laplace_cfg: LaplaceConfig = self.config.laplace
        U, S, _ = compute_ggn_top_eigenvectors(
            self.apply_fn,
            self.base_params,
            self.last_params_map,
            train_ds,
            laplace_cfg.rank,
            seed=self.seed,
        )
        self.U = U
        self.S = S

        # 2. Re‑use LUNOLaplace to calibrate τ
        luno_la = LUNOLaplace(self.apply_fn, self.full_params, self.config)
        luno_la.U = U
        luno_la.S = S
        luno_la.fit(calib_ds)  # will compute best tau using calib_ds
        self.tau = luno_la.tau
        logger.info("Sample‑LA inherited τ = %.6e from LUNO‑LA", self.tau)

    # ------------------------------------------------------------------
    def predict(
        self, inputs: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if self.tau is None or self.U is None or self.S is None:
            raise RuntimeError("Must call fit() before predict().")
        key = jax.random.PRNGKey(self.seed)
        x_batch = jnp.atleast_1d(inputs)
        map_flat = self.last_params_vec

        # Precompute helpers for sampling from (U S Uᵗ + τ I)⁻¹
        r = self.U.shape[1]
        S_plus_tau = self.S + self.tau  # (r,)
        # Matrix square root of (S + τ I)
        sqrt_inv_S_plus_tau = 1.0 / jnp.sqrt(S_plus_tau)  # element‑wise
        sqrt_tau_inv = 1.0 / jnp.sqrt(self.tau)

        def sample_fn(k):
            z = jax.random.normal(k, (self.p_last,))
            # component orthogonal to U: (I - U Uᵗ) z (but we can do simpler)
            # z_orth = z - U @ (Uᵗ z)
            proj = self.U.T @ z  # (r,)
            # The weighted combination: U @ ( (sqrt_inv_S+tau - sqrt_tau_inv) * proj ) + sqrt_tau_inv * z
            diff = sqrt_inv_S_plus_tau - sqrt_tau_inv  # (r,)
            weighted = self.U @ (diff * proj)  # (p_last,)
            sample = sqrt_tau_inv * z + weighted
            return map_flat + sample

        keys = jax.random.split(key, self.num_samples)
        samples_flat = jax.vmap(sample_fn)(keys)

        apply_batch = jax.vmap(self._apply_last_flat, in_axes=(0, None))
        outs = apply_batch(samples_flat, x_batch)  # (S, 1, *spatial, 1)
        outs = outs.squeeze(1)
        mean = jnp.mean(outs, axis=0)
        var = jnp.var(outs, axis=0)
        return mean, var


# =============================================================================
# Ensemble
# =============================================================================

class Ensemble(UQMethod):
    """
    Deep ensemble of independently trained FNOs.

    Parameters
    ----------
    apply_fn : callable
        Forward function (shared across all ensemble members).
    params_list : list of dict
        One parameter dictionary per ensemble member.
    config : UQConfig
    """

    def __init__(
        self,
        apply_fn: Callable,
        params_list: List[Dict],
        config: UQConfig,
        seed: int = 42,
    ) -> None:
        # Base class expects a single params dict; we bypass its splitting by
        # passing an empty placeholder and override relevant methods.
        # However, we still need to initialise base to avoid errors.
        # We'll create a dummy UQMethod using a copy of the first params.
        # Simpler: we won't call super().__init__ directly? We need the config.
        self.apply_fn = apply_fn
        self.config = config
        self.params_list = params_list
        self.seed = seed

        # We'll store some dummy attributes for compatibility.
        self.full_params = params_list[0]  # not really used
        self.num_blocks = _infer_num_blocks(self.full_params)
        # We'll ignore base/last split; it's okay.

    def fit(
        self,
        calib_ds: Tuple[np.ndarray, np.ndarray],
        train_ds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        """Ensemble requires no calibration."""
        pass

    def predict(
        self, inputs: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x_batch = jnp.atleast_1d(inputs)
        outs = []
        for p in self.params_list:
            o = self.apply_fn(p, x_batch)  # (1, *spatial, 1)
            outs.append(o[0])  # drop batch dim
        outs = jnp.stack(outs, axis=0)  # (S, *spatial, 1)
        mean = jnp.mean(outs, axis=0)
        var = jnp.var(outs, axis=0)
        return mean, var


# =============================================================================
# Input Perturbation
# =============================================================================

class InputPerturbation(UQMethod):
    """
    Uncertainty estimates via input perturbation (Gaussian noise added to
    the input).
    """

    def __init__(
        self,
        apply_fn: Callable,
        params: Dict,
        config: UQConfig,
        seed: int = 42,
    ) -> None:
        super().__init__(apply_fn, params, config, seed)
        self.sigma_pert: Optional[float] = None
        self.num_perturb = config.input_perturb.num_samples

    # ------------------------------------------------------------------
    def fit(
        self,
        calib_ds: Tuple[np.ndarray, np.ndarray],
        train_ds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        del train_ds
        calib_x, calib_y = calib_ds
        calib_cfg: CalibrationConfig = self.config.calibration
        sigma_candidates = np.logspace(
            calib_cfg.input_perturb_sigma_range[0],
            calib_cfg.input_perturb_sigma_range[1],
            calib_cfg.grid_size,
        )
        best_nll = float("inf")
        best_sigma = sigma_candidates[0]
        key = jax.random.PRNGKey(self.seed)

        # We'll vectorize over perturbations using vmap
        def perturb_and_forward(x, sigma):
            noise = sigma * jax.random.normal(key, x.shape)
            return self.apply_fn(self.full_params, x + noise)

        # vmap over perturbations
        perturb_batch = jax.vmap(perturb_and_forward, in_axes=(None, 0))

        for sigma in sigma_candidates:
            nll_sum = 0.0
            count = 0
            for i in range(len(calib_x)):
                x = jnp.asarray(calib_x[i : i + 1])
                y = jnp.asarray(calib_y[i : i + 1])
                # Generate perturbations (num_perturb,)
                key, subkey = jax.random.split(key)
                perturb_sigma = jnp.full((self.num_perturb,), sigma)
                preds = perturb_batch(x, perturb_sigma)  # (S, 1, *spatial, 1)
                preds = preds[:, 0]  # (S, *spatial, 1)
                mean = jnp.mean(preds, axis=0)
                var = jnp.var(preds, axis=0)
                nll = jnp.mean(_gaussian_nll(y[0], mean, var))
                nll_sum += nll
                count += 1
            avg_nll = nll_sum / count
            if avg_nll < best_nll:
                best_nll = avg_nll
                best_sigma = sigma

        self.sigma_pert = best_sigma
        logger.info("Input perturbation σ = %.6e", self.sigma_pert)

    # ------------------------------------------------------------------
    def predict(
        self, inputs: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if self.sigma_pert is None:
            raise RuntimeError("Must call fit() before predict().")
        x_batch = jnp.atleast_1d(inputs)
        key = jax.random.PRNGKey(self.seed)
        sigma = self.sigma_pert

        def perturb_and_forward(k, x):
            noise = sigma * jax.random.normal(k, x.shape)
            return self.apply_fn(self.full_params, x + noise)

        keys = jax.random.split(key, self.num_perturb)
        preds = jax.vmap(perturb_and_forward)(keys, jnp.broadcast_to(x_batch, (self.num_perturb,) + x_batch.shape))
        # preds shape: (S, 1, *spatial, 1)
        preds = preds[:, 0]  # (S, *spatial, 1)
        mean = jnp.mean(preds, axis=0)
        var = jnp.var(preds, axis=0)
        return mean, var

