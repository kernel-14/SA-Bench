## uncertainty/weight_space.py
"""Isotropic Gaussian weight-space belief for the LUNO reproduction.

This module implements ``WeightSpaceBelief``, which represents the simplest
weight-space uncertainty model: an isotropic Gaussian N(w*, σ²I) centred at
the MAP parameter vector w*.

Role in the system:
  - Consumed by ``SamplePushforward`` (baselines/) to implement Sample-Iso:
    draw 200 weight samples, push each through the (nonlinear) FNO, compute
    empirical mean and variance.
  - Consumed by ``LUNOInference`` (uncertainty/luno.py) to implement LUNO-Iso:
    the marginal variance simplifies to σ² · ‖J(x)‖² (no Woodbury inversion
    needed), so ``LUNOInference`` only needs ``get_sigma()`` / ``sigma_sq``.
  - Mutated by ``Calibrator`` (evaluation/calibration.py) via
    ``set_sigma_sq()`` during the 500-point log-spaced grid search that
    minimises validation NLL.

Paper references:
  - Appendix D.3.3: "Isotropic Gaussian (*-Iso): N(w*, Σ := σ²I)"
  - Appendix D.5: calibration via grid search over 500 log-spaced values
  - config.yaml calibration.sigma_sq_iso_center: 1.0 (initial grid centre)
  - config.yaml calibration.grid_range_factor: 100.0 (grid spans [c/100, c*100])
  - config.yaml uncertainty.sampling.n_samples: 200

Design constraints:
  - ``sigma_sq`` is stored as a Python ``float`` (not a JAX array) so that
    calibration can mutate it in a plain Python loop without triggering
    JAX retracing.
  - No ``woodbury_matvec`` method — that belongs to ``LaplaceApprox``.
    For isotropic covariance, J·Σ·J^T = σ²·J·J^T, handled in LUNOInference.
  - Zero dependencies on model or data classes (only jax, jnp, utils.jax_utils).
"""

from __future__ import annotations

import logging
from typing import Optional

import jax
import jax.numpy as jnp

import utils.jax_utils  # noqa: F401  imported for project consistency

logger = logging.getLogger(__name__)


class WeightSpaceBelief:
    """Isotropic Gaussian weight-space belief N(w*, σ²I).

    Stores the MAP parameter vector ``mean`` and a scalar variance
    ``sigma_sq``.  Supports weight sampling (reparameterisation trick) and
    exposes ``sigma_sq`` for mutation during calibration.

    This class handles only the isotropic case (``cov_type='iso'``).  The
    low-rank Laplace case is handled by ``LaplaceApprox`` in
    ``uncertainty/luno.py``.

    Attributes:
        mean: Flattened MAP parameter vector w*, shape ``[p]``.  Produced
            by ``utils.jax_utils.flatten_params(params)`` before construction.
        cov_type: Covariance type identifier.  Always ``'iso'`` for this
            class; stored for documentation and forward-compatibility.
        sigma_sq: Scalar variance σ² > 0.  Stored as a Python ``float`` so
            that calibration can mutate it without JAX retracing.

    Example::

        flat_params, _ = flatten_params(trained_params)
        belief = WeightSpaceBelief(
            mean=flat_params,
            cov_type='iso',
            sigma_sq=1.0,  # initial value; calibrated later
        )

        # Draw 200 weight samples
        key = jax.random.PRNGKey(0)
        samples = belief.sample(key, n_samples=200)
        # samples.shape == (200, p)

        # Calibration loop (in Calibrator):
        belief.set_sigma_sq(0.01)
        sigma = belief.get_sigma()  # 0.1
    """

    def __init__(
        self,
        mean: jnp.ndarray,
        cov_type: str = "iso",
        sigma_sq: float = 1.0,
    ) -> None:
        """Initialise the isotropic Gaussian weight-space belief.

        Args:
            mean: Flattened MAP parameter vector w*, shape ``[p]``.
                Must be a 1-D ``jnp.ndarray``.  Produced by calling
                ``utils.jax_utils.flatten_params(trained_params)`` in
                ``main.py`` before constructing this object.
            cov_type: Covariance type identifier.  Must be ``'iso'``.
                Stored for documentation and forward-compatibility.
                Default: ``'iso'``.
            sigma_sq: Initial scalar variance σ² > 0.  Corresponds to
                ``config.calibration.sigma_sq_iso_center = 1.0`` before
                calibration.  Updated via ``set_sigma_sq()`` during the
                calibration grid search.  Stored as a Python ``float`` to
                avoid JAX retracing when mutated.  Default: ``1.0``.

        Raises:
            ValueError: If ``mean`` is not a 1-D array.
            ValueError: If ``cov_type`` is not ``'iso'``.
            ValueError: If ``sigma_sq <= 0``.

        Example::

            flat, _ = flatten_params(params)
            belief = WeightSpaceBelief(mean=flat, cov_type='iso', sigma_sq=1.0)
        """
        # ------------------------------------------------------------------
        # Validate mean
        # ------------------------------------------------------------------
        mean_arr: jnp.ndarray = jnp.asarray(mean)
        if mean_arr.ndim != 1:
            raise ValueError(
                f"mean must be a 1-D array (flattened parameter vector), "
                f"got shape {mean_arr.shape}"
            )

        # ------------------------------------------------------------------
        # Validate cov_type
        # ------------------------------------------------------------------
        if cov_type != "iso":
            raise ValueError(
                f"WeightSpaceBelief only supports cov_type='iso', "
                f"got '{cov_type}'.  For low-rank Laplace, use LaplaceApprox."
            )

        # ------------------------------------------------------------------
        # Validate sigma_sq
        # ------------------------------------------------------------------
        sigma_sq_float: float = float(sigma_sq)
        if sigma_sq_float <= 0.0:
            raise ValueError(
                f"sigma_sq must be positive, got {sigma_sq_float}"
            )

        # ------------------------------------------------------------------
        # Store fields
        # ------------------------------------------------------------------
        self.mean: jnp.ndarray = mean_arr
        """Flattened MAP parameter vector w*, shape [p]."""

        self.cov_type: str = cov_type
        """Covariance type identifier. Always 'iso' for this class."""

        self.sigma_sq: float = sigma_sq_float
        """Scalar variance σ² > 0. Python float for calibration mutability."""

        logger.debug(
            "WeightSpaceBelief initialised: p=%d, cov_type='%s', sigma_sq=%.4e",
            int(mean_arr.shape[0]),
            cov_type,
            sigma_sq_float,
        )

    # -----------------------------------------------------------------------
    # Sampling
    # -----------------------------------------------------------------------

    def sample(
        self,
        key: jax.Array,
        n_samples: int = 200,
    ) -> jnp.ndarray:
        """Draw weight samples from N(w*, σ²I) via the reparameterisation trick.

        Computes:

        .. math::

            w^{(s)} = \\mu + \\sigma \\cdot \\varepsilon^{(s)},
            \\quad \\varepsilon^{(s)} \\sim \\mathcal{N}(0, I_p)

        for ``s = 1, ..., n_samples``.

        Args:
            key: JAX PRNG key for reproducible sampling.  Consumed entirely
                by this call; the caller should split the key before passing
                it if the key will be reused.
            n_samples: Number of weight samples to draw.  From
                ``config.uncertainty.sampling.n_samples = 200`` when called
                from ``SamplePushforward``.  Default: ``200``.

        Returns:
            Weight sample matrix, shape ``[n_samples, p]``.  Each row is
            an independent sample from N(w*, σ²I).  The samples are
            ``jnp.ndarray`` with dtype matching ``self.mean``.

        Raises:
            ValueError: If ``n_samples <= 0``.

        Notes:
            - ``sigma_sq`` is read as a Python float at call time, so
              changes via ``set_sigma_sq()`` are reflected immediately
              without retracing.
            - This function is JAX-traceable and can be used inside
              ``jax.jit``-compiled functions, provided ``n_samples`` is
              treated as a static integer.

        Example::

            key = jax.random.PRNGKey(42)
            samples = belief.sample(key, n_samples=200)
            # samples.shape == (200, p)
            # Each row: w* + σ * ε, ε ~ N(0, I)
        """
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")

        p: int = int(self.mean.shape[0])

        # Draw standard normal noise: shape [n_samples, p]
        eps: jnp.ndarray = jax.random.normal(
            key,
            shape=(n_samples, p),
            dtype=self.mean.dtype,
        )  # [n_samples, p]

        # Reparameterisation: w = μ + σ · ε
        # sigma_sq is a Python float → sqrt is a Python/NumPy scalar
        sigma: float = float(jnp.sqrt(self.sigma_sq))

        # mean[None, :] broadcasts [p] → [1, p] for addition with [n_samples, p]
        samples: jnp.ndarray = self.mean[jnp.newaxis, :] + sigma * eps
        # samples.shape: [n_samples, p]

        return samples

    # -----------------------------------------------------------------------
    # Sigma Access
    # -----------------------------------------------------------------------

    def get_sigma(self) -> float:
        """Return the standard deviation σ = sqrt(σ²) as a Python float.

        Used by ``LUNOInference`` for the LUNO-Iso case, where the marginal
        variance at output point x simplifies to:

        .. math::

            \\text{Var}[F(a)(x)] = \\sigma^2 \\cdot \\|J(a, x)\\|_F^2

        and ``LUNOInference`` needs σ² (or σ) to scale the Jacobian norm.

        Returns:
            Standard deviation σ = sqrt(σ²) as a Python ``float``.
            Always positive.

        Example::

            belief.set_sigma_sq(0.01)
            sigma = belief.get_sigma()  # 0.1
            # marginal_var = sigma**2 * jnp.sum(J**2, axis=-1)
        """
        return float(jnp.sqrt(self.sigma_sq))

    # -----------------------------------------------------------------------
    # Calibration Interface
    # -----------------------------------------------------------------------

    def set_sigma_sq(self, sigma_sq: float) -> None:
        """Update the variance parameter σ² in-place.

        Called by ``Calibrator._eval_nll`` during the 500-point log-spaced
        grid search that minimises validation NLL (Appendix D.5).

        The calibration grid spans:
        ``[sigma_sq_iso_center / grid_range_factor,
           sigma_sq_iso_center * grid_range_factor]``
        = ``[1.0 / 100.0, 1.0 * 100.0]`` = ``[0.01, 100.0]``
        with 500 log-spaced points, per ``config.calibration``.

        Args:
            sigma_sq: New variance value σ² > 0.  Converted to a Python
                ``float`` to ensure mutability and prevent accidental JAX
                array storage.

        Raises:
            ValueError: If ``sigma_sq <= 0``.

        Notes:
            - This is a plain Python mutation (not JAX-traced).  Calibration
              is a Python-level loop, so no JIT recompilation occurs.
            - After calibration, the final best value is set via this method
              before the method is used for test evaluation.

        Example::

            # Calibration loop (in Calibrator):
            for sigma_sq_candidate in log_grid:
                belief.set_sigma_sq(sigma_sq_candidate)
                nll = eval_nll_on_val_set(belief)
            belief.set_sigma_sq(best_sigma_sq)  # final calibrated value
        """
        sigma_sq_float: float = float(sigma_sq)
        if sigma_sq_float <= 0.0:
            raise ValueError(
                f"sigma_sq must be positive, got {sigma_sq_float}"
            )
        self.sigma_sq = sigma_sq_float

        logger.debug("WeightSpaceBelief.set_sigma_sq: sigma_sq=%.4e", sigma_sq_float)

    # -----------------------------------------------------------------------
    # Marginal Variance (convenience for LUNOInference Iso case)
    # -----------------------------------------------------------------------

    def marginal_variance(self, jacobian: jnp.ndarray) -> jnp.ndarray:
        """Compute marginal variances diag(J · σ²I · J^T) = σ² · ‖J‖²_row.

        For the isotropic case, the posterior covariance is Σ = σ²I, so:

        .. math::

            \\text{diag}(J \\Sigma J^\\top)_i
            = \\sigma^2 \\sum_j J_{ij}^2
            = \\sigma^2 \\|J_i\\|^2

        This is a convenience method for ``LUNOInference`` to avoid
        branching on belief type in the marginal variance computation.
        It mirrors the interface of ``LaplaceApprox.marginal_variance``.

        Args:
            jacobian: Jacobian matrix, shape ``[m, p]``, where ``m`` is the
                number of output dimensions (e.g., spatial points × output
                channels) and ``p`` is the number of (last-layer) parameters.

        Returns:
            Marginal variances, shape ``[m]``.  Entry ``i`` equals
            ``sigma_sq * sum_j jacobian[i, j]^2``.

        Example::

            # jacobian has shape [n_spatial * d_out, p_last]
            var = belief.marginal_variance(jacobian)
            std = jnp.sqrt(jnp.maximum(var, 0.0))
        """
        # σ² · sum_j J[i,j]^2 = σ² · ‖J_i‖²
        row_norms_sq: jnp.ndarray = jnp.sum(jacobian ** 2, axis=-1)  # [m]
        return self.sigma_sq * row_norms_sq  # [m]

    # -----------------------------------------------------------------------
    # Representation
    # -----------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a concise string representation of the belief.

        Returns:
            A string summarising the key parameters.

        Example::

            WeightSpaceBelief(p=12345, cov_type='iso', sigma_sq=1.0000e+00,
                              sigma=1.0000e+00)
        """
        p: int = int(self.mean.shape[0])
        sigma: float = self.get_sigma()
        return (
            f"WeightSpaceBelief("
            f"p={p}, "
            f"cov_type='{self.cov_type}', "
            f"sigma_sq={self.sigma_sq:.4e}, "
            f"sigma={sigma:.4e}"
            f")"
        )
