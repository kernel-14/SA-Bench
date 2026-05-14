## evaluation/metrics.py
"""Evaluation metrics for the LUNO reproduction.

Implements the three evaluation metrics used throughout the LUNO paper:
  - RMSE (Appendix D.4.1): root mean squared error of mean predictions
  - Marginal NLL (Appendix D.4.2): negative log-likelihood under independent
    Gaussian marginals at each spatial point
  - Marginal χ² statistic (Appendix D.4.3): mean squared standardized residual

All metrics are computed over batched inputs of shape [n_test, ...] and
averaged over both test pairs and spatial points, matching the "expected
value over 250 test pairs" framing in Appendix D.4.

Paper references:
  - Appendix D.4.1: RMSE definition
  - Appendix D.4.2: Marginal NLL definition
  - Appendix D.4.3: χ² statistic definition
  - Section 5 / Table 1: reported metric values (e.g., RMSE ~3.62e-2,
    NLL ~-2.08, χ² ~1.022 for LUNO-LA on Burgers)
  - config.yaml evaluation.metrics: ['rmse', 'nll', 'chi2']

Design notes:
  - All methods are static (no instance state needed).
  - Inputs are reshaped to [n_test, n_spatial_flat] at the start of each
    method to handle 1D, 2D, and multi-channel cases uniformly.
  - NLL is averaged (not summed) over spatial points to produce
    scale-invariant values comparable across spatial resolutions.
    This is consistent with the reported values (~-2 range for 256 spatial
    points in the Burgers experiment).
  - sigma is clipped from below at eps=1e-8 before division to prevent
    inf/nan for overconfident predictions.
  - safe_log from utils.jax_utils is used for the log(sigma^2) term in NLL.
  - All methods return JAX scalars (shape ()) compatible with both JIT
    contexts and Python calibration loops.
"""

from __future__ import annotations

import logging
from typing import Dict

import jax.numpy as jnp

from utils.jax_utils import safe_log

logger = logging.getLogger(__name__)

# Numerical floor for sigma to prevent division by zero and log(0)
_SIGMA_EPS: float = 1e-8

# Constant: log(2 * pi), precomputed for NLL
_LOG_2PI: float = float(jnp.log(2.0 * jnp.pi))


class Metrics:
    """Static methods for computing RMSE, marginal NLL, and χ² statistics.

    All methods accept batched inputs of shape ``[n_test, ...]`` where the
    trailing dimensions are any combination of spatial and channel dims.
    Inputs are flattened to ``[n_test, n_spatial_flat]`` internally.

    The "expected value over test samples" framing from the paper means:
      - RMSE: mean over n_test of per-sample RMSE (sqrt of mean squared error)
      - NLL: mean over n_test of per-sample NLL (mean over spatial points)
      - χ²: mean over n_test of per-sample χ² (mean over spatial points)

    Example::

        y_true = jnp.ones([250, 256, 1])   # [n_test, spatial, out_channels]
        y_pred = jnp.ones([250, 256, 1]) * 0.9
        sigma  = jnp.ones([250, 256, 1]) * 0.1

        rmse = Metrics.compute_rmse(y_true, y_pred)
        nll  = Metrics.compute_nll(y_true, y_pred, sigma)
        chi2 = Metrics.compute_chi2(y_true, y_pred, sigma)
        all_metrics = Metrics.compute_all(y_true, y_pred, sigma)
        # all_metrics == {'rmse': ..., 'nll': ..., 'chi2': ...}
    """

    @staticmethod
    def compute_rmse(
        y_true: jnp.ndarray,
        y_pred: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute the expected root mean squared error over test pairs.

        Implements the RMSE from Appendix D.4.1:

        .. math::

            \\text{RMSE} = \\sqrt{\\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2}

        The "expected" value is the average over ``n_test`` test pairs of
        the per-pair RMSE (square root of the mean squared error over all
        spatial points within that pair).

        Algorithm:
          1. Flatten inputs to ``[n_test, n_spatial_flat]``.
          2. Compute squared errors: ``(y_true - y_pred) ** 2``.
          3. Mean over spatial dimension → per-pair MSE, shape ``[n_test]``.
          4. Square root → per-pair RMSE, shape ``[n_test]``.
          5. Mean over ``n_test`` → scalar.

        Args:
            y_true: Ground truth values. Shape ``[n_test, ...]`` where
                ``...`` is any combination of spatial and channel dims.
                Typically ``[n_test, spatial_res, out_channels]`` for 1D
                or ``[n_test, H, W, out_channels]`` for 2D.
            y_pred: Predicted mean values. Same shape as ``y_true``.

        Returns:
            Scalar RMSE value (JAX array of shape ``()``). Matches the
            format in Tables 1, 4, 5 of the paper (e.g., ``3.62e-2``).

        Raises:
            ValueError: If ``y_true`` and ``y_pred`` have different shapes.

        Example::

            rmse = Metrics.compute_rmse(y_true, y_pred)
            # rmse ≈ 3.62e-2 for LUNO-LA on Burgers (Table 1)
        """
        y_true_arr: jnp.ndarray = jnp.asarray(y_true, dtype=jnp.float32)
        y_pred_arr: jnp.ndarray = jnp.asarray(y_pred, dtype=jnp.float32)

        if y_true_arr.shape != y_pred_arr.shape:
            raise ValueError(
                f"y_true and y_pred must have the same shape, "
                f"got y_true.shape={y_true_arr.shape} and "
                f"y_pred.shape={y_pred_arr.shape}"
            )

        n_test: int = y_true_arr.shape[0]

        # Flatten to [n_test, n_spatial_flat]
        y_true_flat: jnp.ndarray = y_true_arr.reshape(n_test, -1)
        y_pred_flat: jnp.ndarray = y_pred_arr.reshape(n_test, -1)

        # Squared errors: [n_test, n_spatial_flat]
        sq_errors: jnp.ndarray = (y_true_flat - y_pred_flat) ** 2

        # Per-pair MSE: mean over spatial dimension → [n_test]
        per_pair_mse: jnp.ndarray = jnp.mean(sq_errors, axis=1)  # [n_test]

        # Per-pair RMSE: sqrt → [n_test]
        per_pair_rmse: jnp.ndarray = jnp.sqrt(per_pair_mse)  # [n_test]

        # Expected RMSE: mean over test pairs → scalar
        rmse: jnp.ndarray = jnp.mean(per_pair_rmse)  # scalar

        return rmse

    @staticmethod
    def compute_nll(
        y_true: jnp.ndarray,
        y_pred: jnp.ndarray,
        sigma: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute the expected marginal negative log-likelihood over test pairs.

        Implements the marginal NLL from Appendix D.4.2, treating each
        spatial point as an independent Gaussian:

        .. math::

            \\text{NLL} = -\\sum_{i=1}^{n} \\log \\mathcal{N}(y_i; \\hat{y}_i, \\sigma_i^2)
            = \\sum_{i=1}^{n} \\left[
                \\frac{1}{2} \\log(2\\pi\\sigma_i^2)
                + \\frac{(y_i - \\hat{y}_i)^2}{2\\sigma_i^2}
            \\right]

        The "expected" value is the average over ``n_test`` test pairs of
        the per-pair NLL (averaged over spatial points within each pair).
        Averaging over spatial points (rather than summing) produces
        scale-invariant values comparable across different spatial
        resolutions, consistent with the reported values (~-2 range for
        256 spatial points in the Burgers experiment).

        Algorithm:
          1. Flatten inputs to ``[n_test, n_spatial_flat]``.
          2. Clip ``sigma`` from below at ``_SIGMA_EPS = 1e-8``.
          3. Compute log-normalizer: ``0.5 * (log(2*pi) + safe_log(sigma^2))``.
          4. Compute quadratic term: ``(y_true - y_pred)^2 / (2 * sigma^2)``.
          5. Sum both terms → per-point NLL.
          6. Mean over spatial dimension → per-pair NLL, shape ``[n_test]``.
          7. Mean over ``n_test`` → scalar.

        Args:
            y_true: Ground truth values. Shape ``[n_test, ...]``.
            y_pred: Predicted mean values. Same shape as ``y_true``.
            sigma: Predicted marginal standard deviations. Same shape as
                ``y_true``. Must be positive; values below ``1e-8`` are
                clipped for numerical stability.

        Returns:
            Scalar marginal NLL value (JAX array of shape ``()``). Negative
            values indicate well-calibrated predictions with small errors
            (e.g., ``-2.0787`` for LUNO-LA on Burgers, Table 1).

        Raises:
            ValueError: If ``y_true``, ``y_pred``, and ``sigma`` do not all
                have the same shape.

        Notes:
            - ``safe_log`` from ``utils.jax_utils`` is used for the
              ``log(sigma^2)`` term to prevent ``log(0) = -inf``.
            - The formula is the standard Gaussian NLL; negative values
              arise naturally when ``sigma < 1/sqrt(2*pi*e) ≈ 0.242``.

        Example::

            nll = Metrics.compute_nll(y_true, y_pred, sigma)
            # nll ≈ -2.0787 for LUNO-LA on Burgers (Table 1)
        """
        y_true_arr: jnp.ndarray = jnp.asarray(y_true, dtype=jnp.float32)
        y_pred_arr: jnp.ndarray = jnp.asarray(y_pred, dtype=jnp.float32)
        sigma_arr: jnp.ndarray = jnp.asarray(sigma, dtype=jnp.float32)

        if not (y_true_arr.shape == y_pred_arr.shape == sigma_arr.shape):
            raise ValueError(
                f"y_true, y_pred, and sigma must all have the same shape. "
                f"Got y_true.shape={y_true_arr.shape}, "
                f"y_pred.shape={y_pred_arr.shape}, "
                f"sigma.shape={sigma_arr.shape}"
            )

        n_test: int = y_true_arr.shape[0]

        # Flatten to [n_test, n_spatial_flat]
        y_true_flat: jnp.ndarray = y_true_arr.reshape(n_test, -1)
        y_pred_flat: jnp.ndarray = y_pred_arr.reshape(n_test, -1)
        sigma_flat: jnp.ndarray = sigma_arr.reshape(n_test, -1)

        # Clip sigma from below to prevent division by zero and log(0)
        sigma_clipped: jnp.ndarray = jnp.maximum(sigma_flat, _SIGMA_EPS)

        # Variance: sigma^2, shape [n_test, n_spatial_flat]
        sigma_sq: jnp.ndarray = sigma_clipped ** 2

        # Log-normalizer term: 0.5 * log(2 * pi * sigma^2)
        # = 0.5 * (log(2*pi) + log(sigma^2))
        # Use safe_log for numerical stability
        log_normalizer: jnp.ndarray = 0.5 * (
            _LOG_2PI + safe_log(sigma_sq, eps=_SIGMA_EPS)
        )  # [n_test, n_spatial_flat]

        # Quadratic term: (y_true - y_pred)^2 / (2 * sigma^2)
        residuals: jnp.ndarray = y_true_flat - y_pred_flat  # [n_test, n_spatial_flat]
        quadratic_term: jnp.ndarray = (residuals ** 2) / (2.0 * sigma_sq)
        # [n_test, n_spatial_flat]

        # Per-point NLL: log_normalizer + quadratic_term
        nll_per_point: jnp.ndarray = log_normalizer + quadratic_term
        # [n_test, n_spatial_flat]

        # Per-pair NLL: mean over spatial dimension → [n_test]
        per_pair_nll: jnp.ndarray = jnp.mean(nll_per_point, axis=1)  # [n_test]

        # Expected NLL: mean over test pairs → scalar
        nll: jnp.ndarray = jnp.mean(per_pair_nll)  # scalar

        return nll

    @staticmethod
    def compute_chi2(
        y_true: jnp.ndarray,
        y_pred: jnp.ndarray,
        sigma: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute the expected marginal χ² statistic over test pairs.

        Implements the marginal χ² statistic from Appendix D.4.3:

        .. math::

            \\chi^2 = \\frac{1}{n} \\sum_{i=1}^{n}
                \\frac{(y_i - \\hat{y}_i)^2}{\\sigma_i^2}

        A value close to 1 indicates well-calibrated uncertainty:
          - χ² > 1: overconfident (uncertainty underestimated)
          - χ² < 1: underconfident (uncertainty overestimated)
          - χ² ≈ 1: well-calibrated

        The "expected" value is the average over ``n_test`` test pairs of
        the per-pair χ² (mean over spatial points within each pair).

        Algorithm:
          1. Flatten inputs to ``[n_test, n_spatial_flat]``.
          2. Clip ``sigma`` from below at ``_SIGMA_EPS = 1e-8``.
          3. Compute standardized squared residuals:
             ``(y_true - y_pred)^2 / sigma^2``.
          4. Mean over spatial dimension → per-pair χ², shape ``[n_test]``.
          5. Mean over ``n_test`` → scalar.

        Args:
            y_true: Ground truth values. Shape ``[n_test, ...]``.
            y_pred: Predicted mean values. Same shape as ``y_true``.
            sigma: Predicted marginal standard deviations. Same shape as
                ``y_true``. Must be positive; values below ``1e-8`` are
                clipped for numerical stability.

        Returns:
            Scalar χ² value (JAX array of shape ``()``). Values in the
            paper range from ``0.864`` (LUNO-Iso, slightly underconfident)
            to ``12.674`` (Ensemble, strongly overconfident) for the
            Burgers experiment (Table 1).

        Raises:
            ValueError: If ``y_true``, ``y_pred``, and ``sigma`` do not all
                have the same shape.

        Notes:
            - The χ² statistic is the mean squared standardized residual,
              not the sum. This makes it scale-invariant across different
              numbers of spatial points.
            - Values above 1 indicate overconfidence (the model is too
              certain about its predictions); values below 1 indicate
              underconfidence. From a UQ perspective, underconfidence is
              preferable to overconfidence (Appendix D.4.3).

        Example::

            chi2 = Metrics.compute_chi2(y_true, y_pred, sigma)
            # chi2 ≈ 1.022 for LUNO-LA on Burgers (Table 1)
        """
        y_true_arr: jnp.ndarray = jnp.asarray(y_true, dtype=jnp.float32)
        y_pred_arr: jnp.ndarray = jnp.asarray(y_pred, dtype=jnp.float32)
        sigma_arr: jnp.ndarray = jnp.asarray(sigma, dtype=jnp.float32)

        if not (y_true_arr.shape == y_pred_arr.shape == sigma_arr.shape):
            raise ValueError(
                f"y_true, y_pred, and sigma must all have the same shape. "
                f"Got y_true.shape={y_true_arr.shape}, "
                f"y_pred.shape={y_pred_arr.shape}, "
                f"sigma.shape={sigma_arr.shape}"
            )

        n_test: int = y_true_arr.shape[0]

        # Flatten to [n_test, n_spatial_flat]
        y_true_flat: jnp.ndarray = y_true_arr.reshape(n_test, -1)
        y_pred_flat: jnp.ndarray = y_pred_arr.reshape(n_test, -1)
        sigma_flat: jnp.ndarray = sigma_arr.reshape(n_test, -1)

        # Clip sigma from below to prevent division by zero
        sigma_clipped: jnp.ndarray = jnp.maximum(sigma_flat, _SIGMA_EPS)

        # Variance: sigma^2, shape [n_test, n_spatial_flat]
        sigma_sq: jnp.ndarray = sigma_clipped ** 2

        # Standardized squared residuals: (y_true - y_pred)^2 / sigma^2
        residuals: jnp.ndarray = y_true_flat - y_pred_flat  # [n_test, n_spatial_flat]
        standardized_sq: jnp.ndarray = (residuals ** 2) / sigma_sq
        # [n_test, n_spatial_flat]

        # Per-pair χ²: mean over spatial dimension → [n_test]
        per_pair_chi2: jnp.ndarray = jnp.mean(standardized_sq, axis=1)  # [n_test]

        # Expected χ²: mean over test pairs → scalar
        chi2: jnp.ndarray = jnp.mean(per_pair_chi2)  # scalar

        return chi2

    @staticmethod
    def compute_all(
        y_true: jnp.ndarray,
        y_pred: jnp.ndarray,
        sigma: jnp.ndarray,
    ) -> Dict[str, jnp.ndarray]:
        """Compute all three evaluation metrics and return as a dict.

        Convenience wrapper that calls ``compute_rmse``, ``compute_nll``,
        and ``compute_chi2`` and packages the results into a dictionary.
        Each metric is computed independently (no shared intermediate
        computation) for clarity and correctness.

        The returned dict keys match the metric names in
        ``config.yaml evaluation.metrics: ['rmse', 'nll', 'chi2']``.

        Args:
            y_true: Ground truth values. Shape ``[n_test, ...]``.
            y_pred: Predicted mean values. Same shape as ``y_true``.
            sigma: Predicted marginal standard deviations. Same shape as
                ``y_true``. Must be positive; values below ``1e-8`` are
                clipped internally.

        Returns:
            Dictionary with keys ``'rmse'``, ``'nll'``, ``'chi2'``, each
            mapping to a scalar JAX array (shape ``()``). Example:

            .. code-block:: python

                {
                    'rmse': jnp.array(0.0362),   # e.g., LUNO-LA Burgers
                    'nll':  jnp.array(-2.0787),
                    'chi2': jnp.array(1.022),
                }

        Raises:
            ValueError: If ``y_true``, ``y_pred``, and ``sigma`` do not all
                have the same shape (propagated from individual methods).

        Example::

            results = Metrics.compute_all(y_true, y_pred, sigma)
            print(f"RMSE: {results['rmse']:.4e}")
            print(f"NLL:  {results['nll']:.4f}")
            print(f"chi2: {results['chi2']:.4f}")
        """
        rmse: jnp.ndarray = Metrics.compute_rmse(y_true, y_pred)
        nll: jnp.ndarray = Metrics.compute_nll(y_true, y_pred, sigma)
        chi2: jnp.ndarray = Metrics.compute_chi2(y_true, y_pred, sigma)

        return {
            "rmse": rmse,
            "nll": nll,
            "chi2": chi2,
        }
