"""
calibration.py
==============
Grid‑search calibration of the single hyper‑parameter (e.g., variance, prior
variance, perturbation magnitude) that governs a Gaussian weight‑space belief
or input‑perturbation method.

The ``Calibration`` class follows the design diagram:

  * ``__init__``   – receives the UQ‑method *class*, the trained model, the
                     calibration dataset, and all configuration.
  * ``evaluate_nll`` – computes the average marginal negative log‑likelihood
                       of a given ``UQMethod`` instance on the calibration set.
  * ``run_grid_search`` – loops over a log‑spaced grid of candidate values,
                          selects the one that minimises the NLL, and returns
                          ``(best_value, best_nll)``.

The grid‑search is **generic**: every UQ‑method that accepts a scalar
hyper‑parameter in its constructor (e.g., ``sigma2``, ``sigma_pert``, or
``tau``) is supported.  For Laplace‑based methods the pre‑computed GGN
eigen‑basis ``V`` and the number of training inputs ``n_train`` must be
provided as extra keyword arguments; they are forwarded to the constructor.
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import Any, Dict, Optional, Tuple, Type

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm

# Safe imports: the configuration classes are immutable and already available.
from config import UQConfig  # contains CalibrationConfig
from uncertainty import UQMethod  # abstract base
from uncertainty import _gaussian_nll  # convenient utility (safe to import)

logger = logging.getLogger(__name__)

# ============================================================================
# Calibration class
# ============================================================================


class Calibration:
    """
    Perform a grid‑search over a scalar hyper‑parameter of a ``UQMethod``.

    Parameters
    ----------
    uq_method_class : Type[UQMethod]
        The class of the uncertainty quantification method to calibrate
        (e.g., ``LUNOIsotropic``, ``SampleLaplace``, ``InputPerturbation``).
    model : nn.Module
        The trained FNO model (used to obtain ``apply_fn``).
    params : dict
        Flax parameter dictionary (the MAP estimate).
    calib_ds : tuple of ndarray
        ``(inputs, targets)`` – the calibration (validation) dataset.
        Shapes: ``inputs`` ``(N, T, *spatial, C)``, ``targets`` ``(N, *spatial, 1)``.
    config : UQConfig
        The top‑level UQ configuration, which contains ``CalibrationConfig``.
    extra_constructor_kwargs : dict, optional
        Additional keyword arguments that are passed to the constructor of
        ``uq_method_class`` when creating a candidate instance.  This is used
        for Laplace methods (``V``, ``n_train``, …) or any other extra data
        that the method requires at construction time.
    batch_size : int, optional
        Mini‑batch size for evaluating the NLL on the calibration dataset
        (default: 8).
    seed : int, optional
        Random seed passed to each candidate instance (default: 0).
    """

    def __init__(
        self,
        uq_method_class: Type[UQMethod],
        model: Any,
        params: Dict[str, Any],
        calib_ds: Tuple[np.ndarray, np.ndarray],
        config: UQConfig,
        extra_constructor_kwargs: Optional[Dict[str, Any]] = None,
        batch_size: int = 8,
        seed: int = 0,
    ) -> None:
        self.uq_method_class = uq_method_class
        self.model = model
        self.params = params
        self.calib_x, self.calib_y = calib_ds
        self.config = config
        self.batch_size = batch_size
        self.seed = seed
        self.extra_kwargs = extra_constructor_kwargs or {}

        # Quick validation
        if self.calib_x.shape[0] != self.calib_y.shape[0]:
            raise ValueError(
                "Mismatch between number of calibration inputs "
                f"({self.calib_x.shape[0]}) and targets ({self.calib_y.shape[0]})"
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate_nll(self, uq_instance: UQMethod) -> float:
        """
        Average marginal negative log‑likelihood of ``uq_instance`` on the
        calibration dataset.

        Parameters
        ----------
        uq_instance : UQMethod
            An already‑constructed UQ method whose ``predict(inputs)``
            returns ``(mean, variance)``.

        Returns
        -------
        avg_nll : float
            Mean per‑point NLL over the whole calibration set.
        """
        total_nll = 0.0
        total_points = 0
        n_samples = self.calib_x.shape[0]

        for start in range(0, n_samples, self.batch_size):
            slice_ = slice(start, start + self.batch_size)
            x_batch = jnp.asarray(self.calib_x[slice_])
            y_batch = jnp.asarray(self.calib_y[slice_])

            mean, var = uq_instance.predict(x_batch)
            # Both have shape (batch, *spatial, 1) or (batch, *spatial) if squeezed?
            # The uq_instance should return the same shape as model output.
            # We flatten to treat every spatial point independently.
            mean_flat = mean.ravel()
            var_flat = var.ravel()
            y_flat = y_batch.ravel()

            # Marginal Gaussian NLL per point
            nll_per_point = _gaussian_nll(y_flat, mean_flat, var_flat)
            batch_nll = jnp.sum(nll_per_point)
            batch_points = y_flat.size

            total_nll += float(batch_nll)
            total_points += batch_points

        avg_nll = total_nll / total_points if total_points > 0 else float("inf")
        return avg_nll

    def run_grid_search(self) -> Tuple[float, float]:
        """
        Perform the log‑space grid search and return the optimal hyper‑parameter.

        The search range and number of points are taken from
        ``config.calibration``.  For Laplace / isotropic methods the
        search variable is the variance ``σ²``.  For input perturbation it is
        the perturbation standard deviation ``σ_pert``.  The method is
        auto‑detected from the class name (case‑insensitive partial match).

        Returns
        -------
        best_value : float
            The hyper‑parameter value that achieves the lowest NLL.
        best_nll : float
            The corresponding average marginal NLL.
        """
        cfg = self.config.calibration
        method_name = self.uq_method_class.__name__.lower()

        # ── Ensemble requires no calibration ────────────────────────────────
        if "ensemble" in method_name:
            logger.info("Ensemble does not require hyper‑parameter calibration.")
            return (0.0, 0.0)

        # ── Determine parameter name, range, and conversion function ─────────
        if "input_perturb" in method_name:
            param_range = getattr(cfg, "input_perturb_sigma_range", (-4.0, -1.0))
            param_name = "sigma_pert"
            make_kwargs = lambda val: {"sigma_pert": val}
        else:
            # Assume any method with “laplace” / “iso” uses variance σ².
            param_range = cfg.sigma_range
            param_name = "sigma2"
            # For Laplace we pass τ = 1/σ² (variance → precision), but the
            # constructor of Laplace classes expects `tau` directly, so we
            # need to convert.  However, our design favours passing σ² and
            # letting the class handle the conversion.  To keep things generic
            # we pass both: the class will decide.  But to avoid requiring
            # knowledge of internal naming, we pass `sigma2` to all methods,
            # and for Laplace classes we additionally pass `tau` if they
            # accept it.  A clean way is to pass **{'sigma2': val} and also
            # **{'tau': 1.0/val} for Laplace.  However, the Laplace classes
            # in the paper use τ, while Iso uses σ².  To maintain uniformity
            # we can pass `sigma2` and let the constructor handle it.
            # The provided uncertainty.py classes accept `sigma2` for Iso and
            # `tau` for LA.  But we can detect the class and adapt.
            if "laplace" in method_name:
                # Convert to precision tau = 1/sigma2
                convert = lambda s2: 1.0 / s2
                def make_kwargs(val):
                    tau_val = 1.0 / val
                    kwargs = {"tau": tau_val}
                    # Also pass extra kwrags (V, n_train) if present
                    kwargs.update(self.extra_kwargs)
                    return kwargs
            else:
                make_kwargs = lambda val: {"sigma2": val}

        # ── Generate candidate grid in log space ────────────────────────────
        grid_size = cfg.grid_size
        candidates = np.logspace(
            param_range[0], param_range[1], num=grid_size, base=10.0
        )

        best_nll = float("inf")
        best_val = candidates[0]

        logger.info(
            "Starting grid search for %s (%d candidates, range [%.2e, %.2e])",
            param_name,
            grid_size,
            10.0 ** param_range[0],
            10.0 ** param_range[1],
        )

        # ── Loop over candidates ────────────────────────────────────────────
        # We use tqdm to show progress.
        for val in tqdm(candidates, desc=f"Calibrating {param_name}", unit="cand"):
            # Build constructor kwargs
            kwargs = {
                "apply_fn": self.model.apply,
                "params": self.params,
                "config": self.config,
                "seed": self.seed,
            }
            # Merge the method‑specific hyper‑parameter
            kwargs.update(make_kwargs(val))

            # Instantiate the UQ method (do NOT call .fit() – we only need predict)
            try:
                uq = self.uq_method_class(**kwargs)
            except TypeError as e:
                # If the class does not accept the auto‑generated kwargs, try
                # a more generic approach: use the class directly with only
                # `sigma2` (fallback).
                # This is a safety net; ideally the class constructors are
                # designed uniformly.
                logger.debug(
                    "Constructor failed with %s – falling back to sigma2=%s",
                    kwargs,
                    val,
                )
                kwargs = {
                    "apply_fn": self.model.apply,
                    "params": self.params,
                    "config": self.config,
                    "seed": self.seed,
                    param_name: val,
                }
                uq = self.uq_method_class(**kwargs)

            nll = self.evaluate_nll(uq)
            if nll < best_nll:
                best_nll = nll
                best_val = float(val)

        # ── Warn if optimum lies at the boundary ────────────────────────────
        if best_val == candidates[0] or best_val == candidates[-1]:
            warnings.warn(
                f"Best {param_name} = {best_val:.4e} lies at the boundary of the "
                f"search range [{10.0**param_range[0]:.2e}, {10.0**param_range[1]:.2e}]. "
                "Consider widening the search range.",
                stacklevel=2,
            )

        logger.info(
            "Calibration finished. Best %s = %.6e (NLL = %.6f)",
            param_name,
            best_val,
            best_nll,
        )
        return best_val, best_nll

