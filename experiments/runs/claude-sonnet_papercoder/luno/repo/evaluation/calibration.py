## evaluation/calibration.py
"""Hyperparameter calibration for LUNO uncertainty quantification methods.

Implements the ``Calibrator`` class that tunes the single scalar hyperparameter
of each UQ method by minimising the marginal NLL on the validation set, as
described in Appendix D.5 of the LUNO paper.

Calibration procedure (Appendix D.5):
  - Use 250 validation input-output pairs
  - Minimise the marginal negative log-likelihood
  - Grid search over a logarithmically spaced grid with 500 points
    centred around the relevant value
  - Grid spans [center / grid_range_factor, center * grid_range_factor]
    = [center / 100, center * 100] with grid_range_factor = 100.0

Hyperparameter mapping:
  - LUNO-LA / Sample-LA: ``prior_prec`` → ``method.belief.set_prior_prec(value)``
  - LUNO-Iso / Sample-Iso: ``sigma_sq`` → ``method.belief.set_sigma_sq(value)``
  - InputPerturbations: ``sigma`` → ``method.set_sigma(value)``
  - DeepEnsemble: no calibration needed

Paper references:
  - Appendix D.5: calibration procedure
  - config.yaml calibration section: all grid parameters
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np

from data.dataset import PDEDataset
from evaluation.metrics import Metrics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases for supported UQ methods
# ---------------------------------------------------------------------------
# We use Any to avoid circular imports; the actual types are:
#   - uncertainty.luno.LUNOInference
#   - baselines.ensembles.SamplePushforward (if it exists)
#   - baselines.input_perturbations.InputPerturbations
#   - baselines.ensembles.DeepEnsemble
UQMethod = Any


class Calibrator:
    """Calibrates a UQ method's scalar hyperparameter via grid search on NLL.

    Performs a 500-point log-spaced grid search over the hyperparameter
    space, evaluating the marginal NLL on all 250 validation pairs at each
    grid point. The hyperparameter value that minimises the validation NLL
    is selected and applied to the method.

    The calibration is a Python-level loop (not ``jax.lax.scan``) because
    each grid evaluation involves mutating the method's internal state via
    a Python setter call, which is incompatible with JAX tracing.

    Attributes:
        method: The UQ method to calibrate. Must expose a setter for its
            scalar hyperparameter (``set_prior_prec``, ``set_sigma_sq``,
            or ``set_sigma``).
        val_dataset: Validation dataset with 250 input-output pairs.
            From ``config.calibration.n_val_pairs = 250``.
        grid_size: Number of log-spaced grid points.
            From ``config.calibration.grid_size = 500``.
        grid_range_factor: Grid spans ``[center/factor, center*factor]``.
            From ``config.calibration.grid_range_factor = 100.0``.
        metrics: ``Metrics`` instance for NLL computation.

    Example::

        calibrator = Calibrator(
            method=luno_la,
            val_dataset=val_ds,
            grid_size=500,
            grid_range_factor=100.0,
        )
        best_prior_prec = calibrator.calibrate(
            param_name='prior_prec',
            center=1.0,
            key=jax.random.PRNGKey(0),
        )
        # luno_la.belief.prior_prec is now set to best_prior_prec
    """

    def __init__(
        self,
        method: UQMethod,
        val_dataset: PDEDataset,
        grid_size: int = 500,
        grid_range_factor: float = 100.0,
        metrics: Optional[Metrics] = None,
    ) -> None:
        """Initialise the Calibrator.

        No computation is performed at initialisation time. The grid search
        is triggered by calling ``calibrate()``.

        Args:
            method: The UQ method to calibrate. Supported types:
                - ``LUNOInference`` with ``LaplaceApprox`` belief:
                  calibrate ``prior_prec`` via ``method.belief.set_prior_prec``.
                - ``LUNOInference`` with ``WeightSpaceBelief`` belief:
                  calibrate ``sigma_sq`` via ``method.belief.set_sigma_sq``.
                - ``SamplePushforward`` with ``LaplaceApprox`` belief:
                  calibrate ``prior_prec`` via ``method.belief.set_prior_prec``.
                - ``SamplePushforward`` with ``WeightSpaceBelief`` belief:
                  calibrate ``sigma_sq`` via ``method.belief.set_sigma_sq``.
                - ``InputPerturbations``:
                  calibrate ``sigma`` via ``method.set_sigma``.
                - ``DeepEnsemble``: no calibration needed; pass but do not
                  call ``calibrate()``.
            val_dataset: Validation dataset. Must contain at least
                ``config.calibration.n_val_pairs = 250`` input-output pairs.
                Pairs are accessed via ``val_dataset.get_batch(indices)``.
            grid_size: Number of log-spaced grid points for the search.
                From ``config.calibration.grid_size = 500``. Default: 500.
            grid_range_factor: The grid spans
                ``[center / grid_range_factor, center * grid_range_factor]``
                in log space. From ``config.calibration.grid_range_factor = 100.0``.
                Default: 100.0.
            metrics: Optional ``Metrics`` instance for NLL computation.
                If ``None``, a fresh ``Metrics()`` instance is created.
                Default: ``None``.

        Raises:
            ValueError: If ``grid_size <= 0``.
            ValueError: If ``grid_range_factor <= 1.0``.
            ValueError: If ``val_dataset`` has zero pairs.
        """
        if grid_size <= 0:
            raise ValueError(f"grid_size must be positive, got {grid_size}")
        if grid_range_factor <= 1.0:
            raise ValueError(
                f"grid_range_factor must be > 1.0, got {grid_range_factor}"
            )
        if len(val_dataset) == 0:
            raise ValueError("val_dataset must contain at least one pair.")

        self.method: UQMethod = method
        self.val_dataset: PDEDataset = val_dataset
        self.grid_size: int = int(grid_size)
        self.grid_range_factor: float = float(grid_range_factor)
        self.metrics: Metrics = metrics if metrics is not None else Metrics()

        # Internal state set during calibrate() and used by _eval_nll()
        self._current_param_name: str = ""
        # Pre-split PRNG keys for validation pairs (set in calibrate())
        self._key_pairs: Optional[jnp.ndarray] = None

        logger.info(
            "Calibrator initialised: method=%s, n_val_pairs=%d, "
            "grid_size=%d, grid_range_factor=%.1f",
            type(method).__name__,
            len(val_dataset),
            grid_size,
            grid_range_factor,
        )

    # -----------------------------------------------------------------------
    # Public Interface
    # -----------------------------------------------------------------------

    def calibrate(
        self,
        param_name: str,
        center: float,
        key: jax.Array,
    ) -> float:
        """Run the grid search calibration and return the best hyperparameter.

        Constructs a 500-point log-spaced grid centred at ``center``,
        evaluates the marginal NLL on all validation pairs at each grid
        point, and returns the value that minimises the NLL. The method's
        hyperparameter is left at the best value after this call.

        Grid construction (Appendix D.5):
          ``grid = np.logspace(log10(center/factor), log10(center*factor), 500)``
          where ``factor = self.grid_range_factor = 100.0``.

        Args:
            param_name: Name of the hyperparameter to calibrate. One of:
                - ``'prior_prec'``: prior precision for Laplace approximation.
                  Setter: ``method.belief.set_prior_prec(value)``.
                - ``'sigma_sq'``: isotropic weight-space variance.
                  Setter: ``method.belief.set_sigma_sq(value)``.
                - ``'sigma'``: input perturbation standard deviation.
                  Setter: ``method.set_sigma(value)``.
            center: Centre of the log-spaced grid. Typical values from
                ``config.yaml``:
                - ``prior_prec``: ``config.calibration.prior_prec_center = 1.0``
                - ``sigma_sq``: ``config.calibration.sigma_sq_iso_center = 1.0``
                - ``sigma``: ``config.calibration.sigma_perturb_center = 0.01``
            key: JAX PRNG key. Used to generate per-pair random keys for
                stochastic methods (``SamplePushforward``,
                ``InputPerturbations``). The same per-pair keys are reused
                across all grid evaluations to ensure that NLL differences
                reflect the hyperparameter, not random variation.

        Returns:
            The best hyperparameter value (Python ``float``) that minimises
            the expected marginal NLL on the validation set. The method's
            internal state is updated to this value before returning.

        Raises:
            ValueError: If ``param_name`` is not one of the supported names.
            ValueError: If ``center <= 0``.

        Notes:
            - The grid search is a Python loop over 500 values. Each
              iteration calls ``_eval_nll``, which runs 250 forward passes
              (or 250 × 200 for sample-based methods). The inner JAX
              computations are JIT-compiled and cached.
            - If all NLL values are NaN or Inf (e.g., due to numerical
              issues at extreme grid values), the centre value is returned
              as a fallback.
            - Progress is logged at INFO level every 50 grid points.

        Example::

            best_prior_prec = calibrator.calibrate(
                param_name='prior_prec',
                center=1.0,
                key=jax.random.PRNGKey(42),
            )
            print(f"Best prior_prec: {best_prior_prec:.4e}")
        """
        # ------------------------------------------------------------------
        # Validate inputs
        # ------------------------------------------------------------------
        valid_param_names = {"prior_prec", "sigma_sq", "sigma"}
        if param_name not in valid_param_names:
            raise ValueError(
                f"param_name must be one of {valid_param_names}, "
                f"got '{param_name}'"
            )
        if float(center) <= 0.0:
            raise ValueError(f"center must be positive, got {center}")

        # ------------------------------------------------------------------
        # Store param_name for use in _eval_nll
        # ------------------------------------------------------------------
        self._current_param_name = param_name

        # ------------------------------------------------------------------
        # Pre-split PRNG keys for all validation pairs.
        # Using the same keys across all grid evaluations ensures that NLL
        # differences are due to the hyperparameter, not random variation.
        # This is important for stochastic methods (Sample-*, InputPerturbations).
        # ------------------------------------------------------------------
        n_val: int = len(self.val_dataset)
        self._key_pairs = jax.random.split(key, n_val)
        # self._key_pairs.shape: [n_val, 2]

        # ------------------------------------------------------------------
        # Build the log-spaced grid
        # Grid spans [center / factor, center * factor] in log space.
        # np.logspace is used (not jnp) since this is a static Python array.
        # ------------------------------------------------------------------
        log_low: float = np.log10(center / self.grid_range_factor)
        log_high: float = np.log10(center * self.grid_range_factor)
        grid: np.ndarray = np.logspace(log_low, log_high, self.grid_size)
        # grid.shape: [grid_size] = [500], dtype float64

        logger.info(
            "Calibrating '%s': center=%.4e, grid=[%.4e, %.4e], "
            "n_grid=%d, n_val=%d",
            param_name,
            center,
            float(grid[0]),
            float(grid[-1]),
            self.grid_size,
            n_val,
        )

        # ------------------------------------------------------------------
        # Grid search: evaluate NLL at each grid point
        # ------------------------------------------------------------------
        nll_values: List[float] = []

        for grid_idx, param_value in enumerate(grid):
            nll_val: float = self._eval_nll(
                param_value=float(param_value),
                param_name=param_name,
                key=key,  # key is not used directly; _key_pairs is used
            )
            nll_values.append(nll_val)

            # Log progress every 50 grid points
            if (grid_idx + 1) % 50 == 0 or grid_idx == 0:
                logger.debug(
                    "  Grid [%d/%d]: %s=%.4e → NLL=%.4f",
                    grid_idx + 1,
                    self.grid_size,
                    param_name,
                    float(param_value),
                    nll_val,
                )

        # ------------------------------------------------------------------
        # Find the best value (minimum NLL)
        # Handle NaN/Inf values by replacing them with a large finite value
        # before argmin, then falling back to center if all are invalid.
        # ------------------------------------------------------------------
        nll_array: np.ndarray = np.array(nll_values, dtype=np.float64)

        # Replace NaN and Inf with a large finite value for argmin
        finite_mask: np.ndarray = np.isfinite(nll_array)
        if not np.any(finite_mask):
            logger.warning(
                "All NLL values are NaN or Inf during calibration of '%s'. "
                "Falling back to center value %.4e.",
                param_name,
                center,
            )
            best_value: float = float(center)
        else:
            # Replace non-finite values with the max finite value + 1
            max_finite: float = float(np.max(nll_array[finite_mask]))
            nll_array_clean: np.ndarray = np.where(
                finite_mask, nll_array, max_finite + 1.0
            )
            best_idx: int = int(np.argmin(nll_array_clean))
            best_value = float(grid[best_idx])
            best_nll: float = float(nll_array[best_idx])

            logger.info(
                "Calibration complete: best %s=%.4e (NLL=%.4f) "
                "at grid index %d/%d",
                param_name,
                best_value,
                best_nll,
                best_idx + 1,
                self.grid_size,
            )

        # ------------------------------------------------------------------
        # Apply the best value to the method (leave it in calibrated state)
        # ------------------------------------------------------------------
        self._apply_setter(param_name, best_value)

        return best_value

    # -----------------------------------------------------------------------
    # Private: NLL Evaluation at a Single Grid Point
    # -----------------------------------------------------------------------

    def _eval_nll(
        self,
        param_value: float,
        param_name: str,
        key: jax.Array,
    ) -> float:
        """Evaluate the expected marginal NLL for a given hyperparameter value.

        Sets the method's hyperparameter to ``param_value``, then evaluates
        the marginal NLL on all validation pairs and returns the mean.

        Algorithm:
          1. Set hyperparameter: call the appropriate setter.
          2. For each of the 250 validation pairs:
             a. Retrieve ``(a, y_true)`` from ``val_dataset``.
             b. Compute ``y_pred = method.predict_mean(a [, key_i])``.
             c. Compute ``var = method.predict_marginal_variance(a [, key_i])``.
             d. Compute ``sigma = sqrt(clip(var, min=1e-10))``.
             e. Compute NLL contribution via ``metrics.compute_nll``.
          3. Return the mean NLL over all validation pairs as a Python float.

        Args:
            param_value: The hyperparameter value to evaluate.
            param_name: The name of the hyperparameter (for setter dispatch).
            key: JAX PRNG key (not used directly; ``self._key_pairs`` is
                used for per-pair keys, which are fixed across grid evals).

        Returns:
            Mean marginal NLL over all validation pairs as a Python ``float``.
            Returns ``float('inf')`` if a numerical error occurs.

        Notes:
            - ``self._key_pairs`` must be set before calling this method
              (done in ``calibrate()``).
            - For deterministic methods (``LUNOInference``, ``DeepEnsemble``),
              the key argument to ``predict_mean`` / ``predict_marginal_variance``
              is not needed. The method type is detected via ``hasattr`` to
              handle both cases.
            - ``var`` is clipped to ``[1e-10, inf)`` before taking the square
              root to prevent NaN from zero or negative variances.
        """
        # ------------------------------------------------------------------
        # Step 1: Set the hyperparameter
        # ------------------------------------------------------------------
        self._apply_setter(param_name, param_value)

        # ------------------------------------------------------------------
        # Step 2: Evaluate NLL on all validation pairs
        # ------------------------------------------------------------------
        total_nll: float = 0.0
        n_val: int = len(self.val_dataset)

        # Determine if the method requires a PRNG key for predictions
        needs_key: bool = self._method_needs_key()

        for i in range(n_val):
            try:
                # Retrieve single validation pair
                # get_batch returns (inputs[i:i+1], targets[i:i+1])
                # We use a single-element batch for efficiency
                a_batch: jnp.ndarray
                y_true_batch: jnp.ndarray
                a_batch, y_true_batch = self.val_dataset.get_batch(
                    jnp.array([i])
                )
                # a_batch.shape: [1, spatial, in_channels] (1D) or [1, H, W, in_channels] (2D)
                # y_true_batch.shape: [1, spatial, out_channels] (1D) or [1, H, W, out_channels] (2D)

                # Get per-pair PRNG key (fixed across grid evaluations)
                key_i: jax.Array = self._key_pairs[i]

                # ----------------------------------------------------------
                # Compute mean prediction and marginal variance
                # Dispatch based on whether the method needs a key.
                # ----------------------------------------------------------
                if needs_key:
                    y_pred_batch: jnp.ndarray = self.method.predict_mean(
                        a_batch, key_i
                    )
                    var_batch: jnp.ndarray = self.method.predict_marginal_variance(
                        a_batch, key_i
                    )
                else:
                    y_pred_batch = self.method.predict_mean(a_batch)
                    var_batch = self.method.predict_marginal_variance(a_batch)

                # ----------------------------------------------------------
                # Compute sigma from variance
                # Clip variance to [1e-10, inf) to prevent NaN from sqrt(0)
                # or sqrt(negative) due to numerical issues.
                # ----------------------------------------------------------
                var_clipped: jnp.ndarray = jnp.maximum(var_batch, 1e-10)
                sigma_batch: jnp.ndarray = jnp.sqrt(var_clipped)

                # ----------------------------------------------------------
                # Compute marginal NLL for this pair
                # Metrics.compute_nll expects [n_test, ...] inputs;
                # our batch has n_test=1, so shapes are already correct.
                # ----------------------------------------------------------
                nll_i: jnp.ndarray = self.metrics.compute_nll(
                    y_true=y_true_batch,
                    y_pred=y_pred_batch,
                    sigma=sigma_batch,
                )

                # Extract as Python float
                nll_i_float: float = float(nll_i)

                # Guard against NaN/Inf from individual pairs
                if np.isfinite(nll_i_float):
                    total_nll += nll_i_float
                else:
                    # Non-finite NLL for this pair: treat as a large penalty
                    total_nll += 1e6

            except Exception as e:  # pylint: disable=broad-except
                # If a forward pass fails (e.g., numerical overflow), treat
                # this pair as having a large NLL penalty.
                logger.debug(
                    "_eval_nll: exception at pair %d with %s=%.4e: %s",
                    i,
                    param_name,
                    param_value,
                    e,
                )
                total_nll += 1e6

        # ------------------------------------------------------------------
        # Step 3: Return mean NLL over all validation pairs
        # ------------------------------------------------------------------
        mean_nll: float = total_nll / float(n_val)
        return mean_nll

    # -----------------------------------------------------------------------
    # Private: Hyperparameter Setter Dispatch
    # -----------------------------------------------------------------------

    def _apply_setter(self, param_name: str, value: float) -> None:
        """Apply the hyperparameter setter for the given parameter name.

        Dispatches to the appropriate setter method based on ``param_name``:
          - ``'prior_prec'`` → ``method.belief.set_prior_prec(value)``
          - ``'sigma_sq'`` → ``method.belief.set_sigma_sq(value)``
          - ``'sigma'`` → ``method.set_sigma(value)``

        Args:
            param_name: Name of the hyperparameter. One of
                ``'prior_prec'``, ``'sigma_sq'``, ``'sigma'``.
            value: New hyperparameter value. Must be positive.

        Raises:
            ValueError: If ``param_name`` is not recognised.
            AttributeError: If the method does not have the expected setter
                (indicates a method/param_name mismatch).

        Notes:
            - For ``'prior_prec'`` and ``'sigma_sq'``, the setter is on
              ``method.belief`` (a ``LaplaceApprox`` or ``WeightSpaceBelief``
              instance).
            - For ``'sigma'``, the setter is directly on ``method``
              (an ``InputPerturbations`` instance).
            - The ``hasattr`` check provides a clear error message if the
              method does not support the requested parameter.
        """
        if param_name == "prior_prec":
            if not hasattr(self.method, "belief"):
                raise AttributeError(
                    f"Method {type(self.method).__name__} does not have a "
                    f"'belief' attribute. Cannot set 'prior_prec'. "
                    f"Ensure the method is a LUNOInference or SamplePushforward "
                    f"with a LaplaceApprox belief."
                )
            if not hasattr(self.method.belief, "set_prior_prec"):
                raise AttributeError(
                    f"method.belief ({type(self.method.belief).__name__}) "
                    f"does not have 'set_prior_prec'. "
                    f"Ensure the belief is a LaplaceApprox instance."
                )
            self.method.belief.set_prior_prec(value)

        elif param_name == "sigma_sq":
            if not hasattr(self.method, "belief"):
                raise AttributeError(
                    f"Method {type(self.method).__name__} does not have a "
                    f"'belief' attribute. Cannot set 'sigma_sq'. "
                    f"Ensure the method is a LUNOInference or SamplePushforward "
                    f"with a WeightSpaceBelief."
                )
            if not hasattr(self.method.belief, "set_sigma_sq"):
                raise AttributeError(
                    f"method.belief ({type(self.method.belief).__name__}) "
                    f"does not have 'set_sigma_sq'. "
                    f"Ensure the belief is a WeightSpaceBelief instance."
                )
            self.method.belief.set_sigma_sq(value)

        elif param_name == "sigma":
            if not hasattr(self.method, "set_sigma"):
                raise AttributeError(
                    f"Method {type(self.method).__name__} does not have "
                    f"'set_sigma'. "
                    f"Ensure the method is an InputPerturbations instance."
                )
            self.method.set_sigma(value)

        else:
            raise ValueError(
                f"Unknown param_name '{param_name}'. "
                f"Supported: 'prior_prec', 'sigma_sq', 'sigma'."
            )

    # -----------------------------------------------------------------------
    # Private: Method Type Detection
    # -----------------------------------------------------------------------

    def _method_needs_key(self) -> bool:
        """Determine whether the method's predict_* methods require a PRNG key.

        Deterministic methods (``LUNOInference``, ``DeepEnsemble``) do not
        require a key. Stochastic methods (``SamplePushforward``,
        ``InputPerturbations``) do.

        Detection is done via ``hasattr`` checks on the method object:
          - If the method has a ``sigma`` attribute (``InputPerturbations``)
            or a ``n_samples`` attribute with a ``belief`` that has a
            ``sample`` method (``SamplePushforward``), it needs a key.
          - Otherwise (``LUNOInference``, ``DeepEnsemble``), it does not.

        Returns:
            ``True`` if the method's ``predict_mean`` and
            ``predict_marginal_variance`` require a ``key`` argument.
            ``False`` otherwise.

        Notes:
            - This detection is based on duck typing rather than isinstance
              checks to avoid circular imports.
            - The detection logic is conservative: if uncertain, it defaults
              to ``False`` (no key), which is safe for deterministic methods.
              Stochastic methods will raise a ``TypeError`` if called without
              a key, which will be caught by the try/except in ``_eval_nll``.
        """
        # InputPerturbations: has 'sigma' attribute (float) and 'set_sigma' method
        if hasattr(self.method, "sigma") and hasattr(self.method, "set_sigma"):
            return True

        # SamplePushforward: has 'n_samples' attribute and 'belief' with 'sample'
        if (
            hasattr(self.method, "n_samples")
            and hasattr(self.method, "belief")
            and hasattr(self.method.belief, "sample")
        ):
            return True

        # LUNOInference, DeepEnsemble: deterministic
        return False

    # -----------------------------------------------------------------------
    # Representation
    # -----------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a concise string representation of the Calibrator.

        Returns:
            A string summarising the key configuration parameters.

        Example::

            Calibrator(method=LUNOInference, n_val_pairs=250,
                       grid_size=500, grid_range_factor=100.0)
        """
        return (
            f"Calibrator("
            f"method={type(self.method).__name__}, "
            f"n_val_pairs={len(self.val_dataset)}, "
            f"grid_size={self.grid_size}, "
            f"grid_range_factor={self.grid_range_factor}"
            f")"
        )
