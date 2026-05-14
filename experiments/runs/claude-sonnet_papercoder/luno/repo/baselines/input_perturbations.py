## baselines/input_perturbations.py
"""Input perturbation baseline for the LUNO reproduction.

Implements the input perturbation uncertainty quantification method from
Pathak et al. (2022), as described in Appendix D.3.1 of the LUNO paper.

Method summary:
  For each test input u_n, generate n_samples perturbed copies:
    u_n^{(s)} = u_n + ε^{(s)},  ε^{(s)} ~ N(0, σ²)  (pointwise)
  Forward all perturbed inputs through the trained FNO.
  Estimate uncertainty from the empirical spread of predictions.

The perturbation scale σ is calibrated on the validation set by minimising
the marginal NLL (Appendix D.5), using a 500-point log-spaced grid search
centred at config.calibration.sigma_perturb_center = 0.01.

Paper references:
  - Appendix D.3.1: input perturbation construction
  - Appendix D.3.5: n_samples = 200 for all sample-based methods
  - Appendix D.5: calibration via grid search over 500 log-spaced values
  - config.yaml calibration.sigma_perturb_center: 0.01
  - config.yaml uncertainty.sampling.n_samples: 200

Design notes:
  - sigma is stored as a Python float for calibration mutability (no JAX
    retracing when set_sigma() is called in the calibration loop).
  - sigma is passed as a jnp.float32 argument to the JIT-compiled inner
    function so JAX traces it as a dynamic value, avoiding recompilation
    across the 500-point calibration grid.
  - A private _get_predictions() helper avoids duplicate vmap/forward code
    between predict_mean() and predict_marginal_variance().
  - jax.vmap is used over the n_samples axis for efficient batched inference.
  - All input channels are perturbed (state + auxiliary), consistent with
    the general perturbation approach; calibration compensates for scale.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import jax
import jax.numpy as jnp
from flax import nnx

from models.fno import FNO

logger = logging.getLogger(__name__)


class InputPerturbations:
    """Input perturbation baseline for uncertainty quantification.

    Generates uncertainty estimates by forwarding multiple pointwise-perturbed
    copies of each test input through the trained FNO and computing the
    empirical spread of predictions.

    Following Pathak et al. (2022) and Appendix D.3.1 of the LUNO paper:
      - Perturbations ε_{x,t} ~ N(0, σ²) are sampled independently for
        each spatial location and channel of the input.
      - n_samples = 200 perturbed copies are generated per test input.
      - σ is calibrated on the validation set to minimise marginal NLL.

    Attributes:
        model: The trained FNO instance. Used for forward passes.
        params: MAP parameter pytree (NNX state) of the trained FNO.
            Frozen — this method does not modify weights.
        sigma: Perturbation standard deviation σ > 0. Python float for
            calibration mutability. Initial value from
            ``config.calibration.sigma_perturb_center = 0.01``.
            Updated via ``set_sigma()`` during calibration.
        n_samples: Number of perturbed copies per test input.
            From ``config.uncertainty.sampling.n_samples = 200``.

    Example::

        ip = InputPerturbations(
            model=fno,
            params=trained_state,
            sigma=0.01,       # initial; calibrated later
            n_samples=200,
        )

        key = jax.random.PRNGKey(0)
        a = jnp.ones([256, 12])  # single 1D test input (no batch dim)

        mean = ip.predict_mean(a, key)           # [256, 1]
        var  = ip.predict_marginal_variance(a, key)  # [256, 1]

        # Calibration:
        ip.set_sigma(0.005)
    """

    def __init__(
        self,
        model: FNO,
        params: Any,
        sigma: float = 0.01,
        n_samples: int = 200,
    ) -> None:
        """Initialise the input perturbation method.

        Args:
            model: The trained FNO instance. Must be a Flax NNX module
                with parameters already initialised. Used to run forward
                passes on perturbed inputs.
            params: MAP parameter pytree (NNX state) of the trained FNO.
                Typically the ``state`` returned by ``Trainer.train()``.
                This is the fixed weight configuration used for all
                forward passes; it is never modified by this class.
            sigma: Initial perturbation standard deviation σ > 0.
                From ``config.calibration.sigma_perturb_center = 0.01``.
                Updated via ``set_sigma()`` during the calibration grid
                search. Stored as a Python ``float`` for mutability.
                Default: ``0.01``.
            n_samples: Number of perturbed copies to generate per test
                input. From ``config.uncertainty.sampling.n_samples = 200``.
                Default: ``200``.

        Raises:
            ValueError: If ``sigma <= 0``.
            ValueError: If ``n_samples <= 0``.

        Example::

            ip = InputPerturbations(
                model=fno,
                params=trained_state,
                sigma=0.01,
                n_samples=200,
            )
        """
        if float(sigma) <= 0.0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        if int(n_samples) <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")

        self.model: FNO = model
        self.params: Any = params
        self.sigma: float = float(sigma)
        self.n_samples: int = int(n_samples)

        # ------------------------------------------------------------------
        # Cache the NNX graph definition for functional forward passes.
        # nnx.split(model) → (graphdef, state); graphdef is the static
        # structure needed for nnx.merge(graphdef, params) → model_copy.
        # ------------------------------------------------------------------
        self._graphdef: Optional[Any] = None
        try:
            graphdef, _ = nnx.split(model)
            self._graphdef = graphdef
        except Exception as e:  # pylint: disable=broad-except
            logger.warning(
                "Could not extract graphdef from model: %s. "
                "Will call model directly for forward passes.",
                e,
            )

        logger.info(
            "InputPerturbations initialised: sigma=%.4e, n_samples=%d, "
            "spatial_dims=%d",
            self.sigma,
            self.n_samples,
            model.spatial_dims,
        )

    # -----------------------------------------------------------------------
    # Private Helper: Batched Forward Pass
    # -----------------------------------------------------------------------

    def _get_predictions(
        self,
        a: jnp.ndarray,
        key: jax.Array,
    ) -> jnp.ndarray:
        """Generate n_samples perturbed predictions for a single input.

        Core computation shared by ``predict_mean`` and
        ``predict_marginal_variance``. Generates ``n_samples`` pointwise-
        perturbed copies of ``a``, forwards each through the FNO, and
        returns the stacked predictions.

        Algorithm:
          1. Sample ε ~ N(0, σ²) of shape [n_samples, *a.shape].
          2. Construct a_perturbed = a[None] + ε, shape [n_samples, *a.shape].
          3. Apply vmapped FNO forward: predictions shape [n_samples, *out_shape].

        The FNO forward pass is applied via ``jax.vmap`` over the sample
        axis for efficient batched inference. ``sigma`` is passed as a
        dynamic JAX argument to avoid JIT recompilation during calibration.

        Args:
            a: Single test input (no batch dimension). Shape:
                - 1D: ``[spatial_res, in_channels]`` (e.g., ``[256, 12]``)
                - 2D: ``[H, W, in_channels]`` (e.g., ``[100, 100, 13]``)
                Note: spatial padding is handled inside the FNO's
                ``__call__`` method, so ``a`` should NOT be pre-padded.
            key: JAX PRNG key for sampling perturbations. Consumed
                entirely by this call; the caller should split the key
                before passing it if the key will be reused.

        Returns:
            Stacked predictions from all perturbed inputs, shape:
                - 1D: ``[n_samples, spatial_res, out_channels]``
                - 2D: ``[n_samples, H, W, out_channels]``

        Notes:
            - ``sigma`` is converted to ``jnp.float32`` before use so that
              JAX traces it as a dynamic value. This avoids JIT recompilation
              when ``set_sigma()`` is called during calibration.
            - The vmapped forward function is JIT-compiled by JAX on the
              first call and cached for subsequent calls with the same
              input shapes.
        """
        # ------------------------------------------------------------------
        # Step 1: Sample perturbations ε ~ N(0, σ²)
        # Shape: [n_samples, *a.shape]
        # sigma is converted to jnp.float32 so JAX traces it dynamically,
        # avoiding recompilation when set_sigma() changes self.sigma.
        # ------------------------------------------------------------------
        sigma_jax: jnp.ndarray = jnp.asarray(self.sigma, dtype=jnp.float32)

        # Split key to get a fresh subkey for the normal sample
        key_eps: jax.Array
        key_eps, _ = jax.random.split(key)

        # epsilon shape: [n_samples, *a.shape]
        epsilon: jnp.ndarray = (
            jax.random.normal(key_eps, shape=(self.n_samples,) + a.shape)
            * sigma_jax
        )  # [n_samples, *a.shape]

        # ------------------------------------------------------------------
        # Step 2: Construct perturbed inputs
        # a[None] broadcasts [*a.shape] → [1, *a.shape] for addition
        # a_perturbed shape: [n_samples, *a.shape]
        # ------------------------------------------------------------------
        a_perturbed: jnp.ndarray = a[jnp.newaxis, ...] + epsilon
        # a_perturbed.shape: [n_samples, *a.shape]

        # ------------------------------------------------------------------
        # Step 3: Vmapped FNO forward pass
        # The FNO expects inputs with a batch dimension; vmap adds it.
        # Each a_i has shape [*a.shape]; the FNO adds a batch dim internally
        # via the vmap transformation.
        #
        # We need to add a batch dimension for the FNO: [1, *a.shape]
        # vmap maps over the n_samples axis, so each call receives [*a.shape]
        # and we add the batch dim inside the vmapped function.
        # ------------------------------------------------------------------
        graphdef = self._graphdef
        params = self.params

        def forward_single(a_i: jnp.ndarray) -> jnp.ndarray:
            """Forward pass for a single (unbatched) perturbed input.

            Adds a batch dimension, runs the FNO, and removes the batch dim.

            Args:
                a_i: Single perturbed input, shape ``[*a.shape]``.

            Returns:
                FNO output for this input, shape ``[*out_shape]`` (no batch).
            """
            # Add batch dimension: [*a.shape] → [1, *a.shape]
            a_batched: jnp.ndarray = a_i[jnp.newaxis, ...]  # [1, *a.shape]

            # Run FNO forward pass
            if graphdef is not None:
                model_copy: FNO = nnx.merge(graphdef, params)
                output_batched: jnp.ndarray = model_copy(a_batched)
            else:
                # Fallback: call model directly (may use stale internal state)
                output_batched = self.model(a_batched)

            # Remove batch dimension: [1, *out_shape] → [*out_shape]
            output: jnp.ndarray = output_batched[0]
            return output

        # Apply vmap over the n_samples axis
        # Input: [n_samples, *a.shape] → Output: [n_samples, *out_shape]
        vmapped_forward = jax.vmap(forward_single)
        predictions: jnp.ndarray = vmapped_forward(a_perturbed)
        # predictions.shape: [n_samples, *out_shape]

        return predictions

    # -----------------------------------------------------------------------
    # Public Prediction Interface
    # -----------------------------------------------------------------------

    def predict_mean(
        self,
        a: jnp.ndarray,
        key: jax.Array,
    ) -> jnp.ndarray:
        """Compute the empirical mean prediction over perturbed inputs.

        Generates ``n_samples`` pointwise-perturbed copies of ``a``,
        forwards each through the FNO, and returns the empirical mean.
        This is the point estimate used for RMSE computation.

        Args:
            a: Single test input (no batch dimension). Shape:
                - 1D: ``[spatial_res, in_channels]`` (e.g., ``[256, 12]``)
                - 2D: ``[H, W, in_channels]`` (e.g., ``[100, 100, 13]``)
            key: JAX PRNG key for sampling perturbations. A fresh key
                should be provided for each test input to ensure
                independent perturbation sets.

        Returns:
            Empirical mean prediction, shape:
                - 1D: ``[spatial_res, out_channels]`` (e.g., ``[256, 1]``)
                - 2D: ``[H, W, out_channels]`` (e.g., ``[100, 100, 1]``)

        Notes:
            - The mean is computed as ``jnp.mean(predictions, axis=0)``
              over the ``n_samples`` axis.
            - For large ``n_samples`` (200), the empirical mean is a good
              approximation of the true mean under the perturbation
              distribution.

        Example::

            key = jax.random.PRNGKey(0)
            a = jnp.ones([256, 12])
            mean = ip.predict_mean(a, key)
            # mean.shape == (256, 1)
        """
        # Get all n_samples predictions: [n_samples, *out_shape]
        predictions: jnp.ndarray = self._get_predictions(a, key)

        # Empirical mean over the sample axis
        mean_pred: jnp.ndarray = jnp.mean(predictions, axis=0)
        # mean_pred.shape: [*out_shape]

        return mean_pred

    def predict_marginal_variance(
        self,
        a: jnp.ndarray,
        key: jax.Array,
    ) -> jnp.ndarray:
        """Compute the empirical marginal variance over perturbed inputs.

        Generates ``n_samples`` pointwise-perturbed copies of ``a``,
        forwards each through the FNO, and returns the empirical variance
        at each spatial point independently (marginal variance).

        This is the uncertainty estimate used for the marginal NLL and χ²
        metrics. The ``Evaluator`` takes the square root to obtain the
        pointwise standard deviation σ_i for ``Metrics.compute_nll`` and
        ``Metrics.compute_chi2``.

        Args:
            a: Single test input (no batch dimension). Shape:
                - 1D: ``[spatial_res, in_channels]`` (e.g., ``[256, 12]``)
                - 2D: ``[H, W, in_channels]`` (e.g., ``[100, 100, 13]``)
            key: JAX PRNG key for sampling perturbations. Should be the
                same key as passed to ``predict_mean`` for the same test
                input to ensure consistent perturbation sets (though the
                key is split internally, so the same key produces the same
                perturbations in both methods).

        Returns:
            Empirical marginal variance, shape:
                - 1D: ``[spatial_res, out_channels]`` (e.g., ``[256, 1]``)
                - 2D: ``[H, W, out_channels]`` (e.g., ``[100, 100, 1]``)

            Entry ``[s, c]`` equals the variance of the ``n_samples``
            predictions at spatial point ``s``, channel ``c``.

        Notes:
            - Uses the biased empirical variance estimator (``ddof=0``),
              which is ``jnp.var``'s default. For ``n_samples = 200``,
              the bias is negligible (factor of 199/200 ≈ 0.995).
            - The returned values are non-negative by construction.
            - The calibration of ``sigma`` compensates for any systematic
              over- or under-estimation of variance.

        Example::

            key = jax.random.PRNGKey(0)
            a = jnp.ones([256, 12])
            var = ip.predict_marginal_variance(a, key)
            # var.shape == (256, 1)
            std = jnp.sqrt(var)  # pointwise std for NLL computation
        """
        # Get all n_samples predictions: [n_samples, *out_shape]
        predictions: jnp.ndarray = self._get_predictions(a, key)

        # Biased empirical variance over the sample axis (ddof=0)
        marginal_var: jnp.ndarray = jnp.var(predictions, axis=0)
        # marginal_var.shape: [*out_shape]

        return marginal_var

    # -----------------------------------------------------------------------
    # Calibration Interface
    # -----------------------------------------------------------------------

    def set_sigma(self, sigma: float) -> None:
        """Update the perturbation standard deviation σ in-place.

        Called by ``Calibrator._eval_nll`` during the 500-point log-spaced
        grid search that minimises validation NLL (Appendix D.5).

        The calibration grid spans:
          ``[sigma_perturb_center / grid_range_factor,
             sigma_perturb_center * grid_range_factor]``
          = ``[0.01 / 100.0, 0.01 * 100.0]`` = ``[1e-4, 1.0]``
        with 500 log-spaced points, per ``config.calibration``.

        Args:
            sigma: New perturbation standard deviation σ > 0. Converted
                to a Python ``float`` to ensure mutability and prevent
                accidental JAX array storage.

        Raises:
            ValueError: If ``sigma <= 0``.

        Notes:
            - This is a plain Python mutation (not JAX-traced). Calibration
              is a Python-level loop, so no JIT recompilation occurs for
              the Python-level sigma value.
            - Since ``sigma`` is converted to ``jnp.float32`` inside
              ``_get_predictions`` before use, JAX traces it as a dynamic
              value and does not recompile when its value changes.
            - After calibration, the final best value is set via this method
              before the method is used for test evaluation.

        Example::

            # Calibration loop (in Calibrator):
            for sigma_candidate in log_grid:
                ip.set_sigma(sigma_candidate)
                nll = eval_nll_on_val_set(ip)
            ip.set_sigma(best_sigma)  # final calibrated value
        """
        sigma_float: float = float(sigma)
        if sigma_float <= 0.0:
            raise ValueError(
                f"sigma must be positive, got {sigma_float}"
            )
        self.sigma = sigma_float
        logger.debug("InputPerturbations.set_sigma: sigma=%.4e", sigma_float)

    # -----------------------------------------------------------------------
    # Representation
    # -----------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a concise string representation of the method.

        Returns:
            A string summarising the key configuration parameters.

        Example::

            InputPerturbations(sigma=1.0000e-02, n_samples=200,
                               spatial_dims=1)
        """
        return (
            f"InputPerturbations("
            f"sigma={self.sigma:.4e}, "
            f"n_samples={self.n_samples}, "
            f"spatial_dims={self.model.spatial_dims}"
            f")"
        )
