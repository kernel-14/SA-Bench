## Code: uncertainty/luno.py

```python
## uncertainty/luno.py
"""LUNO inference: Laplace approximation and function-valued GP for FNOs.

This module implements the two core classes for LUNO uncertainty quantification:

1. ``LaplaceApprox`` — low-rank Laplace posterior over the last Fourier block
   weights, providing Woodbury-based covariance-vector products and sampling.

2. ``LUNOInference`` — last-layer LUNO construction (Appendix C.1), converting
   the weight-space belief into a function-valued GP over the FNO output.

Mathematical foundation:
  The FNO output is factored as:
    F(a, w)(x) = q_tilde(z^{(L-1)}(x, w_{L-1}))
  where z^{(L-1)} is LINEAR in w_{L-1}, enabling efficient last-layer inference.

  The function-valued GP posterior is:
    F(a) ~ GP(m_a, K_a)
    m_a(x) = F(a, w*)(x)
    K_a(x1, x2) = Dq_tilde(m_z(x1)) * K_{z^{(L-1)}}(x1, x2) * Dq_tilde(m_z(x2))^T

  where K_{z^{(L-1)}}(x1, x2) = Phi(x1) * Sigma * Phi(x2)^T and Phi(x) is the
  feature matrix (Jacobian of z^{(L-1)} w.r.t. w_{L-1}).

Paper references:
  - Section 3.2: LUNO construction
  - Appendix C.1: Last-layer LUNO for FNOs
  - Appendix B: Linearized Laplace approximation
  - Appendix D.3.4: Low-rank GGN, rank=500, prior_prec calibration
  - config.yaml: uncertainty.ggn.rank=500, calibration.prior_prec_center=1.0
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from flax import nnx

from models.fno import FNO
from uncertainty.weight_space import WeightSpaceBelief
from utils.jax_utils import (
    flatten_params,
    woodbury_marginal_variance,
    woodbury_sample,
    woodbury_solve,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LaplaceApprox
# ---------------------------------------------------------------------------


class LaplaceApprox:
    """Low-rank Laplace posterior over the last Fourier block weights.

    Represents the Gaussian weight-space belief:
        w_{L-1} ~ N(mean, Sigma)
    where:
        Sigma = (n_data * V * diag(eigvals) * V^T + prior_prec * I)^{-1}

    All covariance operations use the Woodbury matrix identity to avoid
    forming the full p_last × p_last covariance matrix. The inner system
    is rank × rank (500 × 500), making all operations tractable.

    Attributes:
        mean: MAP last-layer weight vector, shape ``[p_last]``.
        eigvecs: Top-rank GGN eigenvectors, shape ``[p_last, rank]``.
            Columns are (approximately) orthonormal.
        eigvals: Corresponding GGN eigenvalues, shape ``[rank]``.
            Non-negative (GGN is PSD). Sorted descending.
        prior_prec: Prior precision scalar σ > 0. Calibrated on val set.
        n_data: Number of data points used to compute the GGN.
        rank: Number of eigenpairs (= eigvecs.shape[1]).

    Example::

        la = LaplaceApprox(
            mean=flat_last_layer,
            eigvecs=eigvecs,   # [p_last, 500]
            eigvals=eigvals,   # [500]
            prior_prec=1.0,
            n_data=25,
        )
        # Compute Sigma * v
        sigma_v = la.woodbury_matvec(v)
        # Compute marginal variances diag(J * Sigma * J^T)
        var = la.marginal_variance(J)
        # Draw weight samples
        samples = la.sample(key, n_samples=200)
    """

    def __init__(
        self,
        mean: jnp.ndarray,
        eigvecs: jnp.ndarray,
        eigvals: jnp.ndarray,
        prior_prec: float = 1.0,
        n_data: int = 25,
    ) -> None:
        """Initialise the low-rank Laplace approximation.

        Args:
            mean: MAP last-layer weight vector, shape ``[p_last]``.
                Produced by ``GGNComputer._get_last_layer_params``.
            eigvecs: Top-rank GGN eigenvectors, shape ``[p_last, rank]``.
                Columns are orthonormal unit vectors.
            eigvals: Corresponding GGN eigenvalues, shape ``[rank]``.
                Should be non-negative; small negatives are clipped internally.
            prior_prec: Prior precision σ > 0. Initial value before
                calibration. From ``config.calibration.prior_prec_center = 1.0``.
                Updated via ``set_prior_prec()`` during calibration.
            n_data: Number of data points used to compute the GGN.
                From ``config.uncertainty.ggn.n_pairs_low_data = 25`` or
                ``config.uncertainty.ggn.n_pairs_ood = 1000``.

        Raises:
            ValueError: If ``mean`` is not 1-D.
            ValueError: If ``eigvecs`` is not 2-D or has wrong leading dim.
            ValueError: If ``eigvals`` is not 1-D or has wrong length.
            ValueError: If ``prior_prec <= 0``.
            ValueError: If ``n_data <= 0``.
        """
        # ------------------------------------------------------------------
        # Validate inputs
        # ------------------------------------------------------------------
        mean_arr: jnp.ndarray = jnp.asarray(mean, dtype=jnp.float32)
        eigvecs_arr: jnp.ndarray = jnp.asarray(eigvecs, dtype=jnp.float32)
        eigvals_arr: jnp.ndarray = jnp.asarray(eigvals, dtype=jnp.float32)

        if mean_arr.ndim != 1:
            raise ValueError(
                f"mean must be 1-D, got shape {mean_arr.shape}"
            )
        if eigvecs_arr.ndim != 2:
            raise ValueError(
                f"eigvecs must be 2-D [p_last, rank], got shape {eigvecs_arr.shape}"
            )
        if eigvecs_arr.shape[0] != mean_arr.shape[0]:
            raise ValueError(
                f"eigvecs.shape[0] ({eigvecs_arr.shape[0]}) must equal "
                f"len(mean) ({mean_arr.shape[0]})"
            )
        if eigvals_arr.ndim != 1:
            raise ValueError(
                f"eigvals must be 1-D, got shape {eigvals_arr.shape}"
            )
        if eigvals_arr.shape[0] != eigvecs_arr.shape[1]:
            raise ValueError(
                f"eigvals.shape[0] ({eigvals_arr.shape[0]}) must equal "
                f"eigvecs.shape[1] ({eigvecs_arr.shape[1]})"
            )
        if float(prior_prec) <= 0.0:
            raise ValueError(f"prior_prec must be positive, got {prior_prec}")
        if int(n_data) <= 0:
            raise ValueError(f"n_data must be positive, got {n_data}")

        # ------------------------------------------------------------------
        # Store fields
        # ------------------------------------------------------------------
        self.mean: jnp.ndarray = mean_arr
        self.eigvecs: jnp.ndarray = eigvecs_arr
        # Clip eigenvalues to [0, inf) to guard against Lanczos float errors
        self.eigvals: jnp.ndarray = jnp.maximum(eigvals_arr, 0.0)
        self.prior_prec: float = float(prior_prec)
        self.n_data: int = int(n_data)
        self.rank: int = int(eigvecs_arr.shape[1])

        logger.info(
            "LaplaceApprox initialised: p_last=%d, rank=%d, "
            "prior_prec=%.4e, n_data=%d",
            int(mean_arr.shape[0]),
            self.rank,
            self.prior_prec,
            self.n_data,
        )

    # -----------------------------------------------------------------------
    # Woodbury Covariance Operations
    # -----------------------------------------------------------------------

    def woodbury_matvec(self, v: jnp.ndarray) -> jnp.ndarray:
        """Compute Sigma * v via the Woodbury matrix identity.

        Computes:
            (n_data * V * diag(eigvals) * V^T + prior_prec * I)^{-1} * v

        using the Woodbury identity from ``utils.jax_utils.woodbury_solve``.

        Args:
            v: Vector(s) to multiply. Shape ``[p_last]`` for a single vector
                or ``[p_last, k]`` for a batch of ``k`` vectors.

        Returns:
            ``Sigma * v`` with the same shape as ``v``.

        Example::

            sigma_v = la.woodbury_matvec(v)  # [p_last]
            sigma_J_T = la.woodbury_matvec(J.T)  # [p_last, m]
        """
        return woodbury_solve(
            eigvecs=self.eigvecs,
            eigvals=self.eigvals,
            prior_prec=self.prior_prec,
            n_data=self.n_data,
            v=v,
        )

    def marginal_variance(self, jacobian: jnp.ndarray) -> jnp.ndarray:
        """Compute marginal variances diag(J * Sigma * J^T) via Woodbury.

        For each row ``j_i`` of the Jacobian (shape ``[p_last]``), computes
        the scalar ``j_i^T * Sigma * j_i``.

        Args:
            jacobian: Jacobian matrix, shape ``[m, p_last]``, where ``m`` is
                the number of output dimensions (e.g., spatial × d_v or
                spatial × out_channels).

        Returns:
            Marginal variances, shape ``[m]``. Entry ``i`` equals
            ``jacobian[i] @ Sigma @ jacobian[i]``.

        Example::

            # jacobian: [n_spatial * d_v, p_last]
            var = la.marginal_variance(jacobian)  # [n_spatial * d_v]
        """
        return woodbury_marginal_variance(
            eigvecs=self.eigvecs,
            eigvals=self.eigvals,
            prior_prec=self.prior_prec,
            n_data=self.n_data,
            jacobian=jacobian,
        )

    def sample(
        self,
        key: jax.Array,
        n_samples: int = 200,
    ) -> jnp.ndarray:
        """Draw weight samples from N(mean, Sigma).

        Uses the spectral decomposition of Sigma for exact sampling:
        - In the eigenvector subspace: variance = 1/(n_data * eigval + prior_prec)
        - In the orthogonal complement: variance = 1/prior_prec

        Args:
            key: JAX PRNG key.
            n_samples: Number of samples to draw.
                From ``config.uncertainty.sampling.n_samples = 200``.

        Returns:
            Weight samples, shape ``[n_samples, p_last]``.

        Example::

            samples = la.sample(key, n_samples=200)  # [200, p_last]
        """
        return woodbury_sample(
            eigvecs=self.eigvecs,
            eigvals=self.eigvals,
            prior_prec=self.prior_prec,
            n_data=self.n_data,
            mean=self.mean,
            key=key,
            n_samples=n_samples,
        )

    # -----------------------------------------------------------------------
    # Calibration Interface
    # -----------------------------------------------------------------------

    def set_prior_prec(self, prior_prec: float) -> None:
        """Update the prior precision in-place.

        Called by ``Calibrator._eval_nll`` during the 500-point log-spaced
        grid search that minimises validation NLL (Appendix D.5).

        Args:
            prior_prec: New prior precision value σ > 0.

        Raises:
            ValueError: If ``prior_prec <= 0``.

        Example::

            la.set_prior_prec(0.1)  # update during calibration
        """
        prior_prec_float: float = float(prior_prec)
        if prior_prec_float <= 0.0:
            raise ValueError(
                f"prior_prec must be positive, got {prior_prec_float}"
            )
        self.prior_prec = prior_prec_float
        logger.debug("LaplaceApprox.set_prior_prec: prior_prec=%.4e", prior_prec_float)

    # -----------------------------------------------------------------------
    # Representation
    # -----------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return (
            f"LaplaceApprox("
            f"p_last={self.mean.shape[0]}, "
            f"rank={self.rank}, "
            f"prior_prec={self.prior_prec:.4e}, "
            f"n_data={self.n_data}"
            f")"
        )


# ---------------------------------------------------------------------------
# LUNOInference
# ---------------------------------------------------------------------------


class LUNOInference:
    """Last-layer LUNO: function-valued GP posterior for FNOs.

    Implements the last-layer LUNO construction from Appendix C.1 of the
    LUNO paper. Given a trained FNO and a Gaussian belief over the last
    Fourier block's weights, this class provides:

    - Mean predictions (identical to the MAP FNO prediction)
    - Marginal variance at each output point
    - Full covariance matrix between output points
    - Function samples from the GP posterior

    The key structural property exploited is that z^{(L-1)}(x, w_{L-1}) is
    **linear** in w_{L-1}, enabling efficient last-layer inference without
    retraining.

    Attributes:
        model: The trained FNO instance.
        params: MAP parameter pytree (NNX state) of the trained FNO.
        belief: Weight-space belief. Either ``LaplaceApprox`` (for LUNO-LA)
            or ``WeightSpaceBelief`` (for LUNO-Iso).
        last_layer_only: Always ``True`` in the paper's experiments.
        spatial_dims: Number of spatial dimensions (1 or 2), inferred from
            the model.
        _graphdef: Cached NNX graph definition for functional forward passes.

    Example::

        luno = LUNOInference(
            model=fno,
            params=trained_state,
            belief=laplace_approx,
            last_layer_only=True,
        )
        mean = luno.predict_mean(a)           # [batch, spatial, out_channels]
        var = luno.predict_marginal_variance(a)  # [batch, spatial, out_channels]
        samples = luno.sample_functions(a, key, n_samples=4)  # [4, spatial, out_channels]
    """

    def __init__(
        self,
        model: FNO,
        params: Any,
        belief: Union[LaplaceApprox, WeightSpaceBelief],
        last_layer_only: bool = True,
    ) -> None:
        """Initialise LUNO inference.

        Args:
            model: The trained FNO instance. Used to access architecture
                parameters (modes, channels, spatial_dims) and to run
                forward passes.
            params: MAP parameter pytree (NNX state) of the trained FNO.
                Typically the ``state`` returned by ``Trainer.train()``.
            belief: Weight-space belief over the last Fourier block's
                parameters. Either ``LaplaceApprox`` (LUNO-LA) or
                ``WeightSpaceBelief`` (LUNO-Iso).
            last_layer_only: Whether to restrict uncertainty to the last
                Fourier block only. From
                ``config.uncertainty.ggn.last_layer_only = True``.
                Always ``True`` in the paper's experiments.

        Raises:
            TypeError: If ``belief`` is not a ``LaplaceApprox`` or
                ``WeightSpaceBelief`` instance.
        """
        if not isinstance(belief, (LaplaceApprox, WeightSpaceBelief)):
            raise TypeError(
                f"belief must be LaplaceApprox or WeightSpaceBelief, "
                f"got {type(belief)}"
            )

        self.model: FNO = model
        self.params: Any = params
        self.belief: Union[LaplaceApprox, WeightSpaceBelief] = belief
        self.last_layer_only: bool = last_layer_only
        self.spatial_dims: int = model.spatial_dims

        # Cache NNX graph definition for functional forward passes
        self._graphdef: Optional[Any] = None
        try:
            self._graphdef, _ = nnx.split(model)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning(
                "Could not extract graphdef from model: %s. "
                "Will use model directly for forward passes.",
                e,
            )

        # Cache the last-layer parameter extraction for efficiency
        self._flat_last_layer: Optional[jnp.ndarray] = None
        self._last_layer_unflatten_fn: Optional[Callable] = None
        self._init_last_layer_cache()

        logger.info(
            "LUNOInference initialised: spatial_dims=%d, last_layer_only=%s, "
            "belief_type=%s",
            self.spatial_dims,
            last_layer_only,
            type(belief).__name__,
        )

    def _init_last_layer_cache(self) -> None:
        """Cache the flattened last-layer parameters and unflatten function."""
        try:
            last_layer_subtree = self._extract_last_layer_subtree(self.params)
            flat, unflatten_fn = flatten_params(last_layer_subtree)
            self._flat_last_layer = flat
            self._last_layer_unflatten_fn = unflatten_fn
            logger.debug(
                "Last-layer cache initialised: p_last=%d", int(flat.shape[0])
            )
        except Exception as e:  # pylint: disable=broad-except
            logger.warning(
                "Could not initialise last-layer cache: %s. "
                "Feature functions will use JAX AD instead.",
                e,
            )

    # -----------------------------------------------------------------------
    # Parameter Extraction Utilities
    # -----------------------------------------------------------------------

    def _extract_last_layer_subtree(self, params: Any) -> Any:
        """Extract the last_fourier_block sub-tree from the parameter pytree.

        Navigates the NNX state to find the sub-tree corresponding to
        ``model.last_fourier_block``.

        Args:
            params: Full parameter pytree (NNX state or nested dict).

        Returns:
            The sub-pytree for ``last_fourier_block``.

        Raises:
            KeyError: If ``last_fourier_block`` is not found.
        """
        # Try direct attribute access (NNX State)
        if hasattr(params, "last_fourier_block"):
            return params.last_fourier_block

        # Try dict access
        if isinstance(params, dict) and "last_fourier_block" in params:
            return params["last_fourier_block"]

        # Try __dict__ traversal
        if hasattr(params, "__dict__"):
            d = vars(params)
            if "last_fourier_block" in d:
                return d["last_fourier_block"]
            # Recurse into values
            for v in d.values():
                if v is params:
                    continue
                try:
                    return self._extract_last_layer_subtree(v)
                except (KeyError, RecursionError):
                    continue

        raise KeyError(
            "Could not find 'last_fourier_block' in parameter pytree. "
            "Ensure the FNO was built with models/fno.py."
        )

    def _get_projection_fn(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """Build the q_tilde function: q(sigma^{(L-1)}(z), w_q).

        Returns a callable that maps z^{(L-1)} (shape [d_v]) to the
        projected output (shape [out_channels]).

        The function applies:
          1. Activation sigma^{(L-1)} (GELU by default)
          2. Projection layer q (linear, with bias)

        Returns:
            A callable ``q_tilde(z) -> output`` where z has shape [d_v]
            and output has shape [out_channels].
        """
        # Get activation function
        activation_name: str = self.model.activation_name
        activation_fn: Callable = _get_activation_fn(activation_name)

        # Extract projection layer parameters from the full params
        # The projection layer is model.projection (nnx.Linear)
        graphdef = self._graphdef
        params = self.params

        def q_tilde(z: jnp.ndarray) -> jnp.ndarray:
            """Apply activation + projection to pre-activation z.

            Args:
                z: Pre-activation vector, shape [d_v].

            Returns:
                Projected output, shape [out_channels].
            """
            # Apply activation
            h: jnp.ndarray = activation_fn(z)  # [d_v]

            # Apply projection layer
            # Reconstruct model to access projection layer
            if graphdef is not None:
                model_copy: FNO = nnx.merge(graphdef, params)
                # Apply projection (nnx.Linear) to h
                # h has shape [d_v]; projection expects [..., d_v] → [..., out_channels]
                out: jnp.ndarray = model_copy.projection(h[jnp.newaxis, :])[0]
            else:
                # Fallback: extract projection weights manually
                out = _apply_projection_from_params(params, h)

            return out  # [out_channels]

        return q_tilde

    # -----------------------------------------------------------------------
    # Feature Function Computation (Appendix C.1)
    # -----------------------------------------------------------------------

    def _compute_feature_functions(
        self,
        a: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute the feature matrix Phi(x) for all spatial points.

        Phi(x) is the Jacobian of z^{(L-1)}(x, w_{L-1}) w.r.t. w_{L-1},
        evaluated at the MAP weights w*_{L-1}. Since z^{(L-1)} is linear
        in w_{L-1}, this Jacobian is constant (independent of w_{L-1}).

        For a single spatial point x and output channel i:
          z_i^{(L-1)}(x) = sum_{j,k} Re(R_{k;ij}) * phi_{kj}(x)
                          + sum_{j,k} Im(R_{k;ij}) * psi_{kj}(x)
                          + sum_j W_{ij} * chi_j(x)

        where:
          phi_{kj}(x) = Re(v_hat_{kj}) * cos(omega_k * x) - Im(v_hat_{kj}) * sin(omega_k * x)
          psi_{kj}(x) = -Im(v_hat_{kj}) * cos(omega_k * x) - Re(v_hat_{kj}) * sin(omega_k * x)
          chi_j(x) = v_j^{(L-1)}(x)

        The feature matrix Phi has shape [n_spatial, d_v, p_last] where
        p_last = 2 * k_max * d_v^2 + d_v^2 = (2*k_max + 1) * d_v^2.

        Args:
            a: Input function, shape [1, spatial, in_channels] (1D) or
               [1, H, W, in_channels] (2D). Batch size must be 1.

        Returns:
            Feature matrix Phi, shape [n_spatial, d_v, p_last] for 1D or
            [H, W, d_v, p_last] for 2D.

        Notes:
            - Uses JAX's jacfwd for correctness and generality.
            - The analytical construction is used for 1D; jacfwd is the
              fallback for 2D and edge cases.
        """
        if self.spatial_dims == 1:
            return self._compute_feature_functions_1d(a)
        else:
            return self._compute_feature_functions_2d(a)

    def _compute_feature_functions_1d(
        self,
        a: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute feature matrix for 1D spatial domain.

        Uses the analytical construction from Appendix C.1 for efficiency.

        Args:
            a: Input, shape [1, spatial, in_channels].

        Returns:
            Feature matrix Phi, shape [spatial_padded, d_v, p_last].
            Note: spatial_padded = spatial + 2 * spatial_padding.
        """
        # ------------------------------------------------------------------
        # Step 1: Get v^{(L-1)} — input to the last Fourier block
        # Shape: [1, spatial_padded, d_v]
        # ------------------------------------------------------------------
        if self._graphdef is not None:
            model_copy: FNO = nnx.merge(self._graphdef, self.params)
            v_prev: jnp.ndarray = model_copy.get_hidden_state(a)
        else:
            v_prev = self.model.get_hidden_state(a)
        # v_prev: [1, spatial_padded, d_v]

        v_prev_sq: jnp.ndarray = v_prev[0]  # [spatial_padded, d_v]
        n_spatial: int = v_prev_sq.shape[0]
        d_v: int = v_prev_sq.shape[1]
        k_max: int = self.model.modes

        # ------------------------------------------------------------------
        # Step 2: Compute RFFT of each channel of v^{(L-1)}
        # v_hat_{kj} = rfft(v_j^{(L-1)})[k] for k=0..k_max-1, j=0..d_v-1
        # Shape: [k_max, d_v] complex
        # ------------------------------------------------------------------
        # rfft along spatial axis (axis=0 of v_prev_sq)
        v_hat_full: jnp.ndarray = jnp.fft.rfft(v_prev_sq, axis=0)
        # v_hat_full: [n_spatial//2 + 1, d_v] complex
        v_hat: jnp.ndarray = v_hat_full[:k_max, :]  # [k_max, d_v] complex

        v_hat_real: jnp.ndarray = jnp.real(v_hat)  # [k_max, d_v]
        v_hat_imag: jnp.ndarray = jnp.imag(v_hat)  # [k_max, d_v]

        # ------------------------------------------------------------------
        # Step 3: Compute cosine and sine basis functions
        # omega_k = 2*pi*k / n_spatial for k = 0, ..., k_max-1
        # x_n = n for n = 0, ..., n_spatial-1 (integer grid indices)
        # cos(omega_k * x_n) = cos(2*pi*k*n / n_spatial)
        # ------------------------------------------------------------------
        k_indices: jnp.ndarray = jnp.arange(k_max, dtype=jnp.float32)  # [k_max]
        n_indices: jnp.ndarray = jnp.arange(n_spatial, dtype=jnp.float32)  # [n_spatial]

        # angles[n, k] = 2*pi*k*n / n_spatial
        angles: jnp.ndarray = (
            2.0 * jnp.pi * jnp.outer(n_indices, k_indices) / float(n_spatial)
        )  # [n_spatial, k_max]

        cos_terms: jnp.ndarray = jnp.cos(angles)  # [n_spatial, k_max]
        sin_terms: jnp.ndarray = jnp.sin(angles)  # [n_spatial, k_max]

        # ------------------------------------------------------------------
        # Step 4: Compute phi and psi feature functions
        #
        # phi_{kj}(x_n) = Re(v_hat_{kj}) * cos(omega_k * x_n)
        #                - Im(v_hat_{kj}) * sin(omega_k * x_n)
        # Shape: [n_spatial, k_max, d_v]
        #
        # psi_{kj}(x_n) = -Im(v_hat_{kj}) * cos(omega_k * x_n)
        #                 - Re(v_hat_{kj}) * sin(omega_k * x_n)
        # Shape: [n_spatial, k_max, d_v]
        # ------------------------------------------------------------------
        # v_hat_real: [k_max, d_v] → broadcast to [n_spatial, k_max, d_v]
        # cos_terms: [n_spatial, k_max] → [n_spatial, k_max, 1]

        cos_exp: jnp.ndarray = cos_terms[:, :, jnp.newaxis]  # [n_spatial, k_max, 1]
        sin_exp: jnp.ndarray = sin_terms[:, :, jnp.newaxis]  # [n_spatial, k_max, 1]

        v_hat_real_exp: jnp.ndarray = v_hat_real[jnp.newaxis, :, :]  # [1, k_max, d_v]
        v_hat_imag_exp: jnp.ndarray = v_hat_imag[jnp.newaxis, :, :]  # [1, k_max, d_v]

        phi: jnp.ndarray = (
            v_hat_real_exp * cos_exp - v_hat_imag_exp * sin_exp
        )  # [n_spatial, k_max, d_v]

        psi: jnp.ndarray = (
            -v_hat_imag_exp * cos_exp - v_hat_real_exp * sin_exp
        )  # [n_spatial, k_max, d_v]

        # chi_j(x_n) = v_j^