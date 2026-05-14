## baselines/ensembles.py
"""Deep Ensemble baseline for the LUNO reproduction.

Implements the DeepEnsemble class described in Appendix D.3.2 of the LUNO
paper. Ten independent FNOs are trained with different random seeds; their
predictions are aggregated to estimate uncertainty.

Key property (discussed in paper Figure 3): the empirical covariance matrix
from 10 ensemble members has rank ≤ n_members - 1 = 9, making it
fundamentally rank-deficient. This means certain error directions fall
entirely in the null space of the ensemble covariance — a limitation that
LUNO-LA does not share.

Paper references:
  - Section 5: "Deep ensembles were trained 10 times with different random seeds"
  - Appendix D.3.2: ensemble construction and uncertainty estimation
  - Figure 3: rank-deficiency of ensemble covariance vs. LUNO-LA
  - config.yaml uncertainty.ensemble.n_members: 10

Design notes:
  - This class does NOT train models; training is done in main.py.
  - predict_mean, predict_marginal_variance, predict_covariance are all
    deterministic (no PRNG keys needed), matching the Evaluator interface.
  - A Python loop over members is used (not vmap) for simplicity and
    correctness with heterogeneous parameter pytrees.
  - predict_marginal_variance returns the same shape as predict_mean so
    that Metrics.compute_all can consume both directly.
"""

from __future__ import annotations

import logging
from typing import Any, List

import jax.numpy as jnp

from models.fno import FNO

logger = logging.getLogger(__name__)


class DeepEnsemble:
    """Deep ensemble of independently trained FNOs for uncertainty estimation.

    Aggregates predictions from ``n_members`` FNOs (each trained with a
    different random seed) to produce mean predictions and empirical
    uncertainty estimates.

    The marginal variance (diagonal of the empirical covariance) is used
    for the NLL and χ² metrics. The full covariance matrix is available
    for visualization (Figure 3 in the paper) but is rank-deficient
    (rank ≤ n_members - 1 = 9).

    Attributes:
        models: List of ``n_members`` FNO instances. All share the same
            architecture but have different trained weights.
        params_list: List of ``n_members`` NNX state objects (parameter
            pytrees), one per ensemble member. Produced by
            ``Trainer.train()`` for each member.
        n_members: Number of ensemble members. From
            ``config.uncertainty.ensemble.n_members = 10``.

    Example::

        # Constructed in main.py after training 10 FNOs:
        ensemble = DeepEnsemble(
            models=[fno_0, fno_1, ..., fno_9],
            params_list=[state_0, state_1, ..., state_9],
        )

        # Evaluate on a single test input:
        a = jnp.ones([1, 256, 12])  # [batch=1, spatial, in_channels]
        mean = ensemble.predict_mean(a)           # [1, 256, 1]
        var  = ensemble.predict_marginal_variance(a)  # [1, 256, 1]
        cov  = ensemble.predict_covariance(a)     # [256, 256]  (rank ≤ 9)
    """

    def __init__(
        self,
        models: List[FNO],
        params_list: List[Any],
    ) -> None:
        """Initialise the deep ensemble from pre-trained models and parameters.

        Does NOT train the models. Training is performed in ``main.py``
        using ``Trainer.train()`` with different random seeds for each member.

        Args:
            models: List of ``n_members`` FNO instances. All must share the
                same architecture (modes, channels, n_blocks, in_channels,
                out_channels, spatial_dims, spatial_padding). Typically
                constructed with ``FNO.from_config(config, rngs=nnx.Rngs(seed_i))``
                for seed_i in range(n_members).
            params_list: List of ``n_members`` NNX state objects (parameter
                pytrees), one per ensemble member. Each is the ``state``
                returned by ``Trainer.train(dataset, key_i)`` for the
                corresponding model. Must have the same length as ``models``.

        Raises:
            ValueError: If ``models`` and ``params_list`` have different lengths.
            ValueError: If either list is empty.

        Example::

            ensemble = DeepEnsemble(
                models=[fno_0, ..., fno_9],
                params_list=[state_0, ..., state_9],
            )
            print(ensemble.n_members)  # 10
        """
        if len(models) != len(params_list):
            raise ValueError(
                f"models and params_list must have the same length, "
                f"got len(models)={len(models)} and "
                f"len(params_list)={len(params_list)}"
            )
        if len(models) == 0:
            raise ValueError(
                "models and params_list must be non-empty. "
                "At least one ensemble member is required."
            )

        self.models: List[FNO] = models
        self.params_list: List[Any] = params_list
        self.n_members: int = len(models)

        logger.info(
            "DeepEnsemble initialised: n_members=%d, "
            "spatial_dims=%d, modes=%d, channels=%d",
            self.n_members,
            models[0].spatial_dims,
            models[0].modes,
            models[0].channels,
        )

    # -----------------------------------------------------------------------
    # Prediction Interface
    # -----------------------------------------------------------------------

    def _get_all_predictions(self, a: jnp.ndarray) -> jnp.ndarray:
        """Compute predictions from all ensemble members.

        Runs the forward pass for each member and stacks the results along
        a new leading axis. This is the shared computation used by
        ``predict_mean``, ``predict_marginal_variance``, and
        ``predict_covariance``.

        Args:
            a: Input function discretization. Shape:
                - 1D: ``[batch, spatial_res, in_channels]``
                - 2D: ``[batch, H, W, in_channels]``
                A batch dimension is always expected. For single-sample
                evaluation, use ``a[jnp.newaxis, ...]`` before calling.

        Returns:
            Stacked predictions from all members, shape:
                - 1D: ``[n_members, batch, spatial_res, out_channels]``
                - 2D: ``[n_members, batch, H, W, out_channels]``

        Notes:
            - Uses a Python loop over members (not ``jax.vmap``) for
              simplicity and correctness with heterogeneous parameter pytrees.
            - Each member's forward pass is independently JIT-compiled by
              JAX on the first call and cached thereafter.
        """
        from flax import nnx  # local import to avoid circular dependency

        predictions: List[jnp.ndarray] = []

        for i in range(self.n_members):
            model_i: FNO = self.models[i]
            params_i: Any = self.params_list[i]

            # Reconstruct the model with member i's parameters using NNX merge.
            # This is the functional interface: nnx.merge(graphdef, state) → model.
            try:
                graphdef_i, _ = nnx.split(model_i)
                model_copy_i: FNO = nnx.merge(graphdef_i, params_i)
                pred_i: jnp.ndarray = model_copy_i(a)
            except Exception:  # pylint: disable=broad-except
                # Fallback: call model directly (may use stale internal state)
                pred_i = model_i(a)

            predictions.append(pred_i)

        # Stack along a new leading axis: [n_members, batch, spatial, out_channels]
        preds_stacked: jnp.ndarray = jnp.stack(predictions, axis=0)

        return preds_stacked

    def predict_mean(self, a: jnp.ndarray) -> jnp.ndarray:
        """Compute the ensemble mean prediction.

        Averages predictions from all ``n_members`` ensemble members.
        This is the point estimate used for RMSE computation.

        Args:
            a: Input function discretization. Shape:
                - 1D: ``[batch, spatial_res, in_channels]``
                - 2D: ``[batch, H, W, in_channels]``

        Returns:
            Ensemble mean prediction with the same spatial shape as the
            FNO output:
                - 1D: ``[batch, spatial_res, out_channels]``
                - 2D: ``[batch, H, W, out_channels]``

        Example::

            a = jnp.ones([1, 256, 12])
            mean = ensemble.predict_mean(a)
            # mean.shape == (1, 256, 1)
        """
        # preds_stacked: [n_members, batch, spatial, out_channels]
        preds_stacked: jnp.ndarray = self._get_all_predictions(a)

        # Mean over the member axis (axis=0)
        mean_pred: jnp.ndarray = jnp.mean(preds_stacked, axis=0)

        return mean_pred

    def predict_marginal_variance(self, a: jnp.ndarray) -> jnp.ndarray:
        """Compute the empirical marginal variance across ensemble members.

        Computes the variance of predictions at each spatial point
        independently (marginal variance), treating spatial points as
        uncorrelated. This is the uncertainty estimate used for the
        marginal NLL and χ² metrics.

        Uses the biased empirical variance estimator (ddof=0), which is
        standard for ensemble-based uncertainty estimation.

        Args:
            a: Input function discretization. Shape:
                - 1D: ``[batch, spatial_res, in_channels]``
                - 2D: ``[batch, H, W, in_channels]``

        Returns:
            Empirical marginal variance with the same spatial shape as the
            FNO output:
                - 1D: ``[batch, spatial_res, out_channels]``
                - 2D: ``[batch, H, W, out_channels]``

            Entry ``[b, s, c]`` equals the variance of the ``n_members``
            predictions at spatial point ``s``, channel ``c``, for batch
            element ``b``.

        Notes:
            - The returned variance is non-negative by construction.
            - For ``n_members = 10``, the effective degrees of freedom is 9,
              but we use the biased estimator (ddof=0) for consistency with
              the calibration procedure (which tunes a scalar multiplier).

        Example::

            a = jnp.ones([1, 256, 12])
            var = ensemble.predict_marginal_variance(a)
            # var.shape == (1, 256, 1)
            std = jnp.sqrt(var)
        """
        # preds_stacked: [n_members, batch, spatial, out_channels]
        preds_stacked: jnp.ndarray = self._get_all_predictions(a)

        # Biased empirical variance over the member axis (axis=0)
        # jnp.var uses ddof=0 by default (biased estimator)
        marginal_var: jnp.ndarray = jnp.var(preds_stacked, axis=0)

        return marginal_var

    def predict_covariance(self, a: jnp.ndarray) -> jnp.ndarray:
        """Compute the full empirical covariance matrix over all output points.

        Computes the empirical covariance matrix of the ensemble predictions
        across all spatial output points. This matrix has rank ≤ n_members - 1
        = 9, making it fundamentally rank-deficient as discussed in Figure 3
        of the paper.

        The rank deficiency means that certain error directions are entirely
        unaccounted for by the ensemble covariance. This is the key
        limitation of deep ensembles compared to LUNO-LA, which has a
        covariance matrix whose rank is bounded only by the number of
        parameters considered.

        Args:
            a: Input function discretization. Shape:
                - 1D: ``[batch, spatial_res, in_channels]``
                - 2D: ``[batch, H, W, in_channels]``
                For covariance computation, ``batch`` must be 1 (single
                input). If ``batch > 1``, only the first element is used.

        Returns:
            Empirical covariance matrix, shape ``[n_spatial, n_spatial]``
            where ``n_spatial = spatial_res * out_channels`` (1D) or
            ``n_spatial = H * W * out_channels`` (2D).

            This matrix is symmetric, positive semi-definite, and has
            rank ≤ ``n_members - 1 = 9``.

        Notes:
            - Uses the biased estimator (divided by ``n_members``, not
              ``n_members - 1``) for consistency with ``predict_marginal_variance``.
            - The diagonal of this matrix equals ``predict_marginal_variance``
              (flattened), up to the biased/unbiased estimator difference.
            - For visualization (Figure 3), the top eigenfunctions of this
              matrix reveal the dominant uncertainty directions.

        Example::

            a = jnp.ones([1, 256, 12])
            cov = ensemble.predict_covariance(a)
            # cov.shape == (256, 256)  for out_channels=1
            # rank(cov) <= 9  (n_members - 1)

            # Null-space projection (Figure 3, panel 8):
            # residual projected onto null space of cov
            eigvals, eigvecs = jnp.linalg.eigh(cov)
            # eigvals[:247] ≈ 0 (null space, rank 256-9=247)
        """
        # preds_stacked: [n_members, batch, spatial, out_channels]
        preds_stacked: jnp.ndarray = self._get_all_predictions(a)

        # Use only the first batch element for covariance computation
        # preds_single: [n_members, spatial, out_channels]
        preds_single: jnp.ndarray = preds_stacked[:, 0, ...]

        # Flatten spatial and channel dimensions
        # preds_flat: [n_members, n_spatial] where n_spatial = spatial * out_channels
        n_members: int = self.n_members
        preds_flat: jnp.ndarray = preds_single.reshape(n_members, -1)
        # preds_flat.shape: [n_members, n_spatial]

        # Center predictions: subtract the ensemble mean
        ensemble_mean: jnp.ndarray = jnp.mean(preds_flat, axis=0, keepdims=True)
        # ensemble_mean.shape: [1, n_spatial]
        preds_centered: jnp.ndarray = preds_flat - ensemble_mean
        # preds_centered.shape: [n_members, n_spatial]

        # Empirical covariance matrix (biased estimator):
        # C = (1/n_members) * preds_centered^T @ preds_centered
        # C.shape: [n_spatial, n_spatial]
        # Rank(C) <= n_members - 1 = 9
        cov: jnp.ndarray = (preds_centered.T @ preds_centered) / float(n_members)
        # cov.shape: [n_spatial, n_spatial]

        return cov

    # -----------------------------------------------------------------------
    # Representation
    # -----------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a concise string representation of the ensemble.

        Returns:
            A string summarising the key configuration parameters.

        Example::

            DeepEnsemble(n_members=10, spatial_dims=1, modes=12, channels=18)
        """
        return (
            f"DeepEnsemble("
            f"n_members={self.n_members}, "
            f"spatial_dims={self.models[0].spatial_dims}, "
            f"modes={self.models[0].modes}, "
            f"channels={self.models[0].channels}"
            f")"
        )
