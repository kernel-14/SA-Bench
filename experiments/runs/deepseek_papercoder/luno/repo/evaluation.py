"""
evaluation.py
=============
Quantitative evaluation of a trained Fourier Neural Operator (FNO) equipped with
an uncertainty quantification (UQ) method.  This module provides two modes:

1. **Static metrics**  (`compute_metrics`):
   Computes the expected marginal root mean squared error (RMSE), χ²‑statistic,
   and negative log‑likelihood (NLL) on a fixed set of input–output pairs.

2. **Autoregressive rollout** (`autoregressive_rollout`):
   Simulates recursive next‑step prediction along entire trajectories, recording
   per‑step RMSE and NLL.  This is used to evaluate how predictive uncertainty
   evolves under distribution shift.

The evaluation works transparently with any ``UQMethod`` that implements
``predict(inputs) -> (mean, variance)``.  Numerical stability is ensured by
clamping the predicted variance to a small positive lower‑bound.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
import tqdm

# The following imports are safe because the evaluation module is always loaded
# after the model and UQ modules have been defined during the experiment flow.
from config import Config  # for type hints only (not used internally)
from fno import FourierNeuralOperator  # noqa: F401 (type annotations)
from uncertainty import UQMethod  # noqa: F401

logger = logging.getLogger(__name__)

# Numerical stability: minimum allowed predictive variance
_MIN_VAR = 1e-12


# ============================================================================
# Helper: Gaussian negative log‑likelihood per point (natural log)
# ============================================================================

def _gaussian_nll(
    y: jnp.ndarray, mean: jnp.ndarray, var: jnp.ndarray
) -> jnp.ndarray:
    """Point‑wise Gaussian NLL with natural logarithm."""
    var_safe = jnp.clip(var, _MIN_VAR)
    return 0.5 * (jnp.log(2 * jnp.pi * var_safe) + (y - mean) ** 2 / var_safe)


# ============================================================================
# Evaluation class
# ============================================================================

class Evaluation:
    """
    Evaluation wrapper for a trained FNO + UQ method.

    Parameters
    ----------
    model : FourierNeuralOperator
        The trained FNO model (kept for potential direct access, not used
        internally as the ``uq_method`` already provides predictions).
    uq_method : UQMethod
        An already fitted/calibrated UQ method whose ``predict`` interface
        returns ``(mean, variance)``.
    test_ds : tuple of ndarray
        ``(test_x, test_y)`` with shapes:
        - test_x : ``(N, T, H, W, C_in)`` – input windows.
        - test_y : ``(N, H, W, 1)``        – target next frame.
        The number of samples ``N`` is typically 250.
    config : Config, optional
        Not used by the evaluation itself, but can be stored for logging.
    """

    def __init__(
        self,
        model: FourierNeuralOperator,
        uq_method: UQMethod,
        test_ds: Tuple[np.ndarray, np.ndarray],
        config: Optional[Config] = None,
    ) -> None:
        self.model = model
        self.uq_method = uq_method
        self.test_x, self.test_y = test_ds
        self.config = config

        # Basic sanity checks
        if self.test_x.shape[0] != self.test_y.shape[0]:
            raise ValueError(
                f"Mismatch: test_x has {self.test_x.shape[0]} samples, "
                f"test_y has {self.test_y.shape[0]}"
            )
        self.num_test = self.test_x.shape[0]
        logger.info("Evaluation ready for %d test samples.", self.num_test)

    # ------------------------------------------------------------------
    # Static metrics
    # ------------------------------------------------------------------

    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute expected marginal RMSE, χ²‑statistic, and NLL on the test set.

        Returns
        -------
        metrics : dict
            Dictionary with keys ``"rmse"``, ``"chi2"``, ``"nll"``.
        """
        # Load entire test set into JAX arrays (small enough to fit in memory)
        test_x_jax = jnp.asarray(self.test_x)
        test_y_jax = jnp.asarray(self.test_y)

        # Get predictions for all samples at once
        mean, var = self.uq_method.predict(test_x_jax)

        # Flatten arrays: treat every spatial point independently
        y_flat = test_y_jax.ravel()
        mean_flat = mean.ravel()
        var_flat = var.ravel()

        # Numerically safe variance
        var_safe = jnp.clip(var_flat, _MIN_VAR)

        # Per‑point errors
        sq_err = (y_flat - mean_flat) ** 2
        chi2_per_point = sq_err / var_safe
        nll_per_point = _gaussian_nll(y_flat, mean_flat, var_safe)

        # Aggregate
        total_points = float(sq_err.size)
        rmse = float(jnp.sqrt(jnp.sum(sq_err) / total_points))
        chi2 = float(jnp.sum(chi2_per_point) / total_points)
        nll = float(jnp.sum(nll_per_point) / total_points)

        logger.info(
            "Test metrics -> RMSE: %.6e, χ²: %.4f, NLL: %.4f", rmse, chi2, nll
        )
        return {"rmse": rmse, "chi2": chi2, "nll": nll}

    # ------------------------------------------------------------------
    # Autoregressive rollout
    # ------------------------------------------------------------------

    def autoregressive_rollout(
        self,
        initial_frames: np.ndarray,
        steps: int,
        true_frames: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Perform autoregressive next‑step prediction and record per‑step metrics.

        The method slides the input window using the model’s own predictions.
        It requires the ground‑truth future frames for comparison.  These can
        be provided directly via ``true_frames``, or the user may set
        ``self.true_rollout_frames`` **before** calling this method (for
        compatibility with the design diagram).

        Parameters
        ----------
        initial_frames : ndarray
            Initial input windows for each trajectory.  Shape
            ``(num_traj, input_window, H, W, C_in)``.
        steps : int
            Number of future time steps to simulate.
        true_frames : ndarray, optional
            Ground‑truth scalar field for all future steps.  Shape
            ``(num_traj, steps, H, W, 1)``.  If None, the value of
            ``self.true_rollout_frames`` is used (must be set beforehand).

        Returns
        -------
        rollout_results : dict
            Contains:
            - ``"rmse_per_step"``  : ndarray of shape ``(steps,)``
            - ``"nll_per_step"``   : ndarray of shape ``(steps,)``
            - ``"chi2_per_step"``  : ndarray of shape ``(steps,)``
            - ``"avg_rmse"``       : scalar (mean over steps)
            - ``"avg_nll"``        : scalar
        """
        if true_frames is None:
            true_frames = getattr(self, "true_rollout_frames", None)
            if true_frames is None:
                raise ValueError(
                    "Ground truth frames not provided.  Set "
                    "`true_frames` argument or `self.true_rollout_frames`."
                )

        # Convert to JAX arrays
        init = jnp.asarray(initial_frames)
        truth = jnp.asarray(true_frames)

        num_traj, input_window, H, W, C_in = init.shape
        if truth.shape[0] != num_traj or truth.shape[1] != steps:
            raise ValueError(
                f"Shape mismatch: initial_frames has {num_traj} trajectories, "
                f"true_frames has {truth.shape[0]} trajectories and "
                f"{truth.shape[1]} steps, expected {steps} steps."
            )

        # Identify dynamic (scalar) channel index.  For 1D PDEs C_in=1, this is 0.
        # For 2D advection the scalar field is always the first channel.
        scalar_ch_idx = 0
        # Static channels are the remaining ones; they are assumed unchanged
        static_mask = jnp.ones(C_in, dtype=bool).at[scalar_ch_idx].set(False)
        static_channels = init[0, 0, 0, 0, static_mask]  # one frame's static values

        # Allocate accumulators for per‑step metrics
        rmse_per_step = []
        nll_per_step = []
        chi2_per_step = []

        current_inputs = init  # (N, T, H, W, C)

        # Iterate over prediction steps
        for step in tqdm.trange(steps, desc="Autoregressive rollout", unit="step"):
            # Predict next scalar field
            mean, var = self.uq_method.predict(current_inputs)  # (N, H, W, 1)
            true = truth[:, step]  # (N, H, W, 1)

            # Flatten per trajectory
            y_flat = true.ravel()
            mean_flat = mean.ravel()
            var_flat = jnp.clip(var.ravel(), _MIN_VAR)

            sq_err = (y_flat - mean_flat) ** 2
            chi2 = sq_err / var_flat
            nll = _gaussian_nll(y_flat, mean_flat, var_flat)

            # Average over all points across trajectories
            rmse_step = float(jnp.sqrt(jnp.mean(sq_err)))
            nll_step = float(jnp.mean(nll))
            chi2_step = float(jnp.mean(chi2))

            rmse_per_step.append(rmse_step)
            nll_per_step.append(nll_step)
            chi2_per_step.append(chi2_step)

            # Update input window for next step:
            # 1) Drop the oldest frame from each trajectory.
            # 2) Append a new frame that consists of:
            #    - predicted scalar field at position scalar_ch_idx,
            #    - static channels copied from the *previous* last frame
            #      (since they are invariant over time).
            # Because the input window shape is (N, T, H, W, C), we can do:
            #   current_inputs = jnp.concatenate(
            #       [current_inputs[:, 1:], new_frame], axis=1
            #   )
            # where `new_frame` has shape (N, 1, H, W, C).
            last_frame_static = current_inputs[:, -1, ..., static_mask]  # (N, H, W, C-1)
            new_frame = jnp.concatenate(
                [mean,               # (N, H, W, 1) -> scalar channel
                 last_frame_static], # (N, H, W, C-1)
                axis=-1,
            )[:, None, ...]  # add time axis of size 1 -> (N, 1, H, W, C)

            current_inputs = jnp.concatenate(
                [current_inputs[:, 1:], new_frame], axis=1
            )

        # Convert lists to arrays
        rmse_arr = np.array(rmse_per_step, dtype=np.float64)
        nll_arr = np.array(nll_per_step, dtype=np.float64)
        chi2_arr = np.array(chi2_per_step, dtype=np.float64)

        avg_rmse = float(np.mean(rmse_arr))
        avg_nll = float(np.mean(nll_arr))

        logger.info(
            "Rollout finished – avg RMSE: %.6e, avg NLL: %.4f",
            avg_rmse,
            avg_nll,
        )
        return {
            "rmse_per_step": rmse_arr,
            "nll_per_step": nll_arr,
            "chi2_per_step": chi2_arr,
            "avg_rmse": avg_rmse,
            "avg_nll": avg_nll,
        }

