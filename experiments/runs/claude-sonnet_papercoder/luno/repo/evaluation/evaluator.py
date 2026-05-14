## Code: evaluation/evaluator.py

```python
## evaluation/evaluator.py
"""Evaluator for the LUNO reproduction experiments.

Orchestrates evaluation of all UQ methods on test datasets, computes
RMSE, marginal NLL, and chi-squared metrics, runs autoregressive rollout
evaluations, and formats results for printing.

Paper references:
  - Section 5 / Appendix D.4: evaluation protocol (250 test pairs, three metrics)
  - Appendix D.4.1-D.4.3: RMSE, NLL, chi2 definitions
  - Figure 4 / Appendix D.6.3: autoregressive rollout evaluation
  - Tables 1, 2, 4, 5: result format

Design notes:
  - Handles heterogeneous method interfaces: deterministic methods
    (LUNOInference, DeepEnsemble) take no key; stochastic methods
    (SamplePushforward, InputPerturbations) require a PRNG key.
  - All metric values are returned as Python floats for serialization safety.
  - Uses tqdm for progress tracking on long evaluation loops.
  - The rollout evaluation reconstructs sliding-window inputs by dropping
    the oldest time step and appending the new prediction.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from data.dataset import PDEDataset
from evaluation.metrics import Metrics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Method type detection helpers (duck-typing to avoid circular imports)
# ---------------------------------------------------------------------------

def _method_is_stochastic(method: Any) -> bool:
    """Return True if the method's predict_* calls require a PRNG key.

    Stochastic methods (SamplePushforward, InputPerturbations) expose a
    ``sigma`` or ``n_samples`` attribute alongside a ``belief`` or
    ``set_sigma`` attribute. Deterministic methods (LUNOInference,
    DeepEnsemble) do not.

    Args:
        method: Any UQ method object.

    Returns:
        True if the method requires a key argument for predict_mean /
        predict_marginal_variance. False otherwise.
    """
    # InputPerturbations: has 'sigma' (float) and 'set_sigma' callable
    if hasattr(method, "sigma") and hasattr(method, "set_sigma"):
        return True
    # SamplePushforward: has 'n_samples' and 'belief' with a 'sample' method
    if (
        hasattr(method, "n_samples")
        and hasattr(method, "belief")
        and hasattr(getattr(method, "belief", None), "sample")
    ):
        return True
    return False


def _call_predict_mean(
    method: Any,
    a: jnp.ndarray,
    key: jax.Array,
) -> jnp.ndarray:
    """Call method.predict_mean with or without a key, as appropriate.

    Args:
        method: UQ method object.
        a: Input array (batched, shape [1, spatial, in_channels] or similar).
        key: JAX PRNG key (used only for stochastic methods).

    Returns:
        Predicted mean, same spatial shape as the FNO output.
    """
    if _method_is_stochastic(method):
        return method.predict_mean(a, key)
    return method.predict_mean(a)


def _call_predict_variance(
    method: Any,
    a: jnp.ndarray,
    key: jax.Array,
) -> jnp.ndarray:
    """Call method.predict_marginal_variance with or without a key.

    Args:
        method: UQ method object.
        a: Input array (batched, shape [1, spatial, in_channels] or similar).
        key: JAX PRNG key (used only for stochastic methods).

    Returns:
        Predicted marginal variance, same spatial shape as the FNO output.
    """
    if _method_is_stochastic(method):
        return method.predict_marginal_variance(a, key)
    return method.predict_marginal_variance(a)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """Runs all LUNO experiments and collects evaluation results.

    Provides three evaluation modes:
    1. ``evaluate_method``: evaluate a single UQ method on 250 test pairs.
    2. ``evaluate_all``: evaluate all registered methods and return a nested
       results dict.
    3. ``evaluate_rollout``: autoregressive rollout evaluation tracking how
       RMSE and NLL evolve over multiple prediction steps.

    Attributes:
        methods: Dict mapping method names to method objects. Supported
            method types: LUNOInference, SamplePushforward, DeepEnsemble,
            InputPerturbations.
        test_dataset: PDEDataset holding the 250 test input-output pairs.
        metrics: Metrics instance for computing RMSE, NLL, chi2.

    Example::

        evaluator = Evaluator(
            methods={
                'LUNO-LA': luno_la,
                'LUNO-Iso': luno_iso,
                'Sample-LA': sample_la,
                'Sample-Iso': sample_iso,
                'Ensemble': ensemble,
                'Input Perturbations': input_perturb,
            },
            test_dataset=test_ds,
        )
        key = jax.random.PRNGKey(0)
        results = evaluator.evaluate_all(key)
        evaluator.print_results(results)
    """

    def __init__(
        self,
        methods: Dict[str, Any],
        test_dataset: PDEDataset,
    ) -> None:
        """Initialise the Evaluator.

        Args:
            methods: Dict mapping string method names to method objects.
                Keys are used for display in ``print_results``. Supported
                method types: LUNOInference, SamplePushforward,
                DeepEnsemble, InputPerturbations.
            test_dataset: PDEDataset holding the test input-output pairs.
                Typically 250 pairs (config.evaluation.n_test_pairs = 250).
                Accessed via ``test_dataset.get_batch(indices)``.

        Raises:
            ValueError: If ``test_dataset`` has zero pairs.

        Example::

            evaluator = Evaluator(methods=all_methods, test_dataset=test_ds)
        """
        if len(test_dataset) == 0:
            raise ValueError("test_dataset must contain at least one pair.")

        self.methods: Dict[str, Any] = methods
        self.test_dataset: PDEDataset = test_dataset
        self.metrics: Metrics = Metrics()

        logger.info(
            "Evaluator initialised: n_methods=%d, n_test_pairs=%d",
            len(methods),
            len(test_dataset),
        )

    # -----------------------------------------------------------------------
    # Single-Method Evaluation
    # -----------------------------------------------------------------------

    def evaluate_method(
        self,
        name: str,
        method: Any,
        key: jax.Array,
    ) -> Dict[str, float]:
        """Evaluate a single UQ method on all test pairs.

        Iterates over all test pairs, computes predicted mean and marginal
        standard deviation for each, then aggregates RMSE, marginal NLL,
        and chi-squared statistics averaged over all pairs and spatial points.

        Protocol (Appendix D.4):
          - For each of the 250 test pairs:
            1. Retrieve (a_i, y_true_i) from test_dataset.
            2. Compute y_pred_i = method.predict_mean(a_i [, key_i]).
            3. Compute var_i = method.predict_marginal_variance(a_i [, key_i]).
            4. sigma_i = sqrt(clip(var_i, min=1e-8)).
            5. Compute per-pair RMSE, NLL, chi2 via Metrics.compute_all.
          - Average all per-pair metrics over the 250 pairs.

        Args:
            name: Display name of the method (used for logging only).
            method: UQ method object. Must expose predict_mean and
                predict_marginal_variance (with or without a key argument,
                detected automatically).
            key: JAX PRNG key. Split into n_test_pairs subkeys upfront so
                that stochastic methods get independent randomness per pair
                while remaining reproducible.

        Returns:
            Dict with keys 'rmse', 'nll', 'chi2', each mapping to a Python
            float. These are the expected values over the 250 test pairs.

        Notes:
            - NaN/Inf metric values from individual pairs are replaced with
              a large penalty (1e6) and a warning is logged.
            - The per-pair metrics are computed by Metrics.compute_all,
              which handles the spatial averaging internally.
            - All returned values are Python floats (not JAX arrays) for
              serialization safety.

        Example::

            results = evaluator.evaluate_method('LUNO-LA', luno_la, key)
            # results == {'rmse': 0.0362, 'nll': -2.0787, 'chi2': 1.022}
        """
        n_test: int = len(self.test_dataset)

        # Pre-split keys for all test pairs (fixed across methods for reproducibility)
        pair_keys: jnp.ndarray = jax.random.split(key, n_test)
        # pair_keys.shape: [n_test, 2]

        # Accumulators for per-pair metrics
        rmse_list: List[float] = []
        nll_list: List[float] = []
        chi2_list: List[float] = []

        logger.info("Evaluating method '%s' on %d test pairs ...", name, n_test)

        for i in tqdm(range(n_test), desc=f"Eval {name}", leave=False):
            try:
                # ----------------------------------------------------------
                # Retrieve single test pair (batched: shape [1, spatial, C])
                # ----------------------------------------------------------
                a_batch: jnp.ndarray
                y_true_batch: jnp.ndarray
                a_batch, y_true_batch = self.test_dataset.get_batch(
                    jnp.array([i])
                )
                # a_batch.shape: [1, spatial_res, in_channels] (1D)
                #             or [1, H, W, in_channels] (2D)
                # y_true_batch.shape: [1, spatial_res, out_channels] (1D)
                #                  or [1, H, W, out_channels] (2D)

                key_i: jax.Array = pair_keys[i]

                # ----------------------------------------------------------
                # Compute predicted mean and marginal variance
                # ----------------------------------------------------------
                y_pred_batch: jnp.ndarray = _call_predict_mean(
                    method, a_batch, key_i
                )
                var_batch: jnp.ndarray = _call_predict_variance(
                    method, a_batch, key_i
                )

                # ----------------------------------------------------------
                # Compute sigma: sqrt(clip(var, min=1e-8))
                # Clipping prevents NaN from zero or negative variances.
                # ----------------------------------------------------------
                var_clipped: jnp.ndarray = jnp.maximum(var_batch, 1e-8)
                sigma_batch: jnp.ndarray = jnp.sqrt(var_clipped)

                # ----------------------------------------------------------
                # Ensure shapes match: y_pred may have different shape than
                # y_true if the method returns unbatched output.
                # Reshape to match y_true_batch.
                # ----------------------------------------------------------
                y_pred_batch = y_pred_batch.reshape(y_true_batch.shape)
                sigma_batch = sigma_batch.reshape(y_true_batch.shape)

                # ----------------------------------------------------------
                # Compute per-pair metrics via Metrics.compute_all
                # Metrics expects [n_test, ...] inputs; our batch has n_test=1.
                # ----------------------------------------------------------
                pair_metrics: Dict[str, jnp.ndarray] = self.metrics.compute_all(
                    y_true=y_true_batch,
                    y_pred=y_pred_batch,
                    sigma=sigma_batch,
                )

                # Extract as Python floats, guarding against NaN/Inf
                rmse_val: float = float(pair_metrics["rmse"])
                nll_val: float = float(pair_metrics["nll"])
                chi2_val: float = float(pair_metrics["chi2"])

                if not np.isfinite(rmse_val):
                    logger.warning(
                        "Non-finite RMSE at pair %d for method '%s': %.4e",
                        i, name, rmse_val,
                    )
                    rmse_val = 1e6
                if not np.isfinite(nll_val):
                    logger.warning(
                        "Non-finite NLL at pair %d for method '%s': %.4e",
                        i, name, nll_val,
                    )
                    nll_val = 1e6
                if not np.isfinite(chi2_val):
                    logger.warning(
                        "Non-finite chi2 at pair %d for method '%s': %.4e",
                        i, name, chi2_val,
                    )
                    chi2_val = 1e6

                rmse_list.append(rmse_val)
                nll_list.append(nll_val)
                chi2_list.append(chi2_val)

            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(
                    "Exception at pair %d for method '%s': %s. "
                    "Using penalty values.",
                    i, name, exc,
                )
                rmse_list.append(1e6)
                nll_list.append(1e6)
                chi2_list.append(1e6)

        # ------------------------------------------------------------------
        # Average over all test pairs
        # ------------------------------------------------------------------
        mean_rmse: float = float(np.mean(rmse_list))
        mean_nll: float = float(np.mean(nll_list))
        mean_chi2: float = float(np.mean(chi2_list))

        logger.info(
            "Method '%s': RMSE=%.4e, NLL=%.4f, chi2=%.4f",
            name, mean_rmse, mean_nll, mean_chi2,
        )

        return {
            "rmse": mean_rmse,
            "nll": mean_nll,
            "chi2": mean_chi2,
        }

    # -----------------------------------------------------------------------
    # All-Methods Evaluation
    # -----------------------------------------------------------------------

    def evaluate_all(
        self,
        key: jax.Array,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate all registered methods and return a nested results dict.

        Calls ``evaluate_method`` for each method in ``self.methods``,
        using independent PRNG subkeys derived from ``key``.

        Args:
            key: Master JAX PRNG key. Split into one subkey per method to
                ensure independent randomness across methods.

        Returns:
            Nested dict ``{method_name: {'rmse': float, 'nll': float,
            'chi2': float}}``. Method names are the keys from
            ``self.methods``.

        Example::

            results = evaluator.evaluate_all(jax.random.PRNGKey(0))
            # results['LUNO-LA'] == {'rmse': 0.0362, 'nll': -2.0787, 'chi2': 1.022}
        """
        if not self.methods:
            logger.warning("evaluate_all called with empty methods dict.")
            return {}

        n_methods: int = len(self.methods)
        method_keys: jnp.ndarray = jax.random.split(key, n_methods)
        # method_keys.shape: [n_methods, 2]

        results: Dict[str, Dict[str, float]] = {}

        for idx, (name, method) in enumerate(
            tqdm(self.methods.items(), desc="Evaluating methods")
        ):
            method_key: jax.Array = method_keys[idx]
            results[name] = self.evaluate_method(name, method, method_key)

        return results

    # -----------------------------------------------------------------------
    # Autoregressive Rollout Evaluation
    # -----------------------------------------------------------------------

    def evaluate_rollout(
        self,
        method: Any,
        n_steps: int = 59,
        n_traj: int = 50,
        key: jax.Array = None,
    ) -> Dict[str, np.ndarray]:
        """Evaluate a UQ method on autoregressive rollouts.

        Autoregressively rolls out predictions for ``n_traj`` trajectories,
        feeding the predicted mean back as input at each step. Tracks RMSE
        and NLL at each step, averaged over trajectories.

        This reproduces Figure 4 and Figure 7 of the paper, which show how
        uncertainty estimates evolve as prediction errors accumulate during
        autoregressive rollout.

        Protocol:
          - For each of ``n_traj`` trajectories:
            1. Start from the initial 10-step input window (from test_dataset).
            2. At each step t in [0, n_steps):
               a. Predict mean and variance from the current input window.
               b. Record RMSE and NLL against the ground-truth target.
               c. Update the input window: drop oldest time step, append
                  the predicted mean as the newest step.
          - Average RMSE and NLL over all trajectories at each step.

        Input window update (1D case):
          current_input has shape [1, spatial_res, in_channels] where
          in_channels = input_steps + aux_channels (velocity + reaction).
          The first ``input_steps`` channels are the time history.
          Update: shift channels [1..input_steps-1] → [0..input_steps-2],
          place y_pred at channel [input_steps-1].
          Auxiliary channels (velocity, reaction) remain unchanged.

        Args:
            method: UQ method object to evaluate. Must expose predict_mean
                and predict_marginal_variance.
            n_steps: Number of autoregressive steps. From
                ``config.evaluation.rollout.n_steps = 59``. Default: 59.
            n_traj: Number of trajectories to roll out. From
                ``config.evaluation.rollout.n_trajectories = 50``.
                Default: 50.
            key: JAX PRNG key. If None, uses PRNGKey(0). Default: None.

        Returns:
            Dict with keys 'rmse' and 'nll', each mapping to a NumPy array
            of shape ``[n_steps]`` containing the metric averaged over
            ``n_traj`` trajectories at each rollout step.

        Notes:
            - The test_dataset must contain at least ``n_traj`` trajectories
              worth of pairs. If ``test_dataset.n_traj`` is set, it is used
              to determine the number of pairs per trajectory. Otherwise,
              ``n_steps`` pairs per trajectory are assumed.
            - If the test_dataset has fewer than ``n_traj * pairs_per_traj``
              pairs, ``n_traj`` is reduced to fit.
            - NaN/Inf values from diverging rollouts are replaced with 1e6
              and a warning is logged.

        Example::

            rollout_results = evaluator.evaluate_rollout(
                method=luno_la,
                n_steps=59,
                n_traj=50,
                key=jax.random.PRNGKey(0),
            )
            # rollout_results['rmse'].shape == (59,)
            # rollout_results['nll'].shape == (59,)
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        n_test: int = len(self.test_dataset)

        # ------------------------------------------------------------------
        # Determine pairs_per_traj from dataset metadata
        # ------------------------------------------------------------------
        if self.test_dataset.n_traj is not None and self.test_dataset.n_traj > 0:
            pairs_per_traj: int = n_test // self.test_dataset.n_traj
        else:
            # Fallback: assume n_steps pairs per trajectory
            pairs_per_traj = max(1, n_steps)

        # Clamp n_traj to available data
        max_traj: int = n_test // max(pairs_per_traj, 1)
        n_traj_actual: int = min(n_traj, max_traj)

        if n_traj_actual < n_traj:
            logger.warning(
                "evaluate_rollout: requested n_traj=%d but only %d trajectories "
                "available in test_dataset (n_test=%d, pairs_per_traj=%d). "
                "Using n_traj=%d.",
                n_traj, max_traj, n_test, pairs_per_traj, n_traj_actual,
            )

        if n_traj_actual == 0:
            logger.error(
                "evaluate_rollout: no complete trajectories available. "
                "Returning empty arrays."
            )
            return {
                "rmse": np.zeros(n_steps),
                "nll": np.zeros(n_steps),
            }

        # Clamp n_steps to available pairs per trajectory
        n_steps_actual: int = min(n_steps, pairs_per_traj)
        if n_steps_actual < n_steps:
            logger.warning(
                "evaluate_rollout: requested n_steps=%d but only %d pairs "
                "per trajectory available. Using n_steps=%d.",
                n_steps, pairs_per_traj, n_steps_actual,
            )

        logger.info(
            "evaluate_rollout: n_traj=%d, n_steps=%d, pairs_per_traj=%d",
            n_traj_actual, n_steps_actual, pairs_per_traj,
        )

        # ------------------------------------------------------------------
        # Determine input_steps from dataset shape
        # in_channels = input_steps + aux_channels
        # For 1D: aux_channels = 2 (velocity placeholder + reaction placeholder)
        # For 2D: aux_channels = 3 (vx, vy, reaction)
        # We infer input_steps from the dataset's is_2d flag.
        # ------------------------------------------------------------------
        is_2d: bool = self.test_dataset.is_2d
        aux_channels: int = 3 if is_2d else 2
        in_channels: int = self.test_dataset.in_channels
        input_steps: int = max(1, in_channels - aux_channels)

        # ------------------------------------------------------------------
        # Pre-split PRNG keys: one per (trajectory, step)
        # ------------------------------------------------------------------
        total_keys: int = n_traj_actual * n_steps_actual
        all_keys: jnp.ndarray = jax.random.split(key, total_keys)
        # all_keys.shape: [n_traj_actual * n_steps_actual, 2]

        # ------------------------------------------------------------------
        # Rollout loop
        # ------------------------------------------------------------------
        # Accumulators: [n_traj_actual, n_steps_actual]
        all_rmse: np.ndarray = np.zeros((n_traj_actual, n_steps_actual), dtype=np.float32)
        all_nll: np.ndarray = np.zeros((n_traj_actual, n_steps_actual), dtype=np.float32)

        for traj_idx in tqdm(
            range(n_traj_actual),
            desc="Rollout trajectories",
            leave=False,
        ):
            # ----------------------------------------------------------
            # Get the initial input window for this trajectory.
            # The first pair of trajectory traj_idx is at global index:
            #   global_start = traj_idx * pairs_per_traj
            # ----------------------------------------------------------
            global_start: int = traj_idx * pairs_per_traj

            # Initial input: shape [1, spatial, in_channels] (1D)
            #             or [1, H, W, in_channels] (2D)
            current_input: jnp.ndarray
            _: jnp.ndarray
            current_input, _ = self.test_dataset.get_batch(
                jnp.array([global_start])
            )
            # current_input.shape: [1, spatial, in_channels] or [1, H, W, in_channels]

            for step_idx in range(n_steps_actual):
                # ----------------------------------------------------------
                # Get ground-truth target for this step
                # ----------------------------------------------------------
                global_pair_idx: int = global_start + step_idx
                _input_unused: jnp.ndarray
                y_true_batch: jnp.ndarray
                _input_unused, y_true_batch = self.test_dataset.get_batch(
                    jnp.array([global_pair_idx])
                )
                # y_true_batch.shape: [1, spatial, out_channels] or [1, H, W, out_channels]

                # ----------------------------------------------------------
                # Get PRNG key for this (traj, step) pair
                # ----------------------------------------------------------
                key_ts: jax.Array = all_keys[traj_idx * n_steps_actual + step_idx]

                # ----------------------------------------------------------
                # Predict mean and variance
                # ----------------------------------------------------------
                try:
                    y_pred_batch: jnp.ndarray = _call_predict_mean(
                        method, current_input, key_ts
                    )
                    var_batch: jnp.ndarray = _call_predict_variance(
                        method, current_input, key_ts
                    )

                    # Reshape to match y_true_batch
                    y_pred_batch = y_pred_batch.reshape(y_true_batch.shape)
                    var_batch = var_batch.reshape(y_true_batch.shape)

                    # Compute sigma
                    var_clipped: jnp.ndarray = jnp.maximum(var_batch, 1e-8)
                    sigma_batch: jnp.ndarray = jnp.sqrt(var_clipped)

                    # Compute per-step metrics
                    step_metrics: Dict[str, jnp.ndarray] = self.metrics.compute_all(
                        y_true=y_true_batch,
                        y_pred=y_pred_batch,
                        sigma=sigma_batch,
                    )

                    step_rmse: float = float(step_metrics["rmse"])
                    step_nll: float = float(step_metrics["nll"])

                    if not np.isfinite(step_rmse):
                        step_rmse = 1e6
                    if not np.isfinite(step_nll):
                        step_nll = 1e6

                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning(
                        "Rollout exception at traj=%d, step=%d: %s",
                        traj_idx, step_idx, exc,
                    )
                    step_rmse = 1e6
                    step_nll = 1e6
                    # Use zeros as the prediction to allow rollout to continue
                    y_pred_batch = jnp.zeros_like(y_true_batch)

                all_rmse[traj_idx, step_idx] = step_rmse
                all_nll[traj_idx, step_idx] = step_nll

                # ----------------------------------------------------------
                # Update input window for the next step.
                # Slide the time-history window: drop oldest step, append
                # the current prediction as the newest step.
                # Auxiliary channels (velocity, reaction) are preserved.
                # ----------------------------------------------------------
                current_input = self._update_input_window(
                    current_input=current_input,
                    y_pred=y_pred_batch,
                    input_steps=input_steps,
                    aux_channels=aux_channels,
                    is_2d=is_2d,
                )

        # ------------------------------------------------------------------
        # Average over trajectories: shape [n_steps_actual]
        # ------------------------------------------------------------------
        mean_rmse: np.ndarray = np.mean(all_rmse, axis=0)  # [n_steps_actual]
        mean_nll: np.ndarray = np.mean(all_nll, axis=0)    # [n_steps_actual]

        # Pad to n_steps if n_steps_actual < n_steps
        if n_steps_actual < n_steps:
            pad_len: int = n_steps - n_steps_actual
            mean_rmse = np.concatenate(
                [mean_rmse, np.full(pad_len, mean_rmse[-1])]
            )
            mean_nll = np.concatenate(
                [mean_nll, np.full(pad_len, mean_nll[-1])]
            )

        return {
            "rmse": mean_rmse,
            "nll": mean_nll,
        }

    # -----------------------------------------------------------------------
    # Input Window Update Helper
    # -----------------------------------------------------------------------

    def _update_input_window(
        self,
        current_input: jnp.ndarray,
        y_pred: jnp.ndarray,
        input_steps: int,
        aux_channels: int,
        is_2d: bool,
    ) -> jnp.ndarray:
        """Update the sliding input window for autoregressive rollout.

        Drops the oldest time step from the history window and appends the
        new prediction as the most recent step. Auxiliary channels (velocity
        field, reaction term) are preserved unchanged.

        For 1D inputs (shape [1, spatial_res, in_channels]):
          - Channels 0..input_steps-1: time history (oldest to newest)
          - Channels input_steps..in_channels-1: auxiliary (velocity, reaction)
          - Update: shift history left by 1, place y_pred at channel input_steps-1

        For 2D inputs (shape [1, H, W, in_channels]):
          - Same channel layout, but spatial dims are H and W.

        Args:
            current_input: Current input window, shape [1, spatial, in_channels]
                (1D) or [1, H, W, in_channels] (2D).
            y_pred: Predicted output for the current step, shape
                [1, spatial, out_channels] (1D) or [1, H, W, out_channels] (2D).
                out_channels is typically 1.
            input_steps: Number of time steps in the history window.
                Typically 10 (config.model.input_steps).
            aux_channels: Number of auxiliary channels (velocity + reaction).
                2 for 1D, 3 for 2D.
            is_2d: True if the input has 2D spatial dimensions.

        Returns:
            Updated input window with the same shape as ``current_input``.

        Notes:
            - The update is performed via JAX array slicing and concatenation
              (no in-place mutation, consistent with JAX's functional style).
            - If out_channels > 1, only the first channel of y_pred is used
              as the new time step (since the FNO predicts one scalar field).
            - The auxiliary channels are taken from the current input (not
              from y_pred), preserving the original velocity and reaction fields.
        """
        if is_2d:
            # current_input: [1, H, W, in_channels]
            # y_pred: [1, H, W, out_channels]

            # Extract the time history channels (drop oldest, keep newest input_steps-1)
            # Old history: current_input[:, :, :, 0:input_steps]
            # New history: current_input[:, :, :, 1:input_steps] + y_pred[:, :, :, 0:1]
            old_history: jnp.ndarray = current_input[:, :, :, 1:input_steps]
            # old_history.shape: [1, H, W,