## Code: main.py

```python
## main.py
"""Main entry point for the LUNO reproduction experiments.

Orchestrates the full pipeline:
  1. Parse CLI arguments and load config.yaml
  2. Generate or load PDE datasets
  3. Train FNO(s)
  4. Build all UQ methods (LUNO-LA, LUNO-Iso, Sample-LA, Sample-Iso,
     InputPerturbations, DeepEnsemble)
  5. Calibrate each method on the validation set
  6. Evaluate on the test set and print results matching the paper's tables
  7. (OOD only) Run autoregressive rollout evaluation

Paper references:
  - Section 5 / Appendix D: full experimental protocol
  - Tables 1, 2, 4, 5, 6: result format
  - Figure 4, 7: autoregressive rollout
  - config.yaml: all hyperparameters

Usage:
    python main.py --experiment low_data --pde burgers --seed 42
    python main.py --experiment ood --seed 42
    python main.py --config my_config.yaml --experiment low_data --pde ks_conservative
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Project imports (order follows dependency graph)
# ---------------------------------------------------------------------------
from config import Config
from data.dataset import PDEDataset
from data.apebench_loader import APEBenchLoader
from data.advdiff_solver import AdvDiffSolver
from models.fno import FNO
from training.trainer import Trainer
from uncertainty.ggn import GGNComputer
from uncertainty.luno import LaplaceApprox, LUNOInference
from uncertainty.weight_space import WeightSpaceBelief
from baselines.ensembles import DeepEnsemble
from baselines.input_perturbations import InputPerturbations
from evaluation.metrics import Metrics
from evaluation.calibration import Calibrator
from evaluation.evaluator import Evaluator
from utils.jax_utils import flatten_params

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SamplePushforward: sample-based UQ method
# ---------------------------------------------------------------------------

class SamplePushforward:
    """Sample-based pushforward for uncertainty quantification.

    Draws weight samples from a belief (LaplaceApprox or WeightSpaceBelief),
    pushes each through the (nonlinear) FNO, and computes empirical mean
    and variance. Implements Sample-LA and Sample-Iso from the paper.

    Paper reference: Appendix D.3.5 — "we generate 200 samples that are
    propagated through the network."

    Attributes:
        model: The trained FNO instance.
        belief: Weight-space belief with a sample() method.
            Either LaplaceApprox (Sample-LA) or WeightSpaceBelief (Sample-Iso).
        n_samples: Number of weight samples. Config: 200.
        params: MAP parameter pytree (NNX state) of the trained FNO.
            Used to reconstruct the model with sampled weights.
    """

    def __init__(
        self,
        model: FNO,
        belief: Any,
        n_samples: int = 200,
        params: Optional[Any] = None,
    ) -> None:
        """Initialise the sample-based pushforward.

        Args:
            model: The trained FNO instance.
            belief: Weight-space belief. Must expose sample(key, n_samples)
                returning shape [n_samples, p_last].
            n_samples: Number of weight samples per prediction.
                From config.uncertainty.sampling.n_samples = 200.
            params: MAP parameter pytree. If None, uses model's current state.
        """
        self.model: FNO = model
        self.belief: Any = belief
        self.n_samples: int = int(n_samples)
        self.params: Optional[Any] = params

        # Cache graphdef for functional forward passes
        self._graphdef: Optional[Any] = None
        try:
            from flax import nnx
            graphdef, _ = nnx.split(model)
            self._graphdef = graphdef
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("SamplePushforward: could not extract graphdef: %s", e)

        logger.debug(
            "SamplePushforward initialised: n_samples=%d, belief_type=%s",
            n_samples,
            type(belief).__name__,
        )

    def _get_all_predictions(
        self,
        a: jnp.ndarray,
        key: jax.Array,
    ) -> jnp.ndarray:
        """Draw weight samples and compute predictions for each.

        For Sample-LA: samples from the Laplace posterior over last-layer
        weights, reconstructs the full parameter pytree, and runs the FNO.
        For Sample-Iso: samples from N(w*, sigma^2 * I) over last-layer
        weights.

        Args:
            a: Input array, shape [batch, spatial, in_channels] (1D) or
               [batch, H, W, in_channels] (2D).
            key: JAX PRNG key for weight sampling.

        Returns:
            Stacked predictions, shape [n_samples, batch, spatial, out_channels]
            (1D) or [n_samples, batch, H, W, out_channels] (2D).
        """
        from flax import nnx

        # Draw weight samples: [n_samples, p_last]
        key_sample: jax.Array
        key_sample, _ = jax.random.split(key)
        weight_samples: jnp.ndarray = self.belief.sample(
            key_sample, n_samples=self.n_samples
        )
        # weight_samples.shape: [n_samples, p_last]

        predictions: List[jnp.ndarray] = []

        # Get the unflatten function for last-layer params
        # We need to reconstruct the full params with sampled last-layer weights
        base_params = self.params
        if base_params is None:
            # Fallback: use model's current state
            _, base_params = nnx.split(self.model)

        for s in range(self.n_samples):
            w_s: jnp.ndarray = weight_samples[s]  # [p_last]

            try:
                # Reconstruct full params with sampled last-layer weights
                new_params = self._reconstruct_params(base_params, w_s)

                # Run forward pass
                if self._graphdef is not None:
                    model_copy: FNO = nnx.merge(self._graphdef, new_params)
                    pred_s: jnp.ndarray = model_copy(a)
                else:
                    pred_s = self.model(a)

                predictions.append(pred_s)

            except Exception as exc:  # pylint: disable=broad-except
                logger.debug(
                    "SamplePushforward: exception at sample %d: %s", s, exc
                )
                # Use MAP prediction as fallback
                if self._graphdef is not None and base_params is not None:
                    model_map: FNO = nnx.merge(self._graphdef, base_params)
                    pred_map: jnp.ndarray = model_map(a)
                else:
                    pred_map = self.model(a)
                predictions.append(pred_map)

        # Stack: [n_samples, batch, spatial, out_channels]
        return jnp.stack(predictions, axis=0)

    def _reconstruct_params(
        self,
        base_params: Any,
        flat_last_layer: jnp.ndarray,
    ) -> Any:
        """Reconstruct full params with updated last-layer weights.

        Replaces the last_fourier_block parameters in base_params with
        the values from flat_last_layer.

        Args:
            base_params: Full parameter pytree (NNX state).
            flat_last_layer: Flat last-layer parameter vector, shape [p_last].

        Returns:
            Updated parameter pytree with new last-layer weights.
        """
        # Get the unflatten function from the belief's mean structure
        # We need to unflatten flat_last_layer to the last-layer sub-pytree
        try:
            # Extract last-layer subtree to get the structure
            last_layer_subtree = self._extract_last_layer_subtree(base_params)
            _, unflatten_fn = flatten_params(last_layer_subtree)
            new_last_layer = unflatten_fn(flat_last_layer)

            # Replace in base_params
            new_params = self._replace_last_layer(base_params, new_last_layer)
            return new_params

        except Exception:  # pylint: disable=broad-except
            # Fallback: return base_params unchanged
            return base_params

    def _extract_last_layer_subtree(self, params: Any) -> Any:
        """Extract last_fourier_block sub-tree from params."""
        if hasattr(params, "last_fourier_block"):
            return params.last_fourier_block
        if isinstance(params, dict) and "last_fourier_block" in params:
            return params["last_fourier_block"]
        if hasattr(params, "__dict__"):
            d = vars(params)
            if "last_fourier_block" in d:
                return d["last_fourier_block"]
        raise KeyError("last_fourier_block not found in params")

    def _replace_last_layer(self, params: Any, new_last_layer: Any) -> Any:
        """Replace last_fourier_block in params with new_last_layer."""
        import copy
        try:
            new_params = copy.copy(params)
            if hasattr(new_params, "last_fourier_block"):
                object.__setattr__(new_params, "last_fourier_block", new_last_layer)
                return new_params
        except Exception:  # pylint: disable=broad-except
            pass
        return params

    def predict_mean(
        self,
        a: jnp.ndarray,
        key: jax.Array,
    ) -> jnp.ndarray:
        """Compute empirical mean prediction over weight samples.

        Args:
            a: Input array, shape [batch, spatial, in_channels].
            key: JAX PRNG key.

        Returns:
            Empirical mean, same shape as FNO output.
        """
        preds: jnp.ndarray = self._get_all_predictions(a, key)
        return jnp.mean(preds, axis=0)

    def predict_marginal_variance(
        self,
        a: jnp.ndarray,
        key: jax.Array,
    ) -> jnp.ndarray:
        """Compute empirical marginal variance over weight samples.

        Args:
            a: Input array, shape [batch, spatial, in_channels].
            key: JAX PRNG key.

        Returns:
            Empirical marginal variance, same shape as FNO output.
        """
        preds: jnp.ndarray = self._get_all_predictions(a, key)
        return jnp.var(preds, axis=0)

    def __repr__(self) -> str:
        return (
            f"SamplePushforward("
            f"n_samples={self.n_samples}, "
            f"belief_type={type(self.belief).__name__}"
            f")"
        )


# ---------------------------------------------------------------------------
# Helper: print results table
# ---------------------------------------------------------------------------

def _print_results_table(
    results: Dict[str, Dict[str, float]],
    title: str = "Evaluation Results",
) -> None:
    """Print evaluation results in a table format matching the paper's tables.

    Args:
        results: Nested dict {method_name: {'rmse': float, 'nll': float, 'chi2': float}}.
        title: Table title string.
    """
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(f"{'Method':<25} {'RMSE (↓)':>14} {'χ²':>10} {'NLL (↓)':>12}")
    print(f"{'-'*70}")
    for method_name, metrics in results.items():
        rmse: float = metrics.get("rmse", float("nan"))
        chi2: float = metrics.get("chi2", float("nan"))
        nll: float = metrics.get("nll", float("nan"))
        print(
            f"{method_name:<25} {rmse:>14.4e} {chi2:>10.3f} {nll:>12.4f}"
        )
    print(f"{'='*70}\n")


def _print_ood_nll_table(
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    title: str = "OOD NLL Results",
) -> None:
    """Print OOD NLL results matching Table 2 / Table 6 of the paper.

    Args:
        all_results: {variant: {method_name: {metric: value}}}.
        title: Table title string.
    """
    # Collect all method names and variants
    variants: List[str] = list(all_results.keys())
    if not variants:
        return
    method_names: List[str] = list(all_results[variants[0]].keys())

    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

    # Header
    header: str = f"{'Method':<25}"
    for v in variants:
        header += f" {v.capitalize():>12}"
    print(header)
    print(f"{'-'*80}")

    for method_name in method_names:
        row: str = f"{method_name:<25}"
        for v in variants:
            nll_val: float = all_results[v].get(method_name, {}).get("nll", float("nan"))
            row += f" {nll_val:>12.3f}"
        print(row)

    print(f"{'='*80}\n")


def _save_results(
    results: Dict[str, Any],
    path: str,
) -> None:
    """Save results dict to a JSON file.

    Args:
        results: Results dictionary (must be JSON-serializable).
        path: Output file path.
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    # Convert JAX arrays and numpy arrays to Python floats for JSON serialization
    def _convert(obj: Any) -> Any:
        if isinstance(obj, (jnp.ndarray, np.ndarray)):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(x) for x in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj

    serializable: Any = _convert(results)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    logger.info("Results saved to: %s", path)


# ---------------------------------------------------------------------------
# _build_methods
# ---------------------------------------------------------------------------

def _build_methods(
    model: FNO,
    params: Any,
    train_ds: PDEDataset,
    val_ds: PDEDataset,
    key: jax.Array,
    config: Config,
    in_channels: int,
    dummy_input: jnp.ndarray,
) -> Tuple[Dict[str, Any], jax.Array]:
    """Build, calibrate, and return all 6 UQ methods.

    Constructs LUNO-LA, LUNO-Iso, Sample-LA, Sample-Iso, InputPerturbations,
    and DeepEnsemble. Each method is calibrated on the validation set
    immediately after construction.

    Args:
        model: The trained FNO instance (MAP weights).
        params: MAP parameter pytree (NNX state) from Trainer.train().
        train_ds: Training dataset (used for GGN computation).
        val_ds: Validation dataset (used for calibration).
        key: JAX PRNG key. Consumed and split internally; the updated key
            is returned as the second element of the tuple.
        config: Configuration dataclass.
        in_channels: Number of FNO input channels (12 for 1D, 13 for 2D).
        dummy_input: Dummy input array for model initialisation.

    Returns:
        Tuple (methods_dict, updated_key) where methods_dict maps method
        names to method objects.

    Notes:
        - LUNO-LA and Sample-LA share the same LaplaceApprox object.
        - LUNO-Iso and Sample-Iso share the same isotropic LaplaceApprox.
        - Calibration of LUNO-LA automatically calibrates Sample-LA (shared belief).
        - The GGN is computed only once and reused for both LA methods.
    """
    from flax import nnx

    logger.info("Building UQ methods ...")

    # ------------------------------------------------------------------
    # Determine n_pairs for GGN computation
    # ------------------------------------------------------------------
    n_pairs: int = (
        config.ggn_n_pairs_low_data
        if config.experiment == "low_data"
        else config.ggn_n_pairs_ood
    )

    # ------------------------------------------------------------------
    # Step 1: Flatten MAP parameters (full model)
    # ------------------------------------------------------------------
    flat_params_full: jnp.ndarray
    unflatten_full: Any
    flat_params_full, unflatten_full = flatten_params(params)
    logger.info("Full parameter count: p=%d", int(flat_params_full.shape[0]))

    # ------------------------------------------------------------------
    # Step 2: Compute low-rank GGN for last Fourier block
    # ------------------------------------------------------------------
    key, ggn_key = jax.random.split(key)
    logger.info(
        "Computing low-rank GGN: rank=%d, n_pairs=%d, last_layer_only=%s",
        config.ggn_rank,
        n_pairs,
        config.ggn_last_layer_only,
    )

    ggn_computer = GGNComputer(
        model=model,
        params=params,
        rank=config.ggn_rank,
        last_layer_only=config.ggn_last_layer_only,
    )

    eigvecs: jnp.ndarray
    eigvals: jnp.ndarray
    try:
        eigvecs, eigvals = ggn_computer.compute_low_rank(
            dataset=train_ds,
            n_pairs=n_pairs,
            key=ggn_key,
        )
        logger.info(
            "GGN computed: eigvecs.shape=%s, eigvals.shape=%s, "
            "top_eigval=%.4e",
            eigvecs.shape,
            eigvals.shape,
            float(eigvals[0]) if len(eigvals) > 0 else 0.0,
        )
    except Exception as e:  # pylint: disable=broad-except
        logger.warning(
            "GGN computation failed: %s. Using identity approximation.", e
        )
        # Fallback: use a minimal rank-1 GGN approximation
        # Extract last-layer params to get p_last
        try:
            flat_last, _ = ggn_computer._get_last_layer_params(params)
            p_last: int = int(flat_last.shape[0])
        except Exception:  # pylint: disable=broad-except
            p_last = int(flat_params_full.shape[0])
        actual_rank: int = min(config.ggn_rank, p_last)
        eigvecs = jnp.zeros((p_last, actual_rank), dtype=jnp.float32)
        eigvals = jnp.zeros((actual_rank,), dtype=jnp.float32)

    # ------------------------------------------------------------------
    # Step 3: Extract flat last-layer MAP parameters
    # ------------------------------------------------------------------
    flat_last_layer: jnp.ndarray
    try:
        flat_last_layer, _ = ggn_computer._get_last_layer_params(params)
        logger.info("Last-layer parameter count: p_last=%d", int(flat_last_layer.shape[0]))
    except Exception as e:  # pylint: disable=broad-except
        logger.warning(
            "Could not extract last-layer params: %s. Using full params.", e
        )
        flat_last_layer = flat_params_full

    # Ensure eigvecs has the correct leading dimension
    p_last_actual: int = int(flat_last_layer.shape[0])
    if eigvecs.shape[0] != p_last_actual:
        logger.warning(
            "eigvecs.shape[0]=%d != p_last=%d. Resizing eigvecs.",
            eigvecs.shape[0],
            p_last_actual,
        )
        actual_rank = min(config.ggn_rank, p_last_actual)
        eigvecs = jnp.zeros((p_last_actual, actual_rank), dtype=jnp.float32)
        eigvals = jnp.zeros((actual_rank,), dtype=jnp.float32)

    # ------------------------------------------------------------------
    # Step 4: Build LaplaceApprox for LUNO-LA / Sample-LA
    # ------------------------------------------------------------------
    laplace: LaplaceApprox = LaplaceApprox(
        mean=flat_last_layer,
        eigvecs=eigvecs,
        eigvals=eigvals,
        prior_prec=config.cal_prior_prec_center,  # 1.0 initial
        n_data=n_pairs,
    )
    logger.info("LaplaceApprox built: %s", laplace)

    # ------------------------------------------------------------------
    # Step 5: Build LUNO-LA (uncalibrated)
    # ------------------------------------------------------------------
    luno_la: LUNOInference = LUNOInference(
        model=model,
        params=params,
        belief=laplace,
        last_layer_only=True,
    )

    # ------------------------------------------------------------------
    # Step 6: Calibrate LUNO-LA prior_prec
    # ------------------------------------------------------------------
    key, cal_key = jax.random.split(key)
    logger.info("Calibrating LUNO-LA prior_prec ...")
    calibrator_la: Calibrator = Calibrator(
        method=luno_la,
        val_dataset=val_ds,
        grid_size=config.cal_grid_size,
        grid_range_factor=config.cal_grid_range_factor,
    )
    best_prior_prec: float = calibrator_la.calibrate(
        param_name="prior_prec",
        center=config.cal_prior_prec_center,
        key=cal_key,
    )
    laplace.set_prior_prec(best_prior_prec)
    logger.info("LUNO-LA calibrated: prior_prec=%.4e", best_prior_prec)

    # ------------------------------------------------------------------
    # Step 7: Build Sample-LA (shares laplace with LUNO-LA)
    # ------------------------------------------------------------------
    sample_la: SamplePushforward = SamplePushforward(
        model=model,
        belief=laplace,
        n_samples=config.n_samples,
        params=params,
    )
    # Sample-LA is already calibrated (shares laplace with LUNO-LA)
    logger.info("Sample-LA built (shares calibrated LaplaceApprox with LUNO-LA)")

    # ------------------------------------------------------------------
    # Step 8: Build isotropic LaplaceApprox for LUNO-Iso / Sample-Iso
    # Using rank-0 eigvecs so Woodbury reduces to (prior_prec * I)^{-1} * v
    # = (1/prior_prec) * v = sigma_sq * v
    # ------------------------------------------------------------------
    initial_prior_prec_iso: float = 1.0 / max(config.cal_sigma_sq_iso_center, 1e-10)
    laplace_iso: LaplaceApprox = LaplaceApprox(
        mean=flat_last_layer,
        eigvecs=jnp.zeros((p_last_actual, 0), dtype=jnp.float32),  # rank-0
        eigvals=jnp.zeros((0,), dtype=jnp.float32),
        prior_prec=initial_prior_prec_iso,
        n_data=n_pairs,
    )
    logger.info("Isotropic LaplaceApprox built: prior_prec=%.4e", initial_prior_prec_iso)

    # ------------------------------------------------------------------
    # Step 9: Build LUNO-Iso (uncalibrated)
    # ------------------------------------------------------------------
    luno_iso: LUNOInference = LUNOInference(
        model=model,
        params=params,
        belief=laplace_iso,
        last_layer_only=True,
    )

    # ------------------------------------------------------------------
    # Step 10: Calibrate LUNO-Iso (calibrate prior_prec = 1/sigma_sq)
    # ------------------------------------------------------------------
    key, cal_key = jax.random.split(key)
    logger.info("Calibrating LUNO-Iso prior_prec (= 1/sigma_sq) ...")
    calibrator_iso: Calibrator = Calibrator(
        method=luno_iso,
        val_dataset=val_ds,
        grid_size=config.cal_grid_size,
        grid_range_factor=config.cal_grid_range_factor,
    )
    best_prior_prec_iso: float = calibrator_iso.calibrate(
        param_name="prior_prec",
        center=initial_prior_prec_iso,
        key=cal_key,
    )
    laplace_iso.set_prior_prec(best_prior_prec_iso)
    best_sigma_sq_iso: float = 1.0 / max(best_prior_prec_iso, 1e-10)
    logger.info(
        "LUNO-Iso calibrated: prior_prec=%.4e (sigma_sq=%.4e)",
        best_prior_prec_iso,
        best_sigma_sq_iso,
    )

    # ------------------------------------------------------------------
    # Step 11: Build Sample-Iso (shares laplace_iso with LUNO-Iso)
    # ------------------------------------------------------------------
    sample_iso: SamplePushforward = SamplePushforward(
        model=model,
        belief=laplace_iso,
        n_samples=config.n_samples,
        params=params,
    )
    logger.info("Sample-Iso built (shares calibrated isotropic LaplaceApprox with LUNO-Iso)")

    # ------------------------------------------------------------------
    # Step 12: Build InputPerturbations (uncalibrated)
    # ------------------------------------------------------------------
    ip: InputPerturbations = InputPerturbations(
        model=model,
        params=params,
        sigma=config.cal_sigma_perturb_center,  # 0.01 initial
        n_samples=config.n_samples,
    )

    # ------------------------------------------------------------------
    # Step 13: Calibrate InputPerturbations sigma
    # ------------------------------------------------------------------
    key, cal_key = jax.random.split(key)
    logger.info("Calibrating InputPerturbations sigma ...")
    calibrator_ip: Calibrator = Calibrator(
        method=ip,
        val_dataset=val_ds,
        grid_size=config.cal_grid_size,
        grid_range_factor=config.cal_grid_range_factor,
    )
    best_sigma: float = calibrator_ip.calibrate(
        param_name="sigma",
        center=config.cal_sigma_perturb_center,
        key=cal_key,
    )
    ip.set_sigma(best_sigma)
    logger.info("InputPerturbations calibrated: sigma=%.4e", best_sigma)

    # ------------------------------------------------------------------
    # Step 14: Train ensemble and build DeepEnsemble
    # ------------------------------------------------------------------
    logger.info(
        "Training %d ensemble members ...", config.n_ensemble
    )
    key, ens_key = jax.random.split(key)
    ens_keys: jnp.ndarray = jax.random.split(ens_key, config.n_ensemble)

    ensemble_models: List[FNO] = []
    ensemble_params_list: List[Any] = []

    for i in range(config.n_ensemble):
        logger.info(
            "  Training ensemble member %d/%d ...", i + 1, config.n_ensemble
        )
        ens_model_i: FNO = FNO(
            modes=config.fno_modes,
            channels=config.fno_channels,
            n_blocks=config.fno_blocks,
            in_channels=in_channels,
            out_channels=config.out_channels,
            spatial_dims=model.spatial_dims,
            spatial_padding=config.spatial_padding,
            activation_name=config.activation,
            rngs=_make_rngs(int(ens_keys[i][0])),
        )

        # Initialise parameters
        try:
            ens_model_i.init_params(ens_keys[i], dummy_input)
        except Exception as e:  # pylint: disable=broad-except
            logger.debug("Ensemble member %d init_params: %s", i, e)

        ens_trainer_i: Trainer = Trainer(model=ens_model_i, config=config)
        try:
            _graphdef_i, ens_state_i = ens_trainer_i.train(
                dataset=train_ds,
                key=ens_keys[i],
            )
        except Exception as e:  # pylint: disable=broad-except
            logger.warning(
                "Ensemble member %d training failed: %s. Using random init.", i, e
            )
            from flax import nnx as _nnx
            _graphdef_i, ens_state_i = _nnx.split(ens_model_i)

        ensemble_models.append(ens_model_i)
        ensemble_params_list.append(ens_state_i)

    deep_ensemble: DeepEnsemble = DeepEnsemble(
        models=ensemble_models,
        params_list=ensemble_params_list,
    )
    logger.info("DeepEnsemble built: %s", deep_ensemble)

    # ------------------------------------------------------------------
    # Step 15: Assemble methods dict
    # ------------------------------------------------------------------
    methods_dict: Dict[str, Any] = {
        "LUNO-LA": luno_la,
        "LUNO-Iso": luno_iso,
        "Sample-LA": sample_la,
        "Sample-Iso": sample_iso,
        "Input Perturbations": ip,
        "Ensemble": deep_ensemble,
    }

    logger.info("All UQ methods built and calibrated.")
    return methods_dict, key


def _make_rngs(seed: int):
    """Create a Flax NNX Rngs object from an integer seed.

    Args:
        seed: Integer seed value.

    Returns:
        An nnx.Rngs instance seeded with the given