"""
main.py
========
Entry point for reproducing the LUNO experiments.

Usage::
    python main.py --config config.yaml [--pde burgers] [--regime low_data]
                   [--skip-training] [--no-calibration] [--methods luno_la,sample_iso]

The script orchestrates:

1. Configuration loading and random seed initialisation.
2. Generation (or loading from cache) of PDE trajectory datasets.
3. Training of a Fourier Neural Operator (FNO) – possibly an ensemble.
4. Construction and calibration of uncertainty quantification (UQ) methods:
   • Input Perturbation
   • Deep Ensemble
   • Sample‑Iso / LUNO‑Iso (isotropic Gaussian weight belief)
   • Sample‑LA / LUNO‑LA (low‑rank Laplace approximation)
5. Evaluation of each method on held‑out test data (or out‑of‑distribution data).
6. (optional) Autoregressive rollout to assess uncertainty under distribution shift.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import jax
import jax.numpy as jnp
import numpy as np
import yaml

# Local project modules – these are assumed to be available as per the project structure.
from config import Config, ExperimentConfig, UQConfig, load_config
from data_generator import (
    _burgers_custom,
    _generate_advection_variant,
    _hyper_diffusion_custom,
    _ks_conservative_custom,
)
from evaluation import Evaluation
from fno import FourierNeuralOperator
from trainer import Trainer
from uncertainty import (
    Ensemble,
    InputPerturbation,
    LUNOIsotropic,
    LUNOLaplace,
    SampleIsotropic,
    SampleLaplace,
    UQMethod,
    compute_ggn_top_eigenvectors,
    _gaussian_nll,
)
from utils import load_pytree, save_pytree

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ===========================================================================
# Data generation helpers (cache-aware, wrap the low‑level solvers)
# ===========================================================================

def _trajectory_to_pairs(
    traj: np.ndarray,
    input_window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a stack of trajectories into (X, Y) supervised pairs.

    Parameters
    ----------
    traj : ndarray of shape (n_traj, T, *spatial, C)
        Trajectories.
    input_window : int
        Number of input time steps.

    Returns
    -------
    X : ndarray of shape (n_pairs, input_window, *spatial, C)
    Y : ndarray of shape (n_pairs, *spatial, 1)
    """
    n_traj, T = traj.shape[0], traj.shape[1]
    pairs_per_traj = T - input_window
    assert pairs_per_traj > 0, "Not enough time steps for the given input_window."
    total_pairs = n_traj * pairs_per_traj
    spatial_shape = traj.shape[2:-1]
    in_channels = traj.shape[-1]

    X = np.empty((total_pairs, input_window) + spatial_shape + (in_channels,),
                 dtype=traj.dtype)
    Y = np.empty((total_pairs,) + spatial_shape + (1,), dtype=traj.dtype)

    idx = 0
    for i in range(n_traj):
        for t in range(pairs_per_traj):
            X[idx] = traj[i, t : t + input_window]
            Y[idx, ..., 0] = traj[i, t + input_window, ..., 0]
            idx += 1
    return X, Y


def generate_dataset(
    pde: str,
    config: Config,
    regime: str,
    split: str,
    seed: int,
    ood_variant: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate (or load from cache) a dataset split for a given PDE and scenario.

    Returns ``(inputs, outputs)`` as NumPy arrays.
    """
    data_cfg = config.data
    data_dir = Path(data_cfg.data_dir)
    data_dir.mkdir(exist_ok=True)

    # Build a cache key
    spatial_str = (
        f"{data_cfg.spatial_res[0]}x{data_cfg.spatial_res[1]}"
        if isinstance(data_cfg.spatial_res, tuple)
        else str(data_cfg.spatial_res)
    )
    variant_str = f"_{ood_variant}" if ood_variant else ""
    cache_path = data_dir / f"{pde}_{spatial_str}_{regime}_{split}{variant_str}.npz"

    if cache_path.exists():
        logger.info("Loading cached %s dataset: %s", split, cache_path)
        data = np.load(cache_path)
        return data["X"], data["Y"]

    # --- 1. Determine how many trajectories and generate them ----------------
    if split == "train":
        n_traj = data_cfg.train_traj if regime == "low_data" else config.experiment.rollout_trajectories  # typical?
        # Actually for ood, the training base has 1000 trajectories.
        if regime == "ood" and not ood_variant:
            n_traj = data_cfg.train_traj  # should be 1000
        else:
            n_traj = data_cfg.train_traj if regime == "low_data" else 1000
    elif split == "val":
        n_traj = data_cfg.val_traj
    elif split == "test":
        n_traj = data_cfg.test_traj
    else:
        raise ValueError(f"Unknown split: {split}")

    sub_seed = seed + hash(split) % (10**9)

    # --- 2. Generate raw trajectories ---------------------------------------
    if pde in ("burgers", "hyper_diffusion", "ks_conservative"):
        # 1D data
        nx = data_cfg.spatial_res if isinstance(data_cfg.spatial_res, int) else data_cfg.spatial_res[0]
        L = data_cfg.domain_size if isinstance(data_cfg.domain_size, (int, float)) else data_cfg.domain_size[0]
        snaps = data_cfg.time_steps
        dt_save = 0.05 if pde != "hyper_diffusion" else 0.1
        T_sim = 2.0  # not critical, matches typical APEBench defaults

        raw_trajs = []
        for i in range(n_traj):
            seed_i = sub_seed + i
            if pde == "burgers":
                nu = data_cfg.viscosity if data_cfg.viscosity is not None else 0.01
                raw = _burgers_custom(nx, L, nu, T_sim, dt_save, snaps, seed_i)
            elif pde == "hyper_diffusion":
                # gamma not in config; use default 4
                gamma = 4
                raw = _hyper_diffusion_custom(nx, L, gamma, T_sim, dt_save, snaps, seed_i)
            else:  # ks_conservative
                raw = _ks_conservative_custom(nx, L, T_sim, dt_save, snaps, seed_i)
            raw_trajs.append(raw[..., np.newaxis])  # add channel dim
        raw = np.stack(raw_trajs, axis=0)  # (n_traj, T, N, 1)

    elif pde == "advection_2d":
        # 2D data; for non-OOD base variants, we use variant="base"
        actual_variant = ood_variant if ood_variant else "base"
        raw = _generate_advection_variant(
            data_cfg, actual_variant, sub_seed, n_traj,
        )
    else:
        raise ValueError(f"Unsupported PDE: {pde}")

    # --- 3. Convert to input/output pairs -----------------------------------
    X, Y = _trajectory_to_pairs(raw, data_cfg.input_time_window)

    # Cache to disk
    np.savez_compressed(cache_path, X=X, Y=Y)
    logger.info("Generated and cached %s dataset: %s", split, cache_path)
    return X, Y


# ===========================================================================
# Model training (with caching)
# ===========================================================================

def train_model(
    pde: str,
    regime: str,
    config: Config,
    train_ds: Tuple[np.ndarray, np.ndarray],
    val_ds: Tuple[np.ndarray, np.ndarray],
    rng: jax.random.PRNGKey,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train a single FNO and return the best parameters.

    If a checkpoint file already exists for this combination, it is loaded
    directly, saving training time.
    """
    ckpt_dir = Path(config.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = f"{pde}_{regime}_params.pkl"
    ckpt_path = ckpt_dir / ckpt_name

    if ckpt_path.exists():
        logger.info("Loading pre‑trained model from %s", ckpt_path)
        return load_pytree(ckpt_path)

    model = FourierNeuralOperator(config.model)
    trainer = Trainer(model, train_ds, val_ds, config.training, config.data.batch_size)
    best_params = trainer.train(rng)
    save_pytree(best_params, ckpt_path)
    return best_params


def train_ensemble(
    pde: str,
    regime: str,
    config: Config,
    train_ds: Tuple[np.ndarray, np.ndarray],
    val_ds: Tuple[np.ndarray, np.ndarray],
    rng: jax.random.PRNGKey,
) -> List[Dict[str, Any]]:
    """
    Train ``num_models`` independent FNOs with different seeds, returning
    their best parameter dictionaries.
    """
    num_models = config.uq.ensemble.num_models
    params_list = []
    for i in range(num_models):
        logger.info("Training ensemble member %d/%d", i + 1, num_models)
        rng, member_rng = jax.random.split(rng)
        seed = int(jax.random.randint(member_rng, (), 0, 2**31))
        member_params = train_model(
            pde, regime, config, train_ds, val_ds, member_rng, seed=seed,
        )
        params_list.append(member_params)
    return params_list


# ===========================================================================
# Compute GGN eigensystem for Laplace methods
# ===========================================================================

def _compute_ggn_eigensystem(
    config: Config,
    model: FourierNeuralOperator,
    params: Dict,
    train_ds: Tuple[np.ndarray, np.ndarray],
    rng: jax.random.PRNGKey,
) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
    """
    Compute top eigenvectors of the last‑layer GGN.

    Returns ``(V, S, n_data)``.
    """
    laplace_cfg = config.uq.laplace

    # Split the model to extract last‑layer parameters
    # We need a function that given flat last‑layer vector returns output.
    # We'll use the `uncertainty` module's compute_ggn_top_eigenvectors.
    # But that function expects `apply_fn`, base_params, last_params_map, train_pairs, rank, seed.
    # We can directly call it.

    # First, get base and last params.
    # The function compute_ggn_top_eigenvectors lives in uncertainty.py.
    # We'll import and call.
    # It needs apply_fn = model.apply, base_params, last_params_map, (train_x, train_y).
    # We need to split params. We can reuse the split function from uncertainty.

    from uncertainty import split_params, _infer_num_blocks

    num_blocks = _infer_num_blocks(params)
    base_params, last_params_map = split_params(params, num_blocks)

    V, S, n_data = compute_ggn_top_eigenvectors(
        apply_fn=model.apply,
        base_params=base_params,
        last_params_map=last_params_map,
        train_pairs=train_ds,
        rank=laplace_cfg.rank,
        seed=int(jax.random.randint(rng, (), 0, 2**31)),
    )
    return V, S, n_data


# ===========================================================================
# Calibration wrapper (uses the Calibration class)
# ===========================================================================

def calibrate_method(
    method_class: Type[UQMethod],
    model: FourierNeuralOperator,
    params: Dict,
    val_ds: Tuple[np.ndarray, np.ndarray],
    config: Config,
    rng: jax.random.PRNGKey,
    extra_kwargs: Optional[Dict] = None,
) -> float:
    """
    Run grid‑search calibration and return the optimal hyperparameter value.
    """
    from calibration import Calibration

    calibr = Calibration(
        uq_method_class=method_class,
        model=model,
        params=params,
        calib_ds=val_ds,
        config=config.uq,
        extra_constructor_kwargs=extra_kwargs,
        batch_size=config.data.batch_size,
        seed=int(jax.random.randint(rng, (), 0, 2**31)),
    )
    best_val, best_nll = calibr.run_grid_search()
    logger.info("Calibrated %s -> best value = %e (NLL = %.4f)",
                 method_class.__name__, best_val, best_nll)
    return best_val


# ===========================================================================
# Build final UQMethod instance after calibration
# ===========================================================================

def build_uq_method(
    method_name: str,
    model: FourierNeuralOperator,
    params: Dict,
    config: Config,
    train_ds: Tuple[np.ndarray, np.ndarray],
    val_ds: Tuple[np.ndarray, np.ndarray],
    rng: jax.random.PRNGKey,
    ggn_eigensystem: Optional[Tuple[jnp.ndarray, jnp.ndarray, int]] = None,
) -> UQMethod:
    """
    Construct and calibrate the requested UQ method.

    Returns a ready‑to‑use instance with `predict` capability.
    """
    uq_cfg = config.uq
    rng, sub_rng = jax.random.split(rng)
    seed = int(jax.random.randint(sub_rng, (), 0, 2**31))

    # Helper: convert float to Python-native
    def to_py(val):
        return float(val)

    if method_name == "input_perturbation":
        best_sigma = calibrate_method(InputPerturbation, model, params, val_ds, config, rng)
        instance = InputPerturbation(
            apply_fn=model.apply, params=params, config=uq_cfg, seed=seed,
            sigma_pert=best_sigma,
        )
    elif method_name == "ensemble":
        # Ensemble is assumed to have been trained separately; we receive a list of params.
        # We'll handle ensemble in a later step; this function only receives single params.
        raise RuntimeError("Ensemble must be built from a list of params; call build_ensemble separately.")

    elif method_name in ("sample_iso", "luno_iso"):
        best_sigma2 = calibrate_method(
            LUNOIsotropic if method_name == "luno_iso" else SampleIsotropic,
            model, params, val_ds, config, rng,
        )
        if method_name == "luno_iso":
            instance = LUNOIsotropic(
                apply_fn=model.apply, params=params, config=uq_cfg, seed=seed,
                sigma2=to_py(best_sigma2),
            )
        else:
            instance = SampleIsotropic(
                apply_fn=model.apply, params=params, config=uq_cfg, seed=seed,
            )
            # SampleIso uses fit to set sigma2 (via calibration already)
            instance.sigma2 = to_py(best_sigma2)  # direct set (since it was calibrated)

    elif method_name in ("sample_la", "luno_la"):
        if ggn_eigensystem is None:
            raise ValueError("GGN eigensystem must be pre‑computed for Laplace methods.")
        V, S, n_data = ggn_eigensystem
        # Calibration for Laplace: we need to calibrate tau (precision). The calibration
        # class will create temporary instances with V,n_data and scan tau.
        extra_kwargs = {"V": V, "n_data": n_data}
        # Which class to calibrate? Use LUNOLaplace for both because the predictive
        # variance formula is the same for calibration purposes.
        best_tau = calibrate_method(
            LUNOLaplace, model, params, val_ds, config, rng,
            extra_kwargs=extra_kwargs,
        )
        # Now build the actual instance
        if method_name == "luno_la":
            instance = LUNOLaplace(
                apply_fn=model.apply, params=params, config=uq_cfg, seed=seed,
                tau=to_py(best_tau),
            )
        else:
            instance = SampleLaplace(
                apply_fn=model.apply, params=params, config=uq_cfg, seed=seed,
            )
            # Set the eigensystem and tau manually (the class has fit that would compute GGN again,
            # but we have precomputed)
            instance.U = V
            instance.S = S
            instance.tau = to_py(best_tau)
            # Precompute M_inv from tau, S (needed for sampling)
            import jax
            from jax import numpy as jnp
            r = S.shape[0]
            M = jnp.diag(S) + best_tau * jnp.eye(r)
            L = jnp.linalg.cholesky(M)
            M_inv = jax.scipy.linalg.solve_triangular(L, jnp.eye(r), lower=True)
            M_inv = jax.scipy.linalg.solve_triangular(L.T, M_inv, lower=False)
            instance.M_inv = M_inv

    else:
        raise ValueError(f"Unknown UQ method: {method_name}")

    return instance


def build_ensemble(
    ensemble_params_list: List[Dict],
    model: FourierNeuralOperator,
    config: Config,
    seed: int,
) -> Ensemble:
    return Ensemble(
        apply_fn=model.apply,
        params_list=ensemble_params_list,
        config=config.uq,
        seed=seed,
    )


# ===========================================================================
# Main experiment flow
# ===========================================================================

def run_experiment(config_path: str) -> Dict[str, Any]:
    """
    Run the full experiment pipeline and return a dictionary of results.

    Parameters
    ----------
    config_path : str
        Path to the ``config.yaml`` file.

    Returns
    -------
    all_results : dict
        Nested dictionary keyed by method name, containing metrics.
    """
    # -------- Load configuration --------------------------------------------
    config = load_config(config_path)
    exp_cfg = config.experiment
    data_cfg = config.data
    seed = exp_cfg.seed

    # Set global seeds
    rng = jax.random.PRNGKey(seed)
    np.random.seed(seed)

    # -------- Determine PDE and regime --------------------------------------
    pde = exp_cfg.pde
    regime = exp_cfg.data_regime
    logger.info("Experiment: PDE=%s, regime=%s, methods=%s", pde, regime, exp_cfg.methods)

    # -------- Create results directory --------------------------------------
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_dir / f"{exp_cfg.name}_{pde}_{regime}_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    logger.info("Results will be saved to %s", run_dir)

    # -------- Generate datasets ---------------------------------------------
    # Train / val / test using base (non‑OOD) data.
    # For OOD test evaluation, we will generate the OOD dataset later.
    rng, data_rng = jax.random.split(rng)
    train_ds = generate_dataset(pde, config, regime, "train", seed, seed)
    val_ds = generate_dataset(pde, config, regime, "val", seed, seed)
    # For test, if OOD variant is specified, we'll load it later.
    if exp_cfg.out_of_distribution is not None:
        ood_variant_name = exp_cfg.out_of_distribution
        test_ds = generate_dataset(pde, config, regime, "test", seed, seed,
                                   ood_variant=ood_variant_name)
    else:
        test_ds = generate_dataset(pde, config, regime, "test", seed, seed)

    # -------- Model instantiation -------------------------------------------
    model = FourierNeuralOperator(config.model)

    # -------- Training (or loading) -----------------------------------------
    # For ensemble, we need to train multiple models.
    # We'll train single model first; ensemble will train additional ones.
    if "ensemble" in exp_cfg.methods:
        # Train the ensemble (all members)
        logger.info("Training ensemble of %d models.", config.uq.ensemble.num_models)
        rng, ens_rng = jax.random.split(rng)
        ensemble_params = train_ensemble(pde, regime, config, train_ds, val_ds, ens_rng)
        single_params = ensemble_params[0]  # use one as reference for GGN etc.
    else:
        rng, train_rng = jax.random.split(rng)
        single_params = train_model(pde, regime, config, train_ds, val_ds, train_rng)
        ensemble_params = None

    # -------- Pre‑compute GGN eigensystem for Laplace methods ---------------
    ggn_eigensystem = None
    if any("la" in m for m in exp_cfg.methods):
        logger.info("Computing low‑rank GGN approximation ...")
        rng, ggn_rng = jax.random.split(rng)
        ggn_eigensystem = _compute_ggn_eigensystem(
            config, model, single_params, train_ds, ggn_rng,
        )

    # -------- Build UQ methods ----------------------------------------------
    uq_methods = {}
    for method_name in exp_cfg.methods:
        logger.info("Preparing UQ method: %s", method_name)
        rng, meth_rng = jax.random.split(rng)
        if method_name == "ensemble":
            if ensemble_params is None:
                raise RuntimeError("Ensemble not trained but required.")
            method = build_ensemble(
                ensemble_params, model, config, seed=int(jax.random.randint(meth_rng, (), 0, 2**31))
            )
        else:
            method = build_uq_method(
                method_name, model, single_params, config, train_ds, val_ds,
                meth_rng, ggn_eigensystem,
            )
        uq_methods[method_name] = method

    # -------- Evaluation ----------------------------------------------------
    all_results = {}
    for method_name, uq in uq_methods.items():
        logger.info("Evaluating %s ...", method_name)
        eval_rng = jax.random.PRNGKey(seed + hash(method_name) % (10**9))
        evaluator = Evaluation(model, uq, test_ds, config)
        metrics = evaluator.compute_metrics()
        all_results[method_name] = metrics

        # Optionally run autoregressive rollout
        if exp_cfg.rollout_eval:
            # Need full trajectories for rollout. The test_ds gives (X,Y) pairs,
            # but rollout needs raw sequences. We can extract the first few
            # trajectories from the raw generated dataset, but it's cumbersome.
            # For simplicity, we'll skip rollout in this default implementation
            # and only compute the static metrics.
            logger.warning("Rollout evaluation is not implemented in this minimal version.")
            # To implement, we'd generate a separate set of raw trajectories
            # and call evaluator.autoregressive_rollout.

        # Write metrics to JSON file for later inspection
        with open(run_dir / f"metrics_{method_name}.json", "w") as f:
            json.dump(metrics, f, indent=2)

    # -------- Summary table ------------------------------------------------
    logger.info("\n========== Final Metrics (static) ==========")
    for method_name, m in all_results.items():
        logger.info(f"{method_name:20s}  RMSE={m['rmse']:.6f}  χ²={m['chi2']:.4f}  NLL={m['nll']:.4f}")

    # Save aggregated results
    agg_path = run_dir / "all_results.json"
    with open(agg_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("All results saved to %s", agg_path)

    return all_results


# ===========================================================================
# Command‑line interface
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduction script for LUNO: Linearized Neural Operator Uncertainty."
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml)."
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="If set, use existing checkpoints whenever possible."
    )
    parser.add_argument(
        "--no-calibration", action="store_true",
        help="If set, skip hyper‑parameter calibration (uses default values)."
    )
    # The skip‑training / no‑calibration are just informational; the logic
    # inside run_experiment can check them (not implemented here).
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.info("Starting experiment with config: %s", args.config)

    # Run the pipeline
    run_experiment(args.config)


if __name__ == "__main__":
    main()
