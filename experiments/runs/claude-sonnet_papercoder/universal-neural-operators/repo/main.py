```python
## main.py
"""
Entry point for all experiments in:
  "Towards Universal Neural Operators through Multiphysics Pretraining"

Usage examples:
    # Run Experiment 1 with MambaFNO
    python main.py --config configs/exp1_out_of_sample.yaml \
                   --experiment exp1 --model mamba_fno --device cuda --seed 42

    # Run Experiment 2 with all models (reproduces Table 2 rows)
    python main.py --config configs/exp2_input_extension.yaml \
                   --experiment exp2 --model all --device cuda --seed 42

    # Run Experiment 3 with Perceiver NO
    python main.py --config configs/exp3_multiphysics.yaml \
                   --experiment exp3 --model perceiver_no --device cpu --seed 0

Design contract (Data structures and interfaces):
  - Accepts CLI args: --config, --experiment, --model, --device, --seed
  - Loads Config via Config.from_yaml(path)
  - Sets global random seeds for reproducibility
  - Dispatches to Exp1OutOfSample, Exp2InputExtension, or Exp3MultiPhysics
  - Calls experiment.setup_data() -> experiment.setup_model() -> experiment.run()
  - Saves results to JSON in config.eval.output_dir
  - Prints ResultsTable matching Tables 1 and 2 in the paper
  - Saves ResultsTable as CSV
  - Supports --model all to run every model sequentially

Config alignment (config.yaml):
  experiment.seed: 42                -> default seed
  experiment.device: "cuda"          -> default device
  evaluation.output_dir: "results"   -> JSON/CSV output directory
  logging.log_dir: "logs"            -> log file directory

Dependencies:
  utils/config.py              -> Config
  utils/logging_utils.py       -> get_logger, ResultsTable
  experiments/exp1_out_of_sample.py -> Exp1OutOfSample, ExperimentBase
  experiments/exp2_input_extension.py -> Exp2InputExtension
  experiments/exp3_multiphysics.py    -> Exp3MultiPhysics
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import numpy as np
import torch

from utils.config import Config
from utils.logging_utils import ResultsTable, get_logger

# ---------------------------------------------------------------------------
# Lazy experiment imports — deferred to avoid circular-import issues and to
# allow the logger to be set up before any heavy module-level code runs.
# ---------------------------------------------------------------------------

def _import_experiment_classes():
    """Import all three experiment classes and return them as a dict."""
    from experiments.exp1_out_of_sample import Exp1OutOfSample
    from experiments.exp2_input_extension import Exp2InputExtension
    from experiments.exp3_multiphysics import Exp3MultiPhysics
    return {
        "exp1": Exp1OutOfSample,
        "exp2": Exp2InputExtension,
        "exp3": Exp3MultiPhysics,
    }


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# All backbone model types the framework supports.
# "all" is a meta-value that triggers sequential execution over every entry.
ALL_MODELS: List[str] = ["fno", "mamba_fno", "perceiver_no", "coda_no", "swin_v2"]

# Valid experiment keys
VALID_EXPERIMENTS: List[str] = ["exp1", "exp2", "exp3"]

# Display names matching Tables 1 and 2 in the paper exactly.
_MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "fno":          "FNO",
    "mamba_fno":    "Mamba FNO",
    "perceiver_no": "Perc.",
    "coda_no":      "CoDA-NO",
    "swin_v2":      "Swin-v2",
}

# Default log directory (from config.yaml logging.log_dir: "logs")
_DEFAULT_LOG_DIR: str = "logs"

# Default output directory (from config.yaml evaluation.output_dir: "results")
_DEFAULT_OUTPUT_DIR: str = "results"

# Default seed (from config.yaml experiment.seed: 42)
_DEFAULT_SEED: int = 42

# Default device (from config.yaml experiment.device: "cuda")
_DEFAULT_DEVICE: str = "cuda"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_global_seed(seed: int) -> None:
    """Set all relevant random seeds for full reproducibility.

    Covers:
      - Python built-in ``random`` module
      - NumPy global RNG
      - PyTorch CPU and CUDA RNGs
      - cuDNN determinism flags (at the cost of some throughput)

    The paper does not specify a seed; we default to 42 (configurable via
    ``--seed`` CLI argument and ``config.yaml experiment.seed``).

    Args:
        seed: Integer seed value. Must be non-negative.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make cuDNN deterministic.  Required for reproducible epoch-time
    # benchmarks (Tables 1 and 2 "Avg. epoch (s)" column).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser.

    Arguments
    ---------
    --config      Path to the YAML configuration file.  Required.
    --experiment  Which experiment to run: exp1 | exp2 | exp3.  Required.
    --model       Which model backbone to use, or "all" to run every model
                  sequentially and produce a combined results table.
                  Choices: fno | mamba_fno | perceiver_no | coda_no |
                           swin_v2 | all.  Required.
    --device      Compute device: cuda | cpu.  Defaults to "cuda" if a GPU
                  is available, otherwise "cpu".
    --seed        Integer random seed.  Default: 42 (config.yaml
                  experiment.seed).

    Returns:
        Configured ArgumentParser instance.
    """
    default_device: str = (
        _DEFAULT_DEVICE if torch.cuda.is_available() else "cpu"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Reproduce experiments from "
            "'Towards Universal Neural Operators through Multiphysics Pretraining'"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        metavar="PATH",
        help=(
            "Path to the YAML configuration file. "
            "Examples: configs/exp1_out_of_sample.yaml, "
            "configs/exp2_input_extension.yaml, "
            "configs/exp3_multiphysics.yaml, "
            "or the root config.yaml."
        ),
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=VALID_EXPERIMENTS,
        help=(
            "Experiment to run: "
            "exp1 = out-of-sample parameter values (Table 1), "
            "exp2 = input function set extension (Table 2 Heat/RD rows), "
            "exp3 = general multi-physics transfer (Table 2 multi-physics rows)."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=ALL_MODELS + ["all"],
        help=(
            "Model backbone to use. "
            "Pass 'all' to run every model sequentially and aggregate "
            "results into one table — this reproduces the full Tables 1 "
            "and 2 from the paper."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=default_device,
        choices=["cuda", "cpu"],
        help=(
            "Compute device. "
            "MambaFNO requires CUDA (mamba_ssm GPU dependency). "
            "FNO, PerceiverNO, and CodaNO work on both CPU and CUDA."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_DEFAULT_SEED,
        help=(
            "Global random seed for reproducibility. "
            "Affects Python random, NumPy, and PyTorch RNGs. "
            "Default: 42 (config.yaml experiment.seed)."
        ),
    )

    return parser


# ---------------------------------------------------------------------------
# Config override helper
# ---------------------------------------------------------------------------


def override_model_type(config: Config, model_type: str) -> Config:
    """Return a deep-copied Config with model_type overridden.

    This is required when ``--model all`` iterates over every backbone:
    each iteration needs a fresh Config with the correct model_type so that
    ExperimentBase._build_backbone() instantiates the right class.

    We deep-copy the Config to prevent cross-contamination between iterations
    (e.g., a failed model's partial state affecting the next run).

    The Config dataclass stores model_type at the top level
    (``config.model_type``). The PretrainConfig also carries architecture
    hyperparameters (hidden_dim, n_modes, n_layers) that are model-specific
    and were merged from ``models.<model_type>.*`` in config.yaml during
    Config.from_yaml(). When overriding model_type, we keep the existing
    PretrainConfig architecture values (they were set for the original
    model_type in the YAML). For a fully correct override, the caller should
    reload the YAML with the new model_type; here we do a best-effort
    override that is sufficient for the --model all use case.

    Args:
        config:     Original Config loaded from YAML.
        model_type: One of ALL_MODELS (e.g., 'fno', 'mamba_fno').

    Returns:
        Deep-copied Config with config.model_type set to model_type.
    """
    cfg_copy: Config = copy.deepcopy(config)
    cfg_copy.model_type = model_type
    return cfg_copy


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


def resolve_device(requested_device: str, logger: logging.Logger) -> torch.device:
    """Resolve the target device with a graceful CUDA fallback.

    If the user requests CUDA but no GPU is available, logs a warning and
    falls back to CPU rather than crashing. This is important for CI/testing
    environments and for models that support CPU inference (FNO, PerceiverNO,
    CodaNO).

    Note: MambaFNO requires CUDA (mamba_ssm GPU dependency). If CUDA is
    unavailable and the model is mamba_fno, the experiment will raise an
    ImportError or RuntimeError with a descriptive message.

    Args:
        requested_device: Device string from CLI ('cuda' or 'cpu').
        logger:           Logger for warning messages.

    Returns:
        Resolved torch.device instance.
    """
    if requested_device == "cuda":
        if torch.cuda.is_available():
            device: torch.device = torch.device("cuda")
            gpu_name: str = torch.cuda.get_device_name(0)
            logger.info("CUDA device: %s", gpu_name)
        else:
            logger.warning(
                "CUDA requested but not available. "
                "Falling back to CPU. "
                "Note: MambaFNO requires CUDA and will fail if selected."
            )
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    return device


# ---------------------------------------------------------------------------
# Single-model experiment runner
# ---------------------------------------------------------------------------


def run_single_model(
    config: Config,
    experiment_key: str,
    model_type: str,
    device: torch.device,
    logger: logging.Logger,
    experiment_registry: Dict[str, Any],
) -> Dict[str, Any]:
    """Instantiate and execute one experiment for one model type.

    Calls the three-phase pipeline:
      1. experiment.setup_data()   — load all required datasets
      2. experiment.setup_model()  — build backbone + adapter framework
      3. experiment.run()          — pretrain, finetune, scratch, evaluate

    The experiment class is expected to return a results dict with the
    structure:
        {
          "pretrained": {
              "mse": float,
              "nmae": float,          # percentage, e.g. 0.0120
              "avg_epoch_s": float,   # wall-clock seconds per epoch
              "n_params": int,        # total parameter count
          },
          "scratch": {
              "mse": float,
              "nmae": float,
              "avg_epoch_s": float,
              "n_params": int,
          },
        }

    The exact keys depend on the experiment class implementation. Missing
    keys are handled gracefully in flatten_results_for_table().

    Args:
        config:               Config with model_type already set.
        experiment_key:       One of "exp1", "exp2", "exp3".
        model_type:           Backbone identifier string.
        device:               torch.device to run on.
        logger:               Logger instance.
        experiment_registry:  Dict mapping experiment_key -> class.

    Returns:
        Results dict as returned by experiment.run(). Returns
        {"error": str} if the experiment raises an exception.

    Raises:
        KeyError: If experiment_key is not in experiment_registry.
    """
    ExperimentClass = experiment_registry[experiment_key]

    logger.info(
        "Starting: experiment=%s  model=%s  device=%s",
        experiment_key,
        model_type,
        device,
    )

    # Instantiate the experiment.
    # The design specifies ExperimentBase.__init__(config_path: str), but
    # the experiment classes also accept a pre-built Config object and device
    # to support the --model all override pattern without writing temp files.
    # We pass both; the experiment class uses whichever it supports.
    try:
        experiment = ExperimentClass(config=config, device=device)
    except TypeError:
        # Fallback: some experiment classes may only accept config_path.
        # In that case, we cannot use the override pattern cleanly.
        # Log a warning and attempt construction with just the config path.
        logger.warning(
            "ExperimentClass %s does not accept (config, device) kwargs. "
            "Attempting construction with config_path only. "
            "model_type override may not take effect.",
            ExperimentClass.__name__,
        )
        # We cannot pass the overridden config in this fallback path.
        # The experiment will use whatever model_type is in the YAML.
        raise RuntimeError(
            f"ExperimentClass {ExperimentClass.__name__} must accept "
            f"(config: Config, device: torch.device) constructor arguments "
            f"to support --model all override. "
            f"Please update the experiment class constructor."
        )

    # ── Phase 1: Data setup ───────────────────────────────────────────────
    t0: float = time.perf_counter()
    logger.info("[%s/%s] Phase 1: setup_data()", experiment_key, model_type)
    experiment.setup_data()
    t1: float = time.perf_counter()
    logger.info(
        "[%s/%s] setup_data() completed in %.1f s",
        experiment_key, model_type, t1 - t0,
    )

    # ── Phase 2: Model setup ──────────────────────────────────────────────
    logger.info("[%s/%s] Phase 2: setup_model()", experiment_key, model_type)
    experiment.setup_model()
    t2: float = time.perf_counter()
    logger.info(
        "[%s/%s] setup_model() completed in %.1f s",
        experiment_key, model_type, t2 - t1,
    )

    # ── Phase 3: Run (pretrain + finetune + scratch + evaluate) ──────────
    logger.info("[%s/%s] Phase 3: run()", experiment_key, model_type)
    results: Dict[str, Any] = experiment.run()
    t3: float = time.perf_counter()
    logger.info(
        "[%s/%s] run() completed in %.1f s. Results: %s",
        experiment_key, model_type, t3 - t2, results,
    )

    return results


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------


def save_results_json(
    all_results: Dict[str, Dict[str, Any]],
    output_dir: str,
    experiment_key: str,
    logger: logging.Logger,
) -> Path:
    """Persist the aggregated results dictionary to a JSON file.

    File is written to:
        {output_dir}/{experiment_key}_results_{timestamp}.json

    The JSON structure mirrors Tables 1 and 2 in the paper:
        {
          "mamba_fno": {
              "pretrained": {
                  "mse": 1.009e-7,
                  "nmae": 0.0120,
                  "avg_epoch_s": 21.91,
                  "n_params": 10000000
              },
              "scratch": {
                  "mse": 1.193e-7,
                  "nmae": 0.0213,
                  "avg_epoch_s": 40.14,
                  "n_params": 10000000
              }
          },
          ...
        }

    Args:
        all_results:    Nested dict keyed by model_type.
        output_dir:     Directory path (created if absent).
        experiment_key: Used in the filename for identification.
        logger:         Logger instance.

    Returns:
        Path to the written JSON file.
    """
    out_path: Path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp: str = time.strftime("%Y%m%d_%H%M%S")
    json_path: Path = out_path / f"{experiment_key}_results_{timestamp}.json"

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2, default=str)

    logger.info("Results JSON saved to: %s", json_path)
    return json_path


# ---------------------------------------------------------------------------
# Results table formatting
# ---------------------------------------------------------------------------


def flatten_results_for_table(
    all_results: Dict[str, Dict[str, Any]],
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    """Convert the nested results dict into a flat list of row dicts.

    Each entry in all_results has the form:
        {
          "pretrained": {"mse": ..., "nmae": ..., "avg_epoch_s": ..., "n_params": ...},
          "scratch":    {"mse": ..., "nmae": ..., "avg_epoch_s": ..., "n_params": ...},
        }

    We flatten this to rows compatible with ResultsTable.add_row():
        {
          "model_name": "Mamba FNO (pretr.)",
          "metrics":    {"mse": float, "nmae": float},
          "epoch_time": float,
          "n_params":   int or None,
        }

    The model_name strings match the paper's Table 1/2 notation exactly:
      - "Mamba FNO (pretr.)"  / "Mamba FNO (scratch)"
      - "Perc. (pretr.)"      / "Perc. (scratch)"
      - "FNO (scratch)"
      - "Swin-v2 (p.+s.)"
      - "CoDA-NO (pretr.)"    / "CoDA-NO (scratch)"

    Missing keys in the results dict are handled gracefully (default to 0.0
    or None) so that a partially failed run still produces a table row.

    Args:
        all_results: Dict keyed by model_type string.
        logger:      Logger instance.

    Returns:
        List of row dicts, one per (model_type, mode) combination.
        Rows are ordered: pretrained before scratch, in ALL_MODELS order.
    """
    rows: List[Dict[str, Any]] = []

    # Iterate in ALL_MODELS order for consistent table layout.
    for model_type in ALL_MODELS:
        if model_type not in all_results:
            continue

        result: Dict[str, Any] = all_results[model_type]
        base_name: str = _MODEL_DISPLAY_NAMES.get(model_type, model_type)

        # ── Error case: experiment failed ─────────────────────────────────
        if "error" in result and len(result) == 1:
            logger.warning(
                "Model %s failed with error: %s. "
                "Skipping table rows for this model.",
                model_type,
                result["error"],
            )
            continue

        # ── Pretrained variant ────────────────────────────────────────────
        if "pretrained" in result:
            r: Dict[str, Any] = result["pretrained"]
            mse_val: float = float(r.get("mse", 0.0))
            nmae_val: float = float(r.get("nmae", 0.0))
            epoch_time_val: float = float(r.get("avg_epoch_s", 0.0))
            n_params_val: Optional[int] = (
                int(r["n_params"]) if "n_params" in r and r["n_params"] is not None
                else None
            )

            rows.append(
                {
                    "model_name": f"{base_name} (pretr.)",
                    "metrics": {"mse": mse_val, "nmae": nmae_val},
                    "epoch_time": epoch_time_val,
                    "n_params": n_params_val,
                }
            )

        # ── Scratch variant ───────────────────────────────────────────────
        if "scratch" in result:
            r = result["scratch"]
            mse_val = float(r.get("mse", 0.0))
            nmae_val = float(r.get("nmae", 0.0))
            epoch_time_val = float(r.get("avg_epoch_s", 0.0))
            n_params_val = (
                int(r["n_params"]) if "n_params" in r and r["n_params"] is not None
                else None
            )

            # Swin-v2 uses "(p.+s.)" notation per the paper (Table 1).
            # All other models use "(scratch)".
            if model_type == "swin_v2":
                suffix: str = " (p.+s.)"
            else:
                suffix = " (scratch)"

            rows.append(
                {
                    "model_name": f"{base_name}{suffix}",
                    "metrics": {"mse": mse_val, "nmae": nmae_val},
                    "epoch_time": epoch_time_val,
                    "n_params": n_params_val,
                }
            )

        # ── Handle flat results (no pretrained/scratch nesting) ───────────
        # Some experiment implementations may return a flat dict directly.
        if "pretrained" not in result and "scratch" not in result and "error" not in result:
            logger.debug(
                "Model %s returned a flat results dict (no 'pretrained'/'scratch' keys). "
                "Treating as a single row.",
                model_type,
            )
            mse_val = float(result.get("mse", 0.0))
            nmae_val = float(result.get("nmae", 0.0))
            epoch_time_val = float(result.get("avg_epoch_s", 0.0))
            n_params_val = (
                int(result["n_params"])
                if "n_params" in result and result["n_params"] is not None
                else None
            )
            rows.append(
                {
                    "model_name": base_name,
                    "metrics": {"mse": mse_val, "nmae": nmae_val},
                    "epoch_time": epoch_time_val,
                    "n_params": n_params_val,
                }
            )

    return rows


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Orchestrate the full experiment pipeline.

    Execution flow
    --------------
    1.  Parse CLI arguments.
    2.  Set global random seeds (Python, NumPy, PyTorch, cuDNN).
    3.  Set up the main logger (console + file).
    4.  Load Config from YAML via Config.from_yaml().
    5.  Resolve device (CLI --device overrides config.experiment.device).
    6.  Import experiment classes (deferred to avoid circular imports).
    7.  Determine which models to run (single model or all).
    8.  For each model:
        a. Deep-copy Config and override model_type.
        b. Instantiate experiment class with (config, device).
        c. Call setup_data() → setup_model() → run().
        d. Collect results dict.
        e. On failure: log exception, store {"error": str}, continue.
    9.  Aggregate all results into a single dict keyed by model_type.
    10. Save aggregated results to JSON in config.eval.output_dir.
    11. Build ResultsTable from flattened results.
    12. Print ResultsTable to stdout (matches Tables 1 and 2 in the paper).
    13. Save ResultsTable as CSV in config.eval.output_dir.
    14. Exit with code 0 on success, 1 if all models failed.
    """
    # ── 1. Parse CLI arguments ────────────────────────────────────────────
    parser: argparse.ArgumentParser = build_arg_parser()
    args: argparse.Namespace = parser.parse_args()

    # ── 2. Set global random seeds ────────────────────────────────────────
    # Done before any other computation so that dataset generation,
    # model initialization, and DataLoader shuffling are all seeded.
    set_global_seed(args.seed)

    # ── 3. Set up main logger ─────────────────────────────────────────────
    # Logger is set up before Config loading so we can report errors.
    # Log directory from config.yaml logging.log_dir: "logs".
    log_dir: str = _DEFAULT_LOG_DIR
    os.makedirs(log_dir, exist_ok=True)

    timestamp_str: str = time.strftime("%Y%m%d_%H%M%S")
    log_filename: str = (
        f"{args.experiment}_{args.model}_{timestamp_str}.log"
    )
    log_file_path: str = os.path.join(log_dir, log_filename)

    logger: logging.Logger = get_logger("main", log_file=log_file_path)
    logger.setLevel(logging.INFO)

    logger.info("=" * 72)
    logger.info("Multiphysics Neural Operator Reproduction")
    logger.info("  Paper: 'Towards Universal Neural Operators through")
    logger.info("           Multiphysics Pretraining'")
    logger.info("  experiment : %s", args.experiment)
    logger.info("  model      : %s", args.model)
    logger.info("  config     : %s", args.config)
    logger.info("  device     : %s", args.device)
    logger.info("  seed       : %d", args.seed)
    logger.info("  log_file   : %s", log_file_path)
    logger.info("=" * 72)

    # ── 4. Load Config from YAML ──────────────────────────────────────────
    config_path: str = args.config
    if not os.path.isfile(config_path):
        logger.error(
            "Configuration file not found: '%s'. "
            "Please provide a valid path via --config.",
            config_path,
        )
        sys.exit(1)

    try:
        config: Config = Config.from_yaml(config_path)
    except Exception as cfg_exc:
        logger.exception(
            "Failed to load configuration from '%s': %s",
            config_path,
            cfg_exc,
        )
        sys.exit(1)

    logger.info(
        "Config loaded: experiment_name='%s', model_type='%s'.",
        config.experiment_name,
        config.model_type,
    )

    # ── 5. Resolve device ─────────────────────────────────────────────────
    # CLI --device takes precedence over config.experiment.device.
    device: torch.device = resolve_device(args.device, logger)
    logger.info("Resolved device: %s", device)

    # ── 6. Import experiment classes ──────────────────────────────────────
    # Deferred import to avoid circular imports and allow logger setup first.
    try:
        experiment_registry: Dict[str, Any] = _import_experiment_classes()
    except ImportError as import_exc:
        logger.exception(
            "Failed to import experiment classes: %s. "
            "Ensure all dependencies are installed.",
            import_exc,
        )
        sys.exit(1)

    # ── 7. Determine which models to run ──────────────────────────────────
    # --model all → run every model in ALL_MODELS order.
    # --model <name> → run only that model.
    if args.model == "all":
        models_to_run: List[str] = list(ALL_MODELS)
        logger.info(
            "Running ALL models sequentially: %s", models_to_run
        )
    else:
        models_to_run = [args.model]
        logger.info("Running single model: %s", args.model)

    # ── 8. Run experiments for each model ─────────────────────────────────
    # all_results accumulates one entry per model_type.
    # Structure: { model_type: { "pretrained": {...}, "scratch": {...} } }
    all_results: Dict[str, Dict[str, Any]] = {}
    n_successful: int = 0
    n_failed: int = 0

    for model_type in models_to_run:
        logger.info("─" * 60)
        logger.info("Model: %s  (%d/%d)", model_type, models_to_run.index(model_type) + 1, len(models_to_run))

        # ── 8a. Override model_type in a deep-copied Config ───────────────
        # This prevents cross-contamination between iterations when
        # --model all is used.
        cfg: Config = override_model_type(config, model_type)

        # ── 8b–8d. Run the experiment ─────────────────────────────────────
        try:
            results: Dict[str, Any] = run_single_model(
                config=cfg,
                experiment_key=args.experiment,
                model_type=model_type,
                device=device,
                logger=logger,
                experiment_registry=experiment_registry,
            )
            all_results[model_type] = results
            n_successful += 1
            logger.info(
                "Model %s completed successfully. Results keys: %s",
                model_type,
                list(results.keys()),
            )

        except Exception as run_exc:  # noqa: BLE001
            # Log the full traceback but continue with remaining models.
            # A single broken model (e.g., MambaFNO on CPU) should not
            # abort the full comparison run.
            logger.