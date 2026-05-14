# config.py
"""
Configuration module for reproducing "Conformal Prediction as Bayesian Quadrature".

All hyperparameters and file paths are centralised here. The module provides a
`Config` dataclass with typed fields and a `from_yaml` loader that reads a
`config.yaml` file (see accompanying example).

Usage:
    from config import Config, load_config

    # Direct instantiation (defaults match the paper)
    cfg = Config()

    # Or load from file
    cfg = load_config("config.yaml")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore


@dataclass
class Config:
    """Collection of all parameters needed for the experiments.

    Attributes are grouped thematically.  Default values match the paper and
    the provided `config.yaml`, but can be overridden at construction time or
    via the `from_yaml` loader.
    """

    # ------------------------------------------------------------------
    # Global experiment settings
    # ------------------------------------------------------------------
    seed: int = 42
    """Global random seed (paper uses not specified; 42 for reproducibility)."""

    num_trials: int = 10_000
    """Number of independent Monte Carlo trials M (paper uses 10,000)."""

    # ------------------------------------------------------------------
    # Risk control parameters
    # ------------------------------------------------------------------
    alpha: float = 0.4
    """Default target risk α (used e.g. for binomial experiment)."""

    hetero_alpha: float = 0.1
    """Target miscoverage α for the synthetic heteroskedastic experiment."""

    coco_alpha: float = 0.1
    """Target false negative rate α for the MS‑COCO experiment."""

    B: float = 1.0
    """Upper bound on the loss (always 1.0 for 0‑1 and FNR losses)."""

    confidence_beta: float = 0.95
    """Confidence level β for the HPD interval (0.95)."""

    rcps_delta: float = 0.05
    """Failure probability δ for the Hoeffding‑based RCPS bound (0.05)."""

    # ------------------------------------------------------------------
    # Bayesian quadrature specifics
    # ------------------------------------------------------------------
    num_dirichlet_samples: int = 1000
    """Number of Monte Carlo draws from Dirichlet for L+ distribution."""

    # ------------------------------------------------------------------
    # Lambda grid
    # ------------------------------------------------------------------
    lambda_min: float = 0.0
    lambda_max: float = 1.0
    lambda_grid_size: int = 1001
    """Produces a step of ~0.001 between λ candidates."""

    # ------------------------------------------------------------------
    # Synthetic data: binomial
    # ------------------------------------------------------------------
    n_binomial: int = 10
    """Number of calibration points for binomial experiment."""
    K_binomial: int = 4
    """Number of trials per calibration point for binomial loss."""

    # ------------------------------------------------------------------
    # Synthetic data: heteroskedastic
    # ------------------------------------------------------------------
    n_hetero: int = 200
    """Number of calibration points for heteroskedastic experiment."""
    hetero_test_size: int = 50_000
    """Size of the test set used to estimate the true risk."""

    # ------------------------------------------------------------------
    # MS‑COCO multilabel experiment
    # ------------------------------------------------------------------
    n_coco_cal: int = 1000
    """Number of calibration images sampled from COCO validation."""
    n_coco_test: int = 3952
    """Number of test images (kept fixed per trial)."""

    coco_model_path: str = "path/to/tresnet_m_coco.pth"
    """Filesystem path to the pre‑trained TResNet‑M weights."""

    coco_data_root: str = "path/to/coco2014/val2014"
    """Directory containing COCO validation images."""

    coco_annotation_file: str = "path/to/coco2014/annotations/instances_val2014.json"
    """Path to the COCO instance annotation JSON."""

    # Optionally, a convenience flag to skip the COCO experiment if datasets are missing
    skip_coco: bool = False

    # ------------------------------------------------------------------
    # Utility paths (not in the paper, but useful for reproducibility)
    # ------------------------------------------------------------------
    output_dir: str = "results"
    """Directory where output tables/plots may be saved."""

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Convert to a dictionary (useful for logging)."""
        return {
            k: v for k, v in self.__dict__.items() if not k.startswith("__")
        }

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "Config":
        """
        Load configuration from a YAML file that follows the structure of the
        provided `config.yaml`.  Missing keys fall back to defaults.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns:
            Config instance populated from the file.

        Raises:
            ImportError: If PyYAML is not installed.
            FileNotFoundError: If the YAML file does not exist.
        """
        if yaml is None:
            raise ImportError(
                "PyYAML is required to load configuration from YAML. "
                "Install it with `pip install pyyaml`."
            )

        yaml_path = Path(yaml_path)
        if not yaml_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        # Extract nested values with safe fallbacks
        exp = data.get("experiment", {})
        risk = data.get("risk_control", {})
        bq = data.get("bayesian_quadrature", {})
        lgrid = data.get("lambda_grid", {})
        binsyn = data.get("binomial_synthetic", {})
        hetero = data.get("heteroskedastic_synthetic", {})
        coco = data.get("coco_multilabel", {})

        # Build the Config instance
        cfg = cls(
            # Global
            seed=exp.get("random_seed", cls.seed),
            num_trials=exp.get("num_trials", cls.num_trials),
            # Risk control
            alpha=risk.get("target_risk_alpha", cls.alpha),
            B=risk.get("upper_bound_B", cls.B),
            confidence_beta=risk.get("confidence_beta", cls.confidence_beta),
            rcps_delta=risk.get("rcps_delta", cls.rcps_delta),
            # Bayesian quadrature
            num_dirichlet_samples=bq.get(
                "num_dirichlet_samples", cls.num_dirichlet_samples
            ),
            # Lambda grid
            lambda_min=lgrid.get("min", cls.lambda_min),
            lambda_max=lgrid.get("max", cls.lambda_max),
            lambda_grid_size=lgrid.get("size", cls.lambda_grid_size),
            # Binomial synthetic
            n_binomial=binsyn.get("n_calibration", cls.n_binomial),
            K_binomial=binsyn.get("K", cls.K_binomial),
            # Heteroskedastic synthetic
            n_hetero=hetero.get("n_calibration", cls.n_hetero),
            hetero_test_size=hetero.get("n_test", cls.hetero_test_size),
            # COCO
            coco_alpha=coco.get("target_alpha", cls.coco_alpha),
            n_coco_cal=coco.get("n_calibration", cls.n_coco_cal),
            n_coco_test=coco.get("n_test", cls.n_coco_test),
            coco_model_path=coco.get("model_path", cls.coco_model_path),
            coco_data_root=coco.get("data_root", cls.coco_data_root),
            coco_annotation_file=coco.get(
                "annotation_file", cls.coco_annotation_file
            ),
        )
        return cfg


# For backwards compatibility: alias load_config
load_config = Config.from_yaml
