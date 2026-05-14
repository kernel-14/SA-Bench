"""
config.py
=========
Immutable dataclasses for all experimental configuration, plus a factory function
``load_config`` that parses a YAML file, merges PDE‑specific data fields, resolves
derived values (e.g. training epochs from regime), and returns a fully validated
``Config`` instance.

All other modules import the configuration classes from here, ensuring a single
source of truth and type‑safe access to every hyperparameter.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

# ---------------------------------------------------------------------------
# Supported constants
# ---------------------------------------------------------------------------
SUPPORTED_PDES: List[str] = [
    "burgers",
    "hyper_diffusion",
    "ks_conservative",
    "advection_2d",
]

SUPPORTED_METHODS: List[str] = [
    "input_perturbation",
    "ensemble",
    "sample_iso",
    "luno_iso",
    "sample_la",
    "luno_la",
]

logger = logging.getLogger(__name__)


# ===================================================================
# Dataclass definitions
# ===================================================================

@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    """Top‑level experiment metadata."""

    name: str = "luno_reproduction"
    seed: int = 42
    pde: str = "burgers"  # one of SUPPORTED_PDES
    data_regime: str = "low_data"  # "low_data" or "ood"
    out_of_distribution: Optional[str] = None  # None, "flip", "pos", ...
    methods: List[str] = dataclasses.field(
        default_factory=lambda: [
            "input_perturbation",
            "ensemble",
            "sample_iso",
            "luno_iso",
            "sample_la",
            "luno_la",
        ]
    )
    rollout_eval: bool = False
    rollout_steps: int = 50
    rollout_trajectories: int = 50

    def __post_init__(self) -> None:
        if self.pde not in SUPPORTED_PDES:
            raise ValueError(
                f"Unknown PDE '{self.pde}'. Supported: {SUPPORTED_PDES}"
            )
        if self.data_regime not in ("low_data", "ood"):
            raise ValueError(
                f"`data_regime` must be 'low_data' or 'ood', got '{self.data_regime}'"
            )
        for method in self.methods:
            if method not in SUPPORTED_METHODS:
                raise ValueError(
                    f"Unknown UQ method '{method}'. Supported: {SUPPORTED_METHODS}"
                )
        if self.out_of_distribution is not None and self.data_regime != "ood":
            raise ValueError(
                "`out_of_distribution` can only be set when `data_regime` is 'ood'."
            )


@dataclasses.dataclass(frozen=True)
class DataConfig:
    """
    Data generation / loading parameters.

    PDE‑specific fields (spatial_res, time_steps, domain_size, ...) are populated
    only for the chosen PDE; other fields remain None.
    """

    # ---- Common ----
    batch_size: int = 8
    num_workers: int = 0
    data_dir: str = "./data"
    use_apebench: bool = True

    # ---- PDE‑specific: populated depending on experiment.pde ----
    spatial_res: Union[int, Tuple[int, int]] = 256
    time_steps: int = 59
    domain_size: Union[float, Tuple[float, float]] = 6.283185307
    input_time_window: int = 10
    train_traj: int = 25
    val_traj: int = 250
    test_traj: int = 250
    dt: Optional[float] = None
    viscosity: Optional[float] = None
    diffusion_coef: Optional[float] = None
    solver: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """FNO architecture hyperparameters."""

    modes: int = 12
    hidden_dim: int = 18
    num_blocks: int = 4
    activation: str = "gelu"
    norms: Optional[str] = None
    use_bias: bool = True


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    """
    Training loop hyperparameters.

    Notes
    -----
    ``epochs`` is resolved at load time from the regime‑specific YAML entries
    (``training.epochs.low_data`` / ``training.epochs.ood``).
    """

    epochs: int = 100  # will be overridden during loading
    optimizer: str = "adamw"
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    lr_schedule: str = "cosine_decay"
    warmup_epochs: int = 5
    loss: str = "mse"
    clip_grad_norm: Optional[float] = None
    checkpoint_dir: str = "./checkpoints"
    log_interval: int = 10

    def __post_init__(self) -> None:
        if self.warmup_epochs >= self.epochs:
            logger.warning(
                "warmup_epochs (%d) >= epochs (%d). Setting warmup_epochs to %d.",
                self.warmup_epochs,
                self.epochs,
                max(0, self.epochs - 1),
            )
            object.__setattr__(
                self, "warmup_epochs", max(0, self.epochs - 1)
            )


@dataclasses.dataclass(frozen=True)
class LaplaceConfig:
    """Settings for the low‑rank Laplace approximation."""

    rank: int = 500
    ggn_batch_size: int = 1000
    use_all_data_for_ggn: bool = True

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError(f"Laplace rank must be positive, got {self.rank}")
        if self.ggn_batch_size <= 0:
            raise ValueError(
                f"ggn_batch_size must be positive, got {self.ggn_batch_size}"
            )


@dataclasses.dataclass(frozen=True)
class CalibrationConfig:
    """Grid‑search calibration parameters."""

    grid_size: int = 500
    sigma_range: Tuple[float, float] = (-5.0, 2.0)  # log10 min, max
    input_perturb_sigma_range: Tuple[float, float] = (-4.0, -1.0)

    def __post_init__(self) -> None:
        if self.grid_size <= 0:
            raise ValueError(f"grid_size must be positive, got {self.grid_size}")
        if len(self.sigma_range) != 2 or self.sigma_range[0] >= self.sigma_range[1]:
            raise ValueError(
                f"Invalid sigma_range: {self.sigma_range}. Must be [low, high] in log10 space."
            )
        if (
            len(self.input_perturb_sigma_range) != 2
            or self.input_perturb_sigma_range[0] >= self.input_perturb_sigma_range[1]
        ):
            raise ValueError(
                f"Invalid input_perturb_sigma_range: {self.input_perturb_sigma_range}."
            )


@dataclasses.dataclass(frozen=True)
class PushForwardConfig:
    """Settings for the Jacobian computation during push‑forward."""

    compute_jacobian_batch_size: int = 1
    variance_diagonal_only: bool = True


@dataclasses.dataclass(frozen=True)
class InputPerturbConfig:
    """Simple wrapper for input perturbation settings."""

    num_samples: int = 200


@dataclasses.dataclass(frozen=True)
class EnsembleConfig:
    """Simple wrapper for deep ensemble settings."""

    num_models: int = 10


@dataclasses.dataclass(frozen=True)
class UQConfig:
    """Holder for all uncertainty quantification parameters."""

    laplace: LaplaceConfig = dataclasses.field(default_factory=LaplaceConfig)
    num_samples: int = 200
    input_perturb: InputPerturbConfig = dataclasses.field(
        default_factory=InputPerturbConfig
    )
    ensemble: EnsembleConfig = dataclasses.field(default_factory=EnsembleConfig)
    calibration: CalibrationConfig = dataclasses.field(
        default_factory=CalibrationConfig
    )
    push_forward: PushForwardConfig = dataclasses.field(
        default_factory=PushForwardConfig
    )

    def __post_init__(self) -> None:
        if self.num_samples <= 0:
            raise ValueError(
                f"num_samples must be positive, got {self.num_samples}"
            )


@dataclasses.dataclass(frozen=True)
class Config:
    """
    Master configuration aggregating all sub‑configs.

    Obtain an instance via ``load_config("config.yaml")`` — never construct
    manually unless you know exactly what you are doing.
    """

    experiment: ExperimentConfig = dataclasses.field(
        default_factory=ExperimentConfig
    )
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    training: TrainConfig = dataclasses.field(default_factory=TrainConfig)
    uq: UQConfig = dataclasses.field(default_factory=UQConfig)


# ===================================================================
# Helper functions
# ===================================================================

def get_supported_pdes() -> List[str]:
    """Return the list of valid PDE identifiers."""
    return list(SUPPORTED_PDES)


def get_supported_methods() -> List[str]:
    """Return the list of valid UQ method identifiers."""
    return list(SUPPORTED_METHODS)


def create_default_config(pde: str = "burgers", regime: str = "low_data") -> Config:
    """
    Build a fully default ``Config`` for quick prototyping / testing.

    Parameters
    ----------
    pde : str
        One of ``SUPPORTED_PDES``.
    regime : str
        ``"low_data"`` or ``"ood"``.

    Returns
    -------
    Config
        A configuration with sensible defaults (mirrors ``config.yaml``).
    """
    # Start from defaults but override the minimal fields.
    exp = ExperimentConfig(pde=pde, data_regime=regime)
    # Resolve epochs
    epochs = _resolve_epochs(regime, {"low_data": 100, "ood": 1000})
    train = TrainConfig(epochs=epochs)
    return Config(experiment=exp, data=DataConfig(), model=ModelConfig(),
                  training=train, uq=UQConfig())


def save_config(config: Config, path: Union[str, Path]) -> None:
    """
    Serialise a validated ``Config`` to a YAML file (useful for logging /
    reproducing an exact run).

    Parameters
    ----------
    config : Config
        The configuration to save.
    path : str or Path
        Destination file.
    """
    # dataclasses.asdict with deep copy
    raw = _config_to_dict(config)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(raw, f, default_flow_style=False, sort_keys=False)
    logger.info("Saved configuration to %s", path)


def _config_to_dict(config: Config) -> Dict[str, Any]:
    """Recursively convert Config (and nested dataclasses) to a plain dict."""
    # We use dataclasses.asdict for simplicity. Frozen dataclasses are supported.
    return _dataclass_to_dict(config)


def _dataclass_to_dict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        result = {}
        for field in dataclasses.fields(obj):
            value = getattr(obj, field.name)
            result[field.name] = _dataclass_to_dict(value)
        return result
    if isinstance(obj, tuple):
        # Distinguish tuple from list: YAML will dump as list anyway, but we
        # preserve the hint that it was a tuple by using list (YAML doesn't
        # have tuple concept). The downstream code can convert back.
        return list(obj)
    if isinstance(obj, list):
        return [_dataclass_to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    return obj


# ===================================================================
# YAML loading and merging
# ===================================================================


def load_config(yaml_path: Union[str, Path]) -> Config:
    """
    Parse ``config.yaml`` and return a fully populated, validated ``Config``.

    Parameters
    ----------
    yaml_path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    Config
        Immutable configuration object.
    """
    with open(yaml_path, "r") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    # 1. Experiment
    exp_raw = raw.get("experiment", {})
    experiment = ExperimentConfig(
        name=exp_raw.get("name", "luno_reproduction"),
        seed=int(exp_raw.get("seed", 42)),
        pde=str(exp_raw.get("pde", "burgers")),
        data_regime=str(exp_raw.get("data_regime", "low_data")),
        out_of_distribution=exp_raw.get("out_of_distribution"),
        methods=exp_raw.get("methods", SUPPORTED_METHODS),
        rollout_eval=bool(exp_raw.get("rollout_eval", False)),
        rollout_steps=int(exp_raw.get("rollout_steps", 50)),
        rollout_trajectories=int(exp_raw.get("rollout_trajectories", 50)),
    )

    # 2. Data
    data_raw = raw.get("data", {})
    pde_data_raw = data_raw.get(experiment.pde, {})
    # Merge common top-level keys with PDE-specific dict (PDE-specific wins)
    common_data = {
        k: v
        for k, v in data_raw.items()
        if k not in SUPPORTED_PDES and not isinstance(v, dict)
    }
    merged_data = {**common_data, **pde_data_raw}
    # Convert spatial_res and domain_size to appropriate types
    spatial_res = merged_data.get("spatial_res", 256)
    if isinstance(spatial_res, list):
        spatial_res = tuple(spatial_res)
    domain_size = merged_data.get("domain_size", 6.283185307)
    if isinstance(domain_size, list):
        domain_size = tuple(domain_size)

    data_config = DataConfig(
        batch_size=int(merged_data.get("batch_size", 8)),
        num_workers=int(merged_data.get("num_workers", 0)),
        data_dir=str(merged_data.get("data_dir", "./data")),
        use_apebench=bool(merged_data.get("use_apebench", True)),
        spatial_res=spatial_res,
        time_steps=int(merged_data.get("time_steps", 59)),
        domain_size=domain_size,
        input_time_window=int(merged_data.get("input_time_window", 10)),
        train_traj=int(merged_data.get("train_traj", 25)),
        val_traj=int(merged_data.get("val_traj", 250)),
        test_traj=int(merged_data.get("test_traj", 250)),
        dt=merged_data.get("dt"),
        viscosity=merged_data.get("viscosity"),
        diffusion_coef=merged_data.get("diffusion_coef"),
        solver=merged_data.get("solver"),
    )

    # 3. Model
    model_raw = raw.get("model", {})
    model_config = ModelConfig(
        modes=int(model_raw.get("modes", 12)),
        hidden_dim=int(model_raw.get("hidden_dim", 18)),
        num_blocks=int(model_raw.get("num_blocks", 4)),
        activation=str(model_raw.get("activation", "gelu")),
        norms=model_raw.get("norms"),
        use_bias=bool(model_raw.get("use_bias", True)),
    )

    # 4. Training
    train_raw = raw.get("training", {})
    # Resolve epochs from regime
    epochs_dict = train_raw.get("epochs", {})
    if isinstance(epochs_dict, dict):
        epochs = _resolve_epochs(experiment.data_regime, epochs_dict)
    else:
        # Backward compatibility: scalar epochs
        epochs = int(epochs_dict)

    train_config = TrainConfig(
        epochs=epochs,
        optimizer=str(train_raw.get("optimizer", "adamw")),
        learning_rate=float(train_raw.get("learning_rate", 1.0e-3)),
        weight_decay=float(train_raw.get("weight_decay", 1.0e-4)),
        lr_schedule=str(train_raw.get("lr_schedule", "cosine_decay")),
        warmup_epochs=int(train_raw.get("warmup_epochs", 5)),
        loss=str(train_raw.get("loss", "mse")),
        clip_grad_norm=train_raw.get("clip_grad_norm"),
        checkpoint_dir=str(train_raw.get("checkpoint_dir", "./checkpoints")),
        log_interval=int(train_raw.get("log_interval", 10)),
    )

    # 5. UQ
    uq_raw = raw.get("uq", {})
    laplace_raw = uq_raw.get("laplace", {})
    laplace_config = LaplaceConfig(
        rank=int(laplace_raw.get("rank", 500)),
        ggn_batch_size=int(laplace_raw.get("ggn_batch_size", 1000)),
        use_all_data_for_ggn=bool(laplace_raw.get("use_all_data_for_ggn", True)),
    )

    calib_raw = uq_raw.get("calibration", {})
    sigma_range = tuple(calib_raw.get("sigma_range", [-5.0, 2.0]))
    input_perturb_range = tuple(
        calib_raw.get("input_perturb_sigma_range", [-4.0, -1.0])
    )
    calibration_config = CalibrationConfig(
        grid_size=int(calib_raw.get("grid_size", 500)),
        sigma_range=sigma_range,  # type: ignore[arg-type]
        input_perturb_sigma_range=input_perturb_range,  # type: ignore[arg-type]
    )

    push_raw = uq_raw.get("push_forward", {})
    push_forward_config = PushForwardConfig(
        compute_jacobian_batch_size=int(
            push_raw.get("compute_jacobian_batch_size", 1)
        ),
        variance_diagonal_only=bool(
            push_raw.get("variance_diagonal_only", True)
        ),
    )

    input_perturb_raw = uq_raw.get("input_perturb", {})
    input_perturb_config = InputPerturbConfig(
        num_samples=int(input_perturb_raw.get("num_samples", 200))
    )

    ensemble_raw = uq_raw.get("ensemble", {})
    ensemble_config = EnsembleConfig(
        num_models=int(ensemble_raw.get("num_models", 10))
    )

    uq_config = UQConfig(
        laplace=laplace_config,
        num_samples=int(uq_raw.get("num_samples", 200)),
        input_perturb=input_perturb_config,
        ensemble=ensemble_config,
        calibration=calibration_config,
        push_forward=push_forward_config,
    )

    # 6. Assemble master config
    config = Config(
        experiment=experiment,
        data=data_config,
        model=model_config,
        training=train_config,
        uq=uq_config,
    )

    # Final validation
    _validate_config(config)
    logger.info("Configuration loaded successfully from %s", yaml_path)
    return config


# -------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------


def _resolve_epochs(regime: str, epochs_dict: Dict[str, Any]) -> int:
    """Map the data regime to the corresponding epoch count."""
    if regime == "low_data":
        key = "low_data"
    elif regime == "ood":
        key = "ood"
    else:
        raise ValueError(f"Unknown data regime '{regime}'")
    if key not in epochs_dict:
        raise KeyError(
            f"Epochs for regime '{regime}' not found in training.epochs. "
            f"Available keys: {list(epochs_dict.keys())}"
        )
    return int(epochs_dict[key])


def _validate_config(config: Config) -> None:
    """Perform additional cross‑field consistency checks."""
    # If out_of_distribution is set, ensure data_regime is ood
    if config.experiment.out_of_distribution is not None:
        if config.experiment.data_regime != "ood":
            raise ValueError(
                "`out_of_distribution` requires `data_regime` to be 'ood'."
            )

    # Ensure rollout parameters are positive if rollout is enabled
    if config.experiment.rollout_eval:
        if config.experiment.rollout_steps <= 0:
            raise ValueError("rollout_steps must be > 0")
        if config.experiment.rollout_trajectories <= 0:
            raise ValueError("rollout_trajectories must be > 0")

    # Check PDE-specific required fields (just warnings, not errors)
    if config.experiment.pde == "burgers" and config.data.viscosity is None:
        logger.warning("viscosity not set for Burgers equation; using default?")
    if config.experiment.pde == "advection_2d" and config.data.diffusion_coef is None:
        logger.warning("diffusion_coef not set for advection-diffusion; using default?")
