## utils/config.py
"""
Configuration dataclasses for the multi-physics neural operator pretraining
framework described in:
  "Towards Universal Neural Operators through Multiphysics Pretraining"

Design contract (Data structures and interfaces):
  PretrainConfig  - physics_list, n_epochs, lr, batch_size, hidden_dim,
                    n_modes, n_layers, weight_decay, scheduler, checkpoint_dir
  FinetuneConfig  - target_physics, n_epochs, lr, batch_size,
                    pretrained_checkpoint, freeze_backbone
  DataConfig      - pdebench_root, gray_scott_root, heat_root,
                    n_train, n_val, n_test, normalize
  EvalConfig      - metrics, output_dir
  Config          - experiment_name, model_type, pretrain, finetune, data, eval
                    + from_yaml(path: str) -> Config
                    + to_dict() -> dict

NO imports from other project files. All other modules import from here.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

import yaml


# Supported backbone model types.
_SUPPORTED_MODELS: frozenset = frozenset(
    {"fno", "mamba_fno", "perceiver_no", "coda_no", "swin_v2"}
)


# ---------------------------------------------------------------------------
# Sub-configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PretrainConfig:
    """Configuration for the multi-physics pretraining phase.

    Architecture fields (hidden_dim, n_modes, n_layers) are sourced from
    models.{model_type} in config.yaml and merged here so that the rest of
    the codebase has a single place to read model architecture params.

    physics_list is populated from
    experiments.{experiment_name}.pretrain_physics in config.yaml.
    These strings must match the physics ID keys used in AdapterFramework's
    adapter registry and MultiPhysicsDataset (Shared Knowledge: PHYSICS_ID
    STRINGS — no dots, use underscores for decimals, e.g. 'burgers_nu0p01').

    All numeric hyperparameters are UNSPECIFIED in the paper; defaults follow
    standard FNO literature practice as documented in config.yaml.
    """

    physics_list: List[str] = field(default_factory=list)
    n_epochs: int = 200
    lr: float = 1.0e-3
    batch_size: int = 16
    hidden_dim: int = 64
    n_modes: int = 16
    n_layers: int = 4
    weight_decay: float = 1.0e-4
    scheduler: str = "cosine"
    checkpoint_dir: str = "checkpoints/pretrain"


@dataclass
class FinetuneConfig:
    """Configuration for the adapter-only fine-tuning phase.

    Key paper constraint (Section 3): during fine-tuning, theta_F (backbone)
    is frozen — only the new adapter parameters (theta_{L_ft}, theta_{P_ft})
    are trained. This is enforced by freeze_backbone=True.

    pretrained_checkpoint: path to the checkpoint saved during pretraining,
    loaded by AdapterFramework.load_checkpoint() before fine-tuning begins.
    Derived automatically from pretrain.checkpoint_dir if not set explicitly
    in the YAML.
    """

    target_physics: str = ""
    n_epochs: int = 100
    lr: float = 1.0e-4
    batch_size: int = 16
    pretrained_checkpoint: str = ""
    freeze_backbone: bool = True  # Explicitly stated in paper (Section 3)


@dataclass
class DataConfig:
    """Dataset paths and split sizes.

    pdebench_root: root directory containing PDEBench HDF5 files.
    gray_scott_root: directory for Gray-Scott cached .npz files
                     (GrayScottDataset writes here on first run).
    heat_root: directory for Heat/HeatConvection cached .npz files
               (HeatConvectionDataset writes here on first run).

    n_train / n_val / n_test: sample counts per physics dataset.
    These are UNSPECIFIED in the paper; defaults follow PDEBench conventions.

    normalize: whether to apply zero-mean unit-variance normalization
    (standard practice, UNSPECIFIED in paper).
    """

    pdebench_root: str = "data/pdebench"
    gray_scott_root: str = "data/cache"
    heat_root: str = "data/cache"
    n_train: int = 800
    n_val: int = 100
    n_test: int = 100
    normalize: bool = True


@dataclass
class EvalConfig:
    """Evaluation configuration.

    metrics: list of metric names to compute. The paper explicitly defines
    and reports 'nmae' (equation 3) and 'mse' (Tables 1 and 2).

    output_dir: where Evaluator writes results JSON and CSV files.
    """

    metrics: List[str] = field(default_factory=lambda: ["nmae", "mse"])
    output_dir: str = "results"


# ---------------------------------------------------------------------------
# Top-level Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """Top-level configuration container.

    All other modules receive a Config instance (or its sub-configs) rather
    than reading YAML directly. This is the single source of truth for all
    hyperparameters and paths.

    Attributes:
        experiment_name: Identifies the experiment. One of:
            'exp1_out_of_sample', 'exp2_input_extension',
            'exp3_multiphysics', or the generic
            'multiphysics_neural_operators'.
        model_type: Backbone architecture. One of 'fno', 'mamba_fno',
            'perceiver_no', 'coda_no', 'swin_v2'.
        pretrain: Pretraining phase configuration.
        finetune: Fine-tuning phase configuration.
        data: Dataset paths and split sizes.
        eval: Evaluation metrics and output directory.
    """

    experiment_name: str = "multiphysics_neural_operators"
    model_type: str = "mamba_fno"
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # -----------------------------------------------------------------------
    # Factory: YAML -> Config
    # -----------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load a Config from a YAML file.

        Merge strategy
        --------------
        config.yaml has a richer structure than the flat dataclasses.
        The merge rules are:

        PretrainConfig
          .physics_list   <- experiments.{exp_name}.pretrain_physics
          .n_epochs       <- training.pretrain.n_epochs
          .lr             <- training.pretrain.lr
          .batch_size     <- training.pretrain.batch_size
          .weight_decay   <- training.pretrain.weight_decay
          .scheduler      <- training.pretrain.scheduler
          .checkpoint_dir <- training.pretrain.checkpoint_dir
          .hidden_dim     <- models.{model_type}.hidden_dim
          .n_modes        <- models.{model_type}.n_modes
          .n_layers       <- models.{model_type}.n_layers

        FinetuneConfig
          .target_physics        <- experiments.{exp_name}.finetune_physics
          .n_epochs              <- training.finetune.n_epochs
          .lr                    <- training.finetune.lr
          .batch_size            <- training.finetune.batch_size
          .freeze_backbone       <- training.finetune.freeze_backbone
          .pretrained_checkpoint <- training.finetune.pretrained_checkpoint
                                    (derived as pretrain_checkpoint_dir +
                                    "/best_model.pt" if not set)

        DataConfig
          .pdebench_root   <- data.pdebench_root
          .gray_scott_root <- data.gray_scott_root  (falls back to cache_root)
          .heat_root       <- data.heat_root         (falls back to cache_root)
          .n_train         <- data.n_train
          .n_val           <- data.n_val
          .n_test          <- data.n_test
          .normalize       <- data.normalize

        EvalConfig
          .metrics    <- evaluation.metrics
          .output_dir <- evaluation.output_dir

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Fully populated Config instance.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            ValueError: If model_type is not one of the supported model types.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Config file not found: '{path}'")

        with open(path, "r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh) or {}

        # ── top-level experiment identifiers ──────────────────────────────
        exp_section: Dict[str, Any] = raw.get("experiment", {})
        experiment_name: str = str(
            exp_section.get("name", "multiphysics_neural_operators")
        )
        # model_type lives in experiment section so CLI can override it later
        # by mutating Config.model_type after construction.
        model_type: str = str(exp_section.get("model_type", "mamba_fno"))
        if model_type not in _SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. "
                f"Must be one of {sorted(_SUPPORTED_MODELS)}."
            )

        # ── model architecture params ─────────────────────────────────────
        # Architecture hyperparameters live under models.{model_type} in YAML.
        models_section: Dict[str, Any] = raw.get("models", {})
        model_cfg: Dict[str, Any] = models_section.get(model_type, {})
        hidden_dim: int = int(model_cfg.get("hidden_dim", 64))
        n_modes: int = int(model_cfg.get("n_modes", 16))
        n_layers: int = int(model_cfg.get("n_layers", 4))

        # ── training section ──────────────────────────────────────────────
        training_section: Dict[str, Any] = raw.get("training", {})
        pretrain_tr: Dict[str, Any] = training_section.get("pretrain", {})
        finetune_tr: Dict[str, Any] = training_section.get("finetune", {})

        # ── experiment-specific physics lists ─────────────────────────────
        experiments_section: Dict[str, Any] = raw.get("experiments", {})
        exp_cfg: Dict[str, Any] = experiments_section.get(experiment_name, {})
        pretrain_physics: List[str] = [
            str(p) for p in exp_cfg.get("pretrain_physics", [])
        ]
        finetune_physics: str = str(exp_cfg.get("finetune_physics", ""))

        # ── PretrainConfig ────────────────────────────────────────────────
        pretrain_checkpoint_dir: str = str(
            pretrain_tr.get("checkpoint_dir", "checkpoints/pretrain")
        )
        pretrain_cfg = PretrainConfig(
            physics_list=pretrain_physics,
            n_epochs=int(pretrain_tr.get("n_epochs", 200)),
            lr=float(pretrain_tr.get("lr", 1.0e-3)),
            batch_size=int(pretrain_tr.get("batch_size", 16)),
            hidden_dim=hidden_dim,
            n_modes=n_modes,
            n_layers=n_layers,
            weight_decay=float(pretrain_tr.get("weight_decay", 1.0e-4)),
            scheduler=str(pretrain_tr.get("scheduler", "cosine")),
            checkpoint_dir=pretrain_checkpoint_dir,
        )

        # ── FinetuneConfig ────────────────────────────────────────────────
        # Derive pretrained_checkpoint from pretrain checkpoint_dir if not set.
        default_pretrained_ckpt: str = os.path.join(
            pretrain_checkpoint_dir, "best_model.pt"
        )
        pretrained_checkpoint: str = str(
            finetune_tr.get("pretrained_checkpoint", default_pretrained_ckpt)
        )
        finetune_cfg = FinetuneConfig(
            target_physics=finetune_physics,
            n_epochs=int(finetune_tr.get("n_epochs", 100)),
            lr=float(finetune_tr.get("lr", 1.0e-4)),
            batch_size=int(finetune_tr.get("batch_size", 16)),
            pretrained_checkpoint=pretrained_checkpoint,
            freeze_backbone=bool(finetune_tr.get("freeze_backbone", True)),
        )

        # ── DataConfig ────────────────────────────────────────────────────
        data_section: Dict[str, Any] = raw.get("data", {})
        # cache_root is the fallback for both gray_scott_root and heat_root
        # when those keys are absent (config.yaml uses cache_root as the
        # shared cache directory).
        cache_root: str = str(data_section.get("cache_root", "data/cache"))
        data_cfg = DataConfig(
            pdebench_root=str(
                data_section.get("pdebench_root", "data/pdebench")
            ),
            gray_scott_root=str(
                data_section.get("gray_scott_root", cache_root)
            ),
            heat_root=str(
                data_section.get("heat_root", cache_root)
            ),
            n_train=int(data_section.get("n_train", 800)),
            n_val=int(data_section.get("n_val", 100)),
            n_test=int(data_section.get("n_test", 100)),
            normalize=bool(data_section.get("normalize", True)),
        )

        # ── EvalConfig ────────────────────────────────────────────────────
        # config.yaml uses key 'evaluation'; Config attribute is 'eval'.
        eval_section: Dict[str, Any] = raw.get("evaluation", {})
        eval_cfg = EvalConfig(
            metrics=list(eval_section.get("metrics", ["nmae", "mse"])),
            output_dir=str(eval_section.get("output_dir", "results")),
        )

        return cls(
            experiment_name=experiment_name,
            model_type=model_type,
            pretrain=pretrain_cfg,
            finetune=finetune_cfg,
            data=data_cfg,
            eval=eval_cfg,
        )

    # -----------------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert the entire Config to a plain Python dict.

        Uses dataclasses.asdict() which recursively converts nested dataclasses
        and preserves list fields. The result is JSON-serializable (all values
        are str, int, float, bool, list, or dict).

        The dict structure mirrors the Config dataclass, not the original YAML:
            {
                "experiment_name": str,
                "model_type": str,
                "pretrain": { ... PretrainConfig fields ... },
                "finetune": { ... FinetuneConfig fields ... },
                "data":     { ... DataConfig fields ... },
                "eval":     { ... EvalConfig fields ... },
            }

        Returns:
            Flat-ish dict with keys: experiment_name, model_type, pretrain,
            finetune, data, eval — where pretrain/finetune/data/eval are
            themselves dicts of their respective fields.
        """
        return asdict(self)
