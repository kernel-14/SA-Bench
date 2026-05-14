## Code: experiments/exp1_out_of_sample.py

```python
## experiments/exp1_out_of_sample.py
"""
Experiment 1: Out-of-sample parameter values scenario.

Reproduces the first experimental scenario from:
  "Towards Universal Neural Operators through Multiphysics Pretraining"

The experiment validates that adapter-based pretraining on multiple physics
problems (Burgers nu=0.01, Gray-Scott F=0.035/k=0.065, Navier-Stokes Re=100)
enables efficient transfer to out-of-sample parameter regimes (Burgers nu=0.001)
by freezing the shared backbone and training only new adapter parameters.

Results should match Table 1 in the paper:
  - MambaFNO (pretr.):  MSE=1.009e-7, NMAE=0.0120%, Avg.epoch=21.91s
  - MambaFNO (scratch): MSE=1.193e-7, NMAE=0.0213%, Avg.epoch=40.14s
  - FNO (scratch):      MSE=1.774e-7, NMAE=0.0204%, Avg.epoch=7.44s
  - Perceiver (pretr.): MSE=1.425e-7, NMAE=0.0169%, Avg.epoch=3.21s
  - CoDA-NO (pretr.):   MSE=2.881e-7, NMAE=0.0343%, Avg.epoch=62.91s

Design contract (Data structures and interfaces):
  ExperimentBase (abstract base):
    config: Config
    device: torch.device
    _logger: Logger
    setup_data() -> None
    setup_model() -> None
    run() -> Dict[str, Any]
    _build_backbone(config: PretrainConfig) -> Module
    _build_adapter_framework(backbone: Module, datasets: Dict) -> AdapterFramework

  Exp1OutOfSample(ExperimentBase):
    pretrain_datasets: Dict[str, Dataset]
    finetune_dataset: Dataset
    test_dataset: Dataset
    model: AdapterFramework
    setup_data() -> None
    setup_model() -> None
    run() -> Dict[str, Any]
    _run_pretrain() -> None
    _run_finetune() -> None
    _run_scratch_baseline() -> None
    _evaluate_all() -> Dict[str, Any]

Config alignment (config.yaml):
  experiment.name: "multiphysics_neural_operators"
  experiments.exp1_out_of_sample.pretrain_physics: [burgers_nu0p01, gray_scott_pretrain, ns_Re100]
  experiments.exp1_out_of_sample.finetune_physics: "burgers_nu0p001"
  experiments.exp1_out_of_sample.models_to_run: [mamba_fno, perceiver_no, fno, swin_v2, coda_no]
  data.burgers.filename_template: "1D_Burgers_Sols_Nu{nu}.hdf5"
  data.gray_scott.pretrain_params.{F, k, Du, Dv}
  data.navier_stokes.filename_template: "2D_NS_Sols_Re{Re}.hdf5"
  training.pretrain.{lr, batch_size, n_epochs, scheduler, checkpoint_dir}
  training.finetune.{lr, batch_size, n_epochs, freeze_backbone, checkpoint_dir}
  evaluation.{metrics, output_dir, epoch_time_n_runs, epoch_time_warmup}

Dependencies:
  utils/config.py              -> Config, PretrainConfig, FinetuneConfig, EvalConfig
  utils/logging_utils.py       -> get_logger, ResultsTable
  data/pdebench_loader.py      -> PDEBenchDataset
  data/gray_scott_generator.py -> GrayScottDataset
  data/multiphysics_dataset.py -> MultiPhysicsDataset
  models/fno_backbone.py       -> FNOBackbone
  models/mamba_fno.py          -> MambaFNO (optional, requires CUDA)
  models/perceiver_no.py       -> PerceiverNO
  models/coda_no.py            -> CodaNO
  models/adapter_framework.py  -> AdapterFramework
  training/pretrain.py         -> Pretrainer
  training/finetune.py         -> Finetuner
  evaluation/evaluator.py      -> Evaluator
"""

from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from data.gray_scott_generator import GrayScottDataset
from data.multiphysics_dataset import MultiPhysicsDataset
from data.pdebench_loader import PDEBenchDataset
from evaluation.evaluator import Evaluator
from models.adapter_framework import AdapterFramework
from models.coda_no import CodaNO
from models.fno_backbone import FNOBackbone
from models.perceiver_no import PerceiverNO
from training.finetune import Finetuner
from training.pretrain import Pretrainer
from utils.config import Config, EvalConfig, FinetuneConfig, PretrainConfig
from utils.logging_utils import ResultsTable, get_logger

# ---------------------------------------------------------------------------
# Conditional MambaFNO import (requires CUDA + mamba_ssm)
# ---------------------------------------------------------------------------

try:
    from models.mamba_fno import MambaFNO

    _MAMBA_AVAILABLE: bool = True
except ImportError:
    MambaFNO = None  # type: ignore[assignment, misc]
    _MAMBA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

_logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Physics ID strings for Experiment 1 (no dots — use 'p' for decimal point).
# These must match config.yaml experiments.exp1_out_of_sample.pretrain_physics
# and experiments.exp1_out_of_sample.finetune_physics.
_PHYSICS_BURGERS_PRETRAIN: str = "burgers_nu0p01"
_PHYSICS_GRAY_SCOTT_PRETRAIN: str = "gray_scott_pretrain"
_PHYSICS_NS_PRETRAIN: str = "ns_Re100"
_PHYSICS_BURGERS_FINETUNE: str = "burgers_nu0p001"

# Checkpoint filenames
_PRETRAIN_BEST_CKPT: str = "pretrain_best.pt"
_FINETUNE_BEST_CKPT: str = "finetune_burgers_nu0p001_best.pt"
_SCRATCH_BEST_CKPT: str = "scratch_burgers_nu0p001_best.pt"

# Results filenames
_RESULTS_CSV: str = "exp1_results.csv"
_RESULTS_JSON: str = "exp1_results.json"

# Default n_dims for 1D physics (Burgers, Advection)
_N_DIMS_1D: int = 1
# Default n_dims for 2D physics (NS, Gray-Scott, Heat, RD)
_N_DIMS_2D: int = 2


# ---------------------------------------------------------------------------
# ExperimentBase
# ---------------------------------------------------------------------------


class ExperimentBase:
    """Abstract base class for all three experiments.

    Provides shared infrastructure: config loading, device setup, logging,
    backbone construction, and adapter framework construction.

    Subclasses must implement:
      - setup_data() -> None
      - setup_model() -> None
      - run() -> Dict[str, Any]

    Attributes:
        config: Fully populated Config instance loaded from YAML.
        device: Target torch.device (CPU or CUDA).
        _logger: Instance-specific logger.
    """

    def __init__(self, config_path: str) -> None:
        """Initialise ExperimentBase.

        Loads the Config from the YAML file at config_path, resolves the
        target device, and sets up the instance logger.

        Args:
            config_path: Path to the YAML configuration file. Must exist.

        Raises:
            FileNotFoundError: If config_path does not exist.
            ValueError: If the config contains invalid values.
        """
        # ── Load configuration ────────────────────────────────────────────
        self.config: Config = Config.from_yaml(config_path)

        # ── Resolve device ────────────────────────────────────────────────
        device_str: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device: torch.device = torch.device(device_str)

        # ── Logger ────────────────────────────────────────────────────────
        os.makedirs(self.config.eval.output_dir, exist_ok=True)
        log_file: str = os.path.join(
            self.config.eval.output_dir,
            f"{self.config.experiment_name}.log",
        )
        self._logger: logging.Logger = get_logger(
            self.__class__.__name__,
            log_file=log_file,
        )

        self._logger.info(
            "%s initialized: config='%s', device='%s', model_type='%s'.",
            self.__class__.__name__,
            config_path,
            str(self.device),
            self.config.model_type,
        )

    def setup_data(self) -> None:
        """Load and prepare datasets. Must be implemented by subclasses."""
        raise NotImplementedError(
            f"{self.__class__.__name__}.setup_data() must be implemented."
        )

    def setup_model(self) -> None:
        """Construct backbone and adapter framework. Must be implemented by subclasses."""
        raise NotImplementedError(
            f"{self.__class__.__name__}.setup_model() must be implemented."
        )

    def run(self) -> Dict[str, Any]:
        """Execute the full experiment. Must be implemented by subclasses."""
        raise NotImplementedError(
            f"{self.__class__.__name__}.run() must be implemented."
        )

    def _build_backbone(self, pretrain_config: PretrainConfig) -> nn.Module:
        """Construct the backbone module based on config.model_type.

        Reads architecture hyperparameters from the model-specific section
        of config.yaml (models.<model_type>.*) and instantiates the
        appropriate backbone class.

        Supported model types:
          - 'fno'         -> FNOBackbone
          - 'mamba_fno'   -> MambaFNO (requires CUDA + mamba_ssm)
          - 'perceiver_no' -> PerceiverNO
          - 'coda_no'     -> CodaNO

        The n_dims parameter is set to 2 by default (2D problems dominate
        in Experiment 1). For mixed 1D/2D pretraining, 1D data is treated
        as 2D with height=1 (standard FNO trick).

        Args:
            pretrain_config: PretrainConfig with hidden_dim, n_modes, n_layers
                populated from config.yaml models.<model_type>.* section.

        Returns:
            Instantiated backbone nn.Module.

        Raises:
            ValueError: If config.model_type is not supported.
            ImportError: If 'mamba_fno' is requested but mamba_ssm is not
                installed.
        """
        model_type: str = self.config.model_type
        hidden_dim: int = pretrain_config.hidden_dim
        n_modes: int = pretrain_config.n_modes
        n_layers: int = pretrain_config.n_layers

        # Read model-specific config from the raw YAML via config.to_dict()
        # to access nested model parameters not in PretrainConfig.
        config_dict: Dict[str, Any] = self.config.to_dict()
        # models section is not directly in Config dataclass; read from
        # the YAML-derived dict if available, otherwise use defaults.
        # Since Config.from_yaml merges model params into PretrainConfig,
        # we use pretrain_config for the core dims and fall back to defaults
        # for model-specific params.

        self._logger.info(
            "_build_backbone: model_type='%s', hidden_dim=%d, "
            "n_modes=%d, n_layers=%d.",
            model_type,
            hidden_dim,
            n_modes,
            n_layers,
        )

        if model_type == "fno":
            backbone: nn.Module = FNOBackbone(
                hidden_dim=hidden_dim,
                n_modes=n_modes,
                n_layers=n_layers,
                n_dims=_N_DIMS_2D,
                activation="gelu",
            )

        elif model_type == "mamba_fno":
            if not _MAMBA_AVAILABLE or MambaFNO is None:
                raise ImportError(
                    "MambaFNO requires mamba_ssm and CUDA. "
                    "Install with: pip install mamba-ssm causal-conv1d\n"
                    "Alternatively, use --model fno, perceiver_no, or coda_no."
                )
            backbone = MambaFNO(
                hidden_dim=hidden_dim,
                n_modes=n_modes,
                n_layers=n_layers,
                d_state=16,   # config.yaml models.mamba_fno.mamba.d_state
                d_conv=4,     # config.yaml models.mamba_fno.mamba.d_conv
                expand=2,     # config.yaml models.mamba_fno.mamba.expand
                n_dims=_N_DIMS_2D,
                activation="gelu",
            )

        elif model_type == "perceiver_no":
            backbone = PerceiverNO(
                hidden_dim=hidden_dim,
                latent_dim=256,   # config.yaml models.perceiver_no.perceiver.latent_dim
                n_latents=64,     # config.yaml models.perceiver_no.perceiver.n_latents
                n_heads=8,        # config.yaml models.perceiver_no.perceiver.n_heads
                n_blocks=n_layers,  # use n_layers as n_blocks
            )

        elif model_type == "coda_no":
            backbone = CodaNO(
                hidden_dim=hidden_dim,
                n_modes=n_modes,
                n_layers=n_layers,
                n_heads=8,        # config.yaml models.coda_no.n_heads
                n_dims=_N_DIMS_2D,
            )

        else:
            raise ValueError(
                f"Unsupported model_type='{model_type}'. "
                f"Supported: 'fno', 'mamba_fno', 'perceiver_no', 'coda_no'."
            )

        n_params: int = sum(p.numel() for p in backbone.parameters())
        self._logger.info(
            "_build_backbone: created %s with %d parameters.",
            type(backbone).__name__,
            n_params,
        )

        return backbone

    def _build_adapter_framework(
        self,
        backbone: nn.Module,
        datasets: Dict[str, Dataset],
    ) -> AdapterFramework:
        """Construct AdapterFramework and register adapters for all physics.

        Creates an AdapterFramework wrapping the given backbone, then
        registers a LiftingAdapter + ProjectionAdapter pair for each
        physics dataset in the provided dict.

        Args:
            backbone: Shared backbone module (FNOBackbone, MambaFNO, etc.).
            datasets: Dict mapping physics_id -> Dataset. Each dataset must
                implement get_n_in() and get_n_out(). Used to determine
                the adapter input/output channel counts.

        Returns:
            AdapterFramework with all adapters registered and ready for
            pretraining.
        """
        hidden_dim: int = self.config.pretrain.hidden_dim

        framework: AdapterFramework = AdapterFramework(
            backbone=backbone,
            hidden_dim=hidden_dim,
        )

        for physics_id, dataset in datasets.items():
            n_in: int = dataset.get_n_in()   # type: ignore[attr-defined]
            n_out: int = dataset.get_n_out()  # type: ignore[attr-defined]
            framework.register_adapter(physics_id, n_in=n_in, n_out=n_out)
            self._logger.debug(
                "_build_adapter_framework: registered adapter for "
                "physics_id='%s', n_in=%d, n_out=%d.",
                physics_id,
                n_in,
                n_out,
            )

        return framework


# ---------------------------------------------------------------------------
# Exp1OutOfSample
# ---------------------------------------------------------------------------


class Exp1OutOfSample(ExperimentBase):
    """Experiment 1: Out-of-sample parameter values.

    Validates the adapter-based pretraining framework on the scenario where
    the pretrain and finetune equations differ only in coefficient values.

    Pretrain physics (from config.yaml experiments.exp1_out_of_sample.pretrain_physics):
      - 'burgers_nu0p01':      Burgers' equation, nu=0.01 (PDEBench)
      - 'gray_scott_pretrain': Gray-Scott RD, F=0.035/k=0.065 (generated)
      - 'ns_Re100':            Navier-Stokes, Re=100 (PDEBench)

    Finetune target (from config.yaml experiments.exp1_out_of_sample.finetune_physics):
      - 'burgers_nu0p001':     Burgers' equation, nu=0.001 (out-of-sample)

    The experiment runs four phases:
      1. Pretrain: all parameters jointly optimized on 3 physics.
      2. Finetune: backbone frozen, only new adapter trained on nu=0.001.
      3. Scratch: same architecture trained from scratch on nu=0.001.
      4. Evaluate: MSE, NMAE, epoch time, param count for both models.

    Attributes:
        pretrain_datasets: Dict[str, Dataset] — training splits for pretrain.
        pretrain_val_datasets: Dict[str, Dataset] — val splits for pretrain.
        finetune_train_dataset: Dataset — training split for finetune target.
        finetune_val_dataset: Dataset — validation split for finetune target.
        test_dataset: Dataset — test split for final evaluation.
        model: AdapterFramework — pretrained + finetuned model.
        scratch_model: AdapterFramework — trained from scratch (baseline).
        _pretrain_history: Dict[str, List[float]] — pretrain loss curves.
        _finetune_history: Dict[str, List[float]] — finetune loss curves.
        _scratch_history: Dict[str, List[float]] — scratch training loss curves.
    """

    def __init__(self, config_path: str) -> None:
        """Initialise Exp1OutOfSample.

        Calls ExperimentBase.__init__ to load config, resolve device, and
        set up logger. Dataset and model attributes are initialized to None
        and populated by setup_data() and setup_model() respectively.

        Args:
            config_path: Path to the YAML configuration file.
        """
        super().__init__(config_path)

        # ── Dataset placeholders ──────────────────────────────────────────
        self.pretrain_datasets: Dict[str, Dataset] = {}
        self.pretrain_val_datasets: Dict[str, Dataset] = {}
        self.finetune_train_dataset: Optional[Dataset] = None
        self.finetune_val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        # ── Model placeholders ────────────────────────────────────────────
        self.model: Optional[AdapterFramework] = None
        self.scratch_model: Optional[AdapterFramework] = None

        # ── Training history ──────────────────────────────────────────────
        self._pretrain_history: Dict[str, List[float]] = {}
        self._finetune_history: Dict[str, List[float]] = {}
        self._scratch_history: Dict[str, List[float]] = {}

        self._logger.info(
            "Exp1OutOfSample: initialized. "
            "Call setup_data() then setup_model() then run()."
        )

    # -----------------------------------------------------------------------
    # Public: setup_data
    # -----------------------------------------------------------------------

    def setup_data(self) -> None:
        """Load all datasets required for Experiment 1.

        Loads three pretrain physics datasets and one finetune/test dataset:
          1. Burgers nu=0.01 (PDEBench, 1D) — pretrain
          2. Gray-Scott F=0.035/k=0.065 (generated, 2D) — pretrain
          3. Navier-Stokes Re=100 (PDEBench, 2D) — pretrain
          4. Burgers nu=0.001 (PDEBench, 1D) — finetune + test

        Each physics is loaded for train, val, and test splits. The pretrain
        splits are wrapped in MultiPhysicsDataset for joint training.

        Config sources:
          data.pdebench_root: root directory for PDEBench HDF5 files
          data.burgers.filename_template: "1D_Burgers_Sols_Nu{nu}.hdf5"
          data.gray_scott.pretrain_params.{F, k, Du, Dv}
          data.gray_scott.{grid_size, n_steps, dt, n_samples}
          data.navier_stokes.filename_template: "2D_NS_Sols_Re{Re}.hdf5"
          data.{n_train, n_val, n_test, normalize}

        Raises:
            FileNotFoundError: If PDEBench HDF5 files are not found at the
                expected paths. Download from:
                https://darus.uni-stuttgart.de/dataverse/pdebench
        """
        self._logger.info("setup_data: loading datasets for Experiment 1.")

        data_cfg = self.config.data
        pdebench_root: str = data_cfg.pdebench_root
        cache_root: str = data_cfg.gray_scott_root
        n_train: int = data_cfg.n_train
        n_val: int = data_cfg.n_val
        n_test: int = data_cfg.n_test
        normalize: bool = data_cfg.normalize

        # ── 1. Burgers nu=0.01 (pretrain) ─────────────────────────────────
        burgers_pretrain_path: str = os.path.join(
            pdebench_root, "1D_Burgers_Sols_Nu0.01.hdf5"
        )
        self._logger.info(
            "Loading Burgers pretrain dataset: '%s'", burgers_pretrain_path
        )

        burgers_pretrain_train: PDEBenchDataset = PDEBenchDataset(
            physics_name=_PHYSICS_BURGERS_PRETRAIN,
            data_path=burgers_pretrain_path,
            split="train",
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            normalize=normalize,
        )
        burgers_pretrain_val: PDEBenchDataset = PDEBenchDataset(
            physics_name=_PHYSICS_BURGERS_PRETRAIN,
            data_path=burgers_pretrain_path,
            split="val",
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            normalize=normalize,
        )

        # ── 2. Gray-Scott F=0.035/k=0.065 (pretrain) ──────────────────────
        # Gray-Scott is NOT in PDEBench — generated via numerical simulation.
        # Parameters from config.yaml data.gray_scott.pretrain_params.
        gs_cache_path: str = os.path.join(cache_root, "gray_scott_pretrain.npz")
        self._logger.info(
            "Loading Gray-Scott pretrain dataset (cache: '%s').", gs_cache_path
        )

        gray_scott_train: GrayScottDataset = GrayScottDataset(
            F=0.035,    # config.yaml data.gray_scott.pretrain_params.F
            k=0.065,    # config.yaml data.gray_scott.pretrain_params.k
            Du=0.16,    # config.yaml data.gray_scott.pretrain_params.Du
            Dv=0.08,    # config.yaml data.gray_scott.pretrain_params.Dv
            n_samples=1000,   # config.yaml data.gray_scott.n_samples
            grid_size=128,    # config.yaml data.gray_scott.grid_size
            n_steps=1000,     # config.yaml data.gray_scott.n_steps
            dt=1.0,           # config.yaml data.gray_scott.dt
            split="train",
            cache_path=gs_cache_path,
            normalize=normalize,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
        )
        gray_scott_val: GrayScottDataset = GrayScottDataset(
            F=0.035,
            k=0.065,
            Du=0.16,
            Dv=0.08,
            n_samples=1000,
            grid_size=128,
            n_steps=1000,
            dt=1.0,
            split="val",
            cache_path=gs_cache_path,
            normalize=normalize,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
        )

        # ── 3. Navier-Stokes Re=100 (pretrain) ────────────────────────────
        ns_pretrain_path: str = os.path.join(
            pdebench_root, "2D_NS_Sols_Re100.hdf5"
        )
        self._logger.info(
            "Loading Navier-Stokes pretrain dataset: '%s'", ns_pretrain_path
        )

        ns_pretrain_train: PDEBenchDataset = PDEBenchDataset(
            physics_name=_PHYSICS_NS_PRETRAIN,
            data_path=ns_pretrain_path,
            split="train",
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            normalize=normalize,
        )
        ns_pretrain_val: PDEBenchDataset = PDEBenchDataset(
            physics_name=_PHYSICS_NS_PRETRAIN,
            data_path=ns_pretrain_path,
            split="val",
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            normalize=normalize,
        )

        # ── 4. Burgers nu=0.001 (finetune + test) ─────────────────────────
        burgers_finetune_path: str = os.path.join(
            pdebench_root, "1D_Burgers_Sols_Nu0.001.hdf5"
        )
        self._logger.info(
            "Loading Burgers finetune dataset: '%s'", burgers_finetune_path
        )

        self.finetune_train_dataset = PDEBenchDataset(
            physics_name=_PHYSICS_BURGERS_FINETUNE,
            data_path=burgers_finetune_path,
            split="train",
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            normalize=normalize,
        )
        self.finetune_val_dataset = PDEBenchDataset(
            physics_name=_PHYSICS_BURGERS_FINETUNE,
            data_path=burgers_finetune_path,
            split="val",
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            normalize=normalize,
        )
        self.test_dataset = PDEBenchDataset(
            physics_name=_PHYSICS_BURGERS_FINETUNE,
            data_path=burgers_finetune_path,
            split="test",
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            normalize=normalize,
        )

        # ── 5. Assemble pretrain dicts ─────────────────────────────────────
        self.pretrain_datasets = {
            _PHYSICS_BURGERS_PRETRAIN: burgers_pretrain_train,
            _PHYSICS_GRAY_SCOTT_PRETRAIN: gray_scott_train,
            _PHYSICS_NS_PRETRAIN: ns_pretrain_train,
        }
        self.pretrain_val_datasets = {
            _PHYSICS_BURGERS_PRETRAIN: burgers_pretrain_val,
            _PHYSICS_GRAY_SCOTT_PRETRAIN: gray_scott_val,
            _PHYSICS_NS_PRETRAIN: ns_pretrain_val,
        }

        self._logger.info(
            "setup_data complete: %d pretrain physics, "
            "finetune_train=%d, finetune_val=%d, test=%d samples.",
            len(self.pretrain_datasets),
            len(self.finetune_train_dataset),
            len(self.finetune_val_dataset),
            len(self.test_dataset),
        )

    # -----------------------------------------------------------------------
    # Public: setup_model
    # -----------------------------------------------------------------------

    def setup_model(self) -> None:
        """Construct the backbone and adapter framework for Experiment 1.

        Builds the backbone based on config.model_type, wraps it in an
        AdapterFramework, and registers adapters for all three pretrain
        physics. The scratch model is NOT built here — it is built fresh
        in _run_scratch_baseline() to ensure random initialization.

        Requires setup_data() to have been called first (pretrain_datasets
        must be populated to determine n_in/n_out per physics).

        Raises:
            RuntimeError: If setup_data() has not been called yet.
        """
        if not self.pretrain_datasets:
            raise RuntimeError(
                "setup_model() requires setup_data() to be called first. "
                "pretrain_datasets is empty."
            )

        self._logger.info(
            "setup_model: building %s backbone and adapter framework.",
            self.config.model_type,
        )

        # ── Build backbone ────────────────────────────────────────────────
        backbone: nn.Module = self._build_backbone(self.config.pretrain)

        # ── Build adapter framework with pretrain adapters ────────────────
        self.model = self._build_adapter_framework(
            backbone=backbone,
            datasets=self.pretrain_datasets,
        )

        # ── Log parameter counts ──────────────────────────────────────────
        param_counts: Dict[str, int] = self.model.get_param_count()
        self._logger.info(
            "setup_model complete: backbone=%d params, adapters=%d params, "
            "total=%d params.",
            param_counts["backbone"],
            param_counts["adapters"],
            param_counts["total"],
        )

    # -----------------------------------------------------------------------
    # Public: run
    # -----------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Execute the full Experiment 1 pipeline.

        Orchestrates four phases in sequence