## Code: experiments/exp3_multiphysics.py

```python
## experiments/exp3_multiphysics.py
"""
Experiment 3: General multi-physics pretraining and transfer.

Reproduces the third experimental scenario from:
  "Towards Universal Neural Operators through Multiphysics Pretraining"

The experiment validates the most general transfer learning scenario: the
shared FNO backbone is pretrained on two physically distinct PDEs (Advection
and Burgers from PDEBench), then the frozen backbone is transferred to
Reaction-Diffusion via a new adapter pair.

From the paper (Section 4):
    "In the final stage, we evaluated the capabilities of the developed
    methods to transfer knowledge from the dynamics of advection and
    Burgers' equation to reaction-diffusion, based on the PDEBench dataset."

Results should match Table 2 (multi-physics rows) in the paper:
  - MambaFNO (pretr.):  MSE=3.91e-6,  NMAE=0.0041%, Avg.epoch=131.2s
  - MambaFNO (scratch): MSE=4.291e-6, NMAE=0.0054%, Avg.epoch=261.1s
  - FNO (scratch):      MSE=7.286e-6, NMAE=0.0121%, Avg.epoch=41.3s
  - Perceiver (pretr.): MSE=4.107e-6, NMAE=0.0051%, Avg.epoch=20.4s
  - CoDA-NO (pretr.):   MSE=1.043e-5, NMAE=0.013%,  Avg.epoch=185.1s

Design contract (Data structures and interfaces):
  Exp3MultiPhysics(ExperimentBase):
    pretrain_datasets: Dict[str, Dataset]
    target_dataset: Dataset
    model: AdapterFramework
    setup_data() -> None
    setup_model() -> None
    run() -> Dict[str, Any]
    _run_multiphysics_pretrain() -> None
    _run_finetune_target() -> None
    _run_scratch_baseline() -> None

Config alignment (config.yaml / configs/exp3_multiphysics.yaml):
  experiments.exp3_multiphysics.pretrain_physics: [advection_beta0p1, burgers_nu0p01]
  experiments.exp3_multiphysics.finetune_physics: "reaction_diffusion"
  data.advection.filename_template: "1D_Advection_Sols_beta{beta}.hdf5"
  data.burgers.filename_template: "1D_Burgers_Sols_Nu{nu}.hdf5"
  data.reaction_diffusion.filename_template: "2D_DiffReact_Sols_Nu{nu}.hdf5"
  data.reaction_diffusion.finetune_params[0].nu: "2.0"
  data.{pdebench_root, n_train, n_val, n_test, normalize}
  training.pretrain.{lr, batch_size, n_epochs, scheduler, checkpoint_dir}
  training.finetune.{lr, batch_size, n_epochs, freeze_backbone}
  evaluation.{metrics, output_dir, epoch_time_n_runs, epoch_time_warmup}

Dependencies:
  utils/config.py              -> Config, FinetuneConfig
  utils/logging_utils.py       -> get_logger, ResultsTable
  data/pdebench_loader.py      -> PDEBenchDataset
  data/multiphysics_dataset.py -> MultiPhysicsDataset
  models/adapter_framework.py  -> AdapterFramework
  models/fno_backbone.py       -> FNOBackbone
  models/perceiver_no.py       -> PerceiverNO
  models/coda_no.py            -> CodaNO
  training/pretrain.py         -> Pretrainer
  training/finetune.py         -> Finetuner
  evaluation/evaluator.py      -> Evaluator
  experiments/exp1_out_of_sample.py -> ExperimentBase
"""

from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset

from data.multiphysics_dataset import MultiPhysicsDataset
from data.pdebench_loader import PDEBenchDataset
from evaluation.evaluator import Evaluator
from experiments.exp1_out_of_sample import ExperimentBase
from models.adapter_framework import AdapterFramework
from models.coda_no import CodaNO
from models.fno_backbone import FNOBackbone
from models.perceiver_no import PerceiverNO
from training.finetune import Finetuner
from training.pretrain import Pretrainer
from utils.config import Config, FinetuneConfig, PretrainConfig
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
# Physics ID string constants (no dots — use 'p' for decimal point)
# These must match config.yaml experiments.exp3_multiphysics.pretrain_physics
# and experiments.exp3_multiphysics.finetune_physics.
# ---------------------------------------------------------------------------

_PHYSICS_ADVECTION_PRETRAIN: str = "advection_beta0p1"
_PHYSICS_BURGERS_PRETRAIN: str = "burgers_nu0p01"
_PHYSICS_REACTION_DIFFUSION: str = "reaction_diffusion"

# Default PDEBench filename templates (from config.yaml data section)
_DEFAULT_ADVECTION_FILENAME: str = "1D_Advection_Sols_beta0.1.hdf5"
_DEFAULT_BURGERS_FILENAME: str = "1D_Burgers_Sols_Nu0.01.hdf5"
_DEFAULT_RD_FILENAME_TEMPLATE: str = "2D_DiffReact_Sols_Nu{nu}.hdf5"
_DEFAULT_RD_FINETUNE_NU: str = "2.0"

# Checkpoint filenames
_PRETRAIN_BEST_CKPT: str = "exp3_pretrain_best.pt"
_FINETUNE_BEST_CKPT: str = "exp3_finetune_rd_best.pt"
_SCRATCH_BEST_CKPT: str = "exp3_scratch_rd_best.pt"

# Results filenames
_RESULTS_CSV: str = "exp3_results.csv"
_RESULTS_JSON: str = "exp3_results.json"

# Spatial dimensionality constants
_N_DIMS_1D: int = 1
_N_DIMS_2D: int = 2

# Default n_in / n_out per physics (from config.yaml data section)
_ADVECTION_N_IN: int = 1
_ADVECTION_N_OUT: int = 1
_BURGERS_N_IN: int = 1
_BURGERS_N_OUT: int = 1
_RD_N_IN: int = 2
_RD_N_OUT: int = 2

# Default number of timed epochs for benchmark_epoch_time
_DEFAULT_N_TIMED_EPOCHS: int = 5


# ---------------------------------------------------------------------------
# Helper: reshape 1D tensors to 2D for a 2D backbone
# ---------------------------------------------------------------------------


class Reshape1Dto2DDataset(Dataset):
    """Wrapper dataset that reshapes 1D spatial tensors to 2D.

    Converts tensors of shape [C, L] to [C, L, 1] so that a 2D backbone
    (using SpectralConv2d) can process 1D physics data. This is the standard
    approach when a single backbone must handle both 1D and 2D problems.

    The width dimension is set to 1, making the 2D convolution degenerate
    to a 1D convolution along the height dimension.

    Attributes:
        _base_dataset: Underlying 1D dataset.
        _n_in: Number of input channels (unchanged).
        _n_out: Number of output channels (unchanged).
    """

    def __init__(self, base_dataset: Dataset) -> None:
        """Initialise Reshape1Dto2DDataset.

        Args:
            base_dataset: Underlying dataset returning ([C, L], [C, L]) tuples.
                Must implement get_n_in() and get_n_out().
        """
        super().__init__()
        self._base_dataset: Dataset = base_dataset
        self._n_in: int = int(base_dataset.get_n_in())   # type: ignore[attr-defined]
        self._n_out: int = int(base_dataset.get_n_out())  # type: ignore[attr-defined]

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self._base_dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Return reshaped input and target tensors.

        Args:
            idx: Sample index.

        Returns:
            Tuple of:
              - input_tensor: shape [C, L, 1] (float32)
              - target_tensor: shape [C, L, 1] (float32)
        """
        inp: Tensor
        tgt: Tensor
        inp, tgt = self._base_dataset[idx]  # type: ignore[index]

        # [C, L] -> [C, L, 1]: add dummy width dimension
        if inp.ndim == 2:
            inp = inp.unsqueeze(-1)   # [C, L, 1]
        if tgt.ndim == 2:
            tgt = tgt.unsqueeze(-1)   # [C, L, 1]

        return inp, tgt

    def get_n_in(self) -> int:
        """Return the number of input channels."""
        return self._n_in

    def get_n_out(self) -> int:
        """Return the number of output channels."""
        return self._n_out


# ---------------------------------------------------------------------------
# Exp3MultiPhysics
# ---------------------------------------------------------------------------


class Exp3MultiPhysics(ExperimentBase):
    """Experiment 3: General multi-physics pretraining and transfer.

    Validates the most general transfer learning scenario in the paper:
    the shared FNO backbone is pretrained on two physically distinct PDEs
    (Advection + Burgers from PDEBench), then the frozen backbone is
    transferred to Reaction-Diffusion via a new adapter pair.

    The experiment runs four phases:
      1. Pretrain: all parameters jointly optimized on Advection + Burgers.
      2. Finetune: backbone frozen, only new RD adapter trained.
      3. Scratch: same architecture trained from scratch on RD only.
      4. Evaluate: MSE, NMAE, epoch time, param count for both models.

    Backbone dimensionality strategy:
      Advection and Burgers are 1D problems; Reaction-Diffusion is 2D.
      To use a single backbone for all three, 1D inputs are reshaped to
      [C, L, 1] (adding a dummy width dimension), allowing the 2D backbone
      (SpectralConv2d) to process them. This is the standard approach in
      the FNO literature for mixed-dimensionality problems.

    Attributes:
        pretrain_datasets: Dict[str, Dataset] — training splits for pretrain.
            Keys: 'advection_beta0p1', 'burgers_nu0p01'.
            Values: Reshape1Dto2DDataset wrapping PDEBenchDataset.
        pretrain_val_datasets: Dict[str, Dataset] — val splits for pretrain.
        target_dataset: Dataset — RD training split for fine-tuning.
        target_val_dataset: Dataset — RD validation split.
        target_test_dataset: Dataset — RD test split for evaluation.
        multi_physics_train: MultiPhysicsDataset — combined pretrain dataset.
        multi_physics_val: MultiPhysicsDataset — combined pretrain val dataset.
        model: AdapterFramework — pretrained + finetuned model.
        scratch_model: AdapterFramework — trained from scratch (baseline).
        _pretrain_history: Dict[str, List[float]] — pretrain loss curves.
        _finetune_history: Dict[str, List[float]] — finetune loss curves.
        _scratch_history: Dict[str, List[float]] — scratch training loss curves.
        results: Dict[str, Any] — accumulated experiment results.
    """

    def __init__(self, config_path: str) -> None:
        """Initialise Exp3MultiPhysics.

        Calls ExperimentBase.__init__ to load config, resolve device, and
        set up logger. All dataset and model attributes are initialized to
        None and populated by setup_data() and setup_model() respectively.

        Args:
            config_path: Path to the YAML configuration file
                (typically configs/exp3_multiphysics.yaml).
        """
        super().__init__(config_path)

        # Store config path for YAML re-reading in helper methods
        self._config_path: str = config_path

        # ── Pretrain dataset placeholders ─────────────────────────────────
        self.pretrain_datasets: Dict[str, Dataset] = {}
        self.pretrain_val_datasets: Dict[str, Dataset] = {}

        # ── Target (RD) dataset placeholders ─────────────────────────────
        self.target_dataset: Optional[Dataset] = None
        self.target_val_dataset: Optional[Dataset] = None
        self.target_test_dataset: Optional[Dataset] = None

        # ── MultiPhysicsDataset wrappers ──────────────────────────────────
        self.multi_physics_train: Optional[MultiPhysicsDataset] = None
        self.multi_physics_val: Optional[MultiPhysicsDataset] = None

        # ── Model instances ───────────────────────────────────────────────
        self.model: Optional[AdapterFramework] = None
        self.scratch_model: Optional[AdapterFramework] = None

        # ── Training histories ────────────────────────────────────────────
        self._pretrain_history: Dict[str, List[float]] = {}
        self._finetune_history: Dict[str, List[float]] = {}
        self._scratch_history: Dict[str, List[float]] = {}

        # ── Results accumulator ───────────────────────────────────────────
        self.results: Dict[str, Any] = {}

        self._logger.info(
            "Exp3MultiPhysics initialized. "
            "Call setup_data() then setup_model() then run()."
        )

    # -----------------------------------------------------------------------
    # Private: config field accessors with defaults
    # -----------------------------------------------------------------------

    def _get_advection_filename(self) -> str:
        """Construct the Advection HDF5 filename from config.

        Reads data.advection.filename_template and formats with beta=0.1.
        Falls back to the default filename if config reading fails.

        Returns:
            Advection HDF5 filename string.
        """
        try:
            import yaml
            with open(self._config_path, "r", encoding="utf-8") as fh:
                raw: Dict[str, Any] = yaml.safe_load(fh) or {}
            template: str = str(
                raw.get("data", {})
                .get("advection", {})
                .get("filename_template", "1D_Advection_Sols_beta{beta}.hdf5")
            )
            # Get beta value from pretrain_params
            pretrain_params = (
                raw.get("data", {})
                .get("advection", {})
                .get("pretrain_params", [{"beta": "0.1"}])
            )
            beta: str = "0.1"
            if pretrain_params and isinstance(pretrain_params, list):
                beta = str(pretrain_params[0].get("beta", "0.1"))
            return template.format(beta=beta)
        except Exception:
            return _DEFAULT_ADVECTION_FILENAME

    def _get_burgers_filename(self) -> str:
        """Construct the Burgers HDF5 filename from config.

        Reads data.burgers.filename_template and formats with nu=0.01.
        Falls back to the default filename if config reading fails.

        Returns:
            Burgers HDF5 filename string.
        """
        try:
            import yaml
            with open(self._config_path, "r", encoding="utf-8") as fh:
                raw: Dict[str, Any] = yaml.safe_load(fh) or {}
            template: str = str(
                raw.get("data", {})
                .get("burgers", {})
                .get("filename_template", "1D_Burgers_Sols_Nu{nu}.hdf5")
            )
            pretrain_params = (
                raw.get("data", {})
                .get("burgers", {})
                .get("pretrain_params", [{"nu": "0.01"}])
            )
            nu: str = "0.01"
            if pretrain_params and isinstance(pretrain_params, list):
                nu = str(pretrain_params[0].get("nu", "0.01"))
            return template.format(nu=nu)
        except Exception:
            return _DEFAULT_BURGERS_FILENAME

    def _get_rd_finetune_filename(self) -> str:
        """Construct the Reaction-Diffusion HDF5 filename for fine-tuning.

        Reads data.reaction_diffusion.filename_template and formats with
        the finetune nu value. Falls back to default if config reading fails.

        Returns:
            Reaction-Diffusion HDF5 filename string for fine-tuning.
        """
        try:
            import yaml
            with open(self._config_path, "r", encoding="utf-8") as fh:
                raw: Dict[str, Any] = yaml.safe_load(fh) or {}
            template: str = str(
                raw.get("data", {})
                .get("reaction_diffusion", {})
                .get("filename_template", _DEFAULT_RD_FILENAME_TEMPLATE)
            )
            finetune_params = (
                raw.get("data", {})
                .get("reaction_diffusion", {})
                .get("finetune_params", [{"nu": _DEFAULT_RD_FINETUNE_NU}])
            )
            nu: str = _DEFAULT_RD_FINETUNE_NU
            if finetune_params and isinstance(finetune_params, list):
                nu = str(finetune_params[0].get("nu", _DEFAULT_RD_FINETUNE_NU))
            return template.format(nu=nu)
        except Exception:
            return _DEFAULT_RD_FILENAME_TEMPLATE.format(nu=_DEFAULT_RD_FINETUNE_NU)

    def _get_pretrain_checkpoint_path(self) -> str:
        """Construct the path for the pretrain best checkpoint.

        Returns:
            Full path to the pretrain best checkpoint file.
        """
        return os.path.join(
            self.config.pretrain.checkpoint_dir,
            _PRETRAIN_BEST_CKPT,
        )

    def _get_finetune_checkpoint_path(self) -> str:
        """Construct the path for the finetune best checkpoint.

        Returns:
            Full path to the finetune best checkpoint file.
        """
        # Derive finetune checkpoint dir from pretrain dir
        pretrain_dir: str = self.config.pretrain.checkpoint_dir
        if "pretrain" in pretrain_dir:
            finetune_dir: str = pretrain_dir.replace("pretrain", "finetune")
        else:
            finetune_dir = os.path.join(
                os.path.dirname(pretrain_dir), "finetune"
            )
        return os.path.join(finetune_dir, _FINETUNE_BEST_CKPT)

    def _get_scratch_checkpoint_path(self) -> str:
        """Construct the path for the scratch baseline best checkpoint.

        Returns:
            Full path to the scratch best checkpoint file.
        """
        pretrain_dir: str = self.config.pretrain.checkpoint_dir
        if "pretrain" in pretrain_dir:
            scratch_dir: str = pretrain_dir.replace("pretrain", "scratch")
        else:
            scratch_dir = os.path.join(
                os.path.dirname(pretrain_dir), "scratch"
            )
        return os.path.join(scratch_dir, _SCRATCH_BEST_CKPT)

    # -----------------------------------------------------------------------
    # Public: setup_data
    # -----------------------------------------------------------------------

    def setup_data(self) -> None:
        """Load all datasets required for Experiment 3.

        Loads three PDEBench datasets:
          1. Advection beta=0.1 (1D) — pretrain
          2. Burgers nu=0.01 (1D) — pretrain
          3. Reaction-Diffusion nu=2.0 (2D) — finetune + test

        1D datasets (Advection, Burgers) are wrapped in Reshape1Dto2DDataset
        to add a dummy width dimension [C, L] -> [C, L, 1], enabling a 2D
        backbone to process them alongside the 2D Reaction-Diffusion data.

        All datasets are loaded for train, val, and test splits. The pretrain
        splits are wrapped in MultiPhysicsDataset for joint training.

        Config sources:
          data.pdebench_root: root directory for PDEBench HDF5 files
          data.advection.filename_template
          data.advection.pretrain_params[0].beta
          data.burgers.filename_template
          data.burgers.pretrain_params[0].nu
          data.reaction_diffusion.filename_template
          data.reaction_diffusion.finetune_params[0].nu
          data.{n_train, n_val, n_test, normalize}

        Raises:
            FileNotFoundError: If any PDEBench HDF5 file is not found.
                Download from: https://darus.uni-stuttgart.de/dataverse/pdebench
        """
        self._logger.info("setup_data: loading datasets for Experiment 3.")

        data_cfg = self.config.data
        pdebench_root: str = data_cfg.pdebench_root
        n_train: int = data_cfg.n_train
        n_val: int = data_cfg.n_val
        n_test: int = data_cfg.n_test
        normalize: bool = data_cfg.normalize

        # ── 1. Advection beta=0.1 (pretrain, 1D) ──────────────────────────
        advection_filename: str = self._get_advection_filename()
        advection_path: str = os.path.join(pdebench_root, advection_filename)

        self._logger.info(
            "Loading Advection pretrain dataset: '%s'", advection_path
        )

        if not os.path.isfile(advection_path):
            raise FileNotFoundError(
                f"Advection HDF5 file not found: '{advection_path}'. "
                f"Please download the PDEBench dataset from "
                f"https://darus.uni-stuttgart.de/dataverse/pdebench "
                f"and place it at the expected path."
            )

        advection_train_raw: PDEBenchDataset = PDEBenchDataset(
            physics_name=_PHYSICS_ADVECTION_PRETRAIN,
            data_path=advection_path,
            split="train",
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            normalize=normalize,
        )
        advection_val_raw: PDEBenchDataset = PDEBenchDataset(
            physics_name=_PHYSICS_ADVECTION_PRETRAIN,
            data_path=advection_path,
            split="val",
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            normalize=normalize,
        )

        # Wrap 1D data to 2D for the 2D backbone
        advection_train: Reshape1Dto2DDataset = Reshape1Dto2DDataset(
            advection_train_raw
        )
        advection_val: Reshape1Dto2DDataset = Reshape1Dto2DDataset(
            advection_val_raw
        )

        self._logger.info(
            "Advection dataset loaded: n_in=%d, n_out=%d, "
            "train=%d, val=%d samples.",
            advection_train.get_n_in(),
            advection_train.get_n_out(),
            len(advection_train),
            len(advection_val),
        )

        # ── 2. Burgers nu=0.01 (pretrain, 1D) ─────────────────────────────
        burgers_filename: str = self._get_burgers_filename()
        burgers_path: str = os.path.join(pdebench_root, burgers_filename)

        self._logger.info(
            "Loading Burgers pretrain dataset: '%s'", burgers_path
        )

        if not os.path.isfile(burgers_path):
            raise FileNotFoundError(
                f"Burgers HDF5 file not found: '{burgers_path}'. "
                f"Please download the PDEBench dataset from "
                f"https://darus.uni-stuttgart.de/dataverse/pdebench "
                f"and place it at the expected path."
            )

        burgers_train_raw: PDEBenchDataset = PDEBenchDataset(
            physics_name=_PHYSICS_BURGERS_PRETRAIN,
            data_path=burgers_path,
            split="train",
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            normalize=normalize,
        )
        burgers_val_raw: PDEBenchDataset = PDEBenchDataset(
            physics_name=_PHYSICS_BURGERS_PRETRAIN,
            data_path=burgers_path,
            split="val",
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            normalize=normalize,
        )

        # Wrap 1D data to 2D for the 2D backbone
        burgers_train: Reshape1Dto2DDataset = Reshape1Dto2DDataset(
            burgers_train_raw
        )
        burgers_val: Reshape1Dto2DDataset = Reshape1Dto2DDataset(
            burgers_val_raw
        )

        self._logger.info(
            "Burgers dataset loaded: n_in=%d, n_out=%d, "
            "train=%d, val=%d samples.",
            burgers_train.get_n_in(),
            burgers_train.get_n_out(),
            len(burgers_train),
            len(burgers_val),
        )

        # ── 3. Reaction-Diffusion nu=2.0 (finetune + test, 2D) ────────────
        rd_filename: str = self._get_rd_finetune_filename()
        rd_path: str = os.path.join(pdebench_root, rd_filename)

        self._logger.info(
            "Loading Reaction-Diffusion finetune dataset: '%s'", rd_path
        )

        if not os.path.isfile(rd_path):
            raise FileNotFoundError(
                f"Reaction-Diffusion HDF5 file not found: '{rd_path}'. "
                f"Please download the PDEBench dataset from "
                f"https://darus.uni-stuttgart.de/dataverse/pdebench "
                f"and place it at the expected path."
            )

        self.target_dataset = PDEBenchDataset(
            physics_name=_PHYSICS_REACTION_DIFFUSION,
            data_path=rd_path,
            split="train",
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            normalize=normalize,
        )
        self.target_val_dataset = PDEBenchDataset(
            physics_name=_PHYSICS_REACTION_DIFFUSION,
            data_path=rd_path,
            split="val",
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            normalize=normalize,
        )
        self.target_test_dataset = PDEBenchDataset(
            physics_name=_PHYSICS_REACTION_DIFFUSION,
            data_path=rd_path,
            split="test",
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            normalize=normalize,
        )

        self._logger.info(
            "Reaction-Diffusion dataset loaded: n_in=%d, n_out=%d, "
            "train=%d, val=%d, test=%d samples.",
            self.target_dataset.get_n_in(),   # type: ignore[attr-defined]
            self.target_dataset.get_n_out(),  # type: ignore[attr-defined]
            len(self.target_dataset),         # type: ignore[arg-type]
            len(self.target_val_dataset),     # type: ignore[arg-type]
            len(self.target_test_dataset),    # type: ignore[arg-type]
        )

        # ── 4. Assemble pretrain dicts ─────────────────────────────────────
        self.pretrain_datasets = {
            _PHYSICS_ADVECTION_PRETRAIN: advection_train,
            _PHYSICS_BURGERS_PRETRAIN: burgers_train,
        }
        self.pretrain_val_datasets = {
            _PHYSICS_ADVECTION_PRETRAIN: advection_val,
            _PHYSICS_BURGERS_PRETRAIN: burgers_val,
        }

        # ── 5. Wrap in MultiPhysicsDataset ────────────────────────────────
        # Uniform sampling weights (None) = equal probability per physics.
        # Both Advection and Burgers have n_in=1, n_out=1 after reshaping,
        # so no padding is needed in the collate function for this experiment.
        self.multi_physics_train = MultiPhysicsDataset(
            datasets=self.pretrain_datasets,
            sampling_weights=None,
        )
        self.multi_physics_val = MultiPhysicsDataset(
            datasets=self.pretrain_val_datasets,
            sampling_weights=None,
        )

        self._logger.info(
            "setup_data complete: %d pretrain physics, "
            "multi_physics_train=%d, multi_physics_val=%d, "
            "rd_train=%d, rd_val=%d, rd_test=%d samples.",
            len(self.pretrain_datasets),
            len(self.multi_physics_train),
            len(self.multi_physics_val),
            len(self.target_dataset),         # type: ignore[arg-type]
            len(self.target_val_dataset),     # type: ignore[arg-type]
            len(self.target_test_dataset),    # type: ignore[arg-type]