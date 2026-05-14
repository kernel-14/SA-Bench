```python
## experiments/exp2_input_extension.py
"""
Experiment 2: Input Function Set Extension scenario.

Reproduces the second experimental scenario from:
  "Towards Universal Neural Operators through Multiphysics Pretraining"

The experiment validates that adapter-based pretraining enables efficient
transfer when the fine-tuning problem requires additional input functions
(extended input cardinality). The backbone is frozen; only a new lifting
adapter (accepting more input channels) is trained.

Two sub-scenarios:
  1. Heat → Heat+Convection:
       Pretrain on pure heat equation (n_in=1: u0).
       Fine-tune on heat with convection (n_in=3: u0, vx, vy).
  2. Reaction-Diffusion → RD+Advection:
       Pretrain on standard RD (n_in=2: u0, v0).
       Fine-tune on RD extended with advection (n_in=4: u0, v0, vx, vy).

Results should match Table 2 in the paper (Heat & Reaction-Diffusion rows):
  - MambaFNO (pretr.):  MSE=3.91e-6,  NMAE=0.0041%, Avg.epoch=131.2s
  - MambaFNO (scratch): MSE=4.291e-6, NMAE=0.0054%, Avg.epoch=261.1s
  - FNO (scratch):      MSE=7.286e-6, NMAE=0.0121%, Avg.epoch=41.3s
  - Perceiver (pretr.): MSE=4.107e-6, NMAE=0.0051%, Avg.epoch=20.4s
  - CoDA-NO (pretr.):   MSE=1.043e-5, NMAE=0.013%,  Avg.epoch=185.1s

Design contract (Data structures and interfaces):
  Exp2InputExtension(ExperimentBase):
    base_dataset: HeatConvectionDataset (with_convection=False, n_in=1)
    extended_dataset: HeatConvectionDataset (with_convection=True, n_in=3)
    model: AdapterFramework
    setup_data() -> None
    setup_model() -> None
    run() -> Dict[str, Any]
    _run_pretrain_base() -> None
    _run_finetune_extended() -> None
    _run_scratch_baseline() -> None

Config alignment (config.yaml / configs/exp2_input_extension.yaml):
  data.heat.alpha: 0.01
  data.heat.grid_size: 64
  data.heat.n_steps: 200
  data.heat.dt: 0.001
  data.heat.n_samples: 1000
  data.heat.base.n_in: 1
  data.heat.extended.n_in: 3
  data.heat.extended.with_convection: true
  data.reaction_diffusion.filename_template
  data.reaction_diffusion.pretrain_params[0].nu
  data.reaction_diffusion.extended_with_advection.n_in: 4
  data.{n_train, n_val, n_test, normalize, pdebench_root, cache_root}
  training.pretrain.{lr, batch_size, n_epochs, scheduler, checkpoint_dir}
  training.finetune.{lr, batch_size, n_epochs, freeze_backbone}
  evaluation.{metrics, output_dir, epoch_time_n_runs, epoch_time_warmup}
  experiments.exp2_input_extension.{pretrain_physics, finetune_physics}

Dependencies:
  utils/config.py              -> Config, FinetuneConfig
  utils/logging_utils.py       -> get_logger, ResultsTable
  data/heat_convection_generator.py -> HeatConvectionDataset
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
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from data.heat_convection_generator import HeatConvectionDataset
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
from utils.config import Config, FinetuneConfig
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
# ---------------------------------------------------------------------------

_PHYSICS_HEAT_BASE: str = "heat_base"
_PHYSICS_HEAT_EXTENDED: str = "heat_extended"
_PHYSICS_RD_BASE: str = "rd_base"
_PHYSICS_RD_EXTENDED: str = "rd_extended"

# Checkpoint filename templates
_PRETRAIN_HEAT_CKPT: str = "exp2_pretrain_heat_best.pt"
_FINETUNE_HEAT_CKPT: str = "exp2_finetune_heat_extended_best.pt"
_SCRATCH_HEAT_CKPT: str = "exp2_scratch_heat_extended_best.pt"
_PRETRAIN_RD_CKPT: str = "exp2_pretrain_rd_best.pt"
_FINETUNE_RD_CKPT: str = "exp2_finetune_rd_extended_best.pt"
_SCRATCH_RD_CKPT: str = "exp2_scratch_rd_extended_best.pt"

# Results filenames
_RESULTS_CSV: str = "exp2_results.csv"
_RESULTS_JSON: str = "exp2_results.json"

# Default spatial dimensionality for 2D problems
_N_DIMS_2D: int = 2

# Default number of timed epochs for benchmark
_DEFAULT_N_TIMED_EPOCHS: int = 5

# Default heat simulation parameters (from config.yaml data.heat section)
_DEFAULT_HEAT_ALPHA: float = 0.01
_DEFAULT_HEAT_GRID_SIZE: int = 64
_DEFAULT_HEAT_N_STEPS: int = 200
_DEFAULT_HEAT_DT: float = 0.001
_DEFAULT_HEAT_N_SAMPLES: int = 1000

# Default RD pretrain nu value (from config.yaml data.reaction_diffusion)
_DEFAULT_RD_PRETRAIN_NU: str = "1.0"
_DEFAULT_RD_FILENAME_TEMPLATE: str = "2D_DiffReact_Sols_Nu{nu}.hdf5"

# Default n_in values (from config.yaml data.heat.base/extended)
_DEFAULT_HEAT_BASE_N_IN: int = 1
_DEFAULT_HEAT_EXTENDED_N_IN: int = 3
_DEFAULT_RD_BASE_N_IN: int = 2
_DEFAULT_RD_EXTENDED_N_IN: int = 4


# ---------------------------------------------------------------------------
# RDAdvectionDataset — wrapper that augments RD data with velocity fields
# ---------------------------------------------------------------------------


class RDAdvectionDataset(Dataset):
    """Wrapper dataset that augments a PDEBench RD dataset with velocity fields.

    Extends the standard Reaction-Diffusion dataset (n_in=2: u0, v0) with
    two additional divergence-free velocity field channels (vx, vy), producing
    an extended dataset with n_in=4: [u0, v0, vx, vy].

    This implements the "extended reaction-diffusion equations with advection"
    scenario described in Section 4 of the paper (Experiment 2).

    The velocity fields are generated deterministically per sample using a
    sinusoidal stream function approach (same as HeatConvectionDataset),
    seeded by the sample index for reproducibility.

    Tensor layout: channel-first [B, C, H, W].
      - Base inputs:    [B, 2, H, W]  (u0, v0)
      - Extended inputs: [B, 4, H, W] (u0, v0, vx, vy)
      - Targets:        [B, 2, H, W]  (u(t=T), v(t=T))

    Attributes:
        _base_dataset: Underlying PDEBenchDataset with n_in=2.
        _n_in: Always 4 (u0, v0, vx, vy).
        _n_out: Always 2 (u(t=T), v(t=T)).
        _grid_size: Spatial resolution of the grid.
        _rng_seed_offset: Seed offset for velocity field generation.
    """

    def __init__(
        self,
        base_dataset: PDEBenchDataset,
        grid_size: int = 64,
        rng_seed_offset: int = 0,
    ) -> None:
        """Initialise RDAdvectionDataset.

        Args:
            base_dataset: PDEBenchDataset for reaction-diffusion (n_in=2).
                Must return (input [2, H, W], target [2, H, W]) tuples.
            grid_size: Spatial resolution of the grid. Used to generate
                velocity fields of the correct size. Default 64.
            rng_seed_offset: Seed offset added to sample index for velocity
                field generation. Allows different splits to have different
                velocity fields. Default 0.
        """
        super().__init__()
        self._base_dataset: PDEBenchDataset = base_dataset
        self._grid_size: int = grid_size
        self._rng_seed_offset: int = rng_seed_offset
        self._n_in: int = _DEFAULT_RD_EXTENDED_N_IN  # 4
        self._n_out: int = 2

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self._base_dataset)

    def __getitem__(self, idx: int):
        """Return extended input and target tensors for sample idx.

        Args:
            idx: Sample index.

        Returns:
            Tuple of:
              - extended_input: Tensor [4, H, W] = [u0, v0, vx, vy]
              - target: Tensor [2, H, W] = [u(t=T), v(t=T)]
        """
        import numpy as np
        import torch

        # Get base (u0, v0) input and target from underlying dataset
        base_input, target = self._base_dataset[idx]
        # base_input: [2, H, W], target: [2, H, W]

        # Generate deterministic divergence-free velocity field for this sample
        rng = np.random.default_rng(seed=idx + self._rng_seed_offset + 42000)

        # Generate stream function as superposition of sinusoids
        x_coords = np.linspace(0.0, 1.0, self._grid_size, endpoint=False)
        y_coords = np.linspace(0.0, 1.0, self._grid_size, endpoint=False)
        xx, yy = np.meshgrid(x_coords, y_coords, indexing="ij")

        psi = np.zeros((self._grid_size, self._grid_size), dtype=np.float64)
        n_modes: int = 4
        for _ in range(n_modes):
            fx = int(rng.integers(1, 5))
            fy = int(rng.integers(1, 5))
            amp = float(rng.uniform(0.1, 1.0))
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            psi += amp * np.sin(
                2.0 * np.pi * fx * xx + 2.0 * np.pi * fy * yy + phase
            )

        # Derive divergence-free velocity: vx = dpsi/dy, vy = -dpsi/dx
        # Using finite differences with periodic boundary conditions
        dx = 1.0 / self._grid_size
        vx = (np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) / (2.0 * dx)
        vy = -(np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0)) / (2.0 * dx)

        # Normalize to unit maximum magnitude for CFL stability
        max_mag = np.sqrt(vx**2 + vy**2).max()
        if max_mag > 1e-10:
            vx = vx / max_mag
            vy = vy / max_mag

        # Convert to float32 tensors
        vx_tensor = torch.from_numpy(vx.astype(np.float32)).unsqueeze(0)  # [1, H, W]
        vy_tensor = torch.from_numpy(vy.astype(np.float32)).unsqueeze(0)  # [1, H, W]

        # Concatenate: [u0, v0, vx, vy] -> [4, H, W]
        extended_input = torch.cat([base_input, vx_tensor, vy_tensor], dim=0)

        return extended_input, target

    def get_n_in(self) -> int:
        """Return the number of input channels (4: u0, v0, vx, vy)."""
        return self._n_in

    def get_n_out(self) -> int:
        """Return the number of output channels (2: u(t=T), v(t=T))."""
        return self._n_out


# ---------------------------------------------------------------------------
# Exp2InputExtension
# ---------------------------------------------------------------------------


class Exp2InputExtension(ExperimentBase):
    """Experiment 2: Input Function Set Extension.

    Validates the adapter framework's ability to handle extended input
    cardinality: a pretrained backbone (frozen) is reused with a new
    LiftingAdapter that accepts more input channels than the pretrain adapter.

    Sub-scenario 1 — Heat → Heat+Convection:
      Pretrain: heat equation (n_in=1: u0)
      Fine-tune: heat + convection (n_in=3: u0, vx, vy)

    Sub-scenario 2 — RD → RD+Advection:
      Pretrain: reaction-diffusion (n_in=2: u0, v0)
      Fine-tune: RD + advection (n_in=4: u0, v0, vx, vy)

    Attributes:
        base_dataset: HeatConvectionDataset training split (with_convection=False).
        base_val_dataset: HeatConvectionDataset validation split.
        base_test_dataset: HeatConvectionDataset test split.
        extended_dataset: HeatConvectionDataset training split (with_convection=True).
        extended_val_dataset: HeatConvectionDataset validation split.
        extended_test_dataset: HeatConvectionDataset test split.
        rd_base_train_dataset: PDEBenchDataset RD training split.
        rd_base_val_dataset: PDEBenchDataset RD validation split.
        rd_extended_train_dataset: RDAdvectionDataset training split.
        rd_extended_val_dataset: RDAdvectionDataset validation split.
        rd_extended_test_dataset: RDAdvectionDataset test split.
        model: AdapterFramework for heat sub-scenario.
        rd_model: AdapterFramework for RD sub-scenario.
        scratch_model: AdapterFramework trained from scratch (heat baseline).
        rd_scratch_model: AdapterFramework trained from scratch (RD baseline).
        results: Accumulated experiment results dict.
    """

    def __init__(self, config_path: str) -> None:
        """Initialise Exp2InputExtension.

        Calls ExperimentBase.__init__ to load config, resolve device, and
        set up logger. All dataset and model attributes are initialized to
        None and populated by setup_data() and setup_model().

        Args:
            config_path: Path to the YAML configuration file
                (typically configs/exp2_input_extension.yaml).
        """
        super().__init__(config_path)

        # ── Heat sub-scenario datasets ────────────────────────────────────
        self.base_dataset: Optional[HeatConvectionDataset] = None
        self.base_val_dataset: Optional[HeatConvectionDataset] = None
        self.base_test_dataset: Optional[HeatConvectionDataset] = None
        self.extended_dataset: Optional[HeatConvectionDataset] = None
        self.extended_val_dataset: Optional[HeatConvectionDataset] = None
        self.extended_test_dataset: Optional[HeatConvectionDataset] = None

        # ── RD sub-scenario datasets ──────────────────────────────────────
        self.rd_base_train_dataset: Optional[PDEBenchDataset] = None
        self.rd_base_val_dataset: Optional[PDEBenchDataset] = None
        self.rd_extended_train_dataset: Optional[RDAdvectionDataset] = None
        self.rd_extended_val_dataset: Optional[RDAdvectionDataset] = None
        self.rd_extended_test_dataset: Optional[RDAdvectionDataset] = None

        # ── Model instances ───────────────────────────────────────────────
        self.model: Optional[AdapterFramework] = None
        self.rd_model: Optional[AdapterFramework] = None
        self.scratch_model: Optional[AdapterFramework] = None
        self.rd_scratch_model: Optional[AdapterFramework] = None

        # ── Training histories ────────────────────────────────────────────
        self._pretrain_heat_history: Dict[str, List[float]] = {}
        self._finetune_heat_history: Dict[str, List[float]] = {}
        self._scratch_heat_history: Dict[str, List[float]] = {}
        self._pretrain_rd_history: Dict[str, List[float]] = {}
        self._finetune_rd_history: Dict[str, List[float]] = {}
        self._scratch_rd_history: Dict[str, List[float]] = {}

        # ── Results accumulator ───────────────────────────────────────────
        self.results: Dict[str, Any] = {}

        self._logger.info(
            "Exp2InputExtension initialized. "
            "Call setup_data() then setup_model() then run()."
        )

    # -----------------------------------------------------------------------
    # Private: config field accessors with defaults
    # -----------------------------------------------------------------------

    def _get_heat_alpha(self) -> float:
        """Read heat diffusivity from config with fallback default."""
        try:
            import yaml
            with open(self._config_path, "r") as f:
                raw = yaml.safe_load(f) or {}
            return float(raw.get("data", {}).get("heat", {}).get("alpha", _DEFAULT_HEAT_ALPHA))
        except Exception:
            return _DEFAULT_HEAT_ALPHA

    def _get_heat_grid_size(self) -> int:
        """Read heat grid size from config with fallback default."""
        try:
            import yaml
            with open(self._config_path, "r") as f:
                raw = yaml.safe_load(f) or {}
            return int(raw.get("data", {}).get("heat", {}).get("grid_size", _DEFAULT_HEAT_GRID_SIZE))
        except Exception:
            return _DEFAULT_HEAT_GRID_SIZE

    def _get_heat_n_steps(self) -> int:
        """Read heat n_steps from config with fallback default."""
        try:
            import yaml
            with open(self._config_path, "r") as f:
                raw = yaml.safe_load(f) or {}
            return int(raw.get("data", {}).get("heat", {}).get("n_steps", _DEFAULT_HEAT_N_STEPS))
        except Exception:
            return _DEFAULT_HEAT_N_STEPS

    def _get_heat_dt(self) -> float:
        """Read heat dt from config with fallback default."""
        try:
            import yaml
            with open(self._config_path, "r") as f:
                raw = yaml.safe_load(f) or {}
            return float(raw.get("data", {}).get("heat", {}).get("dt", _DEFAULT_HEAT_DT))
        except Exception:
            return _DEFAULT_HEAT_DT

    def _get_heat_n_samples(self) -> int:
        """Read heat n_samples from config with fallback default."""
        try:
            import yaml
            with open(self._config_path, "r") as f:
                raw = yaml.safe_load(f) or {}
            return int(raw.get("data", {}).get("heat", {}).get("n_samples", _DEFAULT_HEAT_N_SAMPLES))
        except Exception:
            return _DEFAULT_HEAT_N_SAMPLES

    def _get_rd_pretrain_nu(self) -> str:
        """Read RD pretrain nu from config with fallback default."""
        try:
            import yaml
            with open(self._config_path, "r") as f:
                raw = yaml.safe_load(f) or {}
            pretrain_params = (
                raw.get("data", {})
                .get("reaction_diffusion", {})
                .get("pretrain_params", [{}])
            )
            if pretrain_params and isinstance(pretrain_params, list):
                return str(pretrain_params[0].get("nu", _DEFAULT_RD_PRETRAIN_NU))
            return _DEFAULT_RD_PRETRAIN_NU
        except Exception:
            return _DEFAULT_RD_PRETRAIN_NU

    def _get_rd_filename_template(self) -> str:
        """Read RD filename template from config with fallback default."""
        try:
            import yaml
            with open(self._config_path, "r") as f:
                raw = yaml.safe_load(f) or {}
            return str(
                raw.get("data", {})
                .get("reaction_diffusion", {})
                .get("filename_template", _DEFAULT_RD_FILENAME_TEMPLATE)
            )
        except Exception:
            return _DEFAULT_RD_FILENAME_TEMPLATE

    # -----------------------------------------------------------------------
    # Public: setup_data
    # -----------------------------------------------------------------------

    def setup_data(self) -> None:
        """Load all datasets required for Experiment 2.

        Loads datasets for both sub-scenarios:

        Sub-scenario 1 (Heat):
          - HeatConvectionDataset(with_convection=False) — base pretrain
          - HeatConvectionDataset(with_convection=True)  — extended finetune

        Sub-scenario 2 (Reaction-Diffusion):
          - PDEBenchDataset('reaction_diffusion', nu=1.0) — base pretrain
          - RDAdvectionDataset (wraps RD + adds velocity fields) — extended finetune

        Config sources:
          data.heat.{alpha, grid_size, n_steps, dt, n_samples}
          data.reaction_diffusion.{filename_template, pretrain_params}
          data.{pdebench_root, cache_root, n_train, n_val, n_test, normalize}

        Raises:
            FileNotFoundError: If PDEBench RD HDF5 file is not found.
        """
        self._logger.info("setup_data: loading datasets for Experiment 2.")

        data_cfg = self.config.data
        n_train: int = data_cfg.n_train
        n_val: int = data_cfg.n_val
        n_test: int = data_cfg.n_test
        normalize: bool = data_cfg.normalize
        cache_root: str = data_cfg.heat_root
        pdebench_root: str = data_cfg.pdebench_root

        # Read heat parameters from config (with defaults)
        alpha: float = self._get_heat_alpha()
        grid_size: int = self._get_heat_grid_size()
        n_steps: int = self._get_heat_n_steps()
        dt: float = self._get_heat_dt()
        n_samples: int = self._get_heat_n_samples()

        # ── Sub-scenario 1: Heat datasets ─────────────────────────────────

        # Base heat (with_convection=False, n_in=1)
        cache_path_heat_base: str = os.path.join(cache_root, "heat_base.npz")
        self._logger.info(
            "Loading base heat dataset (cache: '%s').", cache_path_heat_base
        )

        self.base_dataset = HeatConvectionDataset(
            alpha=alpha,
            with_convection=False,
            n_samples=n_samples,
            grid_size=grid_size,
            n_steps=n_steps,
            dt=dt,
            split="train",
            cache_path=cache_path_heat_base,
            normalize=normalize,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
        )
        self.base_val_dataset = HeatConvectionDataset(
            alpha=alpha,
            with_convection=False,
            n_samples=n_samples,
            grid_size=grid_size,
            n_steps=n_steps,
            dt=dt,
            split="val",
            cache_path=cache_path_heat_base,
            normalize=normalize,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
        )
        self.base_test_dataset = HeatConvectionDataset(
            alpha=alpha,
            with_convection=False,
            n_samples=n_samples,
            grid_size=grid_size,
            n_steps=n_steps,
            dt=dt,
            split="test",
            cache_path=cache_path_heat_base,
            normalize=normalize,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
        )

        # Validate n_in matches config expectation
        actual_base_n_in: int = self.base_dataset.get_n_in()
        if actual_base_n_in != _DEFAULT_HEAT_BASE_N_IN:
            raise ValueError(
                f"Base heat dataset has n_in={actual_base_n_in}, "
                f"expected {_DEFAULT_HEAT_BASE_N_IN}. "
                f"Check HeatConvectionDataset(with_convection=False)."
            )

        self._logger.info(
            "Base heat dataset loaded: n_in=%d, n_out=%d, "
            "train=%d, val=%d, test=%d samples.",
            self.base_dataset.get_n_in(),
            self.base_dataset.get_n_out(),
            len(self.base_dataset),
            len(self.base_val_dataset),
            len(self.base_test_dataset),
        )

        # Extended heat (with_convection=True, n_in=3)
        cache_path_heat_ext: str = os.path.join(cache_root, "heat_extended.npz")
        self._logger.info(
            "Loading extended heat dataset (cache: '%s').", cache_path_heat_ext
        )

        self.extended_dataset = HeatConvectionDataset(
            alpha=alpha,
            with_convection=True,
            n_samples=n_samples,
            grid_size=grid_size,
            n_steps=n_steps,
            dt=dt,
            split="train",
            cache_path=cache_path_heat_ext,
            normalize=normalize,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
        )
        self.extended_val_dataset = HeatConvectionDataset(
            alpha=alpha,
            with_convection=True,
            n_samples=n_samples,
            grid_size=grid_size,
            n_steps=n_steps,
            dt=dt,
            split="val",
            cache_path=cache_path_heat_ext,
            normalize=normalize,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
        )
        self.extended_test_dataset = HeatConvectionDataset(
            alpha=alpha,
            with_convection=True,
            n_samples=n_samples,
            grid_size=grid_size,
            n_steps=n_steps,
            dt=dt,
            split="test",
            cache_path=cache_path_heat_ext,
            normalize=normalize,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
        )

        # Validate extended n_in
        actual_ext_n_in: int = self.extended_dataset.get_n_in()
        if actual_ext_n_in != _DEFAULT_HEAT_EXTENDED_N_IN:
            raise ValueError(
                f"Extended heat dataset has n_in={actual_ext_n_in}, "
                f"expected {_DEFAULT_HEAT_EXTENDED_N_IN}. "
                f"Check HeatConvectionDataset(with_convection=True)."
            )

        self._logger.info(
            "Extended heat dataset loaded: n_in=%d, n_out=%d, "
            "train=%d, val=%d, test=%d samples.",
            self.extended_dataset.get_n_in(),
            self.extended_dataset.get_n_out(),
            len(self.extended_dataset),
            len(self.extended_val_dataset),
            len(self.extended_test_dataset),
        )

        # ── Sub-scenario 2: Reaction-Diffusion datasets ───────────────────

        rd_nu: str = self._get_rd_pretrain_nu()
        rd_filename_template: str = self._get_rd_filename_template()
        rd_filename: str = rd_filename_template.format(nu=rd_nu)
        rd_path: str = os.path.join(pdebench_root, rd_filename)

        self._logger.info(
            "Loading RD base dataset: '%s'.", rd_path
        )

        if not os.path.isfile(rd_path):
            self._logger.warning(
                "RD HDF5 file not found at '%s'. "
                "RD sub-scenario will be skipped. "
                "Download from https://darus.uni-stuttgart.de/dataverse/pdebench",
                rd