```python
## main.py
"""Entry point for iCT-GC: Improving Consistency Models with Generator-Augmented Flows.

This module orchestrates the full experiment pipeline described in the paper.
It supports four execution modes:

    train              — Full training run (Algorithm 1) + final evaluation
    eval               — Evaluate a trained checkpoint (FID/KID/IS)
    ablation_mu        — Reproduce Figure 5: FID vs steps for μ ∈ {0,0.1,0.3,0.5,0.7,1.0}
    compare_couplings  — Reproduce Table 1: iCT-IC vs iCT-OT vs iCT-GC

Usage::

    # Train iCT-GC on CIFAR-10
    python main.py --config configs/cifar10.yaml --mode train --seed 42

    # Evaluate a checkpoint
    python main.py --config configs/cifar10.yaml --mode eval \
        --checkpoint ./checkpoints/cifar10/best.pt

    # μ ablation study (Figure 5)
    python main.py --config configs/cifar10.yaml --mode ablation_mu

    # Coupling comparison (Table 1)
    python main.py --config configs/cifar10.yaml --mode compare_couplings

Config values are loaded from the dataset-specific YAML (e.g. configs/cifar10.yaml).
The master config.yaml documents all available fields and their defaults.
"""

import argparse
import copy
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

# ---------------------------------------------------------------------------
# Internal imports — order matters to avoid circular dependencies
# ---------------------------------------------------------------------------
from utils.helpers import (
    AverageMeter,
    count_parameters,
    get_device,
    normalize_images,
    set_seed,
)
from utils.ema import EMA
from data.dataset_loader import DatasetLoader
from models.song_unet import SongUNet
from models.consistency_model import ConsistencyModel
from training.trainer import Trainer
from evaluation.evaluator import Evaluator


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


class Config:
    """Flat configuration object loaded from a dataset-specific YAML file.

    All fields have defaults matching the ``defaults`` section of config.yaml.
    Dataset-specific YAMLs (e.g. configs/cifar10.yaml) override these defaults.

    The ``from_yaml`` classmethod handles both flat dataset YAMLs and the
    master config.yaml with nested sections.

    Attributes:
        dataset: Dataset identifier. One of 'cifar10', 'imagenet32',
            'celeba', 'lsun_church', 'ffhq', 'imagenet64_cond'.
        image_size: Spatial resolution (e.g. 32, 64).
        in_channels: Number of input image channels (default 3 for RGB).
        out_channels: Number of output image channels (default 3 for RGB).
        batch_size: Training and generation batch size.
        training_steps: Total number of training iterations K.
        learning_rate: Lion optimizer learning rate.
        optimizer: Optimizer name (always 'lion' per paper).
        weight_decay: Lion optimizer weight decay (default 0.0).
        s0: Initial number of discretization intervals (default 10).
        s1: Final number of discretization intervals (default 1280).
        rho: Noise schedule exponent (default 7.0).
        sigma_min: Minimum noise level σ_0 (default 0.002).
        sigma_max: Maximum noise level σ_T (default 80.0).
        sigma_data: Data pixel std for c_skip/c_out (default 0.5).
        P_mean: Timestep distribution mean (default -1.1).
        P_std: Timestep distribution std (default 2.0).
        model_channels: Base channel count for SongUNet (default 128).
        num_blocks: ResNet blocks per level. int or list of ints.
        channel_mult: Per-level channel multipliers (e.g. [1,2,2]).
        attn_resolutions: Resolutions with attention (e.g. [16] or []).
        dropout: Dropout probability. float or list of floats.
        embedding_type: Time embedding type ('positional' or 'fourier').
        mu: Joint learning parameter μ ∈ [0,1] (default 0.5).
        coupling: Coupling strategy ('ic', 'ot', 'gc').
        ot_batch_size: OT solver batch size (default 512).
        ema_decay: EMA decay coefficient (default 0.9999).
        distance_fn: Distance function mode (default 'pseudo_huber').
        pseudo_huber_c: Pseudo-Huber constant c (default 0.00054).
        eval_every: Evaluation/checkpoint interval in steps (default 10000).
        num_eval_samples: Images per evaluation run (default 50000).
        num_eval_runs: Evaluation runs for confidence intervals (default 5).
        data_root: Root directory for all datasets (default './data').
        save_dir: Checkpoint save directory.
        log_dir: TensorBoard log directory.
        device: Target device ('cuda' or 'cpu').
        num_workers: DataLoader worker processes (default 4).
        seed: Random seed (default 42).
        normalize_mean: Per-channel normalisation mean (default [0.5,0.5,0.5]).
        normalize_std: Per-channel normalisation std (default [0.5,0.5,0.5]).
    """

    # ------------------------------------------------------------------
    # Default values (from config.yaml defaults section)
    # ------------------------------------------------------------------
    _DEFAULTS: Dict[str, Any] = {
        # Dataset
        "dataset": "cifar10",
        "image_size": 32,
        "in_channels": 3,
        "out_channels": 3,
        # Training
        "batch_size": 512,
        "training_steps": 100000,
        "learning_rate": 0.0001,
        "optimizer": "lion",
        "weight_decay": 0.0,
        # Noise schedule
        "s0": 10,
        "s1": 1280,
        "rho": 7.0,
        "sigma_min": 0.002,
        "sigma_max": 80.0,
        "sigma_data": 0.5,
        "P_mean": -1.1,
        "P_std": 2.0,
        # Architecture
        "model_channels": 128,
        "num_blocks": 3,
        "channel_mult": [1, 2, 2],
        "attn_resolutions": [],
        "dropout": 0.0,
        "embedding_type": "positional",
        # GC
        "mu": 0.5,
        "coupling": "gc",
        "ot_batch_size": 512,
        # EMA
        "ema_decay": 0.9999,
        # Loss
        "distance_fn": "pseudo_huber",
        "pseudo_huber_c": 0.00054,
        # Evaluation
        "eval_every": 10000,
        "num_eval_samples": 50000,
        "num_eval_runs": 5,
        # Paths
        "data_root": "./data",
        "save_dir": "./checkpoints",
        "log_dir": "./logs",
        # Device
        "device": "cuda",
        "num_workers": 4,
        "seed": 42,
        # Preprocessing
        "normalize_mean": [0.5, 0.5, 0.5],
        "normalize_std": [0.5, 0.5, 0.5],
    }

    def __init__(self, **kwargs: Any) -> None:
        """Initialise config with defaults, then apply keyword overrides.

        Args:
            **kwargs: Key-value pairs that override the defaults. Any key
                not in ``_DEFAULTS`` is also accepted (for forward
                compatibility with new config fields).
        """
        # Start from defaults
        for key, value in self._DEFAULTS.items():
            setattr(self, key, value)

        # Apply overrides
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_yaml(cls, path: str, dataset_key: Optional[str] = None) -> "Config":
        """Load configuration from a YAML file.

        Handles two YAML formats:
        1. **Flat dataset YAML** (e.g. ``configs/cifar10.yaml``): All keys
           are at the top level. This is the primary format used by the
           experiment scripts.
        2. **Master config.yaml** with nested sections: Contains a
           ``defaults`` section and dataset-specific override sections
           (e.g. ``cifar10``, ``celeba``). The ``dataset_key`` argument
           selects which section to merge over defaults.

        Args:
            path: Path to the YAML configuration file.
            dataset_key: If the YAML has nested sections (master config.yaml
                format), this specifies which dataset section to use
                (e.g. ``'cifar10'``). If ``None``, the YAML is treated as
                a flat dataset config.

        Returns:
            A ``Config`` instance with all fields populated.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            KeyError: If ``dataset_key`` is specified but not found in the
                YAML.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Configuration file not found: '{path}'. "
                "Check the --config argument."
            )

        with open(path, "r", encoding="utf-8") as f:
            raw: Dict[str, Any] = yaml.safe_load(f) or {}

        # Detect master config.yaml format (has 'defaults' key)
        if "defaults" in raw and isinstance(raw["defaults"], dict):
            # Master config format: merge defaults + dataset section
            merged: Dict[str, Any] = dict(raw["defaults"])

            if dataset_key is not None:
                if dataset_key not in raw:
                    raise KeyError(
                        f"Dataset key '{dataset_key}' not found in '{path}'. "
                        f"Available keys: {[k for k in raw if k != 'defaults']}"
                    )
                # Dataset section overrides defaults
                merged.update(raw[dataset_key])
            else:
                # No dataset key — use defaults only (warn user)
                print(
                    f"[Config] WARNING: Master config.yaml loaded without "
                    f"--dataset_key. Using defaults only. "
                    f"Available dataset sections: "
                    f"{[k for k in raw if k not in ('defaults', 'ablation_mu', 'ablation_hyperparams', 'analysis')]}"
                )
        else:
            # Flat dataset YAML format — use directly
            merged = dict(raw)

        return cls(**merged)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the config to a plain dictionary.

        Returns all non-private attributes as a dict. Used for checkpoint
        saving and JSON output.

        Returns:
            Dictionary mapping attribute names to their values.
        """
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        }

    def __repr__(self) -> str:
        """Return a human-readable summary of the configuration."""
        lines = ["Config("]
        for key, value in sorted(self.__dict__.items()):
            if not key.startswith("_"):
                lines.append(f"  {key}={value!r},")
        lines.append(")")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main orchestrator class
# ---------------------------------------------------------------------------


class Main:
    """Top-level orchestrator for iCT-GC experiments.

    Wires together all modules (data loading, model, EMA, training,
    evaluation) and dispatches to mode-specific methods based on the
    CLI ``--mode`` argument.

    The class is stateful: all shared objects (config, model, EMA, trainer,
    evaluator) are stored as instance attributes and reused across method
    calls.

    Attributes:
        args: Parsed CLI arguments from argparse.
        config: Flat Config object loaded from the dataset YAML.
        device: Target torch.device for all tensor operations.
        dataset_loader: DatasetLoader for train and eval splits.
        train_loader: DataLoader for the training split.
        eval_loader: DataLoader for the evaluation split.
        model: ConsistencyModel (online weights, gradient-tracked).
        ema: EMA wrapper for stop-gradient endpoint prediction.
        trainer: Trainer implementing Algorithm 1.
        evaluator: Evaluator computing FID/KID/IS.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialise all experiment components in dependency order.

        The initialisation order is critical:
        1. Config → 2. Seed → 3. Device → 4. Data → 5. Model →
        6. EMA → 7. Trainer → 8. Evaluator

        Args:
            args: Parsed CLI arguments. Must have attributes:
                - ``config`` (str): Path to dataset YAML.
                - ``mode`` (str): Execution mode.
                - ``checkpoint`` (Optional[str]): Checkpoint path.
                - ``seed`` (int): Random seed.
                - ``dataset_key`` (Optional[str]): For master config.yaml.
        """
        self.args: argparse.Namespace = args

        # ------------------------------------------------------------------
        # Step 1: Load configuration
        # ------------------------------------------------------------------
        dataset_key: Optional[str] = getattr(args, "dataset_key", None)
        self.config: Config = Config.from_yaml(
            path=args.config,
            dataset_key=dataset_key,
        )

        # Override seed from CLI if explicitly provided
        if hasattr(args, "seed") and args.seed is not None:
            self.config.seed = int(args.seed)

        # ------------------------------------------------------------------
        # Step 2: Set random seed for reproducibility
        # ------------------------------------------------------------------
        set_seed(self.config.seed)

        # ------------------------------------------------------------------
        # Step 3: Resolve device
        # ------------------------------------------------------------------
        self.device: torch.device = get_device(str(self.config.device))
        # Update config.device to the resolved device string
        self.config.device = str(self.device)

        print(
            f"[Main] Initialising iCT-GC experiment.\n"
            f"  Config: {args.config}\n"
            f"  Mode:   {args.mode}\n"
            f"  Device: {self.device}\n"
            f"  Seed:   {self.config.seed}"
        )

        # ------------------------------------------------------------------
        # Step 4: Create output directories
        # ------------------------------------------------------------------
        os.makedirs(str(self.config.save_dir), exist_ok=True)
        os.makedirs(str(self.config.log_dir), exist_ok=True)

        # Save a copy of the config for reproducibility
        config_save_path: str = os.path.join(
            str(self.config.save_dir), "config.json"
        )
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2, default=str)

        # ------------------------------------------------------------------
        # Step 5: Initialise data loaders
        # ------------------------------------------------------------------
        self.dataset_loader: DatasetLoader = DatasetLoader(self.config)
        self.train_loader = self.dataset_loader.get_train_loader()
        self.eval_loader = self.dataset_loader.get_eval_loader()

        # Get dataset statistics — sigma_data is used in c_skip/c_out
        sigma_data: float
        num_classes: int
        sigma_data, num_classes = self.dataset_loader.get_dataset_stats()

        # Update config with computed sigma_data (may differ from YAML default)
        self.config.sigma_data = sigma_data

        print(
            f"[Main] Dataset: {self.config.dataset} "
            f"({self.config.image_size}×{self.config.image_size}), "
            f"sigma_data={sigma_data:.4f}, "
            f"num_classes={num_classes}"
        )

        # ------------------------------------------------------------------
        # Step 6: Build model
        # ------------------------------------------------------------------
        self.model: ConsistencyModel = self._setup_model(
            checkpoint_path=args.checkpoint
        )

        num_params: int = count_parameters(self.model)
        print(f"[Main] Model parameters: {num_params:,}")

        # ------------------------------------------------------------------
        # Step 7: Build EMA
        # ------------------------------------------------------------------
        self.ema: EMA = EMA(
            model=self.model,
            decay=float(self.config.ema_decay),
        )

        # If resuming from checkpoint, load EMA state
        if args.checkpoint is not None and os.path.exists(args.checkpoint):
            ckpt: Dict[str, Any] = torch.load(
                args.checkpoint, map_location=self.device
            )
            if "ema_state_dict" in ckpt:
                self.ema.load_state_dict(ckpt["ema_state_dict"])
                print(
                    f"[Main] Loaded EMA state from checkpoint: "
                    f"'{args.checkpoint}'"
                )

        # ------------------------------------------------------------------
        # Step 8: Build trainer
        # ------------------------------------------------------------------
        self.trainer: Trainer = self._setup_trainer()

        # If resuming from checkpoint, restore trainer state
        if args.checkpoint is not None and os.path.exists(args.checkpoint):
            self.trainer._load_checkpoint(args.checkpoint)

        # ------------------------------------------------------------------
        # Step 9: Build evaluator
        # ------------------------------------------------------------------
        self.evaluator: Evaluator = self._setup_evaluator()

        print("[Main] Initialisation complete.")

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_model(
        self,
        checkpoint_path: Optional[str] = None,
    ) -> ConsistencyModel:
        """Instantiate and configure the ConsistencyModel.

        Builds a SongUNet backbone with the architecture parameters from
        config, wraps it in ConsistencyModel with EDM preconditioning, and
        optionally loads weights from a checkpoint.

        Args:
            checkpoint_path: Optional path to a ``.pt`` checkpoint file.
                If provided and the file exists, the model weights are loaded
                from ``ckpt['model_state_dict']``. EMA and optimizer states
                are loaded separately.

        Returns:
            A ``ConsistencyModel`` instance on ``self.device`` with all
            parameters initialised.
        """
        # Read architecture parameters from config
        img_resolution: int = int(self.config.image_size)
        in_channels: int = int(self.config.in_channels)
        out_channels: int = int(self.config.out_channels)
        model_channels: int = int(self.config.model_channels)

        # channel_mult: list of ints (e.g. [1, 2, 2])
        channel_mult: List[int] = list(self.config.channel_mult)

        # num_blocks: int or list of ints — SongUNet handles both
        num_blocks = self.config.num_blocks

        # attn_resolutions: list of ints (e.g. [16] or [])
        attn_resolutions: List[int] = list(self.config.attn_resolutions)

        # dropout: float or list of floats — SongUNet handles both
        dropout = self.config.dropout

        # embedding_type: 'positional' (default) or 'fourier'
        embedding_type: str = str(self.config.embedding_type)

        # Build SongUNet backbone
        net: SongUNet = SongUNet(
            img_resolution=img_resolution,
            in_channels=in_channels,
            out_channels=out_channels,
            model_channels=model_channels,
            channel_mult=channel_mult,
            num_blocks=num_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            embedding_type=embedding_type,
        )

        # Wrap with EDM preconditioning
        model: ConsistencyModel = ConsistencyModel(
            net=net,
            sigma_min=float(self.config.sigma_min),
            sigma_data=float(self.config.sigma_data),
        )

        # Move to target device
        model = model.to(self.device)

        # Optionally load weights from checkpoint
        if checkpoint_path is not None:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"Checkpoint file not found: '{checkpoint_path}'. "
                    "Check the --checkpoint argument."
                )
            ckpt: Dict[str, Any] = torch.load(
                checkpoint_path, map_location=self.device
            )
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
                print(
                    f"[Main] Loaded model weights from checkpoint: "
                    f"'{checkpoint_path}'"
                )
            else:
                print(
                    f"[Main] WARNING: Checkpoint '{checkpoint_path}' does not "
                    "contain 'model_state_dict'. Model weights not loaded."
                )

        return model

    def _setup_trainer(self) -> Trainer:
        """Instantiate the Trainer with the current model, EMA, and config.

        Returns:
            A ``Trainer`` instance ready to call ``train()``.
        """
        return Trainer(
            config=self.config,
            model=self.model,
            ema=self.ema,
            train_loader=self.train_loader,
        )

    def _setup_evaluator(self) -> Evaluator:
        """Instantiate the Evaluator with the current model and eval loader.

        Returns:
            An ``Evaluator`` instance ready to call ``evaluate()``.
        """
        return Evaluator(
            config=self.config,
            model=self.model,
            real_loader=self.eval_loader,
        )

    def _build_fresh_model(self, cfg: Config) -> ConsistencyModel:
        """Build a freshly initialised ConsistencyModel from a config.

        Used in ablation studies where multiple independent models are
        trained with different hyperparameters. Unlike ``_setup_model``,
        this method does not load any checkpoint.

        Args:
            cfg: Configuration object for the new model. May differ from
                ``self.config`` (e.g. different ``mu`` or ``coupling``).

        Returns:
            A freshly initialised ``ConsistencyModel`` on ``self.device``.
        """
        net: SongUNet = SongUNet(
            img_resolution=int(cfg.image_size),
            in_channels=int(cfg.in_channels),
            out_channels=int(cfg.out_channels),
            model_channels=int(cfg.model_channels),
            channel_mult=list(cfg.channel_mult),
            num_blocks=cfg.num_blocks,
            attn_resolutions=list(cfg.attn_resolutions),
            dropout=cfg.dropout,
            embedding_type=str(cfg.embedding_type),
        )

        model: ConsistencyModel = ConsistencyModel(
            net=net,
            sigma_min=float(cfg.sigma_min),
            sigma_data=float(cfg.sigma_data),
        )

        return model.to(self.device)

    # ------------------------------------------------------------------
    # Run dispatch
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Dispatch to the appropriate execution mode.

        Reads ``self.args.mode`` and calls the corresponding method.
        All mode methods are self-contained and handle their own output.

        Raises:
            ValueError: If ``args.mode`` is not a recognised mode.
        """
        mode: str = str(self.args.mode)

        dispatch: Dict[str, Any] = {
            "train": self.run_training,
            "eval": self.run_evaluation,
            "ablation_mu": self.run_ablation_mu,
            "compare_couplings": self.run_compare_couplings,
        }

        if mode not in dispatch:
            raise ValueError(
                f"Unknown mode '{mode}'. "
                f"Must be one of: {sorted(dispatch.keys())}."
            )

        print(f"[Main] Running mode: '{mode}'")
        dispatch[mode]()

    # ------------------------------------------------------------------
    # Mode: train
    # ------------------------------------------------------------------

    def run_training(self) -> None:
        """Run the full training loop and evaluate the final model.

        Calls ``trainer.train()`` for ``config.training_steps`` steps,
        then applies EMA weights and runs the full evaluation protocol
        (50k images, 5 runs) to produce the final FID/KID/IS numbers.

        The EMA model is used for final evaluation, matching the paper's
        protocol (Table 3 ablation confirms EMA is critical).
        """
        print(
            f"[Main] Starting training: {self.config.training_steps} steps, "
            f"dataset={self.config.dataset}, "
            f"coupling={self.config.coupling}, "
            f"mu={self.config.mu}"
        )

        # --- Main training loop ---
        self.trainer.train()

        print("[Main] Training complete. Running final evaluation...")

        # --- Apply EMA weights for evaluation ---
        # The paper evaluates with EMA weights (Table 3: removing EMA raises
        # FID from 5.95 to 6.73 on CIFAR-10).
        self.ema.apply_shadow(self.model)

        try:
            # --- Final evaluation ---
            self.run_evaluation(use_current_weights=True)
        finally:
            # Always restore online weights after evaluation
            self.ema.restore(self.model)

    # ------------------------------------------------------------------
    # Mode: eval
    # ------------------------------------------------------------------

    def run_evaluation(
        self,
        use_current_weights: bool = False,
    ) -> Dict[str, float]:
        """Evaluate the model and report FID/KID/IS with confidence intervals.

        Implements the paper's evaluation protocol (Appendix D):
        - 50,000 generated images vs 50,000 training images
        - 5 independent runs for mean ± std confidence intervals
        - Single-step generation (one NFE)

        Args:
            use_current_weights: If ``True``, use the model's current weights
                (caller is responsible for applying EMA if desired). If
                ``False`` (default, for standalone ``--mode eval``), load
                the best checkpoint and apply EMA weights.

        Returns:
            Dictionary with keys ``fid_mean``, ``fid_std``, ``kid_mean``,
            ``kid_std``, ``is_mean``, ``is_std``.
        """
        if not use_current_weights:
            # Load best checkpoint (or latest if best not available)
            best_ckpt_path: str = os.path.join(
                str(self.config.save_dir), "best.pt"
            )
            latest_ckpt_path: str = os.path.join(
                str(self.config.save_dir), "latest.pt"
            )

            ckpt_path: Optional[str] = None
            if os.path.exists(best_ckpt_path):
                ckpt_path = best_ckpt_path
                print(f"[Main] Loading best checkpoint: '{best_ckpt_path}'")
            elif os.path.exists(latest_ckpt_path):
                ckpt_path = latest_ckpt_path
                print(
                    f"[Main] Best checkpoint not found. "
                    f"Loading latest: '{latest_ckpt_path}'"
                )
            elif self.args.checkpoint is not None:
                ckpt_path = self.args.checkpoint
                print(
                    f"[Main] Loading checkpoint from --checkpoint: "
                    f"'{ckpt_path}'"
                )
            else:
                print(
                    "[Main] WARNING: No checkpoint found. "
                    "Evaluating with current (randomly initialised) weights."
                )

            if ckpt_path is not None:
                ckpt: Dict[str, Any] = torch.load(
                    ckpt_path, map_location=self.device
                )
                if "model_state_dict" in ckpt:
                    self.model.load_state_dict(ckpt["model_state_dict"])
                if "ema_state_dict" in ckpt:
                    self.ema.load_state_dict(ckpt["ema_state_dict"])

            # Apply EMA weights for evaluation
            self.ema.apply_shadow(self.model)

        try:
            # --- Run evaluation ---
            metrics: Dict[str, float] = self.evaluator.evaluate(
                num_samples=int(self.config.num_eval_samples),
                num_runs=int(self.config.num_eval_runs),
            )
        finally:
            if not use_current_weights:
                # Restore online weights
                self.ema.restore(self.model)

        # --- Print formatted results table ---
        self._print_metrics_table(metrics)

        # --- Save results to JSON ---
        results_with_meta: Dict[str, Any] = {
            "dataset": self.config.dataset,
            "coupling": self.config.coupling,
            "mu": self.config.mu,
            "training_steps": self.config.training_steps,
            "num_eval_samples": self.config.num_eval_samples,
            "num_eval_runs": self.config.num_eval_runs,
            **metrics,
        }

        results_path: str = os.path.join(
            str(self.config.save_dir), "eval_results.json"
        )
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results_with_meta, f, indent=2)

        print(f"[Main] Evaluation results saved to '{results_path}'.")

        return metrics

    # ------------------------------------------------------------------
    # Mode: ablation_mu
    # ------------------------------------------------------------------

    def run_ablation_mu(self) -> None:
        """Reproduce Figure 5: FID vs training steps for different μ values.

        Trains independent models with μ ∈ {0.0, 0.1, 0.3, 0.5, 0.7, 1.0}
        on CIFAR-10 and records FID at regular intervals to produce
        convergence curves.

        The μ values are read from ``config.yaml`` (``ablation_mu.mu_values``).
        If not present in the loaded config, defaults to the paper's values.

        Results are saved as:
        - ``ablation_mu_results.json``: Raw FID curves per μ value
        - ``ablation_mu_fid_curves.png``: Plot matching Figure 5

        Note: This is computationally expensive — it trains one model per
        μ value. For a quick sanity check, reduce ``training_steps`` in the
        config.
        """
        # Read μ values from config (ablation_mu section) or use paper defaults
        mu_values: List[float] = list(
            getattr(self.config, "mu_values", [0.0, 0.1, 0.3, 0.5, 0.7, 1.0])
        )

        # More frequent evaluation for convergence curves
        eval_every_ablation: int = int(
            getattr(self.config, "eval_every", 5000)
        )

        print(
            f"[Main] Starting μ ablation study.\n"
            f"  μ values: {mu_values}\n"
            f"  Training steps per model: {self.config.training_steps}\n"
            f"  Eval every: {eval_every_ablation} steps"
        )

        # Results storage: {mu: {'steps': [...], 'fid': [...]}}
        ablation_results: Dict[