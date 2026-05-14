## Code: main.py

```python
# main.py
"""Main entry point for the MoE-POT (Mixture-of-Experts Pre-training Operator Transformer).

Orchestrates all experiment modes: pre-training, fine-tuning, downstream task
fine-tuning, zero-shot evaluation, ablation studies, and interpretability analysis.

Usage examples:
    # Pre-training (single GPU):
    python main.py --config config.yaml --mode pretrain --model_size tiny

    # Pre-training (multi-GPU with torchrun):
    torchrun --nproc_per_node=8 main.py --config config.yaml --mode pretrain --model_size small

    # Fine-tuning on all 6 datasets:
    python main.py --config config.yaml --mode finetune --model_size tiny \
        --checkpoint ./checkpoints/tiny/best.pt

    # Fine-tuning on a specific dataset:
    python main.py --config config.yaml --mode finetune --model_size tiny \
        --checkpoint ./checkpoints/tiny/best.pt --dataset fno_ns_1e5

    # Zero-shot evaluation:
    python main.py --config config.yaml --mode evaluate --model_size tiny \
        --checkpoint ./checkpoints/tiny/best.pt

    # Ablation studies:
    python main.py --config config.yaml --mode ablation --model_size tiny

    # Interpretability analysis:
    python main.py --config config.yaml --mode interpretability --model_size tiny \
        --checkpoint ./checkpoints/tiny/best.pt

    # Downstream tasks:
    python main.py --config config.yaml --mode downstream --model_size tiny \
        --checkpoint ./checkpoints/tiny/best.pt
"""

import argparse
import copy
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Internal module imports
# ---------------------------------------------------------------------------
from data.datasets import MultiPDEDataset, PDEDataset
from data.sampler import BalancedMultiDatasetSampler
from evaluation.evaluator import Evaluator
from evaluation.interpretability import InterpretabilityAnalyzer
from models.moe_pot import MoEPOT
from training.finetuner import Finetuner
from training.trainer import Trainer
from utils.checkpoint import Checkpointer
from utils.logger import Logger


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Flat configuration object for MoE-POT experiments.

    Populated from config.yaml by Config.from_yaml(). All hyperparameters
    from the nested YAML structure are flattened into direct attributes,
    making them accessible as config.attn_dim, config.learning_rate, etc.

    Attributes correspond to config.yaml sections:
        - models.{size}: attn_dim, mlp_dim, num_layers, num_heads,
          num_routed_experts, num_shared_experts, top_k
        - architecture: patch_size, input_timesteps, target_resolution,
          max_channels, modes_x, modes_y, load_balance_weight
        - pretraining/finetuning/downstream: num_epochs, learning_rate,
          weight_decay, beta1, beta2, batch_size, warmup_epochs,
          noise_injection, noise_scale, freeze_router
        - paths: data_root, checkpoint_dir, log_dir, results_dir
        - logging: use_wandb, log_interval, save_interval
        - datasets: nested dict with pretraining and downstream dataset configs
    """

    # --- Model architecture (from config.yaml models.{size}) ---
    model_size: str = "tiny"
    attn_dim: int = 512
    mlp_dim: int = 512
    num_layers: int = 4
    num_heads: int = 4
    num_routed_experts: int = 16
    num_shared_experts: int = 2
    top_k: int = 4

    # --- Architecture hyperparameters (from config.yaml architecture) ---
    patch_size: int = 8
    input_timesteps: int = 10
    target_resolution: int = 128
    max_channels: int = 4
    modes_x: int = 8
    modes_y: int = 8
    load_balance_weight: float = 0.1

    # --- Training hyperparameters ---
    num_epochs: int = 1000
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    beta1: float = 0.9
    beta2: float = 0.9
    batch_size: int = 20
    warmup_epochs: int = 200
    noise_injection: bool = True
    noise_scale: float = 0.01
    freeze_router: bool = False

    # --- Paths (from config.yaml paths) ---
    data_root: str = "./data/raw"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    results_dir: str = "./results"

    # --- Logging (from config.yaml logging) ---
    use_wandb: bool = False
    log_interval: int = 10
    save_interval: int = 50

    # --- Dataset configs (nested dict, not flattened) ---
    datasets: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(
        cls,
        path: str,
        model_size: str = "tiny",
        mode: str = "pretrain",
    ) -> "Config":
        """Loads and flattens the hierarchical YAML config into a Config object.

        Merges model-specific settings, architecture settings, and
        mode-specific training settings into a single flat Config.

        Args:
            path: Path to config.yaml.
            model_size: One of 'tiny', 'small', 'medium'. Selects the
                model sub-config from config['models'][model_size].
            mode: Experiment mode. Determines which training config section
                to use: 'pretrain' → pretraining, 'finetune' → finetuning,
                'downstream' → downstream, others → pretraining defaults.

        Returns:
            Populated Config instance with all hyperparameters as flat attrs.

        Raises:
            FileNotFoundError: If the config file does not exist.
            KeyError: If required config sections are missing.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            cfg: Dict[str, Any] = yaml.safe_load(f)

        # --- Model-specific config ---
        if model_size not in cfg.get("models", {}):
            raise KeyError(
                f"Model size '{model_size}' not found in config.yaml models section. "
                f"Available sizes: {list(cfg.get('models', {}).keys())}"
            )
        model_cfg: Dict[str, Any] = cfg["models"][model_size]

        # --- Architecture config ---
        arch_cfg: Dict[str, Any] = cfg.get("architecture", {})

        # --- Training config based on mode ---
        pretrain_cfg: Dict[str, Any] = cfg.get("pretraining", {})
        finetune_cfg: Dict[str, Any] = cfg.get("finetuning", {})
        downstream_cfg: Dict[str, Any] = cfg.get("downstream", {})

        if mode == "pretrain":
            train_cfg = pretrain_cfg
        elif mode == "finetune":
            train_cfg = finetune_cfg
        elif mode == "downstream":
            train_cfg = downstream_cfg
        else:
            # evaluate, ablation, interpretability: use pretraining defaults
            train_cfg = pretrain_cfg

        # --- Paths and logging ---
        paths_cfg: Dict[str, Any] = cfg.get("paths", {})
        log_cfg: Dict[str, Any] = cfg.get("logging", {})

        # --- Build flat Config ---
        # weight_decay, beta1, beta2, batch_size, noise_scale always from
        # pretraining section (same optimizer settings across all modes).
        return cls(
            # Model architecture
            model_size=str(model_cfg.get("model_size", model_size)),
            attn_dim=int(model_cfg.get("attn_dim", 512)),
            mlp_dim=int(model_cfg.get("mlp_dim", 512)),
            num_layers=int(model_cfg.get("num_layers", 4)),
            num_heads=int(model_cfg.get("num_heads", 4)),
            num_routed_experts=int(model_cfg.get("num_routed_experts", 16)),
            num_shared_experts=int(model_cfg.get("num_shared_experts", 2)),
            top_k=int(model_cfg.get("top_k", 4)),
            # Architecture
            patch_size=int(arch_cfg.get("patch_size", 8)),
            input_timesteps=int(arch_cfg.get("input_timesteps", 10)),
            target_resolution=int(arch_cfg.get("target_resolution", 128)),
            max_channels=int(arch_cfg.get("max_channels", 4)),
            modes_x=int(arch_cfg.get("modes_x", 8)),
            modes_y=int(arch_cfg.get("modes_y", 8)),
            load_balance_weight=float(arch_cfg.get("load_balance_weight", 0.1)),
            # Training (mode-specific)
            num_epochs=int(train_cfg.get("num_epochs", 1000)),
            warmup_epochs=int(train_cfg.get("warmup_epochs", 200)),
            noise_injection=bool(train_cfg.get("noise_injection", True)),
            freeze_router=bool(train_cfg.get("freeze_router", False)),
            # Training (always from pretraining section)
            learning_rate=float(pretrain_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(pretrain_cfg.get("weight_decay", 1e-6)),
            beta1=float(pretrain_cfg.get("beta1", 0.9)),
            beta2=float(pretrain_cfg.get("beta2", 0.9)),
            batch_size=int(pretrain_cfg.get("batch_size", 20)),
            noise_scale=float(pretrain_cfg.get("noise_scale", 0.01)),
            # Paths
            data_root=str(paths_cfg.get("data_root", "./data/raw")),
            checkpoint_dir=str(paths_cfg.get("checkpoint_dir", "./checkpoints")),
            log_dir=str(paths_cfg.get("log_dir", "./logs")),
            results_dir=str(paths_cfg.get("results_dir", "./results")),
            # Logging
            use_wandb=bool(log_cfg.get("use_wandb", False)),
            log_interval=int(log_cfg.get("log_interval", 10)),
            save_interval=int(log_cfg.get("save_interval", 50)),
            # Dataset configs (nested dict preserved as-is)
            datasets=cfg.get("datasets", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the Config to a JSON-compatible dictionary.

        Returns:
            Dictionary with all config attributes. Nested dicts (datasets)
            are preserved as-is.
        """
        return asdict(self)


# ---------------------------------------------------------------------------
# Distributed training utilities
# ---------------------------------------------------------------------------

def setup_distributed(local_rank: int) -> Tuple[int, int, bool]:
    """Initializes the distributed process group if running in multi-GPU mode.

    Checks for the WORLD_SIZE environment variable set by torchrun/launch.
    If WORLD_SIZE > 1, initializes the NCCL process group and sets the
    current CUDA device to local_rank.

    Args:
        local_rank: Local GPU rank for this process. Set by torchrun via
            the LOCAL_RANK environment variable or --local_rank argument.

    Returns:
        Tuple (rank, world_size, is_distributed) where:
          - rank: Global rank of this process (0 to world_size-1).
          - world_size: Total number of processes.
          - is_distributed: True if multi-GPU training is active.
    """
    world_size: int = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed: bool = world_size > 1

    if is_distributed:
        # Initialize NCCL process group for GPU-to-GPU communication.
        # NCCL is the recommended backend for GPU training.
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
        )
        rank: int = dist.get_rank()

        # Set the current CUDA device for this process.
        # Each process uses a different GPU identified by local_rank.
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    else:
        rank = 0

    return rank, world_size, is_distributed


def cleanup_distributed() -> None:
    """Destroys the distributed process group after training completes.

    Should be called at the end of main() when distributed training was
    initialized. Safe to call even if dist.is_initialized() is False.
    """
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the MoE-POT experiment runner.

    Returns:
        Parsed argument namespace with all experiment configuration options.
    """
    parser = argparse.ArgumentParser(
        description="MoE-POT: Mixture-of-Experts Pre-training Operator Transformer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["pretrain", "finetune", "downstream", "evaluate", "ablation", "interpretability"],
        help=(
            "Experiment mode: "
            "'pretrain' = pre-train on 6 PDE datasets; "
            "'finetune' = fine-tune on individual datasets (200 epochs); "
            "'downstream' = fine-tune on downstream tasks (500 epochs); "
            "'evaluate' = zero-shot evaluation of a checkpoint; "
            "'ablation' = run ablation studies (Nr, TopK, heads, patch_size); "
            "'interpretability' = router classification accuracy analysis."
        ),
    )

    # Model configuration
    parser.add_argument(
        "--model_size",
        type=str,
        default="tiny",
        choices=["tiny", "small", "medium"],
        help=(
            "Model size variant: "
            "tiny (30M total / 17M activated), "
            "small (166M / 90M), "
            "medium (489M / 288M)."
        ),
    )

    # Checkpoint path (required for finetune, evaluate, interpretability)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a pre-trained checkpoint .pt file. "
            "Required for finetune, downstream, evaluate, and interpretability modes. "
            "If not provided for finetune/downstream, defaults to "
            "{checkpoint_dir}/{model_size}/best.pt."
        ),
    )

    # Dataset selection for fine-tuning
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "Specific dataset key for finetune/downstream mode. "
            "If not provided, fine-tunes on all datasets in the config. "
            "Example: 'fno_ns_1e5', 'pdebench_swe', 'cfdbench'."
        ),
    )

    # Distributed training
    parser.add_argument(
        "--local_rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", "0")),
        help=(
            "Local GPU rank for distributed training. "
            "Set automatically by torchrun via LOCAL_RANK env variable."
        ),
    )

    # Logging control
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        default=False,
        help="Disable Weights & Biases logging even if config.yaml has use_wandb: true.",
    )

    # Results output
    parser.add_argument(
        "--results_file",
        type=str,
        default=None,
        help=(
            "Optional path to save results as a JSON file. "
            "If not provided, results are saved to {results_dir}/{mode}_results.json."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Results saving utility
# ---------------------------------------------------------------------------

def save_results(
    results: Dict[str, Any],
    results_dir: str,
    experiment_name: str,
    custom_path: Optional[str] = None,
) -> None:
    """Saves experiment results to a JSON file.

    Args:
        results: Dictionary of results to save. Values must be JSON-serializable
            (floats, ints, strings, lists, dicts). Tensors are not supported
            directly — convert to lists before passing.
        results_dir: Directory where the results file will be written.
            Created if it does not exist.
        experiment_name: Base name for the results file. The file will be
            saved as {results_dir}/{experiment_name}_results.json.
        custom_path: Optional override for the full output path. If provided,
            results_dir and experiment_name are ignored.
    """
    os.makedirs(results_dir, exist_ok=True)

    if custom_path is not None:
        output_path: str = custom_path
    else:
        output_path = os.path.join(results_dir, f"{experiment_name}_results.json")

    # Convert any non-serializable values (e.g., numpy floats) to Python floats.
    def _make_serializable(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_make_serializable(v) for v in obj]
        elif hasattr(obj, "item"):
            # numpy scalar or torch scalar
            return obj.item()
        elif isinstance(obj, float):
            return obj
        elif isinstance(obj, int):
            return obj
        elif isinstance(obj, str):
            return obj
        else:
            return str(obj)

    serializable_results: Dict[str, Any] = _make_serializable(results)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2)

    print(f"[Results] Saved to {output_path}")


# ---------------------------------------------------------------------------
# Data pipeline construction helpers
# ---------------------------------------------------------------------------

# Ordered list of the 6 pre-training dataset keys from config.yaml.
# Order must match the dataset_idx convention in MultiPDEDataset.
_PRETRAIN_DATASET_KEYS: List[str] = [
    "fno_ns_1e5",
    "fno_ns_1e3",
    "pdebench_cns_0p1_0p01",
    "pdebench_swe",
    "pdebench_dr",
    "cfdbench",
]

# Downstream task dataset keys from config.yaml datasets.downstream.
_DOWNSTREAM_DATASET_KEYS: List[str] = [
    "fno_ns_1e4",
    "pdebench_cns_1_0p01",
    "pdearena",
]


def _compute_per_gpu_batch_size(total_batch_size: int, world_size: int) -> int:
    """Computes the per-GPU batch size from the total batch size.

    With total_batch_size=20 and world_size=8: per_gpu = 2 (floor division).
    Ensures at least 1 sample per GPU.

    Args:
        total_batch_size: Total batch size across all GPUs (config.batch_size).
        world_size: Number of GPUs/processes.

    Returns:
        Per-GPU batch size (at least 1).
    """
    per_gpu: int = max(1, total_batch_size // world_size)
    return per_gpu


def build_pretraining_dataloaders(
    config: Config,
    rank: int,
    world_size: int,
) -> Tuple[DataLoader, Dict[str, DataLoader]]:
    """Builds the multi-dataset training loader and per-dataset validation loaders.

    Constructs PDEDataset instances for all 6 pre-training datasets, wraps
    them in a MultiPDEDataset, and creates a BalancedMultiDatasetSampler
    for inverse-size-weighted sampling during pre-training.

    Gracefully handles missing dataset files by skipping datasets that
    cannot be loaded (FileNotFoundError), logging a warning. This allows
    partial pre-training when not all datasets are available.

    Args:
        config: Config object with data_root, target_resolution,
            input_timesteps, batch_size, and dataset configs.
        rank: Global rank of this process (0 for single-GPU).
        world_size: Total number of processes.

    Returns:
        Tuple (train_loader, val_loaders) where:
          - train_loader: DataLoader backed by BalancedMultiDatasetSampler
            for the combined multi-PDE training set.
          - val_loaders: Dict mapping dataset key to its test DataLoader.
            Only includes datasets that were successfully loaded.
    """
    per_gpu_batch_size: int = _compute_per_gpu_batch_size(
        config.batch_size, world_size
    )

    # --- Build individual PDEDatasets ---
    train_datasets: List[PDEDataset] = []
    test_datasets: List[PDEDataset] = []
    loaded_keys: List[str] = []

    for dataset_key in _PRETRAIN_DATASET_KEYS:
        try:
            train_ds: PDEDataset = PDEDataset(
                name=dataset_key,
                data_root=config.data_root,
                split="train",
                target_resolution=config.target_resolution,
                input_timesteps=config.input_timesteps,
            )
            test_ds: PDEDataset = PDEDataset(
                name=dataset_key,
                data_root=config.data_root,
                split="test",
                target_resolution=config.target_resolution,
                input_timesteps=config.input_timesteps,
            )
            train_datasets.append(train_ds)
            test_datasets.append(test_ds)
            loaded_keys.append(dataset_key)

            if rank == 0:
                print(
                    f"[Data] Loaded '{dataset_key}': "
                    f"train={len(train_ds)}, test={len(test_ds)}"
                )

        except FileNotFoundError as e:
            if rank == 0:
                print(
                    f"[Data] WARNING: Could not load dataset '{dataset_key}': {e}. "
                    f"Skipping this dataset."
                )
        except Exception as e:  # pylint: disable=broad-except
            if rank == 0:
                print(
                    f"[Data] WARNING: Error loading dataset '{dataset_key}': {e}. "
                    f"Skipping this dataset."
                )

    if not train_datasets:
        raise RuntimeError(
            "No datasets could be loaded. Check that data files exist in "
            f"'{config.data_root}' and that dataset names match the expected format."
        )

    # --- Build MultiPDEDataset ---
    multi_train: MultiPDEDataset = MultiPDEDataset(train_datasets)

    # --- Build BalancedMultiDatasetSampler ---
    # w_k = 1 for all datasets (config.yaml pretraining.dataset_weights: all 1)
    dataset_sizes: List[int] = multi_train.get_dataset_sizes()
    weights: List[float] = [1.0] * len(dataset_sizes)
    total_samples: int = sum(dataset_sizes)

    sampler: BalancedMultiDatasetSampler = BalancedMultiDatasetSampler(
        dataset_sizes=dataset_sizes,
        weights=weights,
        total_samples=total_samples,
    )

    # --- Build training DataLoader ---
    # num_workers=4 for efficient data loading; pin_memory=True for async
    # CPU→GPU transfer; drop_last=True to avoid incomplete batches.
    train_loader: DataLoader = DataLoader(
        multi_train,
        batch_size=per_gpu_batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    # --- Build per-dataset validation DataLoaders ---
    val_loaders: Dict[str, DataLoader] = {}
    for dataset_key, test_ds in zip(loaded_keys, test_datasets):
        val_loaders[dataset_key] = DataLoader(
            test_ds,
            batch_size=per_gpu_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    if rank == 0:
        print(
            f"[Data] Pre-training: {len(train_datasets)} datasets loaded, "
            f"total train samples: {total_samples}, "
            f"per-GPU batch size: {per_gpu_batch_size}"
        )

    return train_loader, val_loaders


def build_finetune_dataloaders(
    config: Config,
    dataset_key: str,
    world_size: int,
    is_downstream: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Builds training and validation DataLoaders for a single dataset.

    Used for both dataset fine-tuning (200 epochs on pre-training datasets)
    and downstream task fine-tuning (500 epochs on NS 1e-4, CNS 1 0.01, PDEArena).

    Args:
        config: Config object with data_root, target_resolution,
            input_timesteps, and batch_size.
        dataset_key: Dataset identifier string (e.g., 'fno_ns_1e5',
            'pdearena'). Must be a valid PDEDataset name.
        world_size: Total number of processes for per-GPU batch size calculation.
        is_downstream: If True, uses downstream dataset split sizes (train=2000,
            test=200). If False, uses pre-training split sizes from config.

    Returns:
        Tuple (train_loader, val_loader) where both are DataLoaders for the
        specified dataset's train and test splits respectively.

    Raises:
        FileNotFoundError: If the dataset file cannot be found.
    """
    per_gpu_batch_size: int = _compute_per_gpu_batch_size(
        config.batch_size, world_size
    )

    # Build train and test PDEDataset instances.
    train_ds: PDEDataset = PDEDataset(
        name=dataset_key,
        data_root=config.data_root,
        split="train",
        target_resolution=config.target_resolution,
        input_timesteps=config.input_timesteps,
    )
    test_ds: PDEDataset = PDEDataset(
        name=dataset_key,
        data_root=config.data_root,
        split="test",
        target_resolution=config.target_resolution,
        input_timesteps=config.input_timesteps,
    )

    # Training DataLoader with shuffling for fine-tuning.
    train_loader: DataLoader = DataLoader(
        train_ds,
        batch_size=per_gpu_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    # Validation DataLoader without shuffling for deterministic evaluation.
    val_loader: DataLoader = DataLoader(
        test_ds,
        batch_size=per_gpu_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Model construction helper
# ---------------------------------------------------------------------------

def build_model(
    config: Config,
    device: torch.device,
    is_distributed: bool = False,
    local_rank: int = 0,
    rank: int = 0,
) -> torch.nn.Module:
    """Constructs and initializes a MoEPOT model from the config.

    Creates the model, moves it to the target device, and optionally wraps
    it in DistributedDataParallel for multi-GPU training.

    Args:
        config: Config object with all model architecture hyperparameters.
        device: Target device for the model.
        is_distributed: Whether to wrap the model in DDP.
        local_rank: Local GPU rank for DDP device assignment.
        rank: Global rank for logging (only rank 0 prints parameter counts).

    Returns:
        The model, either a raw MoEPOT or a DistributedDataParallel-wrapped
        MoEPOT. Use model.module to access the underlying MoEPOT when DDP
        is active.
    """
    # Instantiate the MoEPOT model from config.
    model: MoEPOT = MoEPOT(config)
    model = model.to(device)

    # Log parameter counts on rank 0.
    if rank == 0:
        total_params, activated_params = model.count_parameters()
        print(
            f"[Model] MoE-POT-{config.model_size.capitalize()}: "
            f"Total params: {total_params / 1e6:.1f}M, "
            f"Activated params: {activated_params / 1e6:.1f}M"
        )

    # Wrap in DistributedDataParallel for multi-GPU training.
    if is_distributed:
        # find_unused_parameters=True is required for MoE because only top_k=4
        # out of 16 routed experts are activated per forward pass. The other
        # 12 experts produce no output and receive no gradient in a given batch.
        # Without this flag, DDP raises a runtime error about unused parameters.
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    return model


def _get_raw_model(model: torch.nn.Module) -> MoEPOT:
    """Extracts the underlying MoEPOT from a potentially DDP-wrapped model.

    Args:
        model: Either a raw MoEPOT or a DistributedDataParallel wrapper.

    Returns:
        The underlying MoEPOT instance.
    """
    if hasattr(model, "module"):
        return model.module
    return model


def _find_epoch_checkpoints(checkpoint_dir: str) -> List[str]:
    """Scans a checkpoint directory for epoch-specific checkpoint files.

    Looks for files matching the pattern 'epoch_*.pt' and returns them
    sorted by epoch number. Used for tracking routing evolution across
    training epochs in the interpretability analysis.

    Args:
        checkpoint_dir: Directory to scan for checkpoint files.

    Returns:
        Sorted list of absolute paths to epoch checkpoint files.
        Returns an empty list