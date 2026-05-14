## Code: main.py

```python
## main.py
"""Entry point for the NFIG (Next-Frequency Image Generation) framework.

Dispatches to one of four execution modes based on the --mode CLI argument:
  - train_frvae:  Train the Frequency-guided Residual-quantized VAE tokenizer.
  - train_nfig:   Train the NFIG Transformer using a frozen FR-VAE tokenizer.
  - evaluate:     Compute rFID, gFID, IS, Precision, Recall on ImageNet val.
  - sample:       Generate images using the full NFIG pipeline.

Usage examples:
    # Phase 1: Train FR-VAE tokenizer
    python main.py --config config.yaml --mode train_frvae

    # Phase 2: Train NFIG Transformer (requires trained FR-VAE)
    python main.py --config config.yaml --mode train_nfig \
        --frvae_checkpoint checkpoints/frvae_epoch_0099.pt

    # Evaluate both models
    python main.py --config config.yaml --mode evaluate \
        --frvae_checkpoint checkpoints/frvae_epoch_0099.pt \
        --nfig_checkpoint checkpoints/nfig_transformer_epoch_0349.pt

    # Sample images
    python main.py --config config.yaml --mode sample \
        --frvae_checkpoint checkpoints/frvae_epoch_0099.pt \
        --nfig_checkpoint checkpoints/nfig_transformer_epoch_0349.pt \
        --class_label 207 --num_samples 16

    # Distributed training (torchrun)
    torchrun --nproc_per_node=8 main.py --config config.yaml --mode train_nfig \
        --frvae_checkpoint checkpoints/frvae_epoch_0099.pt
"""

import argparse
import logging
import os
import random
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torchvision

from data.imagenet_dataset import ImageNetDataset
from evaluation.evaluator import Evaluator
from inference.sampler import NFIGSampler
from models.frvae.frvae import FRVAE
from models.transformer.nfig_transformer import NFIGTransformer
from training.frvae_trainer import FRVAETrainer
from training.nfig_trainer import NFIGTrainer
from utils.checkpoint import CheckpointManager
from utils.config import Config

# ---------------------------------------------------------------------------
# Module-level logger — configured in _setup_logging().
# ---------------------------------------------------------------------------
logger: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(log_dir: str, rank: int = 0) -> None:
    """Configure the Python logging system.

    Sets up a StreamHandler (stdout) and optionally a FileHandler writing to
    `log_dir/main.log`. Only rank 0 writes to the file in distributed runs;
    all ranks write to stdout at INFO level.

    Args:
        log_dir: Directory where `main.log` will be written.
            Created if it does not exist.
        rank: Process rank. Only rank 0 creates the file handler.
    """
    os.makedirs(log_dir, exist_ok=True)

    # Root logger configuration.
    root_logger: logging.Logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove any pre-existing handlers to avoid duplicate output.
    root_logger.handlers.clear()

    # Formatter: timestamp + level + logger name + message.
    formatter: logging.Formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # StreamHandler: all ranks log to stdout.
    stream_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    # FileHandler: only rank 0 writes to file.
    if rank == 0:
        log_file: str = os.path.join(log_dir, "main.log")
        file_handler: logging.FileHandler = logging.FileHandler(
            log_file, mode="a", encoding="utf-8"
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all relevant libraries.

    Sets seeds for Python's random module, NumPy, PyTorch CPU, and all
    CUDA devices. Optionally enables deterministic CUDA operations.

    Args:
        seed: Integer seed value. Default 42 from CLI argument.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic CUDA operations for full reproducibility.
    # Note: this may reduce performance on some operations.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info("Random seed set to %d.", seed)


# ---------------------------------------------------------------------------
# Distributed training helpers
# ---------------------------------------------------------------------------

def _init_distributed(local_rank: int, backend: str = "nccl") -> int:
    """Initialize the PyTorch distributed process group.

    Reads the MASTER_ADDR, MASTER_PORT, RANK, and WORLD_SIZE environment
    variables set by torchrun. Falls back to single-process mode if these
    are not set.

    Args:
        local_rank: Local GPU index on this node. Injected by torchrun via
            --local_rank or the LOCAL_RANK environment variable.
        backend: Distributed backend. From config.training.backend = "nccl".

    Returns:
        The global rank of this process (0 for single-GPU or master process).
    """
    if not dist.is_available():
        logger.warning(
            "torch.distributed is not available. Running in single-process mode."
        )
        return 0

    # Check if torchrun has set the required environment variables.
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        logger.info(
            "RANK/WORLD_SIZE environment variables not set. "
            "Running in single-process mode (no DDP)."
        )
        return 0

    # Initialize the process group.
    dist.init_process_group(backend=backend)

    # Set the CUDA device for this process.
    torch.cuda.set_device(local_rank)

    rank: int = dist.get_rank()
    world_size: int = dist.get_world_size()

    logger.info(
        "Distributed training initialized: rank=%d, world_size=%d, "
        "backend='%s', local_rank=%d.",
        rank,
        world_size,
        backend,
        local_rank,
    )

    return rank


def _cleanup_distributed() -> None:
    """Destroy the distributed process group if it was initialized.

    Called in the finally block of main() to ensure clean shutdown even
    if training raises an exception.
    """
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed process group destroyed.")


def _is_main_process() -> bool:
    """Return True if this is the main (rank 0) process.

    In single-GPU mode, always returns True.
    In distributed mode, returns True only for rank 0.

    Returns:
        Boolean indicating whether this is the main process.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the NFIG entry point.

    Returns:
        Parsed argument namespace with the following attributes:
            - config (str): Path to config.yaml.
            - mode (str): Execution mode — one of
              ['train_frvae', 'train_nfig', 'evaluate', 'sample'].
            - frvae_checkpoint (Optional[str]): Path to FR-VAE checkpoint.
            - nfig_checkpoint (Optional[str]): Path to NFIG Transformer checkpoint.
            - resume (Optional[str]): Path to checkpoint to resume training from.
            - local_rank (int): Local GPU rank (injected by torchrun).
            - seed (int): Random seed for reproducibility. Default 42.
            - output_dir (Optional[str]): Override config.evaluation.output_dir.
            - num_samples (Optional[int]): Override config.evaluation.num_samples.
            - class_label (Optional[int]): Single class label for sample mode.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="NFIG: Next-Frequency Image Generation via Frequency Ordering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Required arguments ---
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
        choices=["train_frvae", "train_nfig", "evaluate", "sample"],
        help=(
            "Execution mode: "
            "'train_frvae' trains the FR-VAE tokenizer; "
            "'train_nfig' trains the NFIG Transformer; "
            "'evaluate' computes rFID/gFID/IS/Precision/Recall; "
            "'sample' generates images."
        ),
    )

    # --- Checkpoint arguments ---
    parser.add_argument(
        "--frvae_checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a trained FR-VAE checkpoint (.pt file). "
            "Required for modes: train_nfig, evaluate, sample."
        ),
    )
    parser.add_argument(
        "--nfig_checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a trained NFIG Transformer checkpoint (.pt file). "
            "Required for modes: evaluate, sample."
        ),
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Path to a checkpoint to resume training from. "
            "Applicable for modes: train_frvae, train_nfig."
        ),
    )

    # --- Distributed training ---
    parser.add_argument(
        "--local_rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", 0)),
        help=(
            "Local GPU rank for distributed training. "
            "Injected automatically by torchrun via the LOCAL_RANK environment variable."
        ),
    )

    # --- Reproducibility ---
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    # --- Optional overrides ---
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override config.evaluation.output_dir for generated image output.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Override config.evaluation.num_samples for FID/IS computation.",
    )
    parser.add_argument(
        "--class_label",
        type=int,
        default=None,
        help=(
            "Single ImageNet class label [0, 999] for sample mode. "
            "If not provided, samples uniformly across all 1000 classes."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------

def _validate_config(config: Config, mode: str) -> None:
    """Validate configuration consistency for the given execution mode.

    Checks critical invariants that must hold for the NFIG framework to
    function correctly. These are in addition to the validation performed
    by Config._validate() during YAML loading.

    Args:
        config: Fully loaded and validated Config dataclass.
        mode: Execution mode string.

    Raises:
        ValueError: If any validation check fails.
    """
    # Verify total token count matches scale factors.
    # This is critical for the block-wise causal attention mask dimensions.
    expected_tokens: int = sum(s * s for s in config.frvae.scale_factors)
    if config.frvae.total_tokens != expected_tokens:
        raise ValueError(
            f"config.frvae.total_tokens={config.frvae.total_tokens} does not match "
            f"sum(s*s for s in scale_factors)={expected_tokens}. "
            f"scale_factors={config.frvae.scale_factors}. "
            "The block-wise causal attention mask requires exactly total_tokens=680."
        )

    # Verify FR-VAE and NFIG Transformer share the same scale factors.
    if config.frvae.scale_factors != config.nfig.scale_factors:
        raise ValueError(
            f"config.frvae.scale_factors={config.frvae.scale_factors} must match "
            f"config.nfig.scale_factors={config.nfig.scale_factors}. "
            "Both components must use the same frequency band scale sequence."
        )

    # Verify shared codebook size.
    if config.frvae.codebook_size != config.nfig.codebook_size:
        raise ValueError(
            f"config.frvae.codebook_size={config.frvae.codebook_size} must equal "
            f"config.nfig.codebook_size={config.nfig.codebook_size}. "
            "The transformer vocabulary must match the FR-VAE codebook."
        )

    # Verify null class ID equals num_classes.
    if config.nfig.null_class_id != config.nfig.num_classes:
        raise ValueError(
            f"config.nfig.null_class_id={config.nfig.null_class_id} must equal "
            f"config.nfig.num_classes={config.nfig.num_classes}. "
            "The null class index must be exactly one past the last real class."
        )

    # Verify CUDA availability when device="cuda" is configured.
    if config.training.device == "cuda" and not torch.cuda.is_available():
        logger.warning(
            "config.training.device='cuda' but CUDA is not available. "
            "Falling back to CPU. This will be very slow for training."
        )

    logger.info(
        "Configuration validated for mode='%s'. "
        "total_tokens=%d, codebook_size=%d, scale_factors=%s.",
        mode,
        config.frvae.total_tokens,
        config.frvae.codebook_size,
        config.frvae.scale_factors,
    )


def _validate_checkpoint_args(
    mode: str,
    frvae_checkpoint: Optional[str],
    nfig_checkpoint: Optional[str],
) -> None:
    """Validate that required checkpoint paths are provided and exist.

    Args:
        mode: Execution mode string.
        frvae_checkpoint: Path to FR-VAE checkpoint (may be None).
        nfig_checkpoint: Path to NFIG Transformer checkpoint (may be None).

    Raises:
        ValueError: If a required checkpoint argument is not provided.
        FileNotFoundError: If a provided checkpoint file does not exist.
    """
    # Modes that require the FR-VAE checkpoint.
    frvae_required_modes: List[str] = ["train_nfig", "evaluate", "sample"]
    if mode in frvae_required_modes:
        if frvae_checkpoint is None:
            raise ValueError(
                f"--frvae_checkpoint is required for mode='{mode}'. "
                "Provide the path to a trained FR-VAE checkpoint (.pt file). "
                "Train the FR-VAE first with: python main.py --mode train_frvae"
            )
        if not os.path.exists(frvae_checkpoint):
            raise FileNotFoundError(
                f"FR-VAE checkpoint not found: '{frvae_checkpoint}'. "
                "Ensure the path is correct and the file exists."
            )

    # Modes that require the NFIG Transformer checkpoint.
    nfig_required_modes: List[str] = ["evaluate", "sample"]
    if mode in nfig_required_modes:
        if nfig_checkpoint is None:
            raise ValueError(
                f"--nfig_checkpoint is required for mode='{mode}'. "
                "Provide the path to a trained NFIG Transformer checkpoint (.pt file). "
                "Train the transformer first with: python main.py --mode train_nfig"
            )
        if not os.path.exists(nfig_checkpoint):
            raise FileNotFoundError(
                f"NFIG Transformer checkpoint not found: '{nfig_checkpoint}'. "
                "Ensure the path is correct and the file exists."
            )


# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------

def _get_device(config: Config, local_rank: int) -> torch.device:
    """Determine the target device for this process.

    In distributed mode, each process uses a specific GPU identified by
    local_rank. In single-GPU mode, uses the device from config.

    Args:
        config: Root Config dataclass.
        local_rank: Local GPU index for this process.

    Returns:
        torch.device for this process.
    """
    if config.training.device == "cuda" and torch.cuda.is_available():
        if dist.is_available() and dist.is_initialized():
            return torch.device(f"cuda:{local_rank}")
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _load_frvae(
    config: Config,
    checkpoint_path: str,
    device: torch.device,
) -> FRVAE:
    """Instantiate and load a trained FR-VAE from a checkpoint.

    The loaded model is set to eval mode with all parameters frozen.
    Used for modes: train_nfig, evaluate, sample.

    Args:
        config: Root Config dataclass.
        checkpoint_path: Path to the FR-VAE checkpoint file.
        device: Target device for the model.

    Returns:
        Loaded FRVAE instance in eval mode with frozen parameters.
    """
    logger.info("Loading FR-VAE from checkpoint: '%s'", checkpoint_path)

    # Instantiate FR-VAE architecture.
    frvae: FRVAE = FRVAE(config.frvae).to(device)

    # Load checkpoint weights.
    checkpoint_manager: CheckpointManager = CheckpointManager(
        config.training.checkpoint_dir
    )
    epoch: int
    metrics: Dict
    epoch, metrics = checkpoint_manager.load(
        path=checkpoint_path,
        model=frvae,
        optimizer=None,  # No optimizer needed for frozen inference model.
    )

    # Freeze all parameters and set to eval mode.
    frvae.freeze()  # calls requires_grad_(False) and eval()

    logger.info(
        "FR-VAE loaded successfully (checkpoint epoch=%d, metrics=%s). "
        "Model frozen for inference.",
        epoch,
        {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics.items()},
    )

    return frvae


def _load_nfig_transformer(
    config: Config,
    checkpoint_path: str,
    device: torch.device,
) -> NFIGTransformer:
    """Instantiate and load a trained NFIG Transformer from a checkpoint.

    The loaded model is set to eval mode with all parameters frozen.
    Used for modes: evaluate, sample.

    Args:
        config: Root Config dataclass.
        checkpoint_path: Path to the NFIG Transformer checkpoint file.
        device: Target device for the model.

    Returns:
        Loaded NFIGTransformer instance in eval mode with frozen parameters.
    """
    logger.info(
        "Loading NFIG Transformer from checkpoint: '%s'", checkpoint_path
    )

    # Instantiate NFIG Transformer architecture.
    transformer: NFIGTransformer = NFIGTransformer(config.nfig).to(device)

    # Load checkpoint weights.
    checkpoint_manager: CheckpointManager = CheckpointManager(
        config.training.checkpoint_dir
    )
    epoch: int
    metrics: Dict
    epoch, metrics = checkpoint_manager.load(
        path=checkpoint_path,
        model=transformer,
        optimizer=None,  # No optimizer needed for frozen inference model.
    )

    # Set to eval mode and freeze parameters.
    transformer.eval()
    transformer.requires_grad_(False)

    logger.info(
        "NFIG Transformer loaded successfully (checkpoint epoch=%d, metrics=%s). "
        "Model frozen for inference.",
        epoch,
        {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics.items()},
    )

    return transformer


# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------

def _run_train_frvae(
    config: Config,
    resume: Optional[str],
    rank: int,
) -> None:
    """Execute Phase 1: FR-VAE tokenizer training.

    Instantiates FRVAETrainer which internally creates:
      - FRVAE (encoder + decomposer + quantizer + decoder)
      - DINODiscriminator
      - NFIGLosses
      - Adam optimizers for generator and discriminator
      - LR schedulers with linear warmup
      - DataLoaders for ImageNet train/val
      - CheckpointManager
      - TensorBoard SummaryWriter (rank 0 only)

    Args:
        config: Root Config dataclass.
        resume: Optional path to a checkpoint to resume training from.
        rank: Process rank (0 for main process).
    """
    logger.info(
        "=" * 60 + "\n"
        "Phase 1: FR-VAE Training\n"
        "  Target rFID: %.2f (paper Table 2)\n"
        "  Epochs: %d\n"
        "  Batch size: %d\n"
        "  Learning rate: %.2e\n"
        "  Codebook size: %d\n"
        "  Total tokens: %d\n"
        "  Scale factors: %s\n"
        + "=" * 60,
        config.frvae.target_rfid,
        config.frvae.epochs,
        config.frvae.batch_size,
        config.frvae.learning_rate,
        config.frvae.codebook_size,
        config.frvae.total_tokens,
        config.frvae.scale_factors,
    )

    # Instantiate the FR-VAE trainer.
    # FRVAETrainer handles all model/optimizer/dataloader construction internally.
    trainer: FRVAETrainer = FRVAETrainer(config)

    # Resume from checkpoint if provided.
    if resume is not None:
        if not os.path.exists(resume):
            raise FileNotFoundError(
                f"Resume checkpoint not found: '{resume}'. "
                "Ensure the path is correct."
            )
        start_epoch: int = trainer.load_checkpoint(resume)
        logger.info(
            "Resuming FR-VAE training from epoch %d (checkpoint: '%s').",
            start_epoch,
            resume,
        )

    # Run the full training loop.
    # FRVAETrainer.train() handles:
    #   - Epoch loop with distributed sampler set_epoch()
    #   - Generator and discriminator training steps
    #   - Validation every eval_every_epochs epochs
    #   - Checkpoint saving every save_every_epochs epochs
    #   - TensorBoard logging
    trainer.train()

    if rank == 0:
        logger.info(
            "FR-VAE training complete. Best rFID: %.4f (target: %.2f).",
            trainer.best_rfid,
            config.frvae.target_rfid,
        )


def _run_train_nfig(
    config: Config,
    frvae_checkpoint: str,
    resume: Optional[str],
    rank: int,
) -> None:
    """Execute Phase 2: NFIG Transformer training.

    Instantiates NFIGTrainer which internally:
      - Loads and freezes the FR-VAE tokenizer from frvae_checkpoint
      - Creates NFIGTransformer (depth=16, hidden_dim=1024, ~310M params)
      - Sets up Adam optimizer (lr=8e-5, batch_size=768 from paper Section 4.1)
      - Sets up LR scheduler (linear warmup + cosine annealing)
      - Creates DataLoaders for ImageNet train/val
      - Sets up CheckpointManager and TensorBoard writer

    Args:
        config: Root Config dataclass.
        frvae_checkpoint: Path to the trained FR-VAE checkpoint.
        resume: Optional path to a transformer checkpoint to resume from.
        rank: Process rank (0 for main process).
    """
    logger.info(
        "=" * 60 + "\n"
        "Phase 2: NFIG Transformer Training\n"
        "  Target gFID: %.2f (paper Table 2)\n"
        "  Epochs: %d (paper Section 4.1)\n"
        "  Batch size: %d (paper Section 4.1)\n"
        "  Learning rate: %.2e (paper Section 4.1)\n"
        "  Transformer depth: %d\n"
        "  Hidden dim: %d\n"
        "  CFG dropout: %.1f%%\n"
        "  FR-VAE checkpoint: '%s'\n"
        + "=" * 60,
        config.nfig.target_gfid,
        config.nfig.epochs,
        config.nfig.batch_size,
        config.nfig.learning_rate,
        config.nfig.depth,
        config.nfig.hidden_dim,
        config.nfig.cfg_dropout_prob * 100.0,
        frvae_checkpoint,
    )

    # Instantiate the NFIG Transformer trainer.
    # NFIGTrainer handles all model/optimizer/dataloader construction internally,
    # including loading and freezing the FR-VAE tokenizer.
    trainer: NFIGTrainer = NFIGTrainer(
        config=config,
        frvae_checkpoint=frvae_checkpoint,
    )

    # Resume from checkpoint if provided.
    if resume is not None:
        if not os.path.exists(resume):
            raise FileNotFoundError(
                f"Resume checkpoint not found: '{resume}'. "
                "Ensure the path is correct."
            )
        start_epoch: int = trainer.load_checkpoint(resume)
        logger.info(
            "Resuming NFIG Transformer training from epoch %d "
            "(checkpoint: '%s').",
            start_epoch,
            resume,
        )

    # Run the full training loop.
    # NFIGTrainer.train() handles:
    #   - Epoch loop with distributed sampler set_epoch()
    #   - Tokenization via frozen FR-VAE
    #   - CFG dropout application
    #   - Cross-entropy loss computation over all 680 token positions
    #   - Gradient clipping (norm=1.0 from config)
    #   - Validation every eval_every_epochs epochs
    #   - Checkpoint saving every save_every_epochs epochs
    #   - TensorBoard logging
    trainer.train()

    if rank == 0:
        logger.info(
            "NFIG Transformer training complete. "
            "Best validation loss: %.4f (target gFID: %.2f).",
            trainer.best_val_loss,
            config.nfig.target_gfid,
        )


def _run_evaluate(
    config: Config,
    frvae_checkpoint: str,
    nfig_checkpoint: str,
    num_samples_override: Optional[int],
    output_dir_override: Optional[str],
    rank: int,
) -> Dict[str, float]:
    """Execute full evaluation: rFID, gFID, IS, Precision, Recall.

    Loads both trained models, instantiates the NFIGSampler and Evaluator,
    and runs the complete evaluation pipeline. Only rank 0 performs evaluation
    in distributed mode.

    Paper targets (Table 2):
        rFID:      0.85
        gFID:      2.81
        IS:        332.42
        Precision: 0.77
        Recall:    0.59

    Args:
        config: Root Config dataclass.
        frvae_checkpoint: Path to the trained FR-VAE checkpoint.
        nfig_checkpoint: Path to the trained NFIG Transformer checkpoint.
        num_samples_override: Optional override for config.evaluation.num_samples.
        output_dir_override: Optional override for config.evaluation.output_dir.
        rank: Process rank (0 for main process).

    Returns:
        Dictionary with all evaluation metrics. Empty dict for non-main processes.
    """
    # Only rank 0 performs evaluation to avoid redundant computation.
    if rank != 0:
        logger.info(
            "Rank %d: Skipping evaluation (only rank 0 evaluates).", rank
        )
        return {}

    # Apply optional overrides.
    if num_samples_override is not None:
        config.evaluation.num_samples = num_samples_override
        logger.info(
            "Overriding num_samples: %d", num_samples_override
        )

    if output_dir_override is not None:
        config.evaluation.output_dir = output_dir_override
        logger.info(
            "Overriding output_dir: '%s'", output_dir_override
        )

    logger.info(
        "=" * 60 + "\n"
        "Evaluation\n"
        "  num_samples: %d\n"
        "  cfg_scale: %.1f (paper Section 4.1)\n"
        "  top_k: %d (paper Section 4.1)\n"
        "  output_dir: '%s'\n"
        "  FR-VAE checkpoint: '%s'\n"
        "  NFIG checkpoint: '%s'\n"
        + "=" * 60,
        config.evaluation.num_samples,
        config.evaluation.cfg_scale,
        config.evaluation.top_k,
        config.evaluation.output_dir,
        frvae_checkpoint,
        nfig_checkpoint,
    )

    # Determine device for evaluation.
    device: torch.device = _get_device(config, local_rank=0)

    # Load FR-VAE (frozen, eval mode).
    frvae: FRVAE = _load_frvae(config, frvae_checkpoint, device)

    # Load NFIG Transformer (frozen, eval mode).
    transformer: NFIGTransformer = _load_nfig_transformer(
        config, nfig_checkpoint, device
    )

    # Instantiate NFIGSampler.
    # NFIGSampler sets both models to eval() and requires_grad_(False).
    sampler: NFIGSampler = NFIGSampler(
        transformer=transformer,
        frvae=frvae,
        config=config.nfig,
    )

    # Build validation DataLoader for rFID computation.
    val_dataset: ImageNetDataset = ImageNetDataset(
        root=config.data.val_dir,
        split="val",
        image_size=config.data.image_size,
    )
    val_loader = val_dataset.