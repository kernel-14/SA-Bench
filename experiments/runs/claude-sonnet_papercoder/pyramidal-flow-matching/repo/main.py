```python
## main.py
"""Entry point for Pyramidal Flow Matching: training, inference, and evaluation.

Implements the PyramidFlowApp class and main() function that orchestrate all
three operational modes described in the paper:
    - Training: Three-stage procedure (image → low-res video → high-res video)
    - Inference: Text-to-video, image-to-video, and long-video generation
    - Evaluation: VBench, EvalCrafter, FID (MS-COCO), FVD (MSR-VTT)

Usage:
    # Training Stage 1 (image pretraining)
    python main.py train --stage 1 --config configs/default.yaml

    # Training Stage 2 (low-resolution video)
    python main.py train --stage 2 --config configs/default.yaml

    # Training Stage 3 (high-resolution video)
    python main.py train --stage 3 --config configs/default.yaml

    # Text-to-video inference
    python main.py infer --prompt "A sunset over the ocean" \
        --output outputs/video.mp4 --checkpoint checkpoints/step_0050000

    # Long-video autoregressive generation
    python main.py infer --prompt "A person walking" \
        --output outputs/long.mp4 --checkpoint checkpoints/step_0050000 \
        --mode long

    # VBench evaluation
    python main.py eval --benchmark vbench \
        --checkpoint checkpoints/step_0050000
"""

import argparse
import json
import logging
import os
import random
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from omegaconf import DictConfig, OmegaConf

from data.dataset_loader import DatasetLoader
from evaluation.metrics import MetricEvaluator
from inference.sampler import InferenceSampler
from models.mmdit import MMDiT
from models.positional_encoding import PositionalEncoding
from models.pyramid_flow import PyramidFlowModel
from models.text_encoders import TextEncoders
from models.vae_3d import VAE3D
from training.trainer import Trainer
from utils.checkpointing import load_checkpoint, load_pretrained_sd3
from utils.distributed import (
    barrier,
    get_rank,
    get_world_size,
    init_distributed,
    is_main_process,
)
from utils.logging import (
    build_summary_writer,
    configure_logging,
    get_logger,
    log_metrics,
)

## ---------------------------------------------------------------------------
## Module-level logger (configured after config load in __init__)
## ---------------------------------------------------------------------------
logger = get_logger(__name__)


## ---------------------------------------------------------------------------
## Argument parser
## ---------------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    """Builds the top-level argument parser with train/infer/eval subcommands.

    Returns:
        Configured ArgumentParser with three subcommands:
            - train: Three-stage training procedure
            - infer: Video generation (t2v, i2v, long)
            - eval: Benchmark evaluation (vbench, evalcrafter, fid, fvd)
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="pyramidal_flow",
        description=(
            "Pyramidal Flow Matching for Efficient Video Generative Modeling. "
            "Supports training, inference, and evaluation modes."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Global argument: config file path
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )

    # Subcommands
    subparsers = parser.add_subparsers(
        dest="subcommand",
        title="subcommands",
        description="Choose an operational mode.",
    )

    # ----------------------------------------------------------------
    # Subcommand: train
    # ----------------------------------------------------------------
    train_parser = subparsers.add_parser(
        "train",
        help="Run the three-stage training procedure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_parser.add_argument(
        "--stage",
        type=int,
        required=True,
        choices=[1, 2, 3],
        help=(
            "Training stage: "
            "1=image pretraining (50k steps), "
            "2=low-res video (200k steps), "
            "3=high-res video (50k steps)."
        ),
    )
    train_parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )
    train_parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Path to a checkpoint directory to resume training from. "
            "Overrides config.training.resume_from_checkpoint."
        ),
    )

    # ----------------------------------------------------------------
    # Subcommand: infer
    # ----------------------------------------------------------------
    infer_parser = subparsers.add_parser(
        "infer",
        help="Generate videos using a trained model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    infer_parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for video generation.",
    )
    infer_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path for the generated video (e.g., outputs/video.mp4).",
    )
    infer_parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )
    infer_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint directory.",
    )
    infer_parser.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help=(
            "Number of frames to generate. "
            "Overrides config.inference.default_num_frames (121)."
        ),
    )
    infer_parser.add_argument(
        "--resolution",
        type=str,
        default=None,
        help=(
            "Output resolution as HxW (e.g., '768x768'). "
            "Overrides config.inference.default_resolution."
        ),
    )
    infer_parser.add_argument(
        "--mode",
        type=str,
        default="t2v",
        choices=["t2v", "i2v", "long"],
        help=(
            "Generation mode: "
            "t2v=text-to-video, "
            "i2v=image-to-video (requires --image), "
            "long=long autoregressive video."
        ),
    )
    infer_parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Input image path for image-to-video (i2v) mode.",
    )
    infer_parser.add_argument(
        "--cfg_scale",
        type=float,
        default=None,
        help=(
            "Classifier-free guidance scale. "
            "Overrides config.inference.cfg_scale (7.5)."
        ),
    )

    # ----------------------------------------------------------------
    # Subcommand: eval
    # ----------------------------------------------------------------
    eval_parser = subparsers.add_parser(
        "eval",
        help="Evaluate a trained model on standard benchmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    eval_parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["vbench", "evalcrafter", "fid", "fvd"],
        help=(
            "Benchmark to run: "
            "vbench=VBench 16-dimension evaluation, "
            "evalcrafter=EvalCrafter ~17 metrics, "
            "fid=FID on MS-COCO (ablation), "
            "fvd=FVD on MSR-VTT (ablation)."
        ),
    )
    eval_parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )
    eval_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint directory.",
    )
    eval_parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Output directory for generated videos and results. "
            "Overrides the benchmark-specific output_dir in config."
        ),
    )

    return parser


## ---------------------------------------------------------------------------
## PyramidFlowApp
## ---------------------------------------------------------------------------


class PyramidFlowApp:
    """Application class orchestrating training, inference, and evaluation.

    Builds all model components from the configuration file and dispatches
    to the appropriate operational mode based on parsed CLI arguments.

    Attributes:
        args: Parsed CLI arguments from argparse.
        config: OmegaConf DictConfig loaded from the YAML config file.
        rank: Global rank of the current process (0 for single-GPU).
        world_size: Total number of processes (1 for single-GPU).
        is_main: True if this is the main (rank 0) process.
        device: torch.device for the current process.
        dtype: torch.dtype for model weights and inference (bfloat16).
        pos_enc: PositionalEncoding module.
        text_encoders: Frozen T5-XXL + CLIP ViT-L/14 text encoders.
        vae: 3D causal VAE for pixel↔latent compression.
        transformer: MM-DiT backbone for velocity prediction.
        model: PyramidFlowModel combining all sub-modules.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """Initializes the application by loading config and building all components.

        Execution order:
            1. Load YAML config and apply CLI overrides
            2. Configure logging
            3. Set random seeds
            4. Initialize distributed training
            5. Create output directories (rank 0 only)
            6. Build model components in dependency order
            7. Move model to device and cast to bfloat16

        Args:
            args: Parsed CLI arguments. The subcommand determines which
                operational mode will be run via run_train/run_infer/run_eval.
        """
        self.args: argparse.Namespace = args

        # ----------------------------------------------------------------
        # Step 1: Load configuration
        # ----------------------------------------------------------------
        config_path: str = getattr(args, "config", "configs/default.yaml")
        self.config: DictConfig = self._load_config(config_path)

        # Apply CLI overrides to config before passing to any component
        self._apply_cli_overrides()

        # ----------------------------------------------------------------
        # Step 2: Configure logging
        # ----------------------------------------------------------------
        logging_cfg: Dict[str, Any] = dict(self.config.get("logging", {}))
        configure_logging(
            level=str(logging_cfg.get("log_level", "INFO")),
            use_color=True,
            use_wandb=bool(logging_cfg.get("use_wandb", False)),
            wandb_project=str(
                logging_cfg.get("wandb_project", "pyramidal-flow-matching")
            ),
        )

        # Re-get logger after configure_logging sets the level
        global logger
        logger = get_logger(__name__)

        logger.info(
            "PyramidFlowApp initializing: subcommand=%s, config=%s",
            getattr(args, "subcommand", "unknown"),
            config_path,
        )

        # ----------------------------------------------------------------
        # Step 3: Set random seeds for reproducibility
        # ----------------------------------------------------------------
        seed: int = int(self.config.get("seed", 42))
        self._set_seeds(seed)

        # ----------------------------------------------------------------
        # Step 4: Initialize distributed training
        # ----------------------------------------------------------------
        init_distributed()
        self.rank: int = get_rank()
        self.world_size: int = get_world_size()
        self.is_main: bool = is_main_process()

        logger.info(
            "Distributed: rank=%d, world_size=%d, is_main=%s",
            self.rank,
            self.world_size,
            self.is_main,
        )

        # ----------------------------------------------------------------
        # Step 5: Determine device and dtype
        # ----------------------------------------------------------------
        self.device: torch.device = self._get_device()
        dtype_str: str = str(self.config.get("model", {}).get("dtype", "bfloat16"))
        self.dtype: torch.dtype = (
            torch.bfloat16 if dtype_str == "bfloat16"
            else torch.float16 if dtype_str == "float16"
            else torch.float32
        )

        logger.info(
            "Device: %s, dtype: %s",
            self.device,
            dtype_str,
        )

        # ----------------------------------------------------------------
        # Step 6: Create output directories (rank 0 only)
        # ----------------------------------------------------------------
        if self.is_main:
            self._create_output_directories()

        # Synchronize all ranks after directory creation
        barrier()

        # ----------------------------------------------------------------
        # Step 7: Build model components in dependency order
        # ----------------------------------------------------------------
        logger.info("Building model components...")

        # 7a. Positional encoding (no dependencies)
        logger.info("Building PositionalEncoding...")
        self.pos_enc: PositionalEncoding = PositionalEncoding(
            dict(self.config)
        )

        # 7b. Text encoders (T5-XXL + CLIP, frozen)
        logger.info("Building TextEncoders (T5-XXL + CLIP)...")
        self.text_encoders: TextEncoders = TextEncoders(dict(self.config))

        # 7c. 3D causal VAE
        logger.info("Building VAE3D...")
        self.vae: VAE3D = VAE3D(dict(self.config))

        # Load pretrained VAE weights if specified
        vae_pretrained_path: Optional[str] = self.config.get("vae", {}).get(
            "pretrained_path", None
        )
        if vae_pretrained_path is not None:
            logger.info(
                "Loading pretrained VAE weights from '%s'.", vae_pretrained_path
            )
            try:
                load_checkpoint(
                    model=self.vae,
                    optimizer=None,
                    scheduler=None,
                    path=vae_pretrained_path,
                    strict=True,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load pretrained VAE from '%s': %s. "
                    "VAE will use random initialization.",
                    vae_pretrained_path,
                    exc,
                )

        # 7d. MM-DiT transformer backbone
        logger.info("Building MMDiT (24 layers, 2B parameters)...")
        self.transformer: MMDiT = MMDiT(dict(self.config))

        # Load pretrained SD3 Medium weights if specified
        sd3_path: Optional[str] = self.config.get("training", {}).get(
            "pretrained_sd3_path", None
        )
        if sd3_path is not None:
            logger.info(
                "Loading SD3 Medium pretrained weights from '%s'.", sd3_path
            )
            try:
                load_pretrained_sd3(model=self.transformer, sd3_path=sd3_path)
            except Exception as exc:
                logger.warning(
                    "Failed to load SD3 Medium weights from '%s': %s. "
                    "Transformer will use random initialization.",
                    sd3_path,
                    exc,
                )

        # 7e. PyramidFlowModel (combines all sub-modules)
        logger.info("Building PyramidFlowModel...")
        self.model: PyramidFlowModel = PyramidFlowModel(
            vae=self.vae,
            transformer=self.transformer,
            text_encoders=self.text_encoders,
            pos_enc=self.pos_enc,
            config=dict(self.config),
        )

        # ----------------------------------------------------------------
        # Step 8: Move model to device and cast to configured dtype
        # ----------------------------------------------------------------
        logger.info(
            "Moving model to device=%s, dtype=%s...", self.device, dtype_str
        )
        self.model = self.model.to(device=self.device, dtype=self.dtype)

        # Free temporary GPU memory after model construction
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("PyramidFlowApp initialization complete.")

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _load_config(self, config_path: str) -> DictConfig:
        """Loads the YAML configuration file using OmegaConf.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            OmegaConf DictConfig containing all configuration values.

        Raises:
            FileNotFoundError: If the config file does not exist.
            Exception: If the YAML file cannot be parsed.
        """
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"Configuration file not found: '{config_path}'. "
                f"Ensure the path is correct. "
                f"Default config is at 'configs/default.yaml'."
            )

        try:
            config: DictConfig = OmegaConf.load(config_path)
            logger.debug("Config loaded from '%s'.", config_path)
            return config
        except Exception as exc:
            raise RuntimeError(
                f"Failed to parse configuration file '{config_path}': {exc}"
            ) from exc

    def _apply_cli_overrides(self) -> None:
        """Applies CLI argument overrides to the loaded configuration.

        CLI arguments take precedence over config file values. Overrides
        are applied using OmegaConf.update() to maintain type safety.

        Handles:
            - --resume: overrides config.training.resume_from_checkpoint
            - --num_frames: overrides config.inference.default_num_frames
            - --resolution: parses "HxW" and overrides config.inference.default_resolution
            - --cfg_scale: overrides config.inference.cfg_scale
            - --output_dir: overrides benchmark-specific output_dir in config
        """
        args: argparse.Namespace = self.args

        # Override resume checkpoint path
        resume_path: Optional[str] = getattr(args, "resume", None)
        if resume_path is not None:
            OmegaConf.update(
                self.config,
                "training.resume_from_checkpoint",
                resume_path,
                merge=True,
            )
            logger.debug(
                "CLI override: training.resume_from_checkpoint = '%s'",
                resume_path,
            )

        # Override number of frames
        num_frames: Optional[int] = getattr(args, "num_frames", None)
        if num_frames is not None:
            OmegaConf.update(
                self.config,
                "inference.default_num_frames",
                num_frames,
                merge=True,
            )
            logger.debug(
                "CLI override: inference.default_num_frames = %d", num_frames
            )

        # Override resolution (parse "HxW" string)
        resolution_str: Optional[str] = getattr(args, "resolution", None)
        if resolution_str is not None:
            parsed_resolution: Optional[List[int]] = self._parse_resolution(
                resolution_str
            )
            if parsed_resolution is not None:
                OmegaConf.update(
                    self.config,
                    "inference.default_resolution",
                    parsed_resolution,
                    merge=True,
                )
                logger.debug(
                    "CLI override: inference.default_resolution = %s",
                    parsed_resolution,
                )
            else:
                logger.warning(
                    "Could not parse resolution string '%s'. "
                    "Expected format: 'HxW' (e.g., '768x768'). "
                    "Using config default.",
                    resolution_str,
                )

        # Override CFG scale
        cfg_scale: Optional[float] = getattr(args, "cfg_scale", None)
        if cfg_scale is not None:
            OmegaConf.update(
                self.config,
                "inference.cfg_scale",
                cfg_scale,
                merge=True,
            )
            logger.debug(
                "CLI override: inference.cfg_scale = %.2f", cfg_scale
            )

        # Override evaluation output directory
        output_dir: Optional[str] = getattr(args, "output_dir", None)
        benchmark: Optional[str] = getattr(args, "benchmark", None)
        if output_dir is not None and benchmark is not None:
            benchmark_key: str = f"eval.{benchmark}.output_dir"
            OmegaConf.update(
                self.config,
                benchmark_key,
                output_dir,
                merge=True,
            )
            logger.debug(
                "CLI override: %s = '%s'", benchmark_key, output_dir
            )

    def _parse_resolution(self, resolution_str: str) -> Optional[List[int]]:
        """Parses a resolution string in 'HxW' format.

        Args:
            resolution_str: Resolution string, e.g., '768x768' or '384x512'.

        Returns:
            List [H, W] of integers, or None if parsing fails.
        """
        try:
            parts: List[str] = resolution_str.lower().split("x")
            if len(parts) != 2:
                return None
            h: int = int(parts[0].strip())
            w: int = int(parts[1].strip())
            if h <= 0 or w <= 0:
                return None
            return [h, w]
        except (ValueError, AttributeError):
            return None

    def _set_seeds(self, seed: int) -> None:
        """Sets random seeds for reproducibility across all libraries.

        Args:
            seed: Integer seed value from config.seed (default: 42).
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Deterministic mode (disabled by default for performance)
        deterministic: bool = bool(self.config.get("deterministic", False))
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True)
            except Exception as exc:
                logger.warning(
                    "Could not enable fully deterministic algorithms: %s. "
                    "Some operations may still be non-deterministic.",
                    exc,
                )
        else:
            # Enable cuDNN benchmark for faster convolutions (non-deterministic)
            torch.backends.cudnn.benchmark = True

        logger.info(
            "Random seeds set: seed=%d, deterministic=%s", seed, deterministic
        )

    def _get_device(self) -> torch.device:
        """Returns the appropriate torch.device for the current process.

        Uses the LOCAL_RANK environment variable to select the correct
        CUDA device in multi-GPU training. Falls back to CPU if CUDA
        is not available.

        Returns:
            torch.device pointing to cuda:<local_rank> or cpu.
        """
        if torch.cuda.is_available():
            local_rank: int = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            return torch.device(f"cuda:{local_rank}")
        return torch.device("cpu")

    def _create_output_directories(self) -> None:
        """Creates all required output directories.

        Only called on the main process (rank 0) to avoid race conditions.
        Creates directories for checkpoints, logs, and evaluation outputs.
        """
        paths_cfg: Dict[str, Any] = dict(self.config.get("paths", {}))

        directories: List[str] = [
            str(paths_cfg.get("output_dir", "outputs")),
            str(paths_cfg.get("checkpoint_dir", "checkpoints")),
            str(paths_cfg.get("log_dir", "logs")),
            str(paths_cfg.get("cache_dir", ".cache")),
        ]

        # Add evaluation output directories if eval config exists
        eval_cfg: Dict[str, Any] = dict(self.config.get("eval", {}))
        for benchmark_name in ("vbench", "evalcrafter", "fid_coco", "fvd_msrvtt"):
            benchmark_cfg: Dict[str, Any] = dict(
                eval_cfg.get(benchmark_name, {})
            )
            output_dir: Optional[str] = benchmark_cfg.get("output_dir")
            if output_dir is not None:
                directories.append(str(output_dir))

        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
                logger.debug("Directory ensured: '%s'", directory)
            except Exception as exc:
                logger.warning(
                    "Failed to create directory '%s': %s", directory, exc
                )

    def _load_model_for_inference(self, checkpoint_path: str) -> None:
        """Loads model weights from a checkpoint for inference or evaluation.

        Sets the model to eval mode after loading. Does not restore optimizer
        or scheduler state (not needed for inference).

        Args:
            checkpoint_path: Path to the checkpoint directory.

        Raises:
            FileNotFoundError: If the checkpoint directory does not exist.
        """
        if not os.path.isdir(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint directory not found: '{checkpoint_path}'. "
                f"Ensure the path points to a valid checkpoint directory "
                f"(e.g., 'checkpoints/step_0050000/')."
            )

        logger.info(
            "Loading model weights from checkpoint: '%s'", checkpoint_path
        )

        try:
            step: int = load_checkpoint(
                model=self.model,
                optimizer=None,
                scheduler=None,
                path=checkpoint_path,
                strict=False,  # Allow partial loading for fine-tuned models
            )
            logger.info(
                "Model weights loaded from checkpoint at step %d.", step
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load checkpoint from '{checkpoint_path}': {exc}"
            ) from exc

        # Set model to evaluation mode
        self.model.eval()
        logger.info("Model set to eval mode.")

    def _get_inference_resolution(self) -> Tuple[int, int]:
        """Returns the inference resolution as a (H, W) tuple.

        Reads from config.inference.default_resolution, which may have been
        overridden by the --resolution CLI argument.

        Returns:
            Tuple (H, W) of integers.
        """
        inference_cfg: Dict[str, Any] = dict(self.config.get("inference", {}))
        raw_resolution: Any = inference_cfg.get("default_resolution", [768, 768])

        if isinstance(raw_resolution, (list, tuple)) and len(raw_resolution) >= 2:
            return (int(raw_resolution[0]), int(raw_resolution[1]))

        logger.warning(
            "Invalid default_resolution in config: %s. Using (768, 768).",
            raw_resolution,
        )
        return (768, 768)

    def _save_video_output(
        self,
        video_tensor: torch.Tensor,
        output_path: str,
        fps: int = 24,
    ) -> None:
        """Saves a video tensor to an MP4 file.

        Only executes on the main process to avoid duplicate writes.
        Handles both [C, T, H, W] and [B, C, T, H, W] tensor formats.

        Args:
            video_tensor: Video tensor with values in [-1, 1] or [0, 1].
                Accepted shapes: [C, T, H, W] or [B, C, T, H, W].
            output_path: Full path to the output .mp4 file.
            fps: Frames per second. From config.inference.default_fps (24).
        """
        if not self.is_main:
            return

        try:
            import imageio  # type: ignore[import]
        except ImportError:
            logger.error(
                "imageio not available. Cannot save video to '%s'. "
                "Install with: pip install imageio==2.34.0 imageio-ffmpeg==0.4.9",
                output_path,
            )
            return

        # Ensure output directory exists
        output_dir: str = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_dir, exist_ok=True)

        # Normalize tensor shape to [C, T, H, W]
        tensor: torch.Tensor = video_tensor.detach().cpu().float()

        if tensor.dim() == 5:
            # [B, C, T, H, W] — take first sample
            tensor = tensor[0]

        if tensor.dim() == 4:
            # [C, T, H, W] — expected format
            pass
        elif tensor.dim() == 3:
            # [C, H, W] — single frame, add temporal dim
            tensor = tensor.unsqueeze(1)
        else:
            logger.error(
                "Unexpected video tensor shape %s. Cannot save video.",
                tuple(tensor.shape),
            )
            return

        # tensor: [C, T, H, W]
        # Normalize from [-1, 1] to [0, 255] uint8
        tensor_min: float = tensor.min().item()
        if tensor_min < -0.1:
            # Values in [-1, 1]: map to [0, 1]
            tensor = (tensor + 1.0) / 2.0

        tensor = tensor.clamp(0.0, 1.0)

        # Convert to [T, H, W, C] uint8 numpy array
        import numpy as np
        frames_np: np.ndarray = (
            tensor.permute(1, 2, 3, 0).numpy() * 255.0
        ).astype(np.uint8)

        # Ensure 3-channel RGB
        if frames_np.shape[-1] == 4:
            frames_np = frames_np[..., :3]
        elif frames_np.shape[-1] == 1:
            frames_np = np.repeat(frames_np, 3, axis=-1)

        # Write MP4
        try:
            writer = imageio.get_writer(
                output_path,
                fps=fps,
                codec="libx264",
                quality=8,
                macro_block_size=1,
                ffmpeg_log_level="quiet",
            )
            for frame in frames_np:
                writer.append_data(frame)
            writer.close()

            logger.info(
                "Video saved: path='%s', frames=%d, fps=%d, "
                "resolution=%dx%d",
                output_path,
                frames_np.shape[0],
                fps,
                frames_np.shape[2],
                frames_np.shape[1],
            )
        except Exception as exc:
            logger.error(
                "Failed to save video to '%s': %s", output_path, exc
            )
            # Clean up partial file
            if os.path.isfile(output_path):
                try:
                    os.remove(output_path)
                except Exception:
                    pass