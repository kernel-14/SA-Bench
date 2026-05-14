```python
## evaluation/vbench_eval.py
"""VBench evaluation wrapper for Pyramidal Flow Matching.

Wraps the external VBench evaluation toolkit to assess generated video quality
across 16 fine-grained dimensions as described in the paper (Section 4.1, 4.3):
"We utilize the VBench (Huang et al., 2024) for quantitative performance
evaluation. VBench is a comprehensive benchmark that includes 16 fine-grained
dimensions to systematically measure both motion quality and semantic alignment."

The paper evaluates at: 5-second, 121-frame videos at 768p resolution, 24fps.
Reference results (Table 1, Table 5):
    Total Score: 81.72, Quality Score: 84.74, Semantic Score: 69.62
    Motion Smoothness: 99.12, Dynamic Degree: 64.63

Config references (configs/default.yaml):
    eval.vbench.enabled: true
    eval.vbench.prompts_path: "data/eval/vbench_prompts.txt"
    eval.vbench.output_dir: "outputs/vbench"
    eval.vbench.num_frames: 121
    eval.vbench.resolution: [768, 768]
    eval.vbench.fps: 24
    eval.vbench.dimensions: [list of 16 dimension names]
    inference.cfg_scale: 7.5

Usage:
    from evaluation.vbench_eval import VBenchEvaluator

    evaluator = VBenchEvaluator(config)
    evaluator.generate_videos(sampler, evaluator.prompts, config.eval.vbench.output_dir)
    results = evaluator.run(config.eval.vbench.output_dir)
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from utils.distributed import is_main_process
from utils.logging import get_logger

## ---------------------------------------------------------------------------
## Module-level logger
## ---------------------------------------------------------------------------
logger = get_logger(__name__)

## ---------------------------------------------------------------------------
## Optional dependency availability flags
## ---------------------------------------------------------------------------
_IMAGEIO_AVAILABLE: bool = False
_TQDM_AVAILABLE: bool = False
_VBENCH_AVAILABLE: bool = False

try:
    import imageio  # type: ignore[import]
    import imageio_ffmpeg  # type: ignore[import]
    _IMAGEIO_AVAILABLE = True
except ImportError:
    logger.warning(
        "imageio or imageio-ffmpeg not available. Video saving will be disabled. "
        "Install with: pip install imageio==2.34.0 imageio-ffmpeg==0.4.9"
    )

try:
    from tqdm import tqdm as _tqdm  # type: ignore[import]
    _TQDM_AVAILABLE = True
except ImportError:
    def _tqdm(iterable: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        """No-op tqdm fallback when tqdm is not installed."""
        return iterable

try:
    import vbench  # type: ignore[import]
    _VBENCH_AVAILABLE = True
    logger.info("VBench toolkit available.")
except ImportError:
    logger.warning(
        "VBench toolkit not available. VBench evaluation will be limited. "
        "Install from: https://github.com/Vchitect/VBench"
    )

## ---------------------------------------------------------------------------
## Constants
## ---------------------------------------------------------------------------
_DEFAULT_PROMPTS_PATH: str = "data/eval/vbench_prompts.txt"
_DEFAULT_OUTPUT_DIR: str = "outputs/vbench"
_DEFAULT_NUM_FRAMES: int = 121
_DEFAULT_RESOLUTION: List[int] = [768, 768]
_DEFAULT_FPS: int = 24
_DEFAULT_CFG_SCALE: float = 7.5
_MAX_FILENAME_PROMPT_LEN: int = 50
_LOG_EVERY_N_PROMPTS: int = 10
_VIDEO_EXTENSION: str = ".mp4"

## VBench quality dimensions (from paper Table 5)
_QUALITY_DIMENSIONS: Tuple[str, ...] = (
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
)

## VBench semantic dimensions (from paper Table 5)
_SEMANTIC_DIMENSIONS: Tuple[str, ...] = (
    "object_class",
    "multiple_objects",
    "human_action",
    "color",
    "spatial_relationship",
    "scene",
    "appearance_style",
    "temporal_style",
    "overall_consistency",
)

## All 16 VBench dimensions (default list matching config)
_ALL_DIMENSIONS: List[str] = list(_QUALITY_DIMENSIONS) + list(_SEMANTIC_DIMENSIONS)

## VBench official dimension weights for total score aggregation
## These weights follow the VBench paper's standard aggregation scheme.
## Quality dimensions contribute to quality_score; semantic to semantic_score.
## total_score is a weighted combination of both.
_QUALITY_WEIGHT: float = 0.7143   # 5/7 of total weight
_SEMANTIC_WEIGHT: float = 0.2857  # 2/7 of total weight


class VBenchEvaluator:
    """Wrapper around the VBench evaluation toolkit for video quality assessment.

    Generates videos from text prompts using InferenceSampler and evaluates
    them using the VBench benchmark across 16 fine-grained dimensions.

    The paper evaluates at 5-second, 121-frame, 768p, 24fps settings.
    Reference scores from Table 1: Total=81.72, Quality=84.74, Semantic=69.62.

    Attributes:
        config: Full project configuration dictionary.
        prompts: List of text prompts loaded from the prompts file.
        prompts_path: Path to the prompts text file.
        output_dir: Directory where generated videos are saved.
        num_frames: Number of frames per generated video (121 from config).
        resolution: [H, W] output resolution ([768, 768] from config).
        fps: Video frame rate (24 from config).
        cfg_scale: Classifier-free guidance scale (7.5 from config).
        dimensions: List of 16 VBench dimension names to evaluate.
        vbench_available: Whether the VBench toolkit is installed.
        enabled: Whether VBench evaluation is enabled in config.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initializes VBenchEvaluator from the project config.

        Reads all evaluation settings from configs/default.yaml and loads
        the prompt list from the configured prompts file.

        Args:
            config: Project configuration dictionary from configs/default.yaml.
                Expected keys under config['eval']['vbench']:
                    - enabled (bool): Whether VBench eval is enabled. Default: True.
                    - prompts_path (str): Path to prompts text file.
                    - output_dir (str): Directory for generated videos.
                    - num_frames (int): Frames per video. Default: 121.
                    - resolution (list[int]): [H, W] resolution. Default: [768, 768].
                    - fps (int): Video frame rate. Default: 24.
                    - dimensions (list[str]): VBench dimensions to evaluate.
                Also reads:
                    - config['inference']['cfg_scale'] (float): Default: 7.5.
        """
        self.config: Dict[str, Any] = config

        # ----------------------------------------------------------------
        # Parse eval.vbench configuration
        # ----------------------------------------------------------------
        eval_cfg: Dict[str, Any] = config.get("eval", {})
        vbench_cfg: Dict[str, Any] = eval_cfg.get("vbench", {})
        inference_cfg: Dict[str, Any] = config.get("inference", {})

        self.enabled: bool = bool(vbench_cfg.get("enabled", True))

        self.prompts_path: str = str(
            vbench_cfg.get("prompts_path", _DEFAULT_PROMPTS_PATH)
        )
        self.output_dir: str = str(
            vbench_cfg.get("output_dir", _DEFAULT_OUTPUT_DIR)
        )
        self.num_frames: int = int(
            vbench_cfg.get("num_frames", _DEFAULT_NUM_FRAMES)
        )

        # Resolution: stored as [H, W]
        raw_resolution: Any = vbench_cfg.get("resolution", _DEFAULT_RESOLUTION)
        if isinstance(raw_resolution, (list, tuple)) and len(raw_resolution) >= 2:
            self.resolution: List[int] = [int(raw_resolution[0]), int(raw_resolution[1])]
        else:
            self.resolution = list(_DEFAULT_RESOLUTION)
            logger.warning(
                "Invalid resolution format in config: %s. "
                "Using default %s.",
                raw_resolution,
                _DEFAULT_RESOLUTION,
            )

        self.fps: int = int(vbench_cfg.get("fps", _DEFAULT_FPS))

        self.cfg_scale: float = float(
            inference_cfg.get("cfg_scale", _DEFAULT_CFG_SCALE)
        )

        # Parse dimensions list (16 VBench dimensions)
        raw_dimensions: Any = vbench_cfg.get("dimensions", _ALL_DIMENSIONS)
        if isinstance(raw_dimensions, (list, tuple)):
            self.dimensions: List[str] = [str(d) for d in raw_dimensions]
        else:
            self.dimensions = list(_ALL_DIMENSIONS)
            logger.warning(
                "Invalid dimensions format in config. Using all 16 default dimensions."
            )

        # VBench toolkit availability
        self.vbench_available: bool = _VBENCH_AVAILABLE

        # ----------------------------------------------------------------
        # Load prompts from file
        # ----------------------------------------------------------------
        self.prompts: List[str] = self._load_prompts(self.prompts_path)

        logger.info(
            "VBenchEvaluator initialized: enabled=%s, num_prompts=%d, "
            "num_frames=%d, resolution=%s, fps=%d, cfg_scale=%.1f, "
            "num_dimensions=%d, vbench_available=%s, output_dir='%s'",
            self.enabled,
            len(self.prompts),
            self.num_frames,
            self.resolution,
            self.fps,
            self.cfg_scale,
            len(self.dimensions),
            self.vbench_available,
            self.output_dir,
        )

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _load_prompts(self, prompts_path: str) -> List[str]:
        """Loads text prompts from a plain text file (one prompt per line).

        Args:
            prompts_path: Path to the prompts text file. Each non-empty line
                is treated as one prompt. Lines starting with '#' are treated
                as comments and skipped.

        Returns:
            List of stripped, non-empty prompt strings. Returns an empty list
            if the file does not exist (with a warning logged).
        """
        if not prompts_path:
            logger.warning(
                "prompts_path is empty. No prompts loaded for VBench evaluation."
            )
            return []

        if not os.path.isfile(prompts_path):
            logger.warning(
                "VBench prompts file not found: '%s'. "
                "No prompts loaded. "
                "Create the file with one prompt per line to enable evaluation. "
                "VBench prompts can be downloaded from: "
                "https://github.com/Vchitect/VBench",
                prompts_path,
            )
            return []

        prompts: List[str] = []
        skipped_lines: int = 0

        try:
            with open(prompts_path, "r", encoding="utf-8", errors="replace") as f:
                for line_num, line in enumerate(f, start=1):
                    stripped: str = line.strip()

                    # Skip empty lines and comment lines
                    if not stripped or stripped.startswith("#"):
                        skipped_lines += 1
                        continue

                    prompts.append(stripped)

        except Exception as exc:
            logger.error(
                "Failed to load prompts from '%s': %s. "
                "Returning empty prompt list.",
                prompts_path,
                exc,
            )
            return []

        logger.info(
            "Loaded %d prompts from '%s' (skipped %d empty/comment lines).",
            len(prompts),
            prompts_path,
            skipped_lines,
        )

        if len(prompts) == 0:
            logger.warning(
                "Prompts file '%s' contains no valid prompts. "
                "VBench evaluation will be skipped.",
                prompts_path,
            )

        return prompts

    def _sanitize_filename(self, prompt: str, max_len: int = _MAX_FILENAME_PROMPT_LEN) -> str:
        """Converts a prompt string to a safe filename component.

        Replaces spaces and special characters with underscores, truncates
        to max_len characters, and strips leading/trailing underscores.

        Args:
            prompt: Raw prompt string.
            max_len: Maximum length of the sanitized component. Defaults to 50.

        Returns:
            Sanitized string safe for use in file names.
        """
        # Replace any non-alphanumeric character (except hyphens) with underscore
        sanitized: str = re.sub(r"[^a-zA-Z0-9\-]", "_", prompt)
        # Collapse multiple consecutive underscores into one
        sanitized = re.sub(r"_+", "_", sanitized)
        # Strip leading/trailing underscores
        sanitized = sanitized.strip("_")
        # Truncate to max_len
        sanitized = sanitized[:max_len]
        # Final strip after truncation
        sanitized = sanitized.strip("_")
        # Fallback if empty after sanitization
        if not sanitized:
            sanitized = "prompt"
        return sanitized.lower()

    def _get_video_output_path(self, idx: int, prompt: str, output_dir: str) -> str:
        """Constructs the output file path for a generated video.

        Uses a deterministic naming scheme: {idx:04d}_{sanitized_prompt}.mp4
        This ensures consistent naming across runs and allows resuming
        interrupted evaluation by checking for existing files.

        Args:
            idx: Zero-based index of the prompt in the evaluation list.
            prompt: The text prompt string.
            output_dir: Directory where the video will be saved.

        Returns:
            Full absolute path to the output mp4 file.
        """
        sanitized: str = self._sanitize_filename(prompt)
        filename: str = f"{idx:04d}_{sanitized}{_VIDEO_EXTENSION}"
        return os.path.join(output_dir, filename)

    def _tensor_to_video_frames(self, video_tensor: Tensor) -> np.ndarray:
        """Converts a video tensor to a numpy array of uint8 frames.

        Handles both [C, T, H, W] (PyTorch convention) and [T, H, W, C]
        (imageio convention) input formats. Normalizes from [-1, 1] or
        [0, 1] to [0, 255] uint8.

        Args:
            video_tensor: Video tensor. Accepted shapes:
                - [C, T, H, W]: PyTorch convention (C=3 for RGB)
                - [T, H, W, C]: imageio convention
                - [1, C, T, H, W]: Batched PyTorch convention (batch dim removed)
                - [B, C, T, H, W]: Multi-batch (first sample taken)

        Returns:
            numpy array of shape [T, H, W, 3] with dtype uint8, values in [0, 255].
        """
        # Move to CPU and convert to float32 for processing
        tensor: Tensor = video_tensor.detach().cpu().float()

        # Handle batched inputs: take first sample
        if tensor.dim() == 5:
            # [B, C, T, H, W] or [B, T, H, W, C]
            tensor = tensor[0]  # Take first sample → [C, T, H, W] or [T, H, W, C]

        if tensor.dim() == 4:
            # Determine if [C, T, H, W] or [T, H, W, C]
            if tensor.shape[0] == 3 or tensor.shape[0] == 1:
                # Likely [C, T, H, W] — C is small (3 for RGB)
                # Permute to [T, H, W, C]
                tensor = tensor.permute(1, 2, 3, 0)  # [T, H, W, C]
            elif tensor.shape[-1] == 3 or tensor.shape[-1] == 1:
                # Already [T, H, W, C]
                pass
            else:
                # Ambiguous: assume [C, T, H, W] and permute
                logger.warning(
                    "Ambiguous video tensor shape %s. "
                    "Assuming [C, T, H, W] format.",
                    tuple(tensor.shape),
                )
                tensor = tensor.permute(1, 2, 3, 0)  # [T, H, W, C]
        else:
            raise ValueError(
                f"Unexpected video tensor shape: {tuple(tensor.shape)}. "
                f"Expected 4D [C, T, H, W] or [T, H, W, C], "
                f"or 5D [B, C, T, H, W]."
            )

        # tensor is now [T, H, W, C]

        # Normalize to [0, 1]:
        # If values are in [-1, 1] (VAE output convention), map to [0, 1]
        # If already in [0, 1], clamp only
        tensor_min: float = tensor.min().item()
        tensor_max: float = tensor.max().item()

        if tensor_min < -0.1:
            # Likely in [-1, 1] range: map to [0, 1]
            tensor = (tensor + 1.0) / 2.0

        # Clamp to [0, 1] to handle floating point artifacts
        tensor = tensor.clamp(0.0, 1.0)

        # Convert to uint8 [0, 255]
        frames_np: np.ndarray = (tensor.numpy() * 255.0).astype(np.uint8)
        # Shape: [T, H, W, C]

        # Ensure 3-channel RGB (drop alpha if present)
        if frames_np.shape[-1] == 4:
            frames_np = frames_np[..., :3]
        elif frames_np.shape[-1] == 1:
            # Grayscale: replicate to 3 channels
            frames_np = np.repeat(frames_np, 3, axis=-1)

        return frames_np

    def _save_video_as_mp4(
        self,
        frames: np.ndarray,
        output_path: str,
        fps: int,
    ) -> bool:
        """Saves a numpy array of frames as an MP4 video file.

        Args:
            frames: numpy array of shape [T, H, W, 3] with dtype uint8.
            output_path: Full path to the output .mp4 file.
            fps: Frames per second for the output video.

        Returns:
            True if the video was saved successfully, False otherwise.
        """
        if not _IMAGEIO_AVAILABLE:
            logger.error(
                "imageio not available. Cannot save video to '%s'. "
                "Install with: pip install imageio==2.34.0 imageio-ffmpeg==0.4.9",
                output_path,
            )
            return False

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            writer = imageio.get_writer(
                output_path,
                fps=fps,
                codec="libx264",
                quality=8,
                macro_block_size=1,  # Allow non-multiple-of-16 dimensions
                ffmpeg_log_level="quiet",
            )
            for frame in frames:
                writer.append_data(frame)
            writer.close()
            return True

        except Exception as exc:
            logger.error(
                "Failed to save video to '%s': %s",
                output_path,
                exc,
            )
            # Clean up partial file if it exists
            if os.path.isfile(output_path):
                try:
                    os.remove(output_path)
                except Exception:
                    pass
            return False

    def _count_mp4_files(self, directory: str) -> int:
        """Counts the number of .mp4 files in a directory.

        Args:
            directory: Path to the directory to scan.

        Returns:
            Number of .mp4 files found. Returns 0 if directory does not exist.
        """
        if not os.path.isdir(directory):
            return 0
        return sum(
            1 for f in os.listdir(directory)
            if f.lower().endswith(_VIDEO_EXTENSION)
        )

    def _compute_aggregate_scores(
        self,
        per_dimension_scores: Dict[str, float],
    ) -> Dict[str, float]:
        """Computes aggregate VBench scores from per-dimension scores.

        Implements the VBench standard aggregation:
        - quality_score: mean of quality-related dimensions
        - semantic_score: mean of semantic-related dimensions
        - total_score: weighted combination (quality_weight * quality + semantic_weight * semantic)

        The weights follow VBench's official aggregation scheme as used in
        the paper's Table 1 results.

        Args:
            per_dimension_scores: Dict mapping dimension name to score value.
                Scores are expected in [0, 100] range (percentage).

        Returns:
            Dict with keys 'quality_score', 'semantic_score', 'total_score'.
            Returns 0.0 for any score where no contributing dimensions are available.
        """
        # Compute quality score: mean of available quality dimensions
        quality_values: List[float] = []
        for dim in _QUALITY_DIMENSIONS:
            if dim in per_dimension_scores:
                quality_values.append(float(per_dimension_scores[dim]))

        quality_score: float = (
            float(np.mean(quality_values)) if quality_values else 0.0
        )

        # Compute semantic score: mean of available semantic dimensions
        semantic_values: List[float] = []
        for dim in _SEMANTIC_DIMENSIONS:
            if dim in per_dimension_scores:
                semantic_values.append(float(per_dimension_scores[dim]))

        semantic_score: float = (
            float(np.mean(semantic_values)) if semantic_values else 0.0
        )

        # Compute total score: weighted combination
        # Following VBench's standard: quality contributes more than semantic
        if quality_values and semantic_values:
            total_score: float = (
                _QUALITY_WEIGHT * quality_score
                + _SEMANTIC_WEIGHT * semantic_score
            )
        elif quality_values:
            total_score = quality_score
        elif semantic_values:
            total_score = semantic_score
        else:
            total_score = 0.0

        return {
            "quality_score": quality_score,
            "semantic_score": semantic_score,
            "total_score": total_score,
        }

    def _run_vbench_package(
        self,
        video_dir: str,
        results_path: str,
    ) -> Optional[Dict[str, float]]:
        """Attempts to run VBench evaluation using the installed Python package.

        Tries to use the VBench Python API directly. Returns None if the
        package API is not compatible or raises an exception.

        Args:
            video_dir: Directory containing generated .mp4 video files.
            results_path: Path where VBench results JSON will be saved.

        Returns:
            Dict mapping dimension names to scores, or None if this approach fails.
        """
        try:
            from vbench import VBench  # type: ignore[import]

            device: str = "cuda" if torch.cuda.is_available() else "cpu"

            # VBench requires a full_info_dir for its internal data
            # Try to find it from the vbench package installation
            vbench_package_dir: str = os.path.dirname(
                os.path.abspath(vbench.__file__)
            )
            full_info_dir: str = os.path.join(vbench_package_dir, "VBench_full_info.json")

            # Fallback: use a temp directory if full_info not found
            if not os.path.isfile(full_info_dir):
                full_info_dir = os.path.join(vbench_package_dir, "data", "VBench_full_info.json")

            if not os.path.isfile(full_info_dir):
                logger.warning(
                    "VBench full_info_dir not found at '%s'. "
                    "VBench package API may not work correctly.",
                    full_info_dir,
                )
                full_info_dir = vbench_package_dir

            output_path: str = os.path.dirname(results_path)
            os.makedirs(output_path, exist_ok=True)

            logger.info(
                "Running VBench evaluation via Python API: "
                "video_dir='%s', dimensions=%s",
                video_dir,
                self.dimensions,
            )

            bench = VBench(device, full_info_dir, output_path)

            # Run evaluation for each dimension
            dimension_scores: Dict[str, float] = {}

            for dimension in self.dimensions:
                try:
                    logger.info("Evaluating VBench dimension: %s", dimension)
                    bench.evaluate(
                        videos_path=video_dir,
                        name=f"pyramidal_flow_{dimension}",
                        dimension_list=[dimension],
                    )

                    # Try to read the result from VBench's output
                    # VBench typically writes results to JSON files
                    dim_result_path: str = os.path.join(
                        output_path,
                        f"pyramidal_flow_{dimension}_eval_results.json",
                    )
                    if os.path.isfile(dim_result_path):
                        with open(dim_result_path, "r", encoding="utf-8") as f:
                            dim_result: Any = json.load(f)
                        # Extract score from VBench result format
                        score: float = self._extract_score_from_vbench_result(
                            dim_result, dimension
                        )
                        dimension_scores[dimension] = score
                        logger.info(
                            "VBench dimension '%s': %.4f", dimension, score
                        )
                    else:
                        logger.warning(
                            "VBench result file not found for dimension '%s': '%s'",
                            dimension,
                            dim_result_path,
                        )

                except Exception as dim_exc:
                    logger.warning(
                        "VBench evaluation failed for dimension '%s': %s. "
                        "Skipping this dimension.",
                        dimension,
                        dim_exc,
                    )
                    continue

            return dimension_scores if dimension_scores else None

        except ImportError:
            logger.info(
                "VBench Python API not available. "
                "Falling back to subprocess approach."
            )
            return None
        except Exception as exc:
            logger.warning(
                "VBench Python API failed: %s. "
                "Falling back to subprocess approach.",
                exc,
            )
            return None

    def _extract_score_from_vbench_result(
        self,
        result: Any,
        dimension: str,
    ) -> float:
        """Extracts a scalar score from a VBench result dict.

        VBench result format varies by dimension. This method handles
        the common formats and returns a scalar score in [0, 100].

        Args:
            result: VBench result object (dict, list, or scalar).
            dimension: Dimension name for logging context.

        Returns:
            Scalar score in [0, 100]. Returns 0.0 if extraction fails.
        """
        try:
            if isinstance(result, (int, float)):
                score: float = float(result)
            elif isinstance(result, dict):
                # Try common VBench result keys
                for key in ("score", "value", "result", dimension, "average"):
                    if key in result:
                        val: Any = result[key]
                        if isinstance(val, (int, float)):
                            score = float(val)
                            break
                        elif isinstance(val, list) and len(val) > 0:
                            score = float(np.mean([float(v) for v in val if isinstance(v, (int, float))]))
                            break
                else:
                    # Try to find any numeric value
                    numeric_values: List[float] = [
                        float(v) for v in result.values()
                        if isinstance(v, (int, float))
                    ]
                    score = float(np.mean(numeric_values)) if numeric_values else 0.0
            elif isinstance(result, list):
                numeric_values = [float(v) for v in result if isinstance(v, (int, float))]
                score = float(np.mean(numeric_values)) if numeric_values else 0.0
            else:
                score = 0.0

            # VBench scores are typically in [0, 1] or [0, 100]
            # Normalize to [0, 100] if in [0, 1] range
            if 0.0 <= score <= 1.0:
                score = score * 100.0

            return score

        except Exception as exc:
            logger.warning(
                "Failed to extract score for dimension '%s': %s. "
                "Returning 0.0.",
                dimension,
                exc,
            )
            return 0.0

    def _run_vbench_subprocess(
        self,
        video_dir: str,
        results_path: str,
    ) -> Optional[Dict[str, float]]:
        """Attempts to run VBench evaluation via subprocess.

        Falls back to calling VBench's evaluation scripts as a subprocess
        when the Python API is not available or fails.

        Args:
            video_dir: Directory containing generated .mp4 video files.
            results_path: Path where results JSON will be saved.

        Returns:
            Dict mapping dimension names to scores, or None if this approach fails.
        """
        # Try to find VBench evaluation script
        vbench_script_candidates: List[str] = [
            "vbench_eval.py",
            "evaluate.py",
            os.path.join(os.path.dirname(sys.executable), "vbench_eval"),
        ]

        # Check if vbench is available as a command-line tool
        vbench_cmd: Optional[str] = shutil.which("vbench")
        if vbench_cmd is None:
            # Try python -m vbench
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "vbench", "--help"],
                    capture_output=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    vbench_cmd = f"{sys.executable} -m vbench"
            except Exception:
                pass

        if vbench_cmd is None:
            logger.warning(
                "VBench command-line tool not found. "
                "Cannot run VBench via subprocess. "
                "Install VBench from: https://github.com/Vchitect/VBench"
            )
            return None

        output_dir: