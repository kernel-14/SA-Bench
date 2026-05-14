## Code: evaluation/evalcrafter_eval.py

```python
## evaluation/evalcrafter_eval.py
"""EvalCrafter evaluation wrapper for Pyramidal Flow Matching.

Wraps the EvalCrafter evaluation toolkit to assess generated video quality
across ~17 objective metrics as described in the paper (Section 4.1, C.1):
"EvalCrafter is another large-scale evaluation benchmark including around
17 objective metrics for assessing video generation capabilities."

The paper evaluates at: 5-second, 121-frame videos at 768p resolution, 24fps.
Reference results (Table 2, Table 6):
    VQAA=86.09, VQAT=88.31, IS=18.49, CLIP-Temp=99.90
    Warping Error=0.0019, Face Consistency=98.89
    Action-Score=67.58, Motion AC-Score=46.0
    Flow-Score=1.79, CLIP-Score=20.73, BLIP-BLUE=23.29
    SD-Score=68.26, Detection-Score=69.55, Color-Score=47.74
    Count-Score=56.31, OCR-Score=68.55, Celebrity ID Score=44.72

Config references (configs/default.yaml):
    eval.evalcrafter.enabled: true
    eval.evalcrafter.prompts_path: "data/eval/evalcrafter_prompts.txt"
    eval.evalcrafter.output_dir: "outputs/evalcrafter"
    eval.evalcrafter.num_frames: 121
    eval.evalcrafter.resolution: [768, 768]
    eval.evalcrafter.fps: 24
    inference.cfg_scale: 7.5

Usage:
    from evaluation.evalcrafter_eval import EvalCrafterEvaluator

    evaluator = EvalCrafterEvaluator(config)
    evaluator.generate_videos(sampler, evaluator.prompts, config['eval']['evalcrafter']['output_dir'])
    results = evaluator.run(config['eval']['evalcrafter']['output_dir'])
"""

import json
import os
import re
import traceback
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
_EVALCRAFTER_AVAILABLE: bool = False

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
    from tqdm import tqdm as _tqdm_impl  # type: ignore[import]
    _TQDM_AVAILABLE = True
except ImportError:
    def _tqdm_impl(iterable: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        """No-op tqdm fallback when tqdm is not installed."""
        return iterable

try:
    import evalcrafter  # type: ignore[import]
    _EVALCRAFTER_AVAILABLE = True
    logger.info("EvalCrafter toolkit available.")
except ImportError:
    logger.warning(
        "EvalCrafter toolkit not available. EvalCrafter evaluation will be limited. "
        "Install from: https://github.com/EvalCrafter/EvalCrafter"
    )

## ---------------------------------------------------------------------------
## Constants
## ---------------------------------------------------------------------------
_DEFAULT_PROMPTS_PATH: str = "data/eval/evalcrafter_prompts.txt"
_DEFAULT_OUTPUT_DIR: str = "outputs/evalcrafter"
_DEFAULT_NUM_FRAMES: int = 121
_DEFAULT_RESOLUTION: List[int] = [768, 768]
_DEFAULT_FPS: int = 24
_DEFAULT_CFG_SCALE: float = 7.5
_MAX_FILENAME_PROMPT_LEN: int = 50
_LOG_EVERY_N_PROMPTS: int = 10
_VIDEO_EXTENSION: str = ".mp4"
_PROMPT_SIDECAR_EXTENSION: str = ".txt"
_RESULTS_FILENAME: str = "evalcrafter_results.json"

## ---------------------------------------------------------------------------
## EvalCrafter metric groups (from paper Table 2, Table 6)
## ---------------------------------------------------------------------------

## Visual quality metrics
_VISUAL_QUALITY_METRICS: Tuple[str, ...] = (
    "VQAA",
    "VQAT",
    "IS",
)

## Temporal quality metrics
_TEMPORAL_QUALITY_METRICS: Tuple[str, ...] = (
    "CLIP_Temp",
    "Warping_Error",
    "Face_Consistency",
)

## Action and motion metrics
_ACTION_MOTION_METRICS: Tuple[str, ...] = (
    "Action_Score",
    "Motion_AC_Score",
)

## Semantic alignment metrics
_SEMANTIC_METRICS: Tuple[str, ...] = (
    "Flow_Score",
    "CLIP_Score",
    "BLIP_BLUE",
    "SD_Score",
    "Detection_Score",
    "Color_Score",
    "Count_Score",
    "OCR_Score",
    "Celebrity_ID_Score",
)

## All 17 EvalCrafter metrics (canonical names matching paper Table 6)
_ALL_METRICS: Tuple[str, ...] = (
    _VISUAL_QUALITY_METRICS
    + _TEMPORAL_QUALITY_METRICS
    + _ACTION_MOTION_METRICS
    + _SEMANTIC_METRICS
)

## Mapping from EvalCrafter toolkit internal names to our canonical names
## (EvalCrafter's API may use different key names than the paper's notation)
_EVALCRAFTER_KEY_MAP: Dict[str, str] = {
    # Visual quality
    "vqaa": "VQAA",
    "video_quality_aesthetic": "VQAA",
    "aesthetic_quality": "VQAA",
    "vqat": "VQAT",
    "video_quality_technical": "VQAT",
    "technical_quality": "VQAT",
    "is": "IS",
    "inception_score": "IS",
    # Temporal quality
    "clip_temp": "CLIP_Temp",
    "clip_temporal": "CLIP_Temp",
    "temporal_consistency": "CLIP_Temp",
    "warping_error": "Warping_Error",
    "warp_error": "Warping_Error",
    "face_consistency": "Face_Consistency",
    "face_sim": "Face_Consistency",
    # Action and motion
    "action_score": "Action_Score",
    "action": "Action_Score",
    "motion_ac_score": "Motion_AC_Score",
    "motion_ac": "Motion_AC_Score",
    "motion": "Motion_AC_Score",
    # Semantic
    "flow_score": "Flow_Score",
    "flow": "Flow_Score",
    "clip_score": "CLIP_Score",
    "clip": "CLIP_Score",
    "blip_blue": "BLIP_BLUE",
    "blip_bleu": "BLIP_BLUE",
    "bleu": "BLIP_BLUE",
    "sd_score": "SD_Score",
    "sd": "SD_Score",
    "detection_score": "Detection_Score",
    "detection": "Detection_Score",
    "color_score": "Color_Score",
    "color": "Color_Score",
    "count_score": "Count_Score",
    "count": "Count_Score",
    "ocr_score": "OCR_Score",
    "ocr": "OCR_Score",
    "celebrity_id_score": "Celebrity_ID_Score",
    "celebrity": "Celebrity_ID_Score",
    "celeb": "Celebrity_ID_Score",
}

## Paper reference values for logging comparison (Table 6)
_PAPER_REFERENCE_VALUES: Dict[str, float] = {
    "VQAA": 86.09,
    "VQAT": 88.31,
    "IS": 18.49,
    "CLIP_Temp": 99.90,
    "Warping_Error": 0.0019,
    "Face_Consistency": 98.89,
    "Action_Score": 67.58,
    "Motion_AC_Score": 46.0,
    "Flow_Score": 1.79,
    "CLIP_Score": 20.73,
    "BLIP_BLUE": 23.29,
    "SD_Score": 68.26,
    "Detection_Score": 69.55,
    "Color_Score": 47.74,
    "Count_Score": 56.31,
    "OCR_Score": 68.55,
    "Celebrity_ID_Score": 44.72,
}


class EvalCrafterEvaluator:
    """Wrapper around the EvalCrafter evaluation toolkit for video quality assessment.

    Generates videos from text prompts using InferenceSampler and evaluates
    them using the EvalCrafter benchmark across ~17 objective metrics.

    The paper evaluates at 5-second, 121-frame, 768p, 24fps settings.
    Reference scores from Table 2 and Table 6 are documented in this module.

    Attributes:
        config: Full project configuration dictionary.
        prompts: List of text prompts loaded from the prompts file.
        prompts_path: Path to the prompts text file.
        output_dir: Directory where generated videos are saved.
        num_frames: Number of frames per generated video (121 from config).
        resolution: [H, W] output resolution ([768, 768] from config).
        fps: Video frame rate (24 from config).
        cfg_scale: Classifier-free guidance scale (7.5 from config).
        evalcrafter_available: Whether the EvalCrafter toolkit is installed.
        enabled: Whether EvalCrafter evaluation is enabled in config.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initializes EvalCrafterEvaluator from the project config.

        Reads all evaluation settings from configs/default.yaml and loads
        the prompt list from the configured prompts file.

        Args:
            config: Project configuration dictionary from configs/default.yaml.
                Expected keys under config['eval']['evalcrafter']:
                    - enabled (bool): Whether EvalCrafter eval is enabled. Default: True.
                    - prompts_path (str): Path to prompts text or JSON file.
                    - output_dir (str): Directory for generated videos.
                    - num_frames (int): Frames per video. Default: 121.
                    - resolution (list[int]): [H, W] resolution. Default: [768, 768].
                    - fps (int): Video frame rate. Default: 24.
                Also reads:
                    - config['inference']['cfg_scale'] (float): Default: 7.5.
        """
        self.config: Dict[str, Any] = config

        # ----------------------------------------------------------------
        # Parse eval.evalcrafter configuration
        # ----------------------------------------------------------------
        eval_cfg: Dict[str, Any] = config.get("eval", {})
        evalcrafter_cfg: Dict[str, Any] = eval_cfg.get("evalcrafter", {})
        inference_cfg: Dict[str, Any] = config.get("inference", {})

        self.enabled: bool = bool(evalcrafter_cfg.get("enabled", True))

        self.prompts_path: str = str(
            evalcrafter_cfg.get("prompts_path", _DEFAULT_PROMPTS_PATH)
        )
        self.output_dir: str = str(
            evalcrafter_cfg.get("output_dir", _DEFAULT_OUTPUT_DIR)
        )
        self.num_frames: int = int(
            evalcrafter_cfg.get("num_frames", _DEFAULT_NUM_FRAMES)
        )

        # Resolution: stored as [H, W]
        raw_resolution: Any = evalcrafter_cfg.get("resolution", _DEFAULT_RESOLUTION)
        if isinstance(raw_resolution, (list, tuple)) and len(raw_resolution) >= 2:
            self.resolution: List[int] = [
                int(raw_resolution[0]),
                int(raw_resolution[1]),
            ]
        else:
            self.resolution = list(_DEFAULT_RESOLUTION)
            logger.warning(
                "Invalid resolution format in config: %s. "
                "Using default %s.",
                raw_resolution,
                _DEFAULT_RESOLUTION,
            )

        self.fps: int = int(evalcrafter_cfg.get("fps", _DEFAULT_FPS))

        self.cfg_scale: float = float(
            inference_cfg.get("cfg_scale", _DEFAULT_CFG_SCALE)
        )

        # EvalCrafter toolkit availability
        self.evalcrafter_available: bool = _EVALCRAFTER_AVAILABLE

        # ----------------------------------------------------------------
        # Load prompts from file
        # ----------------------------------------------------------------
        self.prompts: List[str] = self._load_prompts()

        logger.info(
            "EvalCrafterEvaluator initialized: enabled=%s, num_prompts=%d, "
            "num_frames=%d, resolution=%s, fps=%d, cfg_scale=%.1f, "
            "evalcrafter_available=%s, output_dir='%s'",
            self.enabled,
            len(self.prompts),
            self.num_frames,
            self.resolution,
            self.fps,
            self.cfg_scale,
            self.evalcrafter_available,
            self.output_dir,
        )

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _load_prompts(self) -> List[str]:
        """Loads text prompts from the configured prompts file.

        Supports two file formats:
        - Plain text (.txt): one prompt per line; empty lines and lines
          starting with '#' are skipped.
        - JSON (.json): list of strings, or list of dicts with a "prompt"
          or "text" key (EvalCrafter's native format).

        Returns:
            List of stripped, non-empty prompt strings. Returns an empty list
            if the file does not exist (with a warning logged).

        Raises:
            FileNotFoundError: If prompts_path is set but the file does not exist.
        """
        if not self.prompts_path:
            logger.warning(
                "prompts_path is empty. No prompts loaded for EvalCrafter evaluation."
            )
            return []

        if not os.path.isfile(self.prompts_path):
            logger.warning(
                "EvalCrafter prompts file not found: '%s'. "
                "No prompts loaded. "
                "Create the file with one prompt per line to enable evaluation. "
                "EvalCrafter prompts can be downloaded from: "
                "https://github.com/EvalCrafter/EvalCrafter",
                self.prompts_path,
            )
            return []

        file_ext: str = Path(self.prompts_path).suffix.lower()

        prompts: List[str] = []

        try:
            if file_ext == ".json":
                prompts = self._load_prompts_from_json(self.prompts_path)
            else:
                # Default: plain text format (one prompt per line)
                prompts = self._load_prompts_from_txt(self.prompts_path)
        except Exception as exc:
            logger.error(
                "Failed to load prompts from '%s': %s. "
                "Returning empty prompt list.",
                self.prompts_path,
                exc,
            )
            return []

        logger.info(
            "Loaded %d prompts from '%s' (format: %s).",
            len(prompts),
            self.prompts_path,
            file_ext if file_ext in (".json", ".txt") else "txt",
        )

        if len(prompts) == 0:
            logger.warning(
                "Prompts file '%s' contains no valid prompts. "
                "EvalCrafter evaluation will be skipped.",
                self.prompts_path,
            )

        return prompts

    def _load_prompts_from_txt(self, path: str) -> List[str]:
        """Loads prompts from a plain text file (one prompt per line).

        Args:
            path: Path to the .txt prompts file.

        Returns:
            List of non-empty, stripped prompt strings.
        """
        prompts: List[str] = []
        skipped_lines: int = 0

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                stripped: str = line.strip()
                # Skip empty lines and comment lines
                if not stripped or stripped.startswith("#"):
                    skipped_lines += 1
                    continue
                prompts.append(stripped)

        if skipped_lines > 0:
            logger.debug(
                "Skipped %d empty/comment lines in '%s'.",
                skipped_lines,
                path,
            )

        return prompts

    def _load_prompts_from_json(self, path: str) -> List[str]:
        """Loads prompts from a JSON file.

        Supports two JSON formats:
        1. List of strings: ["prompt 1", "prompt 2", ...]
        2. List of dicts: [{"prompt": "...", "id": ...}, ...]
           Also accepts "text", "caption", "description" as alternative keys.

        Args:
            path: Path to the .json prompts file.

        Returns:
            List of non-empty prompt strings.

        Raises:
            ValueError: If the JSON structure is not recognized.
        """
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data: Any = json.load(f)

        prompts: List[str] = []

        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    stripped: str = item.strip()
                    if stripped:
                        prompts.append(stripped)
                elif isinstance(item, dict):
                    # Try common prompt key names
                    prompt_text: str = ""
                    for key in ("prompt", "text", "caption", "description", "query"):
                        if key in item and isinstance(item[key], str):
                            prompt_text = item[key].strip()
                            break
                    if prompt_text:
                        prompts.append(prompt_text)
                    else:
                        logger.debug(
                            "Skipping JSON dict item with no recognized prompt key: %s",
                            list(item.keys())[:5],
                        )
                else:
                    logger.debug(
                        "Skipping non-string, non-dict item in JSON prompts list: %s",
                        type(item).__name__,
                    )

        elif isinstance(data, dict):
            # Dict format: {"prompts": [...]} or {"data": [...]}
            for key in ("prompts", "data", "items", "queries"):
                if key in data and isinstance(data[key], list):
                    # Recursively process the nested list
                    nested_prompts: List[str] = []
                    for item in data[key]:
                        if isinstance(item, str):
                            stripped = item.strip()
                            if stripped:
                                nested_prompts.append(stripped)
                        elif isinstance(item, dict):
                            for pkey in ("prompt", "text", "caption", "description"):
                                if pkey in item and isinstance(item[pkey], str):
                                    text: str = item[pkey].strip()
                                    if text:
                                        nested_prompts.append(text)
                                    break
                    if nested_prompts:
                        prompts = nested_prompts
                        break
            else:
                raise ValueError(
                    f"Unrecognized JSON dict structure in '{path}'. "
                    f"Expected a list of prompts or a dict with a 'prompts' key. "
                    f"Found top-level keys: {list(data.keys())[:10]}."
                )
        else:
            raise ValueError(
                f"Unrecognized JSON structure in '{path}'. "
                f"Expected a list of strings or dicts. "
                f"Got type: {type(data).__name__}."
            )

        return prompts

    def _sanitize_filename(
        self,
        prompt: str,
        max_len: int = _MAX_FILENAME_PROMPT_LEN,
    ) -> str:
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

    def _get_video_output_path(
        self,
        idx: int,
        prompt: str,
        output_dir: str,
    ) -> str:
        """Constructs the output file path for a generated video.

        Uses a deterministic naming scheme: {idx:05d}_{sanitized_prompt}.mp4
        This ensures consistent naming across runs and allows resuming
        interrupted generation by checking for existing files.

        Args:
            idx: Zero-based index of the prompt in the evaluation list.
            prompt: The text prompt string.
            output_dir: Directory where the video will be saved.

        Returns:
            Full absolute path to the output mp4 file.
        """
        sanitized: str = self._sanitize_filename(prompt)
        filename: str = f"{idx:05d}_{sanitized}{_VIDEO_EXTENSION}"
        return os.path.join(output_dir, filename)

    def _get_prompt_sidecar_path(
        self,
        video_path: str,
    ) -> str:
        """Returns the path for the prompt sidecar text file.

        EvalCrafter may need prompt-video pairing via sidecar files.
        The sidecar has the same name as the video but with .txt extension.

        Args:
            video_path: Full path to the video file.

        Returns:
            Full path to the corresponding .txt sidecar file.
        """
        return str(Path(video_path).with_suffix(_PROMPT_SIDECAR_EXTENSION))

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
            # [B, C, T, H, W] — take first sample
            tensor = tensor[0]  # → [C, T, H, W]

        if tensor.dim() == 4:
            # Determine if [C, T, H, W] or [T, H, W, C]
            if tensor.shape[0] in (1, 3, 4):
                # Likely [C, T, H, W] — C is small (1, 3, or 4 channels)
                # Permute to [T, H, W, C]
                tensor = tensor.permute(1, 2, 3, 0)  # [T, H, W, C]
            elif tensor.shape[-1] in (1, 3, 4):
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
        elif tensor.dim() == 3:
            # Single frame [C, H, W] — treat as 1-frame video
            if tensor.shape[0] in (1, 3, 4):
                tensor = tensor.permute(1, 2, 0).unsqueeze(0)  # [1, H, W, C]
            else:
                tensor = tensor.unsqueeze(0)  # [1, H, W, C] assuming [H, W, C]
        else:
            raise ValueError(
                f"Unexpected video tensor shape: {tuple(tensor.shape)}. "
                f"Expected 3D [C, H, W], 4D [C, T, H, W] or [T, H, W, C], "
                f"or 5D [B, C, T, H, W]."
            )

        # tensor is now [T, H, W, C]

        # Normalize to [0, 1]:
        # If values are in [-1, 1] (VAE output convention), map to [0, 1]
        # If already in [0, 1], clamp only
        tensor_min: float = tensor.min().item()

        if tensor_min < -0.1:
            # Likely in [-1, 1] range: map to [0, 1]
            tensor = (tensor + 1.0) / 2.0

        # Clamp to [0, 1] to handle floating point artifacts
        tensor = tensor.clamp(0.0, 1.0)

        # Convert to uint8 [0, 255]
        frames_np: np.ndarray = (tensor.numpy() * 255.0).astype(np.uint8)
        # Shape: [T, H, W, C]

        # Ensure 3-channel RGB (drop alpha if present, replicate grayscale)
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
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

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

    def _save_prompt_sidecar(self, prompt: str, sidecar_path: str) -> None:
        """Saves a prompt string to a sidecar text file.

        EvalCrafter may need prompt-video pairing via sidecar files.

        Args:
            prompt: The text prompt string.
            sidecar_path: Full path to the .txt sidecar file.
        """
        try:
            with open(sidecar_path, "w", encoding="utf-8") as f:
                f.write(prompt)
        except Exception as exc:
            logger.warning(
                "Failed to save prompt sidecar to '%s': %s",
                sidecar_path,
                exc,
            )

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

    def _get_sorted_video_paths(self, video_dir: str) -> List[str]:
        """Returns sorted list of .mp4 file paths in a directory.

        Sorts by filename to ensure deterministic prompt-video pairing
        (index-based naming ensures correct ordering).

        Args:
            video_dir: Directory to scan for .mp4 files.

        Returns:
            Sorted list of full absolute paths to .mp4 files.
        """
        if not os.path.isdir(video_dir):
            return []

        video_files: List[str] = [
            os.path.join(video_dir, f)
            for f in sorted(os.listdir(video_dir))
            if f.lower().endswith(_VIDEO_EXTENSION)
        ]
        return video_files

    def _normalize_metric_key(self, raw_key: str) -> Optional[str]:
        """Normalizes a raw EvalCrafter metric key to our canonical name.

        Applies case-insensitive lookup in the key mapping table.

        Args:
            raw_key: Raw metric key from EvalCrafter's output.

        Returns:
            Canonical metric name (e.g., "VQAA"), or None if not recognized.
        