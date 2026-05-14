## evaluate.py
"""
Evaluation module for Pyramidal Flow Matching.

Provides the `Evaluator` class that generates videos for standard benchmarks
(VBench, EvalCrafter) using the trained sampler, saves them to disk, and
invokes the official evaluation scripts to compute quantitative metrics.

All configuration is read from the project’s `Config` object.  The sampler
must be fully configured (model loaded, VAE present, text embedder ready).
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Video I/O helper
# ---------------------------------------------------------------------------

def _save_video(tensor: torch.Tensor, path: str, fps: int = 24) -> None:
    """
    Save a video tensor as an MP4 file.

    Args:
        tensor: float tensor of shape (T, C, H, W) with values in [0, 1].
        path: Output file path (should end with .mp4).
        fps: Frames per second.
    """
    # Convert to uint8
    video_uint8 = (tensor.clamp(0.0, 1.0) * 255.0).to(torch.uint8)

    # Required shape for torchvision.io.write_video: (T, H, W, C)
    if video_uint8.shape[1] == 3:  # (T, C, H, W)
        video_uint8 = video_uint8.permute(0, 2, 3, 1)

    try:
        from torchvision.io import write_video
        write_video(path, video_uint8, fps)
    except ImportError:
        # Fallback to OpenCV
        import cv2
        logger.warning("torchvision not available, using OpenCV to write video.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        T, H, W, C = video_uint8.shape
        out = cv2.VideoWriter(path, fourcc, fps, (W, H))
        if not out.isOpened():
            raise RuntimeError(f"Could not open video writer for {path}")
        for t in range(T):
            frame = video_uint8[t].cpu().numpy()
            # OpenCV expects BGR
            if C == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
        out.release()


def _load_prompts(prompt_file: str) -> List[str]:
    """
    Load a list of text prompts from a JSON file.

    The file is expected to contain either a JSON array of strings or an array
    of objects with a ``prompt`` key.

    Returns:
        Flat list of prompt strings.
    """
    if not os.path.isfile(prompt_file):
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    with open(prompt_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        prompts = []
        for item in data:
            if isinstance(item, str):
                prompts.append(item)
            elif isinstance(item, dict) and "prompt" in item:
                prompts.append(item["prompt"])
            else:
                raise ValueError(f"Unsupported prompt item format: {type(item)}")
        return prompts
    else:
        raise ValueError(f"Prompt file must contain a JSON array, got {type(data)}")


# ---------------------------------------------------------------------------
#  Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Evaluator for video generation benchmarks.

    Args:
        cfg: Global configuration object (must contain ``evaluation`` and
            ``inference`` sections).
        sampler: A fully configured ``Sampler`` instance with loaded model and VAE.
        benchmark: One of ``"VBench"`` or ``"EvalCrafter"``.
    """

    SUPPORTED_BENCHMARKS = {"VBench", "EvalCrafter"}

    def __init__(
        self,
        cfg: Any,   # Config object from config.py, but duck-typed
        sampler: Any,  # Sampler from inference.py
        benchmark: str,
    ) -> None:
        self.cfg = cfg
        self.sampler = sampler
        self.benchmark = benchmark

        if benchmark not in self.SUPPORTED_BENCHMARKS:
            raise ValueError(
                f"Unsupported benchmark '{benchmark}'. Choose from {self.SUPPORTED_BENCHMARKS}"
            )

        # Prompt file
        if benchmark == "VBench":
            prompt_file = cfg.evaluation.vbench_prompts
        else:
            prompt_file = cfg.evaluation.evalcrafter_prompts
        self.prompts = _load_prompts(prompt_file)

        # Output directory
        base_output = cfg.evaluation.output_dir
        self.output_dir = os.path.join(base_output, benchmark)

        # Video generation parameters (from config, can be overridden)
        self.fps: int = cfg.inference.video.default_fps
        self.resolution: tuple = tuple(cfg.inference.video.default_resolution)
        self.guidance_scale: float = cfg.inference.guidance_scale

    # ------------------------------------------------------------------
    #  Main evaluation entry point
    # ------------------------------------------------------------------
    def run_evaluation(self) -> Dict[str, float]:
        """
        Generate videos for all prompts, save them, run benchmark evaluation,
        and return the collected metrics.

        Returns:
            Dictionary mapping metric names to float scores (or numeric values).
        """
        if not self.prompts:
            logger.warning("No prompts found; returning empty metrics.")
            return {}

        logger.info(f"Starting evaluation for {self.benchmark} with {len(self.prompts)} prompts.")
        os.makedirs(self.output_dir, exist_ok=True)

        # 1. Generate videos
        videos = self.generate_videos_for_prompts(self.prompts)

        # 2. Save each video
        for idx, video_tensor in enumerate(videos):
            video_path = os.path.join(self.output_dir, f"prompt_{idx:04d}.mp4")
            _save_video(video_tensor, video_path, fps=self.fps)

        # 3. Run benchmark-specific evaluation
        if self.benchmark == "VBench":
            metrics = self._run_vbench_eval(self.output_dir)
        else:  # EvalCrafter
            metrics = self._run_evalcrafter_eval(self.output_dir)

        logger.info(f"Evaluation finished. {len(metrics)} metrics collected.")
        return metrics

    # ------------------------------------------------------------------
    #  Video generation (sequential)
    # ------------------------------------------------------------------
    def generate_videos_for_prompts(self, prompts: List[str]) -> List[torch.Tensor]:
        """
        Generate a video tensor for each prompt using the sampler.

        Args:
            prompts: List of text prompts.

        Returns:
            List of video tensors, each of shape (T, C, H, W) with values in [0, 1].
        """
        videos: List[torch.Tensor] = []
        for i, prompt in enumerate(prompts):
            logger.info(f"Generating video {i+1}/{len(prompts)}: {prompt[:50]}...")
            try:
                video = self.sampler.sample_text_to_video(
                    prompt=prompt,
                    duration_sec=5,   # paper uses 5‑second videos
                    fps=self.fps,
                    guidance_scale=self.guidance_scale,
                )
                # Ensure proper shape and range
                if video.ndim != 4 or video.shape[1] != 3:
                    raise ValueError(f"Unexpected video shape: {video.shape}")
                # Clamp to [0,1] just in case
                video = video.clamp(0.0, 1.0)
                videos.append(video)
            except Exception as e:
                logger.error(f"Failed to generate video for prompt '{prompt}': {e}")
                # Depending on the benchmark, we might want to skip or re-raise.
                # For reproducibility, stop early.
                raise RuntimeError(f"Video generation failed at prompt {i}: {prompt}") from e

        return videos

    # ------------------------------------------------------------------
    #  VBench evaluation runner
    # ------------------------------------------------------------------
    def _run_vbench_eval(self, video_dir: str) -> Dict[str, float]:
        """
        Invoke VBench evaluation script and parse its output JSON.

        Args:
            video_dir: Directory containing the generated MP4 files.

        Returns:
            Dictionary of all VBench metrics.
        """
        # Determine path to VBench evaluator.
        # We assume the user has cloned VBench and set the path in config, or it's
        # relative to the project root. We will look for an environment variable or
        # a well-known location. For simplicity, assume 'vbench/evaluate.py' inside
        # the project. The config can also contain 'vbench_root'.
        vbench_root = self.cfg.evaluation.get("vbench_root", "vbench")
        eval_script = os.path.join(vbench_root, "evaluate.py")
        if not os.path.isfile(eval_script):
            raise FileNotFoundError(
                f"VBench evaluation script not found at {eval_script}. "
                "Please clone https://github.com/Vchitect/VBench into the project "
                "and set evaluation.vbench_root in config.yaml."
            )

        results_file = os.path.join(video_dir, "vbench_results.json")
        cmd = [
            sys.executable,
            eval_script,
            "--videos_path", video_dir,
            "--result_path", results_file,
        ]

        logger.info(f"Running VBench: {' '.join(cmd)}")
        try:
            proc = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                cwd=vbench_root,
            )
            logger.debug(f"VBench stdout:\n{proc.stdout}")
            if proc.stderr:
                logger.warning(f"VBench stderr:\n{proc.stderr}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"VBench evaluation failed with exit code {e.returncode}.\n"
                f"stdout: {e.stdout}\nstderr: {e.stderr}"
            ) from e

        if not os.path.isfile(results_file):
            raise RuntimeError(f"VBench did not produce expected results file: {results_file}")

        with open(results_file, "r", encoding="utf-8") as f:
            raw_metrics = json.load(f)

        # VBench output format: usually a list of per-video metrics, or a summary dict.
        # The official evaluate.py returns a dictionary with keys like "total_score",
        # "quality_score", "semantic_score", etc. We'll flatten it.
        if isinstance(raw_metrics, dict):
            # If it contains nested dicts, flatten them with prefix.
            flat = {}
            for k, v in raw_metrics.items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        flat[f"{k}/{sub_k}"] = float(sub_v)
                elif isinstance(v, (int, float, list)):
                    if isinstance(v, list):
                        # average if list of numbers
                        if all(isinstance(x, (int, float)) for x in v):
                            flat[k] = float(np.mean(v))
                        else:
                            flat[k] = str(v)   # fallback
                    else:
                        flat[k] = float(v)
            return flat
        elif isinstance(raw_metrics, list):
            # If it's a list of per-video metrics, we need to compute overall averages.
            # This is rare for VBench; we'll assume it's a list of dicts.
            logger.warning("VBench output is a list; averaging per-video metrics.")
            # Expect each element to be a dict of metric: float
            agg: Dict[str, List[float]] = {}
            for vid_result in raw_metrics:
                if not isinstance(vid_result, dict):
                    continue
                for k, v in vid_result.items():
                    if isinstance(v, (int, float)):
                        agg.setdefault(k, []).append(v)
            return {k: float(np.mean(v)) for k, v in agg.items()}
        else:
            raise ValueError(f"Unsupported VBench output format: {type(raw_metrics)}")

    # ------------------------------------------------------------------
    #  EvalCrafter evaluation runner
    # ------------------------------------------------------------------
    def _run_evalcrafter_eval(self, video_dir: str) -> Dict[str, float]:
        """
        Invoke EvalCrafter evaluation script and parse its output JSON.

        Args:
            video_dir: Directory containing the generated MP4 files.

        Returns:
            Dictionary of all EvalCrafter metrics.
        """
        # Similar approach: locate the script.
        ec_root = self.cfg.evaluation.get("evalcrafter_root", "evalcrafter")
        eval_script = os.path.join(ec_root, "eval.py")
        if not os.path.isfile(eval_script):
            raise FileNotFoundError(
                f"EvalCrafter evaluation script not found at {eval_script}. "
                "Please clone https://github.com/EvalCrafter/EvalCrafter into the project "
                "and set evaluation.evalcrafter_root in config.yaml."
            )

        results_file = os.path.join(video_dir, "evalcrafter_results.json")
        cmd = [
            sys.executable,
            eval_script,
            "--video_dir", video_dir,
            "--output", results_file,
        ]

        logger.info(f"Running EvalCrafter: {' '.join(cmd)}")
        try:
            proc = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                cwd=ec_root,
            )
            logger.debug(f"EvalCrafter stdout:\n{proc.stdout}")
            if proc.stderr:
                logger.warning(f"EvalCrafter stderr:\n{proc.stderr}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"EvalCrafter evaluation failed with exit code {e.returncode}.\n"
                f"stdout: {e.stdout}\nstderr: {e.stderr}"
            ) from e

        if not os.path.isfile(results_file):
            raise RuntimeError(f"EvalCrafter did not produce expected results file: {results_file}")

        with open(results_file, "r", encoding="utf-8") as f:
            raw_metrics = json.load(f)

        # EvalCrafter output is typically a flat dictionary of metric: float.
        if isinstance(raw_metrics, dict):
            flat = {}
            for k, v in raw_metrics.items():
                if isinstance(v, (int, float)):
                    flat[k] = float(v)
                elif isinstance(v, list):
                    flat[k] = float(np.mean(v)) if all(isinstance(x, (int, float)) for x in v) else str(v)
                elif isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        flat[f"{k}/{sub_k}"] = float(sub_v)
            return flat
        else:
            raise ValueError(f"Unsupported EvalCrafter output format: {type(raw_metrics)}")


