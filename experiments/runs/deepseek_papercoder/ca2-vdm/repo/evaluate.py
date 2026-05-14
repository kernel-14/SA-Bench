## evaluate.py
"""
Ca2‑VDM evaluation module.

Provides the ``Evaluator`` class that computes Fréchet Video Distance (FVD)
using the pretrained I3D model, in accordance with the evaluation protocols
described in the paper.  Two modes are supported:

1. **Whole‑video FVD** – compares directories of generated and real 16‑frame
   clips (used for Tables 1 & 2).
2. **Chunk‑wise FVD** – splits long autoregressively generated videos into
   non‑overlapping 16‑frame chunks and computes FVD per chunk against a set
   of real 16‑frame clips (used for Tables 3 & 4).

All hyperparameters (chunk size, I3D checkpoint path, device, etc.) are
read from the global ``Config`` object.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from config import Config
from data.preprocess import VideoProcessor
from utils.metrics import (
    InceptionI3d,
    calculate_fvd,
    extract_features,
    load_i3d_model,
    preprocess_video,
)


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Wrap the I3D feature extractor and provide high‑level FVD evaluation
    methods aligned with the paper's experiments.

    Parameters
    ----------
    config : Config
        Global configuration object; must contain valid ``evaluation`` and
        ``system`` sections.
    video_processor : VideoProcessor, optional
        Pre‑instantiated video processor; if ``None``, a new one is created
        from ``config``.
    """

    def __init__(
        self, config: Config, video_processor: Optional[VideoProcessor] = None
    ) -> None:
        self.config = config
        # ------------------------------------------------------------------
        # Device
        # ------------------------------------------------------------------
        self.device = torch.device(config.system.device)
        if torch.cuda.is_available() and self.device.type == "cuda":
            self.device = torch.device("cuda")

        # ------------------------------------------------------------------
        # Video processor (for loading frames)
        # ------------------------------------------------------------------
        if video_processor is None:
            self.video_processor = VideoProcessor(config)
        else:
            self.video_processor = video_processor

        # ------------------------------------------------------------------
        # Load I3D model
        # ------------------------------------------------------------------
        i3d_path = config.evaluation.fvd.i3d_model_path
        if i3d_path is None or not os.path.isfile(i3d_path or ""):
            raise FileNotFoundError(
                f"I3D checkpoint not found at '{i3d_path}'. "
                "Please download a Kinetics‑400 I3D model (e.g. from the StyleGAN‑V "
                "codebase) and set its path in config.yaml under evaluation.fvd.i3d_model_path."
            )

        self.i3d_model: InceptionI3d = load_i3d_model(
            i3d_path, device=self.device
        )  # type: ignore[assignment]
        self.i3d_model.eval()
        self.i3d_feature_layer = self.i3d_model.feature_layer_name

        # ------------------------------------------------------------------
        # Hyperparameters from config
        # ------------------------------------------------------------------
        self.chunk_size = config.evaluation.fvd.chunk_size
        if self.chunk_size < 16:
            raise ValueError(f"chunk_size must be at least 16, got {self.chunk_size}")

        self.num_generated_videos = config.evaluation.fvd.num_generated_videos

        # ------------------------------------------------------------------
        # Kinetics‑400 normalisation (from utils.metrics, but we store here
        # for clarity)
        # ------------------------------------------------------------------
        self.kinetics_mean = torch.tensor(
            [0.43216, 0.394666, 0.37645], device=self.device
        ).view(1, 3, 1, 1, 1)  # (1, 3, 1, 1, 1) to broadcast over (B,C,T,H,W)
        self.kinetics_std = torch.tensor(
            [0.22803, 0.22145, 0.216989], device=self.device
        ).view(1, 3, 1, 1, 1)

    # ------------------------------------------------------------------
    # Public evaluation methods
    # ------------------------------------------------------------------

    def compute_fvd(
        self,
        generated_videos_path: str,
        real_videos_path: str,
    ) -> float:
        """
        Compute FVD for in‑chunk generation quality (Tables 1 & 2).

        Videos in both directories are expected to be exactly 16 frames
        (or at least 16 frames; only the first 16 are used).  All files
        with common video extensions are enumerated.

        Parameters
        ----------
        generated_videos_path : str
            Path to a directory containing generated video files.
        real_videos_path : str
            Path to a directory containing ground‑truth video files.

        Returns
        -------
        float
            FVD score (lower is better).
        """
        # Gather video paths
        gen_paths = self._list_video_files(generated_videos_path)
        real_paths = self._list_video_files(real_videos_path)

        if not gen_paths:
            raise RuntimeError(f"No video files found in {generated_videos_path}")
        if not real_paths:
            raise RuntimeError(f"No video files found in {real_videos_path}")

        logger.info(
            "Computing whole‑video FVD: %d generated, %d real clips.",
            len(gen_paths), len(real_paths),
        )

        # Extract features for both sets
        gen_features = self._extract_features_from_paths(gen_paths, num_frames=16)
        real_features = self._extract_features_from_paths(real_paths, num_frames=16)

        if gen_features.shape[0] < 2 or real_features.shape[0] < 2:
            raise RuntimeError(
                "FVD requires at least 2 samples per set for covariance estimation."
            )

        fvd = calculate_fvd(real_features, gen_features)
        logger.info("FVD: %.4f", fvd)
        return fvd

    def compute_chunkwise_fvd(
        self,
        generated_frames: Union[torch.Tensor, List[torch.Tensor]],
        real_frames: Union[torch.Tensor, List[torch.Tensor]],
    ) -> Dict[str, float]:
        """
        Compute FVD for each non‑overlapping chunk of long generated videos
        (Tables 3 & 4).

        The real videos must be **16‑frame clips**.  Each long generated
        video is split into ``chunk_size`` frames (default 16); if the total
        length is not a multiple of ``chunk_size``, the last partial chunk
        is ignored.  Feature groups are formed by chunk index (i.e., all
        first chunks, all second chunks, …) and FVD is computed per group
        against the real features.

        Parameters
        ----------
        generated_frames : Tensor or list of Tensors
            Long generated videos in pixel space.  Each tensor has shape
            ``(T_i, C, H, W)`` with values in [‑1, 1] (as output by
            ``VideoProcessor.load_video``).  A list of such tensors can
            be passed for multiple videos.
        real_frames : Tensor or list of Tensors
            Real 16‑frame clips, each of shape ``(16, C, H, W)``.

        Returns
        -------
        dict
            Mapping ``"chunk_{i+1}"`` → FVD score for the i‑th chunk group.
        """
        # Normalise inputs
        gen_list = self._to_list(generated_frames)
        real_list = self._to_list(real_frames)

        # ------------------------------------------------------------------
        # 1. Extract features for all real clips
        # ------------------------------------------------------------------
        real_features = self._extract_features_from_tensors(
            real_list, num_frames=16
        )  # shape (N_real, D)

        if real_features.shape[0] < 2:
            raise RuntimeError("Need at least 2 real clips for FVD covariance.")

        # ------------------------------------------------------------------
        # 2. Split generated videos into chunks and extract features per chunk
        # ------------------------------------------------------------------
        # chunk_features[i] will be a list of numpy arrays (one per generated video's i-th chunk)
        max_chunks = 0
        chunk_features: Dict[int, List[np.ndarray]] = {}

        for video in tqdm(gen_list, desc="Chunk‑wise feature extraction"):
            T = video.shape[0]
            if T < self.chunk_size:
                logger.warning(
                    "Skipping video with %d frames (< chunk_size %d).", T, self.chunk_size
                )
                continue

            # Number of complete chunks
            n_chunks = T // self.chunk_size
            if n_chunks > max_chunks:
                max_chunks = n_chunks

            for ci in range(n_chunks):
                start = ci * self.chunk_size
                end = start + self.chunk_size
                chunk = video[start:end, ...]  # (chunk_size, C, H, W)

                # Preprocess for I3D
                preprocessed = self._preprocess_i3d_input(chunk)  # (1, 3, T, 224, 224)
                feats = extract_features(
                    self.i3d_model, preprocessed, self.i3d_feature_layer
                )
                chunk_features.setdefault(ci, []).append(feats)

        if max_chunks == 0:
            raise RuntimeError("No complete chunks found in the generated videos.")

        # ------------------------------------------------------------------
        # 3. Compute FVD for each chunk index
        # ------------------------------------------------------------------
        fvd_scores: Dict[str, float] = {}
        for ci in range(max_chunks):
            feats_list = chunk_features.get(ci, [])
            if not feats_list:
                fvd_scores[f"chunk_{ci+1}"] = float("nan")
                logger.warning("No features for chunk %d; set to NaN.", ci + 1)
                continue

            gen_chunk_features = np.concatenate(feats_list, axis=0)
            if gen_chunk_features.shape[0] < 2:
                logger.warning(
                    "Chunk %d has only %d samples; FVD may be unstable.",
                    ci + 1, gen_chunk_features.shape[0],
                )
            fvd = calculate_fvd(real_features, gen_chunk_features)
            fvd_scores[f"chunk_{ci+1}"] = fvd
            logger.info("Chunk %d FVD: %.4f", ci + 1, fvd)

        return fvd_scores

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _list_video_files(self, directory: str) -> List[str]:
        """Return all video files with common extensions in *directory*."""
        extensions = {".mp4", ".avi", ".mov", ".webm", ".mkv"}
        folder = Path(directory)
        if not folder.is_dir():
            raise NotADirectoryError(f"{directory} is not a directory.")
        files = [
            str(p)
            for p in folder.iterdir()
            if p.suffix.lower() in extensions
        ]
        return sorted(files)

    @staticmethod
    def _to_list(
        obj: Union[torch.Tensor, List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        """Convert a single tensor or a list/tensor of tensors to a list."""
        if isinstance(obj, torch.Tensor):
            # Single video
            return [obj]
        if isinstance(obj, (list, tuple)):
            return list(obj)
        raise TypeError(f"Expected Tensor or list of Tensors, got {type(obj)}")

    def _extract_features_from_paths(
        self, video_paths: List[str], num_frames: int
    ) -> np.ndarray:
        """
        Load first `num_frames` frames from each video, preprocess for I3D,
        extract features, and stack into a matrix.

        Parameters
        ----------
        video_paths : list of str
            Paths to video files.
        num_frames : int
            Exact number of frames to use (e.g. 16).

        Returns
        -------
        numpy.ndarray
            Feature matrix of shape ``(N_videos, feature_dim)``.
        """
        features_list: List[np.ndarray] = []
        for vp in tqdm(video_paths, desc="Extracting features"):
            try:
                # Load first `num_frames` frames (contiguous)
                frames = self.video_processor.load_video(
                    vp, num_frames=num_frames, start_frame=0, uniform=False
                )  # shape (T, 3, H, W), values in [-1, 1]
            except Exception as e:
                logger.warning("Skipping %s: %s", vp, e)
                continue

            if frames.shape[0] < num_frames:
                logger.warning(
                    "Video %s has only %d frames, expected >= %d; skipping.",
                    vp, frames.shape[0], num_frames,
                )
                continue

            # Preprocess for I3D
            preprocessed = self._preprocess_i3d_input(frames[:num_frames])
            feats = extract_features(
                self.i3d_model, preprocessed, self.i3d_feature_layer
            )
            features_list.append(feats)

        if not features_list:
            raise RuntimeError("No valid videos found for feature extraction.")

        return np.concatenate(features_list, axis=0)

    def _extract_features_from_tensors(
        self, video_tensors: List[torch.Tensor], num_frames: int
    ) -> np.ndarray:
        """
        Like `_extract_features_from_paths` but operates directly on
        pixel‑space tensors.

        Each tensor is expected to have shape ``(T, C, H, W)`` with values
        in [‑1, 1] (as produced by `VideoProcessor.load_video`).  Only the
        first `num_frames` are used.
        """
        features_list: List[np.ndarray] = []
        for video in tqdm(video_tensors, desc="Extracting features (tensors)"):
            if video.shape[0] < num_frames:
                logger.warning(
                    "Video tensor has %d frames, expected >= %d; skipping.",
                    video.shape[0], num_frames,
                )
                continue
            # Take first `num_frames`
            clip = video[:num_frames].to(self.device)
            preprocessed = self._preprocess_i3d_input(clip)
            feats = extract_features(
                self.i3d_model, preprocessed, self.i3d_feature_layer
            )
            features_list.append(feats)

        if not features_list:
            raise RuntimeError("No valid video tensors found for feature extraction.")

        return np.concatenate(features_list, axis=0)

    def _preprocess_i3d_input(self, clip: torch.Tensor) -> torch.Tensor:
        """
        Convert a raw video clip to the I3D input format.

        The input `clip` is a tensor of shape ``(T, C, H, W)`` with values
        in [‑1, 1].  The output is ``(1, 3, T, 224, 224)``, normalised with
        Kinetics‑400 mean/std.
        """
        # 1. Convert from [-1, 1] to [0, 1]
        clip = (clip + 1.0) / 2.0                      # (T, C, H, W)
        # 2. Resize to 224x224 and normalise – delegate to preprocess_video
        return preprocess_video(clip)                  # returns (1, 3, T, 224, 224)

