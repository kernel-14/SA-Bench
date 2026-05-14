"""Evaluation metrics for pyramidal flow matching.

Implements evaluation on:
- VBench (Huang et al., 2024): 16 fine-grained dimensions
- EvalCrafter (Liu et al., 2024): 17 objective metrics
- FID on MS-COCO (for image generation ablation)
- FVD on MSR-VTT (for video generation ablation)
"""

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import inception_v3


class FIDCalculator:
    """Frechet Inception Distance for image quality evaluation.

    Used in the ablation study (Fig. 7) to compare spatial pyramid
    vs standard flow matching on MS-COCO.
    """

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.inception = inception_v3(pretrained=True, transform_input=False)
        self.inception.fc = torch.nn.Identity()
        self.inception.eval().to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    @torch.no_grad()
    def get_features(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        """Extract Inception features from images."""
        features = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            tensors = torch.stack([self.transform(img) for img in batch]).to(self.device)
            feat = self.inception(tensors)
            features.append(feat.cpu().numpy())
        return np.concatenate(features, axis=0)

    def compute_fid(
        self,
        real_images: List[Image.Image],
        fake_images: List[Image.Image],
    ) -> float:
        """Compute FID between real and generated images."""
        real_features = self.get_features(real_images)
        fake_features = self.get_features(fake_images)

        mu_real = np.mean(real_features, axis=0)
        mu_fake = np.mean(fake_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)
        sigma_fake = np.cov(fake_features, rowvar=False)

        return self._frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)

    @staticmethod
    def _frechet_distance(
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray,
        eps: float = 1e-6,
    ) -> float:
        """Compute Frechet distance between two Gaussians."""
        diff = mu1 - mu2
        covmean = _sqrtm(sigma1 @ sigma2)

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
        return float(fid)


def _sqrtm(matrix: np.ndarray) -> np.ndarray:
    """Compute matrix square root via eigendecomposition."""
    from scipy.linalg import sqrtm
    return sqrtm(matrix)


class FVDCalculator:
    """Frechet Video Distance for video quality evaluation.

    Used in the ablation study (Fig. 12b) to compare temporal pyramid
    vs full-sequence diffusion on MSR-VTT.
    """

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # I3D model for video feature extraction
        self._i3d = None

    def _load_i3d(self):
        """Lazy load I3D model."""
        if self._i3d is None:
            try:
                from torchvision.models.video import r3d_18
                self._i3d = r3d_18(pretrained=True)
                self._i3d.fc = torch.nn.Identity()
                self._i3d.eval().to(self.device)
            except Exception:
                raise RuntimeError("Could not load I3D model for FVD computation")

    @torch.no_grad()
    def get_video_features(
        self,
        videos: List[List[Image.Image]],
        batch_size: int = 4,
    ) -> np.ndarray:
        """Extract video features using I3D."""
        self._load_i3d()
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        features = []
        for i in range(0, len(videos), batch_size):
            batch_videos = videos[i:i + batch_size]
            # Convert to tensor: (B, C, T, H, W)
            tensors = []
            for video in batch_videos:
                frames = torch.stack([transform(f) for f in video])  # (T, C, H, W)
                tensors.append(frames.permute(1, 0, 2, 3))  # (C, T, H, W)
            batch_tensor = torch.stack(tensors).to(self.device)  # (B, C, T, H, W)
            feat = self._i3d(batch_tensor)
            features.append(feat.cpu().numpy())

        return np.concatenate(features, axis=0)

    def compute_fvd(
        self,
        real_videos: List[List[Image.Image]],
        fake_videos: List[List[Image.Image]],
    ) -> float:
        """Compute FVD between real and generated videos."""
        real_features = self.get_video_features(real_videos)
        fake_features = self.get_video_features(fake_videos)

        mu_real = np.mean(real_features, axis=0)
        mu_fake = np.mean(fake_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)
        sigma_fake = np.cov(fake_features, rowvar=False)

        return FIDCalculator._frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)


class MotionSmoothness:
    """Compute motion smoothness metric.

    Measures temporal consistency by computing optical flow magnitude
    variance across frames. Lower variance = smoother motion.
    """

    def compute(self, frames: List[Image.Image]) -> float:
        """Compute motion smoothness score."""
        try:
            import cv2
        except ImportError:
            return 0.0

        if len(frames) < 2:
            return 1.0

        flows = []
        prev_gray = cv2.cvtColor(np.array(frames[0]), cv2.COLOR_RGB2GRAY)

        for frame in frames[1:]:
            curr_gray = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            flows.append(magnitude.mean())
            prev_gray = curr_gray

        if not flows:
            return 1.0

        # Smoothness = 1 - normalized variance of flow magnitudes
        flow_array = np.array(flows)
        variance = np.var(flow_array) / (np.mean(flow_array) + 1e-8)
        smoothness = 1.0 / (1.0 + variance)
        return float(smoothness)


class CLIPScore:
    """CLIP-based text-video alignment score."""

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):
        self.model_name = model_name
        self._model = None
        self._processor = None

    def _load_model(self):
        if self._model is None:
            from transformers import CLIPModel, CLIPProcessor
            self._model = CLIPModel.from_pretrained(self.model_name)
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._model.eval()

    @torch.no_grad()
    def compute(
        self,
        frames: List[Image.Image],
        text: str,
        device: torch.device = None,
    ) -> float:
        """Compute CLIP score between video frames and text."""
        self._load_model()
        device = device or torch.device("cpu")
        model = self._model.to(device)

        # Sample a subset of frames for efficiency
        sample_frames = frames[::max(1, len(frames) // 8)][:8]

        inputs = self._processor(
            text=[text],
            images=sample_frames,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)
        logits = outputs.logits_per_text  # (1, num_frames)
        score = logits.mean().item()
        return score


class VBenchEvaluator:
    """Wrapper for VBench evaluation.

    VBench measures 16 dimensions:
    - Subject Consistency, Background Consistency, Temporal Flickering
    - Motion Smoothness, Dynamic Degree, Aesthetic Quality, Imaging Quality
    - Object Class, Multiple Objects, Human Action, Color
    - Spatial Relationship, Scene, Appearance Style, Temporal Style
    - Overall Consistency
    """

    def __init__(self, vbench_path: Optional[str] = None):
        self.vbench_path = vbench_path

    def evaluate(
        self,
        video_dir: str,
        prompt_file: str,
        output_path: str,
    ) -> Dict[str, float]:
        """Run VBench evaluation on generated videos.

        Args:
            video_dir: directory containing generated videos
            prompt_file: path to VBench prompt list
            output_path: path to save evaluation results

        Returns:
            dict of metric name -> score
        """
        try:
            from vbench import VBench
            evaluator = VBench(device="cuda", video_path=video_dir, full_info_dir=prompt_file)
            results = evaluator.evaluate(
                videos_path=video_dir,
                name=output_path,
                prompt_list=prompt_file,
            )
            return results
        except ImportError:
            print("VBench not installed. Install from https://github.com/Vchitect/VBench")
            return {}


class EvalCrafterEvaluator:
    """Wrapper for EvalCrafter evaluation.

    EvalCrafter measures ~17 metrics including:
    - VQAA, VQAT (video quality)
    - IS (Inception Score)
    - CLIP-Temp (temporal CLIP consistency)
    - Warping Error, Face Consistency
    - Action Score, Motion AC-Score
    - Flow Score, CLIP Score, BLIP-BLUE, SD-Score
    - Detection Score, Color Score, Count Score, OCR Score, Celebrity ID Score
    """

    def evaluate(
        self,
        video_dir: str,
        prompt_file: str,
        output_path: str,
    ) -> Dict[str, float]:
        """Run EvalCrafter evaluation."""
        try:
            import evalcrafter
            results = evalcrafter.evaluate(
                video_dir=video_dir,
                prompt_file=prompt_file,
                output_path=output_path,
            )
            return results
        except ImportError:
            print("EvalCrafter not installed. See https://github.com/EvalCrafter/EvalCrafter")
            return {}


def compute_token_efficiency(
    num_frames: int,
    tokens_per_frame: int,
    num_stages: int = 3,
) -> Dict[str, float]:
    """Compute token efficiency metrics for pyramidal flow matching.

    From the paper: for a 10s, 241-frame video, pyramidal flow uses
    ≤15,360 tokens vs 119,040 tokens for full-sequence diffusion.

    Args:
        num_frames: total number of frames
        tokens_per_frame: tokens per frame at full resolution
        num_stages: number of pyramid stages

    Returns:
        dict with efficiency metrics
    """
    full_tokens = num_frames * tokens_per_frame

    # Pyramidal: most frames at lowest resolution
    # Stage k has factor 2^(K-1-k), so stage 0 has factor 2^(K-1)
    # Under uniform stage partitioning, each stage has num_frames/K frames
    frames_per_stage = num_frames // num_stages
    pyramid_tokens = 0
    for k in range(num_stages):
        factor = 2 ** (num_stages - 1 - k)
        tokens_k = tokens_per_frame // (factor * factor)
        pyramid_tokens += frames_per_stage * tokens_k

    # Computation: proportional to tokens^2 for attention
    full_compute = full_tokens ** 2
    pyramid_compute = pyramid_tokens ** 2

    return {
        "full_tokens": full_tokens,
        "pyramid_tokens": pyramid_tokens,
        "token_reduction": full_tokens / pyramid_tokens,
        "compute_reduction": full_compute / pyramid_compute,
    }
