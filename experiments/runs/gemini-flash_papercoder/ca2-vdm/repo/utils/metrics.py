```python
import os
import logging
import random
import numpy as np
from PIL import Image
import cv2 # For loading real videos in _get_real_video_frames
import subprocess # For calling external VBench script
import shutil # For temporary directories

from typing import List, Tuple, Dict, Any, Optional, Callable

import torch
import torch.nn as nn
import torchvision.transforms as transforms
# For FVD calculation
from scipy import linalg
from tqdm import tqdm

from config import Config
from pipelines.inferrer import Inferrer
# from data.video_dataset import VideoDataset # Avoiding direct import for type hinting only


logger = logging.getLogger(__name__)

# --- Placeholder for I3D Model (Based on torchvision's R3D_18 as a proxy for feature extractor) ---
# For true reproducibility, this should be replaced with the exact I3D model and weights
# used in the StyleGAN-V codebase or the paper's experiments.
# A common choice for FVD is I3D pre-trained on Kinetics-400, converting from pytorchvideo.
# For simplicity and to avoid external complex dependencies for this task,
# we'll use a simplified R3D_18 from torchvision and adapt its input expectation.
# The paper states "a pretrained I3D model", and that "FVD scores ... from StyleGAN-V codebase".
# StyleGAN-V's codebase often uses a specific I3D provided in their dependencies or a custom one.
# For now, let's create a minimal mock.
# In a real scenario, you'd load something like:
# from pytorchvideo.models.resnet import create_resnet
# self.i3d_model = create_resnet(
#     input_channel=3,
#     model_depth=50, # Or 101, etc.
#     model_num_class=400, # Kinetics-400
#     norm=nn.BatchNorm3d,
#     activation=nn.ReLU,
# ).eval()
# self.i3d_model.load_state_dict(torch.load("path_to_kinetics_pretrained_weights"))
#
# Since we don't have pytorchvideo or specific I3D details, we'll mock its behavior.
# The forward pass will just return random features of expected shape.
# This is a critical point for full reproduction, as FVD depends on the exact feature extractor.

class MockI3DFeatureExtractor(nn.Module):
    """
    A mock I3D feature extractor to simulate behavior for FVD calculation.
    In a real scenario, this would be a loaded and pre-trained I3D model.
    It returns random features of a typical I3D feature dimension.
    """
    def __init__(self, feature_dim: int = 2048): # Typical I3D feature dimension
        super().__init__()
        self.feature_dim = feature_dim
        # I3D typically expects 224x224 frames, normalized.
        # self.preprocess is handled by self.i3d_transform in Evaluator for PIL Images.

    def forward(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """
        Simulates feature extraction.
        Args:
            video_tensor (torch.Tensor): Input video tensor (N, C, T, H, W).
                                         N is batch size, C channels, T frames.
        Returns:
            torch.Tensor: Mock features (N, feature_dim).
        """
        if video_tensor.ndim != 5:
            raise ValueError(f"Expected 5D input (N, C, T, H, W), got {video_tensor.ndim}D: {video_tensor.shape}")

        batch_size = video_tensor.shape[0]
        return torch.randn(batch_size, self.feature_dim, device=video_tensor.device)

# --- End of I3D Placeholder ---


class Evaluator:
    """
    The Evaluator class is responsible for computing evaluation metrics and profiling performance.
    It uses the Inferrer to generate videos and external tools (or their mocks) for metrics.
    """

    def __init__(self, config: Config, inferrer: Inferrer, device: torch.device):
        """
        Initializes the Evaluator instance.

        Args:
            config (Config): The global configuration object.
            inferrer (Inferrer): An instance of the Inferrer class for video generation.
            device (torch.device): The computational device (e.g., 'cuda', 'cpu').
        """
        self.config = config
        self.inferrer = inferrer
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)

        # FVD Initialization
        # IMPORTANT: Replace MockI3DFeatureExtractor with an actual I3D model
        # pre-trained on Kinetics-400 for accurate FVD scores.
        self.i3d_model = MockI3DFeatureExtractor(feature_dim=2048) # Default I3D feature dimension
        self.i3d_model.to(self.device)
        self.i3d_model.eval()
        self.logger.warning("Using MockI3DFeatureExtractor. For true FVD, replace this with a Kinetics-400 pretrained I3D model.")

        # Image transformations for I3D input
        self.i3d_transform = transforms.Compose([
            transforms.ToTensor(), # Converts to [0, 1] range
            transforms.Resize((224, 224), antialias=True), # Standard I3D input resolution
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet/Kinetics normalization
        ])

        # VBench Setup - Placeholder
        self.vbench_output_dir = os.path.join(self.config.save_path, "vbench_results")
        os.makedirs(self.vbench_output_dir, exist_ok=True)
        self.logger.info(f"VBench generated videos will be saved to: {self.vbench_output_dir}")
        self.logger.warning("VBench metrics calculation requires an external VBench setup or API. This implementation is a placeholder.")


    def _compute_fvd(self, features_1: torch.Tensor, features_2: torch.Tensor) -> float:
        """
        Calculates the Frechet Video Distance (FVD) given two sets of I3D features.
        Adapted from FID implementation, using means and covariance matrices.

        Args:
            features_1 (torch.Tensor): Features from the first set of videos (e.g., generated).
                                       Shape: (num_videos, feature_dim).
            features_2 (torch.Tensor): Features from the second set of videos (e.g., real).
                                       Shape: (num_videos, feature_dim).

        Returns:
            float: The computed FVD score.
        """
        # Convert to numpy arrays for scipy
        mu1 = features_1.cpu().numpy().mean(axis=0)
        sigma1 = np.cov(features_1.cpu().numpy(), rowvar=False)
        mu2 = features_2.cpu().numpy().mean(axis=0)
        sigma2 = np.cov(features_2.cpu().numpy(), rowvar=False)

        # Calculate squared difference of means
        mu_diff = mu1 - mu2
        squared_mu_diff = np.dot(mu_diff, mu_diff)

        # Calculate sqrtm of product of covariance matrices
        covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
        if not np.isfinite(covmean).all():
            self.logger.error("Non-finite square root of product of covariance matrices encountered in FVD. "
                              "This can happen with near-singular covariance matrices. "
                              "Adding a small epsilon to diagonal of sigma1@sigma2.")
            # Fallback if sqrtm fails due to numerical instability
            offset = np.eye(sigma1.shape[0]) * 1e-6
            covmean = linalg.sqrtm(sigma1 @ sigma2 + offset, disp=False)[0]
        
        if np.iscomplexobj(covmean):
            self.logger.warning("Complex numbers encountered in FVD covmean. Taking real part.")
            covmean = covmean.real

        # Calculate trace
        trace_term = np.trace(sigma1 + sigma2 - 2 * covmean)

        fvd = squared_mu_diff + trace_term
        return float(fvd)

    def _extract_i3d_features(self, videos: List[Image.Image]) -> torch.Tensor:
        """
        Extracts I3D features from a list of video frames (PIL Images).

        Args:
            videos (List[Image.Image]): A list of PIL Image objects representing video frames
                                        for a single video.

        Returns:
            torch.Tensor: Extracted features for the video, shape (1, feature_dim).
        """
        if not videos:
            self.logger.warning("No video frames provided for I3D feature extraction. Returning dummy features.")
            return torch.zeros(1, self.i3d_model.feature_dim, device=self.device)

        # Convert list of PIL Images to (T, C, H, W) tensor, then to (C, T, H, W) for I3D
        # PIL Images are typically 0-255. self.i3d_transform will handle normalization.
        
        processed_frames: List[torch.Tensor] = []
        for frame in videos:
            # Convert PIL Image to tensor before applying transform
            processed_frames.append(self.i3d_transform(frame))
        
        video_tensor_NCHW = torch.stack(processed_frames) # (T, C, H_i3d, W_i3d)
        video_tensor_CTHW = video_tensor_NCHW.permute(1, 0, 2, 3) # (C, T, H_i3d, W_i3d)

        # Handle minimum frame count for I3D (e.g., 16 frames)
        min_frames = self.config.fvd_chunk_size # This is typically 16
        current_frames = video_tensor_CTHW.shape[1]
        
        if current_frames < min_frames:
            # Pad by repeating the last frame
            padding_needed = min_frames - current_frames
            last_frame = video_tensor_CTHW[:, -1:, :, :] # (C, 1, H, W)
            video_tensor_CTHW = torch.cat([video_