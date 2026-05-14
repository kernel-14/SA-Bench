## utils/metrics.py
"""
Fréchet Video Distance (FVD) metrics for Ca2‑VDM reproduction.

Provides utilities to load a Kinetics‑400 pretrained I3D model, preprocess
video clips, extract I3D features, compute FVD between sets of features,
and evaluate autoregressive generation quality via chunk‑wise FVD.

The I3D implementation is a standard Inception‑3D architecture compatible
with the StyleGAN‑V codebase and commonly used in video generation research.
All functions are stateless and reusable outside the Ca2‑VDM context.
"""

from __future__ import annotations

import math
import os
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ---------------------------------------------------------------------------
# I3D model definition (self‑contained)
# ---------------------------------------------------------------------------

# Basic building blocks taken from the original I3D release
class MaxPool3dSamePadding(nn.MaxPool3d):
    """3D max pooling with same padding (output size = ceil(input_size / stride))."""

    def compute_pad(self, dim: int, s: int, k: int) -> Tuple[int, int]:
        # Ceil division: (dim + stride - 1) // stride
        o = math.ceil(dim / s)
        pad = max(0, (o - 1) * s + k - dim)
        return pad // 2, pad - pad // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        D, H, W = x.shape[2], x.shape[3], x.shape[4]
        pad_d = self.compute_pad(D, self.stride[0], self.kernel_size[0])
        pad_h = self.compute_pad(H, self.stride[1], self.kernel_size[1])
        pad_w = self.compute_pad(W, self.stride[2], self.kernel_size[2])
        x = F.pad(x, pad_w + pad_h + pad_d)
        return super().forward(x)


class AvgPool3dSamePadding(nn.AvgPool3d):
    """3D average pooling with same padding."""

    def compute_pad(self, dim: int, s: int, k: int) -> Tuple[int, int]:
        o = math.ceil(dim / s)
        pad = max(0, (o - 1) * s + k - dim)
        return pad // 2, pad - pad // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        D, H, W = x.shape[2], x.shape[3], x.shape[4]
        pad_d = self.compute_pad(D, self.stride[0], self.kernel_size[0])
        pad_h = self.compute_pad(H, self.stride[1], self.kernel_size[1])
        pad_w = self.compute_pad(W, self.stride[2], self.kernel_size[2])
        x = F.pad(x, pad_w + pad_h + pad_d)
        return super().forward(x)


class Unit3D(nn.Module):
    """Basic 3D conv‑BN‑ReLU unit."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 1,
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, str] = 0,
        use_bias: bool = False,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=use_bias,
        )
        self.bn = nn.BatchNorm3d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class InceptionModule(nn.Module):
    """An Inception module with four branches (1x1x1, 3x3x3, 3x3x3 double, pooling)."""

    def __init__(self, in_channels: int, out_channels: List[int]) -> None:
        super().__init__()
        if len(out_channels) != 6:
            raise ValueError("out_channels must provide 6 values (b0,b1,b2,b3,b4,b5)")
        b0, b1, b2, b3, b4, b5 = out_channels

        self.branch0 = nn.Sequential(
            Unit3D(in_channels, b0, kernel_size=1, use_bn=True)
        )
        self.branch1 = nn.Sequential(
            Unit3D(in_channels, b1, kernel_size=1, use_bn=True),
            Unit3D(b1, b2, kernel_size=3, padding=1, use_bn=True),
        )
        self.branch2 = nn.Sequential(
            Unit3D(in_channels, b3, kernel_size=1, use_bn=True),
            Unit3D(b3, b4, kernel_size=3, padding=1, use_bn=True),
        )
        self.branch3 = nn.Sequential(
            MaxPool3dSamePadding(kernel_size=3, stride=1),
            Unit3D(in_channels, b5, kernel_size=1, use_bn=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b0 = self.branch0(x)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        return torch.cat([b0, b1, b2, b3], dim=1)


class InceptionI3d(nn.Module):
    """
    I3D model adapted for feature extraction (FVD).

    Input: (B, 3, T, H, W) with H=W=224.
    Returns features from the final average pooling layer (1024‑D).

    Weights can be loaded from a Kinetics‑400 checkpoint.
    """

    def __init__(self, num_classes: int = 400) -> None:
        super().__init__()
        # Stem
        self.conv3d_1a_7x7 = Unit3D(3, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3))
        self.maxpool3d_2a_3x3 = MaxPool3dSamePadding(kernel_size=(1, 3, 3), stride=(1, 2, 2))

        self.conv3d_2b_1x1 = Unit3D(64, 64, kernel_size=1)
        self.conv3d_2c_3x3 = Unit3D(64, 192, kernel_size=3, padding=1)

        self.maxpool3d_3a_3x3 = MaxPool3dSamePadding(kernel_size=(1, 3, 3), stride=(1, 2, 2))

        # Mixed blocks
        self.mixed_3b = InceptionModule(192, [64, 96, 128, 16, 32, 32])
        self.mixed_3c = InceptionModule(256, [128, 128, 192, 32, 96, 64])

        self.maxpool3d_4a_3x3 = MaxPool3dSamePadding(kernel_size=(3, 3, 3), stride=(2, 2, 2))

        self.mixed_4b = InceptionModule(480, [192, 96, 208, 16, 48, 64])
        self.mixed_4c = InceptionModule(512, [160, 112, 224, 24, 64, 64])
        self.mixed_4d = InceptionModule(512, [128, 128, 256, 24, 64, 64])
        self.mixed_4e = InceptionModule(512, [112, 144, 288, 32, 64, 64])
        self.mixed_4f = InceptionModule(528, [256, 160, 320, 32, 128, 128])

        self.maxpool3d_5a_2x2 = MaxPool3dSamePadding(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.mixed_5b = InceptionModule(832, [256, 160, 320, 32, 128, 128])
        self.mixed_5c = InceptionModule(832, [384, 192, 384, 48, 128, 128])

        # Final layers (not used for feature extraction, but kept for completeness)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.5)
        self.logits = nn.Linear(1024, num_classes)

        # Store feature layer name for external access
        self.feature_layer_name = "avgpool"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward for classification; to extract features use the hook mechanism."""
        x = self.conv3d_1a_7x7(x)
        x = self.maxpool3d_2a_3x3(x)
        x = self.conv3d_2b_1x1(x)
        x = self.conv3d_2c_3x3(x)
        x = self.maxpool3d_3a_3x3(x)
        x = self.mixed_3b(x)
        x = self.mixed_3c(x)
        x = self.maxpool3d_4a_3x3(x)
        x = self.mixed_4b(x)
        x = self.mixed_4c(x)
        x = self.mixed_4d(x)
        x = self.mixed_4e(x)
        x = self.mixed_4f(x)
        x = self.maxpool3d_5a_2x2(x)
        x = self.mixed_5b(x)
        x = self.mixed_5c(x)
        x = self.avgpool(x)          # (B, 1024, 1, 1, 1)
        x = self.dropout(x)
        x = x.reshape(x.shape[0], -1)  # (B, 1024)
        x = self.logits(x)
        return x

    def load_state_dict_from_file(self, path: str) -> None:
        """Load pretrained weights from a .pth file (Kinetics‑400 I3D)."""
        state_dict = torch.load(path, map_location="cpu")
        # Some checkpoints may have keys prefixed with "module."; strip them.
        new_state = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            new_state[k] = v
        self.load_state_dict(new_state, strict=False)
        print(f"Loaded I3D weights from {path}.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Kinetics‑400 normalization statistics (RGB)
_KINETICS_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
_KINETICS_STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)


def load_i3d_model(
    model_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Tuple[InceptionI3d, str]:
    """
    Load the Inception‑I3D model with Kinetics‑400 weights.

    Args:
        model_path: Path to a .pth file containing I3D state dict.
                    If ``None``, attempts to download from a known URL;
                    if the download fails, raises ``FileNotFoundError``.
        device: Device on which to load the model (default: CPU).

    Returns:
        - `model` (InceptionI3d): The loaded model in evaluation mode.
        - `feature_layer_name` (str): Name of the layer to extract features
          from (`'avgpool'`).
    """
    if device is None:
        device = torch.device("cpu")

    model = InceptionI3d(num_classes=400)

    if model_path is not None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"I3D checkpoint not found: {model_path}")
        model.load_state_dict_from_file(model_path)
    else:
        # Attempt to download a commonly used checkpoint
        try:
            # Public URL to a converted Kinetics‑400 I3D checkpoint (from e.g., PyTorch I3D repo)
            url = "https://github.com/piergiaj/pytorch-i3d/raw/master/models/rgb_imagenet.pt"
            # Note: This URL may not work for everyone; fallback to an error message.
            state_dict = torch.hub.load_state_dict_from_url(
                url, map_location="cpu", check_hash=True
            )
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            raise FileNotFoundError(
                "Unable to load I3D weights automatically. Please provide a valid "
                "'model_path' pointing to a Kinetics‑400 I3D checkpoint.\n"
                f"Download error: {e}"
            )

    model.eval()
    model.to(device)
    return model, model.feature_layer_name


def preprocess_video(video_tensor: torch.Tensor) -> torch.Tensor:
    """
    Preprocess a video clip for I3D feature extraction.

    - Converts to float tensor with shape (T, 3, H, W).
    - Resizes each frame so the shorter side is 224, then centre‑crops to 224×224.
    - Normalises with Kinetics‑400 mean and std.
    - Returns (1, 3, T, 224, 224).

    Args:
        video_tensor: Input video in pixel space. Acceptable shapes:
            - ``(T, H, W, 3)``, values in [0, 255] (uint8) or [0, 1] (float).
            - ``(T, 3, H, W)``, same value range.

    Returns:
        Preprocessed tensor ``(1, 3, T, 224, 224)`` on CPU (usually).
    """
    if video_tensor.dim() == 4 and video_tensor.shape[-1] == 3:
        # (T, H, W, 3) -> (T, 3, H, W)
        video_tensor = video_tensor.permute(0, 3, 1, 2)
    elif not (video_tensor.dim() == 4 and video_tensor.shape[1] == 3):
        raise ValueError(
            f"Expected shape (T, H, W, 3) or (T, 3, H, W), got {video_tensor.shape}"
        )

    if video_tensor.dtype == torch.uint8:
        video_tensor = video_tensor.float() / 255.0
    elif video_tensor.max() > 1.0:
        warnings.warn("Video tensor has values > 1.0, assuming [0,255] range.")
        video_tensor = video_tensor.float() / 255.0

    T, C, H, W = video_tensor.shape
    if C != 3:
        raise ValueError("Video must have 3 colour channels.")

    # Resize to 224 shortest side and centre crop
    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.CenterCrop(224),
    ])
    # Transform expects (C, H, W) per image; apply to each frame
    frames = [transform(video_tensor[t]) for t in range(T)]
    video_tensor = torch.stack(frames, dim=1)   # (3, T, 224, 224)
    video_tensor = video_tensor.permute(1, 0, 2, 3)  # (T, 3, 224, 224)

    # Normalise: (value - mean) / std
    mean = video_tensor.new_tensor(_KINETICS_MEAN).view(1, 3, 1, 1)
    std = video_tensor.new_tensor(_KINETICS_STD).view(1, 3, 1, 1)
    video_tensor = (video_tensor - mean) / std

    # Add batch dimension: (1, 3, T, 224, 224)
    video_tensor = video_tensor.unsqueeze(0)
    return video_tensor


def extract_features(
    model: InceptionI3d,
    video_tensor: torch.Tensor,
    feature_layer: str = "avgpool",
    batch_size: Optional[int] = None,
) -> np.ndarray:
    """
    Extract I3D features from a batch of preprocessed video clips.

    Args:
        model: I3D model in eval mode.
        video_tensor: Batched videos of shape ``(B, 3, T, 224, 224)``.
        feature_layer: Name of the layer whose output is taken as features.
                       Default ``'avgpool'`` (the adaptive average pool layer).
        batch_size: Not used; kept for API compatibility.

    Returns:
        NumPy array of shape ``(B, D)`` where D is the feature dimension (1024).
    """
    model.eval()
    device = next(model.parameters()).device
    video_tensor = video_tensor.to(device)

    features = []

    # Hook to capture output of the specified layer
    def hook_fn(module, input, output):
        # output shape: (B, 1024, 1, 1, 1) -> flatten to (B, 1024)
        features.append(output.detach().cpu().numpy().reshape(output.shape[0], -1))

    # Register hook and run model
    target_module = dict(model.named_modules())[feature_layer]
    handle = target_module.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(video_tensor)   # forward to trigger hook
    handle.remove()

    if not features:
        raise RuntimeError(f"No features captured from layer '{feature_layer}'.")
    return np.concatenate(features, axis=0)


def calculate_fvd(
    real_features: np.ndarray,
    generated_features: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Compute the Fréchet Video Distance (FVD) between two sets of features.

    The distance is defined as:
        FVD = ||μ_real - μ_gen||² + Tr(C_real + C_gen - 2 (C_real C_gen)^{1/2})

    Args:
        real_features: Real video features, shape ``(N_real, D)``.
        generated_features: Generated video features, shape ``(N_gen, D)``.
        eps: Small value added to the diagonal of covariance matrices for
             numerical stability.

    Returns:
        FVD as a float.
    """
    mu_r = np.mean(real_features, axis=0)
    mu_g = np.mean(generated_features, axis=0)

    C_r = np.cov(real_features, rowvar=False)
    C_g = np.cov(generated_features, rowvar=False)

    # Add ridge
    C_r = C_r + eps * np.eye(C_r.shape[0])
    C_g = C_g + eps * np.eye(C_g.shape[0])

    # Compute matrix square root of C_r * C_g
    prod = C_r @ C_g
    try:
        sqrtm = scipy.linalg.sqrtm(prod)
    except scipy.linalg.LinAlgError:
        # fallback to eigenvalue decomposition for stability
        warnings.warn("sqrtm failed, using eigendecomposition approximation.")
        eigvals, eigvecs = np.linalg.eigh(prod)
        sqrtm = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 0))) @ eigvecs.T

    diff = mu_r - mu_g
    diff_sq = np.dot(diff, diff)   # squared L2 distance
    trace_term = np.trace(C_r + C_g - 2 * sqrtm)
    # Ensure non‑negative (numerical noise)
    fvd = max(0.0, diff_sq + trace_term)
    return float(fvd)


def compute_chunkwise_fvd(
    model: InceptionI3d,
    gen_video: torch.Tensor,
    real_features_all: np.ndarray,
    chunk_size: int = 16,
) -> List[float]:
    """
    Split a long generated video into chunks and compute FVD for each chunk
    against a pre‑computed real feature set.

    Args:
        model: I3D model.
        gen_video: Generated video in pixel space, shape ``(T, H, W, 3)``
                   or ``(T, 3, H, W)`` (values [0,255] or [0,1]).
        real_features_all: Feature array for the real dataset ``(N_real, D)``.
        chunk_size: Number of frames per chunk (e.g., 16).

    Returns:
        List of FVD scores, one per chunk.
    """
    T = gen_video.shape[0]
    # Standardise shape to (T, H, W, 3)
    if gen_video.shape[1] == 3:
        gen_video = gen_video.permute(0, 2, 3, 1)  # (T, H, W, 3)

    device = next(model.parameters()).device
    fvd_scores: List[float] = []

    for start in range(0, T, chunk_size):
        end = start + chunk_size
        if end > T:
            break
        chunk = gen_video[start:end]  # (chunk_size, H, W, 3)
        # Preprocess: returns (1, 3, chunk_size, 224, 224)
        preprocessed = preprocess_video(chunk)
        gen_feats = extract_features(model, preprocessed)  # (1, D)
        fvd = calculate_fvd(real_features_all, gen_feats)
        fvd_scores.append(fvd)

    return fvd_scores


def extract_features_from_dataset(
    model: InceptionI3d,
    dataset: Union[Dataset, DataLoader],
    feature_layer: str = "avgpool",
    batch_size: int = 16,
    num_workers: int = 4,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Pre‑compute I3D features for an entire dataset of video clips.

    The ``dataset`` can be a ``Dataset`` object that returns a tensor of shape
    ``(T, H, W, 3)`` or ``(T, 3, H, W)``, OR a ``DataLoader``.  If a
    ``Dataset`` is given, a ``DataLoader`` is constructed internally.

    Args:
        model: I3D model.
        dataset: A dataset (or DataLoader) providing raw video tensors.
        feature_layer: I3D layer from which to extract features.
        batch_size: Batch size for the DataLoader (ignored if ``dataset`` is
                    already a DataLoader).
        num_workers: Number of loader workers.
        device: Device for I3D inference (default: model's current device).

    Returns:
        Feature array of shape ``(N, D)``.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    if isinstance(dataset, DataLoader):
        loader = dataset
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )

    all_features: List[np.ndarray] = []
    for batch in loader:
        # batch may be a tensor or a tuple; handle raw video tensor directly
        if isinstance(batch, (list, tuple)):
            video = batch[0]
        else:
            video = batch

        # video is likely on CPU as raw pixel values; ensure it is a 4‑D or 5‑D tensor
        if video.dim() == 4:   # (B, T, H, W, 3) or (B, T, 3, H, W)
            # add batch dim? Actually it's already batched: B is batch size.
            # We need to preprocess each video individually because they may have different T.
            # However, for I3D we can batch with the same T? To keep simple, we process one by one.
            # We'll loop over B.
            for i in range(video.shape[0]):
                sample = video[i]  # (T, H, W, 3) or (T, 3, H, W)
                preprocessed = preprocess_video(sample)
                feats = extract_features(model, preprocessed, feature_layer)
                all_features.append(feats)
        else:
            # Single video: (T, H, W, 3)
            preprocessed = preprocess_video(video)
            feats = extract_features(model, preprocessed, feature_layer)
            all_features.append(feats)

    if not all_features:
        raise RuntimeError("No features extracted from dataset.")

    return np.concatenate(all_features, axis=0)

