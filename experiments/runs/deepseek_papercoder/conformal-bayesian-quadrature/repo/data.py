# data.py
"""
Data generators for the three experiments described in
"Conformal Prediction as Bayesian Quadrature".

Implements:
- BinomialDataGenerator    : synthetic calibration data for binomial risk.
- HeteroDataGenerator      : synthetic heteroskedastic regression data.
- CocoDataLoader           : MS-COCO validation set with pre‑computed
                             per‑image sigmoid probabilities using a
                             pre‑trained TResNet‑M model.

All generators inherit from the abstract `DataGenerator` base class.
"""

from __future__ import annotations

import abc
import json
import math
import os
from pathlib import Path
from typing import Any, List, Sequence, Tuple, Union, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.datasets import CocoDetection


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class DataGenerator(abc.ABC):
    """Abstract base class for data generators used in the experiments."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    @abc.abstractmethod
    def generate(self) -> Tuple[Any, Any]:
        """
        Produce calibration and test data for one trial.

        Returns:
            A tuple (calib_data, test_data).  The structure depends on the
            experiment.  For the binomial experiment test_data is None.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Synthetic data: binomial experiment
# ---------------------------------------------------------------------------

class BinomialDataGenerator(DataGenerator):
    """Generate synthetic Uniform(0,1) variates for the binomial experiment.

    Each trial consists of `n_cal` calibration points, each with `K`
    independent draws from Uniform(0,1).
    """

    def __init__(
        self,
        n_calibration: int,
        K: int,
        rng: np.random.Generator,
    ) -> None:
        """
        Args:
            n_calibration: Number of calibration points (e.g. 10).
            K: Number of synthetic Bernoulli trials per point (e.g. 4).
            rng: Seeded numpy random generator.
        """
        super().__init__(rng)
        self.n_cal = n_calibration
        self.K = K

    def generate(self) -> Tuple[np.ndarray, None]:
        """Return (calib_array, None).  calib_array has shape (n_cal, K)."""
        calib = self.rng.uniform(0.0, 1.0, size=(self.n_cal, self.K))
        return calib, None


# ---------------------------------------------------------------------------
# Synthetic data: heteroskedastic regression
# ---------------------------------------------------------------------------

class HeteroDataGenerator(DataGenerator):
    """Generate heteroskedastic regression data: X ~ U[0,4], Y|X ~ N(0, X^2)."""

    def __init__(
        self,
        n_calibration: int,
        n_test: int,
        rng: np.random.Generator,
    ) -> None:
        """
        Args:
            n_calibration: Number of calibration points (200).
            n_test: Number of test points for risk estimation (e.g., 50000).
            rng: Seeded numpy random generator.
        """
        super().__init__(rng)
        self.n_cal = n_calibration
        self.n_test = n_test

    def generate(self) -> Tuple[Tuple[np.ndarray, np.ndarray],
                                Tuple[np.ndarray, np.ndarray]]:
        """
        Returns:
            ( (X_cal, Y_cal), (X_test, Y_test) ), each a tuple of 1D arrays.
        """
        # Calibration
        X_cal = self.rng.uniform(0.0, 4.0, size=self.n_cal)
        Y_cal = self.rng.normal(loc=0.0, scale=np.abs(X_cal))

        # Test
        X_test = self.rng.uniform(0.0, 4.0, size=self.n_test)
        Y_test = self.rng.normal(loc=0.0, scale=np.abs(X_test))

        return (X_cal, Y_cal), (X_test, Y_test)


# ---------------------------------------------------------------------------
# TResNet‑M architecture (for COCO experiment)
# ---------------------------------------------------------------------------

# Helper modules from the official TResNet implementation
# (https://github.com/Alibaba-MIIL/TResNet), MIT license.
# They are included here to avoid external dependencies outside PyTorch.

class SpaceToDepth(nn.Module):
    """Space‑to‑Depth transform: rearranges spatial blocks into channel dim."""

    def __init__(self, block_size: int = 4) -> None:
        super().__init__()
        assert block_size == 4, "Only block_size=4 is used in TResNet‑M."
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pixel_unshuffle(x, self.block_size)


class AntiAliasDownsampleLayer(nn.Module):
    """
    BlurPool‑inspired anti‑aliasing down‑sampling layer.
    Applies avg pooling before 1×1 convolution with stride 1.
    """

    def __init__(self, channels: int, filt_size: int = 3, stride: int = 2) -> None:
        super().__init__()
        if filt_size == 1:
            a = np.array([1.0])
        elif filt_size == 2:
            a = np.array([0.5, 0.5])
        elif filt_size == 3:
            a = np.array([1.0, 2.0, 1.0])
        elif filt_size == 4:
            a = np.array([1.0, 3.0, 3.0, 1.0])
        elif filt_size == 5:
            a = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
        elif filt_size == 6:
            a = np.array([1.0, 5.0, 10.0, 10.0, 5.0, 1.0])
        elif filt_size == 7:
            a = np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])
        else:
            raise ValueError(f"Unsupported filt_size: {filt_size}")
        a = a / a.sum()
        filt = torch.tensor(a, dtype=torch.float32).reshape(1, 1, -1)
        self.register_buffer("filt", filt.repeat(channels, 1, 1, 1))
        self.pad = nn.ReflectionPad2d((filt_size // 2, filt_size // 2, 0, 0))
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # horizontal blur + strided average pool
        return F.conv2d(
            self.pad(x),
            self.filt,
            groups=x.shape[1],
            stride=(1, self.stride),
        )


class BasicBlock(nn.Module):
    """TResNet basic block with anti‑aliasing downsampling in the shortcut."""

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        use_aa_downsample: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        if stride != 1 and use_aa_downsample:
            self.aa_downsample = AntiAliasDownsampleLayer(planes)
        else:
            self.aa_downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            if self.aa_downsample is not None and self.stride != 1:
                identity = self.aa_downsample(identity)
        elif self.stride != 1 and self.aa_downsample is not None:
            identity = self.aa_downsample(identity)

        out += identity
        out = self.relu(out)
        return out


class TResNetM(nn.Module):
    """
    TResNet‑M architecture.

    Configuration matches the pre‑trained checkpoint used in
    Angelopoulos & Bates (2023).  The model expects 224×224 inputs and
    outputs 80‑dimensional logits (multi‑label classification).
    """

    def __init__(self, layers: Sequence[int], num_classes: int = 80) -> None:
        super().__init__()
        self.inplanes = 64
        self.base_width = 64
        self.groups = 1
        self.num_classes = num_classes

        # Stem: SpaceToDepth(4) -> Conv2d
        self.space_to_depth = SpaceToDepth(block_size=4)
        self.conv1 = nn.Conv2d(48, self.inplanes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # Layers
        self.layer1 = self._make_layer(BasicBlock, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(BasicBlock, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        # Weight initialisation (as in the original)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: type,
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride=stride,
                  downsample=downsample, use_aa_downsample=True)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, stride=1,
                      downsample=None, use_aa_downsample=False)
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.space_to_depth(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ---------------------------------------------------------------------------
# COCO experiment data loader
# ---------------------------------------------------------------------------

class CocoDataLoader(DataGenerator):
    """
    MS‑COCO validation set loader with pre‑extracted TResNet‑M probabilities.

    Usage:
        loader = CocoDataLoader(data_root, annotation_file, model_path,
                                n_calibration, n_test, seed=123)
        # Per‑trial random splits are obtained by calling .generate()
        (probs_cal, labels_cal), (probs_test, labels_test) = loader.generate()
    """

    def __init__(
        self,
        data_root: str,
        annotation_file: str,
        model_path: str,
        n_calibration: int,
        n_test: int,
        rng: np.random.Generator,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Args:
            data_root: Path to the COCO val2014 directory.
            annotation_file: Path to instances_val2014.json.
            model_path: Path to the pre‑trained TResNet‑M .pth file.
            n_calibration: Number of calibration images per trial.
            n_test: Number of test images per trial.
            rng: Numpy random generator (will be used per trial).
            device: Torch device; defaults to GPU if available.
        """
        super().__init__(rng)
        self.data_root = Path(data_root)
        self.annotation_file = Path(annotation_file)
        self.model_path = Path(model_path)
        self.n_cal = n_calibration
        self.n_test = n_test

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # ----- Load model & compute probabilities once -----
        self._model = self._load_model()
        self._model.eval()
        self._model.to(self.device)

        # Image preprocessing (same as original TResNet‑M pipeline)
        self._transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # Load COCO dataset via torchvision
        coco_dataset = CocoDetection(
            root=str(self.data_root),
            annFile=str(self.annotation_file),
            transform=self._transform,
        )
        # Build multi‑hot label map (sorted by COCO category ID)
        self._class_ids = sorted(coco_dataset.coco.getCatIds())
        self._num_classes = len(self._class_ids)
        assert self._num_classes == 80, "Expected 80 categories in COCO 2014."

        # Compute probabilities for the entire validation set
        self._all_probs, self._all_labels = self._compute_probabilities(coco_dataset)
        self._total = self._all_probs.shape[0]
        print(f"CocoDataLoader: cached {self._total} images.")

    def _load_model(self) -> TResNetM:
        """Instantiate and load the pre‑trained TResNet‑M."""
        # Standard TResNet‑M configuration: [3, 4, 11, 3] layers
        model = TResNetM(layers=[3, 4, 11, 3], num_classes=self._num_classes)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"TResNet‑M checkpoint not found at {self.model_path}. "
                "Please download the checkpoint (e.g. from the conformal "
                "prediction book repository) and update the config."
            )
        state_dict = torch.load(self.model_path, map_location="cpu")
        # Handle common wrapper keys
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if "model" in state_dict:
            state_dict = state_dict["model"]
        # Remove "module." prefix if present (DDP wrapper)
        new_state = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            new_state[k] = v
        model.load_state_dict(new_state, strict=True)
        return model

    def _compute_probabilities(
        self,
        dataset: CocoDetection,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass on the whole dataset and return (probs, labels)."""
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=64, shuffle=False,
            num_workers=4, pin_memory=True,
        )
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(self.device)
                logits = self._model(images)
                probs = torch.sigmoid(logits).cpu().numpy()

                # Build multi‑hot labels
                batch_labels = np.zeros((len(targets), self._num_classes),
                                        dtype=np.float32)
                for i, anns in enumerate(targets):
                    if len(anns) == 0:
                        continue
                    cat_ids = [ann["category_id"] for ann in anns]
                    idxs = [self._class_ids.index(cid) for cid in cat_ids if cid in self._class_ids]
                    batch_labels[i, idxs] = 1.0

                all_probs.append(probs)
                all_labels.append(batch_labels)

        return (np.concatenate(all_probs, axis=0),
                np.concatenate(all_labels, axis=0))

    def generate(
        self, rng: Optional[np.random.Generator] = None
    ) -> Tuple[Tuple[np.ndarray, np.ndarray],
               Tuple[np.ndarray, np.ndarray]]:
        """
        Randomly split the cached dataset into calibration and test subsets.

        Args:
            rng: If provided, overrides the internal generator for this trial.

        Returns:
            ( (probs_cal, labels_cal), (probs_test, labels_test) ),
            each a numpy array of shape (n, 80).
        """
        if rng is not None:
            self.rng = rng
        indices = self.rng.permutation(self._total)
        cal_idx = indices[:self.n_cal]
        test_idx = indices[self.n_cal:self.n_cal + self.n_test]
        assert len(test_idx) == self.n_test, (
            f"Insufficient test images: need {self.n_test}, but only "
            f"{self._total - self.n_cal} available."
        )
        return (
            (self._all_probs[cal_idx], self._all_labels[cal_idx]),
            (self._all_probs[test_idx], self._all_labels[test_idx]),
        )

