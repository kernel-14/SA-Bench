"""
Dataset loading and preprocessing for Hi-MAR.

Supports:
  - ImageNet (class-conditional, 256x256)
  - MS-COCO (text-to-image, 256x256)

VAE encoding is done on-the-fly during training using the KL-16 VAE.
Both low-resolution (128x128) and high-resolution (256x256) images are
encoded to produce the two-scale token sequences.
"""

import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF


# ---------------------------------------------------------------------------
# Image transforms
# ---------------------------------------------------------------------------

def build_train_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


# ---------------------------------------------------------------------------
# ImageNet dataset
# ---------------------------------------------------------------------------

class ImageNetDataset(Dataset):
    """
    ImageNet dataset returning both low-res (128x128) and high-res (256x256) images.

    Returns:
        img_large: (3, 256, 256) normalized tensor
        img_small: (3, 128, 128) normalized tensor (downsampled from img_large)
        label:     int class index
    """

    def __init__(self, root: str, split: str = "train", image_size: int = 256):
        self.image_size = image_size
        self.image_size_small = image_size // 2

        if split == "train":
            transform = build_train_transform(image_size)
        else:
            transform = build_eval_transform(image_size)

        self.dataset = datasets.ImageFolder(
            root=os.path.join(root, split),
            transform=transform,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        img_large, label = self.dataset[idx]
        # Downsample to small resolution
        img_small = F.interpolate(
            img_large.unsqueeze(0),
            size=(self.image_size_small, self.image_size_small),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)
        return img_large, img_small, label


# ---------------------------------------------------------------------------
# MS-COCO dataset
# ---------------------------------------------------------------------------

class COCOTextImageDataset(Dataset):
    """
    MS-COCO dataset for text-to-image generation.

    Each sample returns one image with one randomly selected caption.
    Text embeddings are pre-computed using CLIP (or computed on-the-fly).

    Returns:
        img_large: (3, 256, 256) normalized tensor
        img_small: (3, 128, 128) normalized tensor
        text_embed: (L, text_embed_dim) CLIP text embedding
    """

    def __init__(
        self,
        image_dir: str,
        ann_file: str,
        split: str = "train",
        image_size: int = 256,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        precomputed_embeds_path: Optional[str] = None,
    ):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.image_size_small = image_size // 2
        self.split = split

        if split == "train":
            self.transform = build_train_transform(image_size)
        else:
            self.transform = build_eval_transform(image_size)

        # Load annotations
        with open(ann_file) as f:
            data = json.load(f)

        # Build image_id → file_name mapping
        self.id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

        # Build image_id → list of captions
        self.id_to_captions: Dict[int, List[str]] = {}
        for ann in data["annotations"]:
            iid = ann["image_id"]
            if iid not in self.id_to_captions:
                self.id_to_captions[iid] = []
            self.id_to_captions[iid].append(ann["caption"])

        self.image_ids = list(self.id_to_captions.keys())

        # Pre-computed CLIP embeddings (optional, for speed)
        self.precomputed_embeds = None
        if precomputed_embeds_path and os.path.exists(precomputed_embeds_path):
            self.precomputed_embeds = torch.load(precomputed_embeds_path, map_location="cpu")

        # Lazy CLIP model (loaded on first use if no pre-computed embeds)
        self._clip_model = None
        self._clip_tokenizer = None
        self.clip_model_name = clip_model_name

    def _load_clip(self):
        from transformers import CLIPModel, CLIPTokenizer
        self._clip_tokenizer = CLIPTokenizer.from_pretrained(self.clip_model_name)
        self._clip_model = CLIPModel.from_pretrained(self.clip_model_name)
        self._clip_model.eval()

    def _encode_text(self, caption: str) -> torch.Tensor:
        if self._clip_model is None:
            self._load_clip()
        inputs = self._clip_tokenizer(
            caption, return_tensors="pt", padding="max_length",
            max_length=77, truncation=True
        )
        with torch.no_grad():
            text_features = self._clip_model.get_text_features(**inputs)
        return text_features.squeeze(0)  # (text_embed_dim,)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_id = self.image_ids[idx]
        filename = self.id_to_filename[image_id]
        captions = self.id_to_captions[image_id]

        # Load and transform image
        img_path = self.image_dir / filename
        img = Image.open(img_path).convert("RGB")
        img_large = self.transform(img)
        img_small = F.interpolate(
            img_large.unsqueeze(0),
            size=(self.image_size_small, self.image_size_small),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)

        # Get text embedding
        if self.precomputed_embeds is not None:
            text_embed = self.precomputed_embeds[image_id]
        else:
            caption = random.choice(captions)
            text_embed = self._encode_text(caption)

        return img_large, img_small, text_embed


# ---------------------------------------------------------------------------
# VAE tokenizer wrapper
# ---------------------------------------------------------------------------

class VAETokenizer:
    """
    Wraps the KL-16 VAE to encode images into continuous token sequences.

    The KL-16 VAE has a downsampling factor of 16:
      128x128 → 8x8 latent  → 64 tokens
      256x256 → 16x16 latent → 256 tokens
    """

    def __init__(self, vae_path: str = "stabilityai/sd-vae-ft-ema", device: str = "cuda"):
        from diffusers import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained(vae_path)
        self.vae.eval()
        self.vae.to(device)
        self.device = device
        self.scale_factor = 0.18215  # standard KL-16 scale factor

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to flattened token sequences.

        Args:
            images: (B, 3, H, W) normalized to [-1, 1]
        Returns:
            tokens: (B, H/16 * W/16, 16) continuous tokens
        """
        images = images.to(self.device)
        posterior = self.vae.encode(images).latent_dist
        latents = posterior.sample() * self.scale_factor
        B, C, h, w = latents.shape
        # Reshape to (B, h*w, C)
        tokens = latents.permute(0, 2, 3, 1).reshape(B, h * w, C)
        return tokens

    @torch.no_grad()
    def decode(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Decode token sequences back to images.

        Args:
            tokens: (B, h*w, C) continuous tokens
            h, w:   spatial dimensions of the latent grid
        Returns:
            images: (B, 3, H*16, W*16) in [-1, 1]
        """
        B, N, C = tokens.shape
        latents = tokens.reshape(B, h, w, C).permute(0, 3, 1, 2)
        latents = latents / self.scale_factor
        images = self.vae.decode(latents).sample
        return images.clamp(-1, 1)


# ---------------------------------------------------------------------------
# Masking utilities
# ---------------------------------------------------------------------------

def sample_mask_ratio_phase1(batch_size: int, min_ratio: float = 0.7, max_ratio: float = 1.0) -> torch.Tensor:
    """Sample masking ratio uniformly in [min_ratio, max_ratio] for phase 1."""
    return torch.rand(batch_size) * (max_ratio - min_ratio) + min_ratio


def sample_mask_ratio_cosine(batch_size: int) -> torch.Tensor:
    """MaskGIT cosine masking: r = cos(π/2 * u), u ~ Uniform(0, 1)."""
    u = torch.rand(batch_size)
    return torch.cos(math.pi / 2 * u)


def sample_mask_ratio_beta(batch_size: int, alpha: float = 4.0, beta: float = 1.0) -> torch.Tensor:
    """Beta distribution masking for MS-COCO (AutoNAT-L style)."""
    dist = torch.distributions.Beta(
        torch.tensor(alpha), torch.tensor(beta)
    )
    return dist.sample((batch_size,))


def create_mask(batch_size: int, num_tokens: int, mask_ratios: torch.Tensor) -> torch.Tensor:
    """
    Create random binary masks.

    Args:
        batch_size:  B
        num_tokens:  N
        mask_ratios: (B,) fraction of tokens to mask
    Returns:
        mask: (B, N) bool, True = masked
    """
    mask = torch.zeros(batch_size, num_tokens, dtype=torch.bool)
    for i, ratio in enumerate(mask_ratios):
        num_masked = max(1, int(num_tokens * ratio.item()))
        perm = torch.randperm(num_tokens)[:num_masked]
        mask[i, perm] = True
    return mask


# ---------------------------------------------------------------------------
# DataLoader builders
# ---------------------------------------------------------------------------

def build_imagenet_loaders(
    data_path: str,
    batch_size: int,
    num_workers: int = 8,
    image_size: int = 256,
    distributed: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = ImageNetDataset(data_path, split="train", image_size=image_size)
    val_dataset = ImageNetDataset(data_path, split="val", image_size=image_size)

    train_sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def build_coco_loaders(
    image_dir: str,
    ann_dir: str,
    batch_size: int,
    num_workers: int = 8,
    image_size: int = 256,
    clip_model_name: str = "openai/clip-vit-base-patch32",
    distributed: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = COCOTextImageDataset(
        image_dir=os.path.join(image_dir, "train2014"),
        ann_file=os.path.join(ann_dir, "captions_train2014.json"),
        split="train",
        image_size=image_size,
        clip_model_name=clip_model_name,
    )
    val_dataset = COCOTextImageDataset(
        image_dir=os.path.join(image_dir, "val2014"),
        ann_file=os.path.join(ann_dir, "captions_val2014.json"),
        split="val",
        image_size=image_size,
        clip_model_name=clip_model_name,
    )

    train_sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
