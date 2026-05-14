## dataset.py
"""
Data loading and batching utilities for Pyramidal Flow Matching.

Implements:
- VideoDataset : loads and preprocesses video clips from various sources.
- ImageDataset : loads and preprocesses images with aspect-ratio bucketing.
- PatchPackCollator : packs variable-length token sequences into a single
  "Patch n' Pack" batch for efficient training of the MM-DiT backbone.

All classes read their configuration from the project's global config (dict‑like).
Heavy decoding/encoding (VAE) is performed externally by the Trainer.
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
from torch.utils.data import Dataset
from torchvision.io import read_video, read_video_timestamps
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _resize_and_crop_video(
    video: torch.Tensor,
    target_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Resize a video tensor so that the shorter side equals the shorter side
    of target_size, then center-crop to exactly target_size.

    Args:
        video: Float tensor of shape (T, C, H, W) in [0, 1].
        target_size: (height, width).

    Returns:
        Transformed tensor of shape (T, C, target_h, target_w).
    """
    T, C, H, W = video.shape
    th, tw = target_size

    # Compute scale such that shorter side matches the shorter target side
    min_side = min(H, W)
    min_target = min(th, tw)
    scale = min_target / min_side

    new_h = max(int(H * scale), th)
    new_w = max(int(W * scale), tw)

    # Permute to (C, T, H, W) for F.interpolate (operates on 4D: N, C, H, W)
    video = video.permute(1, 0, 2, 3)  # (C, T, H, W)
    video = F.interpolate(video, size=(new_h, new_w), mode='bilinear', align_corners=False)
    video = video.permute(1, 0, 2, 3)  # back to (T, C, new_h, new_w)

    # Center crop
    h_start = (new_h - th) // 2
    w_start = (new_w - tw) // 2
    video = video[:, :, h_start:h_start + th, w_start:w_start + tw]
    return video


def _resize_and_crop_image(
    image: torch.Tensor,
    target_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Similar to _resize_and_crop_video but for a single image of shape (C, H, W).
    """
    C, H, W = image.shape
    th, tw = target_size
    min_side = min(H, W)
    min_target = min(th, tw)
    scale = min_target / min_side
    new_h = max(int(H * scale), th)
    new_w = max(int(W * scale), tw)

    image = image.unsqueeze(0)  # (1, C, H, W)
    image = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)
    image = image.squeeze(0)  # (C, new_h, new_w)

    h_start = (new_h - th) // 2
    w_start = (new_w - tw) // 2
    image = image[:, h_start:h_start + th, w_start:w_start + tw]
    return image


def _load_video_frames(path: str, target_frames: int) -> Optional[torch.Tensor]:
    """
    Attempt to load a video segment of target_frames uniformly sampled.
    Returns a torch Tensor of shape (T, C, H, W) with values in [0, 1] or None if invalid.
    """
    try:
        video, audio, info = read_video(path, start_pts=0, end_pts=None, pts_unit='sec')
        # video shape: (T, H, W, C) uint8
        if video.numel() == 0:
            return None
        # Permute to (T, C, H, W) and convert to float
        video = video.permute(0, 3, 1, 2).float() / 255.0
        total_frames = video.shape[0]
        if total_frames < target_frames:
            logger.warning(f"Video {path} has only {total_frames} frames, < {target_frames}. Skipping.")
            return None

        # Random crop of exactly target_frames
        start_frame = random.randint(0, total_frames - target_frames)
        video = video[start_frame:start_frame + target_frames]
        return video
    except Exception as e:
        logger.error(f"Failed to load video {path}: {e}")
        return None


def _find_files(root: str, extensions: List[str]) -> List[str]:
    """Recursively collect files with given extensions under root."""
    root_path = Path(root)
    if not root_path.exists():
        logger.warning(f"Data directory does not exist: {root}")
        return []
    files = []
    for ext in extensions:
        files.extend([str(p) for p in root_path.rglob(f"*{ext}")])
    return sorted(files)


def _load_captions_from_json(path: str) -> Dict[str, str]:
    """Load a JSON mapping from identifier -> caption."""
    if not path or not os.path.isfile(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _get_caption_from_cache_or_fallback(
    identifier: str,
    cache: Dict[str, str],
    original_captions: Dict[str, str],
    fallback: str = ""
) -> str:
    """
    Return the best available caption for an item.
    Hierarchy: recaption cache -> original captions -> fallback.
    """
    if identifier in cache and cache[identifier]:
        return cache[identifier]
    if identifier in original_captions and original_captions[identifier]:
        return original_captions[identifier]
    return fallback


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class VideoDataset(Dataset):
    """
    Video dataset for stage‑2 and stage‑3 training.

    Loads video files from all configured directories (WebVid‑10M, OpenVid‑1M,
    Open‑Sora Plan non‑watermark). Each sample is a fixed‑length segment of
    `target_frames` frames, resized and cropped to `resolution` and normalised
    to [-1, 1]. Captions are taken from the recaption cache or a metadata file.

    Args:
        cfg: The global configuration dict (must contain datasets.video keys).
        split: 'train' or 'val'. Only 'train' is used for now.
        target_frames: number of consecutive frames, e.g., 120 for 5 seconds at 24 fps.
        resolution: (H, W) of the output video frames, e.g., (384, 384).
        recaption_cache: Path to a JSON file mapping video filename stems to recaptions,
                         or a pre‑loaded dict. If None, original captions (from a side
                         metadata file) are used.
        original_captions_path: Optional path to a JSON file with original dataset captions.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        split: str = "train",
        target_frames: int = 120,
        resolution: Tuple[int, int] = (384, 384),
        recaption_cache: Optional[Union[str, Dict[str, str]]] = None,
        original_captions_path: Optional[str] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.target_frames = target_frames
        self.resolution = resolution

        # Gather all video files from configured paths
        video_dirs = cfg["datasets"]["video"]
        self.video_paths = []
        for key in ["webvid10m", "openvid1m", "opensora_plan_highres"]:
            if key in video_dirs:
                self.video_paths.extend(_find_files(video_dirs[key], extensions=[".mp4", ".avi", ".mov"]))

        if not self.video_paths:
            raise RuntimeError("No video files found in configured directories!")

        # Load captions
        self.recaption_cache: Dict[str, str] = {}
        if isinstance(recaption_cache, str):
            self.recaption_cache = _load_captions_from_json(recaption_cache)
        elif isinstance(recaption_cache, dict):
            self.recaption_cache = recaption_cache

        self.original_captions: Dict[str, str] = {}
        if original_captions_path:
            self.original_captions = _load_captions_from_json(original_captions_path)

        # Pre‑compute which videos are long enough (optional acceleration)
        self._valid_indices = self._filter_long_enough()

        logger.info(f"VideoDataset: {len(self._valid_indices)} videos (out of {len(self.video_paths)} total) are usable (>= {target_frames} frames).")

    def _filter_long_enough(self) -> List[int]:
        """Quickly check which videos have at least target_frames frames."""
        valid = []
        for idx, path in enumerate(self.video_paths):
            # A fast check: read_video_timestamps returns a list of times; if it's longer than target_frames, okay.
            try:
                timestamps = read_video_timestamps(path, pts_unit='sec')
                if len(timestamps) >= self.target_frames:
                    valid.append(idx)
            except Exception:
                # We'll try loading the video in __getitem__ and skip if it fails.
                # Optimistically include it.
                valid.append(idx)
        return valid

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video_idx = self._valid_indices[idx]
        path = self.video_paths[video_idx]

        # Load video segment
        video = _load_video_frames(path, self.target_frames)
        if video is None:
            # Fallback – extremely unlikely after filter, but retry with another random sample
            new_idx = random.randint(0, len(self._valid_indices) - 1)
            return self.__getitem__(new_idx)

        # Resize and crop to target resolution, then normalize to [-1, 1]
        video = _resize_and_crop_video(video, self.resolution)
        video = 2.0 * video - 1.0  # [0,1] -> [-1,1]

        # Identify video: use stem as identifier for captions
        identifier = Path(path).stem
        caption = _get_caption_from_cache_or_fallback(
            identifier, self.recaption_cache, self.original_captions, fallback="a video"
        )

        return {
            "video": video,           # (T, C, H, W) float in [-1,1]
            "caption": caption,
            "video_id": identifier,
        }


class ImageDataset(Dataset):
    """
    Image dataset for stage‑1 (image‑only) training and for the 12.5% mixture
    in video stages.

    Divides all images into aspect‑ratio buckets so that each sample can be
    resized to a fixed resolution without extreme distortion. Bucket assignment
    is done at initialisation.

    Args:
        cfg: The global configuration dict (must contain datasets.image keys).
        split: 'train' or 'val'.
        buckets: List of (H, W) resolutions that determine the bucket sizes.
        recaption_cache: Path to JSON or dict mapping image filename stems to
                         recaptions. If None, original captions are used.
        original_captions_dir: Directory containing .txt caption files (stem.txt)
                               for original captions. If None, captions are taken
                               from recaption_cache only (empty fallback).
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        split: str = "train",
        buckets: Optional[List[Tuple[int, int]]] = None,
        recaption_cache: Optional[Union[str, Dict[str, str]]] = None,
        original_captions_dir: Optional[str] = None,
    ):
        super().__init__()
        self.cfg = cfg
        if buckets is None:
            # Default buckets covering a range of aspect ratios
            buckets = [(256, 256), (256, 512), (384, 384), (512, 256), (512, 512)]
        self.buckets = buckets

        # Gather all image files
        image_dirs = cfg["datasets"]["image"]
        extensions = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
        self.image_paths = []
        for key in ["laion_high_aesthetic", "cc12m", "sa1b_nonblurred", "journeydb", "synthetic"]:
            if key in image_dirs:
                self.image_paths.extend(_find_files(image_dirs[key], extensions=extensions))

        if not self.image_paths:
            raise RuntimeError("No image files found in configured directories!")

        # Captions
        self.recaption_cache: Dict[str, str] = {}
        if isinstance(recaption_cache, str):
            self.recaption_cache = _load_captions_from_json(recaption_cache)
        elif isinstance(recaption_cache, dict):
            self.recaption_cache = recaption_cache

        self.original_captions_dir = original_captions_dir

        # Assign each image to the best matching bucket and store resolution info
        self.samples = self._assign_buckets()

        logger.info(f"ImageDataset: {len(self.samples)} images assigned to {len(buckets)} buckets.")

    def _get_image_size(self, path: str) -> Tuple[int, int]:
        """Quickly determine image dimensions without fully decoding."""
        try:
            with Image.open(path) as img:
                return img.size  # (W, H) in PIL
        except Exception:
            return (0, 0)

    def _find_best_bucket(self, img_w: int, img_h: int) -> int:
        """Return index of the bucket whose aspect ratio is closest to the image."""
        best_idx = 0
        best_diff = float('inf')
        img_ar = img_w / img_h if img_h > 0 else 1.0
        for i, (bh, bw) in enumerate(self.buckets):
            bucket_ar = bw / bh if bh > 0 else 1.0
            diff = abs(img_ar - bucket_ar)
            if diff < best_diff:
                best_diff = diff
                best_idx = i
        return best_idx

    def _assign_buckets(self) -> List[Dict[str, Any]]:
        """Scan all images, assign bucket, and create a list of sample dicts."""
        sample_list = []
        for path in self.image_paths:
            w, h = self._get_image_size(path)
            if w <= 0 or h <= 0:
                continue
            bucket_idx = self._find_best_bucket(w, h)
            bucket_h, bucket_w = self.buckets[bucket_idx]

            identifier = Path(path).stem
            # Try to get caption from original txt file if directory is provided
            original_caption = ""
            if self.original_captions_dir:
                txt_path = os.path.join(self.original_captions_dir, identifier + ".txt")
                if os.path.isfile(txt_path):
                    try:
                        with open(txt_path, 'r', encoding='utf-8') as f:
                            original_caption = f.read().strip()
                    except Exception:
                        pass

            sample_list.append({
                "path": path,
                "identifier": identifier,
                "original_size": (h, w),  # store as (H, W) convention
                "bucket_idx": bucket_idx,
                "bucket_res": (bucket_h, bucket_w),
                "original_caption": original_caption,
            })
        return sample_list

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        path = sample["path"]
        bucket_h, bucket_w = sample["bucket_res"]

        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Fallback to a random image (very rare)
            new_idx = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(new_idx)

        # Convert to tensor and apply resize + crop to bucket resolution
        img_tensor = TVF.to_tensor(img)  # (C, H, W) in [0,1]
        img_tensor = _resize_and_crop_image(img_tensor, (bucket_h, bucket_w))
        img_tensor = 2.0 * img_tensor - 1.0  # [-1,1]

        caption = _get_caption_from_cache_or_fallback(
            sample["identifier"],
            self.recaption_cache,
            {sample["identifier"]: sample["original_caption"]},
            fallback="an image"
        )

        return {
            "image": img_tensor,                # (C, H, W) in [-1,1]
            "caption": caption,
            "original_size": sample["original_size"],
        }


# ---------------------------------------------------------------------------
# Patch n' Pack collator
# ---------------------------------------------------------------------------

class PatchPackCollator:
    """
    Collates a list of per‑sample token sequences into a single packed batch
    following the "Patch n' Pack" strategy (Dehghani et al., 2023).

    Each sample is expected to be a dict with at least:
        "input_tokens": tensor of shape (n_i, hidden_dim)
        "text_emb":     tensor of shape (t_i, text_dim)
        "sample_id":    (optional) integer; if not present, assigned automatically.

    The collator concatenates all tokens and text, pads to max_tokens,
    and creates the necessary masks for causal self‑attention and
    cross‑attention.

    Args:
        max_tokens: Maximum number of tokens allowed in one packed batch.
    """

    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens

    def __call__(self, samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Args:
            samples: List of dicts, each containing:
                "input_tokens": (n_i, d)  – patchified latent tokens
                "text_emb":     (t_i, td) – text context embeddings
                "pos_emb":      (optional) (n_i, dd) – positional embeddings
                "sample_id":    (optional) int – sample index, assigned if missing

        Returns:
            A dictionary with the packed batch:
                "packed_tokens":   (max_tokens, d)
                "attention_mask":  (max_tokens,)        True for real tokens
                "sample_ids":      (max_tokens,)        long
                "text_emb":        (total_text, td)
                "cross_attn_mask": (max_tokens, total_text)  additive mask (0 for allowed, -inf for disallowed)
                "packed_pos_emb":  (optional) (max_tokens, dd) – positional embeddings if present
                "num_samples":     int
        """
        device = samples[0]["input_tokens"].device
        hidden_dim = samples[0]["input_tokens"].shape[1]
        text_dim = samples[0]["text_emb"].shape[1]

        # Enforce maximum token count by dropping excess samples
        total_tokens = 0
        kept = []
        for i, s in enumerate(samples):
            n = s["input_tokens"].shape[0]
            if total_tokens + n <= self.max_tokens:
                kept.append(s)
                total_tokens += n
            else:
                break  # discard the rest; ideally we ensure upstream batch size avoids this

        if not kept:
            raise ValueError("No sample fits into the allocated max_tokens. Increase max_tokens or decrease batch size.")

        # Concatenate tokens
        all_tokens = torch.cat([s["input_tokens"] for s in kept], dim=0)  # (N, d)
        N = all_tokens.shape[0]

        # Padding mask and sample ids
        attention_mask = torch.zeros(self.max_tokens, dtype=torch.bool, device=device)
        attention_mask[:N] = True
        sample_ids = torch.zeros(self.max_tokens, dtype=torch.long, device=device)
        cur = 0
        for i, s in enumerate(kept):
            n = s["input_tokens"].shape[0]
            sample_ids[cur:cur + n] = i
            cur += n

        # Text embeddings
        all_text = torch.cat([s["text_emb"] for s in kept], dim=0)  # (Tt, td)
        total_text_len = all_text.shape[0]

        # Build cross-attention mask:
        # For each token position i in packed sequence and each text position j,
        # set mask[i,j] = 0.0 if they belong to the same sample, else -inf.
        cross_mask = torch.full((self.max_tokens, total_text_len), -float("inf"), device=device)
        cur_t = 0
        cur_token = 0
        for i, s in enumerate(kept):
            n = s["input_tokens"].shape[0]
            t = s["text_emb"].shape[0]
            cross_mask[cur_token:cur_token + n, cur_t:cur_t + t] = 0.0
            cur_token += n
            cur_t += t

        # Pack tokens with zero padding
        packed_tokens = torch.zeros(self.max_tokens, hidden_dim, dtype=all_tokens.dtype, device=device)
        packed_tokens[:N] = all_tokens

        result = {
            "packed_tokens": packed_tokens,
            "attention_mask": attention_mask,
            "sample_ids": sample_ids,
            "text_emb": all_text,
            "cross_attn_mask": cross_mask,
            "num_samples": len(kept),
        }

        # Optional positional embeddings
        if "pos_emb" in samples[0]:
            pos_dim = samples[0]["pos_emb"].shape[1]
            packed_pos = torch.zeros(self.max_tokens, pos_dim, dtype=all_tokens.dtype, device=device)
            cur = 0
            for s in kept:
                n = s["pos_emb"].shape[0]
                packed_pos[cur:cur + n] = s["pos_emb"]
                cur += n
            result["packed_pos_emb"] = packed_pos

        return result

