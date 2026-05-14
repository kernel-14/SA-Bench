import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, List, Tuple
import random

try:
    from decord import VideoReader
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False


class VideoDataset(Dataset):
    """Dataset for video diffusion training with partial noising.
    
    Loads video clips of varying lengths and returns:
    - latent: (L, C, H, W) 
    - prefix_len: P randomly sampled from allowed values
    """

    def __init__(
        self,
        video_dir: str,
        vae,
        text_encoder=None,
        resolution: int = 256,
        chunk_length: int = 16,
        max_prefix_len: int = 49,
        max_train_len: int = 65,
        is_t2v: bool = True,
        stage: int = 2,
    ):
        self.video_dir = video_dir
        self.vae = vae
        self.text_encoder = text_encoder
        self.resolution = resolution
        self.chunk_length = chunk_length
        self.max_prefix_len = max_prefix_len
        self.max_train_len = max_train_len
        self.is_t2v = is_t2v
        self.stage = stage

        # Collect video paths
        self.video_paths = self._collect_videos()
        
        # Prefix length options: {1, 1+l, 1+2l, ..., 1+nl}
        n = (max_prefix_len - 1) // chunk_length
        self.prefix_options = [1 + i * chunk_length for i in range(n + 1)]

    def _collect_videos(self) -> List[str]:
        paths = []
        for root, _, files in os.walk(self.video_dir):
            for f in files:
                if f.endswith((".mp4", ".avi", ".mov", ".gif")):
                    paths.append(os.path.join(root, f))
        return paths

    def _read_video(self, path: str) -> torch.Tensor:
        """Read video frames and return as tensor (T, H, W, C) normalized to [0, 1]."""
        if not DECORD_AVAILABLE:
            raise ImportError("decord is required for video loading. Install with: pip install decord")

        vr = VideoReader(path)
        total_frames = len(vr)

        if total_frames < self.max_train_len:
            return None

        # Random start
        start = random.randint(0, total_frames - self.max_train_len)
        frames = vr.get_batch(list(range(start, start + self.max_train_len))).asnumpy()
        frames = torch.from_numpy(frames).float() / 255.0  # (L, H, W, C)

        # Resize to target resolution
        if frames.shape[1] != self.resolution or frames.shape[2] != self.resolution:
            frames = frames.permute(0, 3, 1, 2)  # (L, C, H, W)
            frames = F.interpolate(frames, size=(self.resolution, self.resolution), mode="bilinear")
            frames = frames.permute(0, 2, 3, 1)  # (L, H, W, C)

        # Normalize to [-1, 1]
        frames = frames * 2.0 - 1.0
        return frames  # (L, H, W, C)

    def _encode_video(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode video frames to latent space using VAE."""
        B = frames.shape[0]
        frames = frames.permute(0, 3, 1, 2)  # (L, C, H, W)
        # VAE encode
        with torch.no_grad():
            latent = self.vae.encode(frames).latent_dist.sample()
            latent = latent * self.vae.config.scaling_factor
        return latent  # (L, C_latent, H_lat, W_lat)

    def __len__(self):
        return len(self.video_paths) * 10  # oversample

    def __getitem__(self, idx):
        while True:
            path = random.choice(self.video_paths)
            frames = self._read_video(path)
            if frames is not None:
                break

        # Encode to latent
        latent = self._encode_video(frames)

        # Determine prefix length P
        if self.stage == 1:
            # Stage 1: no clean prefix (causal pretrain on stage1_frames)
            L_total = self.max_train_len
            P = 0  # all frames are noisy
        else:
            # Stage 2: sample P from prefix options
            P = random.choice(self.prefix_options)
            L_total = P + self.chunk_length

        latent = latent[:L_total]  # (L_total, C_latent, H_lat, W_lat)

        # Random cyclic TPE offset
        tpe_offset = random.randint(0, self.max_prefix_len - 1)

        output = {
            "latent": latent,
            "prefix_len": P,
            "total_frames": L_total,
            "tpe_offset": tpe_offset,
        }

        # Text encoding for T2V
        if self.is_t2v and self.text_encoder is not None:
            # Use a caption file or placeholder
            caption = self._load_caption(path)
            with torch.no_grad():
                text_embed = self.text_encoder(caption)
            output["text_embed"] = text_embed

        return output

    def _load_caption(self, video_path: str) -> str:
        """Load caption for video. Placeholder — replace with actual caption loading."""
        return "a video"


class VideoPredictionDataset(Dataset):
    """Dataset for video prediction (no text conditioning).
    
    Used for SkyTimelapse training with:
    - l = 8
    - P_max = 25
    - L_train = 33
    """

    def __init__(
        self,
        video_dir: str,
        vae,
        resolution: int = 256,
        chunk_length: int = 8,
        max_prefix_len: int = 25,
        max_train_len: int = 33,
    ):
        self.video_dir = video_dir
        self.vae = vae
        self.resolution = resolution
        self.chunk_length = chunk_length
        self.max_prefix_len = max_prefix_len
        self.max_train_len = max_train_len

        self.video_paths = self._collect_videos()

        n = (max_prefix_len - 1) // chunk_length
        self.prefix_options = [1 + i * chunk_length for i in range(n + 1)]

    def _collect_videos(self):
        paths = []
        for root, _, files in os.walk(self.video_dir):
            for f in files:
                if f.endswith((".mp4", ".avi", ".mov", ".gif")):
                    paths.append(os.path.join(root, f))
        return paths

    def _read_video(self, path):
        if not DECORD_AVAILABLE:
            raise ImportError("decord required")
        vr = VideoReader(path)
        total = len(vr)
        if total < self.max_train_len:
            return None
        start = random.randint(0, total - self.max_train_len)
        frames = vr.get_batch(list(range(start, start + self.max_train_len))).asnumpy()
        frames = torch.from_numpy(frames).float() / 255.0
        if frames.shape[1] != self.resolution:
            frames = frames.permute(0, 3, 1, 2)
            frames = F.interpolate(frames, size=(self.resolution, self.resolution), mode="bilinear")
            frames = frames.permute(0, 2, 3, 1)
        frames = frames * 2.0 - 1.0
        return frames

    def _encode_video(self, frames):
        frames = frames.permute(0, 3, 1, 2)
        with torch.no_grad():
            latent = self.vae.encode(frames).latent_dist.sample()
            latent = latent * self.vae.config.scaling_factor
        return latent

    def __len__(self):
        return len(self.video_paths) * 10

    def __getitem__(self, idx):
        while True:
            path = random.choice(self.video_paths)
            frames = self._read_video(path)
            if frames is not None:
                break

        latent = self._encode_video(frames)
        P = random.choice(self.prefix_options)
        L_total = P + self.chunk_length
        latent = latent[:L_total]
        tpe_offset = random.randint(0, self.max_prefix_len - 1)

        return {
            "latent": latent,
            "prefix_len": P,
            "total_frames": L_total,
            "tpe_offset": tpe_offset,
        }


class FVDDataset(Dataset):
    """Dataset for FVD evaluation.
    
    Provides ground-truth video clips for computing FVD statistics.
    """

    def __init__(
        self,
        video_dir: str,
        resolution: int = 256,
        clip_length: int = 16,
        num_samples: int = 2048,
    ):
        self.video_dir = video_dir
        self.resolution = resolution
        self.clip_length = clip_length
        self.num_samples = num_samples
        self.video_paths = self._collect_videos()

    def _collect_videos(self):
        paths = []
        for root, _, files in os.walk(self.video_dir):
            for f in files:
                if f.endswith((".mp4", ".avi", ".mov")):
                    paths.append(os.path.join(root, f))
        return paths

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if not DECORD_AVAILABLE:
            raise ImportError("decord required")
        
        path = random.choice(self.video_paths)
        vr = VideoReader(path)
        total = len(vr)
        if total < self.clip_length:
            # Pad by looping
            indices = list(range(total)) * (self.clip_length // total + 1)
            indices = indices[:self.clip_length]
        else:
            start = random.randint(0, total - self.clip_length)
            indices = list(range(start, start + self.clip_length))
        
        frames = vr.get_batch(indices).asnumpy()
        frames = torch.from_numpy(frames).float() / 255.0
        if frames.shape[1] != self.resolution:
            frames = frames.permute(0, 3, 1, 2)
            frames = F.interpolate(frames, size=(self.resolution, self.resolution), mode="bilinear")
            frames = frames.permute(0, 2, 3, 1)
        frames = frames * 2.0 - 1.0
        return frames.permute(0, 3, 1, 2)  # (T, C, H, W)


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
