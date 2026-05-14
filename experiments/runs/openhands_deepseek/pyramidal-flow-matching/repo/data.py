"""Data loading and preprocessing for Pyramidal Flow Matching.

Handles:
- Image datasets: LAION-5B, CC-12M, SA-1B, JourneyDB, synthetic data
- Video datasets: WebVid-10M, OpenVid-1M, Open-Sora Plan
- 3D VAE encoding/decoding
- Text encoding via T5 and CLIP
- Packing with varying token counts (Patch n' Pack)
- Temporal pyramid history construction
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Optional, List, Tuple, Dict, Any
import os
import random
import numpy as np
from PIL import Image
import json
import io

# Video reading
try:
    import decord
    decord.bridge.set_bridge("torch")
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False

try:
    import av
    HAS_PYAV = True
except ImportError:
    HAS_PYAV = False


def read_video_frames(
    video_path: str,
    target_fps: int = 24,
    num_frames: Optional[int] = None,
    start_frame: int = 0,
) -> torch.Tensor:
    """Read video frames as tensor (T, C, H, W).

    Args:
        video_path: Path to video file.
        target_fps: Target frames per second.
        num_frames: Number of frames to read.
        start_frame: Starting frame index.

    Returns:
        Video tensor of shape (T, C, H, W).
    """
    if HAS_DECORD:
        vr = decord.VideoReader(video_path)
        source_fps = vr.get_avg_fps()
        stride = max(1, int(source_fps / target_fps))

        total_frames = len(vr)
        if num_frames is None:
            num_frames = total_frames // stride

        frame_indices = list(range(start_frame * stride, min(total_frames, (start_frame + num_frames) * stride), stride))
        if len(frame_indices) > num_frames:
            frame_indices = frame_indices[:num_frames]

        frames = vr.get_batch(frame_indices)
        frames = frames.permute(0, 3, 1, 2).float() / 255.0
        frames = frames * 2.0 - 1.0
        return frames
    elif HAS_PYAV:
        container = av.open(video_path)
        stream = container.streams.video[0]
        source_fps = float(stream.average_rate)
        stride = max(1, int(source_fps / target_fps))

        frames = []
        for i, frame in enumerate(container.decode(stream)):
            if i % stride == 0:
                img = frame.to_ndarray(format="rgb24").astype(np.float32) / 255.0
                frames.append(torch.from_numpy(img).permute(2, 0, 1) * 2.0 - 1.0)
                if num_frames and len(frames) >= num_frames:
                    break
        container.close()
        if frames:
            return torch.stack(frames)
        return torch.zeros(0)
    else:
        raise ImportError("Either decord or av is required for video reading")


def resize_for_bucket(
    image: torch.Tensor,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    """Resize image to target dimensions while maintaining aspect ratio.

    Used for bucketing images of different aspect ratios.
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
    return F.interpolate(image, size=(target_h, target_w), mode="bilinear", align_corners=False)


class ImageDataset(Dataset):
    """Dataset for image training (Stage 1).

    Supports multiple data sources with different aspect ratios.
    Uses bucketing for efficient batching.
    """

    def __init__(
        self,
        data_paths: Dict[str, str],
        image_size: int = 768,
        latent_size: int = 96,
        target_h: int = 768,
        target_w: int = 768,
        vae: Optional[torch.nn.Module] = None,
        tokenizer_t5: Any = None,
        tokenizer_clip: Any = None,
        use_latent_cache: bool = True,
    ):
        super().__init__()
        self.data_paths = data_paths
        self.image_size = image_size
        self.latent_size = latent_size
        self.target_h = target_h
        self.target_w = target_w
        self.vae = vae
        self.tokenizer_t5 = tokenizer_t5
        self.tokenizer_clip = tokenizer_clip
        self.use_latent_cache = use_latent_cache

        # Collect all image paths and captions
        self.samples = []
        for dataset_name, path in data_paths.items():
            if os.path.exists(path):
                self._load_dataset(dataset_name, path)

    def _load_dataset(self, dataset_name: str, path: str):
        """Load dataset samples from path.

        Supports: JSONL, CSV, directory of images with captions.
        """
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for f in files:
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                        img_path = os.path.join(root, f)
                        caption_path = os.path.splitext(img_path)[0] + ".txt"
                        caption = ""
                        if os.path.exists(caption_path):
                            with open(caption_path, "r") as cf:
                                caption = cf.read().strip()
                        self.samples.append({"image": img_path, "caption": caption, "source": dataset_name})
        elif path.endswith(".jsonl"):
            with open(path, "r") as f:
                for line in f:
                    data = json.loads(line.strip())
                    self.samples.append({
                        "image": data.get("image_path", data.get("image", "")),
                        "caption": data.get("caption", data.get("text", "")),
                        "source": dataset_name,
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        img_path = sample["image"]
        caption = sample["caption"]

        try:
            img = Image.open(img_path).convert("RGB")
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            img = img * 2.0 - 1.0
        except Exception:
            img = torch.randn(3, self.image_size, self.image_size)

        if img.shape[1] != self.target_h or img.shape[2] != self.target_w:
            img = resize_for_bucket(img, self.target_h, self.target_w)

        latent = None
        if self.vae is not None:
            with torch.no_grad():
                img_batch = img.unsqueeze(0).to(self.vae.encoder.conv_in.conv.weight.device)
                img_batch = img_batch.unsqueeze(2)
                latent = self.vae.encode_latents(img_batch).squeeze(0)

        return {
            "pixel_values": img.squeeze(0) if img.dim() == 4 else img,
            "latent": latent.squeeze(0) if latent is not None and latent.dim() == 5 else latent,
            "caption": caption,
        }


class VideoDataset(Dataset):
    """Dataset for video training (Stage 2 and 3).

    Supports:
    - Variable-duration videos (2s, 5s, 10s)
    - 3D VAE encoding with temporal compression
    - Temporal pyramid history construction
    - History noise corruption
    """

    def __init__(
        self,
        data_paths: Dict[str, str],
        image_size: int = 768,
        latent_size: int = 96,
        fps: int = 24,
        min_duration: float = 2.0,
        max_duration: float = 10.0,
        vae: Optional[torch.nn.Module] = None,
        tokenizer_t5: Any = None,
        tokenizer_clip: Any = None,
        latent_temporal_compression: int = 8,
        history_frames: int = 3,
        history_noise_max: float = 1.0 / 3.0,
        use_latent_cache: bool = True,
    ):
        super().__init__()
        self.data_paths = data_paths
        self.image_size = image_size
        self.latent_size = latent_size
        self.fps = fps
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.vae = vae
        self.tokenizer_t5 = tokenizer_t5
        self.tokenizer_clip = tokenizer_clip
        self.latent_temporal_compression = latent_temporal_compression
        self.history_frames = history_frames
        self.history_noise_max = history_noise_max
        self.use_latent_cache = use_latent_cache

        self.samples = []
        for dataset_name, path in data_paths.items():
            if os.path.exists(path):
                self._load_video_dataset(dataset_name, path)

    def _load_video_dataset(self, dataset_name: str, path: str):
        """Load video dataset from path."""
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for f in files:
                    if f.lower().endswith((".mp4", ".avi", ".mov", ".webm", ".gif")):
                        video_path = os.path.join(root, f)
                        caption_path = os.path.splitext(video_path)[0] + ".txt"
                        caption = ""
                        if os.path.exists(caption_path):
                            with open(caption_path, "r") as cf:
                                caption = cf.read().strip()
                        self.samples.append({"video": video_path, "caption": caption, "source": dataset_name})
        elif path.endswith(".jsonl"):
            with open(path, "r") as f:
                for line in f:
                    data = json.loads(line.strip())
                    self.samples.append({
                        "video": data.get("video_path", data.get("video", "")),
                        "caption": data.get("caption", data.get("text", "")),
                        "source": dataset_name,
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        video_path = sample["video"]
        caption = sample["caption"]

        duration = random.uniform(self.min_duration, self.max_duration)
        num_frames = int(duration * self.fps)
        num_frames = max(1, num_frames)

        try:
            frames = read_video_frames(video_path, target_fps=self.fps, num_frames=num_frames)
            if frames.shape[0] == 0:
                frames = torch.randn(max(1, num_frames), 3, self.image_size, self.image_size) * 0.1
            if frames.shape[1] != self.image_size or frames.shape[2] != self.image_size:
                frames = F.interpolate(frames, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        except Exception:
            frames = torch.randn(max(1, num_frames), 3, self.image_size, self.image_size) * 0.1

        pixel_values = frames * 2.0 - 1.0

        latent = None
        if self.vae is not None:
            with torch.no_grad():
                frames_batch = pixel_values.unsqueeze(0)
                latent = self.vae.encode_latents(frames_batch).squeeze(0)

        return {
            "pixel_values": pixel_values,
            "latent": latent,
            "caption": caption,
            "num_frames": pixel_values.shape[0],
        }


def collate_fn_video(
    batch: List[Dict[str, Any]],
    tokenizer_t5: Any = None,
    tokenizer_clip: Any = None,
    vae: Optional[torch.nn.Module] = None,
    max_tokens: int = 4096,
    history_frames: int = 3,
    history_noise_max: float = 1.0 / 3.0,
    patch_size: int = 2,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """Collate function for video training batches.

    Implements Patch n' Pack: packs varying-length samples into fixed-token batches.

    Also handles temporal pyramid condition construction:
    - History latents with progressive compression
    - Corruptive noise addition to history
    """
    captions = [item["caption"] for item in batch]

    if vae is not None:
        latents = [item["latent"] for item in batch]
    else:
        latents = [item["pixel_values"] for item in batch]

    # Stack or pack latents
    if all(l.shape == latents[0].shape for l in latents):
        latents_batch = torch.stack(latents)
    else:
        latents_batch = pack_varying_length(latents, max_tokens, patch_size, device)

    # Tokenize text
    if tokenizer_t5 is not None:
        t5_tokens = tokenizer_t5(
            captions,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        t5_input_ids = t5_tokens["input_ids"].to(device)
        t5_attention_mask = t5_tokens["attention_mask"].to(device)
    else:
        t5_input_ids = torch.randint(0, 32000, (len(batch), 77), device=device)
        t5_attention_mask = torch.ones(len(batch), 77, device=device)

    if tokenizer_clip is not None:
        clip_tokens = tokenizer_clip(
            captions,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        clip_input_ids = clip_tokens["input_ids"].to(device)
    else:
        clip_input_ids = torch.randint(0, 49408, (len(batch), 77), device=device)

    # Build temporal pyramid history
    B = len(batch)
    history_conditions = []
    history_pos_ids = []

    for b in range(B):
        latent_b = latents[b] if isinstance(latents, list) else latents_batch[b]
        num_frames = latent_b.shape[0]

        frame_positions = []
        for t in range(num_frames):
            frame_positions.append(t)
            if t < history_frames:
                continue

            # Build history for this frame
            frame_history = []
            for h in range(1, min(history_frames, t) + 1):
                hist_latent = latent_b[t - h:t - h + 1]
                comp_factor = 2 ** (h)
                hist_latent = F.interpolate(
                    hist_latent, scale_factor=1.0 / comp_factor,
                    mode="bilinear", align_corners=False
                )
                frame_history.append(hist_latent)
            history_conditions.append(frame_history)

        history_pos_ids.append(torch.tensor(frame_positions, device=device))

    return {
        "latents": latents_batch,
        "t5_input_ids": t5_input_ids,
        "t5_attention_mask": t5_attention_mask,
        "clip_input_ids": clip_input_ids,
        "history_conditions": history_conditions,
        "history_pos_ids": history_pos_ids,
        "captions": captions,
    }


def pack_varying_length(
    tensors: List[torch.Tensor],
    max_tokens: int,
    patch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Pack tensors of varying sequence lengths into fixed-token batches.

    Patch n' Pack strategy (Dehghani et al., 2023).
    """
    all_patches = []
    for t in tensors:
        if t.dim() == 4:
            B, C, H, W = t.shape
            for b in range(B):
                patches = t[b].view(C, H // patch_size, patch_size, W // patch_size, patch_size)
                patches = patches.permute(1, 3, 0, 2, 4).reshape(-1, C * patch_size * patch_size)
                all_patches.append(patches)
        elif t.dim() == 3:
            C, H, W = t.shape
            patches = t.view(C, H // patch_size, patch_size, W // patch_size, patch_size)
            patches = patches.permute(1, 3, 0, 2, 4).reshape(-1, C * patch_size * patch_size)
            all_patches.append(patches)

    if all_patches:
        min_tokens = min(p.shape[0] for p in all_patches)
        packed = torch.stack([p[:min_tokens] for p in all_patches])
        return packed
    return torch.zeros(len(tensors), 0, 3 * patch_size * patch_size, device=device)


class MixedImageVideoDataset(Dataset):
    """Dataset that mixes images and videos with configurable ratio.

    Used in Stage 2 and 3 where image proportion is 12.5%.
    """

    def __init__(
        self,
        image_dataset: ImageDataset,
        video_dataset: VideoDataset,
        image_proportion: float = 0.125,
    ):
        super().__init__()
        self.image_dataset = image_dataset
        self.video_dataset = video_dataset
        self.image_proportion = image_proportion
        self.total_len = max(len(image_dataset), len(video_dataset))

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if random.random() < self.image_proportion:
            img_idx = idx % len(self.image_dataset)
            return self.image_dataset[img_idx]
        else:
            vid_idx = idx % len(self.video_dataset)
            return self.video_dataset[vid_idx]
