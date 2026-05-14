"""Utility functions for Pyramidal Flow Matching.

Utility functions for:
- Video saving/loading
- Image processing
- Token counting
- Memory estimation
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
import os
import math
from PIL import Image


def calculate_token_count(
    num_frames: int,
    spatial_h: int,
    spatial_w: int,
    patch_size: int = 2,
    num_stages: int = 3,
    history_frames: int = 3,
) -> Tuple[int, int, float]:
    """Calculate token counts and efficiency.

    Returns:
        total_tokens: Total tokens for pyramidal flow.
        full_sequence_tokens: Tokens if using full-sequence diffusion.
        reduction_ratio: Token reduction ratio.
    """
    tokens_per_frame = (spatial_h // patch_size) * (spatial_w // patch_size)

    # Pyramidal tokens: average across stages
    pyramidal_tokens = 0
    for k in range(num_stages):
        factor = 2 ** k
        pyramidal_tokens += tokens_per_frame // (factor ** 2)
    pyramidal_tokens = pyramidal_tokens / num_stages

    # Temporal pyramid history
    history_tokens = 0
    for k in range(history_frames):
        factor = 2 ** (k + 1)
        history_tokens += tokens_per_frame // (factor ** 2)

    total_pyramidal = pyramidal_tokens + history_tokens
    full_sequence = num_frames * tokens_per_frame
    reduction = full_sequence / total_pyramidal if total_pyramidal > 0 else 0

    return int(total_pyramidal), int(full_sequence), reduction


def memory_estimate(
    num_frames: int = 241,
    spatial_h: int = 768,
    spatial_w: int = 768,
    patch_size: int = 2,
    num_stages: int = 3,
) -> dict:
    """Estimate GPU memory requirements for training."""
    tokens_per_frame = (spatial_h // patch_size) * (spatial_w // patch_size)

    pyramidal_tokens = 0
    for k in range(num_stages):
        factor = 2 ** k
        pyramidal_tokens += tokens_per_frame // (factor ** 2)
    pyramidal_tokens /= num_stages

    # Memory estimate per token (bytes)
    bytes_per_token = 4  # float32 for estimation
    hidden_size = 2048  # MM-DiT hidden dim

    activation_memory = pyramidal_tokens * hidden_size * bytes_per_token * 24  # layers

    return {
        "tokens_per_frame": tokens_per_frame,
        "avg_pyramidal_tokens": int(pyramidal_tokens),
        "full_sequence_tokens": num_frames * tokens_per_frame,
        "activation_memory_gb": activation_memory / 1e9,
        "reduction_ratio": (num_frames * tokens_per_frame) / pyramidal_tokens,
    }


class VideoWriter:
    """Video writing utility with compression support."""

    def __init__(self, output_path: str, fps: int = 24, codec: str = "libx264"):
        self.output_path = output_path
        self.fps = fps
        self.codec = codec
        self.frames = []

    def add_frame(self, frame: torch.Tensor):
        """Add a frame to the video.

        Args:
            frame: (C, H, W) tensor in [0, 1] or (H, W, C).
        """
        if frame.dim() == 3 and frame.shape[0] == 3:
            frame = frame.permute(1, 2, 0)
        frame_np = (frame * 255).clamp(0, 255).to(torch.uint8).numpy()
        self.frames.append(frame_np)

    def save(self):
        """Write all frames to video file."""
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        try:
            import imageio
            writer = imageio.get_writer(
                self.output_path,
                fps=self.fps,
                codec=self.codec,
                quality=8,
            )
            for frame in self.frames:
                writer.append_data(frame)
            writer.close()
        except Exception:
            self._save_frames()

    def _save_frames(self):
        """Fallback: save individual frames."""
        frames_dir = self.output_path.replace(".mp4", "_frames")
        os.makedirs(frames_dir, exist_ok=True)
        for i, frame in enumerate(self.frames):
            img = Image.fromarray(frame)
            img.save(os.path.join(frames_dir, f"frame_{i:04d}.png"))

    def __len__(self) -> int:
        return len(self.frames)


def load_video_frames(
    video_path: str,
    target_fps: int = 24,
    max_frames: Optional[int] = None,
) -> torch.Tensor:
    """Load video as (T, C, H, W) tensor."""
    try:
        import decord
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(video_path)
        source_fps = vr.get_avg_fps()
        stride = max(1, int(source_fps / target_fps))

        total = len(vr)
        indices = list(range(0, total, stride))
        if max_frames and len(indices) > max_frames:
            indices = indices[:max_frames]

        frames = vr.get_batch(indices)
        frames = frames.permute(0, 3, 1, 2).float() / 255.0
        return frames
    except ImportError:
        pass

    try:
        import av
        container = av.open(video_path)
        stream = container.streams.video[0]
        source_fps = float(stream.average_rate)
        stride = max(1, int(source_fps / target_fps))

        frames = []
        for i, frame in enumerate(container.decode(stream)):
            if i % stride == 0:
                img = frame.to_ndarray(format="rgb24").astype(np.float32) / 255.0
                frames.append(torch.from_numpy(img).permute(2, 0, 1))
                if max_frames and len(frames) >= max_frames:
                    break
        container.close()
        return torch.stack(frames) if frames else torch.zeros(0)
    except ImportError:
        raise ImportError("Install decord or av for video loading")


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_params(n: int) -> str:
    """Format parameter count."""
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.2f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return str(n)


class ModelEMA:
    """Exponential Moving Average for model weights."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
