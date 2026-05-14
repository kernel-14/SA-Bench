from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ---------------------------------------------------------------------------
# Video Loading Utilities
# ---------------------------------------------------------------------------

def load_video_frames(
    video_path: str,
    num_frames: int,
    resolution: Tuple[int, int] = (256, 256),
    start_frame: int = 0,
    frame_stride: int = 1,
) -> Optional[torch.Tensor]:
    """
    Load frames from a video file using decord.

    Args:
        video_path: path to video file
        num_frames: number of frames to load
        resolution: (H, W) target resolution
        start_frame: starting frame index
        frame_stride: stride between frames
    Returns:
        frames: (L, C, H, W) float tensor in [-1, 1], or None on failure
    """
    try:
        import decord
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(video_path, width=resolution[1], height=resolution[0])
        total_frames = len(vr)
        indices = list(range(start_frame, min(start_frame + num_frames * frame_stride, total_frames), frame_stride))
        if len(indices) < num_frames:
            return None
        indices = indices[:num_frames]
        frames = vr.get_batch(indices)  # (L, H, W, C) uint8
        frames = frames.permute(0, 3, 1, 2).float() / 127.5 - 1.0  # (L, C, H, W) in [-1, 1]
        return frames
    except Exception:
        return None


def load_video_frames_cv2(
    video_path: str,
    num_frames: int,
    resolution: Tuple[int, int] = (256, 256),
    start_frame: int = 0,
    frame_stride: int = 1,
) -> Optional[torch.Tensor]:
    """Fallback video loader using OpenCV."""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_count = 0
        while len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_stride == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (resolution[1], resolution[0]))
                frames.append(frame)
            frame_count += 1
        cap.release()
        if len(frames) < num_frames:
            return None
        frames = np.stack(frames[:num_frames], axis=0)  # (L, H, W, C)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 127.5 - 1.0
        return frames
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Base Video Dataset
# ---------------------------------------------------------------------------

class VideoDataset(Dataset):
    """Base class for video datasets."""

    def __init__(
        self,
        resolution: Tuple[int, int] = (256, 256),
        num_frames: int = 65,
        frame_stride: int = 1,
    ):
        self.resolution = resolution
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.samples: List[Dict] = []

    def _load_frames(self, video_path: str, start_frame: int = 0) -> Optional[torch.Tensor]:
        frames = load_video_frames(video_path, self.num_frames, self.resolution, start_frame, self.frame_stride)
        if frames is None:
            frames = load_video_frames_cv2(video_path, self.num_frames, self.resolution, start_frame, self.frame_stride)
        return frames

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# InternVid Dataset (Text-to-Video Training)
# ---------------------------------------------------------------------------

class InternVidDataset(VideoDataset):
    """
    InternVid dataset for text-to-video training.
    Filtered to 4.9M high-quality video-text pairs at 256x256.

    Expected directory structure:
        data_root/
            videos/
                video_id.mp4
                ...
            annotations.json  # list of {"video_id": ..., "caption": ...}
    """

    def __init__(
        self,
        data_root: str,
        resolution: Tuple[int, int] = (256, 256),
        num_frames: int = 65,
        frame_stride: int = 1,
        split: str = "train",
    ):
        super().__init__(resolution, num_frames, frame_stride)
        self.data_root = Path(data_root)
        self.video_dir = self.data_root / "videos"

        ann_file = self.data_root / f"annotations_{split}.json"
        if ann_file.exists():
            with open(ann_file) as f:
                self.samples = json.load(f)
        else:
            # Fallback: scan video directory
            self.samples = [
                {"video_id": p.stem, "caption": ""}
                for p in sorted(self.video_dir.glob("*.mp4"))
            ]

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        video_path = str(self.video_dir / f"{sample['video_id']}.mp4")

        # Random start frame for data augmentation
        start_frame = random.randint(0, max(0, 30 - self.num_frames))
        frames = self._load_frames(video_path, start_frame)

        if frames is None:
            # Return a random valid sample on failure
            return self.__getitem__(random.randint(0, len(self) - 1))

        return {
            "frames": frames,          # (L, C, H, W) in [-1, 1]
            "caption": sample.get("caption", ""),
            "video_id": sample["video_id"],
        }


# ---------------------------------------------------------------------------
# SkyTimelapse Dataset (Video Prediction)
# ---------------------------------------------------------------------------

class SkyTimelapseDataset(VideoDataset):
    """
    SkyTimelapse dataset for video prediction (without text input).

    Training set: 997 long timelapse videos -> 2392 short clips
    Test set: 111 long timelapse videos -> 225 short clips

    Expected structure:
        data_root/
            train/
                video_0001/
                    frame_0001.jpg
                    ...
            test/
                video_0001/
                    ...
    """

    def __init__(
        self,
        data_root: str,
        resolution: Tuple[int, int] = (256, 256),
        num_frames: int = 33,
        frame_stride: int = 1,
        split: str = "train",
    ):
        super().__init__(resolution, num_frames, frame_stride)
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        split_dir = self.data_root / split
        self.samples = []
        if split_dir.exists():
            for video_dir in sorted(split_dir.iterdir()):
                if video_dir.is_dir():
                    frames = sorted(video_dir.glob("*.jpg")) + sorted(video_dir.glob("*.png"))
                    if len(frames) >= num_frames:
                        # Create sliding window clips
                        for start in range(0, len(frames) - num_frames + 1, num_frames // 2):
                            self.samples.append({
                                "video_dir": str(video_dir),
                                "frame_paths": [str(f) for f in frames[start:start + num_frames]],
                            })

    def _load_frame_sequence(self, frame_paths: List[str]) -> torch.Tensor:
        from PIL import Image
        frames = []
        for path in frame_paths:
            img = Image.open(path).convert("RGB")
            frames.append(self.transform(img))
        return torch.stack(frames, dim=0)  # (L, C, H, W)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        frames = self._load_frame_sequence(sample["frame_paths"])
        return {
            "frames": frames,
            "caption": "",
            "video_id": Path(sample["video_dir"]).name,
        }


# ---------------------------------------------------------------------------
# UCF-101 Dataset (Evaluation)
# ---------------------------------------------------------------------------

class UCF101Dataset(VideoDataset):
    """
    UCF-101 dataset for evaluation.
    Uses descriptive text prompts from PYoCo (Ge et al., 2023).
    Generates 2048 samples with uniform distribution per category.

    Expected structure:
        data_root/
            videos/
                ApplyEyeMakeup/
                    v_ApplyEyeMakeup_g01_c01.avi
                    ...
            classnames.txt
            pyoco_prompts.json  # {classname: [prompt1, prompt2, ...]}
    """

    UCF101_CLASSES = [
        "ApplyEyeMakeup", "ApplyLipstick", "Archery", "BabyCrawling", "BalanceBeam",
        "BandMarching", "BaseballPitch", "Basketball", "BasketballDunk", "BenchPress",
        "Biking", "Billiards", "BlowDryHair", "BlowingCandles", "BodyWeightSquats",
        "Bowling", "BoxingPunchingBag", "BoxingSpeedBag", "BreastStroke", "BrushingTeeth",
        "CleanAndJerk", "CliffDiving", "CricketBowling", "CricketShot", "CuttingInKitchen",
        "Diving", "Drumming", "Fencing", "FieldHockeyPenalty", "FloorGymnastics",
        "FrisbeeCatch", "FrontCrawl", "GolfSwing", "Haircut", "HammerThrow",
        "Hammering", "HandstandPushups", "HandstandWalking", "HeadMassage", "HighJump",
        "HorseRace", "HorseRiding", "HulaHoop", "IceDancing", "JavelinThrow",
        "JugglingBalls", "JumpRope", "JumpingJack", "Kayaking", "Knitting",
        "LongJump", "Lunges", "MilitaryParade", "Mixing", "MoppingFloor",
        "Nunchucks", "ParallelBars", "PizzaTossing", "PlayingCello", "PlayingDaf",
        "PlayingDhol", "PlayingFlute", "PlayingGuitar", "PlayingPiano", "PlayingSitar",
        "PlayingTabla", "PlayingViolin", "PoleVault", "PommelHorse", "PullUps",
        "Punch", "PushUps", "Rafting", "RockClimbingIndoor", "RopeClimbing",
        "Rowing", "SalsaSpin", "ShavingBeard", "Shotput", "SkateBoarding",
        "Skiing", "Skijet", "SkyDiving", "SoccerJuggling", "SoccerPenalty",
        "StillRings", "SumoWrestling", "Surfing", "Swing", "TableTennisShot",
        "TaiChi", "TennisSwing", "ThrowDiscus", "TrampolineJumping", "Typing",
        "UnevenBars", "VolleyballSpiking", "WalkingWithDog", "WallPushups",
        "WritingOnBoard", "YoYo",
    ]

    def __init__(
        self,
        data_root: str,
        resolution: Tuple[int, int] = (256, 256),
        num_frames: int = 16,
        split: str = "test",
        num_samples_per_class: int = 20,
    ):
        super().__init__(resolution, num_frames)
        self.data_root = Path(data_root)
        self.split = split

        # Load prompts
        prompt_file = self.data_root / "pyoco_prompts.json"
        if prompt_file.exists():
            with open(prompt_file) as f:
                self.class_prompts = json.load(f)
        else:
            self.class_prompts = {cls: [f"A video of {cls.lower()}"] for cls in self.UCF101_CLASSES}

        # Build sample list
        self.samples = []
        video_dir = self.data_root / "videos"
        if video_dir.exists():
            for cls_name in self.UCF101_CLASSES:
                cls_dir = video_dir / cls_name
                if cls_dir.exists():
                    videos = sorted(cls_dir.glob("*.avi")) + sorted(cls_dir.glob("*.mp4"))
                    prompts = self.class_prompts.get(cls_name, [f"A video of {cls_name.lower()}"])
                    for i, video in enumerate(videos[:num_samples_per_class]):
                        self.samples.append({
                            "video_path": str(video),
                            "class_name": cls_name,
                            "caption": prompts[i % len(prompts)],
                        })

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        frames = self._load_frames(sample["video_path"])
        if frames is None:
            return self.__getitem__(random.randint(0, len(self) - 1))
        return {
            "frames": frames,
            "caption": sample["caption"],
            "class_name": sample["class_name"],
        }


# ---------------------------------------------------------------------------
# MSR-VTT Dataset (Evaluation)
# ---------------------------------------------------------------------------

class MSRVTTDataset(VideoDataset):
    """
    MSR-VTT dataset for zero-shot T2V evaluation.
    Uses official test split (2990 videos, 20 captions each).
    Randomly selects one caption per video.

    Expected structure:
        data_root/
            videos/
                video0.mp4
                ...
            test_captions.json  # {video_id: [caption1, ...]}
    """

    def __init__(
        self,
        data_root: str,
        resolution: Tuple[int, int] = (256, 256),
        num_frames: int = 16,
        split: str = "test",
    ):
        super().__init__(resolution, num_frames)
        self.data_root = Path(data_root)

        caption_file = self.data_root / f"{split}_captions.json"
        if caption_file.exists():
            with open(caption_file) as f:
                captions = json.load(f)
            self.samples = [
                {
                    "video_id": vid_id,
                    "video_path": str(self.data_root / "videos" / f"{vid_id}.mp4"),
                    "caption": random.choice(caps) if isinstance(caps, list) else caps,
                }
                for vid_id, caps in captions.items()
            ]
        else:
            video_dir = self.data_root / "videos"
            self.samples = [
                {"video_id": p.stem, "video_path": str(p), "caption": ""}
                for p in sorted(video_dir.glob("*.mp4"))
            ] if video_dir.exists() else []

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        frames = self._load_frames(sample["video_path"])
        if frames is None:
            return self.__getitem__(random.randint(0, len(self) - 1))
        return {
            "frames": frames,
            "caption": sample["caption"],
            "video_id": sample["video_id"],
        }


# ---------------------------------------------------------------------------
# Collate Function
# ---------------------------------------------------------------------------

def collate_fn(batch: List[Dict]) -> Dict:
    """Collate a batch of video samples."""
    frames = torch.stack([b["frames"] for b in batch], dim=0)  # (B, L, C, H, W)
    captions = [b.get("caption", "") for b in batch]
    video_ids = [b.get("video_id", "") for b in batch]
    return {
        "frames": frames,
        "captions": captions,
        "video_ids": video_ids,
    }


# ---------------------------------------------------------------------------
# Dataset Factory
# ---------------------------------------------------------------------------

def build_dataset(dataset_name: str, config, split: str = "train") -> VideoDataset:
    """Build dataset from config."""
    if dataset_name == "internvid":
        return InternVidDataset(
            data_root=config.internvid_root,
            resolution=(config.resolution, config.resolution),
            num_frames=config.max_train_frames,
            split=split,
        )
    elif dataset_name == "skytimelapse":
        return SkyTimelapseDataset(
            data_root=config.skytimelapse_root,
            resolution=(config.resolution, config.resolution),
            num_frames=config.max_train_frames,
            split=split,
        )
    elif dataset_name == "ucf101":
        return UCF101Dataset(
            data_root=config.ucf101_root,
            resolution=(config.resolution, config.resolution),
            num_frames=config.chunk_len,
            split=split,
        )
    elif dataset_name == "msrvtt":
        return MSRVTTDataset(
            data_root=config.msrvtt_root,
            resolution=(config.resolution, config.resolution),
            num_frames=config.chunk_len,
            split=split,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def build_dataloader(dataset: VideoDataset, config, shuffle: bool = True) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
