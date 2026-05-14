"""
Dataset classes for Pyramidal Flow Matching training.

Supports:
- Image datasets (LAION-5B, CC-12M, SA-1B, JourneyDB)
- Video datasets (WebVid-10M, OpenVid-1M, Open-Sora-Plan)
- Mixed image/video batches (12.5% image ratio in video training)
- Variable aspect ratio and resolution handling
"""

import os
import json
import random
import math
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image


class ImageDataset(Dataset):
    """
    Dataset for image training.
    
    Supports variable aspect ratios by bucketing images into
    predefined resolution buckets.
    """
    
    # Common aspect ratio buckets (width x height)
    RESOLUTION_BUCKETS = [
        (256, 256), (256, 384), (384, 256),
        (384, 384), (384, 512), (512, 384),
        (512, 512), (512, 768), (768, 512),
        (768, 768), (768, 1024), (1024, 768),
        (1024, 1024),
    ]
    
    def __init__(
        self,
        data_root: str,
        metadata_file: str,
        resolution: int = 512,
        max_resolution: int = 1024,
        text_max_length: int = 256,
        use_buckets: bool = True,
    ):
        """
        Args:
            data_root: Root directory for image files
            metadata_file: JSON file with image paths and captions
            resolution: Default resolution
            max_resolution: Maximum resolution
            text_max_length: Maximum text length
            use_buckets: Whether to use resolution bucketing
        """
        self.data_root = Path(data_root)
        self.resolution = resolution
        self.max_resolution = max_resolution
        self.text_max_length = text_max_length
        self.use_buckets = use_buckets
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        print(f"Loaded {len(self.metadata)} image samples")
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def get_bucket_resolution(self, width: int, height: int) -> Tuple[int, int]:
        """Find the closest resolution bucket for given dimensions."""
        aspect_ratio = width / height
        
        best_bucket = None
        best_diff = float('inf')
        
        for bucket_w, bucket_h in self.RESOLUTION_BUCKETS:
            if bucket_w > self.max_resolution or bucket_h > self.max_resolution:
                continue
            
            bucket_ratio = bucket_w / bucket_h
            diff = abs(math.log(aspect_ratio / bucket_ratio))
            
            if diff < best_diff:
                best_diff = diff
                best_bucket = (bucket_w, bucket_h)
        
        return best_bucket or (self.resolution, self.resolution)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.metadata[idx]
        
        # Load image
        img_path = self.data_root / item['image_path']
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            # Return a random item if loading fails
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        # Get target resolution
        if self.use_buckets:
            target_w, target_h = self.get_bucket_resolution(img.width, img.height)
        else:
            target_w = target_h = self.resolution
        
        # Resize and crop
        img = self._resize_and_crop(img, target_w, target_h)
        
        # Convert to tensor and normalize to [-1, 1]
        img_tensor = torch.from_numpy(np.array(img)).float() / 127.5 - 1.0
        img_tensor = img_tensor.permute(2, 0, 1)  # (C, H, W)
        
        return {
            'image': img_tensor,
            'caption': item.get('caption', ''),
            'width': target_w,
            'height': target_h,
        }
    
    def _resize_and_crop(self, img: Image.Image, target_w: int, target_h: int) -> Image.Image:
        """Resize and center crop image to target dimensions."""
        # Compute scale to fit target while maintaining aspect ratio
        scale = max(target_w / img.width, target_h / img.height)
        new_w = int(img.width * scale)
        new_h = int(img.height * scale)
        
        img = img.resize((new_w, new_h), Image.LANCZOS)
        
        # Center crop
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        img = img.crop((left, top, left + target_w, top + target_h))
        
        return img


class VideoDataset(Dataset):
    """
    Dataset for video training.
    
    Supports variable duration videos and autoregressive training
    with temporal pyramid history conditions.
    """
    
    def __init__(
        self,
        data_root: str,
        metadata_file: str,
        num_frames: int = 121,  # 5 seconds at 24fps
        fps: int = 24,
        resolution: int = 384,
        text_max_length: int = 256,
        history_frames: int = 2,  # Number of history frames for autoregressive training
    ):
        """
        Args:
            data_root: Root directory for video files
            metadata_file: JSON file with video paths and captions
            num_frames: Number of frames to sample per video
            fps: Target frames per second
            resolution: Target resolution
            text_max_length: Maximum text length
            history_frames: Number of history frames for autoregressive conditioning
        """
        self.data_root = Path(data_root)
        self.num_frames = num_frames
        self.fps = fps
        self.resolution = resolution
        self.text_max_length = text_max_length
        self.history_frames = history_frames
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        print(f"Loaded {len(self.metadata)} video samples")
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.metadata[idx]
        
        video_path = self.data_root / item['video_path']
        
        try:
            frames = self._load_video_frames(video_path)
        except Exception:
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        # Convert to tensor (C, T, H, W) and normalize to [-1, 1]
        frames_tensor = torch.stack([
            torch.from_numpy(np.array(f)).float() / 127.5 - 1.0
            for f in frames
        ])  # (T, H, W, C)
        frames_tensor = frames_tensor.permute(3, 0, 1, 2)  # (C, T, H, W)
        
        # Split into history and current frames
        # For autoregressive training, we need history frames
        if frames_tensor.shape[1] > self.history_frames:
            history = [
                frames_tensor[:, i:i+1]  # Each history frame: (C, 1, H, W)
                for i in range(self.history_frames)
            ]
            current = frames_tensor[:, self.history_frames:]  # (C, T-history, H, W)
        else:
            history = []
            current = frames_tensor
        
        return {
            'video': current,
            'history': history,
            'caption': item.get('caption', ''),
            'num_frames': current.shape[1],
        }
    
    def _load_video_frames(self, video_path: Path) -> List[Image.Image]:
        """Load frames from a video file."""
        try:
            import imageio
            
            reader = imageio.get_reader(str(video_path))
            meta = reader.get_meta_data()
            
            # Compute frame indices to sample
            total_frames = meta.get('nframes', self.num_frames)
            if total_frames == float('inf'):
                total_frames = self.num_frames
            
            # Sample frames uniformly
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            
            frames = []
            for idx in frame_indices:
                try:
                    frame = reader.get_data(idx)
                    img = Image.fromarray(frame).convert('RGB')
                    img = img.resize((self.resolution, self.resolution), Image.LANCZOS)
                    frames.append(img)
                except Exception:
                    # Use last valid frame if this one fails
                    if frames:
                        frames.append(frames[-1])
                    else:
                        frames.append(Image.new('RGB', (self.resolution, self.resolution)))
            
            reader.close()
            return frames
            
        except ImportError:
            # Fallback: return random frames
            return [
                Image.new('RGB', (self.resolution, self.resolution))
                for _ in range(self.num_frames)
            ]


class MixedDataset(Dataset):
    """
    Mixed dataset combining image and video data.
    
    As described in the paper: "The image data from stage 1 is also utilized
    at a proportion of 12.5% in each batch."
    """
    
    def __init__(
        self,
        video_dataset: VideoDataset,
        image_dataset: Optional[ImageDataset] = None,
        image_ratio: float = 0.125,  # 12.5% as per paper
    ):
        """
        Args:
            video_dataset: Video dataset
            image_dataset: Optional image dataset for mixing
            image_ratio: Fraction of image samples in each batch
        """
        self.video_dataset = video_dataset
        self.image_dataset = image_dataset
        self.image_ratio = image_ratio
    
    def __len__(self) -> int:
        return len(self.video_dataset)
    
    def __getitem__(self, idx: int) -> Dict:
        # Randomly decide whether to return image or video
        if self.image_dataset is not None and random.random() < self.image_ratio:
            img_idx = random.randint(0, len(self.image_dataset) - 1)
            item = self.image_dataset[img_idx]
            item['is_video'] = False
            return item
        else:
            item = self.video_dataset[idx]
            item['is_video'] = True
            return item


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """
    Create a DataLoader with optional distributed sampling.
    
    Args:
        dataset: Dataset to load from
        batch_size: Batch size per GPU
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        distributed: Whether to use distributed sampling
        rank: Process rank for distributed training
        world_size: Total number of processes
    
    Returns:
        DataLoader instance
    """
    if distributed:
        from torch.utils.data import DistributedSampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
        )
        shuffle = False  # Sampler handles shuffling
    else:
        sampler = None
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function to handle variable-length sequences.
    
    Handles both image and video batches with different resolutions.
    """
    # Separate images and videos
    images = [item for item in batch if not item.get('is_video', True)]
    videos = [item for item in batch if item.get('is_video', True)]
    
    result = {}
    
    if videos:
        # Stack video tensors (assuming same resolution after preprocessing)
        try:
            result['video'] = torch.stack([v['video'] for v in videos])
            result['captions'] = [v['caption'] for v in videos]
            
            # Handle history frames
            if videos[0].get('history'):
                result['history'] = [v['history'] for v in videos]
        except Exception:
            # If stacking fails (different sizes), just return the first item
            result['video'] = videos[0]['video'].unsqueeze(0)
            result['captions'] = [videos[0]['caption']]
    
    if images:
        try:
            result['image'] = torch.stack([i['image'] for i in images])
            result['image_captions'] = [i['caption'] for i in images]
        except Exception:
            result['image'] = images[0]['image'].unsqueeze(0)
            result['image_captions'] = [images[0]['caption']]
    
    return result
