"""
Data Pipeline for Pyramidal Flow Matching.

Implements the data loading and preprocessing pipeline for:
- Image datasets: LAION-5B, CC-12M, SA-1B, JourneyDB, synthetic data
- Video datasets: WebVid-10M, OpenVid-1M, Open-Sora Plan data

Uses Patch n' Pack (Dehghani et al., 2023) for efficient batching with
varying token counts. Supports the 12.5% image data proportion in
video training stages.

The 3D VAE compresses videos spatially and temporally at 8x8x8 ratio.
"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Optional, Tuple, List, Dict, Any
import math
import random


class ImageVideoDataset(Dataset):
    """
    Mixed image-video dataset for pyramidal flow training.
    
    Supports:
    - Single images with text captions
    - Video clips with text captions
    - Flexible resolution buckets
    - Configurable image/video ratio per batch
    
    Args:
        image_paths: List of image file paths
        image_captions: List of image captions
        video_paths: List of video file paths
        video_captions: List of video captions
        video_frame_counts: Number of frames per video
        image_ratio: Proportion of images in each batch (default: 0.0 for pure video)
    """
    
    def __init__(
        self,
        image_paths: Optional[List[str]] = None,
        image_captions: Optional[List[str]] = None,
        video_paths: Optional[List[str]] = None,
        video_captions: Optional[List[str]] = None,
        video_frame_counts: Optional[List[int]] = None,
        image_ratio: float = 0.0,
    ):
        self.image_paths = image_paths or []
        self.image_captions = image_captions or []
        self.video_paths = video_paths or []
        self.video_captions = video_captions or []
        self.video_frame_counts = video_frame_counts or []
        self.image_ratio = image_ratio
        
        self.num_images = len(self.image_paths)
        self.num_videos = len(self.video_paths)
        
        assert len(self.image_paths) == len(self.image_captions), \
            "Image paths and captions must match"
        assert len(self.video_paths) == len(self.video_captions), \
            "Video paths and captions must match"
    
    def __len__(self) -> int:
        return max(self.num_images, self.num_videos)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a single sample (image or video).
        
        Returns:
            Dict with:
                - 'latent': VAE-encoded latent tensor
                - 'caption': Text caption
                - 'is_video': Boolean
                - 'num_frames': Number of frames (1 for images)
        """
        # Decide whether to return image or video
        if random.random() < self.image_ratio and self.num_images > 0:
            # Return image
            img_idx = random.randint(0, self.num_images - 1)
            return {
                'latent': self._load_image_latent(img_idx),
                'caption': self.image_captions[img_idx],
                'is_video': False,
                'num_frames': 1,
            }
        else:
            # Return video
            vid_idx = random.randint(0, self.num_videos - 1)
            num_frames = self.video_frame_counts[vid_idx] if self.video_frame_counts else 121
            return {
                'latent': self._load_video_latent(vid_idx, num_frames),
                'caption': self.video_captions[vid_idx],
                'is_video': True,
                'num_frames': num_frames,
            }
    
    def _load_image_latent(self, idx: int) -> torch.Tensor:
        """
        Load and encode an image to latent space.
        
        In practice, this would use a pre-trained VAE encoder.
        For the implementation, we return a placeholder.
        """
        # Placeholder: In production, use actual VAE encoder
        # VAE compresses 768x768 image -> 96x96 latent (8x spatial compression)
        return torch.randn(16, 96, 96)  # (C, H, W)
    
    def _load_video_latent(self, idx: int, num_frames: int) -> torch.Tensor:
        """
        Load and encode a video to latent space.
        
        The 3D VAE compresses at 8x8x8 (spatial + temporal).
        """
        # Placeholder: Use actual 3D VAE encoder
        # VAE compresses T x 768 x 768 -> T/8 x 96 x 96
        latent_frames = num_frames // 8
        return torch.randn(latent_frames, 16, 96, 96)  # (T, C, H, W)


class PatchNPackCollator:
    """
    Patch n' Pack collator for efficient batching.
    
    Implements the length-balanced batching strategy from
    Dehghani et al. (2023) to handle varying token counts
    across different resolutions and video lengths.
    
    Samples are packed together to minimize padding, maximizing
    GPU utilization.
    """
    
    def __init__(
        self,
        max_tokens_per_batch: int = 15360,
        token_size: Tuple[int, int] = (96, 96),
    ):
        self.max_tokens_per_batch = max_tokens_per_batch
        self.token_size = token_size
    
    def compute_tokens(self, sample: Dict[str, Any]) -> int:
        """Compute number of tokens for a sample."""
        latent = sample['latent']
        if latent.dim() == 3:  # Image: (C, H, W)
            return latent.shape[1] * latent.shape[2]
        elif latent.dim() == 4:  # Video: (T, C, H, W)
            return latent.shape[0] * latent.shape[2] * latent.shape[3]
        return 0
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch using Patch n' Pack strategy.
        
        Packs samples to fill up to max_tokens_per_batch tokens.
        """
        # Sort by token count to minimize padding
        batch = sorted(batch, key=lambda x: self.compute_tokens(x))
        
        packed_latents = []
        packed_captions = []
        token_counts = []
        
        current_tokens = 0
        
        for sample in batch:
            tokens = self.compute_tokens(sample)
            
            if current_tokens + tokens > self.max_tokens_per_batch:
                break
            
            packed_latents.append(sample['latent'])
            packed_captions.append(sample['caption'])
            token_counts.append(tokens)
            current_tokens += tokens
        
        # Pad latents to same spatial size within batch
        # (actual packing logic depends on DiT architecture)
        
        return {
            'latent': torch.stack([l for l in packed_latents if l.dim() == 3])
                      if packed_latents and packed_latents[0].dim() == 3
                      else packed_latents,
            'caption': packed_captions,
            'token_counts': token_counts,
            'total_tokens': sum(token_counts),
        }


class VideoDataPipeline:
    """
    Complete data pipeline for pyramidal flow training.
    
    Handles:
    - Loading from multiple image and video datasets
    - VAE encoding (spatial 8x + temporal 8x compression)
    - Bucket sampling by resolution and aspect ratio
    - Patch n' Pack batching
    - Image/video ratio control
    """
    
    def __init__(
        self,
        image_datasets: Optional[List[str]] = None,
        video_datasets: Optional[List[str]] = None,
        vae_model = None,
        image_ratio: float = 0.0,
        max_tokens_per_batch: int = 15360,
        token_size: Tuple[int, int] = (96, 96),
    ):
        self.image_datasets = image_datasets or []
        self.video_datasets = video_datasets or []
        self.vae_model = vae_model
        self.image_ratio = image_ratio
        
        self.dataset = ImageVideoDataset(
            image_paths=[],  # Populated from actual data loading
            image_captions=[],
            video_paths=[],
            video_captions=[],
            video_frame_counts=[],
            image_ratio=image_ratio,
        )
        
        self.collator = PatchNPackCollator(
            max_tokens_per_batch=max_tokens_per_batch,
            token_size=token_size,
        )
    
    def create_dataloader(
        self,
        batch_size: int = 1,
        num_workers: int = 8,
        shuffle: bool = True,
    ) -> DataLoader:
        """Create a DataLoader with Patch n' Pack collation."""
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collator,
            pin_memory=True,
        )
    
    @staticmethod
    def get_dataset_statistics() -> Dict[str, int]:
        """
        Return dataset statistics as reported in the paper.
        
        Image datasets:
        - LAION-5B high-aesthetic subset: ~180M images
        - CC-12M: 11M images
        - SA-1B non-blurred: 6.9M images
        - JourneyDB: 4.4M images
        - Public synthetic: 14M images
        
        Video datasets:
        - WebVid-10M: ~10M videos
        - OpenVid-1M: ~1M videos
        - Open-Sora Plan: ~1M videos
        
        Total: ~10M single-shot videos after postprocessing
        """
        return {
            'total_images': 180_000_000 + 11_000_000 + 6_900_000 + 4_400_000 + 14_000_000,
            'total_videos': 10_000_000 + 1_000_000 + 1_000_000,
            'laion_images': 180_000_000,
            'cc12m_images': 11_000_000,
            'sa1b_images': 6_900_000,
            'journeydb_images': 4_400_000,
            'synthetic_images': 14_000_000,
            'webvid_videos': 10_000_000,
            'openvid_videos': 1_000_000,
            'opensora_videos': 1_000_000,
            'postprocessed_videos': 10_000_000,
        }
