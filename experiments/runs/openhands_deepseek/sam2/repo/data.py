"""Dataset loading and preprocessing for SAM 2 training.

Supports:
- SA-1B image dataset for pre-training and joint training
- SA-V video dataset with masklet annotations
- Internal video dataset
- VOS datasets (DAVIS, MOSE, YouTubeVOS)

Training data mixture (paper Sec 5.2):
- With open-source: ~15.5% SA-1B, ~49.5% SA-V, ~15.1% Internal, 
  ~1.3% DAVIS, ~9.4% MOSE, ~9.2% YouTubeVOS
- Without open-source: ~15.2% SA-1B, ~70% SA-V, ~14.8% Internal

Data augmentations (paper Table 12):
- Random horizontal flips
- Random affine transforms
- Random color jittering
- Random grayscale
- Mosaic transform (10% probability): tile video into 2x2 grid
- Reverse temporal order (50% probability)

Each training sample:
- 8-frame sequences (16 for fine-tuning)
- Up to 2 prompted frames
- Up to 3 masklets per sequence
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class SAM2VideoAugmentation:
    """Video-specific augmentations for SAM 2 training."""

    def __init__(self, image_size: int = 1024, use_flip: bool = True,
                 use_affine: bool = True, use_color_jitter: bool = True,
                 use_grayscale: bool = True, mosaic_prob: float = 0.1,
                 reverse_time_prob: float = 0.5):
        self.image_size = image_size
        self.use_flip = use_flip
        self.use_affine = use_affine
        self.use_color_jitter = use_color_jitter
        self.use_grayscale = use_grayscale
        self.mosaic_prob = mosaic_prob
        self.reverse_time_prob = reverse_time_prob

        self.color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        self.grayscale = T.RandomGrayscale(p=0.1)

    def _apply_to_frames(self, frames: torch.Tensor, masks: torch.Tensor, transform_fn) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply a spatial transform consistently to all frames and masks."""
        B, C, H, W = frames.shape
        frames_out = []
        masks_out = []
        for i in range(B):
            f, m = transform_fn(frames[i], masks[i])
            frames_out.append(f)
            masks_out.append(m)
        return torch.stack(frames_out), torch.stack(masks_out)

    def _random_horizontal_flip(self, frames: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Random horizontal flip applied consistently to all frames."""
        if random.random() < 0.5:
            frames = torch.flip(frames, dims=[-1])
            masks = torch.flip(masks, dims=[-1])
        return frames, masks

    def _random_affine(self, frames: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Random affine transform (rotation, translation, scaling)."""
        angle = random.uniform(-10, 10)
        translate = (random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1))
        scale = random.uniform(0.9, 1.1)

        T_frames, C_f, H_f, W_f = frames.shape
        frames_out = []
        masks_out = []
        for i in range(T_frames):
            f = TF.affine(frames[i], angle=angle, translate=translate, scale=scale, shear=0, fill=0)
            m = TF.affine(masks[i:i+1], angle=angle, translate=translate, scale=scale, shear=0, fill=0)
            frames_out.append(f)
            masks_out.append(m)
        return torch.stack(frames_out), torch.cat(masks_out)

    def _mosaic_transform(self, frames: torch.Tensor, masks: torch.Tensor,
                          target_masklet_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mosaic augmentation: tile video into 2x2 grid.

        Places the same video in all 4 quadrants. Model must use motion
        cues to distinguish the target object from identical copies.
        Also helps learn to segment small objects (half size).
        """
        T_f, C_f, H_f, W_f = frames.shape
        H_half, W_half = H_f // 2, W_f // 2

        # Target quadrant where the target masklet is
        quad_map = [(0, 0), (0, 1), (1, 0), (1, 1)]
        target_quad = random.randint(0, 3)

        mosaic_frames = torch.zeros(T_f, C_f, H_f, W_f, device=frames.device, dtype=frames.dtype)
        mosaic_masks = torch.zeros(T_f, 1, H_f, W_f, device=masks.device, dtype=masks.dtype)

        for t in range(T_f):
            # Resize to half
            frame_half = F.interpolate(frames[t:t+1], size=(H_half, W_half), mode="bilinear", align_corners=False)
            mask_half = F.interpolate(masks[t:t+1].float(), size=(H_half, W_half), mode="nearest")

            for q_idx, (qy, qx) in enumerate(quad_map):
                y1, y2 = qy * H_half, (qy + 1) * H_half
                x1, x2 = qx * W_half, (qx + 1) * W_half
                mosaic_frames[t, :, y1:y2, x1:x2] = frame_half
                if q_idx == target_quad:
                    mosaic_masks[t, :, y1:y2, x1:x2] = mask_half

        return mosaic_frames, mosaic_masks

    def __call__(self, frames: torch.Tensor, masks: torch.Tensor,
                 target_masklet_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentations to video frames and masks.

        Args:
            frames: [T, 3, H, W] video frames
            masks: [T, 1, H, W] binary masklets
            target_masklet_idx: which masklet is the training target

        Returns:
            augmented_frames: [T, 3, H_img, W_img]
            augmented_masks: [T, 1, H_img, W_img]
        """
        # Resize to square
        frames = F.interpolate(frames, size=(self.image_size, self.image_size),
                               mode="bilinear", align_corners=False)
        masks = F.interpolate(masks.float(), size=(self.image_size, self.image_size),
                             mode="nearest")

        # Reverse temporal order (50% probability)
        if random.random() < self.reverse_time_prob:
            frames = torch.flip(frames, dims=[0])
            masks = torch.flip(masks, dims=[0])

        # Mosaic (10% probability)
        if random.random() < self.mosaic_prob:
            frames, masks = self._mosaic_transform(frames, masks, target_masklet_idx)

        # Horizontal flip
        if self.use_flip:
            frames, masks = self._random_horizontal_flip(frames, masks)

        # Affine
        if self.use_affine and random.random() < 0.5:
            frames, masks = self._random_affine(frames, masks)

        # Color jitter (per-frame)
        if self.use_color_jitter:
            frames_out = []
            for i in range(frames.shape[0]):
                frames_out.append(self.color_jitter(frames[i]))
            frames = torch.stack(frames_out)

        # Grayscale
        if self.use_grayscale:
            frames_out = []
            for i in range(frames.shape[0]):
                frames_out.append(self.grayscale(frames[i]))
            frames = torch.stack(frames_out)

        return frames, masks


class SA1BDataset(Dataset):
    """SA-1B image dataset for pre-training and joint training.

    Filters: masks covering >90% of image are removed.
    Maximum 64 masks per image (randomly sampled).
    """
    def __init__(self, data_root: str, split: str = "train",
                 image_size: int = 1024, max_masks_per_image: int = 64,
                 max_mask_area_ratio: float = 0.9, transform=None):
        super().__init__()
        self.data_root = data_root
        self.image_size = image_size
        self.max_masks_per_image = max_masks_per_image
        self.max_mask_area_ratio = max_mask_area_ratio
        self.transform = transform

        # Load metadata (placeholder - real implementation loads from disk)
        self.metadata = self._load_metadata(split)

    def _load_metadata(self, split: str) -> List[Dict]:
        """Load SA-1B metadata. Placeholder for actual implementation."""
        return []

    def _load_image(self, idx: int) -> torch.Tensor:
        """Load a single image. Placeholder."""
        return torch.zeros(3, self.image_size, self.image_size)

    def _load_masks(self, idx: int) -> torch.Tensor:
        """Load masks for an image, filtered by area."""
        return torch.zeros(1, self.image_size, self.image_size)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        image = self._load_image(idx)
        masks = self._load_masks(idx)

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,       # [3, H, W]
            "masks": masks,       # [N_masks, H, W]
            "type": "image",
        }


class VideoSegmentationDataset(Dataset):
    """Video segmentation dataset for SA-V, Internal, and VOS datasets.

    Each item contains a sequence of frames with corresponding masklets.
    During training, we sample 8-frame sequences with up to 3 masklets.
    """
    def __init__(self, data_root: str, split: str = "train",
                 num_frames: int = 8, image_size: int = 1024,
                 max_masklets: int = 3, augmentation: Optional[SAM2VideoAugmentation] = None):
        super().__init__()
        self.data_root = data_root
        self.num_frames = num_frames
        self.image_size = image_size
        self.max_masklets = max_masklets
        self.augmentation = augmentation

        # Load video metadata
        self.metadata = self._load_metadata(split)

    def _load_metadata(self, split: str) -> List[Dict]:
        """Load video segmentation metadata. Placeholder."""
        return []

    def _load_video_frames(self, video_id: str, start_frame: int, num_frames: int) -> torch.Tensor:
        """Load video frames. Placeholder."""
        return torch.zeros(num_frames, 3, self.image_size, self.image_size)

    def _load_masklets(self, video_id: str, masklet_ids: List[str],
                       start_frame: int, num_frames: int) -> torch.Tensor:
        """Load masklets. Placeholder.

        Returns: [num_masklets, num_frames, H, W]
        """
        return torch.zeros(len(masklet_ids), num_frames, self.image_size, self.image_size)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        meta = self.metadata[idx]
        video_id = meta["video_id"]
        num_total_frames = meta["num_frames"]
        masklet_ids = meta["masklet_ids"]

        # Sample a random starting frame
        max_start = max(0, num_total_frames - self.num_frames)
        start_frame = random.randint(0, max_start)

        # Load frames
        frames = self._load_video_frames(video_id, start_frame, self.num_frames)

        # Select up to max_masklets masklets
        selected_masklets = random.sample(masklet_ids, min(len(masklet_ids), self.max_masklets))

        # Load masklets
        masklets = self._load_masklets(video_id, selected_masklets, start_frame, self.num_frames)

        # Apply augmentations
        if self.augmentation is not None:
            target_masklet_idx = random.randint(0, len(selected_masklets) - 1)
            target_masks = masklets[target_masklet_idx:target_masklet_idx + 1, :, :, :]
            target_masks = target_masks.squeeze(0)  # [num_frames, H, W]
            frames, target_masks = self.augmentation(frames, target_masks.unsqueeze(1), target_masklet_idx)

        return {
            "frames": frames,                    # [T, 3, H, W]
            "masklets": target_masks.squeeze(1),  # [T, H, W] target masklet
            "all_masklets": masklets,             # [M, T, H, W]
            "video_id": video_id,
            "start_frame": start_frame,
            "type": "video",
        }


class MixedBatchSampler:
    """Sampler that alternates between image and video datasets during training.

    In each training iteration, we sample a full batch either from the image or
    video dataset, with probabilities proportional to the size of each data source.
    This approach allows for balanced exposure to both tasks and different batch
    sizes for each data source to maximize compute utilization.
    """
    def __init__(self, image_dataset: Dataset, video_dataset: Dataset,
                 image_batch_size: int = 256, video_batch_size: int = 128,
                 image_prob: float = 0.152, video_prob: float = 0.848):
        self.image_dataset = image_dataset
        self.video_dataset = video_dataset
        self.image_batch_size = image_batch_size
        self.video_batch_size = video_batch_size
        self.image_prob = image_prob
        self.video_prob = video_prob

        self.image_indices = list(range(len(image_dataset)))
        self.video_indices = list(range(len(video_dataset)))

        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        random.seed(epoch)
        random.shuffle(self.image_indices)
        random.shuffle(self.video_indices)

    def __iter__(self):
        ip = 0
        vp = 0
        while ip < len(self.image_indices) and vp < len(self.video_indices):
            if random.random() < self.image_prob:
                batch_indices = self.image_indices[ip:ip + self.image_batch_size]
                ip += self.image_batch_size
                yield [{"dataset": "image", "indices": batch_indices}]
            else:
                batch_indices = self.video_indices[vp:vp + self.video_batch_size]
                vp += self.video_batch_size
                yield [{"dataset": "video", "indices": batch_indices}]

    def __len__(self):
        n_image_batches = len(self.image_indices) // self.image_batch_size
        n_video_batches = len(self.video_indices) // self.video_batch_size
        return n_image_batches + n_video_batches


def build_dataloaders(config: dict) -> Tuple[DataLoader, dict]:
    """Build training and validation dataloaders.

    Args:
        config: training configuration

    Returns:
        train_loader, data_info dict
    """
    augmentation = SAM2VideoAugmentation(
        image_size=config["image_size"],
        use_flip=config.get("use_horizontal_flip", True),
        use_affine=config.get("use_affine_transform", True),
        use_color_jitter=config.get("use_color_jitter", True),
        use_grayscale=config.get("use_grayscale", True),
        mosaic_prob=config.get("mosaic_prob", 0.1),
        reverse_time_prob=config.get("reverse_time_prob", 0.5),
    )

    sa1b = SA1BDataset(
        data_root=config["sa1b_root"],
        image_size=config["image_size"],
        max_masks_per_image=config.get("max_masks_per_image", 64),
        max_mask_area_ratio=config.get("max_mask_area_ratio", 0.9),
    )

    sav = VideoSegmentationDataset(
        data_root=config["sav_root"],
        num_frames=config.get("num_frames", 8),
        image_size=config["image_size"],
        augmentation=augmentation,
    )

    video_datasets = [sav]

    if "davis_root" in config:
        davis = VideoSegmentationDataset(
            data_root=config["davis_root"],
            num_frames=config.get("num_frames", 8),
            image_size=config["image_size"],
            augmentation=augmentation,
        )
        video_datasets.append(davis)

    if "mose_root" in config:
        mose = VideoSegmentationDataset(
            data_root=config["mose_root"],
            num_frames=config.get("num_frames", 8),
            image_size=config["image_size"],
            augmentation=augmentation,
        )
        video_datasets.append(mose)

    if "ytvos_root" in config:
        ytvos = VideoSegmentationDataset(
            data_root=config["ytvos_root"],
            num_frames=config.get("num_frames", 8),
            image_size=config["image_size"],
            augmentation=augmentation,
        )
        video_datasets.append(ytvos)

    combined_video = ConcatDataset(video_datasets) if len(video_datasets) > 1 else video_datasets[0]

    sampler = MixedBatchSampler(
        image_dataset=sa1b,
        video_dataset=combined_video,
        image_batch_size=config.get("image_batch_size", 256),
        video_batch_size=config.get("video_batch_size", 128),
        image_prob=config.get("image_prob", 0.152),
        video_prob=1 - config.get("image_prob", 0.152),
    )

    train_loader = DataLoader(
        dataset={"image": sa1b, "video": combined_video},
        batch_sampler=sampler,
        num_workers=config.get("num_workers", 8),
        pin_memory=True,
    )

    data_info = {
        "num_images": len(sa1b),
        "num_videos": len(combined_video),
        "image_batch_size": sampler.image_batch_size,
        "video_batch_size": sampler.video_batch_size,
    }

    return train_loader, data_info
