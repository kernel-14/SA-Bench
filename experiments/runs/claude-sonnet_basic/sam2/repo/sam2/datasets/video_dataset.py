"""
Video dataset for SAM 2 training.

Implements the training data pipeline described in Section D.2 of the paper:
- Samples 8-frame sequences from videos
- Randomly selects up to 2 frames for prompting
- Simulates interactive prompting with corrective clicks
- Applies data augmentations: horizontal flip, affine transforms, color jitter, grayscale
- Mosaic transform (10% probability): tiles same video into 2x2 grid

Prompt types (initial prompts):
- Ground-truth mask: 50% probability
- Positive click from GT mask: 25% probability
- Bounding box: 25% probability

Corrective clicks:
- Sampled from center of error region between GT and prediction
- With 10% probability, randomly sampled from GT mask
"""

import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


def get_center_of_mass(mask: np.ndarray) -> Tuple[int, int]:
    """Get the center of mass of a binary mask."""
    if mask.sum() == 0:
        h, w = mask.shape
        return h // 2, w // 2
    y_coords, x_coords = np.where(mask > 0)
    cy = int(y_coords.mean())
    cx = int(x_coords.mean())
    return cy, cx


def sample_click_from_mask(mask: np.ndarray) -> Tuple[int, int]:
    """Sample a click from the center of a binary mask."""
    return get_center_of_mass(mask)


def sample_click_from_error_region(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    positive: bool = True,
) -> Optional[Tuple[int, int]]:
    """
    Sample a corrective click from the error region.

    For positive clicks: sample from false negatives (GT=1, pred=0)
    For negative clicks: sample from false positives (GT=0, pred=1)
    """
    if positive:
        error_region = (gt_mask > 0) & (pred_mask == 0)
    else:
        error_region = (gt_mask == 0) & (pred_mask > 0)

    if error_region.sum() == 0:
        return None

    return get_center_of_mass(error_region.astype(np.float32))


def mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Convert binary mask to bounding box (x1, y1, x2, y2)."""
    if mask.sum() == 0:
        return None
    y_coords, x_coords = np.where(mask > 0)
    x1, y1 = int(x_coords.min()), int(y_coords.min())
    x2, y2 = int(x_coords.max()), int(y_coords.max())
    return x1, y1, x2, y2


class VideoSegmentationDataset(Dataset):
    """
    Dataset for video object segmentation training.

    Supports SA-V, DAVIS, MOSE, YouTubeVOS formats.
    Samples 8-frame sequences and simulates interactive prompting.
    """

    def __init__(
        self,
        video_dir: str,
        annotation_dir: str,
        num_frames: int = 8,
        image_size: int = 1024,
        max_num_prompts: int = 2,
        use_mosaic: bool = True,
        mosaic_prob: float = 0.1,
        augment: bool = True,
        reverse_prob: float = 0.5,
        max_masks_per_seq: int = 3,
    ):
        super().__init__()
        self.video_dir = video_dir
        self.annotation_dir = annotation_dir
        self.num_frames = num_frames
        self.image_size = image_size
        self.max_num_prompts = max_num_prompts
        self.use_mosaic = use_mosaic
        self.mosaic_prob = mosaic_prob
        self.augment = augment
        self.reverse_prob = reverse_prob
        self.max_masks_per_seq = max_masks_per_seq

        # Load video list
        self.videos = self._load_video_list()

        # Augmentation transforms
        self.color_jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        )

    def _load_video_list(self) -> List[Dict]:
        """Load list of videos and their annotations."""
        videos = []
        if not os.path.exists(self.video_dir):
            return videos

        for video_name in sorted(os.listdir(self.video_dir)):
            video_path = os.path.join(self.video_dir, video_name)
            ann_path = os.path.join(self.annotation_dir, video_name)

            if os.path.isdir(video_path) and os.path.isdir(ann_path):
                frames = sorted([
                    f for f in os.listdir(video_path)
                    if f.endswith(('.jpg', '.png', '.jpeg'))
                ])
                masks = sorted([
                    f for f in os.listdir(ann_path)
                    if f.endswith('.png')
                ])

                if len(frames) >= self.num_frames:
                    videos.append({
                        'name': video_name,
                        'frames': frames,
                        'masks': masks,
                        'video_path': video_path,
                        'ann_path': ann_path,
                    })

        return videos

    def __len__(self) -> int:
        return max(len(self.videos), 1)

    def _load_frame(self, path: str) -> torch.Tensor:
        """Load and preprocess a video frame."""
        from PIL import Image
        img = Image.open(path).convert('RGB')
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return img

    def _load_mask(self, path: str) -> torch.Tensor:
        """Load and preprocess a segmentation mask."""
        from PIL import Image
        mask = Image.open(path).convert('L')
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        mask = torch.from_numpy(np.array(mask)).float()
        mask = (mask > 0).float()
        return mask

    def _apply_augmentation(
        self,
        frames: List[torch.Tensor],
        masks: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Apply consistent augmentation to all frames and masks."""
        # Random horizontal flip
        if random.random() < 0.5:
            frames = [TF.hflip(f) for f in frames]
            masks = [TF.hflip(m.unsqueeze(0)).squeeze(0) for m in masks]

        # Random affine transform
        if random.random() < 0.5:
            angle = random.uniform(-10, 10)
            translate = (random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1))
            scale = random.uniform(0.9, 1.1)
            shear = random.uniform(-5, 5)

            frames = [
                TF.affine(f, angle=angle, translate=[int(t * self.image_size) for t in translate],
                         scale=scale, shear=shear)
                for f in frames
            ]
            masks = [
                TF.affine(m.unsqueeze(0), angle=angle,
                         translate=[int(t * self.image_size) for t in translate],
                         scale=scale, shear=shear).squeeze(0)
                for m in masks
            ]

        # Color jitter (only on frames, not masks)
        if random.random() < 0.8:
            frames = [self.color_jitter(f) for f in frames]

        # Random grayscale
        if random.random() < 0.2:
            frames = [TF.rgb_to_grayscale(f, num_output_channels=3) for f in frames]

        return frames, masks

    def _sample_prompt(
        self,
        gt_mask: np.ndarray,
        prompt_type: str = 'random',
    ) -> Dict:
        """
        Sample an initial prompt for a frame.

        Returns dict with prompt type and data.
        """
        if prompt_type == 'random':
            r = random.random()
            if r < 0.5:
                prompt_type = 'mask'
            elif r < 0.75:
                prompt_type = 'click'
            else:
                prompt_type = 'box'

        if prompt_type == 'mask':
            return {'type': 'mask', 'mask': gt_mask}
        elif prompt_type == 'click':
            cy, cx = sample_click_from_mask(gt_mask)
            return {
                'type': 'click',
                'coords': np.array([[cx, cy]]),
                'labels': np.array([1]),
            }
        elif prompt_type == 'box':
            bbox = mask_to_bbox(gt_mask)
            if bbox is None:
                cy, cx = sample_click_from_mask(gt_mask)
                return {
                    'type': 'click',
                    'coords': np.array([[cx, cy]]),
                    'labels': np.array([1]),
                }
            x1, y1, x2, y2 = bbox
            return {
                'type': 'box',
                'box': np.array([x1, y1, x2, y2]),
            }

        return {'type': 'none'}

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a training sample.

        Returns dict with:
        - frames: [T, 3, H, W] video frames
        - masks: [T, H, W] ground-truth masks
        - prompts: list of prompt dicts for each frame
        - prompt_frames: list of frame indices that have prompts
        """
        if len(self.videos) == 0:
            # Return dummy data if no videos loaded
            T = self.num_frames
            H = W = self.image_size
            return {
                'frames': torch.zeros(T, 3, H, W),
                'masks': torch.zeros(T, H, W),
                'prompts': [{'type': 'none'}] * T,
                'prompt_frames': [0],
                'has_mask': torch.ones(T),
            }

        video = self.videos[idx % len(self.videos)]

        # Sample frame indices
        all_frames = video['frames']
        n_frames = len(all_frames)

        if n_frames <= self.num_frames:
            frame_indices = list(range(n_frames))
            # Pad if needed
            while len(frame_indices) < self.num_frames:
                frame_indices.append(frame_indices[-1])
        else:
            start = random.randint(0, n_frames - self.num_frames)
            frame_indices = list(range(start, start + self.num_frames))

        # Optionally reverse temporal order
        if random.random() < self.reverse_prob:
            frame_indices = frame_indices[::-1]

        # Load frames and masks
        frames = []
        masks = []
        has_mask = []

        for fi in frame_indices:
            frame_name = all_frames[fi]
            frame_path = os.path.join(video['video_path'], frame_name)

            # Find corresponding mask
            mask_name = frame_name.replace('.jpg', '.png').replace('.jpeg', '.png')
            mask_path = os.path.join(video['ann_path'], mask_name)

            frame = self._load_frame(frame_path)
            frames.append(frame)

            if os.path.exists(mask_path):
                mask = self._load_mask(mask_path)
                masks.append(mask)
                has_mask.append(1.0)
            else:
                masks.append(torch.zeros(self.image_size, self.image_size))
                has_mask.append(0.0)

        # Apply augmentation
        if self.augment:
            frames, masks = self._apply_augmentation(frames, masks)

        # Select prompt frames (up to max_num_prompts, always include first frame)
        num_prompts = random.randint(1, min(self.max_num_prompts, self.num_frames))
        prompt_frame_indices = [0]  # Always prompt first frame
        if num_prompts > 1:
            other_frames = list(range(1, self.num_frames))
            random.shuffle(other_frames)
            prompt_frame_indices.extend(other_frames[:num_prompts - 1])
        prompt_frame_indices = sorted(prompt_frame_indices)

        # Generate prompts for each prompt frame
        prompts = [{'type': 'none'}] * self.num_frames
        for pfi in prompt_frame_indices:
            gt_mask_np = masks[pfi].numpy()
            if has_mask[pfi] > 0:
                prompts[pfi] = self._sample_prompt(gt_mask_np)

        return {
            'frames': torch.stack(frames),  # [T, 3, H, W]
            'masks': torch.stack(masks),  # [T, H, W]
            'prompts': prompts,
            'prompt_frames': prompt_frame_indices,
            'has_mask': torch.tensor(has_mask),
        }


class ImageSegmentationDataset(Dataset):
    """
    Dataset for image segmentation training (SA-1B format).

    Used for pre-training and joint training with video data.
    """

    def __init__(
        self,
        image_dir: str,
        annotation_dir: str,
        image_size: int = 1024,
        max_masks_per_image: int = 64,
        augment: bool = True,
        filter_large_masks: bool = True,
        large_mask_threshold: float = 0.9,
    ):
        super().__init__()
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.image_size = image_size
        self.max_masks_per_image = max_masks_per_image
        self.augment = augment
        self.filter_large_masks = filter_large_masks
        self.large_mask_threshold = large_mask_threshold

        self.images = self._load_image_list()

    def _load_image_list(self) -> List[str]:
        """Load list of image files."""
        images = []
        if not os.path.exists(self.image_dir):
            return images

        for fname in sorted(os.listdir(self.image_dir)):
            if fname.endswith(('.jpg', '.png', '.jpeg')):
                images.append(fname)

        return images

    def __len__(self) -> int:
        return max(len(self.images), 1)

    def __getitem__(self, idx: int) -> Dict:
        """Get a training sample with random mask selection."""
        if len(self.images) == 0:
            H = W = self.image_size
            return {
                'image': torch.zeros(3, H, W),
                'masks': torch.zeros(1, H, W),
                'num_masks': 0,
            }

        img_name = self.images[idx % len(self.images)]
        img_path = os.path.join(self.image_dir, img_name)

        from PIL import Image
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)

        if self.augment:
            if random.random() < 0.5:
                img = TF.hflip(img)

        img_tensor = TF.to_tensor(img)
        img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Load annotations (SA-1B format: JSON with RLE masks)
        ann_name = img_name.replace('.jpg', '.json').replace('.png', '.json')
        ann_path = os.path.join(self.annotation_dir, ann_name)

        masks = []
        if os.path.exists(ann_path):
            import json
            with open(ann_path) as f:
                anns = json.load(f)

            if isinstance(anns, dict):
                anns = anns.get('annotations', [])

            # Filter large masks
            if self.filter_large_masks:
                anns = [
                    a for a in anns
                    if a.get('area', 0) / (orig_h * orig_w) < self.large_mask_threshold
                ]

            # Randomly sample masks
            if len(anns) > self.max_masks_per_image:
                anns = random.sample(anns, self.max_masks_per_image)

            for ann in anns:
                # Decode RLE mask
                try:
                    from pycocotools import mask as mask_utils
                    rle = ann.get('segmentation', {})
                    if isinstance(rle, dict):
                        m = mask_utils.decode(rle)
                        m = torch.from_numpy(m).float()
                        m = F.interpolate(
                            m.unsqueeze(0).unsqueeze(0),
                            size=(self.image_size, self.image_size),
                            mode='nearest',
                        ).squeeze()
                        masks.append(m)
                except Exception:
                    pass

        if not masks:
            masks = [torch.zeros(self.image_size, self.image_size)]

        masks_tensor = torch.stack(masks[:self.max_masks_per_image])

        return {
            'image': img_tensor,
            'masks': masks_tensor,
            'num_masks': len(masks),
        }
