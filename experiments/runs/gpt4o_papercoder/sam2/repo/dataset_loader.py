"""
dataset_loader.py

Module to handle loading, preprocessing, and augmenting datasets for training, validation, and testing.
The DatasetLoader class is responsible for integrating multiple datasets (e.g., SA-V, DAVIS, YouTube-VOS)
into PyTorch DataLoader objects while applying preprocessing and augmentation as specified in config.yaml.
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from typing import Tuple, Dict, List
import random
import yaml
from utils import normalize_frames, random_crop_video, positional_encoding

class DatasetLoader:
    """
    A class responsible for managing datasets used in SAM 2 for training, validation, and testing.
    This includes loading, preprocessing, augmentation, and DataLoader pipeline construction.
    """

    def __init__(self, config: Dict):
        """
        Initialize DatasetLoader with the parameters specified in the config.

        Args:
            config (dict): Configuration dictionary from config.yaml.
        """
        self.config = config
        self.frame_resolution = self.config['data']['frame_resolution']
        self.max_frames_per_video = self.config['data']['max_frames_per_video']
        self.augmentations_enabled = self.config['data']['augmentations']
        self.num_workers = self.config['data']['num_workers']
        self.batch_size = self.config['training']['batch_size']

        # Ensure reproducibility
        random.seed(self.config['reproducibility']['random_seed'])
        torch.manual_seed(self.config['reproducibility']['random_seed'])

    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Load training, validation, and testing datasets.

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: PyTorch DataLoader objects for training, validation, and testing.
        """
        print("Loading datasets...")
        train_dataset = self._build_dataset(split="train")
        val_dataset = self._build_dataset(split="val")
        test_dataset = self._build_dataset(split="test")

        # Create DataLoaders with specified batch sizes and workers
        train_loader = self._build_dataloader(train_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = self._build_dataloader(val_dataset, self.batch_size, shuffle=False, num_workers=self.num_workers)
        test_loader = self._build_dataloader(test_dataset, self.batch_size, shuffle=False, num_workers=self.num_workers)

        return train_loader, val_loader, test_loader

    def _build_dataset(self, split: str) -> Dataset:
        """
        Build a dataset object for a specified split (train/val/test).

        Args:
            split (str): Dataset split type ("train", "val", or "test").

        Returns:
            Dataset: PyTorch-compatible Dataset object.
        """
        assert split in ["train", "val", "test"], "Invalid dataset split type."
        print(f"Building {split} dataset...")

        # Placeholder for actual dataset handling logic:
        # Example datasets (replace this with concrete logic for SA-V, DAVIS, etc.)
        if split == "train":
            dataset_path = self.config['evaluation']['datasets']['zero_shot'][0]  # Example: SA-V for training
        elif split == "val":
            dataset_path = self.config['evaluation']['datasets']['zero_shot'][1]  # Example: DAVIS for validation
        elif split == "test":
            dataset_path = self.config['evaluation']['datasets']['test_set_split']

        dataset = VideoDataset(
            dataset_path, 
            split, 
            frame_resolution=self.frame_resolution, 
            augmentations=self.augmentations_enabled, 
            max_frames=self.max_frames_per_video
        )

        return dataset

    def _build_dataloader(self, dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
        """
        Build a PyTorch DataLoader from a given dataset.

        Args:
            dataset (Dataset): The PyTorch Dataset object to use.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the dataset.
            num_workers (int): The number of processes to use for data loading.

        Returns:
            DataLoader: PyTorch DataLoader object.
        """
        print(f"Creating DataLoader: Batch size = {batch_size}, Shuffle = {shuffle}, Workers = {num_workers}")
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True  # Optimize loading
        )
        return data_loader


class VideoDataset(Dataset):
    """
    Custom PyTorch Dataset for loading and processing video datasets.
    Supports preprocessing, augmentation, and annotation handling.
    """

    def __init__(self, path: str, split: str, frame_resolution: int, augmentations: Dict, max_frames: int):
        """
        Initialize the VideoDataset object.

        Args:
            path (str): Path to the dataset.
            split (str): Dataset split type ("train", "val", or "test").
            frame_resolution (int): Target resolution for video frames (e.g., 1024x1024).
            augmentations (dict): Augmentation configurations (from config.yaml).
            max_frames (int): Max number of frames per video.
        """
        self.path = path
        self.split = split
        self.frame_resolution = frame_resolution
        self.augmentations_enabled = augmentations
        self.max_frames = max_frames

        # Load video file paths and annotations
        self.metadata = self._load_metadata(path)
        self.transform = self._build_transforms()

    def __len__(self) -> int:
        """
        Return the total number of videos in the dataset.
        """
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict:
        """
        Load and preprocess a video sample.

        Args:
            idx (int): Index of the video sample.

        Returns:
            Dict: Preprocessed video frames and corresponding annotations.
        """
        video_info = self.metadata[idx]
        video_frames, annotations = self._load_video_and_annotations(video_info)

        # Apply transformations and augmentations
        if self.transform:
            video_frames = self.transform(video_frames)

        return {"frames": video_frames, "annotations": annotations}

    def _load_metadata(self, path: str) -> List[Dict]:
        """
        Load metadata for the dataset (video file paths, annotations).

        Args:
            path (str): Path to the dataset.

        Returns:
            List[Dict]: Metadata for each video.
        """
        metadata_file = os.path.join(path, f"{self.split}_metadata.yaml")
        with open(metadata_file, "r") as file:
            metadata = yaml.safe_load(file)
        return metadata

    def _build_transforms(self) -> transforms.Compose:
        """
        Build the transformation pipeline for preprocessing and augmentation.

        Returns:
            transforms.Compose: Torchvision transformation pipeline.
        """
        transform_list = []

        # Resize frames to target resolution
        transform_list.append(transforms.Resize((self.frame_resolution, self.frame_resolution)))

        # Apply augmentations if enabled
        if self.split == "train" and self.augmentations_enabled:
            if self.augmentations_enabled.get('horizontal_flip', False):
                transform_list.append(transforms.RandomHorizontalFlip())
            if self.augmentations_enabled.get('random_crop', False):
                transform_list.append(transforms.RandomCrop(self.frame_resolution))
            if self.augmentations_enabled.get('affine_transform', False):
                transform_list.append(transforms.RandomAffine(degrees=10, scale=(0.9, 1.1)))
            if self.augmentations_enabled.get('color_jitter', False):
                transform_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
            if self.augmentations_enabled.get('color_jitter', False):
                transform_list.append(transforms.RandomGrayscale(p=0.1))

        # Normalize video frames
        transform_list.append(transforms.ToTensor())
        transform_list.append(normalize_frames())  # Utility function for normalization

        return transforms.Compose(transform_list)

    def _load_video_and_annotations(self, video_info: Dict) -> Tuple[List, List]:
        """
        Load video frames and corresponding annotations.

        Args:
            video_info (Dict): Metadata for the video, including file paths and FPS info.

        Returns:
            Tuple[List, List]: A tuple of video frames and respective annotations.
        """
        video_path = video_info['video_path']
        annotation_path = video_info['annotation_path']
        video_frames = []  # Load video frames, placeholder
        annotations = []  # Load corresponding annotations (e.g., masks for segmentation)
        return video_frames, annotations
