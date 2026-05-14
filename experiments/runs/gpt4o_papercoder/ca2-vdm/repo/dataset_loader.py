## dataset_loader.py
import os
from typing import Optional, Dict
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoTokenizer
import torch
import json
from config import Config

class VideoTextDataset(Dataset):
    """
    PyTorch Dataset for Text-to-Video generation, such as InternVid.
    Handles loading video-text pairs and applies preprocessing.
    """
    def __init__(
        self,
        data_path: str,
        split: str,
        resolution: int,
        tokenizer: AutoTokenizer,
        max_text_length: int,
        transform: Optional[transforms.Compose] = None
    ) -> None:
        """
        Initialize the dataset.

        Args:
            data_path (str): Path to the dataset split (e.g., train or test).
            split (str): Dataset split type (e.g., "train", "test").
            resolution (int): Resolution to which videos are resized.
            tokenizer (AutoTokenizer): Tokenizer for text prompts (e.g., from T5).
            max_text_length (int): Maximum token length for text encoding.
            transform (Optional[transforms.Compose]): Transformation pipeline for video preprocessing.
        """
        self.data_path = data_path
        self.resolution = resolution
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.transform = transform

        # Load video-text metadata
        with open(os.path.join(data_path, f"{split}.json"), "r") as f:
            self.metadata = json.load(f)

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.
        """
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieve an individual sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing "video" and "text" tensors.
        """
        # Access metadata for the specific index
        item = self.metadata[idx]
        video_path = item['video_path']
        text_prompt = item['text']

        # Load and preprocess video
        video_tensor = self._load_video(video_path)

        if self.transform:
            video_tensor = self.transform(video_tensor)

        # Tokenize text prompt
        text_encoding = self.tokenizer(
            text_prompt,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt"
        )

        return {
            "video": video_tensor,
            "text": text_encoding["input_ids"].squeeze(0)
        }

    def _load_video(self, video_path: str) -> torch.Tensor:
        """
        Load video frames as a tensor from the file path.

        Args:
            video_path (str): Path to the video file.

        Returns:
            torch.Tensor: Tensor representing the video, shape (T, C, H, W).
        """
        # Placeholder for actual video loading logic
        # Replace this with a library like decord or OpenCV for handling video files
        video_tensor = torch.zeros((16, 3, self.resolution, self.resolution))  # Dummy video tensor
        return video_tensor


class VideoOnlyDataset(Dataset):
    """
    PyTorch Dataset for video prediction tasks, such as SkyTimelapse.
    Handles loading video sequences and applies preprocessing.
    """
    def __init__(
        self,
        data_path: str,
        split: str,
        resolution: int,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            data_path (str): Path to the dataset split (e.g., train or test).
            split (str): Dataset split type (e.g., "train", "test").
            resolution (int): Resolution to which frames are resized.
            transform (Optional[transforms.Compose]): Transformation pipeline for video preprocessing.
        """
        self.data_path = os.path.join(data_path, split)
        self.resolution = resolution
        self.transform = transform

        # List all video files in the directory
        self.video_files = [os.path.join(self.data_path, f) for f in os.listdir(self.data_path) if f.endswith(".mp4")]

    def __len__(self) -> int:
        """
        Return the total number of videos in the dataset.
        """
        return len(self.video_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieve an individual video sample.

        Args:
            idx (int): Index of the video sample.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing video frames as a tensor.
        """
        video_path = self.video_files[idx]
        video_tensor = self._load_video(video_path)

        if self.transform:
            video_tensor = self.transform(video_tensor)

        return {"video": video_tensor}

    def _load_video(self, video_path: str) -> torch.Tensor:
        """
        Load video frames as a tensor from the file path.

        Args:
            video_path (str): Path to the video file.

        Returns:
            torch.Tensor: Tensor representing the video, shape (T, C, H, W).
        """
        # Placeholder for actual video loading logic
        video_tensor = torch.zeros((16, 3, self.resolution, self.resolution))  # Dummy video tensor
        return video_tensor


class DatasetLoader:
    """
    Centralized class for loading and preparing datasets for text-to-video and video prediction tasks.
    """
    def __init__(self, config: Config) -> None:
        """
        Initialize the DatasetLoader.

        Args:
            config (Config): Configuration object containing dataset and preprocessing settings.
        """
        self.config = config

    def load_data(self, split: str) -> DataLoader:
        """
        Load a specific dataset split as a PyTorch DataLoader.

        Args:
            split (str): Dataset split to load (e.g., 'train', 'val', or 'test').

        Returns:
            DataLoader: PyTorch DataLoader for the specified dataset split.
        """
        dataset_name = self.config.get("current_dataset", "intern_vid")
        resolution = self.config.get("vae.resolution", 256)
        batch_size = self.config.get(f"training.batch_size_{dataset_name}", 32)

        if dataset_name == "intern_vid":
            tokenizer = AutoTokenizer.from_pretrained("t5-small")
            max_text_length = 64

            dataset = VideoTextDataset(
                data_path=self.config.get("datasets.intern_vid.splits", {}).get(split, ""),
                split=split,
                resolution=resolution,
                tokenizer=tokenizer,
                max_text_length=max_text_length,
                transform=self.prepare_transforms(resolution),
            )
        elif dataset_name == "sky_timelapse":
            dataset = VideoOnlyDataset(
                data_path=self.config.get("datasets.sky_timelapse.splits", {}).get(split, ""),
                split=split,
                resolution=resolution,
                transform=self.prepare_transforms(resolution),
            )
        elif dataset_name == "ucf101":
            dataset = VideoOnlyDataset(
                data_path=self.config.get("datasets.ucf101.split", ""),
                split=split,
                resolution=resolution,
                transform=self.prepare_transforms(resolution),
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=4)

    def prepare_transforms(self, resolution: int) -> transforms.Compose:
        """
        Define the preprocessing pipeline for video frames.

        Args:
            resolution (int): Resolution to resize frames to.

        Returns:
            transforms.Compose: A composed pipeline of transformations.
        """
        return transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalizes to [-1, 1]
        ])
