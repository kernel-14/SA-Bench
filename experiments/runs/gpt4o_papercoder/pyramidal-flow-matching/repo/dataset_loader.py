## dataset_loader.py

import os
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from typing import Dict, List, Tuple
from pyramid_utils import PyramidUtils
from transformers import CLIPProcessor, CLIPModel

class DatasetLoader:
    """
    DatasetLoader handles loading, preprocessing, augmentations, and batching for both image and video datasets.
    It integrates utility functions for spatial and temporal pyramid processing and relies on external captioning models 
    for video-text alignment (e.g., CLIP).
    """

    def __init__(self, config: Dict[str, Dict]) -> None:
        """
        Initialize DatasetLoader.

        Args:
            config (dict): Configuration dictionary containing dataset paths and preprocessing parameters.
        """
        self.config = config
        self.image_paths = config["dataset"]["image_datasets"]
        self.video_paths = config["dataset"]["video_datasets"]
        self.batch_size = config["training"]["batch_size"]
        
        # Transformer models for video captioning and text alignment evaluation
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    def load_images(self) -> DataLoader:
        """
        Load image datasets from specified paths, apply preprocessing, and return a PyTorch DataLoader.

        Returns:
            DataLoader: Batched images ready for training.
        """
        # Transformation pipeline for images
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Construct dataset for image data
        image_files = self._scan_files([
            self.image_paths["LAION-5B_path"],
            self.image_paths["CC-12M_path"],
            self.image_paths["SA-1B_path"],
            self.image_paths["JourneyDB_path"],
            self.image_paths.get("synthetic_data", None)
        ])
        dataset = ImageDataset(image_files, transform)

        # Return DataLoader
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def load_videos(self, duration: str = "short") -> DataLoader:
        """
        Load video datasets, apply preprocessing and text-to-video alignment, and return a PyTorch DataLoader.

        Args:
            duration (str): Video duration type ('short' or 'long') as defined in the configuration.

        Returns:
            DataLoader: Batched video latents ready for training or evaluation.
        """
        # Validate duration
        assert duration in ["short", "long"], "Unsupported duration parameter. Choose 'short' or 'long'."

        video_path_key = f"{duration}_duration"
        video_paths = [
            self.video_paths["WebVid-10M_path"],
            self.video_paths["OpenVid-1M_path"],
            self.video_paths["OpenSoraPlan_path"]
        ]
        
        # Scan video files
        video_files = self._scan_files(video_paths)

        # Construct VideoDataset
        dataset = VideoDataset(
            video_files=video_files,
            clip_processor=self.clip_processor,
            clip_model=self.clip_model,
            transform=None,  # Additional augmentations can be added later
            compression_fn=PyramidUtils.compress_frames,
            noise_fn=PyramidUtils.add_noise,
            compression_levels=self.config["model"]["vae"]["downsampling_ratio"],
            noise_level=self.config["model"]["flow_matching"]["noise_level"][1]
        )

        # Return DataLoader
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def _scan_files(self, paths: List[str]) -> List[str]:
        """
        Scan given dataset paths for available files.

        Args:
            paths (List[str]): List of paths to scan.

        Returns:
            List[str]: List of valid files found in the directories.
        """
        files = []
        for path in paths:
            if path and os.path.exists(path):
                for root, _, filenames in os.walk(path):
                    for file in filenames:
                        if file.endswith((".jpg", ".png", ".mp4", ".mkv")):  # Image and video formats
                            files.append(os.path.join(root, file))
            else:
                print(f"Warning: Path not found or invalid - {path}")
        return files


class ImageDataset(Dataset):
    """
    A PyTorch Dataset class for loading and preprocessing image data.
    """
    def __init__(self, file_paths: List[str], transform: transforms.Compose) -> None:
        """
        Initialize the ImageDataset.

        Args:
            file_paths (List[str]): List of image file paths.
            transform (transforms.Compose): Preprocessing pipeline for images.
        """
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self) -> int:
        """
        Get the total number of images.

        Returns:
            int: Total number of images.
        """
        return len(self.file_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        Load and preprocess an image.

        Args:
            index (int): Index of the image to load.

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        from PIL import Image
        img = Image.open(self.file_paths[index]).convert("RGB")
        return self.transform(img)


class VideoDataset(Dataset):
    """
    A PyTorch Dataset class for loading, preprocessing, and augmenting video data.
    """
    def __init__(
        self,
        video_files: List[str],
        clip_processor: CLIPProcessor,
        clip_model: CLIPModel,
        transform: transforms.Compose,
        compression_fn,
        noise_fn,
        compression_levels: Tuple[int, int, int],
        noise_level: float,
    ) -> None:
        """
        Initialize the VideoDataset.

        Args:
            video_files (List[str]): List of video file paths.
            clip_processor (CLIPProcessor): CLIP model processor for text-to-video alignment.
            clip_model (CLIPModel): CLIP model for text captioning alignment.
            transform (transforms.Compose): Preprocessing pipeline for frames.
            compression_fn (callable): Function to compress video frames into latent space.
            noise_fn (callable): Function to add Gaussian noise.
            compression_levels (Tuple[int, int, int]): Downsampling ratios for spatial and temporal pyramid.
            noise_level (float): Noise strength to apply to frames.
        """
        self.video_files = video_files
        self.clip_processor = clip_processor
        self.clip_model = clip_model
        self.transform = transform
        self.compression_fn = compression_fn
        self.noise_fn = noise_fn
        self.compression_levels = compression_levels
        self.noise_level = noise_level

    def __len__(self) -> int:
        """
        Get the total number of video files.

        Returns:
            int: Total number of video files.
        """
        return len(self.video_files)

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        Load, preprocess, and augment a video file.

        Args:
            index (int): Index of the video to load.

        Returns:
            torch.Tensor: Processed latent representation of the video frames.
        """
        import cv2
        video_path = self.video_files[index]
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0  # [H, W, C] -> [C, H, W]
            if self.transform:
                frame_tensor = self.transform(frame_tensor)
            frames.append(frame_tensor)
        cap.release()

        frames = torch.stack(frames)
        compressed_frames = self.compression_fn(frames, levels=self.compression_levels[0])
        augmented_frames = self.noise_fn(compressed_frames, noise_level=self.noise_level)
        
        return augmented_frames
