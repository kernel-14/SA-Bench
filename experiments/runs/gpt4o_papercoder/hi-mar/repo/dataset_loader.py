## dataset_loader.py

import os
import json
from typing import Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from transformers import CLIPProcessor, CLIPModel

class DatasetLoader:
    """
    DatasetLoader handles loading and preprocessing for ImageNet and MS-COCO datasets.
    It also includes functionality to embed textual data using a pre-trained CLIP model.
    """

    def __init__(self, config: dict):
        """
        Initializes the DatasetLoader with configuration settings.

        Args:
            config (dict): Dictionary containing settings from `config.yaml`.
        """
        # Retrieve configurations
        self.imagenet_path = config["datasets"]["ImageNet"]["path"]
        self.imagenet_resolution = tuple(
            map(int, config["datasets"]["ImageNet"]["resolution"].split("x"))
        )
        self.mscoco_path = config["datasets"]["MSCOCO"]["path"]
        self.mscoco_resolution = tuple(
            map(int, config["datasets"]["MSCOCO"]["resolution"].split("x"))
        )
        self.mscoco_captions_per_image = config["datasets"]["MSCOCO"]["captions_per_image"]
        self.vae_resolutions = config["vae"]["resolutions"]
        
        # CLIP model and processor initialization (Hugging Face)
        self.clip_model_name = "openai/clip-vit-base-patch32"  # Default CLIP model
        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(self.clip_model_name)

        # Image normalization constants (ImageNet)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def load_ImageNet(self) -> Tuple[Dataset, Dataset]:
        """
        Loads the ImageNet dataset for class-conditional image generation.

        Returns:
            Tuple[Dataset, Dataset]: Training and validation datasets.
        """
        # Define image preprocessing pipeline
        transform_pipeline = transforms.Compose([
            transforms.Resize(self.imagenet_resolution),
            transforms.CenterCrop(self.imagenet_resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        # Ensure the dataset directory exists
        if not os.path.exists(self.imagenet_path):
            raise FileNotFoundError(f"ImageNet directory not found at: {self.imagenet_path}")

        # Load full dataset
        imagenet_dataset = datasets.ImageFolder(root=self.imagenet_path, transform=transform_pipeline)

        # Split dataset into training and validation sets (e.g., 80%-20%)
        dataset_length = len(imagenet_dataset)
        train_size = int(dataset_length * 0.8)
        val_size = dataset_length - train_size

        train_dataset, val_dataset = random_split(imagenet_dataset, [train_size, val_size])

        return train_dataset, val_dataset

    def load_MSCOCO(self) -> Tuple[Dataset, Dataset]:
        """
        Loads the MS-COCO dataset for text-to-image generation.

        Returns:
            Tuple[Dataset, Dataset]: Training and validation datasets.
        """
        from pycocotools.coco import COCO  # MS-COCO utility for annotations

        # Paths to images and annotations
        images_path = os.path.join(self.mscoco_path, "images")
        annotations_path = os.path.join(self.mscoco_path, "annotations", "captions_train2017.json")

        # Ensure the paths exist
        if not os.path.exists(images_path):
            raise FileNotFoundError(f"MS-COCO images directory not found at: {images_path}")
        if not os.path.exists(annotations_path):
            raise FileNotFoundError(f"MS-COCO annotations file not found at: {annotations_path}")

        # Dataset transformation pipeline for images
        transform_pipeline = transforms.Compose([
            transforms.Resize(self.mscoco_resolution),
            transforms.CenterCrop(self.mscoco_resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        # Initialize COCO dataset object
        coco = COCO(annotations_path)

        # Get image IDs
        image_ids = list(coco.imgs.keys())

        # Define Dataset class for MS-COCO
        class MSCOCODataset(Dataset):
            def __init__(self, image_ids, coco, images_path, transforms, captions_per_image):
                self.image_ids = image_ids
                self.coco = coco
                self.images_path = images_path
                self.transforms = transforms
                self.captions_per_image = captions_per_image

            def __len__(self):
                return len(self.image_ids)

            def __getitem__(self, idx):
                # Get image ID and corresponding data
                image_id = self.image_ids[idx]
                image_info = self.coco.loadImgs(image_id)[0]
                image_path = os.path.join(self.images_path, image_info["file_name"])
                captions = self.coco.imgToAnns[image_id]

                # Load and preprocess image
                image = transforms.ToPILImage()(image_path)
                image = self.transforms(image)

                # Extract captions (limit to caption_per_image)
                caption_texts = [caption["caption"] for caption in captions[:self.captions_per_image]]

                return image, caption_texts

        # Initialize dataset
        coco_dataset = MSCOCODataset(
            image_ids=image_ids,
            coco=coco,
            images_path=images_path,
            transforms=transform_pipeline,
            captions_per_image=self.mscoco_captions_per_image
        )

        # Split dataset into training and validation sets
        dataset_length = len(coco_dataset)
        train_size = int(dataset_length * 0.8)
        val_size = dataset_length - train_size

        train_dataset, val_dataset = random_split(coco_dataset, [train_size, val_size])

        return train_dataset, val_dataset

    def process_text(self, text: List[str]) -> torch.Tensor:
        """
        Processes a list of text captions into CLIP embeddings.

        Args:
            text (List[str]): List of textual captions.

        Returns:
            torch.Tensor: Embedded representations for the given text captions.
        """
        # Use CLIP tokenizer to encode text into tokens
        inputs = self.clip_processor(
            text=text,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        # Pass tokens through the CLIP model for embeddings
        with torch.no_grad():
            embeddings = self.clip_model.get_text_features(**inputs)

        return embeddings
