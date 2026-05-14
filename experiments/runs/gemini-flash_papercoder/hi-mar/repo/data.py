import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import json
from typing import Dict, Any, Tuple, Callable, Optional, List, Union
import random
import logging

# Assuming tokenizer.py is in the same directory or accessible via import
# The design specifies `tokenizer.VAETokenizer` and `tokenizer.CLIPTextEncoder`
# so these classes are imported directly from a `tokenizer` module.
from tokenizer import VAETokenizer, CLIPTextEncoder

logger = logging.getLogger(__name__)
# Configure logging for better visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class HiMARDataset(Dataset):
    """
    Dataset class for loading image data for Hi-MAR.
    Handles both class-conditional (ImageNet) and text-to-image (MS-COCO) datasets.
    It provides high-resolution and low-resolution versions of the image,
    along with appropriate conditioning information.
    """
    def __init__(
        self,
        root_dir: str,
        split: str,
        transform: Callable,
        is_text_to_image: bool,
        high_res_image_size: int,
        low_res_image_size: int,
        clip_encoder: Optional[CLIPTextEncoder] = None
    ):
        """
        Initializes the HiMARDataset.

        Args:
            root_dir: Base directory of the dataset.
            split: 'train' or 'val'.
            transform: A torchvision.transforms.Compose object for final tensor transformation (ToTensor, Normalize).
                       Does NOT include resizing, which is handled explicitly within __getitem__.
            is_text_to_image: True for MS-COCO, False for ImageNet.
            high_res_image_size: Target size for the high-resolution image (e.g., 256).
            low_res_image_size: Target size for the low-resolution image (e.g., 128).
            clip_encoder: Optional CLIPTextEncoder instance for text-to-image tasks.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.is_text_to_image = is_text_to_image
        self.high_res_image_size = high_res_image_size
        self.low_res_image_size = low_res_image_size
        self.clip_encoder = clip_encoder

        self.image_infos: List[Tuple[str, Any]] = [] # Stores (image_path, raw_condition)

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Dataset root directory not found: {root_dir}")

        if self.is_text_to_image:
            self._load_mscoco_dataset()
        else:
            self._load_imagenet_dataset()

        if not self.image_infos:
            logger.warning(f"No image data loaded for split '{split}' in {root_dir}")

        logger.info(f"Loaded {len(self.image_infos)} samples for {self.split} split.")

    def _load_mscoco_dataset(self):
        """Loads image paths and captions for MS-COCO dataset."""
        # Example structure: root_dir/annotations/captions_train2017.json, root_dir/train2017/
        annotation_file = os.path.join(self.root_dir, 'annotations', f'captions_{self.split}2017.json')
        if not os.path.exists(annotation_file):
            raise FileNotFoundError(f"MS-COCO annotation file not found: {annotation_file}")

        with open(annotation_file, 'r') as f:
            annotations = json.load(f)

        image_id_to_filename = {img['id']: img['file_name'] for img in annotations['images']}
        image_id_to_captions: Dict[int, List[str]] = {}

        for ann in annotations['annotations']:
            image_id = ann['image_id']
            caption = ann['caption']
            if image_id not in image_id_to_captions:
                image_id_to_captions[image_id] = []
            image_id_to_captions[image_id].append(caption)
        
        # Create a list of (image_path, list_of_captions)
        for image_id, captions in image_id_to_captions.items():
            filename = image_id_to_filename.get(image_id)
            if filename:
                # Determine image directory (e.g., 'train2017' or 'val2017')
                image_dir_name = f"{self.split}2017" 
                image_path = os.path.join(self.root_dir, image_dir_name, filename)
                if os.path.exists(image_path):
                    self.image_infos.append((image_path, captions))
                else:
                    logger.warning(f"Image not found at {image_path} for MS-COCO, skipping.")
            else:
                logger.warning(f"Filename not found for MS-COCO image_id {image_id}, skipping.")

    def _load_imagenet_dataset(self):
        """Loads image paths and class IDs for ImageNet dataset."""
        # ImageNet typically has a structure like:
        # root_dir/train/n01440764/ILSVRC2012_00000293.JPEG
        # root_dir/val/n01440764/ILSVRC2012_00000293.JPEG
        
        split_dir = os.path.join(self.root_dir, self.split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"ImageNet split directory not found: {split_dir}")

        class_names = sorted(os.listdir(split_dir))
        class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}

        for class_name in class_names:
            class_path = os.path.join(split_dir, class_name)
            if os.path.isdir(class_path):
                class_idx = class_to_idx[class_name]
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                        img_path = os.path.join(class_path, img_name)
                        self.image_infos.append((img_path, class_idx))

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.image_infos)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int]]:
        """
        Retrieves a preprocessed image pair (high-resolution and low-resolution)
        and its conditioning information.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A dictionary containing:
            - 'high_res_image': torch.Tensor of shape (C, high_res_image_size, high_res_image_size)
            - 'low_res_image': torch.Tensor of shape (C, low_res_image_size, low_res_image_size)
            - 'conditions': torch.Tensor (text embedding) or int (class ID)
        """
        image_path, raw_condition = self.image_infos[idx]

        # 1. Image Loading and Resizing (PIL images first)
        img = Image.open(image_path).convert('RGB')

        # High-resolution image (PIL)
        high_res_img_pil = img.resize((self.high_res_image_size, self.high_res_image_size), Image.BICUBIC)
        
        # Low-resolution image (PIL)
        low_res_img_pil = img.resize((self.low_res_image_size, self.low_res_image_size), Image.BICUBIC)

        # Apply common tensor transform (ToTensor, Normalize)
        high_res_image = self.transform(high_res_img_pil)
        low_res_image = self.transform(low_res_img_pil)

        # 2. Conditioning Preparation
        conditions: Union[torch.Tensor, int]
        if self.is_text_to_image:
            if self.clip_encoder is None:
                raise ValueError("CLIPTextEncoder must be provided for text-to-image tasks.")
            # Raw condition is a list of captions for MS-COCO, pick one randomly
            caption_string = random.choice(raw_condition) if isinstance(raw_condition, list) else raw_condition
            # encode_text returns (1, seq_len, embed_dim), squeeze to (seq_len, embed_dim)
            conditions = self.clip_encoder.encode_text([caption_string]).squeeze(0)
        else:
            # Raw condition is the class_id integer for ImageNet
            conditions = raw_condition
        
        return {
            'high_res_image': high_res_image,
            'low_res_image': low_res_image,
            'conditions': conditions
        }


class DataModule:
    """
    Encapsulates dataset and dataloader setup for Hi-MAR.
    Provides training and validation dataloaders based on the global configuration.
    """
    def __init__(
        self,
        config: Dict[str, Any],
        tokenizer: VAETokenizer,
        clip_encoder: Optional[CLIPTextEncoder] = None
    ):
        """
        Initializes the DataModule.

        Args:
            config: The loaded configuration dictionary, specifically the training section.
            tokenizer: An instance of VAETokenizer (used for its image size configs).
            clip_encoder: An instance of CLIPTextEncoder, if text-to-image task is enabled.
        """
        self.config = config
        self.tokenizer = tokenizer # Stored to access image_size configs, not for actual tokenization here
        self.clip_encoder = clip_encoder

        self.train_dataset: Optional[HiMARDataset] = None
        self.val_dataset: Optional[HiMARDataset] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None

        # Determine active dataset based on config.training.<dataset>.enabled flags
        training_config = self.config.get('training', {})
        imagenet_cfg = training_config.get('imagenet', {})
        mscoco_cfg = training_config.get('mscoco', {})

        imagenet_enabled = imagenet_cfg.get('enabled', False)
        mscoco_enabled = mscoco_cfg.get('enabled', False)

        if imagenet_enabled and mscoco_enabled:
            raise ValueError("Only one dataset (ImageNet or MS-COCO) can be enabled at a time for training.")
        elif imagenet_enabled:
            self._setup_dataset_params = imagenet_cfg
            self._is_text_to_image = False
            self.dataset_name = "ImageNet"
        elif mscoco_enabled:
            self._setup_dataset_params = mscoco_cfg
            self._is_text_to_image = True
            self.dataset_name = "MS-COCO"
        else:
            raise ValueError("No dataset is enabled in the training configuration. Set 'enabled: true' for either 'imagenet' or 'mscoco'.")

        self.data_path = self._setup_dataset_params.get('data_path', 'path/to/dataset')
        self.batch_size = self._setup_dataset_params.get('batch_size', 1)
        self.conditional_type = self._setup_dataset_params.get('conditional_type', 'class')
        
        # Define image transformations (ToTensor and Normalize only, resizing handled in HiMARDataset.__getitem__)
        self.transform = transforms.Compose([
            transforms.ToTensor(), # Converts PIL Image to FloatTensor, scales to [0, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalizes to [-1, 1]
        ])

        self._setup_datasets()

    def _setup_datasets(self):
        """Sets up the training and validation datasets and dataloaders."""
        num_workers = os.cpu_count() // 2 if os.cpu_count() else 0 # Use half available CPU cores, or 0
        if num_workers == 0:
            logger.warning("Number of CPU workers for data loading is 0. This might be slow. Consider increasing if resources allow.")

        self.train_dataset = HiMARDataset(
            root_dir=self.data_path,
            split='train',
            transform=self.transform,
            is_text_to_image=self._is_text_to_image,
            high_res_image_size=self.tokenizer.high_res_image_size,
            low_res_image_size=self.tokenizer.low_res_image_size,
            clip_encoder=self.clip_encoder if self._is_text_to_image else None
        )
        self.val_dataset = HiMARDataset(
            root_dir=self.data_path,
            split='val',
            transform=self.transform,
            is_text_to_image=self._is_text_to_image,
            high_res_image_size=self.tokenizer.high_res_image_size,
            low_res_image_size=self.tokenizer.low_res_image_size,
            clip_encoder=self.clip_encoder if self._is_text_to_image else None
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True # Typically speeds up data transfer to GPU
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False, # No need to shuffle validation data
            num_workers=num_workers,
            pin_memory=True
        )
        logger.info(f"DataModule initialized for {self.dataset_name} with batch size {self.batch_size}.")
        logger.info(f"Train dataset size: {len(self.train_dataset)}, Val dataset size: {len(self.val_dataset)}")

    def get_train_dataloader(self) -> DataLoader:
        """Returns the configured training DataLoader."""
        if self.train_loader is None:
            raise RuntimeError("Train DataLoader not initialized. Call DataModule._setup_datasets() first.")
        return self.train_loader

    def get_val_dataloader(self) -> DataLoader:
        """Returns the configured validation DataLoader."""
        if self.val_loader is None:
            raise RuntimeError("Validation DataLoader not initialized. Call DataModule._setup_datasets() first.")
        return self.val_loader

