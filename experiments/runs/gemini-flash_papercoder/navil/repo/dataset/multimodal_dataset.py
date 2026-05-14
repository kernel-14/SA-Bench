import json
import os
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageOps

from transformers import PreTrainedTokenizer

from config import Config
from utils import logger


def _pad_image_to_multiple(img: Image.Image, multiple: int) -> Image.Image:
    """
    Pads a PIL Image with black pixels so that its width and height
    are multiples of the specified `multiple`.

    Args:
        img: The input PIL Image.
        multiple: The integer value to which width and height should be multiples.

    Returns:
        The padded PIL Image.
    """
    width, height = img.size
    
    pad_w = (multiple - (width % multiple)) % multiple
    pad_h = (multiple - (height % multiple)) % multiple

    if pad_w == 0 and pad_h == 0:
        return img # No padding needed

    # Pad with black color (0, 0, 0)
    padded_img = ImageOps.expand(img, border=(0, 0, pad_w, pad_h), fill=(0, 0, 0))
    logger.debug(f"Padded image from {width}x{height} to {padded_img.size[0]}x{padded_img.size[1]} "
                 f"(target multiple: {multiple})")
    return padded_img


class MultimodalDataset(Dataset):
    """
    Dataset class for loading multimodal (image-text) data.
    Handles image loading, initial padding, Visual Multi-scale Packing (VMP),
    and text tokenization.
    """

    def __init__(self, data_paths: List[str], tokenizer: PreTrainedTokenizer, config: Config, stage: str, is_train: bool):
        """
        Initializes the MultimodalDataset.

        Args:
            data_paths: A list of file paths (e.g., JSONL files) containing the image-text pairs.
            tokenizer: The Hugging Face tokenizer for the LLM.
            config: The global configuration object.
            stage: The current training stage (e.g., "stage_1_1", "stage_1_2", "stage_2").
            is_train: A boolean indicating if the dataset is for training.
        """
        self.tokenizer = tokenizer
        self.config = config
        self.stage = stage
        self.is_train = is_train

        # Retrieve stage-specific configuration
        stage_config = self.config.get(f"training_stages.{self.stage}")
        if stage_config is None:
            raise ValueError(f"Configuration for stage '{self.stage}' not found.")

        self.vmp_enabled = stage_config.visual_multi_scale_packing
        self.vmp_tau = self.config.get("common.visual_multi_scale_packing_tau")
        self.vmp_area_threshold = self.config.get("evaluation.vmp_area_threshold")
        self.patch_embedding_multiple = self.config.get("model_architecture.visual_encoder.patch_embedding_stride") * 2 # Images padded to multiple of 32. Original stride 16 means patches are 16x16. 32 ensures minimum 2x2 patch grid.

        logger.info(f"Dataset for stage '{self.stage}' initialized. VMP enabled: {self.vmp_enabled}")
        if self.vmp_enabled:
            logger.info(f"VMP parameters: tau={self.vmp_tau}, area_threshold={self.vmp_area_threshold}")

        self.data_samples: List[Dict[str, str]] = []
        self._load_data_samples(data_paths)

        # Image transforms
        self.to_tensor = transforms.ToTensor()
        # Resize transform expects (H, W) or an int (for shorter side)
        self.resize_transform = transforms.Resize


    def _load_data_samples(self, data_paths: List[str]) -> None:
        """
        Loads image-text pairs from the specified data paths.
        Assumes JSONL format where each line is a JSON object with 'image_path' and 'text' keys.
        """
        logger.info(f"Loading data samples from {data_paths}...")
        total_samples = 0
        for path in data_paths:
            if not os.path.exists(path):
                logger.warning(f"Data path not found: {path}. Skipping.")
                continue
            
            # Simple check for file type, assuming JSONL
            if os.path.isdir(path):
                logger.info(f"Path is a directory: {path}. Attempting to read all .jsonl files.")
                jsonl_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jsonl')]
            elif os.path.isfile(path) and path.endswith('.jsonl'):
                jsonl_files = [path]
            else:
                logger.warning(f"Unsupported data file format: {path}. Skipping.")
                continue

            for jsonl_file in jsonl_files:
                try:
                    with open(jsonl_file, 'r', encoding='utf-8') as f:
                        for line_idx, line in enumerate(f):
                            try:
                                sample = json.loads(line)
                                if 'image_path' not in sample or 'text' not in sample:
                                    logger.warning(f"Skipping malformed sample in {jsonl_file} line {line_idx}: "
                                                   "Missing 'image_path' or 'text' keys.")
                                    continue
                                
                                # Make image_path absolute if it's relative to the data_path
                                # This assumes image paths in JSONL are relative to the JSONL file's directory
                                # or absolute. If relative to an overarching data_dir, this needs adjustment.
                                if not os.path.isabs(sample['image_path']):
                                    sample['image_path'] = os.path.join(os.path.dirname(jsonl_file), sample['image_path'])

                                self.data_samples.append({
                                    'image_path': sample['image_path'],
                                    'text': sample['text']
                                })
                                total_samples += 1
                            except json.JSONDecodeError:
                                logger.warning(f"Skipping invalid JSON line in {jsonl_file} line {line_idx}.")
                except Exception as e:
                    logger.error(f"Error reading data from {jsonl_file}: {e}")
        logger.info(f"Finished loading data samples. Total samples: {total_samples}")


    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data_samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves an image-text pair from the dataset, applies necessary
        preprocessing, and returns it as a dictionary.

        Args:
            idx: The index of the sample to retrieve.

        Returns:
            A dictionary containing:
                'image_tensors': List[torch.Tensor] (scaled image tensors)
                'text_ids': List[int] (token IDs of the raw text)
                'original_text': str (the original text content)
                'image_path': str (path to the image, for debugging/logging)
        """
        sample = self.data_samples[idx]
        image_path = sample['image_path']
        text = sample['text']

        # Load and preprocess image (including VMP if enabled)
        image_tensors_list = self._load_and_preprocess_image(image_path)

        # Tokenize text (without adding special tokens, collate_fn will handle it)
        text_token_ids = self.tokenizer(text, add_special_tokens=False)['input_ids']

        return {
            'image_tensors': image_tensors_list,
            'text_ids': text_token_ids,
            'original_text': text,
            'image_path': image_path,
        }

    def _load_and_preprocess_image(self, image_path: str) -> List[torch.Tensor]:
        """
        Loads an image, applies initial padding, and optionally performs
        Visual Multi-scale Packing (VMP).

        Args:
            image_path: The file path to the image.

        Returns:
            A list of PyTorch Tensors, each representing an image at a different scale
            (or just one if VMP is disabled), ready for the visual encoder.
        """
        try:
            img_pil = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image from {image_path}: {e}. Returning dummy image.")
            # Return a dummy black image if loading fails
            # This should be handled gracefully by the training loop, e.g., skipping the sample
            # For now, a 3x32x32 black image as a placeholder.
            return [torch.zeros((3, self.patch_embedding_multiple, self.patch_embedding_multiple), dtype=torch.float)]

        # Apply initial padding to ensure dimensions are multiples of 32
        current_image_pil = _pad_image_to_multiple(img_pil, self.patch_embedding_multiple)
        
        image_tensors_list: List[torch.Tensor] = []

        if self.vmp_enabled:
            logger.debug(f"Applying VMP for image: {image_path}")
            # Add the largest (original padded) image first
            image_tensors_list.append(self.to_tensor(current_image_pil))

            while True:
                current_width, current_height = current_image_pil.size
                
                # Calculate new dimensions
                new_width = round(current_width * self.vmp_tau)
                new_height = round(current_height * self.vmp_tau)

                # Stop if area falls below threshold or dimensions become too small
                if new_width * new_height < self.vmp_area_threshold or new_width < self.patch_embedding_multiple or new_height < self.patch_embedding_multiple:
                    break

                # Resize and re-pad the image
                current_image_pil = self.resize_transform((new_height, new_width))(current_image_pil)
                current_image_pil = _pad_image_to_multiple(current_image_pil, self.patch_embedding_multiple)
                
                image_tensors_list.append(self.to_tensor(current_image_pil))
        else:
            # If VMP is disabled, just add the single padded image
            image_tensors_list.append(self.to_tensor(current_image_pil))
            logger.debug(f"VMP disabled. Added single image scale for {image_path}.")

        return image_tensors_list

