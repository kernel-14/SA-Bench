"""
Data processing for NaViL training.

Handles:
1. Image preprocessing (padding to multiples of 32, normalization)
2. Multi-scale image preparation
3. Special token insertion:
   - <begin_of_image> before image tokens
   - <end_of_image> after image tokens
   - <end_of_line> at end of each row of image tokens
   - <end_of_scale> after each scale in multi-scale packing
4. Dataset classes for pre-training and SFT
"""

import os
import math
import json
import random
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


# Image normalization constants (ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Special token strings
SPECIAL_TOKENS = {
    "begin_of_image": "<begin_of_image>",
    "end_of_image": "<end_of_image>",
    "end_of_line": "<end_of_line>",
    "end_of_scale": "<end_of_scale>",
}


@dataclass
class ImageProcessorConfig:
    """Configuration for image preprocessing."""
    patch_size: int = 16
    pixel_shuffle_factor: int = 2
    image_mean: List[float] = None
    image_std: List[float] = None
    max_image_size: int = 4096
    min_image_size: int = 32

    def __post_init__(self):
        if self.image_mean is None:
            self.image_mean = IMAGENET_MEAN
        if self.image_std is None:
            self.image_std = IMAGENET_STD


class ImageProcessor:
    """
    Preprocesses images for NaViL.

    Key operations:
    1. Pad image so H and W are multiples of patch_size (16)
    2. Normalize with ImageNet mean/std
    3. Convert to tensor
    """

    def __init__(self, config: ImageProcessorConfig = None):
        self.config = config or ImageProcessorConfig()

    def pad_to_multiple(self, image: Image.Image, multiple: int = 32) -> Image.Image:
        """
        Pad image so that H and W are multiples of `multiple`.
        From paper: "input images are first padded to ensure its length and
        width are multiples of 32."
        """
        w, h = image.size
        new_w = math.ceil(w / multiple) * multiple
        new_h = math.ceil(h / multiple) * multiple

        if new_w == w and new_h == h:
            return image

        # Pad with zeros (black)
        padded = Image.new(image.mode, (new_w, new_h), 0)
        padded.paste(image, (0, 0))
        return padded

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize tensor with ImageNet mean/std."""
        mean = torch.tensor(self.config.image_mean, dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor(self.config.image_std, dtype=tensor.dtype).view(3, 1, 1)
        return (tensor - mean) / std

    def to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL image to normalized tensor."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        arr = np.array(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (C, H, W)
        return self.normalize(tensor)

    def process(self, image: Union[Image.Image, str]) -> torch.Tensor:
        """
        Process a single image.

        Args:
            image: PIL Image or path to image
        Returns:
            tensor: (C, H, W) normalized image tensor
        """
        if isinstance(image, str):
            image = Image.open(image)

        # Pad to multiple of 32
        image = self.pad_to_multiple(image, multiple=32)

        # Convert to tensor and normalize
        tensor = self.to_tensor(image)
        return tensor

    def process_batch(
        self,
        images: List[Union[Image.Image, str]],
        target_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Process a batch of images.

        Args:
            images: list of PIL Images or paths
            target_size: optional (H, W) to resize all images to
        Returns:
            batch: (B, C, H, W)
        """
        tensors = []
        for img in images:
            t = self.process(img)
            if target_size is not None:
                t = F.interpolate(
                    t.unsqueeze(0),
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            tensors.append(t)

        # Pad to same size if needed
        if len(set(t.shape for t in tensors)) > 1:
            max_h = max(t.shape[1] for t in tensors)
            max_w = max(t.shape[2] for t in tensors)
            padded = []
            for t in tensors:
                pad_h = max_h - t.shape[1]
                pad_w = max_w - t.shape[2]
                t = F.pad(t, (0, pad_w, 0, pad_h))
                padded.append(t)
            tensors = padded

        return torch.stack(tensors)


class MultiScaleImageProcessor:
    """
    Prepares multi-scale image sequences for Visual Multi-scale Packing.

    Given image I_0 and tau = sqrt(2)/2:
    - I_0: original image
    - I_1: I_0 downsampled by tau
    - I_2: I_1 downsampled by tau
    - ... until area < threshold
    """

    def __init__(
        self,
        tau: float = 0.5 * math.sqrt(2),
        min_area: int = 32 * 32,
        patch_size: int = 16,
        image_processor: Optional[ImageProcessor] = None,
    ):
        self.tau = tau
        self.min_area = min_area
        self.patch_size = patch_size
        self.image_processor = image_processor or ImageProcessor()

    def get_scale_sizes(self, H: int, W: int) -> List[Tuple[int, int]]:
        """Get list of (H_i, W_i) for each scale."""
        sizes = [(H, W)]
        h, w = H, W
        while True:
            h_new = int(h * self.tau)
            w_new = int(w * self.tau)
            # Round to multiples of patch_size
            h_new = max(self.patch_size, (h_new // self.patch_size) * self.patch_size)
            w_new = max(self.patch_size, (w_new // self.patch_size) * self.patch_size)
            if h_new * w_new < self.min_area:
                break
            if h_new == h and w_new == w:
                break
            sizes.append((h_new, w_new))
            h, w = h_new, w_new
        return sizes

    def process(self, image: Union[Image.Image, str]) -> List[torch.Tensor]:
        """
        Process image into multi-scale sequence.

        Returns:
            list of tensors, one per scale, from original to smallest
        """
        if isinstance(image, str):
            image = Image.open(image)

        # Pad original to multiple of 32
        image = self.image_processor.pad_to_multiple(image, multiple=32)
        w, h = image.size

        scale_sizes = self.get_scale_sizes(h, w)
        scale_tensors = []

        for h_i, w_i in scale_sizes:
            if h_i == h and w_i == w:
                img_i = image
            else:
                img_i = image.resize((w_i, h_i), Image.BILINEAR)
            tensor_i = self.image_processor.to_tensor(img_i)
            scale_tensors.append(tensor_i)

        return scale_tensors


class TokenBuilder:
    """
    Builds token sequences with special image tokens.

    Inserts:
    - <begin_of_image> before image tokens
    - <end_of_image> after image tokens
    - <end_of_line> at end of each row of image tokens
    - <end_of_scale> after each scale
    """

    def __init__(self, tokenizer, special_token_ids: Dict[str, int]):
        self.tokenizer = tokenizer
        self.special_token_ids = special_token_ids

    def build_image_token_sequence(
        self,
        grid_size: Tuple[int, int],
        image_token_id: int,
        include_eol: bool = True,
        include_eos: bool = True,
    ) -> List[int]:
        """
        Build token sequence for a single image at one scale.

        Args:
            grid_size: (H, W) grid of image tokens after connector
            image_token_id: placeholder token ID for image tokens
            include_eol: whether to include <end_of_line> tokens
            include_eos: whether to include <end_of_scale> token
        Returns:
            list of token IDs
        """
        H, W = grid_size
        tokens = [self.special_token_ids["begin_of_image"]]

        for row in range(H):
            for col in range(W):
                tokens.append(image_token_id)
            if include_eol:
                tokens.append(self.special_token_ids["end_of_line"])

        tokens.append(self.special_token_ids["end_of_image"])
        if include_eos:
            tokens.append(self.special_token_ids["end_of_scale"])

        return tokens

    def build_multiscale_token_sequence(
        self,
        grid_sizes: List[Tuple[int, int]],
        image_token_id: int,
    ) -> List[int]:
        """
        Build token sequence for multi-scale image.

        Args:
            grid_sizes: list of (H_i, W_i) for each scale
            image_token_id: placeholder token ID
        Returns:
            list of token IDs
        """
        all_tokens = []
        for i, grid_size in enumerate(grid_sizes):
            scale_tokens = self.build_image_token_sequence(
                grid_size,
                image_token_id,
                include_eol=True,
                include_eos=(i < len(grid_sizes) - 1),  # No EOS for last scale
            )
            all_tokens.extend(scale_tokens)
        return all_tokens


class PretrainDataset(Dataset):
    """
    Dataset for pre-training NaViL on image-text pairs.

    Supports:
    - Web-scale noisy image-caption pairs (Laion-2B, Coyo-700M, Wukong, SA-1B)
    - Synthetic captions from existing MLLMs (InternVL-8B)

    Training objective: Next-Token-Prediction on image captioning.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        image_processor: ImageProcessor,
        max_seq_len: int = 4096,
        use_synthetic_captions: bool = False,
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_seq_len = max_seq_len
        self.use_synthetic_captions = use_synthetic_captions

        # Load data index
        self.data = self._load_data()

    def _load_data(self) -> List[Dict]:
        """Load data index from JSON/JSONL file."""
        data = []
        if os.path.isfile(self.data_path):
            with open(self.data_path) as f:
                if self.data_path.endswith(".jsonl"):
                    for line in f:
                        data.append(json.loads(line.strip()))
                else:
                    data = json.load(f)
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        # Load image
        image_path = item["image"]
        try:
            image = self.image_processor.process(image_path)
        except Exception:
            # Return a dummy sample on error
            image = torch.zeros(3, 448, 448)

        # Get caption
        if self.use_synthetic_captions and "synthetic_caption" in item:
            caption = item["synthetic_caption"]
        else:
            caption = item.get("caption", item.get("text", ""))

        # Tokenize caption
        tokens = self.tokenizer.encode(caption, add_special_tokens=True)
        tokens = tokens[:self.max_seq_len]

        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "images": image,
            "labels": labels,
        }


class SFTDataset(Dataset):
    """
    Dataset for supervised fine-tuning (Stage 2).

    Supports multi-turn conversations with images.
    Uses InternLM2 conversation format.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        image_processor: ImageProcessor,
        max_seq_len: int = 4096,
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_seq_len = max_seq_len

        self.data = self._load_data()

    def _load_data(self) -> List[Dict]:
        data = []
        if os.path.isfile(self.data_path):
            with open(self.data_path) as f:
                if self.data_path.endswith(".jsonl"):
                    for line in f:
                        data.append(json.loads(line.strip()))
                else:
                    data = json.load(f)
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        # Load image if present
        image = None
        if "image" in item:
            try:
                image = self.image_processor.process(item["image"])
            except Exception:
                image = torch.zeros(3, 448, 448)

        # Build conversation
        conversations = item.get("conversations", [])
        input_ids, labels = self._build_conversation(conversations)

        result = {
            "input_ids": input_ids,
            "labels": labels,
        }
        if image is not None:
            result["images"] = image

        return result

    def _build_conversation(
        self,
        conversations: List[Dict],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build token sequence from conversation.
        Only compute loss on assistant responses.
        """
        all_tokens = []
        all_labels = []

        for turn in conversations:
            role = turn.get("role", turn.get("from", ""))
            content = turn.get("content", turn.get("value", ""))

            tokens = self.tokenizer.encode(content, add_special_tokens=False)

            if role in ["assistant", "gpt"]:
                # Compute loss on assistant responses
                all_tokens.extend(tokens)
                all_labels.extend(tokens)
            else:
                # Don't compute loss on user/system inputs
                all_tokens.extend(tokens)
                all_labels.extend([-100] * len(tokens))

        # Truncate
        all_tokens = all_tokens[:self.max_seq_len]
        all_labels = all_labels[:self.max_seq_len]

        return (
            torch.tensor(all_tokens, dtype=torch.long),
            torch.tensor(all_labels, dtype=torch.long),
        )


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    Pads sequences to same length within batch.
    """
    # Find max lengths
    max_len = max(item["input_ids"].shape[0] for item in batch)

    input_ids_list = []
    labels_list = []
    attention_mask_list = []
    images_list = []
    has_images = any("images" in item for item in batch)

    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad_len = max_len - seq_len

        # Pad input_ids
        input_ids = F.pad(item["input_ids"], (0, pad_len), value=0)
        input_ids_list.append(input_ids)

        # Pad labels
        labels = item.get("labels", item["input_ids"].clone())
        labels = F.pad(labels, (0, pad_len), value=-100)
        labels_list.append(labels)

        # Attention mask
        mask = torch.cat([
            torch.ones(seq_len, dtype=torch.long),
            torch.zeros(pad_len, dtype=torch.long),
        ])
        attention_mask_list.append(mask)

        if has_images:
            images_list.append(item.get("images", torch.zeros(3, 448, 448)))

    result = {
        "input_ids": torch.stack(input_ids_list),
        "labels": torch.stack(labels_list),
        "attention_mask": torch.stack(attention_mask_list),
    }

    if has_images:
        result["images"] = torch.stack(images_list)

    return result
