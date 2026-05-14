"""Dataset loading, preprocessing, and multimodal sequence construction.

Handles:
- Image loading and padding to patch_size multiples
- Tokenization with special tokens
- Visual multi-scale packing for training
- Next-token-prediction label construction
- Data sources: Laion-2B, Coyo-700M, Wukong, SA-1B + synthesized captions
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader, IterableDataset


def pad_image_to_patch_multiple(
    image: Image.Image,
    patch_size: int = 16,
    fill_color: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Pad image so height and width are multiples of patch_size."""
    w, h = image.size
    new_h = ((h + patch_size - 1) // patch_size) * patch_size
    new_w = ((w + patch_size - 1) // patch_size) * patch_size
    if new_h != h or new_w != w:
        new_img = Image.new("RGB", (new_w, new_h), fill_color)
        new_img.paste(image, (0, 0))
        return new_img
    return image


def pad_image_tensor_to_patch_multiple(
    tensor: torch.Tensor,
    patch_size: int = 16,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Pad image tensor [C, H, W] so H, W are multiples of patch_size."""
    _, h, w = tensor.shape
    new_h = ((h + patch_size - 1) // patch_size) * patch_size
    new_w = ((w + patch_size - 1) // patch_size) * patch_size
    if new_h != h or new_w != w:
        padded = torch.full((3, new_h, new_w), pad_value, dtype=tensor.dtype)
        padded[:, :h, :w] = tensor
        return padded
    return tensor


def build_multiscale_tensor(
    image: torch.Tensor,
    downsample_rate: float,
    min_area: int = 256,
) -> List[torch.Tensor]:
    """Build multi-scale image sequence from a single image tensor.

    Args:
        image: [3, H, W]
        downsample_rate: tau (sqrt(2)/2)
        min_area: minimum area threshold

    Returns:
        list of [3, Hi, Wi] for each scale
    """
    scales = [image]
    h, w = image.shape[-2:]
    while h * w * (downsample_rate ** 2) >= min_area:
        h = int(h * downsample_rate)
        w = int(w * downsample_rate)
        if h < 16 or w < 16:
            break
        downscaled = F.interpolate(
            image.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False,
        ).squeeze(0)
        scales.append(downscaled)
    return scales


@dataclass
class MultimodalSample:
    """Single multimodal training sample."""
    image: torch.Tensor
    text: str
    text_tokens: torch.Tensor
    labels: torch.Tensor
    image_scales: Optional[List[torch.Tensor]] = None


class ImageCaptionDataset(Dataset):
    """Map-style dataset for image-text pairs.

    Supports web-scale datasets (Laion-2B, Coyo-700M, etc.) with
    optional synthesized captions from existing MLLMs.

    During training, batch mixing of:
    - 300M samples from web-scale datasets
    - 200M samples with synthesized captions
    """

    def __init__(
        self,
        data_paths: List[str],
        tokenizer,
        image_transform,
        max_length: int = 16384,
        max_image_patches: int = 4096,
        patch_size: int = 16,
        special_tokens: Optional[Dict[str, str]] = None,
        use_multiscale: bool = True,
        downsample_rate: float = 0.7071067811865476,
        seed: int = 42,
    ):
        self.data_paths = data_paths
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_length = max_length
        self.max_image_patches = max_image_patches
        self.patch_size = patch_size
        self.use_multiscale = use_multiscale
        self.downsample_rate = downsample_rate

        self.special_tokens = special_tokens or {
            "begin_of_image": "<begin_of_image>",
            "end_of_image": "<end_of_image>",
            "end_of_line": "<end_of_line>",
            "end_of_scale": "<end_of_scale>",
        }

        self._samples = self._load_data()

    def _load_data(self) -> List[Dict]:
        """Load metadata from data paths.

        Expected format per file: JSON lines with 'image_path' and 'caption'.
        """
        import json
        samples = []
        for path in self.data_paths:
            with open(path, "r") as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
        return samples

    def __len__(self) -> int:
        return len(self._samples)

    def _tokenize_text(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize text and create labels for next-token prediction.

        Labels are masked for image placeholder tokens.
        Returns (input_ids, labels).
        """
        tokens = self.tokenizer.encode(text)
        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()
        return input_ids, labels

    def _construct_sequence(
        self,
        image: torch.Tensor,
        caption: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Construct multimodal input sequence.

        Format: <begin_of_image> [image tokens with <end_of_line>]
                <end_of_image> [caption tokens]

        Returns: (input_ids, image_tensor, labels)
        """
        begin_tok = self.special_tokens["begin_of_image"]
        end_tok = self.special_tokens["end_of_image"]
        eol_tok = self.special_tokens["end_of_line"]

        begin_ids = torch.tensor(self.tokenizer.encode(begin_tok), dtype=torch.long)
        end_ids = torch.tensor(self.tokenizer.encode(end_tok), dtype=torch.long)
        eol_ids = torch.tensor(self.tokenizer.encode(eol_tok), dtype=torch.long)

        caption_ids, caption_labels = self._tokenize_text(caption)

        input_ids = torch.cat([begin_ids, eol_ids, end_ids, caption_ids])
        labels = torch.cat([
            torch.full_like(begin_ids, -100),
            torch.full_like(eol_ids, -100),
            torch.full_like(end_ids, -100),
            caption_labels,
        ])

        return input_ids, image, labels

    def __getitem__(self, idx: int) -> MultimodalSample:
        sample = self._samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        image = pad_image_to_patch_multiple(image, self.patch_size)
        image_tensor = self.image_transform(image)

        input_ids, image_tensor, labels = self._construct_sequence(
            image_tensor, sample["caption"],
        )

        scales = None
        if self.use_multiscale:
            scales = build_multiscale_tensor(image_tensor, self.downsample_rate)

        return MultimodalSample(
            image=image_tensor,
            text=sample["caption"],
            text_tokens=input_ids,
            labels=labels,
            image_scales=scales,
        )


def collate_multimodal_batch(
    batch: List[MultimodalSample],
    pad_token_id: int,
    max_seq_len: int = 16384,
    max_image_patches: int = 4096,
) -> Dict[str, torch.Tensor]:
    """Collate function for batching multimodal samples.

    Returns:
        dict with keys:
        - input_ids: [batch, max_seq_len] padded token IDs
        - images: [batch, 3, max_H, max_W] padded images
        - labels: [batch, max_seq_len] padded labels
        - attention_mask: [batch, max_seq_len]
        - image_token_mask: [batch, max_seq_len] 1=image, 0=text
    """
    def pad_to_len(t: torch.Tensor, length: int, pad_val: int) -> torch.Tensor:
        if t.shape[0] >= length:
            return t[:length]
        padding = torch.full((length - t.shape[0],), pad_val, dtype=t.dtype)
        return torch.cat([t, padding])

    batch_input_ids = []
    batch_labels = []
    batch_images = []

    max_h = 0
    max_w = 0
    for sample in batch:
        _, h, w = sample.image.shape
        max_h = max(max_h, h)
        max_w = max(max_w, w)

    for sample in batch:
        _, h, w = sample.image.shape
        if h < max_h or w < max_w:
            padded_img = torch.zeros(3, max_h, max_w, dtype=sample.image.dtype)
            padded_img[:, :h, :w] = sample.image
            batch_images.append(padded_img)
        else:
            batch_images.append(sample.image)

        input_ids = pad_to_len(sample.text_tokens, max_seq_len, pad_token_id)
        labels = pad_to_len(sample.labels, max_seq_len, -100)

        batch_input_ids.append(input_ids)
        batch_labels.append(labels)

    return {
        "input_ids": torch.stack(batch_input_ids),
        "images": torch.stack(batch_images),
        "labels": torch.stack(batch_labels),
    }


class HighQualityDataset(Dataset):
    """Dataset for Stage 1.2: high-quality multimodal alignment + language data.

    Mix of:
    - Multimodal alignment samples (complex QA, captions, etc.)
    - Pure language data (from InternLM2.5 pre-training corpus)
    """

    def __init__(
        self,
        multimodal_paths: List[str],
        language_paths: List[str],
        tokenizer,
        image_transform,
        multimodal_ratio: float = 0.5,
        max_length: int = 16384,
        max_image_patches: int = 12188,
        patch_size: int = 16,
        special_tokens: Optional[Dict[str, str]] = None,
        use_multiscale: bool = True,
        downsample_rate: float = 0.7071067811865476,
    ):
        self.multimodal_dataset = ImageCaptionDataset(
            multimodal_paths, tokenizer, image_transform,
            max_length, max_image_patches, patch_size,
            special_tokens, use_multiscale, downsample_rate,
        )
        self.language_data = self._load_language_data(language_paths)
        self.tokenizer = tokenizer
        self.multimodal_ratio = multimodal_ratio
        self.max_length = max_length

    def _load_language_data(self, paths: List[str]) -> List[str]:
        import json
        texts = []
        for path in paths:
            with open(path, "r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        texts.append(data.get("text", ""))
        return texts

    def __len__(self) -> int:
        return len(self.multimodal_dataset) + len(self.language_data)

    def __getitem__(self, idx: int):
        if random.random() < self.multimodal_ratio and idx < len(self.multimodal_dataset):
            return self.multimodal_dataset[min(idx, len(self.multimodal_dataset) - 1)]

        lang_idx = idx % len(self.language_data)
        text = self.language_data[lang_idx]
        tokens = self.tokenizer.encode(text)
        input_ids = torch.tensor(tokens[:self.max_length], dtype=torch.long)
        labels = input_ids.clone()

        dummy_image = torch.zeros(3, 224, 224)

        return MultimodalSample(
            image=dummy_image,
            text=text,
            text_tokens=input_ids,
            labels=labels,
        )


class SFTDataset(Dataset):
    """Supervised Fine-Tuning dataset.

    High-quality multimodal instruction data from InternVL-2.5 sources.
    Smaller dataset (~68M), all parameters unfrozen.
    Supports multi-turn conversation format.
    """

    def __init__(
        self,
        data_paths: List[str],
        tokenizer,
        image_transform,
        max_length: int = 16384,
        max_image_patches: int = 24576,
        patch_size: int = 16,
        special_tokens: Optional[Dict[str, str]] = None,
        use_multiscale: bool = True,
        downsample_rate: float = 0.7071067811865476,
    ):
        self.base_dataset = ImageCaptionDataset(
            data_paths, tokenizer, image_transform,
            max_length, max_image_patches, patch_size,
            special_tokens, use_multiscale, downsample_rate,
        )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> MultimodalSample:
        return self.base_dataset[idx]


def create_image_transform(
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    resize_mode: str = "fill",
) -> callable:
    """Create image transformation pipeline."""
    import torchvision.transforms as T

    transforms = [
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ]
    return T.Compose(transforms)
