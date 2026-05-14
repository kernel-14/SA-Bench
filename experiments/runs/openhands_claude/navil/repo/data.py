"""
Data loading and preprocessing for NaViL.

Pre-training data (Stage 1):
  - Web-scale image-text pairs: LAION-2B, COYO-700M, Wukong, SA-1B
  - 300M directly sampled + 200M with synthesized captions (InternVL-8B)
  - Total: 500M image-text pairs

Fine-tuning data (Stage 2):
  - 68M high-quality multimodal data (image captioning, QA, OCR, charts, docs)
  - Pure language data from InternLM2.5

Supports:
  - WebDataset format (tar shards) for large-scale pre-training
  - JSON/JSONL format for fine-tuning
  - Dynamic resolution: images padded to multiples of patch_size=16
  - Conversation format compatible with InternLM2 / Qwen3 tokenizers
"""

import io
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision import transforms

try:
    import webdataset as wds
    HAS_WDS = True
except ImportError:
    HAS_WDS = False


# ── Image preprocessing ───────────────────────────────────────────────────────

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def build_image_transform(
    image_size: int = 448,
    is_train: bool = True,
    patch_size: int = 16,
) -> Callable:
    """
    Build image transform pipeline.
    Images are padded to multiples of patch_size (paper: "padded to ensure
    its length and width are multiples of 32" — 32 = 2 * patch_size for
    pixel shuffle factor 2).
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.5, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(
                image_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])


def dynamic_preprocess(
    image: Image.Image,
    patch_size: int = 16,
    pixel_shuffle_factor: int = 2,
    max_patches: int = 4096,
    min_size: int = 32,
) -> torch.Tensor:
    """
    Resize image to fit within max_patches while preserving aspect ratio.
    Pad to multiples of (patch_size * pixel_shuffle_factor).

    The paper states: "input images are first padded to ensure its length
    and width are multiples of 32" (32 = patch_size * pixel_shuffle_factor = 16 * 2).
    """
    stride = patch_size * pixel_shuffle_factor   # 32

    W, H = image.size
    # Compute scale to fit within max_patches
    num_patches = (H // patch_size) * (W // patch_size)
    if num_patches > max_patches:
        scale = math.sqrt(max_patches / num_patches)
        H = max(min_size, int(H * scale))
        W = max(min_size, int(W * scale))
        image = image.resize((W, H), Image.BICUBIC)

    # Pad to multiples of stride
    W, H = image.size
    pad_h = (stride - H % stride) % stride
    pad_w = (stride - W % stride) % stride

    if pad_h > 0 or pad_w > 0:
        new_img = Image.new(image.mode, (W + pad_w, H + pad_h), (0, 0, 0))
        new_img.paste(image, (0, 0))
        image = new_img

    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    tensor = transforms.ToTensor()(image)
    return normalize(tensor)


# ── Conversation / tokenization helpers ───────────────────────────────────────

SPECIAL_TOKENS = [
    "<begin_of_image>",
    "<end_of_image>",
    "<end_of_line>",
    "<end_of_scale>",
    "<image_patch>",
]

SYSTEM_PROMPT = (
    "You are a helpful multimodal assistant. "
    "You can understand images and answer questions about them."
)


def build_conversation_prompt(
    messages: List[Dict[str, str]],
    tokenizer,
    add_generation_prompt: bool = False,
) -> str:
    """
    Build a conversation string using the tokenizer's chat template.
    Falls back to a simple format if no chat template is available.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    # Fallback format
    text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            text += f"<|system|>\n{content}\n"
        elif role == "user":
            text += f"<|user|>\n{content}\n"
        elif role == "assistant":
            text += f"<|assistant|>\n{content}\n"
    if add_generation_prompt:
        text += "<|assistant|>\n"
    return text


def tokenize_multimodal(
    text: str,
    tokenizer,
    images: Optional[List[Image.Image]],
    max_length: int = 16384,
    patch_size: int = 16,
    pixel_shuffle_factor: int = 2,
    max_patches: int = 4096,
) -> Dict[str, Any]:
    """
    Tokenize a multimodal sample.

    Image placeholders in text (e.g. "<image>") are replaced with the
    appropriate number of <image_patch> tokens based on the actual image size,
    bracketed by <begin_of_image> and <end_of_image>.
    """
    boi = "<begin_of_image>"
    eoi = "<end_of_image>"
    eol = "<end_of_line>"
    eos_scale = "<end_of_scale>"
    patch_tok = "<image_patch>"

    processed_images = []
    image_token_counts = []

    if images:
        for img in images:
            tensor = dynamic_preprocess(
                img, patch_size, pixel_shuffle_factor, max_patches
            )
            processed_images.append(tensor)
            _, H, W = tensor.shape
            num_h = H // patch_size // pixel_shuffle_factor
            num_w = W // patch_size // pixel_shuffle_factor
            image_token_counts.append((num_h, num_w))

    # Replace <image> placeholders with actual token sequences
    img_idx = 0
    new_text = ""
    parts = text.split("<image>")
    for i, part in enumerate(parts):
        new_text += part
        if i < len(parts) - 1 and img_idx < len(image_token_counts):
            num_h, num_w = image_token_counts[img_idx]
            # Build image token string: boi + (patch_tok + eol) * rows + eos_scale + eoi
            img_str = boi
            for row in range(num_h):
                img_str += patch_tok * num_w + eol
            img_str += eos_scale + eoi
            new_text += img_str
            img_idx += 1

    encoding = tokenizer(
        new_text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=False,
    )

    return {
        "input_ids": encoding["input_ids"][0],
        "attention_mask": encoding["attention_mask"][0],
        "images": processed_images,
    }


# ── Pre-training Dataset (WebDataset) ─────────────────────────────────────────

class WebDatasetPretraining(IterableDataset):
    """
    WebDataset-based iterable dataset for large-scale pre-training.
    Supports LAION-2B, COYO-700M, Wukong, SA-1B formats.

    Each shard contains .jpg/.png + .txt/.json pairs.
    """

    def __init__(
        self,
        shard_urls: List[str],
        tokenizer,
        max_length: int = 16384,
        patch_size: int = 16,
        pixel_shuffle_factor: int = 2,
        max_patches: int = 4096,
        shuffle_buffer: int = 10000,
        is_train: bool = True,
    ):
        if not HAS_WDS:
            raise ImportError("webdataset is required for pre-training data loading")

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.patch_size = patch_size
        self.pixel_shuffle_factor = pixel_shuffle_factor
        self.max_patches = max_patches

        self.dataset = (
            wds.WebDataset(shard_urls, resampled=is_train)
            .shuffle(shuffle_buffer)
            .decode("pil")
            .to_tuple("jpg;png", "txt;json")
            .map(self._process_sample)
            .select(lambda x: x is not None)
        )

    def _process_sample(self, sample: Tuple) -> Optional[Dict[str, Any]]:
        image, caption = sample
        if isinstance(image, bytes):
            try:
                image = Image.open(io.BytesIO(image)).convert("RGB")
            except Exception:
                return None
        if not isinstance(image, Image.Image):
            return None

        if isinstance(caption, bytes):
            caption = caption.decode("utf-8", errors="ignore")
        if isinstance(caption, dict):
            caption = caption.get("caption", caption.get("text", ""))

        # Simple image captioning format for pre-training
        text = f"<image>{caption}"

        try:
            result = tokenize_multimodal(
                text=text,
                tokenizer=self.tokenizer,
                images=[image],
                max_length=self.max_length,
                patch_size=self.patch_size,
                pixel_shuffle_factor=self.pixel_shuffle_factor,
                max_patches=self.max_patches,
            )
        except Exception:
            return None

        # Labels: -100 for image tokens, actual token ids for text
        labels = result["input_ids"].clone()
        # Mask image patch tokens in labels
        patch_id = self.tokenizer.convert_tokens_to_ids("<image_patch>")
        if patch_id is not None and patch_id != self.tokenizer.unk_token_id:
            labels[labels == patch_id] = -100

        result["labels"] = labels
        return result

    def __iter__(self):
        return iter(self.dataset)


# ── Fine-tuning Dataset (JSON/JSONL) ──────────────────────────────────────────

class MultimodalSFTDataset(Dataset):
    """
    JSON/JSONL dataset for supervised fine-tuning.

    Expected format:
    {
        "id": "sample_001",
        "image": "path/to/image.jpg",   # optional
        "conversations": [
            {"from": "human", "value": "What is in this image? <image>"},
            {"from": "gpt",   "value": "The image shows ..."}
        ]
    }
    """

    def __init__(
        self,
        data_path: str,
        image_root: str,
        tokenizer,
        max_length: int = 16384,
        patch_size: int = 16,
        pixel_shuffle_factor: int = 2,
        max_patches: int = 4096,
        is_train: bool = True,
    ):
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.patch_size = patch_size
        self.pixel_shuffle_factor = pixel_shuffle_factor
        self.max_patches = max_patches

        self.samples = self._load_data(data_path)

    def _load_data(self, path: str) -> List[Dict]:
        samples = []
        if path.endswith(".jsonl"):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        samples.append(json.loads(line))
        else:
            with open(path) as f:
                data = json.load(f)
            samples = data if isinstance(data, list) else [data]
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        sample = self.samples[idx]

        # Load image if present
        images = []
        if "image" in sample and sample["image"]:
            img_path = os.path.join(self.image_root, sample["image"])
            try:
                img = Image.open(img_path).convert("RGB")
                images.append(img)
            except Exception:
                images = []

        # Build conversation
        conversations = sample.get("conversations", [])
        messages = []
        for turn in conversations:
            role = "user" if turn["from"] in ("human", "user") else "assistant"
            messages.append({"role": role, "content": turn["value"]})

        text = build_conversation_prompt(
            messages, self.tokenizer, add_generation_prompt=False
        )

        try:
            result = tokenize_multimodal(
                text=text,
                tokenizer=self.tokenizer,
                images=images if images else None,
                max_length=self.max_length,
                patch_size=self.patch_size,
                pixel_shuffle_factor=self.pixel_shuffle_factor,
                max_patches=self.max_patches,
            )
        except Exception:
            return None

        # Labels: mask user turns and image tokens with -100
        labels = result["input_ids"].clone()
        patch_id = self.tokenizer.convert_tokens_to_ids("<image_patch>")
        if patch_id is not None and patch_id != self.tokenizer.unk_token_id:
            labels[labels == patch_id] = -100

        # Mask user turns (everything before the first assistant response)
        self._mask_user_turns(labels, result["input_ids"])

        result["labels"] = labels
        return result

    def _mask_user_turns(self, labels: torch.Tensor, input_ids: torch.Tensor):
        """Set labels to -100 for all user turn tokens."""
        # Find assistant response boundaries using tokenizer's special tokens
        # This is a simplified version; production code would use the tokenizer's
        # chat template to identify exact boundaries
        pass   # Labels are set to -100 for image patches; full masking requires
               # tokenizer-specific boundary detection


# ── Pure language dataset ──────────────────────────────────────────────────────

class PureLanguageDataset(Dataset):
    """
    Pure language dataset for maintaining NLP capabilities during pre-training.
    Used in Stage 1.2 alongside multimodal data.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 16384,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._load_data(data_path)

    def _load_data(self, path: str) -> List[str]:
        texts = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        texts.append(obj.get("text", obj.get("content", "")))
                    except json.JSONDecodeError:
                        texts.append(line)
        return texts

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.samples[idx]
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )
        input_ids = encoding["input_ids"][0]
        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"][0],
            "labels": input_ids.clone(),
            "images": [],
        }


# ── Collation ─────────────────────────────────────────────────────────────────

def collate_fn(
    batch: List[Optional[Dict[str, Any]]],
    pad_token_id: int = 0,
    max_length: int = 16384,
) -> Dict[str, Any]:
    """
    Collate a batch of samples with variable-length sequences.
    Pads input_ids, attention_mask, and labels to the longest sequence.
    Images are kept as a list of lists.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return {}

    max_len = min(max(b["input_ids"].shape[0] for b in batch), max_length)

    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    images_list = []

    for b in batch:
        ids = b["input_ids"][:max_len]
        mask = b["attention_mask"][:max_len]
        lbl = b["labels"][:max_len]

        pad_len = max_len - ids.shape[0]
        if pad_len > 0:
            ids  = F.pad(ids,  (0, pad_len), value=pad_token_id)
            mask = F.pad(mask, (0, pad_len), value=0)
            lbl  = F.pad(lbl,  (0, pad_len), value=-100)

        input_ids_list.append(ids)
        attention_mask_list.append(mask)
        labels_list.append(lbl)
        images_list.append(b.get("images", []))

    return {
        "input_ids":      torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "labels":         torch.stack(labels_list),
        "images":         images_list,
    }


# ── DataLoader factory ────────────────────────────────────────────────────────

def build_pretrain_dataloader(
    shard_urls: List[str],
    tokenizer,
    batch_size: int,
    num_workers: int = 8,
    max_length: int = 16384,
    max_patches: int = 4096,
) -> DataLoader:
    dataset = WebDatasetPretraining(
        shard_urls=shard_urls,
        tokenizer=tokenizer,
        max_length=max_length,
        max_patches=max_patches,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn(
            b,
            pad_token_id=tokenizer.pad_token_id or 0,
            max_length=max_length,
        ),
        pin_memory=True,
    )


def build_sft_dataloader(
    data_path: str,
    image_root: str,
    tokenizer,
    batch_size: int,
    num_workers: int = 8,
    max_length: int = 16384,
    max_patches: int = 24576,
    shuffle: bool = True,
) -> DataLoader:
    dataset = MultimodalSFTDataset(
        data_path=data_path,
        image_root=image_root,
        tokenizer=tokenizer,
        max_length=max_length,
        max_patches=max_patches,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn(
            b,
            pad_token_id=tokenizer.pad_token_id or 0,
            max_length=max_length,
        ),
        pin_memory=True,
    )


# ── Tokenizer setup ───────────────────────────────────────────────────────────

def setup_tokenizer(model_name_or_path: str):
    """
    Load tokenizer and add NaViL special tokens.
    Returns (tokenizer, special_token_ids).
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )

    # Add special tokens
    new_tokens = [t for t in SPECIAL_TOKENS if t not in tokenizer.get_vocab()]
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

    special_token_ids = {
        "begin_of_image": tokenizer.convert_tokens_to_ids("<begin_of_image>"),
        "end_of_image":   tokenizer.convert_tokens_to_ids("<end_of_image>"),
        "end_of_line":    tokenizer.convert_tokens_to_ids("<end_of_line>"),
        "end_of_scale":   tokenizer.convert_tokens_to_ids("<end_of_scale>"),
        "image_patch":    tokenizer.convert_tokens_to_ids("<image_patch>"),
    }

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, special_token_ids


# ── Validation dataset for loss monitoring ────────────────────────────────────

class ValidationDataset(Dataset):
    """
    Held-out subset of multimodal data for validation loss monitoring
    (used in Sec. 3.2 ablation studies).
    """

    def __init__(
        self,
        data_path: str,
        image_root: str,
        tokenizer,
        max_length: int = 4096,
        max_patches: int = 1024,
        max_samples: int = 5000,
    ):
        self.inner = MultimodalSFTDataset(
            data_path=data_path,
            image_root=image_root,
            tokenizer=tokenizer,
            max_length=max_length,
            max_patches=max_patches,
            is_train=False,
        )
        self.max_samples = min(max_samples, len(self.inner))

    def __len__(self) -> int:
        return self.max_samples

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        return self.inner[idx]
