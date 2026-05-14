"""
Dataset and dataloader utilities for Adjoint Matching fine-tuning.

The paper uses:
- 40,000 training prompts sampled from a pool of 100,000 licensed text-image pairs
- 1,000 test prompts for evaluation
- 3 independent runs per method, each with different prompt subsets
- Prompts are text-only (images are not used during fine-tuning)

Fine-tuning is purely online: prompts are used to condition the model,
and images are generated on-the-fly during training.
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Prompt dataset
# ---------------------------------------------------------------------------

class PromptDataset(Dataset):
    """
    Dataset of text prompts for fine-tuning.

    During fine-tuning, only text prompts are needed (no images).
    Images are generated on-the-fly by the model.
    """

    def __init__(
        self,
        prompts: List[str],
        tokenizer=None,
        max_length: int = 77,
    ):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Dict:
        prompt = self.prompts[idx]
        item = {"prompt": prompt, "idx": idx}
        return item

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        prompts = [item["prompt"] for item in batch]
        indices = [item["idx"] for item in batch]
        return {"prompts": prompts, "indices": torch.tensor(indices)}


class PromptDatasetWithImages(Dataset):
    """
    Dataset of text-image pairs for DPO fine-tuning.
    Used when preference data (ranked pairs) is available.
    """

    def __init__(
        self,
        data: List[Dict],
        image_transform=None,
    ):
        """
        Args:
            data: List of dicts with keys "prompt", "image_win", "image_lose"
            image_transform: Transform to apply to images
        """
        self.data = data
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        result = {"prompt": item["prompt"]}

        if "image_win" in item and "image_lose" in item:
            img_win = item["image_win"]
            img_lose = item["image_lose"]
            if self.image_transform is not None:
                img_win = self.image_transform(img_win)
                img_lose = self.image_transform(img_lose)
            result["image_win"] = img_win
            result["image_lose"] = img_lose

        return result


# ---------------------------------------------------------------------------
# Prompt loading utilities
# ---------------------------------------------------------------------------

def load_prompts_from_file(filepath: str) -> List[str]:
    """Load prompts from a text file (one prompt per line) or JSON."""
    filepath = Path(filepath)
    if filepath.suffix == ".json":
        with open(filepath) as f:
            data = json.load(f)
        if isinstance(data, list):
            if isinstance(data[0], str):
                return data
            elif isinstance(data[0], dict):
                return [item.get("caption", item.get("prompt", "")) for item in data]
    else:
        with open(filepath) as f:
            return [line.strip() for line in f if line.strip()]


def sample_prompt_subset(
    all_prompts: List[str],
    num_prompts: int,
    seed: int = 42,
) -> List[str]:
    """
    Sample a subset of prompts (for reproducible experiments).
    Paper uses 40k training prompts sampled from 100k total.
    """
    rng = random.Random(seed)
    if num_prompts >= len(all_prompts):
        return all_prompts
    return rng.sample(all_prompts, num_prompts)


def create_train_eval_split(
    all_prompts: List[str],
    num_train: int = 40000,
    num_eval: int = 1000,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Create train/eval split from prompt pool.
    Paper uses 40k train + 1k eval, with different sets per run.
    """
    rng = random.Random(seed)
    shuffled = all_prompts.copy()
    rng.shuffle(shuffled)
    train_prompts = shuffled[:num_train]
    eval_prompts = shuffled[num_train:num_train + num_eval]
    return train_prompts, eval_prompts


# ---------------------------------------------------------------------------
# Text encoding utilities
# ---------------------------------------------------------------------------

class CLIPTextEncoder:
    """
    CLIP text encoder wrapper for generating text embeddings.
    Uses open_clip library (Ilharco et al., 2021).
    """

    def __init__(
        self,
        model_name: str = "ViT-H-14",
        pretrained: str = "laion2b_s32b_b79k",
        device: torch.device = None,
    ):
        import open_clip
        self.device = device or torch.device("cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, prompts: List[str]) -> torch.Tensor:
        """Encode text prompts to embeddings [B, seq_len, dim]."""
        tokens = self.tokenizer(prompts).to(self.device)
        text_features = self.model.encode_text(tokens)
        return text_features

    @torch.no_grad()
    def encode_with_null(
        self, prompts: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode prompts and null (empty) prompts for CFG."""
        text_emb = self.encode(prompts)
        null_prompts = [""] * len(prompts)
        null_emb = self.encode(null_prompts)
        return text_emb, null_emb


# ---------------------------------------------------------------------------
# Reward model wrappers
# ---------------------------------------------------------------------------

class ImageRewardModel:
    """
    ImageReward model wrapper (Xu et al., 2023).
    Reward function: r(x) = lambda * ImageReward(x)
    """

    def __init__(
        self,
        model_name: str = "ImageReward-v1.0",
        device: torch.device = None,
    ):
        import ImageReward as RM
        self.device = device or torch.device("cpu")
        self.model = RM.load(model_name, device=str(self.device))

    def score(
        self,
        images: torch.Tensor,
        prompts: List[str],
    ) -> torch.Tensor:
        """
        Compute ImageReward scores.

        Args:
            images: [B, 3, H, W] in [-1, 1] or [0, 1]
            prompts: List of text prompts

        Returns:
            Scores [B]
        """
        # Convert to PIL images for ImageReward
        from torchvision.transforms.functional import to_pil_image
        import numpy as np

        scores = []
        for i, (img, prompt) in enumerate(zip(images, prompts)):
            # Normalize to [0, 1] if needed
            if img.min() < 0:
                img = (img + 1.0) / 2.0
            img = img.clamp(0, 1)
            pil_img = to_pil_image(img.cpu())
            score = self.model.score(prompt, pil_img)
            scores.append(score)

        return torch.tensor(scores, device=self.device, dtype=torch.float32)

    def __call__(self, images: torch.Tensor, prompts: Optional[List[str]] = None) -> torch.Tensor:
        if prompts is None:
            prompts = [""] * images.shape[0]
        return self.score(images, prompts)


class DifferentiableRewardWrapper(torch.nn.Module):
    """
    Wrapper that makes a reward model differentiable w.r.t. input images.
    Used for gradient-based fine-tuning methods.
    """

    def __init__(self, reward_model, prompts_cache: Optional[List[str]] = None):
        super().__init__()
        self.reward_model = reward_model
        self.prompts_cache = prompts_cache

    def forward(self, images: torch.Tensor, prompts: Optional[List[str]] = None) -> torch.Tensor:
        if prompts is None:
            prompts = self.prompts_cache or [""] * images.shape[0]
        return self.reward_model(images, prompts)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloader(
    prompts: List[str],
    batch_size: int = 40,
    shuffle: bool = True,
    num_workers: int = 4,
    tokenizer=None,
) -> DataLoader:
    """Build DataLoader for prompt-based fine-tuning."""
    dataset = PromptDataset(prompts, tokenizer=tokenizer)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=PromptDataset.collate_fn,
        pin_memory=True,
        drop_last=True,
    )


def build_infinite_dataloader(
    prompts: List[str],
    batch_size: int = 40,
    seed: int = 42,
):
    """
    Infinite dataloader that cycles through prompts.
    Used for fine-tuning where we specify number of iterations, not epochs.
    """
    rng = random.Random(seed)
    prompts = prompts.copy()

    while True:
        rng.shuffle(prompts)
        for i in range(0, len(prompts) - batch_size + 1, batch_size):
            yield prompts[i:i + batch_size]


# ---------------------------------------------------------------------------
# Synthetic prompt generator (for testing without real data)
# ---------------------------------------------------------------------------

SYNTHETIC_PROMPTS = [
    "A beautiful landscape with mountains and a lake",
    "Portrait of a smiling person in natural light",
    "A colorful abstract painting with geometric shapes",
    "A cozy living room with warm lighting",
    "A plate of delicious food on a wooden table",
    "A city skyline at sunset with reflections in water",
    "A close-up of a flower with morning dew",
    "A group of people having a picnic in a park",
    "A futuristic cityscape with flying vehicles",
    "A serene beach scene with palm trees",
    "A winter landscape with snow-covered trees",
    "A vibrant market scene with colorful stalls",
    "A majestic waterfall in a tropical forest",
    "A vintage car on a winding mountain road",
    "A cozy cafe interior with books and plants",
    "A dramatic storm over the ocean",
    "A child playing in autumn leaves",
    "A modern kitchen with stainless steel appliances",
    "A field of sunflowers under a blue sky",
    "A medieval castle on a hilltop at dusk",
]


def get_synthetic_prompts(n: int = 1000, seed: int = 42) -> List[str]:
    """Generate synthetic prompts for testing."""
    rng = random.Random(seed)
    prompts = []
    while len(prompts) < n:
        prompts.extend(SYNTHETIC_PROMPTS)
    rng.shuffle(prompts)
    return prompts[:n]
