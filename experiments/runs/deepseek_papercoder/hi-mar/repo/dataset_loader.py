"""
dataset_loader.py

DatasetLoader – constructs PyTorch DataLoader objects for ImageNet (class‑conditional)
and MS‑COCO (text‑to‑image) tasks, aligned with the Hi‑MAR training pipeline.

Provides:
  - ImageNetDataset : yields high‑res & low‑res RGB tensors and a class label.
  - COCODataset      : yields high‑res & low‑res RGB tensors and a pre‑computed
                       CLIP text embedding (one of five captions chosen randomly).
  - build_coco_embeddings : utility to pre‑compute and cache CLIP embeddings for
                            all COCO captions.  Must be executed once before training.
  - DatasetLoader    : factory that creates train / validation loaders.

The VAE tokenization is NOT performed inside the loader; pixel tensors in [-1, 1]
are returned, and the caller (trainer / inference) applies VAETokenizer.encode.
"""

from __future__ import annotations

import logging
import os
import pickle
import random
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.utils.data
import torchvision.transforms.functional as F_tv
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# Project imports
from config import DataConfig
from vae_tokenizer import VAETokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  ImageNet Dataset
# ---------------------------------------------------------------------------

class ImageNetDataset(Dataset):
    """
    ImageNet ILSVRC2012 class‑conditional dataset.

    Returns a dict:
        - 'high_res' : FloatTensor (3, 256, 256) in [-1, 1]
        - 'low_res'  : FloatTensor (3, 128, 128) in [-1, 1]
        - 'class_id' : LongTensor of the class label
    """

    def __init__(self, root: str, split: str = "train") -> None:
        """
        Args:
            root:  Root directory of ImageNet containing 'train/' and 'val/'
                   subdirectories.
            split: 'train' or 'val'.
        """
        super().__init__()
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split}")

        self.root = root
        self.split = split
        self.data_dir = os.path.join(root, split)

        # Gather all image paths and corresponding integer labels.
        classes = sorted(os.listdir(self.data_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.samples: List[Tuple[str, int]] = []
        for cls in classes:
            cls_dir = os.path.join(self.data_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append(
                        (os.path.join(cls_dir, fname), self.class_to_idx[cls])
                    )
        logger.info(
            "ImageNet %s dataset: %d samples", split, len(self.samples)
        )

        # Transform parameters.
        if split == "train":
            self.crop_scale = (0.8, 1.0)   # RandomResizedCrop scale
            self.hflip = True
        else:
            self.crop_scale = (1.0, 1.0)   # no cropping (just resize)
            self.hflip = False

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        path, label = self.samples[idx]
        pil_img = Image.open(path).convert("RGB")

        if self.split == "train":
            # RandomResizedCrop with shared parameters for both resolutions.
            # Using torchvision's get_params then functional operations ensures
            # the same crop area is used for high‑res and low‑res.
            i, j, h, w = torchvision.transforms.RandomResizedCrop.get_params(
                pil_img, scale=self.crop_scale, ratio=(3.0 / 4.0, 4.0 / 3.0)
            )
            cropped = F_tv.crop(pil_img, i, j, h, w)
            if self.hflip and random.random() < 0.5:
                cropped = F_tv.hflip(cropped)

            high_res = F_tv.resize(
                cropped,
                [256, 256],
                interpolation=F_tv.InterpolationMode.BICUBIC,
            )
            low_res = F_tv.resize(
                cropped,
                [128, 128],
                interpolation=F_tv.InterpolationMode.BICUBIC,
            )
        else:  # validation
            # Resize to a square without cropping.
            high_res = F_tv.resize(
                pil_img, 256, interpolation=F_tv.InterpolationMode.BICUBIC
            )
            low_res = F_tv.resize(
                pil_img, 128, interpolation=F_tv.InterpolationMode.BICUBIC
            )

        # Convert to tensor and normalise to [-1, 1].
        to_tensor = torchvision.transforms.ToTensor()
        normalize = torchvision.transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )

        high_tensor = normalize(to_tensor(high_res))
        low_tensor = normalize(to_tensor(low_res))
        class_tensor = torch.tensor(label, dtype=torch.long)

        return {
            "high_res": high_tensor,
            "low_res": low_tensor,
            "class_id": class_tensor,
        }


# ---------------------------------------------------------------------------
#  COCO Dataset
# ---------------------------------------------------------------------------

class COCODataset(Dataset):
    """
    MS‑COCO text‑to‑image dataset.

    Expects pre‑computed CLIP text embeddings (one per caption) stored in a
    pickle file.  The file must be generated by ``build_coco_embeddings``.

    Returns a dict:
        - 'high_res' : FloatTensor (3, 256, 256) in [-1, 1]
        - 'low_res'  : FloatTensor (3, 128, 128) in [-1, 1]
        - 'text_emb' : FloatTensor (77, 768), CLIP embedding of one randomly
                       chosen caption.
    """

    def __init__(
        self,
        root: str,
        ann_file: str,
        embeddings_path: str,
        split: str = "train",
    ) -> None:
        """
        Args:
            root:             COCO dataset root (e.g., './data/coco'). The images
                              are expected under ``root/{split}2017/``.
            ann_file:         Path to the COCO annotation JSON (e.g.,
                              'captions_train2017.json').
            embeddings_path:  Path to the pickled CLIP embeddings (output of
                              ``build_coco_embeddings``).
            split:            'train' or 'val'.
        """
        super().__init__()
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split}")

        self.root = root
        self.split = split
        self.ann_file = ann_file
        self.image_dir = os.path.join(root, f"{split}2017")

        # Load COCO caption annotations.
        logger.info("Loading COCO annotations from %s", ann_file)
        with open(ann_file, "r") as f:
            coco_data = json.load(f)

        # Build mapping: image_id → filename.
        self.id_to_filename: Dict[int, str] = {
            img["id"]: img["file_name"] for img in coco_data["images"]
        }

        # Build mapping: image_id → list of captions (must have exactly 5).
        id_to_captions: Dict[int, List[str]] = {}
        for ann in coco_data["annotations"]:
            img_id = ann["image_id"]
            caption = ann["caption"]
            id_to_captions.setdefault(img_id, []).append(caption)

        # Keep only images with exactly 5 captions.
        self.image_ids: List[int] = [
            img_id
            for img_id, caps in id_to_captions.items()
            if len(caps) == 5
        ]
        self.id_to_captions = {
            img_id: caps
            for img_id, caps in id_to_captions.items()
            if img_id in self.image_ids
        }

        # Load pre‑computed CLIP embeddings.
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(
                f"CLIP embeddings file not found at '{embeddings_path}'. "
                "Run 'build_coco_embeddings' to generate it."
            )
        with open(embeddings_path, "rb") as f:
            self.embeddings: Dict[int, List[Tensor]] = pickle.load(f)

        # Verify that every image_id has embeddings.
        missing_ids = set(self.image_ids) - set(self.embeddings.keys())
        if missing_ids:
            raise KeyError(
                f"Missing CLIP embeddings for {len(missing_ids)} images, "
                f"e.g. {next(iter(missing_ids))}."
            )

        logger.info(
            "COCO %s dataset: %d samples", split, len(self.image_ids)
        )

        # Transform parameters.
        if split == "train":
            self.crop_scale = (0.8, 1.0)
            self.hflip = True
        else:
            self.crop_scale = (1.0, 1.0)
            self.hflip = False

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        img_id = self.image_ids[idx]
        file_name = self.id_to_filename[img_id]
        pil_img = Image.open(os.path.join(self.image_dir, file_name)).convert("RGB")

        if self.split == "train":
            i, j, h, w = torchvision.transforms.RandomResizedCrop.get_params(
                pil_img, scale=self.crop_scale, ratio=(3.0 / 4.0, 4.0 / 3.0)
            )
            cropped = F_tv.crop(pil_img, i, j, h, w)
            if self.hflip and random.random() < 0.5:
                cropped = F_tv.hflip(cropped)
            high_res = F_tv.resize(
                cropped,
                [256, 256],
                interpolation=F_tv.InterpolationMode.BICUBIC,
            )
            low_res = F_tv.resize(
                cropped,
                [128, 128],
                interpolation=F_tv.InterpolationMode.BICUBIC,
            )
        else:
            high_res = F_tv.resize(
                pil_img, 256, interpolation=F_tv.InterpolationMode.BICUBIC
            )
            low_res = F_tv.resize(
                pil_img, 128, interpolation=F_tv.InterpolationMode.BICUBIC
            )

        to_tensor = torchvision.transforms.ToTensor()
        normalize = torchvision.transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )
        high_tensor = normalize(to_tensor(high_res))
        low_tensor = normalize(to_tensor(low_res))

        # Randomly select one of the 5 caption embeddings.
        caption_idx = random.randint(0, 4)
        text_emb = self.embeddings[img_id][caption_idx]  # shape [77, 768]

        return {
            "high_res": high_tensor,
            "low_res": low_tensor,
            "text_emb": text_emb,
        }


# ---------------------------------------------------------------------------
#  Helper: pre‑compute CLIP embeddings for COCO captions
# ---------------------------------------------------------------------------

def build_coco_embeddings(
    coco_dataset: COCODataset,
    clip_model_name: str,
    save_path: str,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 32,
) -> None:
    """
    Pre‑compute CLIP text embeddings for all captions of a COCO dataset
    and save them as a pickle file.

    Args:
        coco_dataset:    An (uninitialised) COCODataset instance.  Only its
                         ``id_to_captions`` is used; the constructor must have
                         been called to populate it.
        clip_model_name: HuggingFace model identifier, e.g.
                         'openai/clip-vit-large-patch14'.
        save_path:       Destination path for the pickle file.
        device:          Torch device.
        batch_size:      Batch size for CLIP encoding.
    """
    from tqdm import tqdm
    from transformers import CLIPTextModel, CLIPTokenizer

    logger.info("Loading CLIP model: %s", clip_model_name)
    model = CLIPTextModel.from_pretrained(clip_model_name).to(device)
    model.eval()
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)

    # Collect all unique captions: we need one embedding per caption (5 per image).
    img_captions: Dict[int, List[str]] = coco_dataset.id_to_captions
    all_image_ids = list(img_captions.keys())
    all_captions = []         # flat list
    caption_index = []        # (img_id_idx, caption_idx)

    for img_idx, img_id in enumerate(all_image_ids):
        caps = img_captions[img_id]
        for ci, cap in enumerate(caps):
            all_captions.append(cap)
            caption_index.append((img_id, ci))

    # Batch encode.
    embeddings_dict: Dict[int, List[Optional[Tensor]]] = {}
    for i in tqdm(range(0, len(all_captions), batch_size), desc="Encoding COCO captions"):
        batch_texts = all_captions[i : i + batch_size]
        tokens = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=77,
        ).to(device)

        with torch.no_grad():
            # CLIP text encoder outputs last_hidden_state of shape (B, 77, 768)
            emb = model(**tokens).last_hidden_state.cpu()  # (B, 77, 768)

        for j in range(emb.shape[0]):
            img_id, ci = caption_index[i + j]
            if img_id not in embeddings_dict:
                embeddings_dict[img_id] = [None] * 5
            embeddings_dict[img_id][ci] = emb[j]

    # Verify completeness.
    for img_id, embs in embeddings_dict.items():
        if any(e is None for e in embs):
            raise RuntimeError(f"Missing embeddings for image {img_id}")

    # Save.
    with open(save_path, "wb") as f:
        pickle.dump(embeddings_dict, f)
    logger.info("Saved %d image embeddings to %s", len(embeddings_dict), save_path)


# ---------------------------------------------------------------------------
#  DatasetLoader – factory
# ---------------------------------------------------------------------------

class DatasetLoader:
    """
    Constructs train / validation DataLoaders for ImageNet or MS‑COCO.

    Example:
        loader = DatasetLoader(config.data, vae_tokenizer, dataset='imagenet')
        train_loader = loader.get_train_loader()
    """

    def __init__(
        self,
        config: DataConfig,
        vae_tokenizer: VAETokenizer,
        dataset: str = "imagenet",
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> None:
        """
        Args:
            config:        Data configuration (paths, CLIP model).
            vae_tokenizer: Not used internally; kept for potential future
                           latent pre‑computation.
            dataset:       One of 'imagenet' or 'coco'.
            batch_size:    Batch size for the loaders.
            num_workers:   Number of subprocess workers.
        """
        super().__init__()
        self.config = config
        self.vae = vae_tokenizer
        self.dataset = dataset.lower()
        self.batch_size = batch_size
        self.num_workers = num_workers

        if self.dataset not in ("imagenet", "coco"):
            raise ValueError(f"dataset must be 'imagenet' or 'coco', got {dataset}")

        # Pre‑compute COCO embeddings path.
        if self.dataset == "coco":
            self.coco_embeddings_path = os.path.join(
                config.coco_root, "coco_clip_embeddings.pkl"
            )
        else:
            self.coco_embeddings_path = None

    def get_train_loader(self) -> DataLoader:
        """Return the training DataLoader."""
        if self.dataset == "imagenet":
            dataset = ImageNetDataset(self.config.imagenet_root, split="train")
        else:  # coco
            dataset = COCODataset(
                root=self.config.coco_root,
                ann_file=self.config.coco_ann_file,
                embeddings_path=self.coco_embeddings_path,
                split="train",
            )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def get_val_loader(self) -> DataLoader:
        """Return the validation DataLoader."""
        if self.dataset == "imagenet":
            dataset = ImageNetDataset(self.config.imagenet_root, split="val")
        else:
            dataset = COCODataset(
                root=self.config.coco_root,
                ann_file=self.config.coco_ann_file,
                embeddings_path=self.coco_embeddings_path,
                split="val",
            )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

