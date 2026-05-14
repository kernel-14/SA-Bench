"""
MS-COCO dataset for text-to-image generation.
"""

import os
import json
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class COCODataset(Dataset):
    """
    MS-COCO dataset for text-to-image generation.

    Following the paper's setup:
    - 82,783 training images
    - 40,504 validation images
    - Each image has 5 captions
    - Text embeddings from CLIP text encoder
    """

    def __init__(
        self,
        data_path,
        img_size=256,
        split='train',
        clip_model_name='openai/clip-vit-large-patch14',
        max_text_len=77,
    ):
        self.data_path = data_path
        self.img_size = img_size
        self.split = split
        self.max_text_len = max_text_len

        # Load annotations
        ann_file = os.path.join(
            data_path, 'annotations',
            f'captions_{split}2014.json'
        )
        with open(ann_file, 'r') as f:
            annotations = json.load(f)

        # Build image-caption mapping
        self.image_info = {img['id']: img for img in annotations['images']}
        self.captions = {}
        for ann in annotations['annotations']:
            img_id = ann['image_id']
            if img_id not in self.captions:
                self.captions[img_id] = []
            self.captions[img_id].append(ann['caption'])

        self.image_ids = list(self.captions.keys())

        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # Load CLIP text encoder
        self._load_clip(clip_model_name)

    def _load_clip(self, model_name):
        """Load CLIP text encoder."""
        try:
            import clip
            self.clip_model, _ = clip.load('ViT-L/14', device='cpu')
            self.clip_model.eval()
            self.use_clip = True
            self.tokenize = clip.tokenize
        except ImportError:
            try:
                from transformers import CLIPTextModel, CLIPTokenizer
                self.clip_tokenizer = CLIPTokenizer.from_pretrained(model_name)
                self.clip_text_model = CLIPTextModel.from_pretrained(model_name)
                self.clip_text_model.eval()
                self.use_clip = False
                self.use_transformers_clip = True
            except ImportError:
                print('Warning: CLIP not available. Using random text embeddings.')
                self.use_clip = False
                self.use_transformers_clip = False

    def _get_text_embedding(self, caption):
        """Get CLIP text embedding for a caption."""
        if hasattr(self, 'use_clip') and self.use_clip:
            with torch.no_grad():
                tokens = self.tokenize([caption], truncate=True)
                text_emb = self.clip_model.encode_text(tokens)
                # Get full sequence embeddings (not just CLS)
                # For CLIP ViT-L/14, text dim is 768
                return text_emb.float()
        elif hasattr(self, 'use_transformers_clip') and self.use_transformers_clip:
            with torch.no_grad():
                inputs = self.clip_tokenizer(
                    caption,
                    max_length=self.max_text_len,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                outputs = self.clip_text_model(**inputs)
                return outputs.last_hidden_state.squeeze(0).float()
        else:
            # Fallback: random embeddings
            return torch.randn(self.max_text_len, 768)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.image_info[img_id]

        # Load image
        img_path = os.path.join(
            self.data_path,
            f'{self.split}2014',
            img_info['file_name']
        )
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        # Randomly select one caption
        captions = self.captions[img_id]
        caption = random.choice(captions)

        # Get text embedding
        text_emb = self._get_text_embedding(caption)

        return image, text_emb
