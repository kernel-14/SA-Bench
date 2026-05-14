"""
Data loading and preprocessing for Hi-MAR.

Supports:
1. Class-conditional generation on ImageNet 256x256
2. Text-to-image generation on MS-COCO 256x256

Uses a VAE (KL-16) to encode images into latent tokens.
For Hi-MAR, we need both 128x128 (low-res) and 256x256 (high-res) latents.
The VAE has an 8x downsampling factor:
- 256x256 image → 32x32 latent (1024 tokens)
- 128x128 image → 16x16 latent (256 tokens)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torchvision.transforms.functional as TF
import numpy as np
import os
from PIL import Image
import json
from typing import Optional, Tuple, List


class ImageNetLatentDataset(Dataset):
    """
    ImageNet dataset that loads images and encodes them to VAE latents.
    
    For Hi-MAR, each sample consists of:
    - Low-resolution latent: 128x128 image → VAE → 16x16 tokens (256)
    - High-resolution latent: 256x256 image → VAE → 32x32 tokens (1024)
    - Class label
    """
    def __init__(
        self,
        root,
        split='train',
        image_size=256,
        low_res_size=128,
        vae=None,
        latent_cache_dir=None,
        transform=None,
    ):
        self.root = root
        self.split = split
        self.image_size = image_size
        self.low_res_size = low_res_size
        self.vae = vae
        self.latent_cache_dir = latent_cache_dir
        
        # Load ImageNet
        if split == 'train':
            self.data = datasets.ImageNet(root, split='train')
        else:
            self.data = datasets.ImageNet(root, split='val')
        
        # Image transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            self.transform_low = transforms.Compose([
                transforms.Resize(low_res_size),
                transforms.CenterCrop(low_res_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = transform
            self.transform_low = transform
        
        # Create cache dir if needed
        if latent_cache_dir:
            os.makedirs(latent_cache_dir, exist_ok=True)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, label = self.data[idx]
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # High-res (256x256)
        img_high = self.transform(img)
        
        # Low-res (128x128)
        img_low = self.transform_low(img)
        
        # Encode to latents using VAE if available
        if self.vae is not None:
            with torch.no_grad():
                # High-res latent
                latent_high = self.vae.encode(img_high.unsqueeze(0)).latent_dist.sample()
                latent_high = latent_high.squeeze(0)  # (C, H, W)
                latent_high = latent_high.reshape(latent_high.shape[0], -1).permute(1, 0)  # (N, C)
                
                # Low-res latent
                latent_low = self.vae.encode(img_low.unsqueeze(0)).latent_dist.sample()
                latent_low = latent_low.squeeze(0)  # (C, H, W)
                latent_low = latent_low.reshape(latent_low.shape[0], -1).permute(1, 0)  # (N, C)
                
                return (latent_low, latent_high), torch.tensor(label, dtype=torch.long)
        else:
            # Return raw images (will be encoded later or used with pre-computed latents)
            return (img_low, img_high), torch.tensor(label, dtype=torch.long)


class COCOTextToImageDataset(Dataset):
    """
    MS-COCO dataset for text-to-image generation.
    
    Each sample:
    - Low-resolution latent: 128x128
    - High-resolution latent: 256x256
    - Text embeddings from CLIP
    """
    def __init__(
        self,
        root,
        annotation_file,
        split='train',
        image_size=256,
        low_res_size=128,
        vae=None,
        text_encoder=None,
        max_text_len=77,
        transform=None,
    ):
        self.root = root
        self.split = split
        self.image_size = image_size
        self.low_res_size = low_res_size
        self.vae = vae
        self.text_encoder = text_encoder
        self.max_text_len = max_text_len
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Filter by split
        self.images = [ann for ann in self.annotations['images']]
        # Build image_id to caption mapping
        self.captions = {}
        for ann in self.annotations['annotations']:
            img_id = ann['image_id']
            if img_id not in self.captions:
                self.captions[img_id] = []
            self.captions[img_id].append(ann['caption'])
        
        # Image transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            self.transform_low = transforms.Compose([
                transforms.Resize(low_res_size),
                transforms.CenterCrop(low_res_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.root, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')
        img_id = img_info['id']
        
        # Randomly select one caption
        captions = self.captions[img_id]
        caption = captions[np.random.randint(len(captions))]
        
        # Transform images
        img_high = self.transform(img)
        img_low = self.transform_low(img)
        
        # Encode text with CLIP
        if self.text_encoder is not None:
            with torch.no_grad():
                text_embeds = self.text_encoder.encode_text([caption])  # (1, 77, C) or similar
        else:
            text_embeds = None  # Will be computed later
        
        # Encode images with VAE
        if self.vae is not None:
            with torch.no_grad():
                latent_high = self.vae.encode(img_high.unsqueeze(0)).latent_dist.sample()
                latent_high = latent_high.squeeze(0)
                latent_high = latent_high.reshape(latent_high.shape[0], -1).permute(1, 0)
                
                latent_low = self.vae.encode(img_low.unsqueeze(0)).latent_dist.sample()
                latent_low = latent_low.squeeze(0)
                latent_low = latent_low.reshape(latent_low.shape[0], -1).permute(1, 0)
        else:
            latent_high = img_high
            latent_low = img_low
        
        return (latent_low, latent_high), None, text_embeds


class LatentDataset(Dataset):
    """
    Dataset for pre-computed VAE latents.
    
    Loads latent tensors directly from disk for faster training.
    """
    def __init__(
        self,
        latent_dir,
        split='train',
        num_classes=None,
    ):
        self.latent_dir = latent_dir
        self.split = split
        self.num_classes = num_classes
        
        # Load file list
        self.latent_files_high = sorted([
            f for f in os.listdir(os.path.join(latent_dir, f'{split}_high'))
            if f.endswith('.pt')
        ])
        self.latent_files_low = sorted([
            f for f in os.listdir(os.path.join(latent_dir, f'{split}_low'))
            if f.endswith('.pt')
        ])
        
        # Load labels if available
        label_path = os.path.join(latent_dir, f'{split}_labels.pt')
        if os.path.exists(label_path):
            self.labels = torch.load(label_path)
        else:
            self.labels = None
    
    def __len__(self):
        return len(self.latent_files_high)
    
    def __getitem__(self, idx):
        latent_high = torch.load(os.path.join(self.latent_dir, f'{self.split}_high', self.latent_files_high[idx]))
        latent_low = torch.load(os.path.join(self.latent_dir, f'{self.split}_low', self.latent_files_low[idx]))
        
        if self.labels is not None:
            label = self.labels[idx]
        else:
            label = None
        
        return (latent_low, latent_high), label


def create_dataloader(
    dataset,
    batch_size=256,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    drop_last=True,
):
    """Create a DataLoader with appropriate settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def get_vae(version='kl-16'):
    """
    Get the KL-16 VAE for encoding images.
    
    This should be the same VAE used by MAR (KL-16 version).
    In practice, this would be loaded from diffusers or a custom checkpoint.
    """
    try:
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse")
        return vae
    except ImportError:
        print("Warning: diffusers not installed. VAE encoding not available.")
        return None


def get_clip_text_encoder():
    """Get CLIP text encoder for MS-COCO."""
    try:
        import clip
        model, _ = clip.load("ViT-B/32", device='cpu')
        return model
    except ImportError:
        try:
            from transformers import CLIPTextModel, CLIPTokenizer
            model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            return model, tokenizer
        except ImportError:
            print("Warning: CLIP not available. Text encoding not available.")
            return None
