import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTokenizer
import math
import random

class BaseScheduler:
    def __init__(self, num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, prediction_type="epsilon"):
        self.num_train_timesteps = num_train_timesteps
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.prediction_type = prediction_type

    def add_noise(self, original_samples, noise, timesteps):
        # original_samples: (B, C, H, W)
        # noise: (B, C, H, W)
        # timesteps: (B,)
        
        sqrt_alphas_cumprod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - self.alphas_cumprod[timesteps]) ** 0.5

        # Reshape for broadcasting
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.view(-1, 1, 1, 1)

        noisy_samples = sqrt_alphas_cumprod * original_samples + sqrt_one_minus_alphas_cumprod * noise
        return noisy_samples

    def get_velocity(self, sample, noise, timesteps):
        # Predict velocity based on DDPM formulation for prediction_type "epsilon" or "v_prediction"
        # Not directly used for MAR loss, but for completeness or if prediction_type changes.
        if self.prediction_type == "epsilon":
            return noise # In MAR, the model predicts epsilon directly
        else:
            raise NotImplementedError(f"Prediction type {self.prediction_type} not supported.")

    def get_masking_ratio(self, current_step, total_steps, strategy="cosine", beta_alpha=4.0, beta_beta=1.0, r_min=0.7, r_max=1.0):
        if strategy == "cosine":
            # Cosine masking schedule from MaskGIT (Chang et al., 2022)
            # Implemented as in original MaskGIT or similar models.
            # R(t) = cos(0.5 * pi * t)^2 (for t in [0,1])
            t = current_step / total_steps
            ratio = math.cos(0.5 * math.pi * t) ** 2
            return ratio
        elif strategy == "beta_distribution":
            # Sample from Beta distribution Beta(alpha, beta)
            # Used in MS-COCO text-to-image generation.
            return np.random.beta(beta_alpha, beta_beta)
        elif strategy == "uniform":
            # For phase 1 ImageNet: masking ratio is randomly sampled in [0.7, 1.0]
            return random.uniform(r_min, r_max)
        else:
            raise ValueError(f"Unknown masking strategy: {strategy}")

class ImageNetDataset(Dataset):
    def __init__(self, data_path, image_size=256, low_res_image_size=128):
        # This is a placeholder. For actual ImageNet, you'd use a robust dataset loader
        # like torchvision.datasets.ImageFolder or webdataset.
        # For reproduction, we'll assume `data_path` points to a directory
        # with images that can be loaded.
        
        # NOTE: Full ImageNet dataset loading is outside the scope of
        # a static code reproduction as it involves downloading and
        # potentially complex data pipelines. This provides the API.
        
        self.data_path = data_path
        self.image_size = image_size
        self.low_res_image_size = low_res_image_size
        
        # Placeholder for image paths - replace with actual dataset logic
        # For now, let's create dummy data for demonstration
        self.images = [f"dummy_image_{i}.png" for i in range(100)] 
        self.labels = [i % 1000 for i in range(100)] # Assuming 1000 classes

        self.transform_high_res = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.transform_low_res = transforms.Compose([
            transforms.Resize(low_res_image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(low_res_image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # In a real scenario, load image from self.data_path / self.images[idx]
        # For now, return dummy black images
        dummy_image = Image.fromarray(np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))
        
        high_res_img = self.transform_high_res(dummy_image)
        low_res_img = self.transform_low_res(dummy_image)
        label = self.labels[idx]
        
        return {
            "high_res_image": high_res_img,
            "low_res_image": low_res_img,
            "label": torch.tensor(label, dtype=torch.long)
        }

class MSCOCODataset(Dataset):
    def __init__(self, data_path, image_size=256, low_res_image_size=128, clip_model_name="openai/clip-vit-large-patch14"):
        # Placeholder for MS-COCO. Similar to ImageNet, full dataset loading is complex.
        self.data_path = data_path
        self.image_size = image_size
        self.low_res_image_size = low_res_image_size
        
        self.images = [f"dummy_coco_image_{i}.png" for i in range(100)]
        self.captions = [f"a photo of a cat {i}" for i in range(100)]

        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        
        self.transform_high_res = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.transform_low_res = transforms.Compose([
            transforms.Resize(low_res_image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(low_res_image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        dummy_image = Image.fromarray(np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))
        
        high_res_img = self.transform_high_res(dummy_image)
        low_res_img = self.transform_low_res(dummy_image)
        
        caption = self.captions[idx]
        text_inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids
        
        return {
            "high_res_image": high_res_img,
            "low_res_image": low_res_img,
            "text_input_ids": text_inputs.squeeze(0) # Remove batch dim added by tokenizer
        }