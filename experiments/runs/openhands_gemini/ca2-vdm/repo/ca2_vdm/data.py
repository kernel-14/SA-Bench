
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
import os
import random
from typing import List, Optional, Dict, Any

from .config import DataConfig, ModelConfig

# Placeholder for actual video loading
def load_video_frames(video_path: str, num_frames: int, resolution: int, transform: Any) -> torch.Tensor:
    """
    Simulates loading and preprocessing video frames.
    In a real scenario, this would load actual video files (e.g., using decord, torchvision.io).
    For reproduction, we'll create dummy tensors.
    Assumes video_path is a directory containing images, or a video file.
    Returns: (L, C_img, H, W)
    """
    # Dummy implementation: create random frames
    frames = torch.randn(num_frames, 3, resolution, resolution) # 3 channels for RGB
    return transform(frames)

class VideoDataset(Dataset):
    def __init__(self, data_config: DataConfig, model_config: ModelConfig,
                 split: str = "train", is_train_stage1: bool = False):
        self.data_config = data_config
        self.model_config = model_config
        self.split = split
        self.is_train_stage1 = is_train_stage1

        self.video_paths = self._load_video_paths()
        
        # Image transformation
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        # Text encoder (placeholder)
        self.text_encoder = None # Will be initialized outside or passed if needed
        self.vae = None # Will be initialized outside or passed if needed

    def _load_video_paths(self) -> List[str]:
        # Dummy: In a real scenario, this would glob video files from data_config.data_path
        # For now, simulate a few video paths.
        if self.data_config.dataset_name == "internvid":
            num_videos = 1000 if self.split == "train" else 100
            return [f"dummy_video_path/internvid_{i:04d}.mp4" for i in range(num_videos)]
        elif self.data_config.dataset_name == "skytimelapse":
            num_videos = 200 if self.split == "train" else 50
            return [f"dummy_video_path/skytimelapse_{i:04d}.mp4" for i in range(num_videos)]
        elif self.data_config.dataset_name in ["msrvtt", "ucf101"]:
            num_videos = 500 if self.split == "train" else 50
            return [f"dummy_video_path/{self.data_config.dataset_name}_{i:04d}.mp4" for i in range(num_videos)]
        else:
            raise ValueError(f"Unknown dataset: {self.data_config.dataset_name}")

    def _get_text_prompt(self, video_path: str) -> str:
        # Dummy: In a real scenario, load text prompt associated with video
        return f"a video of {os.path.basename(video_path).replace('_', ' ').replace('.mp4', '')}"

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video_path = self.video_paths[idx]
        
        # For training stage 1 (T2V), use t2v_first_stage_frames
        # For training stage 2 (T2V) or video prediction, use vp_train_frames / t2v_second_stage_frames
        if self.is_train_stage1:
            total_frames = self.model_config.t2v_first_stage_frames
            max_condition_frames = 0 # No clean prefix in stage 1
        else:
            total_frames = self.model_config.max_condition_frames + self.model_config.chunk_length
            max_condition_frames = self.model_config.max_condition_frames

        # Load video frames (L, C_img, H, W)
        # Dummy: replace with actual video loading
        video_frames = torch.randn(total_frames, 3, self.data_config.image_size, self.data_config.image_size)
        video_frames = self.transform(video_frames)

        # Encode to latent space (L, C_latent, H_latent, W_latent)
        # Assuming VAE downsamples by 8x for H, W
        latent_H = self.data_config.image_size // 8
        latent_W = self.data_config.image_size // 8
        video_latents = torch.randn(total_frames, self.model_config.latent_channels, latent_H, latent_W) # Dummy latents
        # In actual implementation: video_latents = self.vae.encode(video_frames).sample()
        
        # Randomly select P clean prefix frames
        # P is a multiple of chunk_length for T2V, or random for VP
        if self.is_train_stage1:
            P = 0
        else:
            if self.data_config.dataset_name == "skytimelapse": # Video prediction
                # P can be 1, 1+l, ..., 1+nl where P_max = 1+nl
                P_multiples = [1 + i * self.model_config.chunk_length for i in range(max_condition_frames // self.model_config.chunk_length + 1)]
                P = random.choice(P_multiples)
            else: # Text-to-Video, also use similar logic
                P_multiples = [1 + i * self.model_config.chunk_length for i in range(max_condition_frames // self.model_config.chunk_length + 1)]
                P = random.choice(P_multiples)
            
            # Ensure P does not exceed total_frames - chunk_length for denoising target
            P = min(P, total_frames - self.model_config.chunk_length)
            
        # Create timestep vector and loss mask
        timesteps = torch.zeros(total_frames, dtype=torch.long)
        loss_mask = torch.zeros(total_frames, dtype=torch.float)
        
        # Clean prefix: t=0, loss_mask=0
        # Denoising target: t=random_t, loss_mask=1
        
        # Simulate timestep `t` for denoising target
        random_t = torch.randint(1, self.model_config.timesteps + 1, (1,)).item()
        
        timesteps[P:] = random_t
        loss_mask[P:] = 1.0

        # Create temporal positional embedding indices for cyclic TPEs
        # During training, each sample is assigned a TPE sequence cyclically shifted with a random offset.
        tpe_seq_len = total_frames
        tpe_indices = torch.arange(tpe_seq_len, dtype=torch.long)

        # Apply cyclic shift with a random offset (only for training)
        if self.split == "train" and not self.is_train_stage1:
            offset = random.randint(0, tpe_seq_len - 1)
            tpe_indices = torch.roll(tpe_indices, shifts=offset, dims=0)
        
        # Text conditioning
        text_prompt = self._get_text_prompt(video_path) if self.data_config.dataset_name == "internvid" else ""
        text_embedding = torch.randn(1, 77, self.model_config.context_dim) # Dummy text embedding
        # In actual implementation: text_embedding = self.text_encoder(text_prompt)

        return {
            "video_latents": video_latents, # (L, C_latent, H_latent, W_latent)
            "timesteps": timesteps, # (L,)
            "loss_mask": loss_mask, # (L,)
            "tpe_indices": tpe_indices, # (L,)
            "text_embedding": text_embedding, # (1, N_tokens, C_text)
            "clean_prefix_frames": P, # Number of clean prefix frames
            "video_path": video_path,
        }

def get_dataloader(data_config: DataConfig, model_config: ModelConfig,
                   batch_size: int, split: str, is_train_stage1: bool = False,
                   shuffle: bool = True) -> DataLoader:
    dataset = VideoDataset(data_config, model_config, split=split, is_train_stage1=is_train_stage1)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=data_config.num_workers, pin_memory=True)
    return dataloader

