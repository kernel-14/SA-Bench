import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import math
import random
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Union, Callable

# Assuming Config, VideoVAE, TextEncoder, downsample, upsample, get_default_device are available
# from other modules. These are typically imported in main.py and passed around,
# or accessed via a global config object.
# For standalone testing/linting, we'll use minimal stubs if actual imports fail.
try:
    from config import Config, TrainingStageConfig
    from vae import VideoVAE
    from utils import get_default_device, downsample, upsample
    # TextEncoder will be defined in model.py
    # We define a stub here for type hinting and local testing.
    class TextEncoder(nn.Module):
        def __init__(self, config: Config):
            super().__init__()
            # Placeholder for actual T5/CLIP models
            self.t5_model_name = config.model.text_encoder.t5_model_name
            self.clip_model_name = config.model.text_encoder.clip_model_name
            self.max_text_length = config.model.text_encoder.max_text_length
            
        def get_text_embeddings(self, prompts: List[str], device: torch.device) -> Dict[str, torch.Tensor]:
            """
            Stub: Generates dummy text embeddings for given prompts.
            Actual implementation would use T5 and CLIP models.
            """
            batch_size = len(prompts)
            # Example dimensions (should match actual model output)
            t5_embed_dim = 768 # Placeholder for T5-large
            clip_embed_dim = 1024 # Placeholder for CLIP-ViT-L/14

            t5_embeddings = torch.randn(batch_size, self.max_text_length, t5_embed_dim, device=device)
            clip_embeddings = torch.randn(batch_size, self.max_text_length, clip_embed_dim, device=device)
            
            return {'t5': t5_embeddings, 'clip': clip_embeddings}

except ImportError:
    print("Warning: Could not import Config, VideoVAE, utils functions. Using stub classes for data_loader.py.")

    class TrainingStageConfig:
        name: str = "Dummy Stage"
        dataset_names: List[str] = ["dummy_data"]
        dataset_paths: Dict[str, str] = {"dummy_data": "/dummy/path"}
        global_batch_size: int = 1
        
    class ModelConfig:
        pyramid_stages: int = 3
        text_encoder: Any = None # To be replaced by a stub TextEncoder

    class InferenceConfig:
        output_resolution: Tuple[int, int] = (256, 256)
        output_fps: int = 24
        output_duration: int = 5 # seconds

    class ComputeConfig:
        device: str = "cpu"
        num_gpus: int = 1

    class DataPathsConfig:
        image_data_root: str = "/dummy/images"
        video_data_root: str = "/dummy/videos"

    class Config:
        training: Dict[int, TrainingStageConfig] = {
            1: TrainingStageConfig(), 2: TrainingStageConfig(), 3: TrainingStageConfig()
        }
        model: ModelConfig = ModelConfig()
        inference: InferenceConfig = InferenceConfig()
        compute: ComputeConfig = ComputeConfig()
        data_paths: DataPathsConfig = DataPathsConfig()

    class VideoVAE(nn.Module):
        def __init__(self):
            super().__init__()
            # Placeholder for actual VAE
        def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return x, x, x
        def decode(self, latents: torch.Tensor) -> torch.Tensor:
            return latents

    # Stub for TextEncoder
    class TextEncoder(nn.Module):
        def __init__(self, config_stub: Optional[Any] = None):
            super().__init__()
            self.max_text_length = 77
        def get_text_embeddings(self, prompts: List[str], device: torch.device) -> Dict[str, torch.Tensor]:
            batch_size = len(prompts)
            t5_embed_dim = 768
            clip_embed_dim = 1024
            return {
                't5': torch.randn(batch_size, self.max_text_length, t5_embed_dim, device=device),
                'clip': torch.randn(batch_size, self.max_text_length, clip_embed_dim, device=device)
            }
    
    # Minimal stubs for utils functions
    def get_default_device() -> torch.device:
        return torch.device("cpu")

    def downsample(tensor: torch.Tensor, factor: int, mode: str = "trilinear") -> torch.Tensor:
        if tensor.ndim == 5: # Video (B, C, T, H, W)
            return torch.nn.functional.interpolate(tensor, size=(tensor.shape[2] // factor, tensor.shape[3] // factor, tensor.shape[4] // factor), mode=mode, align_corners=False)
        elif tensor.ndim == 4: # Image (B, C, H, W)
            return torch.nn.functional.interpolate(tensor, size=(tensor.shape[2] // factor, tensor.shape[3] // factor), mode='bilinear', align_corners=False)
        else:
            raise NotImplementedError("Stub downsample for other tensor dimensions not implemented.")

    def upsample(tensor: torch.Tensor, factor: int, mode: str = "trilinear") -> torch.Tensor:
        if tensor.ndim == 5: # Video (B, C, T, H, W)
            return torch.nn.functional.interpolate(tensor, size=(tensor.shape[2] * factor, tensor.shape[3] * factor, tensor.shape[4] * factor), mode=mode, align_corners=False)
        elif tensor.ndim == 4: # Image (B, C, H, W)
            return torch.nn.functional.interpolate(tensor, size=(tensor.shape[2] * factor, tensor.shape[3] * factor), mode='bilinear', align_corners=False)
        else:
            raise NotImplementedError("Stub upsample for other tensor dimensions not implemented.")

    # Stub for load_video_frames (usually in utils.py)
    def load_video_frames(video_path: Union[str, Path], num_frames: int, fps: int, resolution: Tuple[int, int]) -> np.ndarray:
        """
        Stub: Returns a dummy numpy array representing video frames.
        """
        # (T, H, W, C), uint8
        return np.zeros((num_frames, resolution[0], resolution[1], 3), dtype=np.uint8)


# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def _get_dummy_data_list(
    data_root: Union[str, Path],
    dataset_names: List[str],
    dataset_paths_map: Dict[str, str],
    split: str,
    num_samples_per_dataset: int = 100
) -> List[Dict[str, Any]]:
    """
    Internal helper function to generate a dummy list of data entries.
    In a real scenario, this would parse manifest files or scan directories.
    """
    all_data_list = []
    base_caption = "A dummy image." if "image" in str(data_root) else "A dummy video of a cat playing."
    
    for ds_name in dataset_names:
        relative_path = dataset_paths_map.get(ds_name, ds_name.lower().replace(" ", "_"))
        full_ds_path = Path(data_root) / relative_path

        # Simulate splitting for 'train' and 'val'
        current_num_samples = num_samples_per_dataset
        if split == 'val':
            current_num_samples = max(1, num_samples_per_dataset // 10) # 10% for validation

        for i in range(current_num_samples):
            file_name = f"{ds_name.replace(' ', '_').lower()}_{i:05d}.{'jpg' if 'image' in str(data_root) else 'mp4'}"
            file_path = full_ds_path / file_name
            all_data_list.append({
                "file_path": str(file_path),
                "caption": f"{base_caption} (Source: {ds_name}, ID: {i})."
            })
    logger.info(f"Generated {len(all_data_list)} dummy entries for split '{split}' from {dataset_names}.")
    return all_data_list


class ImageDataset(Dataset):
    """
    Dataset for loading image data for Stage 1 of training.
    """
    def __init__(self, config: Config, split: str, stage_idx: int, text_encoders: TextEncoder):
        """
        Args:
            config (Config): Global configuration object.
            split (str): 'train' or 'val'.
            stage_idx (int): The index of the training stage (expected 1 for image training).
            text_encoders (TextEncoder): Instance of TextEncoder for caption tokenization.
        """
        if stage_idx not in config.training:
            raise ValueError(f"Training stage {stage_idx} not found in config.")
        
        self.config = config
        self.split = split
        self.stage_config: TrainingStageConfig = config.training[stage_idx]
        self.text_encoders = text_encoders
        self.text_encoder_device = get_default_device() # Device for text encoder
        
        self.data_list = _get_dummy_data_list(
            data_root=config.data_paths.image_data_root,
            dataset_names=self.stage_config.dataset_names,
            dataset_paths_map=self.stage_config.dataset_paths,
            split=split
        )

        output_res = config.inference.output_resolution
        self.transform = transforms.Compose([
            transforms.Resize(output_res),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        logger.info(f"ImageDataset initialized for stage {stage_idx}, split '{split}' with {len(self.data_list)} samples.")

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.data_list[idx]
        file_path = entry["file_path"]
        caption = entry["caption"]

        try:
            image = Image.open(file_path).convert("RGB")
            image_tensor = self.transform(image)
        except Exception as e:
            logger.warning(f"Error loading or transforming image {file_path}: {e}. Returning black image.")
            output_res = self.config.inference.output_resolution
            image_tensor = torch.zeros(3, output_res[0], output_res[1], dtype=torch.float32) # Default black image normalized
            # If using [-1,1] normalization, black is 0.5*2-1 = 0 (before norm it's 0.5) or fill with 0 if it's already normalized.
            image_tensor = self.transform(Image.fromarray(np.zeros((output_res[0], output_res[1], 3), dtype=np.uint8)))


        text_embeddings = self.text_encoders.get_text_embeddings([caption], self.text_encoder_device)
        # Squeeze batch dimension (B=1) from text embeddings
        text_embeds_t5 = text_embeddings['t5'].squeeze(0)
        text_embeds_clip = text_embeddings['clip'].squeeze(0)

        return {
            'image_frames': image_tensor.unsqueeze(0), # Add temporal dim for consistency (1, C, H, W) -> (1, C, 1, H, W) if it was video. But for images it's (C, H, W)
            'text_prompt': caption,
            'text_embeds_t5': text_embeds_t5,
            'text_embeds_clip': text_embeds_clip
        }


class VideoDataset(Dataset):
    """
    Dataset for loading video data for Stages 2 and 3 of training.
    """
    def __init__(self, config: Config, split: str, vae: VideoVAE, stage_idx: int, text_encoders: TextEncoder):
        """
        Args:
            config (Config): Global configuration object.
            split (str): 'train' or 'val'.
            vae (VideoVAE): Instance of VideoVAE (used for type hinting, encoding done in Trainer).
            stage_idx (int): The index of the training stage (expected 2 or 3 for video training).
            text_encoders (TextEncoder): Instance of TextEncoder for caption tokenization.
        """
        if stage_idx not in config.training:
            raise ValueError(f"Training stage {stage_idx} not found in config.")

        self.config = config
        self.split = split
        self.vae = vae # Stored for type consistency, actual use in Trainer
        self.stage_config: TrainingStageConfig = config.training[stage_idx]
        self.text_encoders = text_encoders
        self.text_encoder_device = get_default_device() # Device for text encoder

        self.data_list = _get_dummy_data_list(
            data_root=config.data_paths.video_data_root,
            dataset_names=self.stage_config.dataset_names,
            dataset_paths_map=self.stage_config.dataset_paths,
            split=split
        )

        output_res = config.inference.output_resolution
        self.transform = transforms.Compose([
            transforms.Resize(output_res),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.output_fps = config.inference.output_fps
        self.output_duration = config.inference.output_duration # in seconds
        self.max_output_duration = config.inference.max_output_duration # in seconds

        # For stage 2 & 3, the paper mentions training for 2s, 5s, and 5-10s videos.
        # Let's assume for simplicity, in __getitem__, we load max_output_duration * output_fps frames
        # and then sample the required number for the current training config.
        # However, for consistency, the model expects a certain number of frames.
        # For this implementation, let's load a fixed duration for now, maybe output_duration or max_output_duration.
        # Let's assume `num_target_frames` is for the current frame to predict.
        # The paper says: "The proposed pyramidal flow matching framework significantly reduces the computational and memory overhead in video generation training. Consider a video with T frame latents, where each frame contains N tokens at the original resolution."
        # And "After postprocessing, around 10M single-shot videos are available for training."
        # And "80,000 steps on 2-second video generation, followed by an additional 120,000 steps on 5-second videos."
        # Let's assume `num_target_frames` refers to the target sequence length for the *current* model prediction.
        # For simplicity, we'll aim for a target duration from `output_duration`.
        self.num_target_frames = self.output_duration * self.output_fps # T frames for the generated video segment
        
        # History condition: `x_t'^{i-2}, x_t'^{i-1}` implies 2 history frames
        self.num_history_frames = 2 
        
        logger.info(f"VideoDataset initialized for stage {stage_idx}, split '{split}' with {len(self.data_list)} samples.")
        logger.info(f"  Target frames per video segment: {self.num_target_frames}")
        logger.info(f"  History frames per video segment: {self.num_history_frames}")

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.data_list[idx]
        file_path = entry["file_path"]
        caption = entry["caption"]

        total_frames_to_load = self.num_target_frames + self.num_history_frames
        video_resolution = self.config.inference.output_resolution

        try:
            # Assumes utils.load_video_frames loads frames as (T, H, W, C) numpy array (uint8)
            raw_frames_np = load_video_frames(
                file_path, total_frames_to_load, self.output_fps, video_resolution
            )
            # Handle case where video might be shorter, load_video_frames should handle padding
            
            # Split into history and target
            history_raw_np = raw_frames_np[:self.num_history_frames]
            video_raw_np = raw_frames_np[self.num_history_frames:]

            history_frames_tensor_list = []
            for frame_np in history_raw_np:
                frame_pil = Image.fromarray(frame_np)
                history_frames_tensor_list.append(self.transform(frame_pil))
            history_frames_tensor = torch.stack(history_frames_tensor_list) # (History_T, C, H, W)

            video_frames_tensor_list = []
            for frame_np in video_raw_np:
                frame_pil = Image.fromarray(frame_np)
                video_frames_tensor_list.append(self.transform(frame_pil))
            video_frames_tensor = torch.stack(video_frames_tensor_list) # (T, C, H, W)
            
        except Exception as e:
            logger.warning(f"Error loading or transforming video {file_path}: {e}. Returning black frames.")
            output_res = self.config.inference.output_resolution
            # Create black frames tensors
            history_frames_tensor = torch.zeros(
                self.num_history_frames, 3, output_res[0], output_res[1], dtype=torch.float32
            )
            video_frames_tensor = torch.zeros(
                self.num_target_frames, 3, output_res[0], output_res[1], dtype=torch.float32
            )
        
        text_embeddings = self.text_encoders.get_text_embeddings([caption], self.text_encoder_device)
        text_embeds_t5 = text_embeddings['t5'].squeeze(0)
        text_embeds_clip = text_embeddings['clip'].squeeze(0)

        return {
            'video_frames': video_frames_tensor, # (T, C, H, W)
            'text_prompt': caption,
            'text_embeds_t5': text_embeds_t5,
            'text_embeds_clip': text_embeds_clip,
            'history_frames': history_frames_tensor # (History_T, C, H, W)
        }


def create_data_loaders(
    config: Config, vae: VideoVAE, text_encoders: TextEncoder
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """
    Factory function to instantiate and return DataLoader objects for training and validation.

    Args:
        config (Config): The global configuration object.
        vae (VideoVAE): The VAE instance.
        text_encoders (TextEncoder): The TextEncoder instance.

    Returns:
        Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
        A tuple containing (train_img_loader, train_vid_loader, val_vid_loader).
        Loaders will be None if their respective training stages are not configured.
    """
    train_img_loader: Optional[DataLoader] = None
    train_vid_loader: Optional[DataLoader] = None
    val_vid_loader: Optional[DataLoader] = None

    num_gpus = config.compute.num_gpus
    world_size = int(os.environ.get("WORLD_SIZE", "1")) if num_gpus > 1 else 1

    # --- Image Training Loader (Stage 1) ---
    if 1 in config.training and config.training[1].dataset_type == "image":
        stage1_config = config.training[1]
        img_train_dataset = ImageDataset(config, 'train', 1, text_encoders)
        
        if num_gpus > 1 and world_size > 1:
            img_train_sampler = DistributedSampler(img_train_dataset, num_replicas=world_size, rank=int(os.environ.get("RANK", "0")), shuffle=True)
            img_batch_size = stage1_config.global_batch_size // world_size
        else:
            img_train_sampler = None
            img_batch_size = stage1_config.global_batch_size

        train_img_loader = DataLoader(
            img_train_dataset,
            batch_size=img_batch_size,
            shuffle=(img_train_sampler is None),
            sampler=img_train_sampler,
            num_workers=os.cpu_count() // world_size if os.cpu_count() else 0,
            pin_memory=True,
            drop_last=True
        )
        logger.info(f"Created Image Training DataLoader with batch size {img_batch_size * world_size} (global).")

    # --- Video Training Loader (Stages 2 & 3 combined, or separate if needed) ---
    # For simplicity, we create one video DataLoader that will be used across stages 2 and 3.
    # The `Trainer` will handle stage-specific batch size and learning rate.
    if (2 in config.training and config.training[2].dataset_type == "video") or \
       (3 in config.training and config.training[3].dataset_type == "video"):
        
        # Use stage 2 config for initial video dataset parameters
        stage_video_config = config.training.get(2, config.training.get(3))
        if not stage_video_config: # Fallback if neither 2 nor 3 exists
            raise ValueError("No video training stage (2 or 3) found in config.")

        vid_train_dataset = VideoDataset(config, 'train', vae, 2, text_encoders) # Pass stage 2 idx for dataset loading
        
        if num_gpus > 1 and world_size > 1:
            vid_train_sampler = DistributedSampler(vid_train_dataset, num_replicas=world_size, rank=int(os.environ.get("RANK", "0")), shuffle=True)
            vid_batch_size = stage_video_config.global_batch_size // world_size
        else:
            vid_train_sampler = None
            vid_batch_size = stage_video_config.global_batch_size

        train_vid_loader = DataLoader(
            vid_train_dataset,
            batch_size=vid_batch_size,
            shuffle=(vid_train_sampler is None),
            sampler=vid_train_sampler,
            num_workers=os.cpu_count() // world_size if os.cpu_count() else 0,
            pin_memory=True,
            drop_last=True
        )
        logger.info(f"Created Video Training DataLoader with batch size {vid_batch_size * world_size} (global).")

        # --- Video Validation Loader ---
        vid_val_dataset = VideoDataset(config, 'val', vae, 2, text_encoders) # Pass stage 2 idx for dataset loading
        
        if num_gpus > 1 and world_size > 1:
            vid_val_sampler = DistributedSampler(vid_val_dataset, num_replicas=world_size, rank=int(os.environ.get("RANK", "0")), shuffle=False)
            val_batch_size = stage_video_config.global_batch_size // world_size # Use same batch size for val
        else:
            vid_val_sampler = None
            val_batch_size = stage_video_config.global_batch_size

        val_vid_loader = DataLoader(
            vid_val_dataset,
            batch_size=val_batch_size,
            shuffle=False, # No shuffling for validation
            sampler=vid_val_sampler,
            num_workers=os.cpu_count() // world_size if os.cpu_count() else 0,
            pin_memory=True,
            drop_last=False # Don't drop last for validation to evaluate all samples
        )
        logger.info(f"Created Video Validation DataLoader with batch size {val_batch_size * world_size} (global).")

    return train_img_loader, train_vid_loader, val_vid_loader


if __name__ == "__main__":
    print("--- Testing data_loader.py ---")

    # Create a dummy config for testing
    dummy_config = Config()
    dummy_config.training[1] = TrainingStageConfig(
        name="Stage 1: Image Training",
        dataset_type="image",
        dataset_names=["LAION-5B-Aesthetic-Subset"],
        dataset_paths={"LAION-5B-Aesthetic-Subset": "/dummy/images/laion"}
    )
    dummy_config.training[2] = TrainingStageConfig(
        name="Stage 2: Low-Resolution Video Training",
        dataset_type="video",
        dataset_names=["WebVid-10M"],
        dataset_paths={"WebVid-10M": "/dummy/videos/webvid"}
    )
    dummy_config.training[3] = TrainingStageConfig(
        name="Stage 3: High-Resolution Video Training",
        dataset_type="video",
        dataset_names=["WebVid-10M"], # Same for simplicity
        dataset_paths={"WebVid-10M": "/dummy/videos/webvid"}
    )
    dummy_config.compute.num_gpus = 1 # Test single GPU behavior

    # Instantiate dummy VAE and TextEncoder
    dummy_vae = VideoVAE()
    dummy_text_encoders = TextEncoder(dummy_config) # Pass config_stub

    # Create data loaders
    train_img_loader, train_vid_loader, val_vid_loader = create_data_loaders(
        dummy_config, dummy_vae, dummy_text_encoders
    )

    # Test Image DataLoader
    if train_img_loader:
        print(f"\nImage Training DataLoader has {len(train_img_loader)} batches.")
        for i, batch in enumerate(train_img_loader):
            print(f"  Batch {i}: Image Frames Shape={batch['image_frames'].shape}, "
                  f"Text T5 Embeds Shape={batch['text_embeds_t5'].shape}, "
                  f"Text CLIP Embeds Shape={batch['text_embeds_clip'].shape}")
            assert batch['image_frames'].ndim == 4 # (B, C, H, W) from (B, C, 1, H, W) squeeze
            assert batch['image_frames'].shape[1] == 3 # RGB channels
            assert batch['text_embeds_t5'].ndim == 2 # (B=1 was squeezed to (N, E))
            if i >= 1: break # Check a few batches
    else:
        print("\nImage Training DataLoader not created.")

    # Test Video DataLoader
    if train_vid_loader:
        print(f"\nVideo Training DataLoader has {len(train_vid_loader)} batches.")
        for i, batch in enumerate(train_vid_loader):
            print(f"  Batch {i}: Video Frames Shape={batch['video_frames'].shape}, "
                  f"History Frames Shape={batch['history_frames'].shape}, "
                  f"Text T5 Embeds Shape={batch['text_embeds_t5'].shape}")
            assert batch['video_frames'].ndim == 4 # (B, T, C, H, W)
            assert batch['history_frames'].ndim == 4 # (B, History_T, C, H, W)
            assert batch['video_frames'].shape[1] == dummy_config.inference.output_duration * dummy_config.inference.output_fps
            assert batch['history_frames'].shape[1] == 2 # 2 history frames
            if i >= 1: break # Check a few batches
    else:
        print("\nVideo Training DataLoader not created.")

    if val_vid_loader:
        print(f"\nVideo Validation DataLoader has {len(val_vid_loader)} batches.")
        for i, batch in enumerate(val_vid_loader):
            print(f"  Batch {i}: Video Frames Shape={batch['video_frames'].shape}, "
                  f"History Frames Shape={batch['history_frames'].shape}")
            assert batch['video_frames'].ndim == 4
            assert batch['history_frames'].ndim == 4
            if i >= 1: break # Check a few batches
    else:
        print("\nVideo Validation DataLoader not created.")

    print("\nAll data_loader.py tests completed.")

