```python
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import os
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from accelerate import Accelerator
from accelerate.logging import get_logger
from pathlib import Path

# Project-specific imports
try:
    from config import Config, TrainingStageConfig
    from model import PyramidalFlowMatchingModel, TextEncoder
    from vae import VideoVAE
    from pyramid_logic import PyramidFlowMatcher
    from utils import create_optimizer_scheduler, save_checkpoint, load_checkpoint, get_default_device
    from data_loader import ImageDataset, VideoDataset
except ImportError as e:
    # Minimal Stubs for testing trainer.py independently
    print(f"Failed to import project modules: {e}. Using stub classes for local testing.")

    class Config:
        def __init__(self):
            self.model = self.ModelConfig()
            self.compute = self.ComputeConfig()
            self.training = {
                1: self.TrainingStageConfig(),
                2: self.TrainingStageConfig(),
                3: self.TrainingStageConfig(),
            }
            self.data_paths = self.DataPathsConfig()

        class TrainingStageConfig:
            name: str = "Dummy Stage"
            global_batch_size: int = 1
            learning_rate: float = 1e-4
            training_steps: int = 10
            warmup_steps: int = 1
            optimizer_beta1: float = 0.9
            optimizer_beta2: float = 0.999
            optimizer_epsilon: float = 1e-6
            gradient_clipping: float = 1.0
            weight_decay: float = 1e-4
            dataset_type: str = "image"
            dataset_names: List[str] = ["dummy"]
            dataset_paths: Dict[str, str] = {"dummy": "/dummy"}
            history_condition_noise_strength: Optional[List[float]] = None

        class ModelConfig:
            pyramid_stages: int = 3
            spatial_pyramid_time_windows: List[Tuple[float, float]] = [(0.0, 1.0)]
            vae: Any = None
            text_encoder: Any = None

        class ComputeConfig:
            device: str = "cpu"
            num_gpus: int = 1
            mixed_precision: str = "no"

        class DataPathsConfig:
            model_weights: str = "dummy_model.pth"

    class PyramidalFlowMatchingModel(torch.nn.Module):
        def __init__(self, config: Config, text_encoders: Any):
            super().__init__()
            self.config = config
            self.linear = torch.nn.Linear(4, 4) # Dummy layer
        def forward(self, latent_xt: torch.Tensor, time_batch: torch.Tensor, text_cond_t5: torch.Tensor, text_cond_clip: torch.Tensor, history_cond: Optional[List[torch.Tensor]] = None, pyramid_stage_k: int = 0) -> torch.Tensor:
            # Dummy forward pass: just pass latent_xt through a linear layer
            # In a real scenario, it would process all inputs
            # The DiT model expects (B, N_tokens, C_embed) for transformer.
            # Assuming latent_xt is (B, C, T, H, W) and gets flattened to (B, T*H*W, C)
            # The dummy model just returns an output of the same shape as latent_xt
            return self.linear(latent_xt.permute(0,2,3,4,1)).permute(0,4,1,2,3)


    class VideoVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Minimal VAE stub to mock encode/decode
        def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            # Simulate encoding to a 4-channel latent, downsampled 8x
            latent_channels = 4
            if x.ndim == 5: # video (B, C, T, H, W)
                return torch.zeros(x.shape[0], latent_channels, x.shape[2]//8, x.shape[3]//8, x.shape[4]//8), None, torch.randn(x.shape[0], latent_channels, x.shape[2]//8, x.shape[3]//8, x.shape[4]//8)
            elif x.ndim == 4: # image (B, C, H, W)
                return torch.zeros(x.shape[0], latent_channels, x.shape[2]//8, x.shape[3]//8), None, torch.randn(x.shape[0], latent_channels, x.shape[2]//8, x.shape[3]//8)
            else:
                raise ValueError("Unsupported input dimensions for VAE stub.")
        def decode(self, latents: torch.Tensor) -> torch.Tensor:
            # Simulate decoding
            if latents.ndim == 5:
                return torch.randn(latents.shape[0], 3, latents.shape[2]*8, latents.shape[3]*8, latents.shape[4]*8)
            elif latents.ndim == 4:
                return torch.randn(latents.shape[0], 3, latents.shape[2]*8, latents.shape[3]*8)
            else:
                raise ValueError("Unsupported latent dimensions for VAE stub.")

    class PyramidFlowMatcher:
        def __init__(self, config: Config, vae: VideoVAE):
            self.config = config
            self.vae = vae
            self.K = config.model.pyramid_stages
            self.spatial_pyramid_timesteps = config.model.spatial_pyramid_time_windows
            self.device = get_default_device()

        def sample_pyramid_stage(self) -> int:
            return 0 # Always return 0 for stub

        def get_pyramid_timesteps(self, k_stage: int) -> Tuple[float, float]:
            return (0.0, 1.0) # Always return full range for stub

        def get_down_factor(self, k_stage: int) -> int:
            return 2**k_stage

        def compute_pyramid_endpoints(self, x1: torch.Tensor, k_stage: int, t_prime: torch.Tensor, noise: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            # Stub implementation
            down_factor = self.get_down_factor(k_stage)
            target_shape = list(x1.shape)
            if x1.ndim == 5: # Video
                target_shape[2] = max(1, target_shape[2] // down_factor) # Ensure T is at least 1
                target_shape[3] = max(1, target_shape[3] // down_factor)
                target_shape[4] = max(1, target_shape[4] // down_factor)
            elif x1.ndim == 4: # Image
                target_shape[2] = max(1, target_shape[2] // down_factor)
                target_shape[3] = max(1, target_shape[3] // down_factor)
            
            x1_down = torch.randn(target_shape, device=x1.device)
            noise_down = torch.randn_like(x1_down)

            x_ek = t_prime * x1_down + (1 - t_prime) * noise_down
            x_sk = (1 - t_prime) * x1_down + t_prime * noise_down # Flipped for testing
            target_vector = x_ek - x_sk
            return x_ek, x_sk, target_vector

    class ImageDataset(torch.utils.data.Dataset):
        def __init__(self, config: Config, split: str, stage_idx: int, text_encoders: Any):
            self.len = 10
            self.config = config
            self.text_encoders = text_encoders
            self.text_encoder_device = get_default_device()
        def __len__(self): return self.len
        def __getitem__(self, idx):
            t5_embeds = torch.randn(77, 768)
            clip_embeds = torch.randn(77, 1024)
            return {
                'image_frames': torch.randn(3, 256, 256), # C, H, W
                'text_prompt': "a dog",
                'text_embeds_t5': t5_embeds,
                'text_embeds_clip': clip_embeds
            }

    class VideoDataset(torch.utils.data.Dataset):
        def __init__(self, config: Config, split: str, vae: VideoVAE, stage_idx: int, text_encoders: Any):
            self.len = 10
            self.config = config
            self.text_encoders = text_encoders
            self.vae = vae
            self.text_encoder_device = get_default_device()
            self.num_history_frames = 2
            self.history_condition_noise_strength = config.training[stage_idx].history_condition_noise_strength or [0.0, 0.0]

        def __len__(self): return self.len
        def __getitem__(self, idx):
            t5_embeds = torch.randn(77, 768)
            clip_embeds = torch.randn(77, 1024)
            
            # Simulate video frames and history frames in pixel space
            video_frames_pixel = torch.randn(3, 24, 256, 256) # C, T, H, W
            history_frames_pixel = [torch.randn(3, 256, 256), torch.randn(3, 256, 256)] # List of C, H, W

            # Encode history frames to latents and apply noise/downsampling
            history_latent_list = []
            for i, h_frame_pixel in enumerate(history_frames_pixel):
                # VAE expects B,C,T,H,W for video, or B,C,H,W for image.
                # Assuming history frames are processed as individual images by VAE then stacked.
                # For this stub, we create dummy latents matching expected output shape for DiT model.
                # The DiT model expects history_cond as a List[torch.Tensor]
                
                # Simplified dummy for history_latent_list
                # For k+1 resolution
                h_latent_k_plus_1 = torch.randn(4, 256//(2**(0+1)*8), 256//(2**(0+1)*8)) # 4, 16, 16 for full res, then k=0 -> k+1=1 -> 2x downsample
                history_latent_list.append(h_latent_k_plus_1.unsqueeze(0)) # Add dummy T=1 dim
                
                # For k resolution
                h_latent_k = torch.randn(4, 256//(2**(0)*8), 256//(2**(0)*8)) # 4, 32, 32 for full res, then k=0 -> 1x downsample
                history_latent_list.append(h_latent_k.unsqueeze(0)) # Add dummy T=1 dim


            # Encode main video frames to latent
            # Placeholder: The vae.encode in stub expects (B, C, H, W) or (B, C, T, H, W)
            # VideoDataset provides (C, T, H, W)
            _, _, video_latent = self.vae.encode(video_frames_pixel.unsqueeze(0)) # (1, C, T, H, W) -> VAE -> (1, C_l, T_l, H_l, W_l)
            
            return {
                'video_latent': video_latent.squeeze(0), # C_l, T_l, H_l, W_l
                'text_prompt': "a cat video",
                'text_embeds_t5': t5_embeds,
                'text_embeds_clip': clip_embeds,
                'history_latent_frames': history_latent_list # List[Tensor], each (C_l, T_l, H_l, W_l)
            }

    def create_optimizer_scheduler(model: torch.nn.Module, config: Config, stage_idx: int) -> Tuple[torch.optim.Optimizer, Any]:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
        return optimizer, scheduler

    def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any, step: int, stage_idx: int, config: Config, save_path: Optional[Union[str, Path]] = None):
        print(f"Stub: Saving checkpoint for stage {stage_idx}, step {step} to {save_path or config.data_paths.model_weights}")

    def load_checkpoint(model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer], scheduler: Optional[Any], config: Config, checkpoint_path: Optional[Union[str, Path]] = None) -> Tuple[int, int]:
        print(f"Stub: Loading checkpoint from {checkpoint_path or config.data_paths.model_weights}")
        return 0, 0

    def get_default_device() -> torch.device:
        return torch.device("cpu")

    class TextEncoder(torch.nn.Module):
        def __init__(self, config: Config):
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


logger = get_logger(__name__)


class Trainer:
    """
    The Trainer class orchestrates the three-stage training process for Pyramidal Flow Matching.
    It handles data loading, loss computation, model updates, logging, and checkpointing.
    """

    def __init__(
        self,
        config: Config,
        model: PyramidalFlowMatchingModel,
        vae: VideoVAE,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        pyramid_logic: PyramidFlowMatcher,
        train_img_loader: Optional[DataLoader],
        train_vid_loader: Optional[DataLoader],
        val_vid_loader: Optional[DataLoader],
    ):
        """
        Initializes the Trainer.

        Args:
            config (Config): The global configuration object.
            model (PyramidalFlowMatchingModel): The main Pyramidal Flow Matching model.
            vae (VideoVAE): The Video VAE.
            optimizer (torch.optim.Optimizer): The optimizer for the model.
            scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
            pyramid_logic (PyramidFlowMatcher): The logic handler for pyramid operations.
            train_img_loader (Optional[DataLoader]): DataLoader for image training (Stage 1).
            train_vid_loader (Optional[DataLoader]): DataLoader for video training (Stages 2 & 3).
            val_vid_loader (Optional[DataLoader]): DataLoader for video validation.
        """
        self.config = config
        self.pyramid_logic = pyramid_logic
        self.vae = vae
        self.global_step = 0
        self.current_stage_idx = 0

        self.accelerator = Accelerator(
            mixed_precision=self.config.compute.mixed_precision,
        )

        # Prepare model, optimizer, scheduler, and data loaders for distributed training
        # Note: optimizer and scheduler might change across stages. They will be re-prepared.
        (
            self.model,
            self.optimizer,
            self.scheduler,
            self.train_img_loader,
            self.train_vid_loader,
            self.val_vid_loader,
        ) = self.accelerator.prepare(
            model, optimizer, scheduler, train_img_loader, train_vid_loader, val_vid_loader
        )

        logger.info(f"Trainer initialized on device: {self.accelerator.device}, "
                    f"mixed precision: {self.accelerator.mixed_precision}")

    def _compute_loss(self, model_output: torch.Tensor, target_vector: torch.Tensor) -> torch.Tensor:
        """
        Calculates the mean squared error (MSE) loss between model output and target vector.

        Args:
            model_output (torch.Tensor): The velocity field predicted by the model.
            target_vector (torch.Tensor): The target velocity vector from pyramid logic.

        Returns:
            torch.Tensor: The computed MSE loss.
        """
        return F.mse_loss(model_output, target_vector, reduction='mean')

    def _log_metrics(self, stage_idx: int, step: int, loss: float, lr: float) -> None:
        """
        Logs training metrics, visible only from the main process.

        Args:
            stage_idx (int): Current training stage index.
            step (int): Current global training step.
            loss (float): Current training loss.
            lr (float): Current learning rate.
        """
        if self.accelerator.is_main_process:
            self.accelerator.log(
                {"loss": loss, "learning_rate": lr},
                step=step,
                group=f"stage_{stage_idx}"
            )
            logger.info(f"Stage {stage_idx} | Step {step} | Loss: {loss:.4f} | LR: {lr:.6f}")

    def _save_checkpoint(self, stage_idx: int, current_step_in_stage: int) -> None:
        """
        Saves the current training state (model, optimizer, scheduler, step, stage).
        Only saves from the main process.

        Args:
            stage_idx (int): Current training stage index.
            current_step_in_stage (int): Current step within the current stage.
        """
        if self.accelerator.is_main_process:
            # Unwrap the DDP model before saving its state_dict
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            
            # Construct a more descriptive save path
            save_dir = Path(self.config.data_paths.model_weights).parent / f"stage_{stage_idx}"
            save_dir.mkdir(parents=True, exist_ok=True)
            filename = f"checkpoint_step_{current_step_in_stage}.pth"
            save_path = save_dir / filename

            # Use utils.save_checkpoint
            save_checkpoint(
                model=unwrapped_model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                step=self.global_step, # Use global step for overall progress
                stage_idx=stage_idx,
                config=self.config,
                save_path=save_path
            )

    def _train_step(self, batch: Dict[str, Any], stage_idx: int) -> float:
        """
        Performs a single training step for a given batch and current training stage.

        Args:
            batch (Dict[str, Any]): A batch of data from the DataLoader.
            stage_idx (int): The current training stage index (1, 2, or 3).

        Returns:
            float: The computed loss for the current step as a scalar.
        """
        self.model.train()
        
        stage_config = self.config.training[stage_idx]
        current_device = self.accelerator.device

        # 1. Prepare clean latent x1
        x1: torch.Tensor
        if stage_config.dataset_type == "image":
            # For images, image_frames is (B, C, H, W). VAE encode handles 4D.
            # Convert to latent space: (B, 3, H, W) -> (B, latent_C, latent_H, latent_W)
            pixel_frames = batch['image_frames'] # (B, C, H, W)
            # VAE might expect 5D input for consistency even with images (T=1)
            # Check VAE input dimension requirement for `encode`
            # For simplicity, we make `pixel_frames` 5D for VAE encoding for consistency
            pixel_frames_5d = pixel_frames.unsqueeze(2) # (B, C, 1, H, W)
            _, _, x1_latent = self.vae.encode(pixel_frames_5d)
            x1 = x1_latent # (B, C_l, T_l=1, H_l, W_l)
            
            # For image training, we typically only use the finest pyramid stage (k=0)
            # The paper says: "the first frame in a video acts as an image."
            # and DiT usually processes a sequence.
            # So, k=0 means highest resolution.
            pyramid_k = 0
        elif stage_config.dataset_type == "video":
            # VideoDataset should already return