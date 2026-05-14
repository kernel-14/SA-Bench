
import torch

class ModelConfig:
    # Model architecture parameters, inspired by Open-Sora and similar Transformer-based VDMs
    latent_channels: int = 4 # VAE latent channels
    model_channels: int = 320 # Base channel dimension for the UNet/Transformer
    num_res_blocks: int = 2 # Number of ResNet blocks per stage
    channel_mult: tuple = (1, 2, 4, 4) # Channel multipliers for each UNet stage
    num_heads: int = 8 # Number of attention heads
    transformer_depth: int = 1 # Number of transformer blocks per stage
    context_dim: int = 768 # Dimension for text conditioning (e.g., T5-large output)
    num_frames: int = 16 # Default number of frames for a clip during training
    resolution: int = 256 # Spatial resolution (H=W)

    # Ca2-VDM specific
    chunk_length: int = 16 # l: length of frames generated per AR step (T2V) or 8 for Video Prediction
    max_condition_frames: int = 49 # P_max: Max length of clean prefix for T2V, 25 for Video Prediction
    prefix_enhancement_frames: int = 3 # P': sub-prefix length for spatial attention

class DiffusionConfig:
    timesteps: int = 1000 # Total diffusion timesteps
    beta_schedule: str = "linear" # "linear" (DDPM) or "cosine"
    beta_start: float = 0.0001 # Beta_1 for DDPM
    beta_end: float = 0.02 # Beta_T for DDPM
    num_inference_steps: int = 100 # Number of steps for inference (Improved DDPM)
    guidance_scale: float = 7.5 # Classifier-free guidance scale for T2V

class TrainingConfig:
    task_type: str = "text_to_video" # "text_to_video" or "video_prediction"
    epochs: int = 10 # Example, actual steps are defined by paper
    batch_size: int = 1 # Per GPU batch size
    learning_rate: float = 2e-5 # AdamW learning rate
    weight_decay: float = 0.01 # Decoupled weight decay regularization
    gradient_accumulation_steps: int = 1
    # Specific to T2V training (InternVid)
    t2v_first_stage_frames: int = 32 # For causal modeling without clean prefix
    t2v_first_stage_steps: int = 32000 # 32k steps
    t2v_first_stage_batch_size: int = 288
    t2v_second_stage_frames: int = 65 # P_max + l
    t2v_second_stage_steps: int = 21000 # 21k steps
    t2v_second_stage_batch_size: int = 144
    # Specific to Video Prediction training (SkyTimelapse)
    vp_train_frames: int = 33 # P_max + l
    vp_train_steps: int = 11000 # 11k steps
    vp_train_batch_size: int = 8
    vp_fixed_P: int = 8 # For OS-Fix baseline

    # Loss weights for combined loss
    lambda_vlb: float = 0.001 # Weight for the VLB loss (D_KL term)

class DataConfig:
    dataset_name: str = "internvid" # "internvid", "skytimelapse", "msrvtt", "ucf101"
    data_path: str = "./data" # Base path for datasets
    image_size: int = 256 # Input image size
    num_workers: int = 8 # Dataloader workers

class SystemConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    mixed_precision: str = "fp16" # "no", "fp16", "bf16"
    output_dir: str = "./output"

# Combine all configs
class Config:
    model: ModelConfig = ModelConfig()
    diffusion: DiffusionConfig = DiffusionConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    system: SystemConfig = SystemConfig()

    def __init__(self, task_type: str = "text_to_video"):
        self.training.task_type = task_type
        if task_type == "text_to_video":
            self.model.chunk_length = 16
            self.model.max_condition_frames = 49 # 1 + 3 * 16
            self.training.batch_size = self.training.t2v_second_stage_batch_size # Default to second stage batch size
            self.data.dataset_name = "internvid"
        elif task_type == "video_prediction":
            self.model.chunk_length = 8
            self.model.max_condition_frames = 25 # 1 + 3 * 8
            self.training.batch_size = self.training.vp_train_batch_size
            self.data.dataset_name = "skytimelapse"
