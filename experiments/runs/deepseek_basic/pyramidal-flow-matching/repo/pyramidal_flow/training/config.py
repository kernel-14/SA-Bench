"""
Training Configuration for Pyramidal Flow Matching.

Contains the hyperparameters and settings used in the paper's
three-stage training procedure (Table 4).
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class TrainingConfig:
    """Training hyperparameters matching Table 4 in the paper."""
    
    # Optimizer
    optimizer: str = "AdamW"
    beta1: float = 0.9
    beta2: float = 0.999  # Stage 1: 0.999, Stage 2-3: 0.95
    epsilon: float = 1e-6
    weight_decay: float = 1e-4
    
    # Learning rate
    learning_rate: float = 1e-4
    lr_schedule: str = "constant_with_warmup"
    warmup_steps: int = 1000
    
    # Batch
    global_batch_size: int = 1536
    
    # Training steps
    max_steps: int = 50000
    gradient_clipping: float = 1.0
    
    # Precision
    numerical_precision: str = "bfloat16"
    
    # GPU
    num_gpus: int = 128
    
    # Flow matching
    num_spatial_stages: int = 3
    num_temporal_levels: int = 3
    max_history_frames: int = 12
    gamma: float = -1.0 / 3.0
    
    # History noise range for training (Section 3.3)
    history_noise_min: float = 0.0
    history_noise_max: float = 1.0 / 3.0
    
    # Model architecture
    hidden_dim: int = 3072
    num_heads: int = 24
    num_layers: int = 24
    input_dim: int = 16  # VAE latent channels
    text_embed_dim: int = 4096
    use_causal_attention: bool = True
    
    # Data
    image_size: Tuple[int, int] = (768, 768)
    latent_size: Tuple[int, int] = (96, 96)  # 768/8
    max_video_frames: int = 241  # 10s at 24fps
    fps: int = 24
    
    # Classifier-free guidance
    cfg_dropout_prob: float = 0.1  # Probability of dropping text conditioning
    
    # Checkpointing
    save_every: int = 5000
    log_every: int = 100
    
    def get_stage_config(self, stage: int) -> 'TrainingConfig':
        """Get configuration for a specific training stage (1, 2, or 3)."""
        config = TrainingConfig()
        
        if stage == 1:
            # Image training
            config.beta2 = 0.999
            config.global_batch_size = 1536
            config.learning_rate = 1e-4
            config.max_steps = 50000
            config.max_video_frames = 1  # single images
        elif stage == 2:
            # Low-resolution video training
            config.beta2 = 0.95
            config.global_batch_size = 768
            config.learning_rate = 1e-4
            config.max_steps = 200000
            config.max_video_frames = 121  # 5s at 24fps
        elif stage == 3:
            # High-resolution video fine-tuning
            config.beta2 = 0.95
            config.global_batch_size = 384
            config.learning_rate = 5e-5
            config.max_steps = 50000
            config.max_video_frames = 241  # 10s at 24fps
        
        return config
    
    def get_gpu_hours(self, stage: int) -> int:
        """Get estimated GPU hours for a stage (from Table 4)."""
        stage_hours = {1: 1536, 2: 11520, 3: 7680}
        return stage_hours.get(stage, 0)
