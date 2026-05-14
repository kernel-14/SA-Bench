
import torch

class Config:
    # General
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 # Bfloat16 precision

    # SDE and Noise Schedule
    K_TIMESTEPS = 40  # Number of timesteps for fine-tuning and inference
    H = 1.0 / K_TIMESTEPS # Step size
    ALPHA_T_FN = lambda t: t # alpha_t = t
    BETA_T_FN = lambda t: 1 - t # beta_t = 1 - t
    
    # Noise schedule for fine-tuning (memoryless noise schedule)
    # sigma(t) = sqrt(2 * (1 - t + h) / (t + h)) as per G.1
    def get_sigma_t(self, t: float):
        return torch.sqrt(2 * (self.BETA_T_FN(t) + self.H) / (self.ALPHA_T_FN(t) + self.H))

    # Optimizer
    LEARNING_RATE = 2e-5
    ADAM_BETA1 = 0.95
    ADAM_BETA2 = 0.999
    ADAM_EPS = 1e-8
    WEIGHT_DECAY = 1e-2
    GRADIENT_NORM_CLIPPING = 1.0

    # Training
    NUM_FINE_TUNE_ITERATIONS = 10000 # Example, paper implies 1000 iter per epoch with 40k prompts
    BATCH_SIZE = 20 # Per GPU, effective batch size 40 with 2 GPUs
    EFFECTIVE_BATCH_SIZE = 40
    NUM_PROMPTS_PER_EPOCH = 40000
    TOTAL_PROMPTS_AVAILABLE = 100000 # Total pool from which fine-tuning prompts are sampled

    # Adjoint Matching Specific
    LAMBDA_REWARD_SCALING = 12500 # Example value, can be 1000, 2500, 12500
    LCT_ADJOINT_MATCHING = 1.6 * LAMBDA_REWARD_SCALING**2 # Loss Clipping Threshold
    LCT_CONTINUOUS_ADJOINT = 1600 * LAMBDA_REWARD_SCALING**2 # For comparison/baseline

    # Gradient Evaluation Timesteps (G.2)
    NUM_UNIFORM_TIMESTEPS_FOR_GRAD = 10
    UNIFORM_TIMESTEPS_RANGE_END = 0.725
    NUM_LAST_TIMESTEPS_FOR_GRAD = 10
    LAST_TIMESTEPS_RANGE_START = 0.75

    # Reward Model
    REWARD_MODEL_NAME = "ImageReward" # Xu et al., 2023
    # r(x) = lambda * RewardModel(x)
    
    # Model (U-Net)
    UNET_CONFIG = { # Placeholder, specific details for U-Net not fully provided in paper
        "in_channels": 4, # Latent diffusion setup
        "out_channels": 4,
        "down_block_types": ("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        "up_block_types": ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        "block_out_channels": (320, 640, 1280, 1280),
        "layers_per_block": 2,
        "cross_attention_dim": 768, # CLIP text embedding dimension
        "attention_head_dim": 8,
    }
