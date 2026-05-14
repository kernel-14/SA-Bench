
import os
import torch

class Config:
    # Environment
    ENV_NAME = "dmc"  # or "gym"
    DMC_TASK = "quadruped-walk" # Example: "cheetah-run", "reacher-hard", "finger-turn-hard"
    DMC_PIXEL_TASK = "walker-walk" # Example: "cheetah-run"
    OBS_TYPE = "state" # "state" or "pixel"
    ACTION_REPEAT = 1 # Default for DMC, some tasks (like DMLab) may use 4

    # Training
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TOTAL_ENV_STEPS = 100_000 # 100K for most tasks, 300K for Finger-Turn-Hard
    BATCH_SIZE = 256
    BUFFER_SIZE = 1_000_000 # 1M transitions
    UTD_RATIO = 20 # Update-to-data ratio
    GRADIENT_STEPS = 1 # Policy gradient steps per update
    POLICY_LR = 1e-4
    Q_LR = 1e-4
    ACTOR_BETA = 0.9 # SAC actor beta for polyak update
    CRITIC_BETA = 0.9 # SAC critic beta for polyak update
    DISCOUNT = 0.99
    TAU = 0.005 # Polyak update coefficient for target networks
    TARGET_UPDATE_INTERVAL = 1

    # Generative Model (Diffusion)
    GENERATIVE_MODEL_TYPE = "diffusion" # "diffusion" or "vae"
    DIFF_LR = 1e-4
    DIFF_EMBED_DIM = 256 # Example, not specified directly, typical for latent diffusion
    DIFF_TIME_EMBED_DIM = 256
    DIFF_N_TIMESTEPS = 1000 # Default for DDPM
    DIFF_DROP_UNCOND_PROB = 0.25 # Probability of dropping condition y for CFG
    DIFF_GUIDANCE_SCALE = 7.5 # Hyperparameter omega for CFG (common value)
    DIFF_MODEL_CAPACITY_FACTOR = 1.0 # Multiplier for diffusion model layers/width if scaling
    DIFF_TRAIN_EPOCHS = 1 # Number of epochs to train diffusion model per inner loop

    # PGR Specific
    SYNTHETIC_DATA_RATIO = 0.5 # r in Algorithm 1, ratio of synthetic data in mixed batch
    SYNTHETIC_BUFFER_SIZE = 1_000_000 # 1M transitions for D_syn
    GENERATION_BATCH_SIZE = 1024 # Batch size for generating synthetic data
    RELEVANCE_FUNCTION = "curiosity" # "return", "td_error", "curiosity", "rnd", "cts", "eco"
    INNER_LOOP_FREQUENCY = 10_000 # Retrain generative model every 10K environment steps

    # Relevance Function - Curiosity (ICM)
    ICM_FEATURE_DIM = 256 # Latent space dimension for feature encoder h
    ICM_HIDDEN_DIM = 256 # Hidden dimension for forward dynamics model g
    ICM_LR = 1e-4
    ICM_UPDATE_RATIO = 0.05 # Updated for 5% of policy gradient steps

    # Relevance Function - RND
    RND_FEATURE_DIM = 512 # Feature output dimension, Table A.1
    RND_LATENT_DIM = 64 # Bottleneck latent dimension, Table A.1
    RND_LR = 1e-4

    # Relevance Function - CTS (Context-Tree Switching)
    CTS_IMAGE_SIZE = 42 # Resized visual observations for CTS, Table A.1
    CTS_CONTEXT_BINS = 8 # Context bins for CTS, Table A.1
    CTS_LR = 1e-4

    # Relevance Function - ECO (Episodic Curiosity)
    ECO_ALPHA = 0.03 # Table A.2
    ECO_BETA = 0.5 # Table A.2
    ECO_MEMORY_SIZE = 200 # |M|, Table A.2
    ECO_PERCENTILE = 90 # F = percentile-90, Table A.2
    ECO_EMBEDDER_OUT_DIM = 512 # ResNet-18 output dim, Table A.2
    ECO_MLP_LAYERS = 4 # Four-layer MLP, Table A.2
    ECO_MLP_DIM = 512 # Feature and output dimensions, Table A.2
    ECO_LR = 1e-4

    # Policy Network (MLP for state-based, CNN for pixel-based)
    POLICY_HIDDEN_LAYERS = 2 # Default for state-based, can be 3 for scaling
    POLICY_HIDDEN_DIM = 256 # Default for state-based, can be 512 for scaling
    CNN_ENCODER_TYPE = "drq_v2" # As in Lu et al. (2022)
    CNN_ENCODER_OUT_DIM = 50 # Example output dim for CNN encoder

    # Evaluation
    EVAL_INTERVAL = 10_000
    EVAL_EPISODES = 10

    # Logging and Checkpointing
    LOG_DIR = "runs"
    SAVE_MODEL_INTERVAL = 50_000

    # Ablations/Scaling (from Section 5.3)
    SCALE_POLICY_NETWORK = False # If True, use larger policy network (3 layers, 512 dim)
    INCREASE_SYNTHETIC_RATIO = False # If True, use r=0.75 or r=0.875
    INCREASE_UTD_RATIO = False # If True, use UTD=40 and D_syn=2M

    def __init__(self):
        if self.SCALE_POLICY_NETWORK:
            self.POLICY_HIDDEN_LAYERS = 3
            self.POLICY_HIDDEN_DIM = 512
            self.BATCH_SIZE = 1024 # Increased for per-parameter throughput
        if self.INCREASE_SYNTHETIC_RATIO:
            # Assumes base batch size of 256 and 128 real transitions
            # r=0.75 -> batch_size=512 (128 real, 384 syn)
            # r=0.875 -> batch_size=1024 (128 real, 896 syn)
            if self.SYNTHETIC_DATA_RATIO == 0.75:
                self.BATCH_SIZE = 512
            elif self.SYNTHETIC_DATA_RATIO == 0.875:
                self.BATCH_SIZE = 1024
            else:
                raise ValueError("SYNTHETIC_DATA_RATIO must be 0.75 or 0.875 when INCREASE_SYNTHETIC_RATIO is True")
        if self.INCREASE_UTD_RATIO:
            self.UTD_RATIO = 40
            self.SYNTHETIC_BUFFER_SIZE = 2_000_000 # 2M transitions
            self.BATCH_SIZE = 512 # From Fig 7c setup
            self.SYNTHETIC_DATA_RATIO = 0.75 # From Fig 7c setup

        if self.OBS_TYPE == "pixel":
            # Pixel-based tasks typically use CNN encoders
            self.POLICY_HIDDEN_LAYERS = None # MLP not directly used for observations
            self.POLICY_HIDDEN_DIM = None
            self.DMC_TASK = self.DMC_PIXEL_TASK # Ensure pixel task is selected
            # Latent space generation is used for pixel observations
            # self.DIFF_EMBED_DIM would be CNN_ENCODER_OUT_DIM

        # Specific task adjustments
        if self.DMC_TASK == "finger-turn-hard":
            self.TOTAL_ENV_STEPS = 300_000


cfg = Config()
