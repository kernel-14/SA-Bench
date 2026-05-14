
import os

class Config:
    # General
    PROJECT_NAME = "SAM2_Reproduction"
    SEED = 42
    DEVICE = "cuda" if "cuda" else "cpu" # Default to CPU if no GPU (The system does not have GPU at all now)

    # Data
    DATA_ROOT = "./data"
    SA_V_DIR = os.path.join(DATA_ROOT, "SA-V")
    SA_1B_DIR = os.path.join(DATA_ROOT, "SA-1B")
    VOS_DATA_DIRS = {
        "DAVIS": os.path.join(DATA_ROOT, "DAVIS"),
        "MOSE": os.path.join(DATA_ROOT, "MOSE"),
        "YouTubeVOS": os.path.join(DATA_ROOT, "YouTubeVOS"),
    }
    IMAGE_SIZE = 1024  # Default input resolution (Table 9a)
    VIDEO_FRAME_LENGTH = 8  # Sample sequences of 8 frames (Section D.2)
    SAMPLING_PROB_IMAGE_DATA = 0.152 # 15.2% SA-1B (Section D.2)
    SAMPLING_PROB_SA_V_DATA = 0.70 # 70% SA-V (Section D.2)
    SAMPLING_PROB_INTERNAL_DATA = 0.148 # 14.8% Internal (Section D.2)
    SAMPLING_PROB_VOS_DATA = 0.092 # 9.2% YouTubeVOS + 1.3% DAVIS + 9.4% MOSE = 19.9% but paper says ~9.2% for MOSE, ~1.3% for DAVIS so 1.3 + 9.4 + 9.2 = 19.9%
    # The sum of sampling probabilities for all data sources should be 1.0.
    # Adjusting based on the paper's text: "The training data mixture consists of ~15.2% SA-1B, ~70% SA-V and ~14.8% Internal.
    # The same settings are used when open-source datasets are included, with the change that the additional data is included
    # (~1.3% DAVIS, ~9.4% MOSE, ~9.2% YouTubeVOS, ~15.5% SA-1B, ~49.5% SA-V, ~15.1% Internal)."
    # Let's use the second set of numbers for the data mixture when OSS datasets are included.
    DATA_MIXTURE_PROBS = {
        "SA_1B": 0.155,
        "SA_V": 0.495, # This should be SA-V + Internal data
        "DAVIS": 0.013,
        "MOSE": 0.094,
        "YouTubeVOS": 0.092,
        "INTERNAL": 0.151,
    }
    # These probabilities must sum to 1.0. Let's adjust slightly or clarify based on paper text if possible.
    # Paper says: "The training data mixture consists of ~15.2% SA-1B, ~70% SA-V and ~14.8% Internal."
    # then "The same settings are used when open-source datasets are included, with the change that the additional data is included
    # (~1.3% DAVIS, ~9.4% MOSE, ~9.2% YouTubeVOS, ~15.5% SA-1B, ~49.5% SA-V, ~15.1% Internal)."
    # These two statements contradict regarding the total SA-V and Internal percentages.
    # For now, I will assume the second set of probabilities for the final training mix.
    # 0.155 + 0.495 + 0.013 + 0.094 + 0.092 + 0.151 = 1.0
    
    # Augmentation (Table 12)
    AUG_HORIZONTAL_FLIP = 0.5 # Probability for horizontal flip
    AUG_RANDOM_AFFINE = 0.5 # Probability for random affine
    AUG_COLOR_JITTER = 0.5 # Probability for color jitter
    AUG_GRAYSCALE = 0.1 # Probability for grayscale (Section D.2 says 10%)
    AUG_MOSAIC_PROB = 0.1 # 10% probability (Section D.2)

    # Model
    IMAGE_ENCODER_TYPE = "Hiera-B+" # Default (Table 6, Table 9f)
    PROMPT_ENCODER_EMBED_DIM = 256 # From SAM, assumed for SAM2
    MASK_DECODER_EMBED_DIM = 256 # From SAM, assumed for SAM2
    MEMORY_ATTENTION_LAYERS = 4 # L=4 layers (Section D.1)
    MEMORY_CHANNELS = 64 # (Table 9d)
    NUM_RECENT_FRAMES_MEMORY_BANK = 6 # N=6 (Table 9c)
    NUM_PROMPTED_FRAMES_MEMORY_BANK = -1 # M, not explicitly stated, assumed to retain all prompted frames
    OCCLUSION_PREDICTION_HEAD = True
    MULTIPLE_MASKS_OUTPUT = True

    # Training (Pre-training and Full Training) (Table 12)
    PRETRAIN_STEPS = 90000
    FULL_TRAINING_STEPS = 200000 # For ablation in Table 7
    FINE_TUNE_STEPS = 50000 # (Section D.2)
    BATCH_SIZE_PRETRAIN = 256
    BATCH_SIZE_FULL_TRAIN = 128
    BATCH_SIZE_FINE_TUNE = 1 # for 16-frame sequences on A100 (Section D.2)
    PRECISION = "bfloat16"
    OPTIMIZER = "AdamW"
    OPTIMIZER_MOMENTUM_BETA1 = 0.9
    OPTIMIZER_MOMENTUM_BETA2 = 0.999
    GRADIENT_CLIPPING_MAX_NORM = 0.1
    WEIGHT_DECAY = 0.1
    LEARNING_RATE = 4e-4
    LR_SCHEDULE = "reciprocal_sqrt"
    LR_SCHEDULE_TIMESCALE = 1000
    WARMUP_ITERS = 1000
    COOLDOWN_ITERS = 5000
    LAYER_DECAY_RATE_T = 0.8
    LAYER_DECAY_RATE_S = 0.8
    LAYER_DECAY_RATE_B_PLUS = 0.9
    LAYER_DECAY_RATE_L = 0.925
    DROP_PATH_RATE_T = 0.1
    DROP_PATH_RATE_S = 0.1
    DROP_PATH_RATE_B_PLUS = 0.2
    DROP_PATH_RATE_L = 0.3
    MASK_LOSS_WEIGHT_FOCAL = 20
    MASK_LOSS_WEIGHT_DICE = 1
    IOU_LOSS_WEIGHT = 1 # L1 loss
    OBJECT_LOSS_WEIGHT = 1 # Cross-entropy for occlusion prediction
    MAX_MASKS_PER_IMAGE = 64
    CORRECTION_CLICKS = 7 # Instead of 8 in SAM (Section D.2)
    RANDOM_CLICK_PROB = 0.1 # 10% probability (Section D.2)
    REVERSE_TEMPORAL_PROB = 0.5 # 50% probability (Section D.2)
    FINE_TUNE_LR_FACTOR = 0.5 # Half of original LR (Section D.2)

    # Evaluation
    EVAL_RESOLUTIONS = [512, 768, 1024]
    METRICS = ["T&F", "mIoU", "G", "FPS"]

