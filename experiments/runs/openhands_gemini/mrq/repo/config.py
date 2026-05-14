
import torch

class Config:
    # General
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 0
    NUM_SEEDS = 10 # For experiments
    # Environment
    ENV_NAME = "CartPole-v1" # Placeholder, actual environments vary
    ACTION_REPEAT = 2 # For DMC, 4 for Atari
    IMAGE_SIZE = (84, 84) # For visual observations
    FRAME_STACK = 3 # For visual DMC, 4 for Atari

    # Model Architecture (B.2 NETWORK ARCHITECTURE)
    ZS_DIM = 512
    ZA_DIM = 256
    ZSA_DIM = 512
    HIDDEN_DIM = 512
    STATE_CHANNELS = 3 # For RGB images, adjust for grayscale
    REWARD_BINS = 65
    REWARD_RANGE = [-10, 10] # Effective: [-22k, 22k]

    # Encoder (4.2.1 ENCODER & B.1 HYPERPARAMETERS)
    LAMBDA_DYNAMICS = 1.0 # Default, no explicit value given but implied by sum
    LAMBDA_REWARD = 0.1
    LAMBDA_TERMINAL = 0.1
    ENCODER_HORIZON = 5 # H_Enc

    # TD3 (4.2.2 VALUE FUNCTION & B.1 HYPERPARAMETERS)
    MULTI_STEP_RETURNS_HORIZON = 3 # H_Q
    TARGET_POLICY_NOISE_STD = 0.2
    TARGET_POLICY_NOISE_CLIP = 0.5 # Corresponds to c from table. The paper states (-0.3, 0.3) for clipping with noise std of 0.22. Clipping to 0.5 (abs) for (N(0, 0.2)^2) seems more consistent with general TD3. Let's use 0.5 for a general clip.

    # LAP (B.1 HYPERPARAMETERS)
    PROBABILITY_SMOOTHING_ALPHA = 0.4
    MINIMUM_PRIORITY = 1.0

    # Exploration (B.1 HYPERPARAMETERS)
    EXPLORATION_NOISE_STD = 0.2
    INITIAL_RANDOM_EXPLORATION_STEPS = 10000

    # Common (B.1 HYPERPARAMETERS)
    DISCOUNT_FACTOR = 0.99 # gamma
    REPLAY_BUFFER_CAPACITY = 1_000_000 # 1M
    MINI_BATCH_SIZE = 256
    TARGET_UPDATE_FREQUENCY = 250 # T_target
    REPLAY_RATIO = 1 # Not explicitly mentioned but common for off-policy.
    OPTIMIZER = "AdamW"
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    ACTIVATION_FUNCTION = "ELU" # For most networks
    POLICY_ACTIVATION_FUNCTION = "ReLU" # For policy network hidden layers
    WEIGHT_INITIALIZATION = "Xavier uniform"
    BIAS_INITIALIZATION = 0
    GRADIENT_CLIP_NORM = 20 # For policy network, no explicit value for others

    # Policy Network Specific (B.1 HYPERPARAMETERS)
    LAMBDA_PRE_ACTIV = 1e-5 # Pre-activation loss weight
    GUMBEL_SOFTMAX_TAU = 10 # For discrete actions

    # Training
    TOTAL_TIME_STEPS = 1_000_000 # 1M for Gym, 500k for DMC, 2.5M for Atari
    EVAL_FREQUENCY = 5000 # 5k for Gym/DMC, 100k for Atari
