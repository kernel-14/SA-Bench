class MRQConfig:
    # Encoder Hyperparameters
    LAMBDA_DYNAMICS = 1.0 # Assumed, not explicitly stated in Table 3
    LAMBDA_REWARD = 0.1
    LAMBDA_TERMINAL = 0.1 # Corrected from placeholder, based on Table 3
    LAMBDA_PRE_ACTIV = 1e-5 # Interpreted from "1e-5 5" in Table 3
    ENCODER_HORIZON = 3 # Assumed, not explicitly stated in Table 3 but consistent with HQ

    # TD3 Hyperparameters
    MULTI_STEP_RETURNS_HORIZON = 3
    TARGET_POLICY_NOISE_SIGMA = 0.2 # From N(0, 0.2^2)
    TARGET_POLICY_NOISE_CLIP = 0.3 # From (-0.3, 0.3)

    # LAP Hyperparameters
    LAP_PROBABILITY_SMOOTHING_ALPHA = 0.4
    MINIMUM_PRIORITY = 1

    # Exploration Hyperparameters
    EXPLORATION_NOISE = 0.2 # From N(0, 0.2^2)
    INITIAL_RANDOM_EXPLORATION_TIME_STEPS = 10000 # 10k

    # Common Hyperparameters
    DISCOUNT_FACTOR = 0.99
    REPLAY_BUFFER_CAPACITY = 1000000 # 1M
    MINI_BATCH_SIZE = 256
    TARGET_UPDATE_FREQUENCY = 250
    REPLAY_RATIO = 1 # Assumed, not explicitly stated in Table 3

    # Optimizer Hyperparameters (for Encoder, Value, Policy)
    OPTIMIZER = "AdamW"
    LEARNING_RATE_ENCODER_VALUE = 1e-4 # Learning rate for Encoder and Value (from table)
    LEARNING_RATE_POLICY = 3e-4 # Learning rate for Policy (from table)
    WEIGHT_DECAY = 1e-4
    GRADIENT_CLIP_NORM = 20 # For Policy Network

    # Network Architecture Hyperparameters
    ZS_DIM = 512 # State embedding dimension
    ZA_DIM = 256 # Action embedding dimension (only used within architecture)
    ZSA_DIM = 512 # State-action embedding dimension
    HIDDEN_DIM = 512
    ACTIVATION_FUNCTION = "ELU"
    WEIGHT_INITIALIZATION = "Xavier uniform"
    BIAS_INITIALIZATION = 0

    # Reward related Hyperparameters
    REWARD_BINS = 65
    REWARD_RANGE = [-10, 10] # Effective: [-22k, 22k]

    # Policy Network Specific
    GUMBEL_SOFTMAX_TAU = 10

    # Environment specifics (placeholders, these will be passed to the model constructor)
    IMAGE_OBSERVATION_SPACE = False
    STATE_CHANNELS = 3 # For image observations (e.g., RGB)
    STATE_DIM = None # For vector observations
    ACTION_DIM = None # For both continuous and discrete actions
    ACTION_RANGE = [-1, 1] # Assumed default for continuous actions

