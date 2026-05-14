import torch

class AgentConfig:
    """
    Configuration for the Deep Repeated ConvLSTM (DRC) agent.
    """
    D_CONVLSTM_LAYERS: int = 3
    N_INTERNAL_TICKS: int = 3
    CHANNELS: int = 32  # G_d
    KERNEL_SIZE: int = 3
    PADDING: int = 1  # Single layer of zero padding for kernel size 3
    GRID_SIZE: int = 8  # H_d, W_d
    OBSERVATION_CHANNELS: int = 7 # Sokoban symbolic representation x_t in R^(8x8x7)
    NUM_ACTIONS: int = 5 # Up, Down, Left, Right, No-op (not explicitly stated for No-op, but standard for grid worlds)

class TrainingConfig:
    """
    Configuration for training the DRC agent.
    """
    TOTAL_TRANSITIONS: int = 250_000_000
    UNFILTERED_BOXOBAN_TRAINING_LEVELS: int = 900_000 # Guez et al. (2018a)
    ALGORITHM: str = "IMPALA"
    DISCOUNT_RATE: float = 0.97  # gamma
    V_TRACE_LAMBDA: float = 0.97  # lambda
    L2_ACTION_LOGITS_PENALTY: float = 1e-3
    L2_POLICY_VALUE_HEADS_REGULARIZATION: float = 1e-5
    ENTROPY_PENALTY: float = 1e-2
    UNROLL_LENGTH: int = 20
    OPTIMIZER: str = "Adam"
    BATCH_SIZE: int = 16
    LEARNING_RATE_MAX: float = 4e-4
    LEARNING_RATE_MIN: float = 0.0
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SokobanEnvConfig:
    """
    Configuration for the Sokoban environment.
    """
    GRID_SIZE: int = 8
    NUM_BOXES: int = 4
    NUM_TARGETS: int = 4
    SYMBOLIC_CHANNELS: int = 7
    REWARD_STEP: float = -0.01
    REWARD_BOX_ON_TARGET: float = 1.0
    REWARD_BOX_OFF_TARGET: float = -1.0
    REWARD_LEVEL_SOLVED: float = 10.0
    EPISODE_LENGTH_MIN: int = 115
    EPISODE_LENGTH_MAX: int = 120

class ProbeConfig:
    """
    Configuration for linear probing experiments.
    """
    PROBE_EPOCHS: int = 10
    PROBE_BATCH_SIZE: int = 16
    PROBE_LEARNING_RATE: float = 0.001
    PROBE_WEIGHT_DECAY: float = 0.001
    PROBE_OPTIMIZER: str = "AdamW"
    NUM_PROBE_SEEDS: int = 5
    PROBE_TRAIN_EPISODES: int = 3000 # For fully trained agent
    PROBE_TEST_EPISODES: int = 1000  # For fully trained agent
    PROBE_TRAIN_TRANSITIONS: int = 106_600 # Approx. from 3000 episodes
    PROBE_TEST_TRANSITIONS: int = 25_700   # Approx. from 1000 episodes
    PROBE_TYPES: list[str] = ["1x1", "3x3", "5x5", "7x7"] # From paper and appendix
    
    # Concept classes
    CONCEPT_CA_CLASSES: list[str] = ["UP", "DOWN", "LEFT", "RIGHT", "NEVER"]
    CONCEPT_CB_CLASSES: list[str] = ["UP", "DOWN", "LEFT", "RIGHT", "NEVER"]

class EvalConfig:
    """
    Configuration for evaluation and intervention experiments.
    """
    THINKING_STEPS: int = 5
    MEDIUM_LEVELS_COUNT: int = 1000
    HARD_LEVELS_COUNT: int = 1000
    INTERVENTION_ALPHA: float = 1.0 # Default scaling factor, varied in experiments
    INTERVENTION_DIRECTIONAL_SQUARES: int = 1 # Default, varied in experiments
