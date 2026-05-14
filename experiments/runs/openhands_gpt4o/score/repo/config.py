# Configuration file for SCoRe

# Model configuration
BASE_MODEL_NAME = "gpt2"
KL_BETA = 0.01
REWARD_ALPHA = 10

# Training configuration
LEARNING_RATE = 5e-6
BATCH_SIZE = 32
NUM_EPOCHS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset configuration
DATASET_NAME = "MATH"