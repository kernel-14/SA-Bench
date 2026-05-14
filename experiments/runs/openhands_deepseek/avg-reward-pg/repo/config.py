"""Configuration for average reward policy gradient experiments."""

# Step size for projected policy gradient
# Should be < 1 / L2^Pi for guaranteed convergence
ETA = 0.01

# Number of iterations for PPG
NUM_ITERS_EXP1 = 2000
NUM_ITERS_EXP2 = 3000
NUM_ITERS_EXP3 = 3000

# Tracking frequency
TRACK_EVERY = 50

# State/action sizes for Experiment 1
SIZES_EXP1 = [(3, 3), (9, 9), (81, 81)]

# State/action size for Experiments 2 and 3
S_EXP23 = 16
A_EXP23 = 16

# Number of restarts for finding empirical optimal policy
N_RESTARTS_OPT = 10

# Number of samples for computing MDP constants
N_SAMPLES_CONSTANTS = 100

# Random seeds
SEED = 42
