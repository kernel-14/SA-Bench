import yaml
import os
from typing import List, Tuple

# Define the path to the config.yaml file
CONFIG_FILE_PATH: str = os.path.join(os.path.dirname(__file__), 'config.yaml')

# Load configuration from the YAML file
with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# --- Training Parameters ---

# LEARNING_RATE (float): The step size (η) for the Projected Policy Gradient (PPG) algorithm.
# Sourced from: config.yaml -> training.learning_rate
LEARNING_RATE: float = float(config['training']['learning_rate'])

# NUM_ITERATIONS_EXP1 (int): Number of policy gradient iterations for Experiment 1.
# Sourced from: config.yaml -> training.num_iterations_exp1
NUM_ITERATIONS_EXP1: int = int(config['training']['num_iterations_exp1'])

# NUM_ITERATIONS_EXP2 (int): Number of policy gradient iterations for Experiment 2.
# Sourced from: config.yaml -> training.num_iterations_exp2
NUM_ITERATIONS_EXP2: int = int(config['training']['num_iterations_exp2'])

# NUM_ITERATIONS_EXP3 (int): Number of policy gradient iterations for Experiment 3.
# Sourced from: config.yaml -> training.num_iterations_exp3
NUM_ITERATIONS_EXP3: int = int(config['training']['num_iterations_exp3'])

# --- MDP Configuration for Experiments ---

# MDP_SIZES_EXP1 (List[Tuple[int, int]]): List of (S, A) tuples for varying MDP sizes in Experiment 1.
# Sourced from: config.yaml -> mdp_config.experiment1.mdp_sizes
MDP_SIZES_EXP1: List[Tuple[int, int]] = [
    (int(s), int(a)) for s, a in config['mdp_config']['experiment1']['mdp_sizes']
]

# FIXED_S_EXP2_3 (int): Fixed state space size for Experiment 2 and Experiment 3.
# Sourced from: config.yaml -> mdp_config.experiment2.fixed_S
# Note: As per the plan, fixed_S from experiment2 section is used for both experiment 2 and 3.
FIXED_S_EXP2_3: int = int(config['mdp_config']['experiment2']['fixed_S'])

# FIXED_A_EXP2_3 (int): Fixed action space size for Experiment 2 and Experiment 3.
# Sourced from: config.yaml -> mdp_config.experiment2.fixed_A
# Note: As per the plan, fixed_A from experiment2 section is used for both experiment 2 and 3.
FIXED_A_EXP2_3: int = int(config['mdp_config']['experiment2']['fixed_A'])

# REWARD_VARIANCE_LEVELS_EXP2 (List[str]): Identifiers for different reward variance levels in Experiment 2.
# Sourced from: config.yaml -> mdp_config.experiment2.reward_variance_levels
REWARD_VARIANCE_LEVELS_EXP2: List[str] = config['mdp_config']['experiment2']['reward_variance_levels']

# TRANSITION_TYPES_EXP3 (List[str]): Identifiers for different transition kernel types in Experiment 3.
# Sourced from: config.yaml -> mdp_config.experiment3.transition_types
TRANSITION_TYPES_EXP3: List[str] = config['mdp_config']['experiment3']['transition_types']

# EXP1_REWARD_VARIANCE_LEVEL (str): Reward variance level for Experiment 1.
# Sourced from: config.yaml -> mdp_config.experiment1.reward_variance_level
EXP1_REWARD_VARIANCE_LEVEL: str = str(config['mdp_config']['experiment1']['reward_variance_level'])

# EXP3_REWARD_VARIANCE_LEVEL (str): Reward variance level for Experiment 3.
# Sourced from: config.yaml -> mdp_config.experiment3.reward_variance_level
EXP3_REWARD_VARIANCE_LEVEL: str = str(config['mdp_config']['experiment3']['reward_variance_level'])

# --- General Settings ---

# RANDOM_SEED (int): Seed for numpy's random number generator to ensure reproducibility.
# Sourced from: config.yaml -> general.random_seed
RANDOM_SEED: int = int(config['general']['random_seed'])

# PLOT_DIR (str): Directory path to save generated plots.
# Sourced from: config.yaml -> general.plot_dir
PLOT_DIR: str = str(config['general']['plot_dir'])
