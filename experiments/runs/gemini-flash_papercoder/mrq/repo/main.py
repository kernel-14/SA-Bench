import argparse
import os
import random
import numpy as np
import torch
import yaml # Required for loading config.yaml
import sys
from datetime import datetime

# Adjust the Python path to ensure local imports work
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config import Config
from environments import EnvironmentWrapper
from replay_buffer import PrioritizedReplayBuffer
from networks import Models
from agent import MRQAgent
from trainer import Trainer
from metrics import Logger, RewardNormalizer


def main():
    """
    The main entry point for the MR.Q training system.
    Parses arguments, loads configuration, initializes components, and starts training.
    """
    parser = argparse.ArgumentParser(description="MR.Q Reinforcement Learning Agent")
    parser.add_argument("--config-path", type=str, default="config.yaml",
                        help="Path to the configuration YAML file.")
    parser.add_argument("--env-name", type=str, default=None,
                        help="Name of the environment to run (overrides config.environment.env_name).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (overrides config.environment.seed).")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Base directory for TensorBoard logs and checkpoints (overrides config.logging_evaluation.log_dir).")
    parser.add_argument("--device", type=str, default=None,
                        help="Computation device (e.g., 'cuda' or 'cpu', overrides config.device).")
    args = parser.parse_args()

    # 1. Configuration Loading and Merging
    # Load default configuration values from the YAML file
    config = Config.from_yaml(args.config_path)

    # Override config values with command-line arguments if provided
    if args.env_name:
        config.env_name = args.env_name
    if args.seed is not None:
        config.seed = args.seed
    if args.log_dir:
        config.log_dir = args.log_dir
    if args.device:
        config.device = args.device

    # 2. Environment-specific Configuration Adjustment
    env_name_lower = config.env_name.lower()

    if any(s in env_name_lower for s in ["ant", "halfcheetah", "hopper", "humanoid", "walker", "-v4", "-v3"]): # Heuristic for Gym Mujoco
        config.image_obs = False
        config.discrete_actions = False
        config.action_repeat = 1
        config.total_timesteps = 1_000_000 # 1M time steps
        config.eval_interval = 5_000 # 5k time steps
    elif "dmc" in env_name_lower:
        config.action_repeat = 2
        config.total_timesteps = 500_000 # 500k time steps
        config.eval_interval = 5_000 # 5k time steps
        if "visual" in env_name_lower:
            config.image_obs = True
            config.discrete_actions = False
            config.frame_stack = 3 # Previous 3 observations
        else: # Proprioceptive DMC
            config.image_obs = False
            config.discrete_actions = False
            config.frame_stack = 1 # Vector observations, no explicit frame stacking
    elif "ale/" in env_name_lower or "atari" in env_name_lower:
        config.image_obs = True
        config.discrete_actions = True
        config.action_repeat = 4
        config.total_timesteps = 2_500_000 # 2.5M time steps
        config.eval_interval = 100_000 # 100k time steps
        config.frame_stack = 4 # Previous 4 processed observations
    else:
        print(f"Warning: Environment name '{config.env_name}' not recognized for automatic config adjustment. "
              "Using default values from config.yaml and CLI. This might lead to incorrect experimental setup.")

    # Print the final effective configuration
    config.print_config()

    # 3. Device Setup
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 4. Seed Initialization
    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True # For reproducibility
        torch.backends.cudnn.benchmark = False    # For reproducibility

    # 5. Logger Initialization
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(config.log_dir, config.env_name, f"seed_{seed}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    logger = Logger(log_dir=log_dir)
    logger.log_hparams(config.to_dict()) # Save config to TensorBoard HParams

    # 6. Environment Setup
    env_wrapper = EnvironmentWrapper(config)
    obs_space_info = env_wrapper.get_observation_space_info()
    action_space_info = env_wrapper.get_action_space_info()

    # 7. Replay Buffer Initialization
    # k_step_sampling in replay_buffer.py init is not needed, max_horizon is passed to sample method.
    replay_buffer = PrioritizedReplayBuffer(
        capacity=config.replay_buffer_capacity,
        obs_shape=obs_space_info["shape"],
        action_dim=action_space_info["dim"],
        device=device,
        alpha=config.prioritized_replay_alpha,
        min_priority_initial=config.min_priority,
    )

    # 8. Reward Normalizer Initialization
    reward_normalizer = RewardNormalizer()

    # 9. Models Initialization
    models = Models(config, obs_space_info, action_space_info, device)

    # 10. Agent Initialization
    agent = MRQAgent(config, models, reward_normalizer, device, action_space_info)

    # 11. Trainer Initialization and Execution
    trainer = Trainer(config, env_wrapper, replay_buffer, agent, models, reward_normalizer, logger)
    trainer.run()

    # 12. Post-training Actions
    print(f"Training completed for {config.env_name} with seed {seed}.")
    env_wrapper.close()
    logger.close()

if __name__ == "__main__":
    main()

