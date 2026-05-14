# main.py

import argparse
import os
import sys
import random
import numpy as np
import torch
from typing import Dict, Any, Optional

# Import core utilities and experiment runner
from config import Config
from utils.logger import Logger
from experiments.runner import ExperimentRunner


def main() -> None:
    """
    Main entry point for the emergent planning reproduction project.
    Parses command-line arguments, initializes configuration and logger, and dispatches
    to the ExperimentRunner to execute specified experiments.
    """
    parser = argparse.ArgumentParser(
        description="Reproduce 'INTERPRETING EMERGENT PLANNING IN MODEL-FREE REINFORCEMENT LEARNING'."
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to the configuration YAML file. Default: config.yaml'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        default='all',
        help='Specify a single experiment to run (e.g., rl_training_drc3_3) or "all" to run all enabled experiments.'
             'Available experiments are defined in config.yaml under `experiments_to_run`.'
    )
    args = parser.parse_args()

    config: Optional[Config] = None
    logger: Optional[Logger] = None
    exit_code: int = 0

    try:
        # 1. Configuration Initialization
        if not os.path.exists(args.config):
            print(f"Error: Configuration file not found at '{args.config}'. Please provide a valid path.", file=sys.stderr)
            sys.exit(1)

        config = Config(args.config)

        # 2. Logger Initialization (needs config, so config must be loaded first)
        logger = Logger(config)
        logger.log_info(f"Loaded configuration from: {args.config}")

        # Override `experiments_to_run` if a specific experiment is requested via CLI
        if args.experiment != 'all':
            logger.log_info(f"CLI override: Running only experiment '{args.experiment}'.")
            all_experiments_config: Dict[str, bool] = config.get('experiments_to_run', {})
            
            # Create a modified experiments_to_run dict where all are False by default
            modified_experiments: Dict[str, bool] = {k: False for k in all_experiments_config.keys()}
            
            if args.experiment in modified_experiments:
                modified_experiments[args.experiment] = True
                config.set('experiments_to_run', modified_experiments)
                logger.log_info(f"Configuration `experiments_to_run` updated for single experiment: {args.experiment}.")
            else:
                logger.log_error(
                    f"Specified experiment '{args.experiment}' not found in `config.experiments_to_run`."
                    f"Available: {list(all_experiments_config.keys())}. Exiting."
                )
                sys.exit(1)
        else:
            logger.log_info("Running all experiments enabled in config.yaml.")
            
        # Set global random seeds for reproducibility (where not overridden by experiment-specific seeds)
        global_seed: int = config.get('seed', 42)
        random.seed(global_seed)
        np.random.seed(global_seed)
        torch.manual_seed(global_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(global_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.log_info(f"Global random seed set to: {global_seed}")

        # 3. ExperimentRunner Instantiation
        # The runner itself will handle environment and agent creation/caching.
        runner = ExperimentRunner(config=config, logger=logger)
        
        # 4. Experiment Execution Orchestration
        runner.run_all_experiments()

        logger.log_info("All requested experiments completed successfully.")

    except Exception as e:
        if logger:
            logger.log_error(f"An unexpected error occurred during execution: {e}", exc_info=True)
        else:
            print(f"An unexpected error occurred before logger was fully initialized: {e}", file=sys.stderr)
        exit_code = 1
    finally:
        if logger:
            logger.close()
            # If an error occurred and the logger was active,
            # save the final config to reflect any modifications or state before crash.
            # This can be helpful for debugging.
            if exit_code != 0:
                error_config_path = os.path.join(logger.run_dir, "config_on_error.yaml")
                config.save(error_config_path)
                logger.log_info(f"Saved configuration at time of error to: {error_config_path}")
        sys.exit(exit_code)


if __name__ == '__main__':
    main()

