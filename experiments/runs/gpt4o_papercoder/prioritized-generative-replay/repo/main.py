## main.py

import argparse
from utils import Utils
from data_loader import DataLoader
from replay_buffer import ReplayBuffer
from rl_agent import RLAgent
from curiosity_module import CuriosityModule
from generative_model import GenerativeModel
from synthetic_replay import SyntheticReplayManager
from trainer import Trainer
from evaluation import Evaluation

def main(config_path: str) -> None:
    """
    Main execution entry point for the Prioritized Generative Replay project.

    Args:
        config_path (str): Path to the configuration file (YAML format).
    """
    # Step 1: Parse configuration
    config = Utils.parse_config(config_path)
    logger = Utils.setup_logging(log_dir=config["output"]["logs_dir"], log_file="main.log")
    logger.info("Configuration loaded successfully.")

    # Step 2: Set up the environment
    logger.info("Setting up the environment...")
    data_loader = DataLoader(env_name=config["environment"]["name"], config=config)
    env, env_config = data_loader.setup_environment()
    logger.info(f"Environment '{config['environment']['task']}' initialized successfully.")

    # Step 3: Initialize replay buffers
    logger.info("Initializing replay buffers...")
    real_buffer = ReplayBuffer(size=config["replay_buffer"]["max_size"])
    synthetic_buffer = ReplayBuffer(size=config["replay_buffer"]["max_size"])

    # Step 4: Initialize RL Agent
    logger.info("Initializing reinforcement learning agent...")
    agent = RLAgent(config=config)
    logger.info(f"RL Agent initialized with algorithm: {config['training']['rl_algorithm']}.")

    # Step 5: Initialize generative model
    logger.info("Initializing generative model...")
    generative_model = GenerativeModel(config=config)

    # Step 6: Initialize curiosity module
    logger.info("Initializing curiosity-based relevance module...")
    curiosity_module = CuriosityModule(config=config)

    # Step 7: Initialize synthetic replay manager
    logger.info("Initializing synthetic replay manager...")
    synthetic_replay_manager = SyntheticReplayManager(
        generative_model=generative_model, 
        relevance_module=curiosity_module, 
        config=config
    )

    # Step 8: Set up the trainer
    logger.info("Setting up the trainer...")
    trainer = Trainer(
        agent=agent,
        generative_model=generative_model,
        replay_buffers={"real_buffer": real_buffer, "synthetic_buffer": synthetic_buffer},
        config=config
    )

    # Step 9: Set up evaluation module
    logger.info("Initializing evaluation module...")
    evaluator = Evaluation(env_name=config["environment"]["task"], config=config)

    # Step 10: Start training and evaluation
    logger.info("Starting training process...")
    try:
        trainer.train_agent(num_iterations=config["training"]["epochs"])
        logger.info("Training completed successfully.")

        logger.info("Starting evaluation...")
        metrics = evaluator.evaluate(agent)
        evaluator.visualize_results(metrics)
        logger.info("Evaluation completed. Metrics saved successfully.")
    except Exception as e:
        Utils.handle_error(module="main", exception=e, logger=logger)
        raise

    # Step 11: Save final model and environment state
    logger.info("Saving final model and replay buffers...")
    Utils.save_checkpoint(agent.actor, path=f"{config['output']['checkpoints_dir']}/final_agent.pth")
    Utils.save_checkpoint(generative_model.noise_predictor, path=f"{config['output']['checkpoints_dir']}/final_gen_model.pth")
    logger.info("All components saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prioritized Generative Replay")
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to the configuration file (default: config.yaml)"
    )
    args = parser.parse_args()

    # Execute main program
    main(config_path=args.config)
