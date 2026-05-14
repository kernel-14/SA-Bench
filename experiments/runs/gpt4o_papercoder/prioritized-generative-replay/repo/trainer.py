## trainer.py

import logging
from typing import Any, Dict

from rl_agent import RLAgent
from generative_model import GenerativeModel
from replay_buffer import ReplayBuffer
from synthetic_replay import SyntheticReplayManager
from curiosity_module import CuriosityModule
from utils import Utils
from evaluation import Evaluation


class Trainer:
    """
    Trainer class to coordinate the training of reinforcement learning agents, 
    periodic synthetic replay buffer generations, and evaluation.
    """

    def __init__(
        self,
        agent: RLAgent,
        generative_model: GenerativeModel,
        replay_buffers: Dict[str, ReplayBuffer],
        config: Dict[str, Any],
    ) -> None:
        """
        Initializes the Trainer object.

        Args:
            agent (RLAgent): Reinforcement learning agent.
            generative_model (GenerativeModel): Generative model for conditional synthetic data.
            replay_buffers (Dict[str, ReplayBuffer]): Real and synthetic replay buffers.
            config (Dict[str, Any]): Parsed configuration dictionary from `config.yaml`.
        """
        self.agent = agent
        self.generative_model = generative_model
        self.real_buffer = replay_buffers["real_buffer"]
        self.synthetic_buffer = replay_buffers["synthetic_buffer"]
        self.config = config

        # Relevance-based synthetic replay manager
        self.replay_manager = SyntheticReplayManager(
            generative_model=self.generative_model,
            relevance_module=CuriosityModule(config),
            config=config,
        )

        # Setup logging
        self.logger = Utils.setup_logging(
            log_dir=config["output"]["logs_dir"], log_file="training.log"
        )

        # Evaluation setup
        self.evaluator = Evaluation(
            env_name=config["environment"]["task"], config=config
        )

        # Parse hyperparameters
        self.max_epochs = config["training"]["epochs"]
        self.batch_size = config["training"]["batch_size"]
        self.synthetic_to_real_ratio = config["training"]["synthetic_to_real_ratio"]
        self.synthetic_regen_interval = config["generative_model"]["training_steps"]
        self.eval_frequency = config["evaluation"].get("eval_frequency", 10_000)

        # Seed setup
        Utils.set_seeds(config["evaluation"].get("num_seeds", 42))

    def train_agent(self, num_iterations: int) -> None:
        """
        Trains the RL agent over a number of iterations, periodically regenerates 
        synthetic replay buffers, and evaluates policy performance.

        Args:
            num_iterations (int): Total number of iterations to train the agent.
        """
        self.logger.info(f"Starting training for {num_iterations} iterations.")

        for iteration in range(1, num_iterations + 1):
            self.logger.info(f"Iteration {iteration} / {num_iterations}")

            # Step 1: Collect transitions from the environment
            self.logger.info("Collecting transitions from the environment...")
            self.agent.collect_transitions(env=self.evaluator.env, replay_buffer=self.real_buffer, num_steps=self.batch_size)

            # Step 2: Periodically regenerate synthetic replay buffer
            if iteration % self.synthetic_regen_interval == 0:
                self.logger.info("Regenerating synthetic replay buffer...")
                self.synthetic_buffer = self.replay_manager.regenerate_synthetic_buffer(self.real_buffer)
                synthetic_metrics = self.replay_manager.evaluate_synthetic_buffer()
                self.logger.info(f"Synthetic buffer stats: {synthetic_metrics}")

            # Step 3: Train the agent's policy
            self.logger.info("Updating policy using mixed replay buffers...")
            self.agent.update_policy(real_buffer=self.real_buffer, synthetic_buffer=self.synthetic_buffer)

            # Step 4: Periodic evaluation
            if iteration % self.eval_frequency == 0 or iteration == num_iterations:
                self.logger.info("Evaluating agent performance...")
                metrics = self.evaluator.evaluate(self.agent)
                self.logger.info(f"Evaluation metrics: {metrics}")
                self.evaluator.visualize_results(metrics)

                # Save model checkpoint
                agent_checkpoint_path = f"{self.config['output']['checkpoints_dir']}/agent_iter_{iteration}.pth"
                gen_model_checkpoint_path = f"{self.config['output']['checkpoints_dir']}/gen_model_iter_{iteration}.pth"
                Utils.save_checkpoint(self.agent.actor, path=agent_checkpoint_path)
                Utils.save_checkpoint(self.generative_model.noise_predictor, path=gen_model_checkpoint_path)
                self.logger.info(f"Checkpoints saved at iteration {iteration}.")

    def _log_training_progress(self, iteration: int, iterations: int, metrics: Dict[str, Any]) -> None:
        """
        Logs the training progress with metrics.

        Args:
            iteration (int): Current training iteration.
            iterations (int): Total training iterations.
            metrics (Dict[str, Any]): Training metrics tracked for logging.
        """
        log_message = (
            f"Iteration: {iteration}/{iterations} | "
            f"Average Return: {metrics.get('average_return', 'N/A')} | "
            f"Dormant Ratio: {metrics.get('dormant_ratio', 'N/A')}"
        )
        self.logger.info(log_message)
