import os
import numpy as np
import torch
import random
import scipy.stats # For trim_mean (IQM)
from typing import Dict, Any, Union, Tuple, List

from config import Config
from environments import EnvironmentWrapper
from replay_buffer import PrioritizedReplayBuffer
from agent import MRQAgent
from networks import Models
from metrics import RewardNormalizer, Logger, ScoreNormalizer


class Trainer:
    """The central orchestrator of the training and evaluation process.

    Ties together the environment, replay buffer, agent, models, and logging
    components to execute the learning process as described in the paper.
    """

    def __init__(
        self,
        config: Config,
        env_wrapper: EnvironmentWrapper,
        replay_buffer: PrioritizedReplayBuffer,
        agent: MRQAgent,
        models: Models,
        reward_normalizer: RewardNormalizer,
        logger: Logger,
    ):
        """Initializes the Trainer.

        Args:
            config: Configuration object holding all hyperparameters.
            env_wrapper: An instance of EnvironmentWrapper for environment interaction.
            replay_buffer: An instance of PrioritizedReplayBuffer for experience storage and sampling.
            agent: An instance of MRQAgent containing the learning logic.
            models: An instance of Models containing all neural networks and optimizers.
            reward_normalizer: An instance of RewardNormalizer for value target scaling.
            logger: An instance of Logger for recording metrics.
        """
        self.config = config
        self.env_wrapper = env_wrapper
        self.replay_buffer = replay_buffer
        self.agent = agent
        self.models = models
        self.reward_normalizer = reward_normalizer
        self.logger = logger

        self.current_timestep: int = 0
        self.current_episode: int = 0

        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Determine benchmark for ScoreNormalizer based on environment name pattern
        benchmark_str: str
        env_name_lower = config.environment.env_name.lower()
        if "ale/" in env_name_lower:
            benchmark_str = "atari"
        # Heuristic for Gym Mujoco -v4 environments. May need refinement for other Gym envs.
        elif "-v4" in env_name_lower or "-v3" in env_name_lower or any(gym_env in env_name_lower for gym_env in ["ant", "halfcheetah", "hopper", "humanoid", "walker"]):
            benchmark_str = "gym"
        else: # Assume DMC for other cases (e.g. cartpole-swingup, cheetah-run)
            benchmark_str = "dmc"
        
        self.score_normalizer = ScoreNormalizer(
            benchmark=benchmark_str, task_name=config.environment.env_name
        )

    def run(self):
        """Executes the main training and evaluation loop."""
        print("Starting initial experience collection...")
        self._collect_initial_experience()
        print(f"Initial experience collection complete. Timesteps: {self.current_timestep}, Episodes: {self.current_episode}")

        # Reset environment once after initial collection to get the first observation for the main loop
        obs = self.env_wrapper.reset() 

        while self.current_timestep < self.config.training.total_timesteps:
            # Perform one training step (collect experience, learn)
            train_metrics, obs = self._train_step(obs)

            # Log training metrics
            if self.current_timestep % self.config.logging_evaluation.log_interval == 0:
                self.logger.log(train_metrics, self.current_timestep)
                self.logger.log({"train/episode": self.current_episode}, self.current_timestep)


            # Target Network and Reward Normalizer Update
            if self.current_timestep % self.config.training.target_update_frequency == 0:
                self.models.update_targets()
                self.reward_normalizer.update_target_mean()
                self.logger.log(
                    {"system/target_update_count": self.current_timestep // self.config.training.target_update_frequency}, 
                    self.current_timestep
                )

            # Evaluation
            if self.current_timestep % self.config.logging_evaluation.eval_interval == 0:
                print(f"Evaluating at timestep {self.current_timestep}...")
                eval_metrics = self._evaluate_policy()
                self.logger.log(eval_metrics, self.current_timestep)
                print(f"Evaluation finished at timestep {self.current_timestep}: "
                      f"Mean Reward = {eval_metrics['eval/mean_episode_reward']:.2f}, "
                      f"Normalized Mean Reward = {eval_metrics.get('eval/mean_normalized_reward', eval_metrics['eval/mean_episode_reward']):.2f}")

            # Checkpoint Saving
            if self.current_timestep % self.config.logging_evaluation.checkpoint_interval == 0:
                self._save_checkpoint(self.current_timestep)
                print(f"Checkpoint saved at timestep {self.current_timestep}")

        print(f"Training finished after {self.config.training.total_timesteps} timesteps.")
        self.logger.close()

    def _collect_initial_experience(self):
        """Populates the replay buffer with initial experiences using random actions."""
        obs = self.env_wrapper.reset()
        
        for _ in range(self.config.training.initial_random_steps):
            if self.current_timestep >= self.config.training.total_timesteps:
                break # Ensure initial steps don't exceed total_timesteps if initial_random_steps is large

            action = self.agent.act(obs, add_noise=True) # Agent.act handles exploration noise

            next_obs, reward, done, info = self.env_wrapper.step(action)
            
            self.replay_buffer.add(obs, action, reward, next_obs, done)
            self.reward_normalizer.update_mean(abs(reward))
            
            obs = next_obs
            self.current_timestep += 1
            if done:
                obs = self.env_wrapper.reset()
                self.current_episode += 1
                
        # Ensure that `agent._terminal_loss_active` gets set if any terminal transitions occurred
        # during initial collection. This is handled implicitly by `agent.compute_encoder_loss`
        # when it processes `dones_seq` from sampled batches.

    def _train_step(self, obs: np.ndarray) -> Tuple[Dict[str, Union[float, np.ndarray]], np.ndarray]:
        """Performs a single environment step and subsequent learning updates.

        Args:
            obs: The current observation from the environment before taking an action.

        Returns:
            A tuple containing:
            - A dictionary of aggregated training metrics (losses, TD errors).
            - The next observation after the step.
        """
        # 1. Collect experience
        action = self.agent.act(obs, add_noise=True)
        next_obs, reward, done, info = self.env_wrapper.step(action)
        self.replay_buffer.add(obs, action, reward, next_obs, done)
        self.reward_normalizer.update_mean(abs(reward))
        
        # 2. Perform learning updates (replay_ratio times)
        aggregated_metrics: Dict[str, list] = {}
        for _ in range(self.config.training.replay_ratio):
            if self.replay_buffer.size() < self.config.training.minibatch_size:
                # Not enough samples in buffer yet, skip learning this iteration
                break

            # Calculate k_step for replay buffer sampling: max of encoder_horizon and value_horizon
            # As discussed in internal thoughts, replay_buffer.sample needs to provide enough steps.
            max_horizon = max(self.config.losses.encoder_horizon, self.config.losses.value_horizon)
            
            # Sample batch from replay buffer. Assume `replay_buffer.sample` provides `next_obs_seq`.
            batch = self.replay_buffer.sample(self.config.training.minibatch_size, max_horizon)
            
            # Agent learns from the batch
            learning_output = self.agent.learn(batch)
            
            # Accumulate metrics
            for key, value in learning_output.items():
                if key not in aggregated_metrics:
                    aggregated_metrics[key] = []
                aggregated_metrics[key].append(value)
            
            # Update PER priorities using TD errors
            self.replay_buffer.update_priorities(
                batch["initial_indices"], learning_output["td_errors"]
            )

        # Average/process accumulated metrics for logging
        mean_metrics: Dict[str, Union[float, np.ndarray]] = {}
        if aggregated_metrics: # Only process if learning actually happened
            for k, v_list in aggregated_metrics.items():
                # If values are numpy arrays (e.g., td_errors), concatenate first then mean/process
                if isinstance(v_list[0], np.ndarray):
                    mean_metrics[k] = np.concatenate(v_list)
                else: # Assume scalar floats
                    mean_metrics[k] = np.mean(v_list)
        
        # Add a prefix to training metrics for logging
        train_metrics_logged = {f"train/{k}": v for k, v in mean_metrics.items()}


        # 3. Handle episode termination
        if done:
            next_obs = self.env_wrapper.reset() # Environment automatically resets on done
            self.current_episode += 1
        
        self.current_timestep += 1

        return train_metrics_logged, next_obs

    def _evaluate_policy(self) -> Dict[str, float]:
        """Evaluates the current policy's performance over a number of episodes.

        Returns:
            A dictionary containing evaluation metrics (mean, std, IQM of rewards and normalized rewards).
        """
        episode_rewards: List[float] = []
        
        for _ in range(self.config.logging_evaluation.num_eval_episodes):
            obs = self.env_wrapper.reset()
            episode_reward = 0.0
            done = False
            
            while not done:
                # Act without exploration noise during evaluation
                action = self.agent.act(obs, add_noise=False) 
                next_obs, reward, done, info = self.env_wrapper.step(action)
                episode_reward += reward
                obs = next_obs
            
            episode_rewards.append(episode_reward)

        # Calculate statistics for raw rewards
        rewards_np = np.array(episode_rewards)
        mean_reward = float(np.mean(rewards_np))
        std_reward = float(np.std(rewards_np))
        iqm_reward = float(scipy.stats.trim_mean(rewards_np, 0.25))

        eval_metrics: Dict[str, float] = {
            "eval/mean_episode_reward": mean_reward,
            "eval/std_episode_reward": std_reward,
            "eval/iqm_episode_reward": iqm_reward,
        }

        # Calculate normalized scores if applicable
        if self.score_normalizer.benchmark != "dmc": # DMC uses raw scores as per paper
            normalized_scores = [self.score_normalizer.normalize_score(r) for r in episode_rewards]
            normalized_scores_np = np.array(normalized_scores)
            
            mean_normalized_reward = float(np.mean(normalized_scores_np))
            std_normalized_reward = float(np.std(normalized_scores_np))
            iqm_normalized_reward = float(scipy.stats.trim_mean(normalized_scores_np, 0.25))

            eval_metrics.update({
                "eval/mean_normalized_reward": mean_normalized_reward,
                "eval/std_normalized_reward": std_normalized_reward,
                "eval/iqm_normalized_reward": iqm_normalized_reward,
            })
        
        return eval_metrics

    def _save_checkpoint(self, step: int):
        """Saves the current state of the models, optimizers, and trainer state to a checkpoint file.

        Args:
            step: The current global timestep, used for naming the checkpoint.
        """
        checkpoint_dir = os.path.join(self.logger.writer.log_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step:07d}.pt")

        checkpoint_state = {
            "current_timestep": self.current_timestep,
            "current_episode": self.current_episode,
            "reward_normalizer_state": {
                "_current_reward_sum_abs": self.reward_normalizer._current_reward_sum_abs,
                "_current_reward_count": self.reward_normalizer._current_reward_count,
                "_target_reward_mean_abs": self.reward_normalizer._target_reward_mean_abs,
            },
            # Model states
            "state_encoder_state_dict": self.models.state_encoder.state_dict(),
            "target_state_encoder_state_dict": self.models.target_state_encoder.state_dict(),
            "state_action_encoder_state_dict": self.models.state_action_encoder.state_dict(),
            "value_net1_state_dict": self.models.value_net1.state_dict(),
            "value_net2_state_dict": self.models.value_net2.state_dict(),
            "target_value_net1_state_dict": self.models.target_value_net1.state_dict(),
            "target_value_net2_state_dict": self.models.target_value_net2.state_dict(),
            "policy_net_state_dict": self.models.policy_net.state_dict(),
            "target_policy_net_state_dict": self.models.target_policy_net.state_dict(),
            # Optimizer states
            "encoder_optimizer_state_dict": self.models.encoder_optimizer.state_dict(),
            "value_optimizer_state_dict": self.models.value_optimizer.state_dict(),
            "policy_optimizer_state_dict": self.models.policy_optimizer.state_dict(),
            # Agent's internal state (e.g., terminal loss flag)
            "agent_terminal_loss_active": self.agent._terminal_loss_active,
        }
        torch.save(checkpoint_state, checkpoint_path)

    def _load_checkpoint(self, path: str):
        """Loads the training state from a checkpoint file.

        Args:
            path: The file path to the checkpoint.
        """
        checkpoint_state = torch.load(path, map_location=self.device)

        # Restore trainer state
        self.current_timestep = checkpoint_state["current_timestep"]
        self.current_episode = checkpoint_state["current_episode"]
        self.reward_normalizer._current_reward_sum_abs = checkpoint_state["reward_normalizer_state"]["_current_reward_sum_abs"]
        self.reward_normalizer._current_reward_count = checkpoint_state["reward_normalizer_state"]["_current_reward_count"]
        self.reward_normalizer._target_reward_mean_abs = checkpoint_state["reward_normalizer_state"]["_target_reward_mean_abs"]
        self.agent._terminal_loss_active = checkpoint_state["agent_terminal_loss_active"]

        # Restore model states
        self.models.state_encoder.load_state_dict(checkpoint_state["state_encoder_state_dict"])
        self.models.target_state_encoder.load_state_dict(checkpoint_state["target_state_encoder_state_dict"])
        self.models.state_action_encoder.load_state_dict(checkpoint_state["state_action_encoder_state_dict"])
        self.models.value_net1.load_state_dict(checkpoint_state["value_net1_state_dict"])
        self.models.value_net2.load_state_dict(checkpoint_state["value_net2_state_dict"])
        self.models.target_value_net1.load_state_dict(checkpoint_state["target_value_net1_state_dict"])
        self.models.target_value_net2.load_state_dict(checkpoint_state["target_value_net2_state_dict"])
        self.models.policy_net.load_state_dict(checkpoint_state["policy_net_state_dict"])
        self.models.target_policy_net.load_state_dict(checkpoint_state["target_policy_net_state_dict"])

        # Restore optimizer states
        # It's important to ensure optimizers are created before loading state_dict
        self.models.encoder_optimizer.load_state_dict(checkpoint_state["encoder_optimizer_state_dict"])
        self.models.value_optimizer.load_state_dict(checkpoint_state["value_optimizer_state_dict"])
        self.models.policy_optimizer.load_state_dict(checkpoint_state["policy_optimizer_state_dict"])

        print(f"Checkpoint loaded successfully from {path}. Resuming from timestep {self.current_timestep}.")

