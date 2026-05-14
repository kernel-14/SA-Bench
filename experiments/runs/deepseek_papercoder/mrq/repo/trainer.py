# trainer.py

"""
Trainer class for MR.Q.

The trainer manages the environment interaction, data collection,
periodic training updates, evaluation, and logging. It uses the
agent and replay buffer implemented in the project and follows the
configuration defined in config.yaml.
"""

import time
import numpy as np
import torch
from typing import Dict, List, Optional
from collections import deque

from config import Config, normalize_score
from env_utils import make_env
from replay_buffer import ReplayBuffer, SequenceBatch
from agent import MRQAgent


class Trainer:
    """
    Orchestrator for training the MR.Q agent on a single task.

    Parameters
    ----------
    cfg : Config
        Full configuration object, including benchmark, task, seed, etc.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # Random seeds for reproducibility
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        # Create training environment
        self.env = make_env(cfg.benchmark, cfg.task, cfg.seed)
        # Determine observation and action spaces
        obs_space = self.env.observation_space
        act_space = self.env.action_space

        # Observation shape (without batch)
        obs_shape = obs_space.shape
        # action dimension
        if hasattr(act_space, "n"):
            # Discrete
            action_dim = act_space.n
            self.discrete_actions = True
        else:
            # Box
            action_dim = act_space.shape[0]
            self.discrete_actions = False

        # Build agent
        self.agent = MRQAgent(
            cfg,
            obs_shape=obs_shape,
            action_dim=action_dim,
            discrete_actions=self.discrete_actions,
            device=cfg.device,
        )

        # Build replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=cfg.replay_buffer_capacity,
            state_shape=obs_shape,
            action_shape=(action_dim,),
            alpha=cfg.lap_alpha,
            min_priority=cfg.lap_min_priority,
            state_dtype=np.float32,
            action_dtype=np.float32,  # one-hot for discrete, raw vector for continuous
        )

        # Sequence length for sampling (encoder horizon + 1)
        self.seq_len = cfg.encoder_horizon + 1

        # Training state
        self.total_steps = 0
        self.episode_return = 0.0
        self.episode_length = 0
        self.recent_returns = deque(maxlen=100)
        self.eval_results: List[Dict[str, float]] = []  # evaluation log

        # Benchmark-specific settings
        self.total_timesteps = cfg.total_timesteps
        self.eval_frequency = cfg.eval_frequency
        self.warmup_steps = cfg.initial_random_steps

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> List[Dict[str, float]]:
        """
        Execute the full training schedule.

        Returns
        -------
        eval_results : list of dict
            Each element contains { "step": int, "mean_return": float, "normalized": float }.
        """
        state, _ = self.env.reset()
        state = np.asarray(state, dtype=np.float32)

        # Progress bar (optional)
        try:
            from tqdm import tqdm
            progress = tqdm(range(self.total_timesteps), desc=f"Training {self.cfg.task}")
        except ImportError:
            progress = range(self.total_timesteps)

        start_time = time.time()
        metrics = {}

        for step in progress:
            # Action selection
            if step < self.warmup_steps:
                action = self.env.action_space.sample()
                if self.discrete_actions:
                    action_idx = action  # int index
                    action_vec = self._discrete_to_one_hot(action_idx)
                else:
                    action_vec = action.astype(np.float32)
                    action_idx = action_vec
            else:
                action_agent = self.agent.select_action(state, explore=True, step=step)
                if self.discrete_actions:
                    action_idx = action_agent  # int
                    action_vec = self._discrete_to_one_hot(action_idx)
                else:
                    action_vec = action_agent  # np.array
                    action_idx = action_vec

            # Environment step
            next_state, reward, done, info = self.env.step(action_idx)
            next_state = np.asarray(next_state, dtype=np.float32)
            reward = float(reward)

            # Store transition (action stored as one-hot / vector)
            self.replay_buffer.add(
                state=state,
                action=action_vec,
                reward=reward,
                next_state=next_state,
                done=done,
            )
            # Inform agent about absolute reward for scaling
            self.agent.observe_reward(reward)

            # Episode tracking
            self.episode_return += reward
            self.episode_length += 1
            if done:
                self.recent_returns.append(self.episode_return)
                self.episode_return = 0.0
                self.episode_length = 0
                state, _ = self.env.reset()
                state = np.asarray(state, dtype=np.float32)
            else:
                state = next_state

            # Training update
            if step >= self.warmup_steps:
                batch = self.replay_buffer.sample(
                    batch_size=self.cfg.batch_size, seq_len=self.seq_len
                )
                metrics = self.agent.update(batch, step)
                self.agent.update_lap_priorities(batch, self.replay_buffer)

            # Periodic logging
            if (step + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                avg_return = np.mean(self.recent_returns) if self.recent_returns else 0.0
                log_str = (
                    f"Step {step+1}/{self.total_timesteps} | "
                    f"AvgReturn(last100)={avg_return:.2f} | "
                    f"Episodes: {len(self.recent_returns)} | "
                    f"Time: {elapsed:.1f}s"
                )
                if metrics:
                    log_str += (
                        f" | L_enc={metrics.get('loss_encoder',0):.4f} "
                        f"V={metrics.get('loss_value',0):.4f} "
                        f"P={metrics.get('loss_policy',0):.4f}"
                    )
                if hasattr(progress, "set_description"):
                    progress.set_description(log_str)

            # Evaluation
            if (step + 1) % self.eval_frequency == 0:
                eval_score = self.evaluate()
                norm_score = self._normalize_score(eval_score)
                self.eval_results.append({
                    "step": step + 1,
                    "mean_return": eval_score,
                    "normalized": norm_score,
                })
                log_str = f"EVAL step={step+1} mean_raw={eval_score:.2f} norm={norm_score:.2f}"
                if hasattr(progress, "write"):
                    progress.write(log_str)
                else:
                    print(log_str)

        # Final evaluation
        final_score = self.evaluate()
        final_norm = self._normalize_score(final_score)
        print(f"Final evaluation at step {self.total_timesteps}: raw={final_score:.2f}, norm={final_norm:.2f}")

        self.env.close()
        return self.eval_results

    # ------------------------------------------------------------------
    # Evaluation method
    # ------------------------------------------------------------------

    def evaluate(self, num_episodes: int = None) -> float:
        """
        Run evaluation episodes using a separate environment instance
        without exploration noise.

        Parameters
        ----------
        num_episodes : int, optional
            Number of episodes to run (default from config).

        Returns
        -------
        mean_return : float
            Average undiscounted return over the episodes.
        """
        if num_episodes is None:
            num_episodes = self.cfg.num_eval_episodes
        eval_env = make_env(self.cfg.benchmark, self.cfg.task, self.cfg.seed + 10000)
        returns = []
        for _ in range(num_episodes):
            state, _ = eval_env.reset()
            state = np.asarray(state, dtype=np.float32)
            done = False
            episode_return = 0.0
            while not done:
                action_agent = self.agent.select_action(state, explore=False, step=0)
                if self.discrete_actions:
                    env_action = action_agent  # int
                else:
                    env_action = action_agent
                state, reward, done, _ = eval_env.step(env_action)
                state = np.asarray(state, dtype=np.float32)
                episode_return += reward
            returns.append(episode_return)
        eval_env.close()
        return float(np.mean(returns))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _discrete_to_one_hot(self, action_idx: int) -> np.ndarray:
        """Convert a discrete action index to a one-hot vector."""
        action_dim = self.env.action_space.n
        one_hot = np.zeros(action_dim, dtype=np.float32)
        one_hot[action_idx] = 1.0
        return one_hot

    def _normalize_score(self, raw_score: float) -> float:
        """Apply benchmark-specific normalization to a raw score."""
        return normalize_score(raw_score, self.cfg)


# For quick testing
if __name__ == "__main__":
    from config import create_config
    # Example: Gym locomotion, Ant-v4
    cfg = create_config(
        benchmark="gym_locomotion", task="Ant-v4", seed=0, device="cpu"
    )
    trainer = Trainer(cfg)
    trainer.train()
