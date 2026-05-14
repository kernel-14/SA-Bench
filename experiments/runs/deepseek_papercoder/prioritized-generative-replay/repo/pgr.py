## pgr.py
"""
Prioritized Generative Replay (PGR) Algorithm – main orchestrator.
Implements the outer/inner loop from Algorithm 1 of the paper.
"""

import os
import logging
import numpy as np
import torch

from config import Config
from environment import Environment
from replay_buffer import ReplayBuffer
from agent import PolicyAgent, REDQAgent, DRQv2Agent
from relevance import (
    RelevanceFunction,
    RewardRelevance,
    ReturnRelevance,
    TDErrorRelevance,
    CuriosityRelevance,
)
from diffusion import ConditionalDiffusion
from utils import set_seeds, get_device, setup_logging


class PGRAlgorithm:
    """
    Prioritized Generative Replay algorithm.

    Orchestrates online interaction, synthetic data generation via a conditional
    diffusion model, and policy training with mixed real/synthetic replays.

    Parameters
    ----------
    config : Config
        Complete experiment configuration (parsed from config.yaml).
    """

    def __init__(self, config: Config) -> None:
        self.config = config

        # Reproducibility
        set_seeds(config.environment.random_seed)

        # Device (CPU / GPU)
        self.device = get_device()

        # Logging
        self.logger = setup_logging(config)

        # ---- Environment ----
        self.env = Environment(
            env_name=config.environment.task_name,
            state_based=config.environment.state_based,
            seed=config.environment.random_seed,
        )

        # Determine observation and action dimensions
        self.state_shape = self.env.observation_space.shape
        self.action_dim = self.env.action_space.shape[0]
        self.is_pixel = not config.environment.state_based

        # ---- Agent ----
        if config.policy.algorithm.lower() == "redq":
            self.agent = REDQAgent(
                state_dim=np.prod(self.state_shape),  # for state-based
                action_dim=self.action_dim,
                config=config,
            )
        elif config.policy.algorithm.lower() == "drqv2":
            self.agent = DRQv2Agent(
                state_dim=self.state_shape,  # (C, H, W)
                action_dim=self.action_dim,
                config=config,
            )
        else:
            raise ValueError(f"Unknown policy algorithm: {config.policy.algorithm}")

        # ---- Replay Buffers ----
        # Determine the state dimension for buffer storage: for pixel tasks,
        # we store encoded latent vectors (dim = 256), not raw pixels.
        if self.is_pixel:
            # The stored state is the latent representation produced by the agent's encoder.
            self.stored_state_dim = self.agent.latent_dim if hasattr(self.agent, 'latent_dim') else 256
        else:
            self.stored_state_dim = np.prod(self.state_shape)

        self.real_buffer = ReplayBuffer(
            capacity=config.replay_buffer.real_capacity,
            state_shape=(self.stored_state_dim,),
            action_shape=(self.action_dim,),
            condition_dim=1,  # scalar relevance (not stored, but placeholder)
        )
        self.syn_buffer = ReplayBuffer(
            capacity=config.replay_buffer.synthetic_capacity,
            state_shape=(self.stored_state_dim,),
            action_shape=(self.action_dim,),
            condition_dim=1,
        )

        # ---- Relevance Function ----
        self.relevance_fn = self._build_relevance_fn(config)

        # ---- Conditional Diffusion Model ----
        # Pass visual encoder if applicable (for pixel tasks, the encoder is part of the agent).
        # The diffusion model will store it for potential use (not used in generation currently).
        self.diffusion = ConditionalDiffusion(
            state_dim=self.stored_state_dim,
            action_dim=self.action_dim,
            config=config,  # pass the whole config; ConditionalDiffusion will extract what it needs
            visual_encoder=self.agent.encoder if self.is_pixel and hasattr(self.agent, 'encoder') else None,
        )

        # ---- Algorithm state ----
        self.env_steps = 0
        self.inner_loop_counter = 0
        self.next_eval_step = config.pgr_algorithm.eval_interval
        self.best_eval_return = -np.inf

        # ---- Training state ----
        self._raw_state = None   # current raw observation for action selection
        self._state = None       # current encoded observation for buffer storage
        self._done = True        # flag to reset environment at start

        # Cache effective synthetic ratio (kept constant for scaling)
        self.effective_synthetic_ratio = (
            (config.replay_buffer.batch_size - config.replay_buffer.real_per_batch)
            / config.replay_buffer.batch_size
        )

        # Log basic info
        self.logger.info("PGR algorithm initialized.")
        self.logger.info(f"Environment: {config.environment.task_name}")
        self.logger.info(f"Total env steps: {config.environment.total_env_steps}")
        self.logger.info(f"Agent: {config.policy.algorithm}")
        self.logger.info(f"Relevance: {config.relevance.type}")
        self.logger.info(f"Inner loop freq: {config.pgr_algorithm.inner_loop_frequency}")
        self.logger.info(f"Synthetic ratio: {self.effective_synthetic_ratio:.3f}")

    # ------------------------------------------------------------------
    # Build relevance function from config
    # ------------------------------------------------------------------
    def _build_relevance_fn(self, config: Config) -> RelevanceFunction:
        """Instantiate the relevance function specified in the configuration."""
        fn_type = config.relevance.type.lower()
        if fn_type == "reward":
            return RewardRelevance()
        elif fn_type == "return":
            return ReturnRelevance(self.agent)
        elif fn_type == "td_error":
            return TDErrorRelevance(self.agent, gamma=config.policy.discount_gamma)
        elif fn_type == "curiosity":
            visual_encoder = None
            if self.is_pixel and hasattr(self.agent, 'encoder'):
                visual_encoder = self.agent.encoder  # frozen later in CuriosityRelevance
            return CuriosityRelevance(
                state_shape=self.state_shape if self.is_pixel else self.stored_state_dim,
                action_dim=self.action_dim,
                config=config,
                feature_extractor=visual_encoder,
                batch_size=config.replay_buffer.batch_size,
            )
        else:
            raise ValueError(f"Unknown relevance type: {fn_type}")

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Execute the PGR algorithm until the environment step budget is exhausted."""

        total_steps = self.config.environment.total_env_steps

        while self.env_steps < total_steps:
            # ---- 1. Reset episode if needed ----
            if self._done:
                raw_obs = self.env.reset()
                self._raw_state = raw_obs
                self._state = self._encode_obs(raw_obs)
                self._done = False

            # ---- 2. Select action ----
            action = self.agent.select_action(self._raw_state)

            # ---- 3. Environment step ----
            next_raw_obs, reward, done, _ = self.env.step(action)

            # ---- 4. Store transition in real buffer ----
            encoded_next_obs = self._encode_obs(next_raw_obs)
            self.real_buffer.push(
                state=self._state,
                action=action,
                reward=reward,
                next_state=encoded_next_obs,
                done=done,
                condition=0.0,  # dummy condition (not used)
            )

            # ---- 5. Update learnable relevance (curiosity) ----
            # Update once per env step, corresponding to ~5% of policy gradient steps.
            if isinstance(self.relevance_fn, CuriosityRelevance):
                self.relevance_fn.update(self.real_buffer)

            # ---- 6. Inner loop checkpoint ----
            self.inner_loop_counter += 1
            if self.inner_loop_counter >= self.config.pgr_algorithm.inner_loop_frequency:
                self._inner_loop_generation()
                self.inner_loop_counter = 0

            # ---- 7. Policy training ----
            # Use synthetic data only if the buffer has enough transitions.
            use_synthetic = len(self.syn_buffer) >= self.config.replay_buffer.batch_size * self.effective_synthetic_ratio
            for _ in range(self.config.policy.utd_ratio):
                if use_synthetic:
                    self.agent.train(
                        real_buffer=self.real_buffer,
                        syn_buffer=self.syn_buffer,
                        synthetic_ratio=self.effective_synthetic_ratio,
                    )
                else:
                    # Fallback to real-only training (synthetic_ratio = 0)
                    self.agent.train(
                        real_buffer=self.real_buffer,
                        syn_buffer=None,
                        synthetic_ratio=0.0,
                    )

            # ---- 8. Update tracking ----
            self._state = encoded_next_obs
            self._raw_state = next_raw_obs
            self._done = done
            self.env_steps += 1

            # ---- 9. Periodic evaluation ----
            if self.env_steps >= self.next_eval_step:
                eval_return = self._evaluate_policy()
                self.next_eval_step += self.config.pgr_algorithm.eval_interval
                self.logger.info(
                    f"Step {self.env_steps}/{total_steps} | Eval return: {eval_return:.2f}"
                )
                if eval_return > self.best_eval_return:
                    self.best_eval_return = eval_return
                    self._save_checkpoint("best")
                # Optional: wandb logging
                if self.config.logging.use_wandb:
                    import wandb
                    wandb.log({"eval_return": eval_return, "env_steps": self.env_steps})

        # Final evaluation
        final_return = self._evaluate_policy()
        self.logger.info(f"Training finished. Final eval return: {final_return:.2f}")
        self._save_checkpoint("final")

    # ------------------------------------------------------------------
    # Encode observation for buffer storage (latent for pixels, identity for state)
    # ------------------------------------------------------------------
    def _encode_obs(self, obs: np.ndarray) -> np.ndarray:
        """Convert a raw observation into the format stored in replay buffer."""
        if self.is_pixel:
            if not hasattr(self.agent, 'encode'):
                raise AttributeError("The policy agent for pixel tasks must provide an `encode` method.")
            latent = self.agent.encode(obs)
            return latent
        else:
            return obs.astype(np.float32)

    # ------------------------------------------------------------------
    # Inner loop: retrain diffusion model and regenerate synthetic buffer
    # ------------------------------------------------------------------
    def _inner_loop_generation(self) -> None:
        """Periodic retraining of the conditional diffusion model and
        replacement of the synthetic replay buffer with newly generated
        transitions."""
        self.logger.info(f"Inner loop at env step {self.env_steps}: retraining diffusion...")

        # 1. Train diffusion for a fixed number of gradient steps
        train_steps = self.config.diffusion.inner_train_steps
        for i in range(train_steps):
            loss = self.diffusion.train_step(
                real_buffer=self.real_buffer,
                relevance_fn=self.relevance_fn,
            )
            if i % 500 == 0 or i == train_steps - 1:
                self.logger.debug(f"Diffusion train step {i}/{train_steps}, loss={loss:.4f}")

        # 2. Generate synthetic transitions
        self.logger.info("Generating synthetic data...")
        num_samples = self.config.pgr_algorithm.synthetic_samples_per_loop
        states, actions, next_states, rewards = self.diffusion.generate(
            num_samples=num_samples,
            real_buffer=self.real_buffer,
            relevance_fn=self.relevance_fn,
            guidance_scale=self.config.diffusion.guidance_scale,
        )

        # 3. Replace the synthetic buffer with newly generated data
        self.syn_buffer = ReplayBuffer(
            capacity=self.config.replay_buffer.synthetic_capacity,
            state_shape=(self.stored_state_dim,),
            action_shape=(self.action_dim,),
        )
        for i in range(num_samples):
            self.syn_buffer.push(
                state=states[i],
                action=actions[i],
                reward=rewards[i],
                next_state=next_states[i],
                done=False,      # synthetic transitions are non‑terminal
                condition=0.0,   # dummy condition
            )

        self.logger.info(f"Synthetic buffer populated with {len(self.syn_buffer)} transitions.")

    # ------------------------------------------------------------------
    # Evaluation of current policy
    # ------------------------------------------------------------------
    def _evaluate_policy(self) -> float:
        """Run several evaluation episodes with deterministic actions.

        Returns the average undiscounted return.
        """
        eval_episodes = self.config.pgr_algorithm.eval_episodes
        total_return = 0.0

        # Set agent to evaluation mode (deterministic actions, no exploration noise)
        # We assume the agent has `train()` and `eval()` methods to switch modes.
        agent_was_training = self.agent.actor.training  # rough check
        self.agent.eval()

        for _ in range(eval_episodes):
            raw_obs = self.env.reset()
            episode_return = 0.0
            done = False
            while not done:
                # Use deterministic action (mean of the stochastic policy)
                action = self.agent.select_action(raw_obs, deterministic=True)
                next_raw_obs, reward, done, _ = self.env.step(action)
                episode_return += reward
                raw_obs = next_raw_obs
            total_return += episode_return

        # Restore original training mode
        if agent_was_training:
            self.agent.train()
        else:
            self.agent.eval()

        return total_return / eval_episodes

    # ------------------------------------------------------------------
    # Checkpointing (save / load)
    # ------------------------------------------------------------------
    def _save_checkpoint(self, tag: str = "latest") -> None:
        """Save agent and diffusion model to disk."""
        checkpoint_dir = self.config.logging.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        agent_path = os.path.join(checkpoint_dir, f"agent_{tag}.pt")
        diffusion_path = os.path.join(checkpoint_dir, f"diffusion_{tag}.pt")
        self.agent.save_checkpoint(agent_path)
        torch.save(self.diffusion.denoiser.state_dict(), diffusion_path)
        self.logger.info(f"Checkpoint saved ({tag}).")

    def load_checkpoint(self, tag: str = "latest") -> None:
        """Restore agent and diffusion model from disk."""
        checkpoint_dir = self.config.logging.checkpoint_dir
        agent_path = os.path.join(checkpoint_dir, f"agent_{tag}.pt")
        diffusion_path = os.path.join(checkpoint_dir, f"diffusion_{tag}.pt")
        if os.path.exists(agent_path):
            self.agent.load_checkpoint(agent_path)
        if os.path.exists(diffusion_path):
            self.diffusion.denoiser.load_state_dict(torch.load(diffusion_path))
        self.logger.info(f"Checkpoint loaded ({tag}).")

