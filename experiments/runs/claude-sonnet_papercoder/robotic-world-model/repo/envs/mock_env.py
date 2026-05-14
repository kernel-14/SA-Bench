## envs/mock_env.py
"""Mock environment for testing the RWM pipeline without Isaac Lab.

This module provides ``MockEnv``, a lightweight drop-in replacement for
``ANYmalEnv`` and ``UnitreeG1Env`` that allows the entire ML pipeline to be
tested without Isaac Lab installed. It inherits from ``BaseEnv`` and satisfies
the same interface contract, producing tensors of the correct shapes and dtypes
so that ``RWMTrainer``, ``MBPOPPOTrainer``, ``TrajectoryDataset``, and
``Benchmark`` can all run end-to-end without a simulator.

The mock dynamics are intentionally simple:
    next_obs = obs + dt * action_padded + noise

where ``action_padded`` pads the action to ``obs_dim`` with zeros, and
``noise ~ N(0, 0.01)``. This is sufficient to exercise all code paths
without requiring a physics simulator.

Usage:
    # In main.py when env_backend == "mock"
    env = MockEnv(config, num_envs=1)
    obs, command = env.reset()
    next_obs, priv, reward, done, info = env.step(action)
    env.close()
"""

from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

from envs.base_env import BaseEnv


class MockEnv(BaseEnv):
    """Lightweight mock environment for unit testing without Isaac Lab.

    Implements the full ``BaseEnv`` interface using simple random dynamics.
    All returned tensors have the correct shapes and dtypes for the configured
    robot type (ANYmal D or Unitree G1), making this suitable for end-to-end
    pipeline testing.

    The mock environment uses a simple linear integrator dynamics model:
        next_obs = obs + dt * action_padded + noise

    where ``action_padded`` zero-pads the action vector to ``obs_dim``, and
    ``noise ~ N(0, 0.01^2)``. This exercises all downstream code paths
    (dataset construction, world model training, MBPO-PPO) without requiring
    a physics simulator.

    Episode termination occurs either:
      - Randomly with 5% probability per step (to test reset logic), or
      - After ``max_episode_steps`` steps (timeout).

    Attributes:
        max_episode_steps: Maximum number of steps per episode before timeout
            termination. Default: 1000.
        noise_std: Standard deviation of Gaussian observation noise added at
            each step. Default: 0.01.
        termination_prob: Per-step probability of random episode termination.
            Default: 0.05 (5%).
        _obs: Current world model observation buffer of shape
            ``[num_envs, obs_dim]``.
        _priv: Current privileged information buffer of shape
            ``[num_envs, priv_dim]``.
        _step_count: Per-environment step counter of shape ``[num_envs]``.
            Used for timeout termination detection.
    """

    def __init__(
        self,
        config: Any,
        num_envs: int = 1,
        max_episode_steps: int = 1000,
        noise_std: float = 0.01,
        termination_prob: float = 0.05,
    ) -> None:
        """Initialize the mock environment from the hydra config.

        Calls ``super().__init__`` to resolve all robot-specific dimensions
        and reward parameters from the config, then initializes internal
        state tensors on the target device.

        Args:
            config: Hydra ``DictConfig`` or plain dict containing the full
                experiment configuration from ``config.yaml``. Must contain:
                - ``config.robot``: "anymal_d" or "unitree_g1"
                - ``config.anymal_d`` or ``config.unitree_g1``: robot sub-config
                  with ``obs_dim``, ``action_dim``, ``priv_dim``,
                  ``policy_obs_dim``, ``reward_weights``, ``obs_slices``,
                  ``priv_slices``.
                - ``config.device``: device string ("cuda" or "cpu")
                - ``config.simulation.dt``: time step (0.02 from config.yaml)
                - ``config.collision_handling.terminate_on_base_contact``:
                  whether to terminate on base contact
            num_envs: Number of parallel mock environments. Default: 1.
            max_episode_steps: Maximum steps per episode before timeout
                termination. Default: 1000.
            noise_std: Standard deviation of Gaussian noise added to
                observations at each step. Default: 0.01.
            termination_prob: Per-step probability of random episode
                termination (independent of base contact). Default: 0.05.

        Raises:
            ValueError: If ``config.robot`` is not "anymal_d" or "unitree_g1"
                (raised by ``BaseEnv.__init__``).
        """
        # Delegate all config parsing and dimension resolution to BaseEnv.
        # This sets: self.obs_dim, self.action_dim, self.priv_dim,
        # self.policy_obs_dim, self.num_envs, self.device, self.dt,
        # self.robot_type, self.reward_weights, self.obs_slices,
        # self.policy_obs_slices, self.priv_slices, self.prev_actions,
        # self.default_joint_pos, self.commands, self.terminate_on_base_contact
        super().__init__(config, num_envs)

        # Mock-specific hyperparameters
        self.max_episode_steps: int = int(max_episode_steps)
        self.noise_std: float = float(noise_std)
        self.termination_prob: float = float(termination_prob)

        # ----------------------------------------------------------------
        # Internal state buffers — all on self.device
        # ----------------------------------------------------------------

        # Current world model observation: small random values near zero
        # to simulate a near-default robot state at initialization.
        self._obs: Tensor = torch.randn(
            self.num_envs,
            self.obs_dim,
            dtype=torch.float32,
            device=self.device,
        ) * 0.1

        # Current privileged information: zeros at initialization.
        # Updated each step with random binary/continuous values.
        self._priv: Tensor = torch.zeros(
            self.num_envs,
            self.priv_dim,
            dtype=torch.float32,
            device=self.device,
        )

        # Per-environment step counter for timeout termination detection.
        self._step_count: Tensor = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )

    # ----------------------------------------------------------------
    # Abstract method implementations
    # ----------------------------------------------------------------

    def reset(self) -> Tuple[Tensor, Tensor]:
        """Reset all environments and return initial observations and commands.

        Reinitializes all internal state tensors with small random values,
        samples new velocity commands, and resets all step counters and
        action buffers.

        Returns:
            A tuple ``(obs, command)`` where:
              - ``obs``: Initial world model observation of shape
                ``[num_envs, obs_dim]``. Small random values (std=0.1)
                simulating a near-default robot state.
              - ``command``: Sampled velocity commands of shape
                ``[num_envs, 3]``. Each row is ``[vx, vy, yaw_rate]``
                sampled uniformly from reasonable locomotion ranges.
        """
        # Reinitialize observation with small random values near zero.
        # std=0.1 keeps values in a reasonable range for all obs dimensions
        # (velocities, gravity, joint positions, velocities, torques).
        self._obs = torch.randn(
            self.num_envs,
            self.obs_dim,
            dtype=torch.float32,
            device=self.device,
        ) * 0.1

        # Reset privileged info to zeros (no contacts at episode start).
        self._priv = torch.zeros(
            self.num_envs,
            self.priv_dim,
            dtype=torch.float32,
            device=self.device,
        )

        # Sample new velocity commands using the BaseEnv helper.
        # Ranges: vx in [-1.0, 1.0], vy in [-0.5, 0.5], yaw in [-1.0, 1.0]
        self.commands = self._sample_commands(self.num_envs)

        # Reset all stateful buffers to zero.
        self.prev_actions.zero_()
        self._step_count.zero_()

        return self._obs.clone(), self.commands.clone()

    def step(
        self,
        action: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]:
        """Apply actions and advance the mock environment by one step.

        Applies a simple linear integrator dynamics model:
            next_obs = obs + dt * action_padded + noise

        where ``action_padded`` zero-pads the action to ``obs_dim``, and
        ``noise ~ N(0, noise_std^2)``. This is sufficient to exercise all
        downstream code paths without a physics simulator.

        Privileged information is updated with random values:
          - For ANYmal D (priv_dim=8): all binary contact flags (Bernoulli 0.5)
          - For Unitree G1 (priv_dim=30): first 26 binary contacts, last 4
            continuous uniform in [0, 1] for foot heights and velocities

        Episode termination occurs when:
          1. Random termination: Bernoulli(termination_prob) per environment
          2. Timeout: step_count >= max_episode_steps

        Args:
            action: Joint position targets of shape ``[num_envs, action_dim]``.
                Will be moved to ``self.device`` if on a different device.

        Returns:
            A tuple ``(next_obs, priv, reward, done, info)`` where:
              - ``next_obs``: Next world model observation, shape
                ``[num_envs, obs_dim]``.
              - ``priv``: Privileged information, shape
                ``[num_envs, priv_dim]``.
              - ``reward``: Zero reward tensor, shape ``[num_envs]``.
                The mock env returns zeros — reward signal quality is not
                the purpose of this environment.
              - ``done``: Float termination flags (0.0 or 1.0), shape
                ``[num_envs]``. 1.0 for terminated environments.
              - ``info``: Dict with "episode_length" and "timeout" keys.
        """
        # Ensure action is on the correct device.
        action = action.to(self.device)

        # ----------------------------------------------------------------
        # Mock dynamics: next_obs = obs + dt * action_padded + noise
        # ----------------------------------------------------------------

        # Zero-pad action from [num_envs, action_dim] to [num_envs, obs_dim].
        # action_dim < obs_dim for both robots (12 < 45 for ANYmal D,
        # 29 < 96 for G1), so we cannot slice action[:, :obs_dim] directly.
        action_padded: Tensor = torch.zeros(
            self.num_envs,
            self.obs_dim,
            dtype=torch.float32,
            device=self.device,
        )
        # Fill the first action_dim dimensions with the actual action values.
        action_padded[:, : self.action_dim] = action

        # Gaussian observation noise for testing noise robustness code paths.
        noise: Tensor = torch.randn(
            self.num_envs,
            self.obs_dim,
            dtype=torch.float32,
            device=self.device,
        ) * self.noise_std

        # Apply integrator dynamics.
        next_obs: Tensor = self._obs + self.dt * action_padded + noise

        # ----------------------------------------------------------------
        # Update privileged information with random values
        # ----------------------------------------------------------------
        self._priv = self._generate_privileged_info()

        # ----------------------------------------------------------------
        # Compute zero reward (mock env does not provide meaningful reward)
        # ----------------------------------------------------------------
        reward: Tensor = torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )

        # ----------------------------------------------------------------
        # Increment step counters and detect termination
        # ----------------------------------------------------------------
        self._step_count += 1

        # Random termination: Bernoulli(termination_prob) per environment.
        # This exercises episode reset logic in MBPOPPOTrainer.
        random_done: Tensor = torch.bernoulli(
            torch.full(
                (self.num_envs,),
                self.termination_prob,
                dtype=torch.float32,
                device=self.device,
            )
        ).bool()

        # Timeout termination: episode length exceeded.
        timeout_done: Tensor = self._step_count >= self.max_episode_steps

        # Combined termination: either random or timeout.
        done_bool: Tensor = random_done | timeout_done  # shape [num_envs], bool

        # ----------------------------------------------------------------
        # Reset terminated environments
        # ----------------------------------------------------------------
        if done_bool.any():
            reset_indices = done_bool.nonzero(as_tuple=True)[0]
            n_reset: int = len(reset_indices)

            # Reset observations for terminated environments.
            next_obs[reset_indices] = torch.randn(
                n_reset,
                self.obs_dim,
                dtype=torch.float32,
                device=self.device,
            ) * 0.1

            # Reset step counters for terminated environments.
            self._step_count[reset_indices] = 0

            # Zero previous actions for terminated environments.
            # This prevents stale action history from corrupting the
            # action rate penalty at the start of the next episode.
            self.prev_actions[reset_indices] = 0.0

        # ----------------------------------------------------------------
        # Update internal state
        # ----------------------------------------------------------------
        self._obs = next_obs.clone()

        # Update prev_actions AFTER reward computation (reward uses old
        # prev_actions) and AFTER resetting done envs (which zero prev_actions
        # for done envs). The _update_prev_actions helper handles this ordering.
        self._update_prev_actions(action, done=done_bool)

        # ----------------------------------------------------------------
        # Build info dict
        # ----------------------------------------------------------------
        info: Dict[str, Any] = {
            "episode_length": self._step_count.float().clone(),
            "timeout": timeout_done.clone(),
        }

        # Return done as float tensor (0.0 or 1.0) for compatibility with
        # MBPOPPOTrainer._compute_gae which uses done as a float mask.
        done_float: Tensor = done_bool.float()

        return next_obs, self._priv.clone(), reward, done_float, info

    def get_privileged_info(self) -> Tensor:
        """Return the current privileged information for all environments.

        Returns the most recently computed privileged information buffer.
        This is consistent with the real environment behavior where
        ``get_privileged_info()`` is called after ``step()`` to retrieve
        the contact and foot state information for the current timestep.

        Returns:
            Privileged information tensor of shape ``[num_envs, priv_dim]``.
            For ANYmal D: 8-dim (4 knee contacts + 4 foot contacts, Table S3).
            For Unitree G1: 30-dim (26 body contacts + 2 foot heights +
            2 foot velocities, Table S3).
        """
        return self._priv.clone()

    def compute_reward(
        self,
        obs: Tensor,
        action: Tensor,
        next_obs: Tensor,
        priv: Tensor,
        command: Tensor,
        t_fa: Optional[Tensor] = None,
    ) -> Tensor:
        """Return zero reward for all environments.

        The mock environment does not provide a meaningful reward signal.
        Returning zeros is intentional — the mock env is for testing pipeline
        correctness (tensor shapes, data flow, gradient computation), not
        reward signal quality.

        The ``MBPOPPOTrainer`` calls this method during imagination rollouts.
        Zero rewards will still exercise the GAE computation and PPO update
        code paths, which is the purpose of the mock env.

        Args:
            obs: Current world model observation, shape ``[B, obs_dim]``.
                Unused in mock env.
            action: Current action, shape ``[B, action_dim]``. Unused.
            next_obs: Next world model observation, shape ``[B, obs_dim]``.
                Unused in mock env.
            priv: Privileged information, shape ``[B, priv_dim]``. Unused.
            command: Velocity commands, shape ``[B, 3]``. Unused.
            t_fa: Feet air time tensor, shape ``[B]``. Unused. Default: None.

        Returns:
            Zero reward tensor of shape ``[B]`` on ``self.device``.
        """
        batch_size: int = obs.shape[0]
        return torch.zeros(
            batch_size,
            dtype=torch.float32,
            device=self.device,
        )

    def close(self) -> None:
        """Release all resources (no-op for mock environment).

        The mock environment holds no external resources (no simulator
        connections, no GPU contexts beyond standard PyTorch tensors).
        This method is a no-op but is provided for interface compliance.

        Idempotent: calling close() multiple times is safe.
        """
        if self._closed:
            return
        self._closed = True
        # No resources to release in the mock environment.

    # ----------------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------------

    def _generate_privileged_info(self) -> Tensor:
        """Generate random privileged information for the current step.

        Produces robot-type-appropriate random privileged information:

        **ANYmal D (priv_dim=8, Table S3):**
          - Dims 0:4 — knee contact flags: Bernoulli(0.5) binary values
          - Dims 4:8 — foot contact flags: Bernoulli(0.5) binary values

        **Unitree G1 (priv_dim=30, Table S3):**
          - Dims 0:26 — body contact flags: Bernoulli(0.5) binary values
          - Dims 26:28 — foot heights: Uniform(0.0, 0.3) continuous values
            (reasonable foot clearance range in meters)
          - Dims 28:30 — foot velocities: Uniform(-1.0, 1.0) continuous values
            (reasonable foot velocity range in m/s)

        Returns:
            Privileged information tensor of shape ``[num_envs, priv_dim]``
            on ``self.device``.
        """
        priv: Tensor = torch.zeros(
            self.num_envs,
            self.priv_dim,
            dtype=torch.float32,
            device=self.device,
        )

        if self.robot_type == "anymal_d":
            # All 8 dims are binary contact flags (Table S3: knee + foot contacts)
            priv = torch.bernoulli(
                torch.full(
                    (self.num_envs, self.priv_dim),
                    0.5,
                    dtype=torch.float32,
                    device=self.device,
                )
            )

        elif self.robot_type == "unitree_g1":
            # Dims 0:26 — binary body contact flags (Table S3)
            priv[:, 0:26] = torch.bernoulli(
                torch.full(
                    (self.num_envs, 26),
                    0.5,
                    dtype=torch.float32,
                    device=self.device,
                )
            )

            # Dims 26:28 — foot heights: continuous in [0.0, 0.3] meters
            # Represents swing foot clearance height above ground
            priv[:, 26:28] = torch.rand(
                self.num_envs,
                2,
                dtype=torch.float32,
                device=self.device,
            ) * 0.3  # scale to [0.0, 0.3] meters

            # Dims 28:30 — foot velocities: continuous in [-1.0, 1.0] m/s
            priv[:, 28:30] = (
                torch.rand(
                    self.num_envs,
                    2,
                    dtype=torch.float32,
                    device=self.device,
                ) * 2.0 - 1.0  # scale to [-1.0, 1.0] m/s
            )

        return priv
