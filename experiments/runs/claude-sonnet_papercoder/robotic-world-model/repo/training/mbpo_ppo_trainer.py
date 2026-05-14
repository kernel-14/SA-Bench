## Code: training/mbpo_ppo_trainer.py

```python
## training/mbpo_ppo_trainer.py
"""MBPO-PPO trainer implementing Algorithm 1 from the Robotic World Model paper.

This module provides ``MBPOPPOTrainer``, the central orchestrator of the
Model-Based Policy Optimization with PPO framework described in Section 3.3
and Algorithm 1 of the paper.

The training loop combines:
  1. Real environment data collection (Algorithm 1, step 3)
  2. World model fine-tuning on replay buffer data (Algorithm 1, step 4)
  3. Imagination rollouts using the world model (Algorithm 1, steps 5-6)
  4. PPO policy updates on imagined trajectories (Algorithm 1, step 7)

Key design decisions:
  - Policy learning happens entirely in imagination (4096 parallel envs, 100 steps)
  - World model is fine-tuned online using a single real environment
  - PPO uses zeroth-order gradients (no backprop through world model dynamics)
  - GAE-λ with γ=0.99, λ=0.95 for advantage estimation
  - KL early stopping at 1.5× target KL to prevent policy collapse

Training parameters (Table S11):
  - Imagination environments: 4096
  - Imagination steps per iteration: 100
  - Buffer size: 1000 trajectories
  - Learning rate: 0.001
  - PPO epochs: 5
  - Mini-batches: 4
  - KL target: 0.01
  - Clip range: 0.2
  - Entropy coefficient: 0.005
  - Discount factor γ: 0.99

Usage:
    trainer = MBPOPPOTrainer(
        world_model=rwm, policy=policy, value_fn=value_fn,
        env=env, config=cfg, logger=logger
    )
    trainer.train(num_iterations=2500)
"""

import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from data.replay_buffer import ReplayBuffer
from data.trajectory_dataset import TrajectoryDataset
from envs.base_env import BaseEnv
from models.policy import PolicyNetwork, ValueNetwork
from models.rwm import GRUWorldModel
from training.rwm_trainer import RWMTrainer
from utils.common import get_device, sample_gaussian
from utils.logger import Logger

# ---------------------------------------------------------------------------
# Training constants
# ---------------------------------------------------------------------------

# Maximum gradient norm for policy and value function gradient clipping.
# Prevents gradient explosion during PPO updates, especially important
# when the world model produces high-variance imagined trajectories.
_MAX_GRAD_NORM: float = 1.0

# KL early stopping multiplier: stop PPO epochs if approx_kl > target * multiplier
# Standard PPO practice to prevent policy collapse.
_KL_EARLY_STOP_MULTIPLIER: float = 1.5

# Small epsilon for advantage normalization to prevent division by zero.
_ADV_NORM_EPS: float = 1e-8

# Default checkpoint directory
_DEFAULT_CHECKPOINT_DIR: str = "checkpoints"

# Default log interval (iterations between metric logging)
_DEFAULT_LOG_INTERVAL: int = 10

# Default eval interval (iterations between policy evaluation)
_DEFAULT_EVAL_INTERVAL: int = 100

# Default save interval (iterations between checkpoint saves)
_DEFAULT_SAVE_INTERVAL: int = 500

# Default number of evaluation episodes
_DEFAULT_EVAL_EPISODES: int = 10

# Minimum number of trajectories in buffer before training world model
_MIN_BUFFER_SIZE: int = 1


class MBPOPPOTrainer:
    """Model-Based Policy Optimization with PPO for the RWM framework.

    Implements Algorithm 1 from the paper, orchestrating the full MBPO-PPO
    training loop: real data collection, world model fine-tuning, imagination
    rollouts, and PPO policy updates.

    The trainer operates on a single real environment for online data collection
    (``simulation.num_envs_real=1``) while using 4096 parallel imagined
    environments for policy optimization (``mbpo_ppo.imagination_envs=4096``).

    **Key architectural insight:** The policy gradient does NOT flow through
    the world model dynamics. PPO is a zeroth-order method that uses importance
    sampling ratios computed from stored log-probabilities. This is the key
    difference from SHAC (which uses first-order gradients through the world
    model) and is why MBPO-PPO is more robust to discontinuous dynamics
    (Section 4.4 of the paper).

    Attributes:
        world_model: The GRU world model for imagination rollouts and fine-tuning.
        policy: The policy network mapping observations to action distributions.
        value_fn: The value function network for GAE advantage estimation.
        env: The real environment for data collection and policy evaluation.
        config: Full experiment configuration from config.yaml.
        logger: Shared logger for metrics and checkpoints.
        device: Target device (CUDA or CPU) for all tensor operations.
        robot_type: Robot identifier string ("anymal_d" or "unitree_g1").
        obs_dim: World model observation dimension (45 for ANYmal D, 96 for G1).
        action_dim: Action space dimension (12 for ANYmal D, 29 for G1).
        priv_dim: Privileged information dimension (8 for ANYmal D, 30 for G1).
        policy_obs_dim: Policy observation dimension (48 for ANYmal D, 99 for G1).
        history_horizon: GRU context length M=32 (Table S10).
        imagination_envs: Number of parallel imagination environments (4096).
        imagination_steps: Steps per imagination rollout T=100 (Table S11).
        gamma: Discount factor γ=0.99 (Table S11).
        gae_lambda: GAE lambda λ=0.95 (config.yaml).
        clip_range: PPO clip range ε=0.2 (Table S11).
        entropy_coef: Entropy bonus coefficient=0.005 (Table S11).
        kl_target: KL divergence target=0.01 (Table S11).
        learning_epochs: PPO update epochs=5 (Table S11).
        num_mini_batches: PPO mini-batches=4 (Table S11).
        termination_threshold: Threshold for imagined episode termination=0.5.
        policy_optimizer: Adam optimizer for the policy network.
        value_optimizer: Adam optimizer for the value function.
        rwm_trainer: RWMTrainer instance for world model fine-tuning.
        replay_buffer: Trajectory-level replay buffer for real data storage.
        current_iteration: Current training iteration counter.
        checkpoint_dir: Directory for saving checkpoints.
        log_interval: Iterations between metric logging.
        eval_interval: Iterations between policy evaluation.
        save_interval: Iterations between checkpoint saves.
        eval_episodes: Number of episodes for policy evaluation.
    """

    def __init__(
        self,
        world_model: GRUWorldModel,
        policy: PolicyNetwork,
        value_fn: ValueNetwork,
        env: BaseEnv,
        config: Any,
        logger: Logger,
    ) -> None:
        """Initialize the MBPO-PPO trainer from the experiment configuration.

        Resolves all training hyperparameters from the config, initializes
        the Adam optimizers for policy and value function, creates the
        RWMTrainer for world model fine-tuning, and initializes the replay
        buffer for real data storage.

        Args:
            world_model: Instantiated GRU world model from ``models/rwm.py``.
                Must already be on the target device. Used for both imagination
                rollouts (``step`` method) and fine-tuning (via ``rwm_trainer``).
            policy: Instantiated policy network from ``models/policy.py``.
                Must already be on the target device. Maps policy observations
                to Gaussian action distributions.
            value_fn: Instantiated value function from ``models/policy.py``.
                Must already be on the target device. Maps policy observations
                to scalar value estimates for GAE.
            env: Real environment instance (``ANYmalEnv``, ``UnitreeG1Env``,
                or ``MockEnv``). Used for data collection and policy evaluation.
                Must implement the ``BaseEnv`` interface.
            config: Hydra ``DictConfig`` or plain dict containing the full
                experiment configuration from ``config.yaml``. Must contain:
                - ``config.robot``: "anymal_d" or "unitree_g1"
                - ``config.anymal_d`` or ``config.unitree_g1``: robot sub-config
                - ``config.mbpo_ppo``: MBPO-PPO hyperparameters (Table S11)
                - ``config.rwm``: world model architecture config
                - ``config.simulation``: simulation parameters
                - ``config.collision_handling``: collision handling config
                - ``config.device``: "cuda" or "cpu"
                - ``config.logging``: logging configuration
                - ``config.checkpoint_dir``: checkpoint directory path
            logger: Shared logger instance from ``utils/logger.py``.
                Used for metric logging and checkpoint tracking.

        Raises:
            ValueError: If ``config.robot`` is not "anymal_d" or "unitree_g1".
            KeyError: If required config fields are missing.
        """
        # ----------------------------------------------------------------
        # 1. Store component references
        # ----------------------------------------------------------------
        self.world_model: GRUWorldModel = world_model
        self.policy: PolicyNetwork = policy
        self.value_fn: ValueNetwork = value_fn
        self.env: BaseEnv = env
        self.config: Any = config
        self.logger: Logger = logger

        # ----------------------------------------------------------------
        # 2. Resolve device
        # ----------------------------------------------------------------
        self.device: torch.device = get_device(str(config.device))

        # ----------------------------------------------------------------
        # 3. Resolve robot type and extract robot-specific dimensions
        # ----------------------------------------------------------------
        robot_type: str = str(config.robot)
        _supported_robots: Tuple[str, ...] = ("anymal_d", "unitree_g1")
        if robot_type not in _supported_robots:
            raise ValueError(
                f"Unsupported robot type '{robot_type}' in config.robot. "
                f"Expected one of: {_supported_robots}. "
                "Check the 'robot' field in config.yaml."
            )
        self.robot_type: str = robot_type

        # Access robot-specific sub-config (e.g., config.anymal_d)
        robot_cfg = config[robot_type]

        # Dimensions from Tables S2-S5
        self.obs_dim: int = int(robot_cfg.obs_dim)
        self.action_dim: int = int(robot_cfg.action_dim)
        self.priv_dim: int = int(robot_cfg.priv_dim)
        self.policy_obs_dim: int = int(robot_cfg.policy_obs_dim)

        # Store obs slices for observation space conversion
        self.obs_slices: Any = robot_cfg.obs_slices
        self.policy_obs_slices: Any = robot_cfg.policy_obs_slices
        self.priv_slices: Any = robot_cfg.priv_slices

        # Reward weights from Table S6
        self.reward_weights: Dict[str, float] = {
            k: float(v) for k, v in robot_cfg.reward_weights.items()
        }

        # ----------------------------------------------------------------
        # 4. Extract MBPO-PPO hyperparameters from config.mbpo_ppo (Table S11)
        # ----------------------------------------------------------------
        ppo_cfg = config.mbpo_ppo

        # Imagination settings (Table S11)
        self.imagination_envs: int = int(ppo_cfg.imagination_envs)   # 4096
        self.imagination_steps: int = int(ppo_cfg.imagination_steps)  # 100

        # Replay buffer size (Table S11: |D| = 1000 trajectories)
        self.buffer_size: int = int(ppo_cfg.buffer_size)  # 1000

        # Optimization hyperparameters (Table S11)
        self.learning_rate: float = float(ppo_cfg.learning_rate)    # 0.001
        self.weight_decay: float = float(ppo_cfg.weight_decay)      # 0.0
        self.learning_epochs: int = int(ppo_cfg.learning_epochs)    # 5
        self.num_mini_batches: int = int(ppo_cfg.num_mini_batches)  # 4
        self.kl_target: float = float(ppo_cfg.kl_divergence_target) # 0.01
        self.gamma: float = float(ppo_cfg.discount_factor)          # 0.99
        self.clip_range: float = float(ppo_cfg.clip_range)          # 0.2
        self.entropy_coef: float = float(ppo_cfg.entropy_coefficient) # 0.005

        # GAE lambda — standard PPO value, from config.yaml
        self.gae_lambda: float = float(ppo_cfg.gae_lambda)  # 0.95

        # Max training iterations (Table S11)
        self.max_iterations: int = int(ppo_cfg.max_iterations)  # 2500

        # ----------------------------------------------------------------
        # 5. Extract simulation parameters
        # ----------------------------------------------------------------
        sim_cfg = config.simulation
        self.dt: float = float(sim_cfg.dt)  # 0.02s (50 Hz)

        # ----------------------------------------------------------------
        # 6. Extract collision handling config (Section A.4.3)
        # ----------------------------------------------------------------
        collision_cfg = config.collision_handling
        self.termination_threshold: float = float(
            collision_cfg.termination_threshold
        )  # 0.5

        # ----------------------------------------------------------------
        # 7. Extract RWM architecture parameters for history encoding
        # ----------------------------------------------------------------
        rwm_cfg = config.rwm
        self.history_horizon: int = int(rwm_cfg.history_horizon)  # 32

        # ----------------------------------------------------------------
        # 8. Initialize Adam optimizers (Table S11)
        # ----------------------------------------------------------------
        # Policy optimizer: lr=0.001, weight_decay=0.0 (Table S11)
        self.policy_optimizer: torch.optim.Adam = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Value function optimizer: same hyperparameters as policy
        self.value_optimizer: torch.optim.Adam = torch.optim.Adam(
            self.value_fn.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # ----------------------------------------------------------------
        # 9. Initialize RWMTrainer for world model fine-tuning
        # ----------------------------------------------------------------
        # The RWMTrainer handles the world model optimization (Algorithm 1,
        # step 4). It is initialized here and called once per outer iteration
        # to fine-tune the world model on newly collected real data.
        self.rwm_trainer: RWMTrainer = RWMTrainer(
            model=self.world_model,
            config=config,
            logger=logger,
        )

        # ----------------------------------------------------------------
        # 10. Initialize replay buffer for real data storage
        # ----------------------------------------------------------------
        # buffer_size=1000 trajectories (Table S11: |D| = 1000)
        # Stores complete trajectories from the single real environment.
        self.replay_buffer: ReplayBuffer = ReplayBuffer(
            buffer_size=self.buffer_size,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            priv_dim=self.priv_dim,
            device=str(config.device),
        )

        # ----------------------------------------------------------------
        # 11. Extract logging and checkpoint configuration
        # ----------------------------------------------------------------
        try:
            self.log_interval: int = int(config.logging.log_interval)
        except (AttributeError, KeyError):
            self.log_interval = _DEFAULT_LOG_INTERVAL

        try:
            self.eval_interval: int = int(config.logging.eval_interval)
        except (AttributeError, KeyError):
            self.eval_interval = _DEFAULT_EVAL_INTERVAL

        try:
            self.save_interval: int = int(config.logging.save_interval)
        except (AttributeError, KeyError):
            self.save_interval = _DEFAULT_SAVE_INTERVAL

        try:
            self.eval_episodes: int = int(config.logging.eval_episodes)
        except (AttributeError, KeyError):
            self.eval_episodes = _DEFAULT_EVAL_EPISODES

        try:
            self.checkpoint_dir: str = str(config.checkpoint_dir)
        except (AttributeError, KeyError):
            self.checkpoint_dir = _DEFAULT_CHECKPOINT_DIR

        # ----------------------------------------------------------------
        # 12. Initialize training state tracking
        # ----------------------------------------------------------------
        # Current iteration counter for checkpoint naming and logging
        self.current_iteration: int = 0

        # Estimated reward history for Fig. 5 reproduction
        self._estimated_reward_history: List[float] = []
        self._gt_reward_history: List[float] = []

        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        print(
            f"[MBPOPPOTrainer] Initialized for robot '{self.robot_type}'. "
            f"obs_dim={self.obs_dim}, action_dim={self.action_dim}, "
            f"policy_obs_dim={self.policy_obs_dim}, "
            f"imagination_envs={self.imagination_envs}, "
            f"imagination_steps={self.imagination_steps}, "
            f"device={self.device}"
        )

    # ----------------------------------------------------------------
    # Private helper methods
    # ----------------------------------------------------------------

    def _construct_policy_obs(
        self,
        wm_obs: Tensor,
        command: Tensor,
        last_action: Tensor,
    ) -> Tensor:
        """Convert world model observations to policy observations.

        The policy observation (Table S5) differs from the world model
        observation (Table S2) in two ways:
          1. Does NOT contain joint torques (which are in the world model obs).
          2. DOES contain the velocity command and last action.

        This conversion is performed at every step of the imagination loop
        and during real data collection.

        **ANYmal D mapping (45-dim wm_obs → 48-dim policy_obs):**
          - base_lin_vel [0:3] from wm_obs[0:3]
          - base_ang_vel [3:6] from wm_obs[3:6]
          - gravity [6:9] from wm_obs[6:9]
          - velocity_command [9:12] from command[0:3]
          - joint_pos [12:24] from wm_obs[9:21]
          - joint_vel [24:36] from wm_obs[21:33]
          - last_actions [36:48] from last_action[0:12]
          Total: 3+3+3+3+12+12+12 = 48 ✓

        **Unitree G1 mapping (96-dim wm_obs → 99-dim policy_obs):**
          - base_lin_vel [0:3] from wm_obs[0:3]
          - base_ang_vel [3:6] from wm_obs[3:6]
          - gravity [6:9] from wm_obs[6:9]
          - velocity_command [9:12] from command[0:3]
          - joint_pos [12:41] from wm_obs[9:38]
          - joint_vel [41:70] from wm_obs[38:67]
          - last_actions [70:99] from last_action[0:29]
          Total: 3+3+3+3+29+29+29 = 99 ✓

        Args:
            wm_obs: World model observation of shape ``[B, obs_dim]``.
                Contains base velocities, gravity, joint positions,
                velocities, and torques (Table S2). Torques are NOT
                included in the policy observation.
            command: Velocity commands of shape ``[B, 3]``.
                ``[vx, vy, yaw_rate]`` to be included in policy obs.
            last_action: Last action taken, shape ``[B, action_dim]``.
                Joint position targets from the previous step.

        Returns:
            Policy observation tensor of shape ``[B, policy_obs_dim]``.
            For ANYmal D: ``[B, 48]``. For Unitree G1: ``[B, 99]``.
        """
        # Extract the shared prefix: base_lin_vel + base_ang_vel + gravity
        # These are always at obs[0:9] for both robots (Table S2).
        base_state: Tensor = wm_obs[:, 0:9]  # shape [B, 9]

        # Extract joint positions and velocities (robot-specific slices).
        # Joint torques (the remaining dims) are NOT included in policy obs.
        if self.robot_type == "anymal_d":
            # ANYmal D: joint_pos=9:21 (12-dim), joint_vel=21:33 (12-dim)
            joint_pos: Tensor = wm_obs[:, 9:21]   # shape [B, 12]
            joint_vel: Tensor = wm_obs[:, 21:33]  # shape [B, 12]
        else:
            # Unitree G1: joint_pos=9:38 (29-dim), joint_vel=38:67 (29-dim)
            joint_pos = wm_obs[:, 9:38]    # shape [B, 29]
            joint_vel = wm_obs[:, 38:67]   # shape [B, 29]

        # Concatenate: [base_state | command | joint_pos | joint_vel | last_action]
        # This matches Table S5 ordering exactly.
        policy_obs: Tensor = torch.cat(
            [base_state, command, joint_pos, joint_vel, last_action],
            dim=-1,
        )  # shape [B, policy_obs_dim]

        return policy_obs

    def _get_history_for_initial_obs(
        self,
        init_obs: Tensor,
        init_commands: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Retrieve M-step history context for GRU initialization.

        For each of the n_envs sampled initial observations, retrieves the
        preceding M=32 steps from the replay buffer to warm up the GRU
        hidden state. If insufficient history is available (trajectory shorter
        than M), pads with zeros at the beginning.

        This implements the "Initialize imagination agents with observations
        sampled from D" step of Algorithm 1, extended to include the GRU
        context needed for accurate predictions.

        Args:
            init_obs: Initial observations sampled from replay buffer,
                shape ``[n_envs, obs_dim]``. These are the starting
                observations for imagination rollouts.
            init_commands: Corresponding velocity commands,
                shape ``[n_envs, 3]``.

        Returns:
            A tuple ``(obs_history, action_history)`` where:
              - ``obs_history``: shape ``[n_envs, M, obs_dim]`` — M historical
                observations for GRU inner autoregression context.
              - ``action_history``: shape ``[n_envs, M, action_dim]`` — M
                historical actions for GRU inner autoregression context.
        """
        n_envs: int = init_obs.shape[0]
        M: int = self.history_horizon

        # Pre-allocate history tensors (zero-padded by default)
        obs_history: Tensor = torch.zeros(
            n_envs, M, self.obs_dim,
            dtype=torch.float32,
            device=self.device,
        )
        action_history: Tensor = torch.zeros(
            n_envs, M, self.action_dim,
            dtype=torch.float32,
            device=self.device,
        )

        # Get all trajectories from the buffer for history lookup
        trajectories: List[Dict[str, object]] = self.replay_buffer.get_all_trajectories()

        if not trajectories:
            # Buffer is empty — return zero-padded history
            # This happens at the very start of training before any data is collected
            return obs_history, action_history

        # For each environment, sample a trajectory and extract M steps of history
        # ending at a random position (consistent with how init_obs was sampled)
        import random
        num_trajectories: int = len(trajectories)

        for i in range(n_envs):
            # Sample a random trajectory
            traj_idx: int = random.randint(0, num_trajectories - 1)
            traj: Dict[str, object] = trajectories[traj_idx]

            traj_length: int = int(traj["length"])  # type: ignore[arg-type]
            traj_obs: Tensor = traj["obs"]           # type: ignore[assignment]  [T, obs_dim]
            traj_actions: Tensor = traj["actions"]   # type: ignore[assignment]  [T, action_dim]

            if traj_length <= 0:
                continue

            # Sample a random end position within the trajectory
            # (the position corresponding to init_obs[i])
            end_t: int = random.randint(0, traj_length - 1)

            # Extract up to M steps of history ending at end_t
            start_t: int = max(0, end_t - M + 1)
            available_steps: int = end_t - start_t + 1

            # Fill the history buffer (right-aligned, zero-padded on the left)
            # This means the most recent steps are at the end of the M-step window
            fill_start: int = M - available_steps  # where to start filling in the M-dim window

            obs_history[i, fill_start:, :] = traj_obs[start_t : end_t + 1].to(self.device)
            action_history[i, fill_start:, :] = traj_actions[start_t : end_t + 1].to(self.device)

        return obs_history, action_history

    def _detect_termination(self, priv_mean: Tensor) -> Tensor:
        """Detect episode termination from world model privileged info predictions.

        The paper states: "We explicitly train RWM to predict such terminations
        in its privileged information prediction head." (Section A.4.3)

        For ANYmal D (priv_dim=8): knee contacts (0:4) and foot contacts (4:8).
        Base contact is not directly in the standard priv space. We use a
        heuristic: if the gravity vector indicates the robot is upside down
        (gravity z-component > 0 in robot frame), terminate.

        For practical implementation, we use the last dimension of the priv
        prediction as a termination logit (sigmoid > threshold). This is
        consistent with the paper's statement about training RWM to predict
        terminations.

        **Implementation note:** The termination signal is derived from the
        priv head's predictions. For ANYmal D, we use the maximum knee contact
        probability as a proxy for base contact (knees touching ground often
        precedes base contact). For Unitree G1, we use the body contact flags.

        Args:
            priv_mean: Predicted privileged info means (logits for binary dims)
                of shape ``[B, priv_dim]``. For ANYmal D: ``[B, 8]``.
                For Unitree G1: ``[B, 30]``.

        Returns:
            Boolean termination flags of shape ``[B]``. True for environments
            where the world model predicts an unsafe state (base contact).
        """
        if self.robot_type == "anymal_d":
            # ANYmal D: use knee contact probability as termination proxy
            # knee_contact: priv[0:4] — high knee contact probability suggests
            # the robot is falling or has collapsed
            # Use sigmoid to convert logits to probabilities
            knee_contact_probs: Tensor = torch.sigmoid(priv_mean[:, 0:4])
            # Terminate if any knee contact probability exceeds threshold
            # AND all foot contacts are also high (robot is on its knees/side)
            max_knee_prob: Tensor = knee_contact_probs.max(dim=-1).values  # [B]
            done: Tensor = max_knee_prob > self.termination_threshold
        else:
            # Unitree G1: use body contact flags (0:26) as termination signal
            # High body contact probability (especially torso/pelvis) indicates fall
            body_contact_probs: Tensor = torch.sigmoid(priv_mean[:, 0:26])
            # Use mean body contact probability as termination signal
            mean_body_contact: Tensor = body_contact_probs.mean(dim=-1)  # [B]
            done = mean_body_contact > self.termination_threshold

        return done  # shape [B], bool

    # ----------------------------------------------------------------
    # Public interface methods
    # ----------------------------------------------------------------

    def collect_real_data(self, n_steps: int = 100) -> None:
        """Collect real environment data and store in the replay buffer.

        Runs the current policy in the real environment for ``n_steps`` steps,
        collecting (obs, action, priv, command) tuples. Stores complete
        trajectories in the replay buffer for world model fine-tuning.

        This implements Algorithm 1, step 3: "Collect observation-action pairs
        in D by interacting with the environment using π_θ."

        The policy is run in inference mode (``torch.no_grad()``) for
        efficiency. The real environment uses a single environment instance
        (``simulation.num_envs_real=1``) as described in Section A.4.3.

        Collision handling (Section A.4.3): If the environment terminates
        due to base contact, the partial trajectory up to termination is
        stored in the buffer (it is still valid training data for the world
        model, as it contains the transition leading to the unsafe state).

        Args:
            n_steps: Number of environment steps to collect. Corresponds to
                ``mbpo_ppo.imagination_steps=100`` in config.yaml (same
                length as imagination rollouts for consistency). Default: 100.
        """
        # Set policy to eval mode for inference (disables dropout if any)
        self.policy.eval()

        # Initialize trajectory storage
        obs_list: List[Tensor] = []
        action_list: List[Tensor] = []
        priv_list: List[Tensor] = []
        command_list: List[Tensor] = []

        # Reset environment and get initial state
        obs: Tensor
        command: Tensor
        obs, command = self.env.reset()
        # obs: [num_envs, obs_dim], command: [num_envs, 3]
        # For single env: [1, obs_dim] and [1, 3]

        # Initialize last action to zeros (no previous action at episode start)
        num_envs: int = obs.shape[0]
        last_action: Tensor = torch.zeros(
            num_envs,
            self.action_dim,
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            for step in range(n_steps):
                # ----------------------------------------------------------------
                # 1. Construct policy observation from world model observation
                # ----------------------------------------------------------------
                policy_obs: Tensor = self._construct_policy_obs(
                    wm_obs=obs,
                    command=command,
                    last_action=last_action,
                )
                # policy_obs: [num_envs, policy_obs_dim]

                # ----------------------------------------------------------------
                # 2. Sample action from current policy
                # ----------------------------------------------------------------
                action: Tensor
                _log_prob: Tensor
                action, _log_prob = self.policy.get_action(policy_obs)
                