## envs/base_env.py
"""Abstract base class defining the environment interface for the RWM project.

This module provides ``BaseEnv``, the abstract contract that all environment
implementations must satisfy. It sits between simulation backends (Isaac Lab,
mock) and the training/evaluation code. Both ``RWMTrainer`` (via data
collection) and ``MBPOPPOTrainer`` (via ``collect_real_data`` and
``_evaluate_policy``) interact exclusively through this interface.

The base class provides:
  - Shared initialization logic (dimension extraction, reward config, buffers)
  - A concrete default ``compute_reward`` implementation delegating to
    ``envs/rewards.py`` stateless functions
  - A concrete ``construct_policy_obs`` helper for converting world model
    observations to policy observations
  - Helper utilities for command sampling and partial environment resets

Concrete subclasses (``ANYmalEnv``, ``UnitreeG1Env``, ``MockEnv``) must
implement the four abstract methods: ``reset``, ``step``,
``get_privileged_info``, and ``close``.

Observation indexing follows Tables S2–S5 from the paper exactly. All
robot-specific dimensions are resolved from the hydra config at construction
time to avoid repeated OmegaConf lookups in hot paths.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

from envs.rewards import (
    action_rate_penalty,
    angular_vel_tracking,
    angular_vel_xy_penalty,
    feet_air_time,
    flat_orientation,
    foot_clearance,
    joint_accel_penalty,
    joint_deviation,
    joint_torque_penalty,
    linear_vel_tracking,
    linear_vel_z_penalty,
    undesired_contacts,
)


class BaseEnv(ABC):
    """Abstract base class for all RWM environment wrappers.

    Defines the interface contract for environment interaction and provides
    shared initialization, reward computation, and observation construction
    logic. All simulator-specific code lives in concrete subclasses.

    The environment operates on batches of ``num_envs`` parallel environments.
    All returned tensors have a leading batch dimension of size ``num_envs``.

    Attributes:
        obs_dim: Dimension of the world model observation (Table S2).
            45 for ANYmal D, 96 for Unitree G1.
        action_dim: Dimension of the action space (Table S4).
            12 for ANYmal D, 29 for Unitree G1.
        priv_dim: Dimension of the privileged information space (Table S3).
            8 for ANYmal D, 30 for Unitree G1.
        policy_obs_dim: Dimension of the policy observation (Table S5).
            48 for ANYmal D, 99 for Unitree G1.
        num_envs: Number of parallel environments in this instance.
        device: PyTorch device string for all tensors.
        robot_type: Robot identifier string. Either "anymal_d" or "unitree_g1".
        dt: Simulation time step in seconds (0.02 = 1/50 Hz).
        control_freq: Control frequency in Hz (50 Hz per paper Section 4.1).
        reward_weights: Dict of reward term weights from Table S6.
        sigma_vxy: Temperature factor for linear velocity tracking reward (0.25).
        sigma_wz: Temperature factor for angular velocity tracking reward (0.25).
        obs_slices: Dict mapping observation field names to [start, end] lists.
        policy_obs_slices: Dict mapping policy obs field names to [start, end].
        priv_slices: Dict mapping privileged info field names to [start, end].
        terminate_on_base_contact: Whether to terminate on base ground contact.
        prev_actions: Previous action buffer of shape [num_envs, action_dim].
            Updated after each step for action rate penalty computation.
        default_joint_pos: Default joint positions of shape [action_dim].
            Initialized to zeros; subclasses override with actual robot pose.
    """

    def __init__(self, config: Any, num_envs: int = 1) -> None:
        """Initialize shared environment state from the hydra config.

        Resolves robot-specific configuration from the master config object,
        stores all dimensions and reward parameters as plain Python attributes,
        and initializes stateful buffers (prev_actions, default_joint_pos).

        Subclasses must call ``super().__init__(config, num_envs)`` before
        performing any simulator-specific initialization.

        Args:
            config: Hydra ``DictConfig`` or plain dict containing the full
                experiment configuration from ``config.yaml``. Must contain:
                - ``config.robot``: "anymal_d" or "unitree_g1"
                - ``config.anymal_d`` or ``config.unitree_g1``: robot sub-config
                - ``config.device``: device string
                - ``config.simulation``: simulation parameters
                - ``config.collision_handling``: collision config
            num_envs: Number of parallel environments to manage. Corresponds
                to ``simulation.num_envs_real`` (1) for online fine-tuning or
                ``simulation.num_envs_pretrain`` (4096) for data collection.
                Default: 1.

        Raises:
            ValueError: If ``config.robot`` is not "anymal_d" or "unitree_g1".
            KeyError: If required config fields are missing.
        """
        # ----------------------------------------------------------------
        # Validate and resolve robot type
        # ----------------------------------------------------------------
        robot_type: str = str(config.robot)
        _supported_robots = ("anymal_d", "unitree_g1")
        if robot_type not in _supported_robots:
            raise ValueError(
                f"Unsupported robot type '{robot_type}' in config.robot. "
                f"Expected one of: {_supported_robots}. "
                "Check the 'robot' field in config.yaml."
            )
        self.robot_type: str = robot_type

        # Resolve robot-specific sub-config (e.g., config.anymal_d)
        # OmegaConf DictConfig supports attribute and item access interchangeably.
        robot_cfg = config[robot_type]

        # ----------------------------------------------------------------
        # Core dimensions (Tables S2–S5)
        # ----------------------------------------------------------------
        self.obs_dim: int = int(robot_cfg.obs_dim)
        self.action_dim: int = int(robot_cfg.action_dim)
        self.priv_dim: int = int(robot_cfg.priv_dim)
        self.policy_obs_dim: int = int(robot_cfg.policy_obs_dim)

        # ----------------------------------------------------------------
        # Structural metadata
        # ----------------------------------------------------------------
        self.num_envs: int = int(num_envs)
        self.device: str = str(config.device)
        self.dt: float = float(config.simulation.dt)
        self.control_freq: int = int(config.simulation.control_freq_hz)

        # ----------------------------------------------------------------
        # Observation and privileged info slices
        # Stored as plain dicts of [start, end] lists (from OmegaConf).
        # Convert to plain Python dicts to avoid OmegaConf overhead in
        # hot paths and to ensure JSON serializability.
        # ----------------------------------------------------------------
        self.obs_slices: Dict[str, Any] = dict(robot_cfg.obs_slices)
        self.policy_obs_slices: Dict[str, Any] = dict(robot_cfg.policy_obs_slices)
        self.priv_slices: Dict[str, Any] = dict(robot_cfg.priv_slices)

        # ----------------------------------------------------------------
        # Reward configuration (Table S6)
        # ----------------------------------------------------------------
        self.reward_weights: Dict[str, float] = {
            k: float(v) for k, v in robot_cfg.reward_weights.items()
        }
        self.sigma_vxy: float = float(robot_cfg.sigma_vxy)
        self.sigma_wz: float = float(robot_cfg.sigma_wz)

        # ----------------------------------------------------------------
        # Collision handling config (Section A.4.3)
        # ----------------------------------------------------------------
        self.terminate_on_base_contact: bool = bool(
            config.collision_handling.terminate_on_base_contact
        )

        # ----------------------------------------------------------------
        # Stateful buffers — initialized on the target device
        # ----------------------------------------------------------------
        # Previous action buffer for action rate penalty: r_adot = w * ||a' - a||^2
        # Reset to zeros at episode start; updated after each step.
        self.prev_actions: Tensor = torch.zeros(
            self.num_envs,
            self.action_dim,
            dtype=torch.float32,
            device=self.device,
        )

        # Default joint positions for joint deviation reward: r_qd = w * ||q - q_0||_1
        # Initialized to zeros; subclasses override with actual robot default pose.
        self.default_joint_pos: Tensor = torch.zeros(
            self.action_dim,
            dtype=torch.float32,
            device=self.device,
        )

        # Current velocity commands — stored for use in compute_reward during step.
        # Shape: [num_envs, 3] — [vx, vy, yaw_rate]
        self.commands: Tensor = torch.zeros(
            self.num_envs,
            3,
            dtype=torch.float32,
            device=self.device,
        )

        # Closed flag for idempotent close() calls
        self._closed: bool = False

    # ----------------------------------------------------------------
    # Abstract methods — must be implemented by all subclasses
    # ----------------------------------------------------------------

    @abstractmethod
    def reset(self) -> Tuple[Tensor, Tensor]:
        """Reset all environments and return initial observations and commands.

        Must reset ``self.prev_actions`` to zeros for all environments.
        Must sample new velocity commands and store them in ``self.commands``.

        Returns:
            A tuple ``(obs, command)`` where:
              - ``obs``: Initial world model observation of shape
                ``[num_envs, obs_dim]``. Contains base velocities, gravity,
                joint positions, velocities, and torques (Table S2).
              - ``command``: Sampled velocity commands of shape
                ``[num_envs, 3]``. Each row is ``[vx, vy, yaw_rate]`` in m/s
                and rad/s respectively.
        """

    @abstractmethod
    def step(
        self,
        action: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]:
        """Apply actions to all environments and return the resulting state.

        Applies joint position targets, advances the simulation by one step
        (dt=0.02s), reads back the new state, computes rewards, and detects
        episode terminations.

        After computing rewards, must update ``self.prev_actions = action.clone()``
        for the next step's action rate penalty computation. Must zero
        ``self.prev_actions`` for any environments where ``done=True``.

        Args:
            action: Joint position targets of shape ``[num_envs, action_dim]``.
                Values are in radians (joint position targets for PD control).

        Returns:
            A tuple ``(next_obs, priv, reward, done, info)`` where:
              - ``next_obs``: Next world model observation of shape
                ``[num_envs, obs_dim]``.
              - ``priv``: Privileged information of shape
                ``[num_envs, priv_dim]``. Contains contact flags, foot
                heights, foot velocities (Table S3).
              - ``reward``: Scalar reward for each environment, shape
                ``[num_envs]``.
              - ``done``: Boolean termination flags, shape ``[num_envs]``.
                True when base contact is detected (if
                ``terminate_on_base_contact=True``) or episode time limit
                is reached.
              - ``info``: Dictionary of diagnostic information. May include
                keys like "episode_length", "tracking_error", etc.
        """

    @abstractmethod
    def get_privileged_info(self) -> Tensor:
        """Return the current privileged information for all environments.

        Privileged information is used as an additional learning objective
        for the world model (Section 3.2). It includes contact flags, foot
        heights, and foot velocities that are not directly observable by the
        policy but help the world model learn accurate dynamics.

        Returns:
            Privileged information tensor of shape ``[num_envs, priv_dim]``.
            For ANYmal D: 8-dim (4 knee contacts + 4 foot contacts, Table S3).
            For Unitree G1: 30-dim (26 body contacts + 2 foot heights +
            2 foot velocities, Table S3).
        """

    @abstractmethod
    def close(self) -> None:
        """Release all simulator resources and shut down the environment.

        Must be idempotent — calling close() multiple times must not raise
        errors. Subclasses should guard with ``self._closed`` flag.

        After calling close(), the environment instance should not be used.
        """

    # ----------------------------------------------------------------
    # Concrete methods — shared across all subclasses
    # ----------------------------------------------------------------

    def compute_reward(
        self,
        obs: Tensor,
        action: Tensor,
        next_obs: Tensor,
        priv: Tensor,
        command: Tensor,
        t_fa: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute the total reward by summing all 12 reward terms.

        Delegates to the stateless functions in ``envs/rewards.py``, passing
        the stored reward weights, sigma values, and stateful buffers
        (``self.prev_actions``, ``self.default_joint_pos``).

        This method is **not abstract** — all subclasses inherit this
        implementation. Override only if non-standard reward logic is needed.

        **Observation type requirement:** ``obs`` and ``next_obs`` must be
        the **world model observation** (45-dim for ANYmal D, 96-dim for G1)
        which contains joint torques (Table S2). The policy observation
        (48-dim / 99-dim) does NOT contain torques and cannot be used here.

        The method does NOT update ``self.prev_actions`` — that is the
        responsibility of ``step()`` to ensure correct ordering.

        Args:
            obs: Current world model observation of shape ``[B, obs_dim]``.
                Must contain joint torques at the robot-specific slice.
            action: Current action (joint position targets) of shape
                ``[B, action_dim]``.
            next_obs: Next world model observation of shape ``[B, obs_dim]``.
                Used for joint acceleration finite differencing.
            priv: Privileged information of shape ``[B, priv_dim]``.
                Contains contact flags, foot heights, foot velocities.
            command: Velocity commands of shape ``[B, 3]``.
                ``[vx, vy, yaw_rate]`` for tracking reward computation.
            t_fa: Pre-computed feet air time tensor of shape ``[B]``.
                Sum of air time (seconds) across all feet per environment.
                If None, defaults to zeros (acceptable when ``w_fa=0.0``
                for Unitree G1 or when contact tracking is unavailable in
                imagination rollouts). Default: None.

        Returns:
            Total reward tensor of shape ``[B]``. Sum of all 12 reward terms
            weighted by ``self.reward_weights`` (Table S6).
        """
        batch_size: int = obs.shape[0]
        w = self.reward_weights

        # Default feet air time to zeros if not provided.
        # This is safe because w_fa=0.0 for Unitree G1, and in imagination
        # rollouts contact tracking is not available.
        if t_fa is None:
            t_fa = torch.zeros(batch_size, dtype=obs.dtype, device=obs.device)

        # ----------------------------------------------------------------
        # Compute all 12 reward terms (Section A.1.2)
        # ----------------------------------------------------------------

        # 1. Linear velocity tracking (x, y): r_vxy = w * exp(-||c_xy - v_xy||^2 / sigma^2)
        r_vxy: Tensor = linear_vel_tracking(
            obs=obs,
            command=command,
            w=w["w_vxy"],
            sigma=self.sigma_vxy,
        )

        # 2. Angular velocity tracking (yaw): r_wz = w * exp(-||c_z - omega_z||^2 / sigma^2)
        r_wz: Tensor = angular_vel_tracking(
            obs=obs,
            command=command,
            w=w["w_wz"],
            sigma=self.sigma_wz,
        )

        # 3. Linear velocity z penalty: r_vz = w * ||v_z||^2
        r_vz: Tensor = linear_vel_z_penalty(obs=obs, w=w["w_vz"])

        # 4. Angular velocity xy penalty: r_wxy = w * ||omega_xy||^2
        r_wxy: Tensor = angular_vel_xy_penalty(obs=obs, w=w["w_wxy"])

        # 5. Joint torque penalty: r_qtau = w * ||tau||^2
        r_qtau: Tensor = joint_torque_penalty(
            obs=obs,
            w=w["w_qtau"],
            robot_type=self.robot_type,
        )

        # 6. Joint acceleration penalty: r_qdd = w * ||q_ddot||^2
        #    Requires consecutive observations for finite differencing.
        r_qdd: Tensor = joint_accel_penalty(
            obs=obs,
            next_obs=next_obs,
            w=w["w_qdd"],
            robot_type=self.robot_type,
            dt=self.dt,
        )

        # 7. Action rate penalty: r_adot = w * ||a' - a||^2
        #    Uses self.prev_actions (updated by step() after reward computation)
        r_adot: Tensor = action_rate_penalty(
            action=action,
            prev_action=self.prev_actions[:batch_size],
            w=w["w_adot"],
        )

        # 8. Feet air time reward: r_fa = w * t_fa
        r_fa: Tensor = feet_air_time(t_fa=t_fa, w=w["w_fa"])

        # 9. Undesired contacts penalty: r_c = w * c_u
        r_c: Tensor = undesired_contacts(
            priv=priv,
            w=w["w_c"],
            robot_type=self.robot_type,
        )

        # 10. Flat orientation penalty: r_g = w * g_xy^2
        r_g: Tensor = flat_orientation(obs=obs, w=w["w_g"])

        # 11. Foot clearance reward: r_fc = w * h_fc
        r_fc: Tensor = foot_clearance(
            priv=priv,
            w=w["w_fc"],
            robot_type=self.robot_type,
        )

        # 12. Joint deviation penalty: r_qd = w * ||q - q_0||_1
        r_qd: Tensor = joint_deviation(
            obs=obs,
            default_pos=self.default_joint_pos,
            w=w["w_qd"],
            robot_type=self.robot_type,
        )

        # ----------------------------------------------------------------
        # Sum all reward terms element-wise
        # ----------------------------------------------------------------
        total_reward: Tensor = (
            r_vxy + r_wz + r_vz + r_wxy + r_qtau + r_qdd
            + r_adot + r_fa + r_c + r_g + r_fc + r_qd
        )  # shape [B]

        return total_reward

    def construct_policy_obs(
        self,
        wm_obs: Tensor,
        command: Tensor,
        last_action: Tensor,
    ) -> Tensor:
        """Construct the policy observation from the world model observation.

        The policy observation (Table S5) differs from the world model
        observation (Table S2) in two ways:
          1. It does NOT contain joint torques (which are in the world model obs).
          2. It DOES contain the velocity command and last action.

        This conversion is needed in ``MBPOPPOTrainer.imagine_trajectories``
        to feed the correct observation to the policy network during rollouts.

        **ANYmal D (48-dim policy obs from 45-dim world model obs):**
          - base_lin_vel [0:3] from wm_obs[0:3]
          - base_ang_vel [3:6] from wm_obs[3:6]
          - gravity [6:9] from wm_obs[6:9]
          - joint_pos [9:21] from wm_obs[9:21]
          - joint_vel [21:33] from wm_obs[21:33]
          - velocity_command [33:36] from command[0:3]
          - last_actions [36:48] from last_action[0:12]
          Total: 9 + 12 + 12 + 3 + 12 = 48 ✓

        **Unitree G1 (99-dim policy obs from 96-dim world model obs):**
          - base_lin_vel [0:3] from wm_obs[0:3]
          - base_ang_vel [3:6] from wm_obs[3:6]
          - gravity [6:9] from wm_obs[6:9]
          - joint_pos [9:38] from wm_obs[9:38]
          - joint_vel [38:67] from wm_obs[38:67]
          - velocity_command [67:70] from command[0:3]
          - last_actions [70:99] from last_action[0:29]
          Total: 9 + 29 + 29 + 3 + 29 = 99 ✓

        Args:
            wm_obs: World model observation of shape ``[B, obs_dim]``.
                Contains base velocities, gravity, joint positions,
                velocities, and torques (Table S2).
            command: Velocity commands of shape ``[B, 3]``.
                ``[vx, vy, yaw_rate]`` to be appended to policy obs.
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
        # For ANYmal D: joint_pos=9:21, joint_vel=21:33 (no torques in policy obs)
        # For Unitree G1: joint_pos=9:38, joint_vel=38:67 (no torques in policy obs)
        if self.robot_type == "anymal_d":
            joint_pos: Tensor = wm_obs[:, 9:21]   # shape [B, 12]
            joint_vel: Tensor = wm_obs[:, 21:33]  # shape [B, 12]
        else:  # unitree_g1
            joint_pos = wm_obs[:, 9:38]    # shape [B, 29]
            joint_vel = wm_obs[:, 38:67]   # shape [B, 29]

        # Concatenate: [base_state | joint_pos | joint_vel | command | last_action]
        policy_obs: Tensor = torch.cat(
            [base_state, joint_pos, joint_vel, command, last_action],
            dim=-1,
        )  # shape [B, policy_obs_dim]

        return policy_obs

    def _sample_commands(self, num_envs: int) -> Tensor:
        """Sample random velocity commands for environment reset.

        Samples uniformly from a reasonable velocity command range for
        velocity tracking tasks. The command distribution covers the range
        of velocities the robot should learn to track.

        Command ranges (reasonable defaults for legged locomotion):
          - vx (forward): [-1.0, 1.0] m/s
          - vy (lateral): [-0.5, 0.5] m/s
          - yaw_rate: [-1.0, 1.0] rad/s

        Subclasses can override this method to use task-specific command
        distributions (e.g., curriculum learning with increasing ranges).

        Args:
            num_envs: Number of command vectors to sample. Typically
                ``self.num_envs``.

        Returns:
            Velocity command tensor of shape ``[num_envs, 3]`` on
            ``self.device``. Each row is ``[vx, vy, yaw_rate]``.
        """
        # Sample uniformly from command ranges
        commands: Tensor = torch.zeros(
            num_envs, 3, dtype=torch.float32, device=self.device
        )
        # vx: forward velocity in [-1.0, 1.0] m/s
        commands[:, 0].uniform_(-1.0, 1.0)
        # vy: lateral velocity in [-0.5, 0.5] m/s
        commands[:, 1].uniform_(-0.5, 0.5)
        # yaw_rate: angular velocity in [-1.0, 1.0] rad/s
        commands[:, 2].uniform_(-1.0, 1.0)
        return commands

    def _reset_done_envs(
        self,
        obs: Tensor,
        done: Tensor,
        reset_obs: Tensor,
        reset_commands: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Reset observations and commands for environments where done=True.

        Applies partial resets: only environments with ``done[i]=True`` are
        reset to the provided ``reset_obs`` and ``reset_commands``. Environments
        with ``done[i]=False`` retain their current observations and commands.

        Also zeros ``self.prev_actions`` for done environments to prevent
        stale action history from affecting the action rate penalty after reset.

        This helper is called by subclass ``step()`` implementations after
        detecting termination conditions.

        Args:
            obs: Current observations of shape ``[num_envs, obs_dim]``.
                Will be updated in-place for done environments.
            done: Boolean termination flags of shape ``[num_envs]``.
                True for environments that need resetting.
            reset_obs: Reset observations of shape ``[num_envs, obs_dim]``.
                Provides the initial observation for each done environment.
                Typically the output of the simulator's reset call.
            reset_commands: Reset velocity commands of shape ``[num_envs, 3]``.
                New commands sampled for the reset environments.

        Returns:
            A tuple ``(updated_obs, updated_commands)`` where done environments
            have been replaced with reset values:
              - ``updated_obs``: shape ``[num_envs, obs_dim]``
              - ``updated_commands``: shape ``[num_envs, 3]``
        """
        if not done.any():
            # Fast path: no environments need resetting
            return obs, self.commands

        done_mask: Tensor = done.bool()  # shape [num_envs]

        # Replace observations for done environments
        obs[done_mask] = reset_obs[done_mask]

        # Update stored commands for done environments
        self.commands[done_mask] = reset_commands[done_mask]

        # Zero previous actions for done environments to prevent stale
        # action history from corrupting the action rate penalty
        self.prev_actions[done_mask] = 0.0

        return obs, self.commands.clone()

    def _update_prev_actions(
        self,
        action: Tensor,
        done: Optional[Tensor] = None,
    ) -> None:
        """Update the previous action buffer after a step.

        Must be called by subclass ``step()`` implementations AFTER computing
        rewards (which use the old ``self.prev_actions``) and AFTER handling
        episode terminations (which zero ``self.prev_actions`` for done envs).

        The update order in ``step()`` must be:
          1. Compute reward using current ``self.prev_actions``
          2. Detect done environments
          3. Zero ``self.prev_actions`` for done environments (via
             ``_reset_done_envs`` or directly)
          4. Call ``_update_prev_actions(action, done)`` to store current
             action as previous for the next step

        Args:
            action: Current action tensor of shape ``[num_envs, action_dim]``.
                Will be stored as ``self.prev_actions`` for the next step.
            done: Optional boolean termination flags of shape ``[num_envs]``.
                If provided, done environments will have their prev_actions
                zeroed BEFORE storing the new action. This handles the case
                where the next step starts a new episode. Default: None.
        """
        if done is not None and done.any():
            # Zero prev_actions for done environments — the next step will
            # start a new episode where the "previous action" should be zero
            self.prev_actions[done.bool()] = 0.0

        # Store current action as previous for the next step
        self.prev_actions.copy_(action.detach())
