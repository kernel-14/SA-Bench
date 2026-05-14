"""Task definitions for velocity tracking experiments.

Defines velocity command generation used during both real environment interaction
and imagination rollouts. The policy receives a velocity command as part of its
observation space (Table S5).
"""

from typing import Tuple
import numpy as np
import torch


class VelocityCommandGenerator:
    """Generates random velocity commands for locomotion tasks.

    Commands are sampled uniformly from specified ranges:
      - Linear velocity x: [-1.0, 1.0] m/s
      - Linear velocity y: [-0.5, 0.5] m/s
      - Angular velocity z: [-1.0, 1.0] rad/s
    """

    def __init__(
        self,
        vx_range: Tuple[float, float] = (-1.0, 1.0),
        vy_range: Tuple[float, float] = (-0.5, 0.5),
        omega_z_range: Tuple[float, float] = (-1.0, 1.0),
        resample_interval: int = 100,  # steps between command resamples
    ):
        self.vx_range = vx_range
        self.vy_range = vy_range
        self.omega_z_range = omega_z_range
        self.resample_interval = resample_interval
        self._step_counter = 0
        self._current_command = np.zeros(3)

    def reset(self) -> np.ndarray:
        self._step_counter = 0
        self._current_command = self._sample()
        return self._current_command.copy()

    def step(self) -> np.ndarray:
        self._step_counter += 1
        if self._step_counter % self.resample_interval == 0:
            self._current_command = self._sample()
        return self._current_command.copy()

    def _sample(self) -> np.ndarray:
        vx = float(np.random.uniform(*self.vx_range))
        vy = float(np.random.uniform(*self.vy_range))
        omega_z = float(np.random.uniform(*self.omega_z_range))
        return np.array([vx, vy, omega_z])

    def sample_batch(self, batch_size: int) -> np.ndarray:
        """Sample a batch of random commands."""
        commands = np.stack([self._sample() for _ in range(batch_size)], axis=0)
        return commands


class VelocityTrackingTask:
    """Velocity tracking task wrapper.

    Provides the interface between policy observations and velocity commands.
    The policy observation space includes the velocity command as defined in
    Table S5:
      ANYmal D:
        - base linear velocity [0:3]
        - base angular velocity [3:6]
        - projected gravity [6:9]
        - velocity command [9:12]
        - joint positions [12:24]
        - joint velocities [24:36]
        - last actions [36:48]
    """

    def __init__(self, robot_spec, device: str = "cuda"):
        self.robot_spec = robot_spec
        self.device = device
        self.command_gen = VelocityCommandGenerator()

    def build_policy_obs(
        self,
        world_model_obs: torch.Tensor,
        velocity_command: torch.Tensor,
        prev_action: torch.Tensor,
    ) -> torch.Tensor:
        """Build the full policy observation from world model observation components.

        Args:
            world_model_obs: (..., obs_dim) — full world model observation
            velocity_command: (..., 3) — [cmd_vx, cmd_vy, cmd_omega_z]
            prev_action: (..., action_dim) — previous action

        Returns:
            policy_obs: (..., policy_obs_dim)
        """
        spec = self.robot_spec

        # Extract relevant components from world model observation
        base_lin_vel = world_model_obs[..., spec.obs_base_lin_vel[0]:spec.obs_base_lin_vel[1]]
        base_ang_vel = world_model_obs[..., spec.obs_base_ang_vel[0]:spec.obs_base_ang_vel[1]]
        projected_gravity = world_model_obs[..., spec.obs_projected_gravity[0]:spec.obs_projected_gravity[1]]

        num_joints = spec.num_joints
        joint_pos = world_model_obs[..., spec.obs_joint_pos_start:spec.obs_joint_pos_start + num_joints]

        # Joint velocities may or may not be present
        if spec.obs_joint_vel_start is not None:
            joint_vel = world_model_obs[..., spec.obs_joint_vel_start:spec.obs_joint_vel_start + num_joints]
        else:
            joint_vel = torch.zeros(
                *world_model_obs.shape[:-1], num_joints, device=world_model_obs.device
            )

        policy_obs = torch.cat(
            [
                base_lin_vel,   # 3
                base_ang_vel,   # 3
                projected_gravity,  # 3
                velocity_command,  # 3
                joint_pos,      # num_joints
                joint_vel,      # num_joints
                prev_action,    # action_dim
            ],
            dim=-1,
        )

        return policy_obs
