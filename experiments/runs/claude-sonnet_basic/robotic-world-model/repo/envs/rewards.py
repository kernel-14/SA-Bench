"""
Reward functions for velocity tracking tasks.

Reward formulations from Section A.1.2 and Table S6.

ANYmal D reward weights:
  w_vxy = 1.0, w_wz = 0.5
  w_vz = -2.0, w_wxy = -0.05
  w_qt = -2.5e-5, w_q = -2.5e-7
  w_a = -0.01, w_fa = 0.5
  w_c = -1.0, w_g = -5.0
  w_fc = 0.0, w_qd = 0.0

Unitree G1 reward weights:
  w_vxy = 1.0, w_wz = 0.5
  w_vz = -2.0, w_wxy = -0.05
  w_qt = -2.5e-5, w_q = -2.5e-7
  w_a = -0.05, w_fa = 0.0
  w_c = -1.0, w_g = -5.0
  w_fc = 1.0, w_qd = -1.0
"""

import torch
import numpy as np
from typing import Optional, Dict


# Observation space indices for ANYmal D (Table S5)
ANYMAL_OBS = {
    "base_lin_vel": slice(0, 3),
    "base_ang_vel": slice(3, 6),
    "projected_gravity": slice(6, 9),
    "velocity_command": slice(9, 12),
    "joint_positions": slice(12, 24),
    "joint_velocities": slice(24, 36),
    "last_actions": slice(36, 48),
}

# Observation space indices for Unitree G1 (Table S5)
G1_OBS = {
    "base_lin_vel": slice(0, 3),
    "base_ang_vel": slice(3, 6),
    "projected_gravity": slice(6, 9),
    "velocity_command": slice(9, 12),
    "joint_positions": slice(12, 41),
    "joint_velocities": slice(41, 70),
    "last_actions": slice(70, 99),
}

# World model observation space for ANYmal D (Table S2)
ANYMAL_WM_OBS = {
    "base_lin_vel": slice(0, 3),
    "base_ang_vel": slice(3, 6),
    "projected_gravity": slice(6, 9),
    "joint_positions": slice(9, 21),
    "joint_velocities": slice(21, 33),
    "joint_torques": slice(33, 45),
}

# World model observation space for Unitree G1 (Table S2)
G1_WM_OBS = {
    "base_lin_vel": slice(0, 3),
    "base_ang_vel": slice(3, 6),
    "projected_gravity": slice(6, 9),
    "joint_positions": slice(9, 38),
    "joint_velocities": slice(38, 67),
    "joint_torques": slice(67, 96),
}

# Privileged info for ANYmal D (Table S3)
ANYMAL_PRIV = {
    "knee_contact": slice(0, 4),
    "foot_contact": slice(4, 8),
}

# Privileged info for Unitree G1 (Table S3)
G1_PRIV = {
    "body_contact": slice(0, 26),
    "foot_height": slice(26, 28),
    "foot_velocity": slice(28, 30),
}


class VelocityTrackingReward:
    """
    Velocity tracking reward for legged locomotion.

    Implements all reward terms from Section A.1.2.
    """

    def __init__(
        self,
        robot: str = "anymal",
        sigma_vxy: float = 0.25,
        sigma_wz: float = 0.25,
    ):
        assert robot in ("anymal", "g1"), f"Unknown robot: {robot}"
        self.robot = robot
        self.sigma_vxy = sigma_vxy
        self.sigma_wz = sigma_wz

        if robot == "anymal":
            self.obs_indices = ANYMAL_OBS
            self.weights = {
                "w_vxy": 1.0,
                "w_wz": 0.5,
                "w_vz": -2.0,
                "w_wxy": -0.05,
                "w_qt": -2.5e-5,
                "w_q": -2.5e-7,
                "w_a": -0.01,
                "w_fa": 0.5,
                "w_c": -1.0,
                "w_g": -5.0,
                "w_fc": 0.0,
                "w_qd": 0.0,
            }
        else:  # g1
            self.obs_indices = G1_OBS
            self.weights = {
                "w_vxy": 1.0,
                "w_wz": 0.5,
                "w_vz": -2.0,
                "w_wxy": -0.05,
                "w_qt": -2.5e-5,
                "w_q": -2.5e-7,
                "w_a": -0.05,
                "w_fa": 0.0,
                "w_c": -1.0,
                "w_g": -5.0,
                "w_fc": 1.0,
                "w_qd": -1.0,
            }

    def __call__(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        priv_info: Optional[torch.Tensor] = None,
        prev_action: Optional[torch.Tensor] = None,
        joint_torques: Optional[torch.Tensor] = None,
        joint_accel: Optional[torch.Tensor] = None,
        feet_air_time: Optional[torch.Tensor] = None,
        undesired_contacts: Optional[torch.Tensor] = None,
        foot_clearance: Optional[torch.Tensor] = None,
        default_joint_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute total reward.

        Args:
            obs: (batch, obs_size) - current observation (policy obs space)
            action: (batch, action_size) - current action
            next_obs: (batch, obs_size) - next observation
            priv_info: (batch, priv_size) - privileged information
            prev_action: (batch, action_size) - previous action for action rate
            joint_torques: (batch, n_joints) - joint torques
            joint_accel: (batch, n_joints) - joint accelerations
            feet_air_time: (batch,) - sum of feet air time
            undesired_contacts: (batch,) - count of undesired contacts
            foot_clearance: (batch,) - foot clearance height
            default_joint_pos: (batch, n_joints) - default joint positions

        Returns:
            reward: (batch,)
        """
        w = self.weights
        idx = self.obs_indices

        # Extract from observation
        base_lin_vel = obs[:, idx["base_lin_vel"]]  # (batch, 3)
        base_ang_vel = obs[:, idx["base_ang_vel"]]  # (batch, 3)
        projected_gravity = obs[:, idx["projected_gravity"]]  # (batch, 3)
        velocity_command = obs[:, idx["velocity_command"]]  # (batch, 3)
        joint_pos = obs[:, idx["joint_positions"]]
        joint_vel = obs[:, idx["joint_velocities"]]
        last_action = obs[:, idx["last_actions"]]

        # Linear velocity tracking (xy)
        cmd_xy = velocity_command[:, :2]
        vel_xy = base_lin_vel[:, :2]
        r_vxy = w["w_vxy"] * torch.exp(
            -torch.sum((cmd_xy - vel_xy) ** 2, dim=-1) / (self.sigma_vxy ** 2)
        )

        # Angular velocity tracking (z)
        cmd_z = velocity_command[:, 2]
        ang_vel_z = base_ang_vel[:, 2]
        r_wz = w["w_wz"] * torch.exp(
            -(cmd_z - ang_vel_z) ** 2 / (self.sigma_wz ** 2)
        )

        # Linear velocity z penalty
        vel_z = base_lin_vel[:, 2]
        r_vz = w["w_vz"] * vel_z ** 2

        # Angular velocity xy penalty
        ang_vel_xy = base_ang_vel[:, :2]
        r_wxy = w["w_wxy"] * torch.sum(ang_vel_xy ** 2, dim=-1)

        # Joint torque penalty
        r_qt = torch.zeros_like(r_vxy)
        if joint_torques is not None:
            r_qt = w["w_qt"] * torch.sum(joint_torques ** 2, dim=-1)

        # Joint acceleration penalty
        r_q = torch.zeros_like(r_vxy)
        if joint_accel is not None:
            r_q = w["w_q"] * torch.sum(joint_accel ** 2, dim=-1)

        # Action rate penalty
        r_a = torch.zeros_like(r_vxy)
        if prev_action is not None:
            r_a = w["w_a"] * torch.sum((last_action - action) ** 2, dim=-1)

        # Feet air time reward
        r_fa = torch.zeros_like(r_vxy)
        if feet_air_time is not None:
            r_fa = w["w_fa"] * feet_air_time

        # Undesired contacts penalty
        r_c = torch.zeros_like(r_vxy)
        if undesired_contacts is not None:
            r_c = w["w_c"] * undesired_contacts.float()

        # Flat orientation penalty
        g_xy = projected_gravity[:, :2]
        r_g = w["w_g"] * torch.sum(g_xy ** 2, dim=-1)

        # Foot clearance reward
        r_fc = torch.zeros_like(r_vxy)
        if foot_clearance is not None:
            r_fc = w["w_fc"] * foot_clearance

        # Joint deviation penalty
        r_qd = torch.zeros_like(r_vxy)
        if default_joint_pos is not None:
            r_qd = w["w_qd"] * torch.sum(torch.abs(joint_pos - default_joint_pos), dim=-1)

        total_reward = r_vxy + r_wz + r_vz + r_wxy + r_qt + r_q + r_a + r_fa + r_c + r_g + r_fc + r_qd

        return total_reward

    def compute_from_wm_obs(
        self,
        wm_obs: torch.Tensor,
        action: torch.Tensor,
        next_wm_obs: torch.Tensor,
        velocity_command: torch.Tensor,
        priv_info: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute reward from world model observation space.

        World model obs doesn't include velocity command, so it's passed separately.

        Args:
            wm_obs: (batch, wm_obs_size) - world model observation
            action: (batch, action_size)
            next_wm_obs: (batch, wm_obs_size)
            velocity_command: (batch, 3) - [vx_cmd, vy_cmd, wz_cmd]
            priv_info: (batch, priv_size)

        Returns:
            reward: (batch,)
        """
        if self.robot == "anymal":
            wm_idx = ANYMAL_WM_OBS
        else:
            wm_idx = G1_WM_OBS

        w = self.weights

        base_lin_vel = wm_obs[:, wm_idx["base_lin_vel"]]
        base_ang_vel = wm_obs[:, wm_idx["base_ang_vel"]]
        projected_gravity = wm_obs[:, wm_idx["projected_gravity"]]
        joint_torques = wm_obs[:, wm_idx["joint_torques"]]

        # Linear velocity tracking (xy)
        cmd_xy = velocity_command[:, :2]
        vel_xy = base_lin_vel[:, :2]
        r_vxy = w["w_vxy"] * torch.exp(
            -torch.sum((cmd_xy - vel_xy) ** 2, dim=-1) / (self.sigma_vxy ** 2)
        )

        # Angular velocity tracking (z)
        cmd_z = velocity_command[:, 2]
        ang_vel_z = base_ang_vel[:, 2]
        r_wz = w["w_wz"] * torch.exp(
            -(cmd_z - ang_vel_z) ** 2 / (self.sigma_wz ** 2)
        )

        # Linear velocity z penalty
        vel_z = base_lin_vel[:, 2]
        r_vz = w["w_vz"] * vel_z ** 2

        # Angular velocity xy penalty
        ang_vel_xy = base_ang_vel[:, :2]
        r_wxy = w["w_wxy"] * torch.sum(ang_vel_xy ** 2, dim=-1)

        # Joint torque penalty
        r_qt = w["w_qt"] * torch.sum(joint_torques ** 2, dim=-1)

        # Flat orientation penalty
        g_xy = projected_gravity[:, :2]
        r_g = w["w_g"] * torch.sum(g_xy ** 2, dim=-1)

        total_reward = r_vxy + r_wz + r_vz + r_wxy + r_qt + r_g

        return total_reward
