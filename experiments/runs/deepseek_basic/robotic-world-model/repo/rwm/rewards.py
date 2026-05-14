"""
Reward functions for velocity tracking tasks.

Based on Section A.1.2 of the paper (Table S6).

The total reward is sum of the following components:
- Linear velocity tracking (x, y)
- Angular velocity tracking (z)
- Linear velocity penalty (z)
- Angular velocity penalty (x, y)
- Joint torque penalty
- Joint acceleration penalty
- Action rate penalty
- Feet air time
- Undesired contacts
- Flat orientation
- Foot clearance
- Joint deviation
"""

import torch
import numpy as np
from typing import Dict, Optional


class VelocityTrackingReward:
    """
    Reward function for velocity tracking tasks (ANYmal D and Unitree G1).
    
    Observation space (Table S5):
        - base linear velocity v[0:3]
        - base angular velocity omega[3:6]
        - projected gravity g[6:9]
        - velocity command c[9:12]
        - joint positions q[12:24/41]
        - joint velocities qdot[24:36/70]
        - last actions a_prev[36:48/99]
    """
    
    def __init__(
        self,
        robot_type: str = 'anymal_d',
        device: torch.device = None,
    ):
        self.device = device or torch.device('cpu')
        
        if robot_type == 'anymal_d':
            self._init_anymal_weights()
            self.num_joints = 12
        elif robot_type == 'unitree_g1':
            self._init_unitree_weights()
            self.num_joints = 29
        else:
            raise ValueError(f"Unknown robot type: {robot_type}")
            
    def _init_anymal_weights(self):
        """Reward weights for ANYmal D (Table S6)."""
        self.w_vxy = 1.0      # Linear velocity tracking
        self.w_omega_z = 0.5  # Angular velocity tracking
        self.w_vz = -2.0       # Linear velocity z penalty
        self.w_omega_xy = -0.05  # Angular velocity xy penalty
        self.w_q_tau = -2.5e-5  # Joint torque penalty
        self.w_q_ddot = -2.5e-7  # Joint acceleration penalty
        self.w_a_dot = -0.01    # Action rate penalty
        self.w_fa = 0.5         # Feet air time
        self.w_c = -1.0         # Undesired contacts
        self.w_g = -5.0         # Flat orientation
        self.w_fc = 0.0         # Foot clearance
        self.w_qd = 0.0         # Joint deviation
        
    def _init_unitree_weights(self):
        """Reward weights for Unitree G1 (Table S6)."""
        self.w_vxy = 1.0
        self.w_omega_z = 0.5
        self.w_vz = -2.0
        self.w_omega_xy = -0.05
        self.w_q_tau = -2.5e-5
        self.w_q_ddot = -2.5e-7
        self.w_a_dot = -0.05
        self.w_fa = 0.0
        self.w_c = -1.0
        self.w_g = -5.0
        self.w_fc = 1.0
        self.w_qd = -1.0
        
    def __call__(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        prev_action: Optional[torch.Tensor] = None,
        priv: Optional[torch.Tensor] = None,
        prev_obs: Optional[torch.Tensor] = None,
        default_joint_pos: Optional[torch.Tensor] = None,
        feet_air_time: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute total reward.
        
        Args:
            obs: (batch, obs_dim) current observation
            action: (batch, act_dim) current action
            prev_action: (batch, act_dim) previous action (for action rate)
            priv: (batch, priv_dim) privileged info (for contacts)
            prev_obs: (batch, obs_dim) previous observation (for joint accel)
            default_joint_pos: (batch, num_joints) default joint positions
            feet_air_time: (batch,) feet air time sum
            
        Returns:
            (batch,) total reward
        """
        reward = torch.zeros(obs.shape[0], device=obs.device)
        
        # Extract observation components
        v = obs[:, 0:3]       # Base linear velocity
        omega = obs[:, 3:6]   # Base angular velocity
        g = obs[:, 6:9]       # Projected gravity
        cmd = obs[:, 9:12]    # Velocity command
        
        # Need to know exact slicing for joints which depends on robot
        joint_start = 12
        joint_end = joint_start + self.num_joints
        q = obs[:, joint_start:joint_end]  # Joint positions
        
        vel_start = joint_end
        vel_end = vel_start + self.num_joints
        qdot = obs[:, vel_start:vel_end]  # Joint velocities
        
        # 1. Linear velocity tracking (x, y)
        v_xy = v[:, :2]
        cmd_xy = cmd[:, :2]
        r_vxy = self.w_vxy * torch.exp(
            -torch.sum((cmd_xy - v_xy) ** 2, dim=-1) / (0.25 ** 2)
        )
        reward += r_vxy
        
        # 2. Angular velocity tracking (z)
        omega_z = omega[:, 2]
        cmd_z = cmd[:, 2]
        r_omega_z = self.w_omega_z * torch.exp(
            -((cmd_z - omega_z) ** 2) / (0.25 ** 2)
        )
        reward += r_omega_z
        
        # 3. Linear velocity z penalty
        v_z = v[:, 2]
        r_vz = self.w_vz * (v_z ** 2)
        reward += r_vz
        
        # 4. Angular velocity xy penalty
        omega_xy = omega[:, :2]
        r_omega_xy = self.w_omega_xy * torch.sum(omega_xy ** 2, dim=-1)
        reward += r_omega_xy
        
        # 5. Joint torque penalty
        # Torques are in obs for world model; for policy obs they may not be
        # We estimate from action if needed
        r_tau = self.w_q_tau * torch.sum(action ** 2, dim=-1)
        reward += r_tau
        
        # 6. Joint acceleration penalty
        if prev_obs is not None:
            prev_qdot = prev_obs[:, vel_start:vel_end]
            q_ddot = (qdot - prev_qdot) / 0.02  # dt = 0.02
            r_q_ddot = self.w_q_ddot * torch.sum(q_ddot ** 2, dim=-1)
            reward += r_q_ddot
        
        # 7. Action rate penalty
        if prev_action is not None:
            r_a_dot = self.w_a_dot * torch.sum((action - prev_action) ** 2, dim=-1)
            reward += r_a_dot
        
        # 8. Feet air time
        if feet_air_time is not None:
            r_fa = self.w_fa * feet_air_time
            reward += r_fa
        
        # 9. Undesired contacts
        if priv is not None:
            # priv contains contact information
            # Count contacts as undesired
            r_c = self.w_c * torch.sum(priv, dim=-1)
            reward += r_c
        
        # 10. Flat orientation
        g_xy = g[:, :2]
        r_g = self.w_g * torch.sum(g_xy ** 2, dim=-1)
        reward += r_g
        
        # 11. Foot clearance (would need foot height from priv)
        if priv is not None and self.w_fc != 0:
            # Simplified: use relevant priv components
            r_fc = self.w_fc * torch.sum(priv[:, :2], dim=-1)  # heuristic
            reward += r_fc
        
        # 12. Joint deviation
        if default_joint_pos is not None and self.w_qd != 0:
            r_qd = self.w_qd * torch.sum(torch.abs(q - default_joint_pos), dim=-1)
            reward += r_qd
        
        return reward


def create_velocity_tracking_reward(robot_type: str = 'anymal_d'):
    """Factory function for velocity tracking reward."""
    return VelocityTrackingReward(robot_type=robot_type)
