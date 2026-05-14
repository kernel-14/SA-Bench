"""
Reward functions for velocity-tracking locomotion tasks (Sec. A.1.2).

All rewards are computed from the policy observation vector, which contains:
  ANYmal D  (48-dim): v(3) | omega(3) | g(3) | cmd(3) | q(12) | q_dot(12) | a_prev(12)
  Unitree G1 (99-dim): v(3) | omega(3) | g(3) | cmd(3) | q(29) | q_dot(29) | a_prev(29)

Privileged information (for contact-based rewards):
  ANYmal D  (8-dim):  knee_contact(4) | foot_contact(4)
  Unitree G1 (30-dim): body_contact(26) | foot_height(2) | foot_velocity(2)
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Observation index helpers
# ---------------------------------------------------------------------------

@dataclass
class AnymalDObsIndex:
    """Slice indices for ANYmal D policy observation (48-dim)."""
    v_lin: slice = slice(0, 3)
    v_ang: slice = slice(3, 6)
    gravity: slice = slice(6, 9)
    cmd: slice = slice(9, 12)
    q: slice = slice(12, 24)
    q_dot: slice = slice(24, 36)
    a_prev: slice = slice(36, 48)


@dataclass
class UnitreeG1ObsIndex:
    """Slice indices for Unitree G1 policy observation (99-dim)."""
    v_lin: slice = slice(0, 3)
    v_ang: slice = slice(3, 6)
    gravity: slice = slice(6, 9)
    cmd: slice = slice(9, 12)
    q: slice = slice(12, 41)
    q_dot: slice = slice(41, 70)
    a_prev: slice = slice(70, 99)


@dataclass
class AnymalDPrivIndex:
    """Slice indices for ANYmal D privileged info (8-dim)."""
    knee_contact: slice = slice(0, 4)
    foot_contact: slice = slice(4, 8)


@dataclass
class UnitreeG1PrivIndex:
    """Slice indices for Unitree G1 privileged info (30-dim)."""
    body_contact: slice = slice(0, 26)
    foot_height: slice = slice(26, 28)
    foot_velocity: slice = slice(28, 30)


ANYMAL_OBS_IDX = AnymalDObsIndex()
G1_OBS_IDX = UnitreeG1ObsIndex()
ANYMAL_PRIV_IDX = AnymalDPrivIndex()
G1_PRIV_IDX = UnitreeG1PrivIndex()


# ---------------------------------------------------------------------------
# Individual reward terms
# ---------------------------------------------------------------------------

def reward_linear_velocity_xy(
    obs: torch.Tensor,
    obs_idx,
    sigma: float = 0.25,
    weight: float = 1.0,
) -> torch.Tensor:
    """
    r_vxy = w * exp(-||c_xy - v_xy||^2 / sigma^2)   (Eq. A.4)
    """
    v_xy = obs[..., obs_idx.v_lin][..., :2]
    cmd_xy = obs[..., obs_idx.cmd][..., :2]
    error = ((cmd_xy - v_xy) ** 2).sum(dim=-1)
    return weight * torch.exp(-error / (sigma ** 2))


def reward_angular_velocity_z(
    obs: torch.Tensor,
    obs_idx,
    sigma: float = 0.25,
    weight: float = 0.5,
) -> torch.Tensor:
    """
    r_wz = w * exp(-||c_z - omega_z||^2 / sigma^2)   (Eq. A.5)
    """
    omega_z = obs[..., obs_idx.v_ang][..., 2:3]
    cmd_z = obs[..., obs_idx.cmd][..., 2:3]
    error = ((cmd_z - omega_z) ** 2).sum(dim=-1)
    return weight * torch.exp(-error / (sigma ** 2))


def reward_linear_velocity_z(
    obs: torch.Tensor,
    obs_idx,
    weight: float = -2.0,
) -> torch.Tensor:
    """
    r_vz = w * ||v_z||^2   (Eq. A.6)
    """
    v_z = obs[..., obs_idx.v_lin][..., 2:3]
    return weight * (v_z ** 2).sum(dim=-1)


def reward_angular_velocity_xy(
    obs: torch.Tensor,
    obs_idx,
    weight: float = -0.05,
) -> torch.Tensor:
    """
    r_wxy = w * ||omega_xy||^2   (Eq. A.7)
    """
    omega_xy = obs[..., obs_idx.v_ang][..., :2]
    return weight * (omega_xy ** 2).sum(dim=-1)


def reward_joint_torque(
    torques: torch.Tensor,
    weight: float = -2.5e-5,
) -> torch.Tensor:
    """
    r_qtau = w * ||tau||^2   (Eq. A.8)
    """
    return weight * (torques ** 2).sum(dim=-1)


def reward_joint_acceleration(
    q_ddot: torch.Tensor,
    weight: float = -2.5e-7,
) -> torch.Tensor:
    """
    r_qddot = w * ||q_ddot||^2   (Eq. A.9)
    """
    return weight * (q_ddot ** 2).sum(dim=-1)


def reward_action_rate(
    obs: torch.Tensor,
    actions: torch.Tensor,
    obs_idx,
    weight: float = -0.01,
) -> torch.Tensor:
    """
    r_adot = w * ||a_prev - a||^2   (Eq. A.10)
    """
    a_prev = obs[..., obs_idx.a_prev]
    return weight * ((a_prev - actions) ** 2).sum(dim=-1)


def reward_feet_air_time(
    feet_air_time: torch.Tensor,
    weight: float = 0.5,
) -> torch.Tensor:
    """
    r_fa = w * t_fa   (Eq. A.11)
    feet_air_time: sum of time feet are in the air
    """
    return weight * feet_air_time


def reward_undesired_contacts(
    undesired_contact_count: torch.Tensor,
    weight: float = -1.0,
) -> torch.Tensor:
    """
    r_c = w * c_u   (Eq. A.12)
    """
    return weight * undesired_contact_count


def reward_flat_orientation(
    obs: torch.Tensor,
    obs_idx,
    weight: float = -5.0,
) -> torch.Tensor:
    """
    r_g = w * g_xy^2   (Eq. A.13)
    """
    g_xy = obs[..., obs_idx.gravity][..., :2]
    return weight * (g_xy ** 2).sum(dim=-1)


def reward_foot_clearance(
    foot_clearance_height: torch.Tensor,
    weight: float = 0.0,
) -> torch.Tensor:
    """
    r_fc = w * h_fc   (Eq. A.14)
    """
    return weight * foot_clearance_height


def reward_joint_deviation(
    obs: torch.Tensor,
    obs_idx,
    default_joint_pos: torch.Tensor,
    weight: float = 0.0,
) -> torch.Tensor:
    """
    r_qd = w * ||q - q_0||_1   (Eq. A.15)
    """
    q = obs[..., obs_idx.q]
    return weight * (q - default_joint_pos).abs().sum(dim=-1)


# ---------------------------------------------------------------------------
# Composite reward computers
# ---------------------------------------------------------------------------

class AnymalDRewardComputer:
    """
    Computes the total reward for ANYmal D velocity tracking (Table S6).

    Requires:
      - policy_obs: (B, 48) policy observation
      - actions:    (B, 12) current actions
      - privileged: (B, 8)  privileged info (contacts)
      - torques:    (B, 12) joint torques (from world model obs or sim)
      - q_ddot:     (B, 12) joint accelerations
      - feet_air_time: (B,) sum of feet air time
      - undesired_contacts: (B,) count of undesired contacts
      - foot_clearance: (B,) foot clearance height
      - default_joint_pos: (12,) default joint positions
    """

    def __init__(self, weights: Dict[str, float]):
        self.w = weights
        self.idx = ANYMAL_OBS_IDX
        self.priv_idx = ANYMAL_PRIV_IDX

    def compute(
        self,
        policy_obs: torch.Tensor,
        actions: torch.Tensor,
        torques: torch.Tensor,
        q_ddot: torch.Tensor,
        feet_air_time: torch.Tensor,
        undesired_contacts: torch.Tensor,
        foot_clearance: torch.Tensor,
        default_joint_pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        w = self.w
        idx = self.idx

        r_vxy = reward_linear_velocity_xy(policy_obs, idx, w["sigma_vxy"], w["w_vxy"])
        r_wz = reward_angular_velocity_z(policy_obs, idx, w["sigma_wz"], w["w_wz"])
        r_vz = reward_linear_velocity_z(policy_obs, idx, w["w_vz"])
        r_wxy = reward_angular_velocity_xy(policy_obs, idx, w["w_wxy"])
        r_qtau = reward_joint_torque(torques, w["w_qtau"])
        r_qddot = reward_joint_acceleration(q_ddot, w["w_qddot"])
        r_adot = reward_action_rate(policy_obs, actions, idx, w["w_adot"])
        r_fa = reward_feet_air_time(feet_air_time, w["w_fa"])
        r_c = reward_undesired_contacts(undesired_contacts, w["w_c"])
        r_g = reward_flat_orientation(policy_obs, idx, w["w_g"])
        r_fc = reward_foot_clearance(foot_clearance, w["w_fc"])
        r_qd = reward_joint_deviation(policy_obs, idx, default_joint_pos, w["w_qd"])

        total = r_vxy + r_wz + r_vz + r_wxy + r_qtau + r_qddot + r_adot + r_fa + r_c + r_g + r_fc + r_qd

        breakdown = {
            "r_vxy": r_vxy, "r_wz": r_wz, "r_vz": r_vz, "r_wxy": r_wxy,
            "r_qtau": r_qtau, "r_qddot": r_qddot, "r_adot": r_adot,
            "r_fa": r_fa, "r_c": r_c, "r_g": r_g, "r_fc": r_fc, "r_qd": r_qd,
        }
        return total, breakdown


class UnitreeG1RewardComputer:
    """
    Computes the total reward for Unitree G1 velocity tracking (Table S6).
    """

    def __init__(self, weights: Dict[str, float]):
        self.w = weights
        self.idx = G1_OBS_IDX
        self.priv_idx = G1_PRIV_IDX

    def compute(
        self,
        policy_obs: torch.Tensor,
        actions: torch.Tensor,
        torques: torch.Tensor,
        q_ddot: torch.Tensor,
        feet_air_time: torch.Tensor,
        undesired_contacts: torch.Tensor,
        foot_clearance: torch.Tensor,
        default_joint_pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        w = self.w
        idx = self.idx

        r_vxy = reward_linear_velocity_xy(policy_obs, idx, w["sigma_vxy"], w["w_vxy"])
        r_wz = reward_angular_velocity_z(policy_obs, idx, w["sigma_wz"], w["w_wz"])
        r_vz = reward_linear_velocity_z(policy_obs, idx, w["w_vz"])
        r_wxy = reward_angular_velocity_xy(policy_obs, idx, w["w_wxy"])
        r_qtau = reward_joint_torque(torques, w["w_qtau"])
        r_qddot = reward_joint_acceleration(q_ddot, w["w_qddot"])
        r_adot = reward_action_rate(policy_obs, actions, idx, w["w_adot"])
        r_fa = reward_feet_air_time(feet_air_time, w["w_fa"])
        r_c = reward_undesired_contacts(undesired_contacts, w["w_c"])
        r_g = reward_flat_orientation(policy_obs, idx, w["w_g"])
        r_fc = reward_foot_clearance(foot_clearance, w["w_fc"])
        r_qd = reward_joint_deviation(policy_obs, idx, default_joint_pos, w["w_qd"])

        total = r_vxy + r_wz + r_vz + r_wxy + r_qtau + r_qddot + r_adot + r_fa + r_c + r_g + r_fc + r_qd

        breakdown = {
            "r_vxy": r_vxy, "r_wz": r_wz, "r_vz": r_vz, "r_wxy": r_wxy,
            "r_qtau": r_qtau, "r_qddot": r_qddot, "r_adot": r_adot,
            "r_fa": r_fa, "r_c": r_c, "r_g": r_g, "r_fc": r_fc, "r_qd": r_qd,
        }
        return total, breakdown


# ---------------------------------------------------------------------------
# Reward from world model observations
# ---------------------------------------------------------------------------

class WorldModelRewardComputer:
    """
    Computes rewards from world model observations during imagination rollouts.

    The world model observation contains: v(3) | omega(3) | g(3) | q | q_dot | tau
    The policy observation contains:      v(3) | omega(3) | g(3) | cmd(3) | q | q_dot | a_prev

    During imagination, we construct the policy obs from the world model obs
    by inserting the velocity command and tracking the previous action.
    """

    def __init__(self, robot: str, weights: Dict[str, float]):
        self.robot = robot
        if robot == "anymal_d":
            self.reward_computer = AnymalDRewardComputer(weights)
            self.wm_obs_dim = 45
            self.action_dim = 12
            self.n_joints = 12
        elif robot == "unitree_g1":
            self.reward_computer = UnitreeG1RewardComputer(weights)
            self.wm_obs_dim = 96
            self.action_dim = 29
            self.n_joints = 29
        else:
            raise ValueError(f"Unknown robot: {robot}")

    def wm_obs_to_policy_obs(
        self,
        wm_obs: torch.Tensor,
        cmd: torch.Tensor,
        a_prev: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert world model observation to policy observation.

        wm_obs: (B, obs_dim)  — v | omega | g | q | q_dot | tau
        cmd:    (B, 3)        — velocity command
        a_prev: (B, action_dim) — previous action

        Returns policy_obs: (B, policy_obs_dim)
        """
        # Extract v, omega, g from world model obs
        v = wm_obs[..., :3]
        omega = wm_obs[..., 3:6]
        g = wm_obs[..., 6:9]
        q = wm_obs[..., 9 : 9 + self.n_joints]
        q_dot = wm_obs[..., 9 + self.n_joints : 9 + 2 * self.n_joints]
        return torch.cat([v, omega, g, cmd, q, q_dot, a_prev], dim=-1)

    def compute_from_wm_obs(
        self,
        wm_obs: torch.Tensor,
        actions: torch.Tensor,
        cmd: torch.Tensor,
        a_prev: torch.Tensor,
        default_joint_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute reward from world model observation.

        wm_obs:  (B, wm_obs_dim)
        actions: (B, action_dim)
        cmd:     (B, 3)
        a_prev:  (B, action_dim)
        default_joint_pos: (n_joints,)

        Returns reward: (B,)
        """
        policy_obs = self.wm_obs_to_policy_obs(wm_obs, cmd, a_prev)
        # Extract torques from world model obs
        tau_start = 9 + 2 * self.n_joints
        torques = wm_obs[..., tau_start : tau_start + self.n_joints]

        # Approximate joint acceleration as zero (not available from single step)
        q_ddot = torch.zeros_like(torques)
        feet_air_time = torch.zeros(wm_obs.size(0), device=wm_obs.device)
        undesired_contacts = torch.zeros(wm_obs.size(0), device=wm_obs.device)
        foot_clearance = torch.zeros(wm_obs.size(0), device=wm_obs.device)

        total, _ = self.reward_computer.compute(
            policy_obs, actions, torques, q_ddot,
            feet_air_time, undesired_contacts, foot_clearance, default_joint_pos,
        )
        return total
