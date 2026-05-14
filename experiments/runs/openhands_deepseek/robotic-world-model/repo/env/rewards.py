"""Reward computation for velocity tracking tasks (Sec A.1.2).

Implements all reward terms from equations in the paper:
- Linear velocity tracking (xy)      : r_vxy = w_vxy * exp(-||c_xy - v_xy||^2 / sigma_vxy^2)
- Angular velocity tracking (z)      : r_omega_z = w_omega_z * exp(-||c_z - omega_z||^2 / sigma_omega_z^2)
- Linear velocity z penalty          : r_vz = w_vz * v_z^2
- Angular velocity xy penalty        : r_omega_xy = w_omega_xy * ||omega_xy||^2
- Joint torque penalty               : r_q_tau = w_q_tau * ||tau||^2
- Joint acceleration penalty         : r_q_ddot = w_q_ddot * ||q_ddot||^2
- Action rate penalty                : r_a_dot = w_a_dot * ||a' - a||^2
- Feet air time                      : r_f_air = w_f_air * t_f_air
- Undesired contacts                 : r_c = w_c * c_u
- Flat orientation                   : r_g = w_g * g_xy^2
- Foot clearance                     : r_fc = w_fc * h_fc
- Joint deviation                    : r_qd = w_qd * ||q - q_0||_1

Temperature factors: sigma_vxy = 0.25, sigma_omega_z = 0.25
"""

from typing import Dict, Optional
import torch

from config import RewardWeights


class RewardComputer:
    """Computes velocity tracking rewards from imagined/predicted observations."""

    def __init__(
        self,
        weights: RewardWeights,
        robot_spec,
        default_joint_pos: Optional[torch.Tensor] = None,
    ):
        self.weights = weights
        self.robot_spec = robot_spec
        self.default_joint_pos = default_joint_pos

        self.sigma_vxy = weights.sigma_vxy
        self.sigma_omega_z = weights.sigma_omega_z

    def compute_rewards(
        self,
        observations: torch.Tensor,
        privileged: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        prev_actions: Optional[torch.Tensor] = None,
        feet_in_air_time: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute total reward from observation components.

        Args:
            observations: (..., obs_dim) — predicted/imagined observations
            privileged: (..., priv_dim) — predicted privileged info (contacts, etc.)
            actions: (..., action_dim) — current actions
            prev_actions: (..., action_dim) — previous actions (for action rate)
            feet_in_air_time: (...) — accumulated air time per foot

        Returns:
            rewards: (...) — total reward per environment
        """
        spec = self.robot_spec
        w = self.weights

        # Extract observation components
        v = observations[..., spec.obs_base_lin_vel[0]:spec.obs_base_lin_vel[1]]  # (..., 3)
        omega = observations[..., spec.obs_base_ang_vel[0]:spec.obs_base_ang_vel[1]]  # (..., 3)
        g = observations[..., spec.obs_projected_gravity[0]:spec.obs_projected_gravity[1]]  # (..., 3)

        joint_pos_start = spec.obs_joint_pos_start
        num_joints = spec.num_joints
        q = observations[..., joint_pos_start:joint_pos_start + num_joints]  # joint positions

        r_total = torch.zeros_like(v[..., 0])

        # --- Linear velocity tracking (xy) ---
        # Command velocity c_xy would come from the task, here we assume v[..., :2] is tracking some command
        # For the reward computation from world model predictions:
        # We compute the velocity tracking error using the actual velocity components
        # In the full implementation, the velocity command c is part of the policy observation
        # Here we compute based on what the world model predicts
        # The command c_xy should be provided separately
        v_xy = v[..., :2]  # (..., 2)
        c_xy = torch.zeros_like(v_xy)  # placeholder — commands come from task
        r_vxy = w.w_vxy * torch.exp(-((c_xy - v_xy) ** 2).sum(dim=-1) / (self.sigma_vxy ** 2))
        r_total = r_total + r_vxy

        # --- Angular velocity tracking (z) ---
        omega_z = omega[..., 2]  # (...,)
        c_z = torch.zeros_like(omega_z)  # placeholder
        r_omega_z = w.w_omega_z * torch.exp(-((c_z - omega_z) ** 2) / (self.sigma_omega_z ** 2))
        r_total = r_total + r_omega_z

        # --- Linear velocity z penalty ---
        v_z = v[..., 2]
        r_vz = w.w_vz * (v_z ** 2)
        r_total = r_total + r_vz

        # --- Angular velocity xy penalty ---
        omega_xy = omega[..., :2]
        r_omega_xy = w.w_omega_xy * (omega_xy ** 2).sum(dim=-1)
        r_total = r_total + r_omega_xy

        # --- Joint torque penalty ---
        if spec.obs_joint_tau_start is not None:
            tau_start = spec.obs_joint_tau_start
            tau = observations[..., tau_start:tau_start + num_joints]
            r_tau = w.w_q_tau * (tau ** 2).sum(dim=-1)
            r_total = r_total + r_tau

        # --- Joint acceleration penalty ---
        if spec.obs_joint_vel_start is not None:
            vel_start = spec.obs_joint_vel_start
            q_dot = observations[..., vel_start:vel_start + num_joints]
            # Acceleration approximated as velocity magnitude
            r_q_ddot = w.w_q_ddot * (q_dot ** 2).sum(dim=-1)
            r_total = r_total + r_q_ddot

        # --- Action rate penalty ---
        if actions is not None and prev_actions is not None:
            r_a_dot = w.w_a_dot * ((actions - prev_actions) ** 2).sum(dim=-1)
            r_total = r_total + r_a_dot

        # --- Feet air time ---
        if feet_in_air_time is not None:
            r_f_air = w.w_f_air * feet_in_air_time
            r_total = r_total + r_f_air

        # --- Undesired contacts ---
        if privileged is not None:
            # For ANYmal D: first 4 entries are knee contact, next 4 are foot contact
            # Undesired contacts = knee contacts
            num_feet = spec.num_feet
            if privileged.shape[-1] >= num_feet:
                knee_contact = privileged[..., :num_feet]
                c_u = knee_contact.sum(dim=-1)
                r_c = w.w_undesired_contact * c_u
                r_total = r_total + r_c

        # --- Flat orientation ---
        g_xy = g[..., :2]
        r_g = w.w_flat_orientation * (g_xy ** 2).sum(dim=-1)
        r_total = r_total + r_g

        # --- Joint deviation ---
        if self.default_joint_pos is not None:
            r_qd = w.w_joint_deviation * (q - self.default_joint_pos).abs().sum(dim=-1)
            r_total = r_total + r_qd

        # --- Foot clearance ---
        # Requires height information, not directly from this observation space
        # Placeholder: set to 0 unless additional info provided
        r_fc = w.w_foot_clearance * 0.0
        r_total = r_total + r_fc

        return r_total

    def compute_rewards_with_commands(
        self,
        observations: torch.Tensor,
        commands: torch.Tensor,
        privileged: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        prev_actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute rewards with explicit velocity commands.

        Args:
            observations: (..., obs_dim)
            commands: (..., 3) — [cmd_vx, cmd_vy, cmd_omega_z]
            privileged: (..., priv_dim)
            actions: (..., action_dim)
            prev_actions: (..., action_dim)

        Returns:
            rewards: (...)
        """
        spec = self.robot_spec
        w = self.weights

        v = observations[..., spec.obs_base_lin_vel[0]:spec.obs_base_lin_vel[1]]
        omega = observations[..., spec.obs_base_ang_vel[0]:spec.obs_base_ang_vel[1]]
        g = observations[..., spec.obs_projected_gravity[0]:spec.obs_projected_gravity[1]]

        num_joints = spec.num_joints
        q = observations[..., spec.obs_joint_pos_start:spec.obs_joint_pos_start + num_joints]

        r_total = torch.zeros_like(v[..., 0])

        # Linear velocity tracking
        c_xy = commands[..., :2]
        v_xy = v[..., :2]
        r_vxy = w.w_vxy * torch.exp(-((c_xy - v_xy) ** 2).sum(dim=-1) / (self.sigma_vxy ** 2))
        r_total = r_total + r_vxy

        # Angular velocity tracking
        c_z = commands[..., 2]
        omega_z = omega[..., 2]
        r_omega_z = w.w_omega_z * torch.exp(-((c_z - omega_z) ** 2) / (self.sigma_omega_z ** 2))
        r_total = r_total + r_omega_z

        # Rest are same as base
        v_z = v[..., 2]
        r_total = r_total + w.w_vz * (v_z ** 2)

        omega_xy = omega[..., :2]
        r_total = r_total + w.w_omega_xy * (omega_xy ** 2).sum(dim=-1)

        if spec.obs_joint_tau_start is not None:
            tau_start = spec.obs_joint_tau_start
            tau = observations[..., tau_start:tau_start + num_joints]
            r_total = r_total + w.w_q_tau * (tau ** 2).sum(dim=-1)

        if spec.obs_joint_vel_start is not None:
            vel_start = spec.obs_joint_vel_start
            q_dot = observations[..., vel_start:vel_start + num_joints]
            r_total = r_total + w.w_q_ddot * (q_dot ** 2).sum(dim=-1)

        if actions is not None and prev_actions is not None:
            r_total = r_total + w.w_a_dot * ((actions - prev_actions) ** 2).sum(dim=-1)

        if privileged is not None:
            num_feet = spec.num_feet
            if privileged.shape[-1] >= 2 * num_feet:
                knee_contact = privileged[..., :num_feet]
                c_u = knee_contact.sum(dim=-1)
                r_total = r_total + w.w_undesired_contact * c_u

        g_xy = g[..., :2]
        r_total = r_total + w.w_flat_orientation * (g_xy ** 2).sum(dim=-1)

        if self.default_joint_pos is not None:
            r_total = r_total + w.w_joint_deviation * (q - self.default_joint_pos).abs().sum(dim=-1)

        return r_total
