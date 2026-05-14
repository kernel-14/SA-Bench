```python
## envs/rewards.py
"""Stateless reward computation module for the Robotic World Model (RWM) project.

This module implements all 12 reward terms from Section A.1.2 of the paper,
plus the ``compute_total_reward`` aggregator. Every function is a pure,
stateless operation on batched PyTorch tensors — no state is maintained
between calls.

Observation indexing follows Tables S2–S5 exactly:
  - World model observation (45-dim ANYmal D / 96-dim G1): contains torques.
  - Policy observation (48-dim ANYmal D / 99-dim G1): contains velocity
    command and last actions instead of torques.

All individual reward functions return a tensor of shape ``[B]`` where ``B``
is the batch size (number of parallel environments). The aggregator sums all
terms element-wise to produce the total reward ``[B]``.

Reward weight keys match ``config.yaml`` exactly:
    w_vxy, w_wz, w_vz, w_wxy, w_qtau, w_qdd, w_adot,
    w_fa, w_c, w_g, w_fc, w_qd

This module has no imports from other project modules — it is a leaf module
that can be tested in complete isolation.
"""

from typing import Dict, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Module-level observation slice constants (Tables S2, S3)
# Stored as (start, end) tuples for use with obs[:, start:end].
# These match the obs_slices and priv_slices fields in config.yaml exactly.
# ---------------------------------------------------------------------------

# World model observation slices (Table S2)
_OBS_SLICES: Dict[str, Dict[str, Tuple[int, int]]] = {
    "anymal_d": {
        "base_lin_vel": (0, 3),
        "base_ang_vel": (3, 6),
        "gravity": (6, 9),
        "joint_pos": (9, 21),
        "joint_vel": (21, 33),
        "joint_torques": (33, 45),
    },
    "unitree_g1": {
        "base_lin_vel": (0, 3),
        "base_ang_vel": (3, 6),
        "gravity": (6, 9),
        "joint_pos": (9, 38),
        "joint_vel": (38, 67),
        "joint_torques": (67, 96),
    },
}

# Privileged information slices (Table S3)
_PRIV_SLICES: Dict[str, Dict[str, Tuple[int, int]]] = {
    "anymal_d": {
        "knee_contact": (0, 4),
        "foot_contact": (4, 8),
        # Undesired contacts for ANYmal D are knee contacts (not foot contacts)
        "undesired_contact": (0, 4),
        # ANYmal D has no foot height/velocity in privileged info (Table S3)
        "foot_height": (4, 8),  # Reuse foot_contact as placeholder; w_fc=0.0
    },
    "unitree_g1": {
        "body_contact": (0, 26),
        "foot_height": (26, 28),
        "foot_velocity": (28, 30),
        # Undesired contacts for G1 are body contacts (not foot contacts)
        "undesired_contact": (0, 26),
    },
}

# Supported robot types — used for validation in all robot-type-dependent fns
_SUPPORTED_ROBOTS: Tuple[str, ...] = ("anymal_d", "unitree_g1")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_robot_type(robot_type: str) -> None:
    """Raise ValueError if robot_type is not a supported robot identifier.

    Args:
        robot_type: Robot identifier string to validate.

    Raises:
        ValueError: If robot_type is not in ``_SUPPORTED_ROBOTS``.
    """
    if robot_type not in _SUPPORTED_ROBOTS:
        raise ValueError(
            f"Unsupported robot_type '{robot_type}'. "
            f"Expected one of: {_SUPPORTED_ROBOTS}."
        )


def _get_obs_slice(robot_type: str, field: str) -> Tuple[int, int]:
    """Return the (start, end) index tuple for a world model observation field.

    Args:
        robot_type: Robot identifier. Must be "anymal_d" or "unitree_g1".
        field: Observation field name. Must be a key in ``_OBS_SLICES``.

    Returns:
        A ``(start, end)`` tuple for use as ``obs[:, start:end]``.

    Raises:
        ValueError: If robot_type or field is not recognized.
    """
    _validate_robot_type(robot_type)
    if field not in _OBS_SLICES[robot_type]:
        raise ValueError(
            f"Unknown observation field '{field}' for robot '{robot_type}'. "
            f"Available fields: {list(_OBS_SLICES[robot_type].keys())}."
        )
    return _OBS_SLICES[robot_type][field]


def _get_priv_slice(robot_type: str, field: str) -> Tuple[int, int]:
    """Return the (start, end) index tuple for a privileged information field.

    Args:
        robot_type: Robot identifier. Must be "anymal_d" or "unitree_g1".
        field: Privileged info field name. Must be a key in ``_PRIV_SLICES``.

    Returns:
        A ``(start, end)`` tuple for use as ``priv[:, start:end]``.

    Raises:
        ValueError: If robot_type or field is not recognized.
    """
    _validate_robot_type(robot_type)
    if field not in _PRIV_SLICES[robot_type]:
        raise ValueError(
            f"Unknown privileged info field '{field}' for robot '{robot_type}'. "
            f"Available fields: {list(_PRIV_SLICES[robot_type].keys())}."
        )
    return _PRIV_SLICES[robot_type][field]


# ---------------------------------------------------------------------------
# Individual reward functions
# All functions return Tensor of shape [B] (one scalar reward per environment).
# ---------------------------------------------------------------------------


def linear_vel_tracking(
    obs: Tensor,
    command: Tensor,
    w: float,
    sigma: float = 0.25,
) -> Tensor:
    """Compute the linear velocity tracking reward (x, y axes).

    Paper formula (Section A.1.2):
        r_vxy = w_vxy * exp(-||c_xy - v_xy||_2^2 / sigma_vxy^2)

    Encourages the robot to match the commanded x-y base linear velocity.
    The exponential kernel provides a smooth, bounded reward in [0, w_vxy].

    The base linear velocity is always at ``obs[:, 0:3]`` for both robots
    (Table S2), so this function is robot-type agnostic.

    Args:
        obs: World model observation tensor of shape ``[B, obs_dim]``.
            ``obs[:, 0:2]`` contains the x and y base linear velocities.
        command: Velocity command tensor of shape ``[B, 3]``.
            ``command[:, 0:2]`` contains the commanded x and y velocities.
            ``command[:, 2]`` contains the commanded yaw rate (unused here).
        w: Reward weight ``w_vxy``. From config: ``anymal_d.reward_weights.w_vxy=1.0``,
            ``unitree_g1.reward_weights.w_vxy=1.0``.
        sigma: Temperature factor controlling the width of the exponential
            kernel. From config: ``anymal_d.sigma_vxy=0.25``,
            ``unitree_g1.sigma_vxy=0.25``. Default: 0.25.

    Returns:
        Reward tensor of shape ``[B]``. Values in ``[0, w]`` when ``w > 0``.
    """
    # Extract x, y components of base linear velocity: obs[:, 0:2]
    v_xy: Tensor = obs[:, 0:2]  # shape [B, 2]
    # Extract x, y velocity commands: command[:, 0:2]
    c_xy: Tensor = command[:, 0:2]  # shape [B, 2]

    # Squared L2 norm of the tracking error: ||c_xy - v_xy||_2^2
    diff_sq: Tensor = ((c_xy - v_xy) ** 2).sum(dim=-1)  # shape [B]

    # Exponential kernel: exp(-error^2 / sigma^2)
    reward: Tensor = w * torch.exp(-diff_sq / (sigma ** 2))  # shape [B]
    return reward


def angular_vel_tracking(
    obs: Tensor,
    command: Tensor,
    w: float,
    sigma: float = 0.25,
) -> Tensor:
    """Compute the angular velocity tracking reward (yaw axis).

    Paper formula (Section A.1.2):
        r_wz = w_wz * exp(-||c_z - omega_z||_2^2 / sigma_wz^2)

    Encourages the robot to match the commanded yaw rate. The base angular
    velocity is at ``obs[:, 3:6]`` for both robots (Table S2), and the yaw
    component is the last element (index 5).

    Args:
        obs: World model observation tensor of shape ``[B, obs_dim]``.
            ``obs[:, 5:6]`` contains the z (yaw) base angular velocity.
        command: Velocity command tensor of shape ``[B, 3]``.
            ``command[:, 2:3]`` contains the commanded yaw rate.
        w: Reward weight ``w_wz``. From config: ``w_wz=0.5`` for both robots.
        sigma: Temperature factor. From config: ``sigma_wz=0.25``.
            Default: 0.25.

    Returns:
        Reward tensor of shape ``[B]``. Values in ``[0, w]`` when ``w > 0``.
    """
    # Extract z component of base angular velocity: obs[:, 5:6]
    # base_ang_vel is at obs[:, 3:6]; yaw (z) is the last element at index 5
    omega_z: Tensor = obs[:, 5:6]  # shape [B, 1]
    # Extract yaw rate command: command[:, 2:3]
    c_z: Tensor = command[:, 2:3]  # shape [B, 1]

    # Squared L2 norm of the yaw tracking error
    diff_sq: Tensor = ((c_z - omega_z) ** 2).sum(dim=-1)  # shape [B]

    reward: Tensor = w * torch.exp(-diff_sq / (sigma ** 2))  # shape [B]
    return reward


def linear_vel_z_penalty(
    obs: Tensor,
    w: float,
) -> Tensor:
    """Compute the vertical linear velocity penalty.

    Paper formula (Section A.1.2):
        r_vz = w_vz * ||v_z||_2^2

    Penalizes vertical base motion to encourage stable locomotion. The z
    component of base linear velocity is at index 2 of ``obs`` (Table S2).

    Args:
        obs: World model observation tensor of shape ``[B, obs_dim]``.
            ``obs[:, 2:3]`` contains the z base linear velocity.
        w: Reward weight ``w_vz``. From config: ``w_vz=-2.0`` for both robots.
            Negative weight makes this a penalty.

    Returns:
        Reward tensor of shape ``[B]``. Values <= 0 when ``w < 0``.
    """
    # Extract z component of base linear velocity: obs[:, 2:3]
    v_z: Tensor = obs[:, 2:3]  # shape [B, 1]

    # Squared L2 norm: ||v_z||_2^2
    reward: Tensor = w * (v_z ** 2).sum(dim=-1)  # shape [B]
    return reward


def angular_vel_xy_penalty(
    obs: Tensor,
    w: float,
) -> Tensor:
    """Compute the roll and pitch angular velocity penalty.

    Paper formula (Section A.1.2):
        r_wxy = w_wxy * ||omega_xy||_2^2

    Penalizes roll and pitch angular velocities to encourage stable upright
    posture. The x and y components of base angular velocity are at
    ``obs[:, 3:5]`` (first two dims of ``base_ang_vel`` at ``obs[:, 3:6]``).

    Args:
        obs: World model observation tensor of shape ``[B, obs_dim]``.
            ``obs[:, 3:5]`` contains the x and y base angular velocities.
        w: Reward weight ``w_wxy``. From config: ``w_wxy=-0.05`` for both
            robots. Negative weight makes this a penalty.

    Returns:
        Reward tensor of shape ``[B]``. Values <= 0 when ``w < 0``.
    """
    # Extract x, y components of base angular velocity: obs[:, 3:5]
    omega_xy: Tensor = obs[:, 3:5]  # shape [B, 2]

    # Squared L2 norm: ||omega_xy||_2^2
    reward: Tensor = w * (omega_xy ** 2).sum(dim=-1)  # shape [B]
    return reward


def joint_torque_penalty(
    obs: Tensor,
    w: float,
    robot_type: str = "anymal_d",
) -> Tensor:
    """Compute the joint torque penalty.

    Paper formula (Section A.1.2):
        r_qtau = w_qtau * ||tau||_2^2

    Penalizes large joint torques to encourage energy-efficient motion.
    Joint torques are in the **world model observation** (Table S2), NOT
    in the policy observation or privileged info. The caller must pass the
    world model observation (45-dim for ANYmal D, 96-dim for G1).

    Args:
        obs: World model observation tensor of shape ``[B, obs_dim]``.
            For ANYmal D: ``obs[:, 33:45]`` contains joint torques (12-dim).
            For Unitree G1: ``obs[:, 67:96]`` contains joint torques (29-dim).
        w: Reward weight ``w_qtau``. From config: ``w_qtau=-2.5e-5`` for
            both robots. Very small negative weight.
        robot_type: Robot identifier. Must be "anymal_d" or "unitree_g1".
            Default: "anymal_d".

    Returns:
        Reward tensor of shape ``[B]``. Values <= 0 when ``w < 0``.

    Raises:
        ValueError: If robot_type is not supported.
    """
    start, end = _get_obs_slice(robot_type, "joint_torques")
    tau: Tensor = obs[:, start:end]  # shape [B, n_joints]

    # Squared L2 norm: ||tau||_2^2
    reward: Tensor = w * (tau ** 2).sum(dim=-1)  # shape [B]
    return reward


def joint_accel_penalty(
    obs: Tensor,
    next_obs: Tensor,
    w: float,
    robot_type: str = "anymal_d",
    dt: float = 0.02,
) -> Tensor:
    """Compute the joint acceleration penalty via finite differencing.

    Paper formula (Section A.1.2):
        r_qdd = w_qdd * ||q_ddot||_2^2

    Joint acceleration is not directly observed — it is estimated by finite
    differencing consecutive joint velocity observations:
        q_ddot ≈ (q_dot_next - q_dot_curr) / dt

    The ``dt=0.02`` value corresponds to the 50 Hz control frequency from
    ``config.yaml`` (``simulation.dt: 0.02``).

    Args:
        obs: Current world model observation of shape ``[B, obs_dim]``.
            Contains current joint velocities.
        next_obs: Next world model observation of shape ``[B, obs_dim]``.
            Contains next-step joint velocities. In imagination rollouts,
            this is the world model's predicted next observation.
        w: Reward weight ``w_qdd``. From config: ``w_qdd=-2.5e-7`` for both
            robots. Very small negative weight.
        robot_type: Robot identifier. Must be "anymal_d" or "unitree_g1".
            Default: "anymal_d".
        dt: Time step in seconds for finite differencing. From config:
            ``simulation.dt: 0.02``. Default: 0.02.

    Returns:
        Reward tensor of shape ``[B]``. Values <= 0 when ``w < 0``.

    Raises:
        ValueError: If robot_type is not supported or dt <= 0.
    """
    if dt <= 0.0:
        raise ValueError(
            f"Time step dt must be positive, got dt={dt}. "
            "Check simulation.dt in config.yaml."
        )

    start, end = _get_obs_slice(robot_type, "joint_vel")
    q_dot_curr: Tensor = obs[:, start:end]       # shape [B, n_joints]
    q_dot_next: Tensor = next_obs[:, start:end]  # shape [B, n_joints]

    # Finite difference approximation of joint acceleration
    q_ddot: Tensor = (q_dot_next - q_dot_curr) / dt  # shape [B, n_joints]

    # Squared L2 norm: ||q_ddot||_2^2
    reward: Tensor = w * (q_ddot ** 2).sum(dim=-1)  # shape [B]
    return reward


def action_rate_penalty(
    action: Tensor,
    prev_action: Tensor,
    w: float,
) -> Tensor:
    """Compute the action rate penalty (smoothness regularization).

    Paper formula (Section A.1.2):
        r_adot = w_adot * ||a' - a||_2^2

    Penalizes large changes between consecutive actions to encourage smooth
    joint position target trajectories. This is robot-type agnostic since
    it only depends on the action tensors.

    Args:
        action: Current action tensor of shape ``[B, action_dim]``.
            Joint position targets for the current step.
        prev_action: Previous action tensor of shape ``[B, action_dim]``.
            Joint position targets from the previous step.
        w: Reward weight ``w_adot``. From config:
            ANYmal D: ``w_adot=-0.01``, Unitree G1: ``w_adot=-0.05``.
            Negative weight makes this a penalty.

    Returns:
        Reward tensor of shape ``[B]``. Values <= 0 when ``w < 0``.
    """
    # Squared L2 norm of action difference: ||a' - a||_2^2
    # Note: paper uses a' for previous and a for current (Section A.1.2)
    diff: Tensor = prev_action - action  # shape [B, action_dim]
    reward: Tensor = w * (diff ** 2).sum(dim=-1)  # shape [B]
    return reward


def feet_air_time(
    t_fa: Tensor,
    w: float,
) -> Tensor:
    """Compute the feet air time reward.

    Paper formula (Section A.1.2):
        r_fa = w_fa * t_fa

    Rewards the robot for keeping its feet in the air (encouraging a
    dynamic gait). The air time ``t_fa`` is a pre-computed scalar per
    environment, representing the sum of air time across all feet.

    This function is stateless — the environment wrapper is responsible
    for tracking contact history and computing ``t_fa`` before calling
    this function. In imagination rollouts where contact tracking is not
    available, pass ``t_fa = torch.zeros(B)`` (acceptable since
    ``w_fa=0.0`` for Unitree G1 and the gait emerges from other terms).

    Args:
        t_fa: Pre-computed feet air time tensor of shape ``[B]``.
            Sum of air time (in seconds) across all feet for each
            environment in the batch. Computed by the environment wrapper
            from contact history.
        w: Reward weight ``w_fa``. From config:
            ANYmal D: ``w_fa=0.5``, Unitree G1: ``w_fa=0.0``.

    Returns:
        Reward tensor of shape ``[B]``. Values >= 0 when ``w > 0``.
    """
    # Direct linear reward: r_fa = w_fa * t_fa
    reward: Tensor = w * t_fa  # shape [B]
    return reward


def undesired_contacts(
    priv: Tensor,
    w: float,
    robot_type: str = "anymal_d",
) -> Tensor:
    """Compute the undesired contact penalty.

    Paper formula (Section A.1.2):
        r_c = w_c * c_u

    Penalizes contacts on body parts that should not touch the ground:
      - ANYmal D: knee contacts (``priv[:, 0:4]``, 4 binary flags)
      - Unitree G1: body contacts (``priv[:, 0:26]``, 26 binary flags)

    Foot contacts are desired and are NOT penalized here.

    Args:
        priv: Privileged information tensor of shape ``[B, priv_dim]``.
            Contains binary contact flags (0.0 = no contact, 1.0 = contact).
            For ANYmal D: priv_dim=8 (Table S3).
            For Unitree G1: priv_dim=30 (Table S3).
        w: Reward weight ``w_c``. From config: ``w_c=-1.0`` for both robots.
            Negative weight makes this a penalty.
        robot_type: Robot identifier. Must be "anymal_d" or "unitree_g1".
            Default: "anymal_d".

    Returns:
        Reward tensor of shape ``[B]``. Values <= 0 when ``w < 0``.

    Raises:
        ValueError: If robot_type is not supported.
    """
    start, end = _get_priv_slice(robot_type, "undesired_contact")
    contact_flags: Tensor = priv[:, start:end]  # shape [B, n_contacts]

    # Count undesired contacts: sum of binary flags across contact points
    c_u: Tensor = contact_flags.sum(dim=-1)  # shape [B]

    reward: Tensor = w * c_u  # shape [B]
    return reward


def flat_orientation(
    obs: Tensor,
    w: float,
) -> Tensor:
    """Compute the flat orientation penalty.

    Paper formula (Section A.1.2):
        r_g = w_g * g_xy^2

    Penalizes tilting by penalizing the x and y components of the projected
    gravity vector. When the robot is perfectly upright, gravity projects to
    ``[0, 0, -1]`` in the robot frame, so ``g_xy ≈ [0, 0]`` and the penalty
    is near zero.

    The gravity vector is at ``obs[:, 6:9]`` for both robots (Table S2),
    so this function is robot-type agnostic.

    Args:
        obs: World model observation tensor of shape ``[B, obs_dim]``.
            ``obs[:, 6:8]`` contains the x and y components of the projected
            gravity vector in the robot frame.
        w: Reward weight ``w_g``. From config: ``w_g=-5.0`` for both robots.
            Large negative weight strongly penalizes tilting.

    Returns:
        Reward tensor of shape ``[B]``. Values <= 0 when ``w < 0``.
    """
    # Extract x, y components of projected gravity: obs[:, 6:8]
    # gravity is at obs[:, 6:9]; g_xy is the first two components
    g_xy: Tensor = obs[:, 6:8]  # shape [B, 2]

    # Squared penalty: w_g * g_xy^2 (element-wise square, then sum)
    reward: Tensor = w * (g_xy ** 2).sum(dim=-1)  # shape [B]
    return reward


def foot_clearance(
    priv: Tensor,
    w: float,
    robot_type: str = "anymal_d",
) -> Tensor:
    """Compute the foot clearance reward.

    Paper formula (Section A.1.2):
        r_fc = w_fc * h_fc

    Rewards swing foot clearance height to encourage the robot to lift its
    feet during locomotion. For ANYmal D, ``w_fc=0.0`` so this term is zero.
    For Unitree G1, foot height is in ``priv[:, 26:28]`` (Table S3).

    Args:
        priv: Privileged information tensor of shape ``[B, priv_dim]``.
            For Unitree G1: ``priv[:, 26:28]`` contains foot heights (2-dim).
            For ANYmal D: ``w_fc=0.0`` so the priv content is irrelevant.
        w: Reward weight ``w_fc``. From config:
            ANYmal D: ``w_fc=0.0``, Unitree G1: ``w_fc=1.0``.
        robot_type: Robot identifier. Must be "anymal_d" or "unitree_g1".
            Default: "anymal_d".

    Returns:
        Reward tensor of shape ``[B]``. Returns zeros for ANYmal D since
        ``w_fc=0.0``. Values >= 0 for G1 when ``w > 0``.

    Raises:
        ValueError: If robot_type is not supported.
    """
    _validate_robot_type(robot_type)

    # Short-circuit for ANYmal D: w_fc=0.0 means this term is always zero.
    # Avoids unnecessary tensor operations and potential index errors.
    if w == 0.0:
        batch_size: int = priv.shape[0]
        return torch.zeros(batch_size, dtype=priv.dtype, device=priv.device)

    start, end = _get_priv_slice(robot_type, "foot_height")
    h_fc: Tensor = priv[:, start:end]  # shape [B, n_feet]

    # Sum foot clearance heights across all feet
    reward: Tensor = w * h_fc.sum(dim=-1)  # shape [B]
    return reward


def joint_deviation(
    obs: Tensor,
    default_pos: Tensor,
    w: float,
    robot_type: str = "anymal_d",
) -> Tensor:
    """Compute the joint deviation penalty from default position.

    Paper formula (Section A.1.2):
        r_qd = w_qd * ||q - q_0||_1

    Penalizes deviation from the default (nominal) joint configuration using
    the L1 norm. For ANYmal D, ``w_qd=0.0`` so this term is zero. For
    Unitree G1, ``w_qd=-1.0`` encourages the robot to stay near its default
    joint configuration.

    Args:
        obs: World model observation tensor of shape ``[B, obs_dim]``.
            Contains current joint positions at the robot-specific slice.
            For ANYmal D: ``obs[:, 9:21]`` (12-dim joint positions).
            For Unitree G1: ``obs[:, 9:38]`` (29-dim joint positions).
        default_pos: Default joint position tensor. Shape ``[n_joints]`` or
            ``[B, n_joints]`` — PyTorch broadcasting handles both cases.
            Should match the joint position dimension for the robot.
        w: Reward weight ``w_qd``. From config:
            ANYmal D: ``w_qd=0.0``, Unitree G1: ``w_qd=-1.0``.
        robot_type: Robot identifier. Must be "anymal_d" or "unitree_g1".
            Default: "anymal_d".

    Returns:
        Reward tensor of shape ``[B]``. Returns zeros for ANYmal D since
        ``w_qd=0.0``. Values <= 0 for G1 when ``w < 0``.

    Raises:
        ValueError: If robot_type is not supported.
    """
    _validate_robot_type(robot_type)

    # Short-circuit for ANYmal D: w_qd=0.0 means this term is always zero.
    if w == 0.0:
        batch_size: int = obs.shape[0]
        return torch.zeros(batch_size, dtype=obs.dtype, device=obs.device)

    start, end = _get_obs_slice(robot_type, "joint_pos")
    q: Tensor = obs[:, start:end]  # shape [B, n_joints]

    # L1 norm: ||q - q_0||_1 = sum of absolute deviations
    # default_pos broadcasts from [n_joints] to [B, n_joints] automatically
    deviation: Tensor = (q - default_pos).abs().sum(dim=-1)  # shape [B]

    reward: Tensor = w * deviation  # shape [B]
    return reward


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def compute_total_reward(
    obs: Tensor,
    action: Tensor,
    next_obs: Tensor,
    priv: Tensor,
    command: Tensor,
    prev_action: Tensor,
    default_pos: Tensor,
    t_fa: Tensor,
    weights: Dict[str, float],
    robot_type: str = "anymal_d",
    dt: float = 0.02,
) -> Tensor:
    """Compute the total reward by summing all 12 reward terms.

    This is the main entry point for reward computation, called by:
      - Environment wrappers (``anymal_env.py``, ``unitree_g1_env.py``,
        ``mock_env.py``) during real environment interaction.
      - ``MBPOPPOTrainer.imagine_trajectories`` during imagination rollouts.

    All 12 reward terms from Section A.1.2 are computed and summed
    element-wise. The weight dict keys match ``config.yaml`` exactly.

    **Important observation type note:**
    ``obs`` and ``next_obs`` must be the **world model observation** (45-dim
    for ANYmal D, 96-dim for G1) which contains joint torques. The policy
    observation (48-dim / 99-dim) does NOT contain torques and cannot be
    used here. The MBPO-PPO trainer must maintain the world model observation
    separately from the policy observation for reward computation.

    Args:
        obs: Current world model observation of shape ``[B, obs_dim]``.
            Must contain joint torques (Table S2). Used for all reward terms
            except joint acceleration (which also needs next_obs).
        action: Current action (joint position targets) of shape
            ``[B, action_dim]``. Used for action rate penalty.
        next_obs: Next world model observation of shape ``[B, obs_dim]``.
            Used for joint acceleration finite differencing. In real
            environments, this is the observation after stepping. In
            imagination, this is the world model's predicted next observation.
        priv: Privileged information of shape ``[B, priv_dim]``.
            