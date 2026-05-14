"""
env_utils.py

Provides the IsaacEnvWrapper class to interface with Isaac Lab simulation environments
for the RWM paper experiments. It abstracts the creation of the environment, extracts
policy observations, world-model observations, and privileged information, and provides
a static method to compute the reward function (Sec. A.1.2) used during imagination
and evaluation.

Requires Isaac Lab (orbit) to be installed. If Isaac Lab is not available, the code
falls back to a simple mock environment that returns random tensors of the correct
shapes for testing purposes.

Configuration is loaded from a dictionary (typically from config.yaml).
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, Any
from gymnasium import spaces
import gymnasium as gym

# Optional Isaac Lab imports – only needed if the real environment is used.
try:
    import isaac_lab  # noqa: F401
    from isaac_lab.envs import DirectRLEnv, DirectMARLEnv
    ISAAC_LAB_AVAILABLE = True
except ImportError:
    ISAAC_LAB_AVAILABLE = False
    # We will define a simple mock environment below for fallback.


# ----------------------------------------------------------------------
# Robot‑specific configuration (extracted from Tables S2‑S6, Sec. A.1)
# ----------------------------------------------------------------------
ROBOT_CONFIGS = {
    "anymal_d": {
        "policy_obs_dim": 48,   # Table S5
        "world_obs_dim": 45,    # Table S2
        "priv_dim": 8,          # Table S3
        "action_dim": 12,       # Table S4
        # Slices for the world‑model observation (assumed order from info['obs_critic'])
        "world_slices": {
            "base_lin_vel":  (0, 3),
            "base_ang_vel":  (3, 6),
            "gravity":       (6, 9),
            "joint_pos":     (9, 21),
            "joint_vel":     (21, 33),
            "joint_torque":  (33, 45),
        },
        # Slices for privileged information (assumed order from info['privileged_info'])
        "priv_slices": {
            "knee_contact": (0, 4),
            "foot_contact": (4, 8),
        },
        # Reward weights (Table S6 left column)
        "reward_weights": {
            "w_v_xy": 1.0,
            "w_omega_z": 0.5,
            "w_v_z": -2.0,
            "w_omega_xy": -0.05,
            "w_q_tau": -2.5e-5,
            "w_q_ddot": -2.5e-7,
            "w_a_rate": -0.01,
            "w_fa": 0.5,
            "w_c": -1.0,
            "w_g": -5.0,
            "w_fc": 0.0,
            "w_qd": 0.0,
        },
        # Default joint positions (neutral standing pose) – not specified in the paper,
        # here set to zero (nominal). If known, replace with robot‑specific values.
        "default_joint_pos": np.zeros(12, dtype=np.float32),
        "command_slice": (9, 12),        # within policy observation
        "joint_vel_dim": 12,             # number of joint velocity components
    },
    "unitree_g1": {
        "policy_obs_dim": 99,
        "world_obs_dim": 96,
        "priv_dim": 30,
        "action_dim": 29,
        "world_slices": {
            "base_lin_vel":  (0, 3),
            "base_ang_vel":  (3, 6),
            "gravity":       (6, 9),
            "joint_pos":     (9, 38),
            "joint_vel":     (38, 67),
            "joint_torque":  (67, 96),
        },
        "priv_slices": {
            "body_contact":  (0, 26),
            "foot_height":   (26, 28),
            "foot_velocity": (28, 30),
        },
        "reward_weights": {
            "w_v_xy": 1.0,
            "w_omega_z": 0.5,
            "w_v_z": -2.0,
            "w_omega_xy": -0.05,
            "w_q_tau": -2.5e-5,
            "w_q_ddot": -2.5e-7,
            "w_a_rate": -0.05,
            "w_fa": 0.0,              # see Table S6: 0.0 for G1
            "w_c": -1.0,
            "w_g": -5.0,
            "w_fc": 1.0,              # foot clearance active
            "w_qd": -1.0,
        },
        "default_joint_pos": np.zeros(29, dtype=np.float32),
        "command_slice": (9, 12),
        "joint_vel_dim": 29,
    },
}


# ----------------------------------------------------------------------
# Reward computation (static helper, usable for imagination & evaluation)
# ----------------------------------------------------------------------
def compute_reward(
    world_obs: torch.Tensor,       # shape (B, world_obs_dim)
    privileged: torch.Tensor,      # shape (B, priv_dim)
    command: torch.Tensor,         # shape (B, 3) : [lin_vel_x, lin_vel_y, ang_vel_z]
    prev_action: torch.Tensor,     # shape (B, action_dim)
    action: torch.Tensor,          # shape (B, action_dim)
    prev_joint_vel: torch.Tensor,  # shape (B, joint_vel_dim)
    joint_vel: torch.Tensor,       # shape (B, joint_vel_dim)
    step_time: float,
    robot: str,
    robot_cfg: Optional[Dict] = None,
) -> torch.Tensor:
    """
    Compute the reward according to Sec. A.1.2 of the paper.
    All terms are weighted and summed. Returns a 1‑D tensor of shape (B,).
    """
    if robot_cfg is None:
        robot_cfg = ROBOT_CONFIGS[robot]

    B = world_obs.shape[0]
    device = world_obs.device
    weights = robot_cfg["reward_weights"]
    slices = robot_cfg["world_slices"]
    priv_slices = robot_cfg["priv_slices"]
    joint_vel_dim = robot_cfg["joint_vel_dim"]

    # Unpack world observation components
    base_lin_vel = world_obs[:, slices["base_lin_vel"][0]:slices["base_lin_vel"][1]]
    base_ang_vel = world_obs[:, slices["base_ang_vel"][0]:slices["base_ang_vel"][1]]
    gravity = world_obs[:, slices["gravity"][0]:slices["gravity"][1]]
    joint_pos = world_obs[:, slices["joint_pos"][0]:slices["joint_pos"][1]]
    joint_torque = world_obs[:, slices["joint_torque"][0]:slices["joint_torque"][1]]
    # joint_vel is already provided directly (the slice [joint_vel] is not used if the caller passes joint_vel separately)
    # but we still might need the full joint_vel slice for checking; use it from world_obs as well?
    # For clarity, we use the passed joint_vel argument, which is the same as the world_obs slice.

    # Command
    cmd_lin_xy = command[:, :2]   # x,y
    cmd_ang_z = command[:, 2:3]   # z

    # Linear velocity tracking (x,y)
    diff_lin_xy = cmd_lin_xy - base_lin_vel[:, :2]
    r_v_xy = weights["w_v_xy"] * torch.exp(-torch.sum(diff_lin_xy ** 2, dim=1) / 0.25 ** 2)

    # Angular velocity tracking (z)
    diff_ang_z = cmd_ang_z.squeeze(-1) - base_ang_vel[:, 2]
    r_omega_z = weights["w_omega_z"] * torch.exp(-diff_ang_z ** 2 / 0.25 ** 2)

    # Vertical velocity penalty
    r_v_z = weights["w_v_z"] * (base_lin_vel[:, 2] ** 2)

    # Roll/pitch rate penalty
    r_omega_xy = weights["w_omega_xy"] * (torch.sum(base_ang_vel[:, :2] ** 2, dim=1))

    # Joint torque penalty
    r_q_tau = weights["w_q_tau"] * torch.sum(joint_torque ** 2, dim=1)

    # Joint acceleration penalty (approximated via finite differences)
    if prev_joint_vel is not None and joint_vel is not None:
        joint_acc = (joint_vel - prev_joint_vel) / step_time
        r_q_ddot = weights["w_q_ddot"] * torch.sum(joint_acc ** 2, dim=1)
    else:
        r_q_ddot = torch.zeros(B, device=device)

    # Action rate penalty
    if prev_action is not None and action is not None:
        r_a_rate = weights["w_a_rate"] * torch.sum((action - prev_action) ** 2, dim=1)
    else:
        r_a_rate = torch.zeros(B, device=device)

    # Feet air time reward (per step: add w_fa * dt for each foot not in contact)
    if robot == "anymal_d":
        # ANYmal D: foot contact is the last 4 entries of privileged (privil_slices["foot_contact"])
        foot_contact = privileged[:, priv_slices["foot_contact"][0]:priv_slices["foot_contact"][1]]
        air_mask = (foot_contact < 0.5).float()  # 0 -> in air, 1 -> on ground; we want air = 1 when foot is off ground
        # actually contact == 0 means foot in air. Use (1 - foot_contact) to get air time bonus.
        air_mask = 1.0 - foot_contact
        # Air time per foot per step
        r_fa = weights["w_fa"] * step_time * air_mask.sum(dim=1)
    elif robot == "unitree_g1":
        # G1 has foot_height but no explicit foot contact; weight is 0.0 anyway
        r_fa = torch.zeros(B, device=device)
    else:
        r_fa = torch.zeros(B, device=device)

    # Undesired contacts
    if robot == "anymal_d":
        # knee contact
        knee_contact = privileged[:, priv_slices["knee_contact"][0]:priv_slices["knee_contact"][1]]
        undesired_count = knee_contact.sum(dim=1)
    elif robot == "unitree_g1":
        # body contact (0:26) – any contact?
        body_contact = privileged[:, priv_slices["body_contact"][0]:priv_slices["body_contact"][1]]
        undesired_count = body_contact.sum(dim=1)
    else:
        undesired_count = torch.zeros(B, device=device)
    r_c = weights["w_c"] * undesired_count

    # Flat orientation – penalise xy components of gravity vector
    g_xy = gravity[:, :2]
    r_g = weights["w_g"] * (torch.sum(g_xy ** 2, dim=1))

    # Foot clearance – only for G1 (weight != 0)
    if robot == "unitree_g1" and weights["w_fc"] != 0.0:
        # foot_height is in privileged (26:28)
        foot_height = privileged[:, priv_slices["foot_height"][0]:priv_slices["foot_height"][1]]
        # Use the minimum of the two foot heights as clearance? The paper uses h_fc, the clearance height of the swing feet.
        # We'll approximate by taking the mean foot height when the foot is in the air? But we don't have swing phase detection.
        # Simplest: take the mean of the two foot heights. This may be rough but matches the spirit.
        h_fc = foot_height.mean(dim=1)
        r_fc = weights["w_fc"] * h_fc
    else:
        r_fc = torch.zeros(B, device=device)

    # Joint deviation
    default_joint_pos = torch.tensor(robot_cfg["default_joint_pos"], device=device, dtype=torch.float32)
    r_qd = weights["w_qd"] * torch.sum(torch.abs(joint_pos - default_joint_pos), dim=1)

    # Total reward
    total_reward = (
        r_v_xy + r_omega_z + r_v_z + r_omega_xy +
        r_q_tau + r_q_ddot + r_a_rate +
        r_fa + r_c + r_g + r_fc + r_qd
    )
    return total_reward


# ----------------------------------------------------------------------
# IsaacEnvWrapper – main interface for simulation interaction
# ----------------------------------------------------------------------
class IsaacEnvWrapper:
    """
    Wraps an Isaac Lab (or fallback mock) environment and provides a uniform
    interface for data collection, training, and evaluation.
    """

    def __init__(self, robot: str, task: str, config: Optional[Dict] = None):
        """
        Args:
            robot: one of "anymal_d" or "unitree_g1".
            task:  task identifier, e.g. "velocity_tracking".
            config: dictionary containing configuration (typically from config.yaml).
                    If None, a minimal default is used.
        """
        self.robot = robot
        self.task = task
        self.config = config if config is not None else {}

        # Load robot‑specific parameters
        if robot not in ROBOT_CONFIGS:
            raise ValueError(f"Unknown robot '{robot}'. Must be one of {list(ROBOT_CONFIGS.keys())}.")
        self.robot_cfg = ROBOT_CONFIGS[robot]

        # Environment creation
        env_id = self._get_env_id(robot, task)

        if ISAAC_LAB_AVAILABLE:
            # Real Isaac Lab environment
            # The environment must be configured to return privileged information.
            # Typical Isaac Lab tasks accept an `env_kwargs` dict; we assume:
            # - flat_obs = False → observation is a dict.
            # - return_privileged_info = True → info dict contains "privileged_info".
            env_kwargs = {
                "flat_obs": False,
                "return_privileged_info": True,
                # Add any other necessary parameters for the specific task.
            }
            self.env = gym.make(env_id, **env_kwargs)
        else:
            # Fallback to mock environment for code testing
            self.env = _MockEnv(self.robot_cfg)

        # Determine the observation space for the policy
        if isinstance(self.env.observation_space, spaces.Dict):
            self.observation_space = self.env.observation_space["policy"]
        else:
            # Fallback: assume flat observation is the policy observation
            self.observation_space = self.env.observation_space

        self.action_space = self.env.action_space

        # Store the step time from config; default 0.02 s if not given
        self.step_time = self.config.get("environment", {}).get("step_time", 0.02)

        # Internal cache for the last world‑obs and privileged info
        self._last_world_obs = None
        self._last_priv = None

    def _get_env_id(self, robot: str, task: str) -> str:
        """
        Map (robot, task) to the Isaac Lab environment identifier.
        These identifiers are specific to the Isaac Lab release used.
        Modify if your installation uses different names.
        """
        mapping = {
            ("anymal_d", "velocity_tracking"): "Isaac-Velocity-Tracking-Anymal-D-v0",
            ("unitree_g1", "velocity_tracking"): "Isaac-Velocity-Tracking-Unitree-G1-v0",
            # Add more tasks as needed
        }
        key = (robot, task)
        if key not in mapping:
            # Fallback for unsupported tasks (e.g., manipulation) – placeholder
            return f"Isaac-{task.replace('_', '-')}-{robot.replace('_', '-')}-v0"
        return mapping[key]

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Any, Dict]:
        """Reset the environment and return the initial policy observation and an info dict
           enriched with 'world_model_obs' and 'privileged'."""
        obs, info = self.env.reset(seed=seed, options=options)
        self._update_cache(obs, info)
        return self._get_policy_obs(obs), self._enriched_info(info)

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict]:
        """Apply an action, return (policy_obs, reward, terminated, truncated, enriched_info)."""
        # Ensure action is numpy array if required
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        obs, reward, term, trunc, info = self.env.step(action)
        done = term or trunc
        self._update_cache(obs, info)
        return self._get_policy_obs(obs), float(reward), term, trunc, self._enriched_info(info)

    def close(self):
        """Close the underlying environment."""
        self.env.close()

    def seed(self, seed: int = None) -> None:
        """Set the random seed of the environment."""
        self.env.reset(seed=seed)

    def get_observation_space(self) -> spaces.Space:
        """Return the policy observation space."""
        return self.observation_space

    def get_action_space(self) -> spaces.Space:
        """Return the action space."""
        return self.action_space

    def get_privileged_info(self) -> Optional[np.ndarray]:
        """Return the most recent privileged information tensor (numpy array)."""
        return self._last_priv

    # ─── Private helpers ──────────────────────────────────────────────

    def _get_policy_obs(self, raw_obs):
        """Extract policy observation from a possibly dict observation."""
        if isinstance(raw_obs, dict):
            return raw_obs["policy"]
        return raw_obs

    def _enriched_info(self, info: Dict) -> Dict:
        """Append world_model_obs and privileged to the info dict."""
        info["world_model_obs"] = self._last_world_obs
        info["privileged"] = self._last_priv
        return info

    def _update_cache(self, obs, info: Dict) -> None:
        """Extract and store the world‑model observation and privileged info."""
        # Retrieve raw critic observation and privileged info from the environment.
        # The exact keys depend on the Isaac Lab task configuration; adjust if necessary.
        if "obs_critic" in info:
            world_obs = np.array(info["obs_critic"], dtype=np.float32)
        else:
            # If the environment does not provide a separate critic observation,
            # we must construct it from the full policy observation and other
            # available signals. For now, assume it is given.
            world_obs = np.zeros(self.robot_cfg["world_obs_dim"], dtype=np.float32)

        if "privileged_info" in info:
            priv = np.array(info["privileged_info"], dtype=np.float32)
        else:
            priv = np.zeros(self.robot_cfg["priv_dim"], dtype=np.float32)

        self._last_world_obs = world_obs
        self._last_priv = priv


# ----------------------------------------------------------------------
# Fallback mock environment (for testing without Isaac Lab)
# ----------------------------------------------------------------------
class _MockEnv:
    """A simple mock environment that returns random observations of the correct dimensions.
       Used only when Isaac Lab is not installed."""
    def __init__(self, robot_cfg: Dict):
        self.robot_cfg = robot_cfg
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(robot_cfg["policy_obs_dim"],), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(robot_cfg["action_dim"],), dtype=np.float32
        )
        self._step_count = 0
        self._ep_len = 1000  # arbitrary

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        self._step_count = 0
        obs = {
            "policy": np.random.randn(self.robot_cfg["policy_obs_dim"]).astype(np.float32)
        }
        info = {
            "obs_critic": np.random.randn(self.robot_cfg["world_obs_dim"]).astype(np.float32),
            "privileged_info": np.random.randn(self.robot_cfg["priv_dim"]).astype(np.float32),
        }
        return obs, info

    def step(self, action):
        self._step_count += 1
        done = self._step_count >= self._ep_len
        obs = {
            "policy": np.random.randn(self.robot_cfg["policy_obs_dim"]).astype(np.float32)
        }
        reward = 0.0
        info = {
            "obs_critic": np.random.randn(self.robot_cfg["world_obs_dim"]).astype(np.float32),
            "privileged_info": np.random.randn(self.robot_cfg["priv_dim"]).astype(np.float32),
        }
        return obs, reward, done, False, info   # term, trunc

    def close(self):
        pass
