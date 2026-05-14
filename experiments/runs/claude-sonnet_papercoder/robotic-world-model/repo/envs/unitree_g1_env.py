## Code: envs/unitree_g1_env.py

```python
## envs/unitree_g1_env.py
"""Isaac Lab wrapper for the Unitree G1 humanoid robot environment.

This module provides ``UnitreeG1Env``, a concrete implementation of ``BaseEnv``
that wraps Isaac Lab's Unitree G1 simulation. It constructs the 96-dim world
model observation (Table S2), 30-dim privileged information (Table S3), and
manages the 29-dim joint position target action space (Table S4).

If Isaac Lab is not installed, ``__init__`` raises ``ImportError`` with a
clear installation message. Use ``MockEnv`` for pipeline testing without
Isaac Lab.

Observation construction follows Table S2 exactly:
    obs[0:3]   = base linear velocity in robot body frame
    obs[3:6]   = base angular velocity in robot body frame
    obs[6:9]   = projected gravity vector in robot body frame
    obs[9:38]  = joint positions (29 joints, radians)
    obs[38:67] = joint velocities (29 joints, rad/s)
    obs[67:96] = applied joint torques (29 joints, N·m)

Privileged information follows Table S3:
    priv[0:26]  = body contact flags (26 body links, binary float)
    priv[26:28] = foot heights above terrain (2 feet, meters, continuous)
    priv[28:30] = foot velocities (2 feet, m/s, continuous)

Key differences from ANYmal D:
    - 29 joints vs 12 joints
    - Heterogeneous priv vector: binary contacts (0:26) + continuous (26:30)
    - w_adot=-0.05 (stricter smoothness), w_fa=0.0, w_fc=1.0, w_qd=-1.0

Control frequency: 50 Hz (dt=0.02s), matching Section 4.1 of the paper.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from envs.base_env import BaseEnv

# ---------------------------------------------------------------------------
# Optional Isaac Lab import — graceful fallback for environments without
# Isaac Sim installed. MockEnv provides a drop-in replacement for testing.
# ---------------------------------------------------------------------------

ISAAC_LAB_AVAILABLE: bool = False

try:
    from isaaclab.app import AppLauncher  # noqa: F401
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    import isaaclab.sim as sim_utils

    ISAAC_LAB_AVAILABLE = True
except ImportError:
    SimulationContext = None  # type: ignore[assignment,misc]
    Articulation = None       # type: ignore[assignment,misc]
    ContactSensor = None      # type: ignore[assignment,misc]
    ISAAC_LAB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Unitree G1 joint and link name constants
# Standard naming convention from the Unitree G1 USD asset in Isaac Lab.
# The G1 has 29 DOF covering legs, arms, and torso.
# ---------------------------------------------------------------------------

# 29 joints in order (standard Unitree G1 joint ordering):
# Left leg (6): left_hip_pitch, left_hip_roll, left_hip_yaw,
#               left_knee, left_ankle_pitch, left_ankle_roll
# Right leg (6): right_hip_pitch, right_hip_roll, right_hip_yaw,
#                right_knee, right_ankle_pitch, right_ankle_roll
# Waist (1): waist_yaw
# Left arm (7): left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw,
#               left_elbow, left_wrist_roll, left_wrist_pitch, left_wrist_yaw
# Right arm (7): right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw,
#                right_elbow, right_wrist_roll, right_wrist_pitch, right_wrist_yaw
# Head (2): head_yaw, head_pitch
_G1_JOINT_NAMES: List[str] = [
    # Left leg
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    # Right leg
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    # Waist
    "waist_yaw_joint",
    # Left arm
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # Right arm
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    # Head
    "head_yaw_joint", "head_pitch_joint",
]

# Body links for contact detection (26 links, Table S3: "body contact 0:26")
# Excludes the two foot end-effectors which are tracked separately.
_G1_BODY_LINK_NAMES: List[str] = [
    "pelvis",
    "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
    "left_knee_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
    "right_knee_link",
    "waist_yaw_link",
    "torso_link",
    "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
    "left_elbow_link", "left_wrist_roll_link",
    "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
    "right_elbow_link", "right_wrist_roll_link",
    "head_link",
    "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_ankle_pitch_link", "right_ankle_roll_link",
]

# Foot end-effector link names (2 feet) — for height and velocity tracking
_G1_FOOT_LINK_NAMES: List[str] = [
    "left_foot_link",
    "right_foot_link",
]

# Base link name — used for termination detection (Section A.4.3)
_G1_BASE_LINK_NAME: str = "pelvis"

# Default Unitree G1 standing joint positions (radians).
# Nominal standing pose with slight knee bend for stability.
# Order matches _G1_JOINT_NAMES exactly (29 values).
_G1_DEFAULT_JOINT_POS: List[float] = [
    # Left leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    # Right leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    # Waist: yaw
    0.0,
    # Left arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
    #           wrist_roll, wrist_pitch, wrist_yaw
    0.0, 0.2, 0.0, 0.5, 0.0, 0.0, 0.0,
    # Right arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
    #            wrist_roll, wrist_pitch, wrist_yaw
    0.0, -0.2, 0.0, 0.5, 0.0, 0.0, 0.0,
    # Head: yaw, pitch
    0.0, 0.0,
]

# PD controller gains for Unitree G1 joint position control.
# Leg joints use higher gains for stability; arm joints use lower gains.
_G1_KP_LEGS: float = 100.0   # Proportional gain for leg joints (N·m/rad)
_G1_KD_LEGS: float = 5.0     # Derivative gain for leg joints (N·m·s/rad)
_G1_KP_ARMS: float = 40.0    # Proportional gain for arm joints (N·m/rad)
_G1_KD_ARMS: float = 2.0     # Derivative gain for arm joints (N·m·s/rad)
_G1_KP_WAIST: float = 80.0   # Proportional gain for waist joint (N·m/rad)
_G1_KD_WAIST: float = 4.0    # Derivative gain for waist joint (N·m·s/rad)

# Action scale: policy output is scaled before adding to default joint pos.
# joint_target = default_joint_pos + action_scale * policy_output
_G1_ACTION_SCALE: float = 0.5  # radians

# Contact force threshold for binary contact detection (Newtons).
_CONTACT_FORCE_THRESHOLD: float = 1.0  # N

# Base contact force threshold for termination detection (Section A.4.3).
_BASE_CONTACT_THRESHOLD: float = 1.0  # N

# Number of simulation warmup steps after reset to allow physics to settle.
_WARMUP_STEPS: int = 5

# Maximum episode length in steps before timeout termination.
# At 50 Hz, 1000 steps = 20 seconds per episode.
_MAX_EPISODE_STEPS: int = 1000

# Joint position limits for action clipping (radians).
# Conservative limits to prevent joint damage.
_G1_JOINT_POS_LOWER: float = -3.14159  # -pi
_G1_JOINT_POS_UPPER: float = 3.14159   # +pi

# Number of body links tracked for contact detection (Table S3: body_contact 0:26)
_G1_NUM_BODY_CONTACTS: int = 26

# Number of feet tracked for height and velocity (Table S3: foot_height 26:28, foot_velocity 28:30)
_G1_NUM_FEET: int = 2


class UnitreeG1Env(BaseEnv):
    """Isaac Lab wrapper for the Unitree G1 humanoid robot.

    Implements the full ``BaseEnv`` interface for the Unitree G1 humanoid,
    providing the 96-dim world model observation, 30-dim privileged information,
    and 29-dim joint position target action space as specified in Tables S2-S4
    of the paper.

    The privileged information vector is heterogeneous:
      - Dims 0:26 — binary body contact flags (BCE loss in trainer)
      - Dims 26:28 — continuous foot heights in meters (MSE/NLL loss)
      - Dims 28:30 — continuous foot velocities in m/s (MSE/NLL loss)

    This heterogeneity is a key difference from ANYmal D (which has only
    binary contacts) and must be handled correctly in ``RWMTrainer._compute_loss``.

    **Requires Isaac Lab (Isaac Sim 4.x) to be installed.** If Isaac Lab is
    not available, ``__init__`` raises ``ImportError`` with installation
    instructions. Use ``MockEnv`` for pipeline testing without Isaac Lab.

    Attributes:
        robot: Isaac Lab ``Articulation`` object for the Unitree G1 robot.
        sim: Isaac Lab ``SimulationContext`` managing the physics simulation.
        scene: Isaac Lab ``InteractiveScene`` containing all simulation assets.
        contact_sensor: Isaac Lab ``ContactSensor`` for reading contact forces.
        action_scale: Scale factor applied to policy outputs. Default: 0.5 rad.
        base_contact_threshold: Force threshold (N) for base contact detection.
        _obs_buf: Pre-allocated observation buffer ``[num_envs, 96]``.
        _priv_buf: Pre-allocated privileged info buffer ``[num_envs, 30]``.
        _episode_length_buf: Per-environment step counter ``[num_envs]``.
        _prev_joint_vel: Previous joint velocities ``[num_envs, 29]`` for
            joint acceleration finite differencing.
        _gravity_world: Constant gravity vector in world frame ``[num_envs, 3]``.
        _body_link_indices: Indices of 26 body links in contact sensor data.
        _foot_link_indices: Indices of 2 foot links in contact sensor data.
        _base_link_index: Index of the base (pelvis) link in contact sensor data.
        _foot_body_indices: Indices of foot links in the articulation body list
            for querying foot positions and velocities.
    """

    def __init__(
        self,
        config: Any,
        num_envs: int = 1,
    ) -> None:
        """Initialize the Unitree G1 Isaac Lab environment.

        Validates Isaac Lab availability, resolves all configuration parameters,
        initializes the Isaac Lab simulation, loads the Unitree G1 asset, sets
        up contact sensors, and pre-allocates state buffers.

        Args:
            config: Hydra ``DictConfig`` or plain dict containing the full
                experiment configuration from ``config.yaml``. Must contain:
                - ``config.robot``: Must be "unitree_g1"
                - ``config.unitree_g1``: Robot sub-config with obs_dim=96,
                  action_dim=29, priv_dim=30, policy_obs_dim=99,
                  reward_weights (Table S6 G1 column), obs_slices, priv_slices,
                  sigma_vxy=0.25, sigma_wz=0.25
                - ``config.device``: "cuda" or "cpu"
                - ``config.simulation.dt``: 0.02 (50 Hz)
                - ``config.simulation.control_freq_hz``: 50
                - ``config.collision_handling.terminate_on_base_contact``: true
            num_envs: Number of parallel simulation environments. Use 1 for
                online fine-tuning (``simulation.num_envs_real``) or 4096 for
                pretraining data collection (``simulation.num_envs_pretrain``).
                Default: 1.

        Raises:
            ImportError: If Isaac Lab (Isaac Sim 4.x) is not installed.
                Install following: https://isaac-sim.github.io/IsaacLab/
            ValueError: If ``config.robot`` is not "unitree_g1".
        """
        if not ISAAC_LAB_AVAILABLE:
            raise ImportError(
                "Isaac Lab is required for UnitreeG1Env but is not installed. "
                "Install Isaac Lab following the instructions at: "
                "https://isaac-sim.github.io/IsaacLab/\n"
                "Alternatively, use MockEnv (env_backend='mock' in config.yaml) "
                "for pipeline testing without Isaac Lab."
            )

        # Validate robot type before calling super().__init__ to provide a
        # clearer error message than the generic BaseEnv validation.
        robot_type: str = str(config.robot)
        if robot_type != "unitree_g1":
            raise ValueError(
                f"UnitreeG1Env requires config.robot='unitree_g1', got '{robot_type}'. "
                "Use ANYmalEnv for the ANYmal D robot."
            )

        # Delegate all config parsing and dimension resolution to BaseEnv.
        # Sets: self.obs_dim=96, self.action_dim=29, self.priv_dim=30,
        # self.policy_obs_dim=99, self.num_envs, self.device, self.dt=0.02,
        # self.robot_type="unitree_g1", self.reward_weights, self.obs_slices,
        # self.policy_obs_slices, self.priv_slices, self.prev_actions,
        # self.default_joint_pos, self.commands, self.terminate_on_base_contact
        super().__init__(config, num_envs)

        # ----------------------------------------------------------------
        # Unitree G1 specific configuration
        # ----------------------------------------------------------------
        self.action_scale: float = _G1_ACTION_SCALE
        self.base_contact_threshold: float = _BASE_CONTACT_THRESHOLD
        self._max_episode_steps: int = _MAX_EPISODE_STEPS

        # Override default_joint_pos with G1's actual standing pose.
        # BaseEnv initializes this to zeros; we replace with the nominal pose.
        self.default_joint_pos = torch.tensor(
            _G1_DEFAULT_JOINT_POS,
            dtype=torch.float32,
            device=self.device,
        )  # shape [29]

        # ----------------------------------------------------------------
        # Pre-allocate state buffers for efficient tensor reuse in hot paths
        # ----------------------------------------------------------------
        self._obs_buf: Tensor = torch.zeros(
            self.num_envs,
            self.obs_dim,  # 96
            dtype=torch.float32,
            device=self.device,
        )
        self._priv_buf: Tensor = torch.zeros(
            self.num_envs,
            self.priv_dim,  # 30
            dtype=torch.float32,
            device=self.device,
        )
        self._episode_length_buf: Tensor = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        # Previous joint velocities for joint acceleration finite differencing.
        # Used in compute_reward → joint_accel_penalty.
        self._prev_joint_vel: Tensor = torch.zeros(
            self.num_envs,
            self.action_dim,  # 29 joints
            dtype=torch.float32,
            device=self.device,
        )
        # Gravity vector in world frame — constant, pre-allocated for efficiency.
        # [0, 0, -1] normalized gravity direction in world frame.
        self._gravity_world: Tensor = torch.zeros(
            self.num_envs,
            3,
            dtype=torch.float32,
            device=self.device,
        )
        self._gravity_world[:, 2] = -1.0  # [0, 0, -1] in world frame

        # ----------------------------------------------------------------
        # Initialize Isaac Lab simulation
        # ----------------------------------------------------------------
        self._init_simulation(config)

        # ----------------------------------------------------------------
        # Resolve link indices for contact sensor and body queries
        # ----------------------------------------------------------------
        self._resolve_link_indices()

        print(
            f"[UnitreeG1Env] Initialized with {self.num_envs} environments. "
            f"obs_dim={self.obs_dim}, action_dim={self.action_dim}, "
            f"priv_dim={self.priv_dim}, device={self.device}, dt={self.dt}s"
        )

    # ----------------------------------------------------------------
    # Isaac Lab initialization helpers
    # ----------------------------------------------------------------

    def _init_simulation(self, config: Any) -> None:
        """Initialize the Isaac Lab simulation, scene, robot, and sensors.

        Sets up the complete Isaac Lab simulation stack:
        1. SimulationContext with physics parameters
        2. Ground plane and lighting
        3. Unitree G1 articulation asset
        4. Contact sensors for body and foot links
        5. Scene finalization and physics warmup

        Args:
            config: Full experiment configuration from config.yaml.
        """
        # ----------------------------------------------------------------
        # 1. Create simulation context
        # ----------------------------------------------------------------
        sim_cfg = SimulationCfg(
            dt=self.dt,
            render_interval=1,
            gravity=(0.0, 0.0, -9.81),
            device=self.device,
        )
        self.sim: Any = SimulationContext(sim_cfg)

        # ----------------------------------------------------------------
        # 2. Set up scene with ground plane and lighting
        # ----------------------------------------------------------------
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/GroundPlane", ground_cfg)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # ----------------------------------------------------------------
        # 3. Load Unitree G1 articulation asset
        # ----------------------------------------------------------------
        try:
            from isaaclab_assets.robots.unitree import UNITREE_G1_CFG  # type: ignore[import]
            g1_cfg: Any = UNITREE_G1_CFG.replace(
                prim_path="/World/envs/env_.*/Robot",
            )
        except ImportError:
            warnings.warn(
                "Could not import UNITREE_G1_CFG from isaaclab_assets. "
                "Attempting to load from default USD path. "
                "Ensure isaaclab_assets is installed.",
                UserWarning,
                stacklevel=2,
            )
            # Build default joint position dict from the constant list
            default_joint_pos_dict: Dict[str, float] = {
                name: pos
                for name, pos in zip(_G1_JOINT_NAMES, _G1_DEFAULT_JOINT_POS)
            }
            g1_cfg = ArticulationCfg(
                prim_path="/World/envs/env_.*/Robot",
                spawn=sim_utils.UsdFileCfg(
                    usd_path="{ISAACLAB_ASSETS_DATA}/Robots/Unitree/G1/g1.usd",
                    activate_contact_sensors=True,
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 0.8),  # G1 is taller than ANYmal D
                    joint_pos=default_joint_pos_dict,
                ),
                actuators={
                    # Leg joints — higher gains for stability
                    "legs": sim_utils.ImplicitActuatorCfg(
                        joint_names_expr=[
                            ".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"
                        ],
                        effort_limit=100.0,
                        velocity_limit=20.0,
                        stiffness=_G1_KP_LEGS,
                        damping=_G1_KD_LEGS,
                    ),
                    # Waist joint
                    "waist": sim_utils.ImplicitActuatorCfg(
                        joint_names_expr=["waist_yaw_joint"],
                        effort_limit=80.0,
                        velocity_limit=10.0,
                        stiffness=_G1_KP_WAIST,
                        damping=_G1_KD_WAIST,
                    ),
                    # Arm joints — lower gains for compliance
                    "arms": sim_utils.ImplicitActuatorCfg(
                        joint_names_expr=[
                            ".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"
                        ],
                        effort_limit=40.0,
                        velocity_limit=20.0,
                        stiffness=_G1_KP_ARMS,
                        damping=_G1_KD_ARMS,
                    ),
                    # Head joints — minimal gains
                    "head": sim_utils.ImplicitActuatorCfg(
                        joint_names_expr=["head_.*"],
                        effort_limit=10.0,
                        velocity_limit=10.0,
                        stiffness=20.0,
                        damping=1.0,
                    ),
                },
            )

        # ----------------------------------------------------------------
        # 4. Set up contact sensors for all body links
        # ----------------------------------------------------------------
        contact_sensor_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/.*",
            update_period=0.0,  # Update every simulation step
            history_length=3,
            debug_vis=False,
            filter_prim_paths_expr=[
                "/World/GroundPlane",
            ],
        )

        # ----------------------------------------------------------------
        # 5. Create interactive scene
        # ----------------------------------------------------------------
        scene_cfg = InteractiveSceneCfg(
            num_envs=self.num_envs,
            env_spacing=3.0,  # 3.0m spacing — G1 is larger than ANYmal D
            replicate_physics=True,
        )
        self.scene: Any = InteractiveScene(scene_cfg)

        # Add robot and contact sensor to scene
        self.scene.articulations["robot"] = g1_cfg
        self.scene.sensors["contact_sensor"] = contact_sensor_cfg

        # ----------------------------------------------------------------
        # 6. Finalize scene and reset simulation
        # ----------------------------------------------------------------
        self.sim.reset()
        self.scene.reset()

        # Retrieve asset handles after scene finalization
        self.robot: Any = self.scene.articulations["robot"]
        self.contact_sensor: Any = self.scene.sensors["contact_sensor"]

        # ----------------------------------------------------------------
        # 7. Physics warmup — step with zero actions to settle contacts
        # ----------------------------------------------------------------
        zero_actions: Tensor = torch.zeros(
            self.num_envs,
            self.action_dim,
            dtype=torch.float32,
            device=self.device,
        )
        for _ in range(_WARMUP_STEPS):
            self._apply_actions(zero_actions)
            self.sim.step()
            self.scene.update(self.dt)

    def _resolve_link_indices(self) -> None:
        """Resolve link indices for contact sensor and body queries.

        Maps link names (body links, foot links, base) to their indices in
        the contact sensor's body name list. These indices are used in
        ``get_privileged_info`` and ``_check_base_contact`` to efficiently
        slice the contact force tensor.

        Also resolves foot body indices in the articulation for querying
        foot positions and velocities.

        Stores:
            self._body_link_indices: List[int] — indices of 26 body links
            self._foot_link_indices: List[int] — indices of 2 foot links
            self._base_link_index: int — index of the base (pelvis) link
            self._foot_body_indices: List[int] — body indices for foot
                position/velocity queries in the articulation
        """
        # Get all body names tracked by the contact sensor
        body_names: List[str] = list(self.contact_sensor.data.body_names)

        # ----------------------------------------------------------------
        # Resolve 26 body link indices (Table S3: body_contact 0:26)
        # ----------------------------------------------------------------
        self._body_link_indices: List[int] = []
        for body_name in _G1_BODY_LINK_NAMES:
            try:
                idx: int = body_names.index(body_name)
                self._body_link_indices.append(idx)
            except ValueError:
                warnings.warn(
                    f"[UnitreeG1Env] Body link '{body_name}' not found in "
                    f"contact sensor body names. "
                    "Body contact detection may be incorrect.",
                    UserWarning,
                    stacklevel=2,
                )
                # Use index 0 as fallback to avoid IndexError in hot path
                self._body_link_indices.append(0)

        # Ensure exactly 26 body link indices (pad or truncate if needed)
        if len(self._body_link_indices) != _G1_NUM_BODY_CONTACTS:
            warnings.warn(
                f"[UnitreeG1Env] Expected {_G1_NUM_BODY_CONTACTS} body link "
                f"indices, got {len(self._body_link_indices)}. "
                "Padding/truncating to match priv_dim.",
                UserWarning,
                stacklevel=2,
            )
            # Pad with zeros or truncate to exactly 26
            while len(self._body_link_indices) < _G1_NUM_BODY_CONTACTS:
                self._body_link_indices.append(0)
            self._body_link_indices = self._body_link_indices[:_G1_NUM_BODY_CONTACTS]

        # ----------------------------------------------------------------
        # Resolve 2 foot link indices (Table S3: foot_height 26:28, foot_velocity 28:30)
        # ----------------------------------------------------------------
        self._foot_link_indices: List[int] = []
        for foot_name in _G1_FOOT_LINK_NAMES:
            try:
                idx = body_names.index(foot_name)
                self._foot_link_indices.append(idx)
            except ValueError:
                warnings.warn(
                    f"[UnitreeG1Env] Foot link '{foot_name}' not found in "
                    f"contact sensor body names. "
                    "Foot height/velocity tracking may be incorrect.",
                    UserWarning,
                    stacklevel=2,
                )
                self._foot_link_indices.append(0)

        # Ensure exactly 2 foot link indices
        if len(self._foot_link_indices) != _G1_NUM_FEET:
            warnings.warn(
                f"[UnitreeG1Env] Expected {_G1_NUM_FEET} foot link indices, "
                f"got {len(self._foot_link_indices)}. Padding/truncating.",
                UserWarning,
                stacklevel=2,
            )
            while len(self._foot_link_indices) < _G1_NUM_FEET:
                self._foot_link_indices.append(0)
            self._foot_link_indices = self._foot_link_indices[:_G1_NUM_FEET]

        # ----------------------------------------------------------------
        # Resolve base (pelvis) link index for termination detection
        # ----------------------------------------------------------------
        try:
            self._base_link_index: int = body_names.index(_G1_BASE_LINK_NAME)
        except ValueError:
            warnings.warn(
                f"[UnitreeG1Env] Base link '{_G1_BASE_LINK_NAME}' not found in "
                f"contact sensor body names. "
                "Base contact termination detection may be incorrect.",
                UserWarning,
                stacklevel=2,
            )
            self._base_link_index = 0

        # ----------------------------------------------------------------
        # Resolve foot body indices in the articulation for position/velocity
        # ----------------------------------------------------------------
        # Isaac Lab articulation body names may differ from contact sensor names
        try:
            articulation_body_names: List[str] = list(self.robot.data.body_names)
            self._foot_body_indices: List[int] = []
            for foot_name in _G1_FOOT_LINK_NAMES:
                try:
                    idx = articulation_body_names.index(foot_name)
                    self._foot_body_indices.append(idx)
                except ValueError:
                    warnings.warn(
                        f"[UnitreeG1Env] Foot link '{foot_name}' not found in "
                        f"articulation body names. Using index 0 as fallback.",
                        UserWarning,
                        stacklevel=2,
                    )
                    self._foot_body_indices.append(0)
        except AttributeError:
            #