## Code: envs/anymal_env.py

```python
## envs/anymal_env.py
"""Isaac Lab wrapper for the ANYmal D quadruped robot environment.

This module provides ``ANYmalEnv``, a concrete implementation of ``BaseEnv``
that wraps Isaac Lab's ANYmal D simulation. It constructs the 45-dim world
model observation (Table S2), 8-dim privileged information (Table S3), and
manages the 12-dim joint position target action space (Table S4).

If Isaac Lab is not installed, ``__init__`` raises ``ImportError`` with a
clear installation message. Use ``MockEnv`` for pipeline testing without
Isaac Lab.

Observation construction follows Table S2 exactly:
    obs[0:3]   = base linear velocity in robot body frame
    obs[3:6]   = base angular velocity in robot body frame
    obs[6:9]   = projected gravity vector in robot body frame
    obs[9:21]  = joint positions (12 joints, radians)
    obs[21:33] = joint velocities (12 joints, rad/s)
    obs[33:45] = applied joint torques (12 joints, N·m)

Privileged information follows Table S3:
    priv[0:4]  = knee contact flags (4 knees, binary float)
    priv[4:8]  = foot contact flags (4 feet, binary float)

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
    # Isaac Lab core simulation
    from isaaclab.app import AppLauncher  # noqa: F401
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from isaaclab.utils.math import (
        quat_rotate_inverse,
        subtract_frame_transforms,
    )
    import isaaclab.sim as sim_utils

    ISAAC_LAB_AVAILABLE = True
except ImportError:
    # Provide stub types so the class body parses without Isaac Lab.
    # All actual usage is guarded by ISAAC_LAB_AVAILABLE checks in __init__.
    SimulationContext = None  # type: ignore[assignment,misc]
    Articulation = None       # type: ignore[assignment,misc]
    ContactSensor = None      # type: ignore[assignment,misc]
    ISAAC_LAB_AVAILABLE = False


# ---------------------------------------------------------------------------
# ANYmal D joint and link name constants
# Standard naming convention from the ANYmal D USD asset in Isaac Lab.
# Order matches the articulation's joint ordering in the USD file.
# ---------------------------------------------------------------------------

# 12 joints in order: LF_HAA, LF_HFE, LF_KFE, RF_HAA, RF_HFE, RF_KFE,
#                     LH_HAA, LH_HFE, LH_KFE, RH_HAA, RH_HFE, RH_KFE
_ANYMAL_JOINT_NAMES: List[str] = [
    "LF_HAA", "LF_HFE", "LF_KFE",
    "RF_HAA", "RF_HFE", "RF_KFE",
    "LH_HAA", "LH_HFE", "LH_KFE",
    "RH_HAA", "RH_HFE", "RH_KFE",
]

# Knee links (4 knees) — used for undesired contact detection (Table S3)
_ANYMAL_KNEE_LINK_NAMES: List[str] = [
    "LF_KFE", "RF_KFE", "LH_KFE", "RH_KFE",
]

# Foot links (4 feet) — used for foot contact detection (Table S3)
_ANYMAL_FOOT_LINK_NAMES: List[str] = [
    "LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT",
]

# Base link name — used for termination detection (Section A.4.3)
_ANYMAL_BASE_LINK_NAME: str = "base"

# Default ANYmal D standing joint positions (radians).
# These are the nominal joint angles for a stable standing pose.
# Source: ANYmal D hardware documentation and Isaac Lab default config.
_ANYMAL_DEFAULT_JOINT_POS: List[float] = [
    0.0,   0.4,  -0.8,   # LF: HAA, HFE, KFE
    0.0,   0.4,  -0.8,   # RF: HAA, HFE, KFE
    0.0,  -0.4,   0.8,   # LH: HAA, HFE, KFE
    0.0,  -0.4,   0.8,   # RH: HAA, HFE, KFE
]

# PD controller gains for ANYmal D joint position control.
# These are standard values from Isaac Lab's ANYmal D configuration.
_ANYMAL_KP: float = 80.0   # Proportional gain (N·m/rad)
_ANYMAL_KD: float = 2.0    # Derivative gain (N·m·s/rad)

# Action scale: policy output is scaled before adding to default joint pos.
# joint_target = default_joint_pos + action_scale * policy_output
_ANYMAL_ACTION_SCALE: float = 0.5  # radians

# Contact force threshold for binary contact detection (Newtons).
# Forces above this threshold are classified as contact (1.0), else no contact (0.0).
_CONTACT_FORCE_THRESHOLD: float = 1.0  # N

# Base contact force threshold for termination detection (Section A.4.3).
_BASE_CONTACT_THRESHOLD: float = 1.0  # N

# Number of simulation warmup steps after reset to allow physics to settle.
_WARMUP_STEPS: int = 5

# Maximum episode length in steps before timeout termination.
# At 50 Hz, 1000 steps = 20 seconds per episode.
_MAX_EPISODE_STEPS: int = 1000


def _quat_rotate_inverse_torch(
    quat: Tensor,
    vec: Tensor,
) -> Tensor:
    """Rotate a vector by the inverse of a quaternion (body frame transform).

    Implements the rotation v_body = R^T @ v_world, where R is the rotation
    matrix corresponding to the quaternion. This is used to transform world-
    frame velocities and gravity into the robot body frame for observations.

    Uses the scalar-first quaternion convention [w, x, y, z] as used by
    Isaac Lab. The formula is:
        v_body = q^{-1} ⊗ [0, v_world] ⊗ q

    This pure-PyTorch fallback is used when Isaac Lab's math utilities are
    not available (e.g., in unit tests with MockEnv).

    Args:
        quat: Quaternion tensor of shape ``[B, 4]`` in [w, x, y, z] order.
            Must be unit quaternions (normalized).
        vec: Vector tensor of shape ``[B, 3]`` to rotate.

    Returns:
        Rotated vector tensor of shape ``[B, 3]`` in the body frame.
    """
    # Extract quaternion components (scalar-first: [w, x, y, z])
    w: Tensor = quat[:, 0:1]   # shape [B, 1]
    x: Tensor = quat[:, 1:2]   # shape [B, 1]
    y: Tensor = quat[:, 2:3]   # shape [B, 1]
    z: Tensor = quat[:, 3:4]   # shape [B, 1]

    # Extract vector components
    vx: Tensor = vec[:, 0:1]   # shape [B, 1]
    vy: Tensor = vec[:, 1:2]   # shape [B, 1]
    vz: Tensor = vec[:, 2:3]   # shape [B, 1]

    # Rotation matrix R^T applied to vec (inverse rotation = transpose of R)
    # R^T @ v = q^{-1} ⊗ [0, v] ⊗ q
    # Expanded form of the rotation formula:
    rx: Tensor = (
        (1.0 - 2.0 * (y * y + z * z)) * vx
        + 2.0 * (x * y + w * z) * vy
        + 2.0 * (x * z - w * y) * vz
    )
    ry: Tensor = (
        2.0 * (x * y - w * z) * vx
        + (1.0 - 2.0 * (x * x + z * z)) * vy
        + 2.0 * (y * z + w * x) * vz
    )
    rz: Tensor = (
        2.0 * (x * z + w * y) * vx
        + 2.0 * (y * z - w * x) * vy
        + (1.0 - 2.0 * (x * x + y * y)) * vz
    )

    return torch.cat([rx, ry, rz], dim=-1)  # shape [B, 3]


class ANYmalEnv(BaseEnv):
    """Isaac Lab wrapper for the ANYmal D quadruped robot.

    Implements the full ``BaseEnv`` interface for the ANYmal D quadruped,
    providing the 45-dim world model observation, 8-dim privileged information,
    and 12-dim joint position target action space as specified in Tables S2-S4
    of the paper.

    The environment manages ``num_envs`` parallel simulation instances in
    Isaac Lab, supporting both single-environment online fine-tuning
    (``num_envs=1``) and large-scale pretraining data collection
    (``num_envs=4096``).

    **Requires Isaac Lab (Isaac Sim 4.x) to be installed.** If Isaac Lab is
    not available, ``__init__`` raises ``ImportError`` with installation
    instructions. Use ``MockEnv`` for pipeline testing without Isaac Lab.

    Attributes:
        robot: Isaac Lab ``Articulation`` object for the ANYmal D robot.
            Provides access to joint states, root state, and applied torques.
        sim: Isaac Lab ``SimulationContext`` managing the physics simulation.
        scene: Isaac Lab ``InteractiveScene`` containing all simulation assets.
        contact_sensor: Isaac Lab ``ContactSensor`` for reading contact forces
            on knee and foot links.
        action_scale: Scale factor applied to policy outputs before adding to
            default joint positions. Default: 0.5 rad.
        base_contact_threshold: Force threshold (N) for base contact detection.
            Episodes terminate when base contact force exceeds this threshold.
        _obs_buf: Pre-allocated observation buffer of shape
            ``[num_envs, obs_dim]`` for efficient tensor reuse.
        _priv_buf: Pre-allocated privileged info buffer of shape
            ``[num_envs, priv_dim]``.
        _episode_length_buf: Per-environment step counter of shape
            ``[num_envs]`` for timeout termination detection.
        _knee_link_indices: Indices of knee links in the contact sensor data.
        _foot_link_indices: Indices of foot links in the contact sensor data.
        _base_link_index: Index of the base link in the contact sensor data.
    """

    def __init__(
        self,
        config: Any,
        num_envs: int = 1,
    ) -> None:
        """Initialize the ANYmal D Isaac Lab environment.

        Validates Isaac Lab availability, resolves all configuration parameters,
        initializes the Isaac Lab simulation, loads the ANYmal D asset, sets up
        contact sensors, and pre-allocates state buffers.

        Args:
            config: Hydra ``DictConfig`` or plain dict containing the full
                experiment configuration from ``config.yaml``. Must contain:
                - ``config.robot``: Must be "anymal_d"
                - ``config.anymal_d``: Robot sub-config with obs_dim=45,
                  action_dim=12, priv_dim=8, policy_obs_dim=48,
                  reward_weights (Table S6), obs_slices, priv_slices,
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
            ValueError: If ``config.robot`` is not "anymal_d".
        """
        if not ISAAC_LAB_AVAILABLE:
            raise ImportError(
                "Isaac Lab is required for ANYmalEnv but is not installed. "
                "Install Isaac Lab following the instructions at: "
                "https://isaac-sim.github.io/IsaacLab/\n"
                "Alternatively, use MockEnv (env_backend='mock' in config.yaml) "
                "for pipeline testing without Isaac Lab."
            )

        # Validate robot type before calling super().__init__ to provide a
        # clearer error message than the generic BaseEnv validation.
        robot_type: str = str(config.robot)
        if robot_type != "anymal_d":
            raise ValueError(
                f"ANYmalEnv requires config.robot='anymal_d', got '{robot_type}'. "
                "Use UnitreeG1Env for the Unitree G1 robot."
            )

        # Delegate all config parsing and dimension resolution to BaseEnv.
        # Sets: self.obs_dim=45, self.action_dim=12, self.priv_dim=8,
        # self.policy_obs_dim=48, self.num_envs, self.device, self.dt=0.02,
        # self.robot_type="anymal_d", self.reward_weights, self.obs_slices,
        # self.policy_obs_slices, self.priv_slices, self.prev_actions,
        # self.default_joint_pos, self.commands, self.terminate_on_base_contact
        super().__init__(config, num_envs)

        # ----------------------------------------------------------------
        # ANYmal D specific configuration
        # ----------------------------------------------------------------
        self.action_scale: float = _ANYMAL_ACTION_SCALE
        self.base_contact_threshold: float = _BASE_CONTACT_THRESHOLD
        self._max_episode_steps: int = _MAX_EPISODE_STEPS

        # Override default_joint_pos with ANYmal D's actual standing pose.
        # BaseEnv initializes this to zeros; we replace with the nominal pose.
        self.default_joint_pos = torch.tensor(
            _ANYMAL_DEFAULT_JOINT_POS,
            dtype=torch.float32,
            device=self.device,
        )  # shape [12]

        # ----------------------------------------------------------------
        # Pre-allocate state buffers for efficient tensor reuse in hot paths
        # ----------------------------------------------------------------
        self._obs_buf: Tensor = torch.zeros(
            self.num_envs,
            self.obs_dim,
            dtype=torch.float32,
            device=self.device,
        )
        self._priv_buf: Tensor = torch.zeros(
            self.num_envs,
            self.priv_dim,
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
            self.action_dim,  # 12 joints
            dtype=torch.float32,
            device=self.device,
        )
        # Gravity vector in world frame — constant, pre-allocated for efficiency.
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
        # Resolve link indices for contact sensor queries
        # ----------------------------------------------------------------
        self._resolve_link_indices()

        print(
            f"[ANYmalEnv] Initialized with {self.num_envs} environments. "
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
        3. ANYmal D articulation asset
        4. Contact sensors for knee, foot, and base links
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
        self.sim: SimulationContext = SimulationContext(sim_cfg)

        # ----------------------------------------------------------------
        # 2. Set up scene with ground plane and lighting
        # ----------------------------------------------------------------
        # Ground plane
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/GroundPlane", ground_cfg)

        # Distant light for rendering
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # ----------------------------------------------------------------
        # 3. Load ANYmal D articulation asset
        # ----------------------------------------------------------------
        # Use Isaac Lab's built-in ANYmal D asset configuration.
        # The asset is loaded from the Isaac Lab asset registry.
        try:
            from isaaclab_assets.robots.anymal import ANYMAL_D_CFG  # type: ignore[import]
            anymal_cfg: ArticulationCfg = ANYMAL_D_CFG.replace(
                prim_path="/World/envs/env_.*/Robot",
            )
        except ImportError:
            # Fallback: construct a minimal ArticulationCfg if the asset
            # registry is not available. This requires the USD file to be
            # present at the specified path.
            warnings.warn(
                "Could not import ANYMAL_D_CFG from isaaclab_assets. "
                "Attempting to load from default USD path. "
                "Ensure isaaclab_assets is installed.",
                UserWarning,
                stacklevel=2,
            )
            anymal_cfg = ArticulationCfg(
                prim_path="/World/envs/env_.*/Robot",
                spawn=sim_utils.UsdFileCfg(
                    usd_path="{ISAACLAB_ASSETS_DATA}/Robots/ANYbotics/ANYmal-D/anymal_d.usd",
                    activate_contact_sensors=True,
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 0.6),
                    joint_pos={
                        "LF_HAA": 0.0, "LF_HFE": 0.4, "LF_KFE": -0.8,
                        "RF_HAA": 0.0, "RF_HFE": 0.4, "RF_KFE": -0.8,
                        "LH_HAA": 0.0, "LH_HFE": -0.4, "LH_KFE": 0.8,
                        "RH_HAA": 0.0, "RH_HFE": -0.4, "RH_KFE": 0.8,
                    },
                ),
                actuators={
                    "legs": sim_utils.ImplicitActuatorCfg(
                        joint_names_expr=[".*_HAA", ".*_HFE", ".*_KFE"],
                        effort_limit=80.0,
                        velocity_limit=10.0,
                        stiffness=_ANYMAL_KP,
                        damping=_ANYMAL_KD,
                    ),
                },
            )

        # ----------------------------------------------------------------
        # 4. Set up contact sensors for knee, foot, and base links
        # ----------------------------------------------------------------
        # All contact links combined into a single sensor for efficiency.
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
            env_spacing=2.5,  # 2.5m spacing between parallel environments
            replicate_physics=True,
        )
        self.scene: InteractiveScene = InteractiveScene(scene_cfg)

        # Add robot and contact sensor to scene
        self.scene.articulations["robot"] = anymal_cfg
        self.scene.sensors["contact_sensor"] = contact_sensor_cfg

        # ----------------------------------------------------------------
        # 6. Finalize scene and reset simulation
        # ----------------------------------------------------------------
        self.sim.reset()
        self.scene.reset()

        # Retrieve asset handles after scene finalization
        self.robot: Articulation = self.scene.articulations["robot"]
        self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]

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
        """Resolve link indices for contact sensor queries.

        Maps link names (knee, foot, base) to their indices in the contact
        sensor's body name list. These indices are used in ``get_privileged_info``
        and ``_check_base_contact`` to efficiently slice the contact force tensor.

        Stores:
            self._knee_link_indices: List[int] — indices of 4 knee links
            self._foot_link_indices: List[int] — indices of 4 foot links
            self._base_link_index: int — index of the base link
        """
        # Get all body names tracked by the contact sensor
        body_names: List[str] = self.contact_sensor.data.body_names

        # Resolve knee link indices
        self._knee_link_indices: List[int] = []
        for knee_name in _ANYMAL_KNEE_LINK_NAMES:
            try:
                idx: int = body_names.index(knee_name)
                self._knee_link_indices.append(idx)
            except ValueError:
                warnings.warn(
                    f"[ANYmalEnv] Knee link '{knee_name}' not found in contact "
                    f"sensor body names: {body_names}. "
                    "Knee contact detection may be incorrect.",
                    UserWarning,
                    stacklevel=2,
                )
                # Use index 0 as fallback to avoid IndexError
                self._knee_link_indices.append(0)

        # Resolve foot link indices
        self._foot_link_indices: List[int] = []
        for foot_name in _ANYMAL_FOOT_LINK_NAMES:
            try:
                idx = body_names.index(foot_name)
                self._foot_link_indices.append(idx)
            except ValueError:
                warnings.warn(
                    f"[ANYmalEnv] Foot link '{foot_name}' not found in contact "
                    f"sensor body names: {body_names}. "
                    "Foot contact detection may be incorrect.",
                    UserWarning,
                    stacklevel=2,
                )
                self._foot_link_indices.append(0)

        # Resolve base link index
        try:
            self._base_link_index: int = body_names.index(_ANYMAL_BASE_LINK_NAME)
        except ValueError:
            warnings.warn(
                f"[ANYmalEnv] Base link '{_ANYMAL_BASE_LINK_NAME}' not found in "
                f"contact sensor body names: {body_names}. "
                "Base contact termination detection may be incorrect.",
                UserWarning,
                stacklevel=2,
            )
            self._base_link_index = 0

    # ----------------------------------------------------------------
    # Abstract method implementations
    # ----------------------------------------------------------------

    def reset(self) -> Tuple[Tensor, Tensor]:
        """Reset all environments and return initial observations and commands.

        Resets the Isaac Lab scene to default states with small random
        perturbations for diversity, samples new velocity commands, and
        performs physics warmup steps to settle contacts.

        Returns:
            A tuple ``(obs, command)`` where:
              - ``obs``: Initial world model observation of shape
                ``[num_envs, 45]``. Contains base velocities, gravity,
                joint positions, velocities, and torques (Table S2).
              - ``command``: Sampled velocity commands of shape
                ``[num_envs, 3]``. Each row is ``[vx, vy, yaw_rate]``.
        """
        # ----------------------------------------------------------------
        # 1. Reset Isaac Lab scene — reinitializes all environments
        # ----------------------------------------------------------------
        # Create reset indices for all environments
        reset_env_ids: Tensor = torch.arange(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        self.scene.reset(env_ids=reset_env_ids)

        # ----------------------------------------------------------------
        # 2. Reset all stateful buffers
        # ----------------------------------------------------------------
        self._episode_length_buf.zero_()
        self.prev_actions.zero_()
        self._prev_joint_vel.zero_()

        # ----------------------------------------------------------------
        # 3. Sample new velocity commands for all environments
        # ----------------------------------------------------------------
        self.commands = self._sample_commands(self.num_envs)

        # ----------------------------------------------------------------
        # 4. Physics warmup — step with zero actions to settle contacts
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

        # ----------------------------------------------------------------
        # 5. Construct initial observation
        # ----------------------------------------------------------------
        obs: Tensor = self._get_observations()

        return obs, self.commands.clone()

    def step(
        self,
        action: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]:
        """Apply joint position targets and advance the simulation by one step.

        Applies the action through the PD controller, steps the physics
        simulation by dt=0.02s, reads back the new state, computes rewards,
        detects terminations, and handles partial resets for done environments.

        Args:
            action: Joint position targets of shape ``[num_envs, 12]``.
                Policy outputs scaled by ``action_scale`` and offset by
                ``default_joint_pos``. Values in radians.

        Returns:
            A tuple ``(next_obs, priv, reward, done, info)`` where:
              - ``next_obs``: Next world model observation, shape
                ``[num_envs, 45]``.
              - ``priv``: Privileged information, shape ``[num_envs, 8]``.
              - ``reward``: Scalar reward per environment, shape
                ``[num_envs]``.
              - ``done``: Float termination flags (0.0 or 1.0), shape
                ``[num_envs]``. 1.0 for terminated environments.
              - ``info``: Dict with "base_contact", "episode_length",
                "timeout" keys.
        """
        # Ensure action is on the correct device
        action = action.to(self.device)

        # Store current observation for reward computation (before stepping)
        current_obs: Tensor = self._obs_buf.clone()

        # ----------------------------------------------------------------
        # 1. Apply actions through PD controller
        # ----------------------------------------------------------------
        self._apply_actions(action)

        # ----------------------------------------------------------------
        # 2. Step physics simulation
        # ----------------------------------------------------------------
        self.sim.step()
        self.scene.update(self.dt)

        # ----------------------------------------------------------------
        # 3. Increment episode length counters
        # ----------------------------------------------------------------
        self._episode_length_buf += 1

        # ----------------------------------------------------------------
        # 4. Read new state
        # ----------------------------------------------------------------
        next_obs: Tensor = self._get_observations()
        priv: Tensor = self.get_privileged_info()

        # ----------------------------------------------------------------
        # 5. Detect termination conditions
        # ----------------------------------------------------------------
        # Base contact termination (Section A.4.3)
        base_contact: Tensor = self._check_base_contact()  # bool [num_envs]

        # Timeout termination
        timeout: Tensor = self._episode_length_buf >= self._max_episode_steps

        # Combined termination
        done_bool: Tensor = base_contact | timeout  # bool [num_envs]

        # ----------------------------------------------------------------
        # 6. Compute rewards
        # ----------------------------------------------------------------
        # Compute feet air time from contact sensor data
        t_fa: Tensor = self._compute_feet_air_time()

        reward: Tensor = self.compute_reward(
            obs=current_obs,
            action=action,
            next_obs=next_obs,
            priv=priv,
            command=self.commands,
            t_fa=t_fa,
        )

        # ----------------------------------------------------------------
        # 7. Handle partial