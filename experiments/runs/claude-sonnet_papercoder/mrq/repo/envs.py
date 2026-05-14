## envs.py
"""Environment wrappers for MR.Q across all four benchmark families.

Provides a single EnvWrapper class that abstracts over:
  - Gym locomotion (Ant-v4, HalfCheetah-v4, Hopper-v4, Humanoid-v4, Walker2d-v4)
  - DMC proprioceptive (28 tasks, vector observations, action_repeat=2)
  - DMC visual (28 tasks, 84x84 RGB images, 3-frame stack, action_repeat=2)
  - Atari (57 games, 84x84 grayscale, 4-frame stack with max-pooling, action_repeat=4)

All benchmarks present a uniform 4-tuple step interface to the agent:
    (next_obs: np.ndarray, reward: float, done: bool, info: dict)

Observations are always float32 numpy arrays. Image observations are returned
as raw uint8-range float32 values in [0, 255]; normalization to [-0.5, 0.5]
is performed inside the CNN forward pass (state / 255.0 - 0.5).

Action scaling convention:
  - Continuous: agent works in [-1, 1]; wrapper scales to env range.
  - Discrete: agent works with one-hot vectors; wrapper converts to int.
"""

import collections
from typing import Deque, Dict, Optional, Tuple

import cv2
import gymnasium
import numpy as np


class EnvWrapper:
    """Unified environment wrapper for all MR.Q benchmarks.

    Handles benchmark-specific preprocessing, action repeat, frame stacking,
    and action space normalization. Presents a consistent interface to the
    MRQAgent regardless of the underlying environment type.

    Attributes:
        env_name: Environment identifier string (e.g., 'HalfCheetah-v4').
        benchmark: Benchmark category ('gym', 'dmc_proprio', 'dmc_visual',
            'atari').
        seed: Random seed used for environment initialization.
        action_repeat: Number of times each action is repeated per step.
        frame_stack: Number of frames stacked as a single observation.
        image_obs: Whether observations are images (True) or vectors (False).
        discrete: Whether the action space is discrete.
        state_shape: Tuple describing the observation shape returned by
            reset() and step(). Channels-first for image observations.
        action_dim: Number of action dimensions (or number of discrete
            actions for Atari).
        action_scale: Half-range of the continuous action space, shape
            (action_dim,). Used to scale from [-1, 1] to env range.
            None for discrete environments.
        action_bias: Midpoint of the continuous action space, shape
            (action_dim,). Used to center the action range.
            None for discrete environments.
    """

    # Image size for all visual benchmarks (84x84 per paper Appendix B.3)
    _IMAGE_SIZE: int = 84

    def __init__(
        self,
        env_name: str,
        benchmark: str,
        seed: int = 0,
        render: bool = False,
    ) -> None:
        """Initialise the environment wrapper.

        Dispatches to the appropriate factory method based on benchmark,
        then derives all derived properties (state_shape, action_dim, etc.)
        from the created environment.

        Args:
            env_name: Environment identifier. For Gym: 'HalfCheetah-v4'.
                For DMC: 'cheetah-run'. For Atari: 'Alien'.
            benchmark: One of 'gym', 'dmc_proprio', 'dmc_visual', 'atari'.
            seed: Random seed for reproducibility.
            render: If True, enable rendering (not used during training).

        Raises:
            ValueError: If benchmark is not a recognised benchmark identifier.
        """
        self.env_name: str = env_name
        self.benchmark: str = benchmark
        self.seed: int = seed
        self._render: bool = render
        self._first_reset: bool = True

        # Benchmark-specific settings (set by factory methods)
        self.action_repeat: int = 1
        self.frame_stack: int = 1
        self.image_obs: bool = False
        self.discrete: bool = False

        # Frame deque for image observations (filled by factory methods)
        self._frames: Optional[Deque[np.ndarray]] = None

        # Dispatch to the appropriate factory
        _valid_benchmarks = ("gym", "dmc_proprio", "dmc_visual", "atari")
        if benchmark not in _valid_benchmarks:
            raise ValueError(
                f"Invalid benchmark '{benchmark}'. "
                f"Must be one of {_valid_benchmarks}."
            )

        if benchmark == "gym":
            self.env = self._make_gym_env(env_name, seed)
        elif benchmark == "dmc_proprio":
            self.env = self._make_dmc_proprio_env(env_name, seed)
        elif benchmark == "dmc_visual":
            self.env = self._make_dmc_visual_env(env_name, seed)
        elif benchmark == "atari":
            self.env = self._make_atari_env(env_name, seed)

        # Derive state_shape and action properties from the created environment
        self._derive_properties()

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    def _make_gym_env(self, env_name: str, seed: int) -> gymnasium.Env:
        """Create a Gymnasium MuJoCo locomotion environment.

        From config.yaml benchmarks.gym:
            action_repeat: 1
            frame_stack: 1
            image_obs: false
            discrete: false
            env_version: v4

        No preprocessing is applied (paper Appendix B.3: "No preprocessing
        is applied").

        Args:
            env_name: Environment name including version suffix (e.g.,
                'HalfCheetah-v4').
            seed: Random seed (applied on first reset).

        Returns:
            Gymnasium environment instance.
        """
        self.action_repeat = 1
        self.frame_stack = 1
        self.image_obs = False
        self.discrete = False

        env = gymnasium.make(env_name)
        return env

    def _make_dmc_proprio_env(self, env_name: str, seed: int) -> gymnasium.Env:
        """Create a DMC proprioceptive environment via shimmy.

        From config.yaml benchmarks.dmc_proprio:
            action_repeat: 2
            frame_stack: 1
            image_obs: false
            discrete: false

        Action repeat is handled inside step() rather than via a wrapper.
        DMC environments are registered by shimmy as 'dm_control/domain-task-v0'.

        Args:
            env_name: DMC task name in 'domain-task' format (e.g.,
                'cheetah-run', 'ball_in_cup-catch').
            seed: Random seed (applied on first reset).

        Returns:
            Gymnasium-compatible DMC environment instance.
        """
        self.action_repeat = 2
        self.frame_stack = 1
        self.image_obs = False
        self.discrete = False

        gym_id = f"dm_control/{env_name}-v0"
        env = gymnasium.make(gym_id)
        return env

    def _make_dmc_visual_env(self, env_name: str, seed: int) -> gymnasium.Env:
        """Create a DMC visual environment with pixel observations via shimmy.

        From config.yaml benchmarks.dmc_visual:
            action_repeat: 2
            frame_stack: 3
            image_obs: true
            image_size: 84

        Requests pixel observations via render_mode='rgb_array'. Frame
        stacking (3 RGB frames → 9 channels) is handled inside the wrapper.

        Args:
            env_name: DMC task name in 'domain-task' format.
            seed: Random seed (applied on first reset).

        Returns:
            Gymnasium-compatible DMC environment with pixel observations.
        """
        self.action_repeat = 2
        self.frame_stack = 3
        self.image_obs = True
        self.discrete = False

        # Initialize frame deque: 3 frames, each (3, 84, 84) CHW float32
        self._frames = collections.deque(maxlen=self.frame_stack)

        gym_id = f"dm_control/{env_name}-v0"
        env = gymnasium.make(gym_id, render_mode="rgb_array")
        return env

    def _make_atari_env(self, env_name: str, seed: int) -> gymnasium.Env:
        """Create an Atari ALE environment with sticky actions.

        From config.yaml benchmarks.atari:
            action_repeat: 4
            frame_stack: 4
            image_obs: true
            image_size: 84
            grayscale: true
            sticky_actions: true
            sticky_action_prob: 0.25
            discrete: true
            env_version: v5

        Frame construction follows the paper exactly (Appendix B.3):
            o_j = max(f_{4j+2}, f_{4j+3})
            state = [o_0, o_1, o_2, o_3]
        This is implemented inside step() via _max_pool_frames().

        Args:
            env_name: Atari game name without version (e.g., 'Alien').
            seed: Random seed (applied on first reset).

        Returns:
            Gymnasium ALE environment instance.
        """
        self.action_repeat = 4
        self.frame_stack = 4
        self.image_obs = True
        self.discrete = True

        # Initialize frame deque: 4 max-pooled observations, each (84, 84)
        self._frames = collections.deque(maxlen=self.frame_stack)

        # ALE v5 with sticky actions (repeat_action_probability=0.25)
        gym_id = f"ALE/{env_name}-v5"
        env = gymnasium.make(
            gym_id,
            repeat_action_probability=0.25,  # sticky actions per Machado et al. 2018
        )
        return env

    # ------------------------------------------------------------------
    # Property derivation
    # ------------------------------------------------------------------

    def _derive_properties(self) -> None:
        """Derive state_shape, action_dim, and action scaling from the env.

        Called once after the environment is created. Sets:
            self.state_shape: Observation shape tuple.
            self.action_dim: Number of action dimensions.
            self.action_scale: Half-range for continuous action scaling.
            self.action_bias: Midpoint for continuous action centering.
        """
        obs_space = self.env.observation_space
        act_space = self.env.action_space

        if self.image_obs:
            if self.benchmark == "dmc_visual":
                # 3 stacked RGB frames: (3 * 3, 84, 84) = (9, 84, 84)
                self.state_shape: Tuple[int, ...] = (
                    self.frame_stack * 3,
                    self._IMAGE_SIZE,
                    self._IMAGE_SIZE,
                )
            else:
                # Atari: 4 stacked grayscale frames: (4, 84, 84)
                self.state_shape = (
                    self.frame_stack,
                    self._IMAGE_SIZE,
                    self._IMAGE_SIZE,
                )
        else:
            # Vector observation: use environment's native shape
            self.state_shape = tuple(obs_space.shape)

        if self.discrete:
            # Discrete action space: action_dim = number of actions
            self.action_dim: int = int(act_space.n)
            self.action_scale: Optional[np.ndarray] = None
            self.action_bias: Optional[np.ndarray] = None
        else:
            # Continuous action space
            self.action_dim = int(act_space.shape[0])
            # Scale and bias for mapping [-1, 1] ↔ [low, high]
            low: np.ndarray = act_space.low.astype(np.float32)
            high: np.ndarray = act_space.high.astype(np.float32)
            self.action_scale = (high - low) / 2.0
            self.action_bias = (high + low) / 2.0

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset the environment and return the initial observation.

        On the first call, passes the seed to the environment for
        reproducibility. Subsequent calls reset without a seed.

        For image environments, initialises the frame deque by filling
        all slots with copies of the first preprocessed frame.

        Returns:
            Initial observation as float32 numpy array of shape
            self.state_shape.
        """
        if self._first_reset:
            obs, info = self.env.reset(seed=self.seed)
            self._first_reset = False
        else:
            obs, info = self.env.reset()

        if self.benchmark == "dmc_visual":
            # Get the rendered pixel frame (env was created with render_mode='rgb_array')
            raw_frame = self.env.render()
            if raw_frame is None:
                # Fallback: try to use obs directly if it's already an image
                raw_frame = obs if isinstance(obs, np.ndarray) and obs.ndim == 3 else None
            if raw_frame is None:
                raise RuntimeError(
                    "DMC visual environment did not return a pixel observation. "
                    "Ensure render_mode='rgb_array' is set."
                )
            processed_frame: np.ndarray = self._preprocess_dmc_visual(raw_frame)
            # Fill deque with frame_stack copies of the initial frame
            assert self._frames is not None
            self._frames.clear()
            for _ in range(self.frame_stack):
                self._frames.append(processed_frame.copy())
            return self._stack_frames_dmc_visual()

        elif self.benchmark == "atari":
            processed_frame = self._preprocess_atari(obs)
            # Fill deque with frame_stack copies of the initial frame
            assert self._frames is not None
            self._frames.clear()
            for _ in range(self.frame_stack):
                self._frames.append(processed_frame.copy())
            return self._stack_frames_atari()

        else:
            # Vector observation (Gym, DMC-proprio)
            return obs.astype(np.float32)

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """Apply an action and return the resulting transition.

        Handles action repeat, action scaling/conversion, frame construction
        for image environments, and merging of terminated/truncated signals.

        Args:
            action: Action array. For continuous environments: shape
                (action_dim,) in [-1, 1] normalized space. For discrete
                environments: one-hot array of shape (action_dim,).

        Returns:
            Tuple of:
                next_obs (np.ndarray): Next observation, shape self.state_shape,
                    dtype float32.
                reward (float): Accumulated reward over action_repeat steps.
                done (bool): True if the episode has ended (terminated or
                    truncated).
                info (dict): Info dict from the last environment step.
        """
        # Convert action to environment format
        env_action = self._convert_action(action)

        accumulated_reward: float = 0.0
        done: bool = False
        info: Dict = {}
        last_obs: Optional[np.ndarray] = None

        if self.benchmark == "atari":
            # Atari: collect all 4 raw frames, max-pool last 2
            raw_frames_in_repeat: list = []
            for step_idx in range(self.action_repeat):
                obs, reward, terminated, truncated, info = self.env.step(env_action)
                accumulated_reward += float(reward)
                raw_frames_in_repeat.append(obs)
                done = bool(terminated or truncated)
                if done:
                    # Pad remaining frames with the last frame
                    while len(raw_frames_in_repeat) < self.action_repeat:
                        raw_frames_in_repeat.append(obs)
                    break

            # Max-pool the last 2 raw frames of the action repeat block
            pooled_obs = self._max_pool_frames(
                raw_frames_in_repeat[-2], raw_frames_in_repeat[-1]
            )
            processed_frame = self._preprocess_atari(pooled_obs)
            assert self._frames is not None
            self._frames.append(processed_frame)
            next_obs = self._stack_frames_atari()

        elif self.benchmark == "dmc_visual":
            # DMC visual: repeat action, use rendered pixel frame
            for _ in range(self.action_repeat):
                obs, reward, terminated, truncated, info = self.env.step(env_action)
                accumulated_reward += float(reward)
                done = bool(terminated or truncated)
                if done:
                    break

            # Get the rendered pixel frame after action repeat
            raw_frame = self.env.render()
            if raw_frame is None:
                raw_frame = obs if isinstance(obs, np.ndarray) and obs.ndim == 3 else None
            if raw_frame is None:
                raise RuntimeError(
                    "DMC visual environment did not return a pixel observation "
                    "during step(). Ensure render_mode='rgb_array' is set."
                )
            processed_frame = self._preprocess_dmc_visual(raw_frame)
            assert self._frames is not None
            self._frames.append(processed_frame)
            next_obs = self._stack_frames_dmc_visual()

        else:
            # Vector observation (Gym repeat=1, DMC-proprio repeat=2)
            for _ in range(self.action_repeat):
                obs, reward, terminated, truncated, info = self.env.step(env_action)
                accumulated_reward += float(reward)
                last_obs = obs
                done = bool(terminated or truncated)
                if done:
                    break
            next_obs = last_obs.astype(np.float32)  # type: ignore[union-attr]

        return next_obs, accumulated_reward, done, info

    def sample_action(self) -> np.ndarray:
        """Sample a random action in the agent's normalized action space.

        For continuous environments: samples uniformly from [-1, 1]^action_dim.
        For discrete environments: samples a random integer and returns its
        one-hot encoding.

        Returns:
            Random action as float32 numpy array of shape (action_dim,).
        """
        if self.discrete:
            # Sample random integer action, convert to one-hot
            random_int: int = int(self.env.action_space.sample())
            one_hot: np.ndarray = np.zeros(self.action_dim, dtype=np.float32)
            one_hot[random_int] = 1.0
            return one_hot
        else:
            # Sample uniformly from [-1, 1] (agent's normalized action space)
            return np.random.uniform(
                -1.0, 1.0, size=(self.action_dim,)
            ).astype(np.float32)

    def close(self) -> None:
        """Close the underlying environment and release resources."""
        self.env.close()

    # ------------------------------------------------------------------
    # Action conversion
    # ------------------------------------------------------------------

    def _convert_action(self, action: np.ndarray) -> np.ndarray:
        """Convert agent action to environment-native format.

        For continuous environments: scales from [-1, 1] to the environment's
        native action range [low, high] and clips to valid bounds.

        For discrete environments: converts one-hot vector to integer index.

        Args:
            action: Agent action. Continuous: shape (action_dim,) in [-1, 1].
                Discrete: one-hot shape (action_dim,).

        Returns:
            Environment-native action. Continuous: float32 array in [low, high].
                Discrete: integer scalar.
        """
        if self.discrete:
            # Convert one-hot to integer
            return np.array(int(np.argmax(action)))
        else:
            # Scale from [-1, 1] to [low, high]
            assert self.action_scale is not None
            assert self.action_bias is not None
            env_action: np.ndarray = (
                action.astype(np.float32) * self.action_scale + self.action_bias
            )
            # Clip to valid action bounds
            low: np.ndarray = self.env.action_space.low
            high: np.ndarray = self.env.action_space.high
            return np.clip(env_action, low, high).astype(np.float32)

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------

    def _preprocess_dmc_visual(self, obs: np.ndarray) -> np.ndarray:
        """Preprocess a DMC visual observation frame.

        Resizes the RGB frame to 84x84 and converts from HWC to CHW format.
        Pixel values remain in [0, 255] as float32; normalization is done
        inside the CNN forward pass.

        Args:
            obs: Raw RGB frame of shape (H, W, 3), dtype uint8.

        Returns:
            Preprocessed frame of shape (3, 84, 84), dtype float32.
        """
        # Resize to 84x84 using bilinear interpolation
        resized: np.ndarray = cv2.resize(
            obs,
            (self._IMAGE_SIZE, self._IMAGE_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )
        # Convert HWC (84, 84, 3) → CHW (3, 84, 84)
        chw: np.ndarray = np.transpose(resized, (2, 0, 1))
        return chw.astype(np.float32)

    def _preprocess_atari(self, obs: np.ndarray) -> np.ndarray:
        """Preprocess a single Atari frame.

        Converts RGB to grayscale and resizes to 84x84. Pixel values remain
        in [0, 255] as float32.

        Args:
            obs: Raw RGB frame of shape (H, W, 3) or grayscale (H, W),
                dtype uint8.

        Returns:
            Preprocessed grayscale frame of shape (84, 84), dtype float32.
        """
        if obs.ndim == 3 and obs.shape[2] == 3:
            # Convert RGB to grayscale
            gray: np.ndarray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        elif obs.ndim == 2:
            gray = obs
        else:
            # Handle unexpected formats gracefully
            gray = obs[..., 0] if obs.ndim == 3 else obs

        # Resize to 84x84 using area interpolation (standard for downsampling)
        resized: np.ndarray = cv2.resize(
            gray,
            (self._IMAGE_SIZE, self._IMAGE_SIZE),
            interpolation=cv2.INTER_AREA,
        )
        return resized.astype(np.float32)

    def _max_pool_frames(
        self, frame1: np.ndarray, frame2: np.ndarray
    ) -> np.ndarray:
        """Compute element-wise maximum of two raw Atari frames.

        This implements the frame max-pooling described in Appendix B.3:
            o_j = max(f_{4j+2}, f_{4j+3})

        Applied to raw frames before grayscale conversion and resizing.

        Args:
            frame1: First raw frame, shape (H, W, 3) or (H, W), uint8.
            frame2: Second raw frame, same shape as frame1, uint8.

        Returns:
            Element-wise maximum frame, same shape as inputs.
        """
        return np.maximum(frame1, frame2)

    # ------------------------------------------------------------------
    # Frame stacking
    # ------------------------------------------------------------------

    def _stack_frames_dmc_visual(self) -> np.ndarray:
        """Stack the DMC visual frame deque into a single observation.

        Concatenates 3 CHW frames along the channel dimension:
            3 frames × 3 channels = 9 channels total.

        Returns:
            Stacked observation of shape (9, 84, 84), dtype float32.
        """
        assert self._frames is not None
        # Each frame: (3, 84, 84) → concatenate along axis 0 → (9, 84, 84)
        return np.concatenate(list(self._frames), axis=0)

    def _stack_frames_atari(self) -> np.ndarray:
        """Stack the Atari frame deque into a single observation.

        Stacks 4 grayscale frames along a new leading dimension:
            4 frames × (84, 84) → (4, 84, 84).

        Returns:
            Stacked observation of shape (4, 84, 84), dtype float32.
        """
        assert self._frames is not None
        # Each frame: (84, 84) → stack along axis 0 → (4, 84, 84)
        return np.stack(list(self._frames), axis=0)

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a human-readable description of the wrapper.

        Returns:
            String with key wrapper properties.
        """
        return (
            f"EnvWrapper("
            f"env_name={self.env_name!r}, "
            f"benchmark={self.benchmark!r}, "
            f"seed={self.seed}, "
            f"state_shape={self.state_shape}, "
            f"action_dim={self.action_dim}, "
            f"discrete={self.discrete}, "
            f"image_obs={self.image_obs}, "
            f"action_repeat={self.action_repeat}, "
            f"frame_stack={self.frame_stack}"
            f")"
        )
