```python
import gym
import numpy as np
import collections
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

# Conditional imports for DeepMind Control Suite
try:
    import dm_control.suite
    from dm_env import specs
except ImportError:
    dm_control = None
    specs = None
    print("Warning: dm_control not installed. DMC environments will not be available.")

# Conditional imports for DeepMind Lab
try:
    import deepmind_lab
except ImportError:
    deepmind_lab = None
    print("Warning: deepmind_lab not installed. DMLab environments will not be available.")

from skimage.transform import resize

# Assuming config.py is in the same directory or accessible via sys.path
from config import Config


# --- Helper Wrappers ---

class DMControlWrapper(gym.Env):
    """
    A wrapper for DeepMind Control Suite environments to conform to the OpenAI Gym API.
    Handles state vector or pixel observations and normalizes pixel values.
    """
    def __init__(self, env: Any, pixel_based: bool, pixel_img_size: Optional[Tuple[int, int]] = None):
        """
        Initializes the DMControlWrapper.

        Args:
            env (Any): The dm_control environment instance.
            pixel_based (bool): If True, extract 'pixels' observation; otherwise, flatten state observations.
            pixel_img_size (Optional[Tuple[int, int]]): The (height, width) for pixel observations, used for rendering.
        """
        self._env = env
        self._pixel_based = pixel_based
        self._pixel_img_size = pixel_img_size if pixel_img_size is not None else (240, 320) # Default for rendering

        # Determine observation space
        if self._pixel_based:
            obs_spec = self._env.observation_spec().get('pixels')
            if obs_spec is None:
                raise ValueError("DMC environment does not provide 'pixels' observation for pixel_based=True.")
            # DMControl pixels are (H, W, 3) uint8. We normalize them to [0, 1] float32.
            self.observation_space = gym.spaces.Box(
                low=0.0, high=1.0, shape=(obs_spec.shape[0], obs_spec.shape[1], obs_spec.shape[2]),
                dtype=np.float32
            )
        else:
            # Concatenate all state observations into a single vector
            flat_obs_dim = 0
            for k, v in self._env.observation_spec().items():
                if k != 'pixels':
                    if isinstance(v, specs.Array):
                        flat_obs_dim += np.prod(v.shape)
                    else:
                        raise TypeError(f"Unsupported observation spec type for key {k}: {type(v)}")

            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(flat_obs_dim,), dtype=np.float32
            )

        # Determine action space
        action_spec = self._env.action_spec()
        self.action_space = gym.spaces.Box(
            low=action_spec.minimum.astype(np.float32),
            high=action_spec.maximum.astype(np.float32),
            shape=action_spec.shape,
            dtype=np.float32
        )

        self.reward_range = (-float('inf'), float('inf'))
        self.metadata = {'render_modes': ['rgb_array']}
        self._last_timestep: Optional[Any] = None # Stores last timestep for rendering if needed

    def _extract_obs(self, timestep: Any) -> np.ndarray:
        """Extracts and formats observation from a dm_env.TimeStep."""
        if self._pixel_based:
            obs = timestep.observation['pixels']
            # Normalize uint8 to float32 [0, 1]
            obs = obs.astype(np.float32) / 255.0
            return obs # HWC, float32 [0,1]
        else:
            # Flatten all state observations into a single vector
            obs_parts: List[np.ndarray] = []
            for k, v in timestep.observation.items():
                if k != 'pixels':
                    obs_parts.append(v.ravel()) # Flatten any non-1D arrays
            return np.concatenate(obs_parts).astype(np.float32)

    def reset(self, **kwargs) -> np.ndarray:
        """Resets the environment."""
        timestep = self._env.reset()
        self._last_timestep = timestep
        return self._extract_obs(timestep)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Takes an action in the environment."""
        # Ensure action dtype matches env's expectation
        action = action.astype(self.action_space.dtype)
        timestep = self._env.step(action)
        self._last_timestep = timestep
        observation = self._extract_obs(timestep)
        reward = timestep.reward if timestep.reward is not None else 0.0
        done = timestep.last()
        info: Dict[str, Any] = {}
        return observation, reward, done, info

    def render(self, mode: str = 'rgb_array') -> np.ndarray:
        """Renders the environment."""
        if mode == 'rgb_array':
            return self._env.physics.render(camera_id=0, height=self._pixel_img_size[0], width=self._pixel_img_size[1])
        raise NotImplementedError(f"Render mode {mode} not supported for DMControlWrapper.")

    def close(self) -> None:
        """Closes the environment."""
        if hasattr(self._env, 'close'):
            self._env.close()


class DMLabWrapper(gym.Env):
    """
    A wrapper for DeepMind Lab environments to conform to the OpenAI Gym API.
    Handles observations, action mapping, and implements the "Noisy-TV" problem.
    """
    # Action mapping for 9 discrete actions as often used in DMLab Gym wrappers.
    # This is an interpretation given the paper states "9 discrete actions".
    # DMLab `action_space` parameter defines available controls for continuous input.
    # We map our discrete actions to specific values for these controls.
    DMLAB_DISCRETE_ACTIONS: List[Dict[str, Union[float, int]]] = [
        # Action 0: No-op
        {'MOVE_BACK_LOOK_ARC_DEG_LEFT_RIGHT': 0, 'MOVE_BACK_LOOK_ARC_DEG_UP_DOWN': 0,
         'MOVE_BACK_VELOCITY_FORWARD': 0, 'MOVE_BACK_VELOCITY_STRAFE': 0,
         'FIRE': 0, 'JUMP': 0, 'CROUCH': 0},
        # Action 1: Look Left
        {'MOVE_BACK_LOOK_ARC_DEG_LEFT_RIGHT': -90, 'MOVE_BACK_LOOK_ARC_DEG_UP_DOWN': 0,
         'MOVE_BACK_VELOCITY_FORWARD': 0, 'MOVE_BACK_VELOCITY_STRAFE': 0,
         'FIRE': 0, 'JUMP': 0, 'CROUCH': 0},
        # Action 2: Look Right
        {'MOVE_BACK_LOOK_ARC_DEG_LEFT_RIGHT': 90, 'MOVE_BACK_LOOK_ARC_DEG_UP_DOWN': 0,
         'MOVE_BACK_VELOCITY_FORWARD': 0, 'MOVE_BACK_VELOCITY_STRAFE': 0,
         'FIRE': 0, 'JUMP': 0, 'CROUCH': 0},
        # Action 3: Move Forward
        {'MOVE_BACK_LOOK_ARC_DEG_LEFT_RIGHT': 0, 'MOVE_BACK_LOOK_ARC_DEG_UP_DOWN': 0,
         'MOVE_BACK_VELOCITY_FORWARD': 1, 'MOVE_BACK_VELOCITY_STRAFE': 0,
         'FIRE': 0, 'JUMP': 0, 'CROUCH': 0},
        # Action 4: Move Backward
        {'MOVE_BACK_LOOK_ARC_DEG_LEFT_RIGHT': 0, 'MOVE_BACK_LOOK_ARC_DEG_UP_DOWN': 0,
         'MOVE_BACK_VELOCITY_FORWARD': -1, 'MOVE_BACK_VELOCITY_STRAFE': 0,
         'FIRE': 0, 'JUMP': 0, 'CROUCH': 0},
        # Action 5: Strafe Left
        {'MOVE_BACK_LOOK_ARC_DEG_LEFT_RIGHT': 0, 'MOVE_BACK_LOOK_ARC_DEG_UP_DOWN': 0,
         'MOVE_BACK_VELOCITY_FORWARD': 0, 'MOVE_BACK_VELOCITY_STRAFE': -1,
         'FIRE': 0, 'JUMP': 0, 'CROUCH': 0},
        # Action 6: Strafe Right
        {'MOVE_BACK_LOOK_ARC_DEG_LEFT_RIGHT': 0, 'MOVE_BACK_LOOK_ARC_DEG_UP_DOWN': 0,
         'MOVE_BACK_VELOCITY_FORWARD': 0, 'MOVE_BACK_VELOCITY_STRAFE': 1,
         'FIRE': 0, 'JUMP': 0, 'CROUCH': 0},
        # Action 7: Fire
        {'MOVE_BACK_LOOK_ARC_DEG_LEFT_RIGHT': 0, 'MOVE_BACK_LOOK_ARC_DEG_UP_DOWN': 0,
         'MOVE_BACK_VELOCITY_FORWARD': 0, 'MOVE_BACK_VELOCITY_STRAFE': 0,
         'FIRE': 1, 'JUMP': 0, 'CROUCH': 0},
        # Action 8: Jump
        {'MOVE_BACK_LOOK_ARC_DEG_LEFT_RIGHT': 0, 'MOVE_BACK_LOOK_ARC_DEG_UP_DOWN': 0,
         'MOVE_BACK_VELOCITY_FORWARD': 0, 'MOVE_BACK_VELOCITY_STRAFE': 0,
         'FIRE': 0, 'JUMP': 1, 'CROUCH': 0},
    ]

    def __init__(self, env: Any, pixel_img_size: Tuple[int, int], noisy_tv: bool = True):
        """
        Initializes the DMLabWrapper.

        Args:
            env (Any): The deepmind_lab.LabEnvironment instance.
            pixel_img_size (Tuple[int, int]): The (height, width) for RGB observations (e.g., 84x84).
            noisy_tv (bool): If True, apply the Noisy-TV stochasticity to observations.
        """
        self._env = env
        self._pixel_img_size = pixel_img_size
        self._noisy_tv = noisy_tv
        
        self.action_space = gym.spaces.Discrete(len(self.DMLAB_DISCRETE_ACTIONS))

        # DMLab observations are RGB images, normalized to [0, 1] float32 after processing.
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self._pixel_img_size[0], self._pixel_img_size[1], 3),
            dtype=np.float32
        )

        self.reward_range = (-float('inf'), float('inf'))
        self.metadata = {'render_modes': ['rgb_array']}

        self._last_obs: Optional[np.ndarray] = None # Store last observation for rendering

    def _get_obs(self) -> np.ndarray:
        """Retrieves and processes the current observation, applying Noisy-TV if enabled."""
        obs = self._env.observations().get('RGB_INTERLACED', None)
        if obs is None:
            raise ValueError("DMLab environment did not provide 'RGB_INTERLACED' observation.")

        # DMLab observations are (H, W, C) uint8. Normalize to [0, 1] float32.
        obs = obs.astype(np.float32) / 255.0
        
        if self._noisy_tv:
            # Apply Noisy-TV: lower right 42x42 pixels of 84x84 RGB observation
            # with noise sampled uniformly from [0, 255].
            # Since obs is already normalized to [0, 1], noise should also be [0, 1].
            h, w, c = obs.shape
            # Assuming expected 84x84 input, lower right quadrant is [42:84, 42:84]
            # This is specific to the 84x84 size mentioned in the paper.
            if h >= 84 and w >= 84:
                noise_region_start_h = h // 2 # 42
                noise_region_start_w = w // 2 # 42
                noise_region_end_h = h # 84
                noise_region_end_w = w # 84
                
                noise_shape = (noise_region_end_h - noise_region_start_h,
                               noise_region_end_w - noise_region_start_w,
                               c)
                
                # Generate uniform noise in [0, 1]
                noise = np.random.uniform(low=0.0, high=1.0, size=noise_shape).astype(np.float32)
                
                obs[noise_region_start_h:noise_region_end_h,
                    noise_region_start_w:noise_region_end_w, :] = noise
            else:
                print(f"Warning: Noisy-TV not applied. Observation size {h}x{w} is smaller than expected 84x84 for noise region.")
        
        self._last_obs = obs
        return obs

    def reset(self, **kwargs) -> np.ndarray:
        """Resets the environment."""
        self._env.reset()
        return self._get_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Takes a discrete action in the environment."""
        if not (0 <= action < len(self.DMLAB_DISCRETE_ACTIONS)):
            raise ValueError(f"Action {action} is out of bounds for DMLab discrete action space (0-{len(self.DMLAB_DISCRETE_ACTIONS)-1}).")

        # Get the DMLab command dictionary for the selected discrete action
        lab_action_dict = self.DMLAB_DISCRETE_ACTIONS[action]
        
        # Convert dictionary values to np.intc as DMLab expects
        # DMLab typically wants integer types for discrete controls and float for continuous.
        # Here we assume all these values are to be passed as np.intc for simplicity of discrete mapping.
        # This aligns with common DMLab PPO setups where 1/-1 movement is passed as integer.
        lab_action_dict_typed = {k: np.array(v, dtype=np.intc) for k, v in lab_action_dict.items()}

        reward = self._env.step(lab_action_dict_typed)
        observation = self._get_obs()
        done = not self._env.is_running()
        info: Dict[str, Any] = {}
        return observation, reward, done, info

    def render(self, mode: str = 'rgb_array') -> Optional[np.ndarray]:
        """Renders the environment."""
        if mode == 'rgb_array' and self._last_obs is not None:
            # `_last_obs` is normalized float32. Convert back to uint8 for rendering.
            return (self._last_obs * 255).astype(np.uint8)
        raise NotImplementedError(f"Render mode {mode} not supported or no observation available for DMLabWrapper.")

    def close(self) -> None:
        """Closes the environment."""
        self._env.close()


class ActionRepeatWrapper(gym.Wrapper):
    """
    Repeats the given action for `amount` steps, sums rewards,
    and returns the final observation and done status.
    """
    def __init__(self, env: gym.Env, amount: int):
        """
        Initializes the ActionRepeatWrapper.

        Args:
            env (gym.Env): The environment to wrap.
            amount (int): The number of times to repeat the action.
        """
        super().__init__(env)
        self._amount = amount

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Steps the environment by repeating the action."""
        total_reward: float = 0.0
        done: bool = False
        info: Dict[str, Any] = {}
        obs: np.ndarray = self.env.observation_space.sample() # Initialize with a dummy observation

        for _ in range(self._amount):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class ResizeObservationWrapper(gym.ObservationWrapper):
    """Resizes image observations to a specified (height, width)."""
    def __init__(self, env: gym.Env, shape: Tuple[int, int]):
        """
        Initializes the ResizeObservationWrapper.

        Args:
            env (gym.Env): The environment to wrap.
            shape (Tuple[int, int]): The target (height, width) for observations.
        """
        super().__init__(env)
        if not isinstance(shape, tuple) or len(shape) != 2:
            raise ValueError(f"Shape must be (height, width), got {shape}")
        self._shape = shape

        # Update observation space shape
        original_shape = self.observation_space.shape
        if len(original_shape) == 3: # Assuming HWC (Height, Width, Channel) input
            new_shape = (*self._shape, original_shape[2])
        elif len(original_shape) == 2: # Assuming HW (Height, Width) grayscale input
            new_shape = self._shape
        else:
            raise ValueError(f"Unsupported observation space shape for resizing: {original_shape}. Expected 2D or 3D.")

        self.observation_space = gym.spaces.Box(
            low=self.observation_space.low.min(), # Keep original low/high, assuming normalization is handled
            high=self.observation_space.high.max(),
            shape=new_shape,
            dtype=np.float32 # Ensure output dtype is float32
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Resizes the observation."""
        if observation.shape[:2] != self._shape:
            # `resize` expects (H, W, C) or (H, W). `preserve_range=False` for float output.
            resized_obs = resize(observation, self.observation_space.shape, anti_aliasing=True, preserve_range=False)
            return resized_obs.astype(np.float32)
        return observation.astype(np.float32) # Ensure dtype is float32


class ChannelFirstWrapper(gym.ObservationWrapper):
    """Transposes image observations from (H, W, C) to (C, H, W)."""
    def __init__(self, env: gym.Env):
        """
        Initializes the ChannelFirstWrapper.

        Args:
            env (gym.Env): The environment to wrap.
        """
        super().__init__(env)
        original_shape = self.observation_space.shape
        if len(original_shape) != 3:
            raise ValueError(f"Expected 3D observation (H, W, C), got {original_shape} for ChannelFirstWrapper.")
        
        new_shape = (original_shape[2], original_shape[0], original_shape[1])
        self.observation_space = gym.spaces.Box(
            low=self.observation_space.low.min(),
            high=self.observation_space.high.max(),
            shape=new_shape,
            dtype=self.observation_space.dtype
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Transposes the observation."""
        return np.transpose(observation, (2, 0, 1))


class FrameStackWrapper(gym.ObservationWrapper):
    """
    Stacks the last `num_stack` observations along the channel dimension.
    Assumes observations are already in (C, H, W) format.
    """
    def __init__(self, env: gym.Env, num_stack: int):
        """
        Initializes the FrameStackWrapper.

        Args:
            env (gym.Env): The environment to wrap.
            num_stack (int): The number of frames to stack.
        """
        super().__init__(env)
        self._num_stack = num_stack
        self._frames: collections.deque = deque([], maxlen=num_stack)

        original_shape = self.observation_space.shape
        if len(original_shape) != 3: # Expected (C, H, W) after ChannelFirstWrapper
            raise ValueError(f"Expected 3D observation (C, H, W) for FrameStack, got {original_shape}.")

        new_shape = (original_shape[0] * num_stack, original_shape[1], original_shape[2])
        self.observation_space = gym.spaces.Box(
            low=self.observation_space.low.min(),
            high=self.observation_space.high.max(),
            shape=new_shape,
            dtype=self.observation_space.dtype
        )

    def _get_observation(self) -> np.ndarray:
        """Returns the stacked observation."""
        assert len(self._frames) == self._num_stack, "Frame stack not full during _get_observation call."
        return np.concatenate(list(self._frames), axis=0)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Steps the environment and stacks the new frame."""
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_observation(), reward, done, info

    def reset(self, **kwargs) -> np.ndarray:
        """Resets the environment and fills the frame stack with the initial observation."""
        obs = self.env.reset(**kwargs)
        # Fill deque with initial observation
        for _ in range(self._num_stack):
            self._frames.append(obs)
        return self._get_observation()


# --- Environment Manager ---

class EnvironmentManager:
    """
    Manages and provides a unified interface for different reinforcement learning environments
    (DeepMind Control Suite, OpenAI Gym, DMLab) with configurable pre-processing.
    """
    def __init__(self, config: Config, seed: int):
        """
        Initializes the environment manager.

        Args:
            config (Config): An instance of the Config class containing environment settings.
            seed (int): An integer for seeding the environment's randomness.
        """
        self._config = config
        self._seed = seed

        env_name: str = config.get_hyperparam('environment.name')
        env_suite: str = config.get_hyperparam('environment.suite')
        pixel_based: bool = config.get_hyperparam('environment.pixel_based')
        pixel_img_size: Tuple[int, int] = tuple(config.get_hyperparam('environment.pixel_img_size'))
        pixel_stack_frames: int = config.get_hyperparam('environment.pixel_stack_frames')
        action_repeat: int = config.get_hyperparam('environment.action_repeat')
        
        self.env: gym.Env # Type hint for the wrapped gym environment

        if env_suite == "DMC":
            if dm_control is None:
                raise ImportError("dm_control is not installed, but required for DMC environments.")
            
            domain_name, task_name = env_name.split('-', 1)
            raw_env: Any = dm_control.suite.load(
                domain_name=domain_name,
                task_name=task_name,
                task_kwargs={'random': seed},
                # environment_kwargs={'flat_observation': not pixel_based} # This flag handles observation structure
            )
            self.env = DMControlWrapper(raw_env, pixel_based, pixel_img_size)

        elif env_suite == "OpenAI_Gym":
            raw_env = gym.make(env_name)
            raw_env.seed(seed)
            self.env = raw_env # Directly use gym env as base
            
            # For state-based Gym envs, ensure observation is float32
            if not pixel_based and self.env.observation_space.dtype != np.float32:
                class EnsureFloat32Wrapper(gym.ObservationWrapper):
                    def __init__(self, env_: gym.Env): 
                        super().__init__(env_)
                        # Update observation_space dtype without changing shape/bounds
                        self.observation_space = gym.spaces.Box(
                            low=self.observation_space.low,
                            high=self.observation_space.high,
                            shape=self.observation_space.shape,
                            dtype=np.float32
                        )
                    def observation(self, obs: np.ndarray) -> np.ndarray: return obs.astype(np.float32)
                self.env = EnsureFloat32Wrapper(self.env)


        elif env_suite == "DMLab":
            if deepmind_lab is None:
                raise ImportError("deepmind_lab is not installed, but required for DMLab environments.")
            
            # DMLab level configuration from paper Appendix A.2
            # "contributed/dmlab/rooms_with_bad_motives/sparse_reward_maze"
            dmlab_level_id = "contributed/dmlab/rooms_with_bad_motives/sparse_reward_maze"

            dmlab_config: Dict[str, str] = {
                'fps': '30',
                'width': f'{pixel_img_size[1]}',
                'height': f'{pixel_img_