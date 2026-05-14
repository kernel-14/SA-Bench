"""
Environment wrappers for MR.Q experiments.

Supports:
- Gym locomotion (MuJoCo, continuous actions, vector obs)
- DMC proprioceptive (continuous actions, vector obs)
- DMC visual (continuous actions, image obs)
- Atari (discrete actions, image obs)
"""

import numpy as np
import gymnasium as gym


# ============================================================================
# Gym Locomotion
# ============================================================================

class GymWrapper:
    """
    Wrapper for Gymnasium environments (Gym locomotion tasks).
    No preprocessing applied.
    """

    def __init__(self, env_name, seed=0):
        self.env = gym.make(env_name)
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)

        obs, _ = self.env.reset(seed=seed)
        self.state_dim = obs.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_scale = float(self.env.action_space.high[0])
        self.discrete = False
        self.image_obs = False
        self.state_channels = None

    def reset(self):
        obs, _ = self.env.reset()
        return obs.astype(np.float32)

    def step(self, action):
        # Scale action from [-1, 1] to actual range
        scaled_action = action * self.action_scale
        obs, reward, terminated, truncated, info = self.env.step(scaled_action)
        done = terminated or truncated
        return obs.astype(np.float32), float(reward), bool(done), bool(terminated), info

    def close(self):
        self.env.close()


# ============================================================================
# DMC Wrappers
# ============================================================================

class DMCWrapper:
    """
    Wrapper for DeepMind Control Suite environments.
    Uses dm_control directly.
    
    Action repeat of 2 (as in paper).
    """

    def __init__(self, domain, task, seed=0, action_repeat=2,
                 image_obs=False, image_size=84, frame_stack=3):
        try:
            from dm_control import suite
        except ImportError:
            raise ImportError("dm_control required for DMC environments. "
                              "Install with: pip install dm_control")

        self.env = suite.load(domain, task, task_kwargs={"random": seed})
        self.action_repeat = action_repeat
        self.image_obs = image_obs
        self.image_size = image_size
        self.frame_stack = frame_stack
        self.discrete = False

        # Get dimensions
        action_spec = self.env.action_spec()
        self.action_dim = action_spec.shape[0]
        self.action_scale = 1.0  # DMC actions are in [-1, 1]

        if image_obs:
            self.state_channels = frame_stack * 3  # RGB * frame_stack
            self.state_dim = None
            self._frames = np.zeros(
                (frame_stack, 3, image_size, image_size), dtype=np.uint8
            )
        else:
            obs_spec = self.env.observation_spec()
            self.state_dim = sum(
                int(np.prod(v.shape)) for v in obs_spec.values()
            )
            self.state_channels = None

    def _get_obs(self):
        """Get observation from DMC environment."""
        if self.image_obs:
            frame = self.env.physics.render(
                height=self.image_size, width=self.image_size, camera_id=0
            )
            return frame.transpose(2, 0, 1).astype(np.uint8)  # (3, H, W)
        else:
            obs = self.env.observation()
            return np.concatenate([
                v.flatten() for v in obs.values()
            ]).astype(np.float32)

    def reset(self):
        time_step = self.env.reset()
        if self.image_obs:
            frame = self._get_obs()
            for i in range(self.frame_stack):
                self._frames[i] = frame
            return self._frames.reshape(
                self.state_channels, self.image_size, self.image_size
            )
        else:
            return self._get_obs()

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.action_repeat):
            time_step = self.env.step(action)
            total_reward += time_step.reward or 0.0
            if time_step.last():
                break

        done = time_step.last()

        if self.image_obs:
            frame = self._get_obs()
            # Shift frames
            self._frames[:-1] = self._frames[1:]
            self._frames[-1] = frame
            obs = self._frames.reshape(
                self.state_channels, self.image_size, self.image_size
            )
        else:
            obs = self._get_obs()

        return obs, float(total_reward), bool(done), bool(done), {}

    def close(self):
        pass


# ============================================================================
# Atari Wrapper
# ============================================================================

class AtariWrapper:
    """
    Atari preprocessing wrapper following standard protocol:
    - Action repeat of 4
    - Grayscale + resize to 84x84
    - Max of 3rd and 4th frames (within each action repeat)
    - Frame stack of 4 observations
    - Sticky actions (p=0.25)
    
    Uses -v5 version with sticky actions.
    """

    def __init__(self, game_name, seed=0, frame_stack=4,
                 action_repeat=4, sticky_action_prob=0.25,
                 noop_max=30, image_size=84):
        try:
            import ale_py
            gym.register_envs(ale_py)
        except ImportError:
            pass

        # Use -v5 with sticky actions
        env_name = f"ALE/{game_name}-v5"
        self.env = gym.make(
            env_name,
            frameskip=1,  # We handle frame skip manually
            repeat_action_probability=sticky_action_prob,
            full_action_space=False,
        )
        self.env.action_space.seed(seed)

        self.frame_stack = frame_stack
        self.action_repeat = action_repeat
        self.image_size = image_size
        self.noop_max = noop_max

        self.action_dim = self.env.action_space.n
        self.state_channels = frame_stack  # Grayscale frames
        self.state_dim = None
        self.discrete = True
        self.image_obs = True
        self.action_scale = 1.0

        # Frame buffer for max pooling
        self._frame_buffer = np.zeros((2, image_size, image_size), dtype=np.uint8)
        # Stacked frames
        self._frames = np.zeros((frame_stack, image_size, image_size), dtype=np.uint8)

    def _preprocess_frame(self, frame):
        """Convert to grayscale and resize to 84x84."""
        try:
            import cv2
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(gray, (self.image_size, self.image_size),
                                 interpolation=cv2.INTER_AREA)
        except ImportError:
            # Fallback without cv2
            gray = np.mean(frame, axis=2).astype(np.uint8)
            # Simple resize by subsampling
            h, w = gray.shape
            resized = gray[::h//self.image_size, ::w//self.image_size]
            resized = resized[:self.image_size, :self.image_size]
        return resized.astype(np.uint8)

    def reset(self):
        obs, _ = self.env.reset()
        # Random no-ops at start
        n_noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(n_noops):
            obs, _, terminated, truncated, _ = self.env.step(0)
            if terminated or truncated:
                obs, _ = self.env.reset()

        frame = self._preprocess_frame(obs)
        for i in range(self.frame_stack):
            self._frames[i] = frame
        return self._frames.copy()

    def step(self, action):
        """
        Execute action with action repeat.
        Max pool over last 2 frames within each repeat group.
        """
        total_reward = 0.0
        done = False

        # Action repeat with max pooling over last 2 frames
        for i in range(self.action_repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            done = terminated or truncated

            # Store last 2 frames for max pooling
            if i == self.action_repeat - 2:
                self._frame_buffer[0] = self._preprocess_frame(obs)
            elif i == self.action_repeat - 1:
                self._frame_buffer[1] = self._preprocess_frame(obs)

            if done:
                break

        # Max pool over last 2 frames
        max_frame = np.maximum(self._frame_buffer[0], self._frame_buffer[1])

        # Update frame stack
        self._frames[:-1] = self._frames[1:]
        self._frames[-1] = max_frame

        return self._frames.copy(), float(total_reward), bool(done), bool(terminated), info

    def close(self):
        self.env.close()


# ============================================================================
# Factory function
# ============================================================================

def make_env(env_type, env_name, seed=0, **kwargs):
    """
    Create environment by type.
    
    Args:
        env_type: "gym", "dmc_proprio", "dmc_visual", "atari"
        env_name: Environment name
        seed: Random seed
        **kwargs: Additional arguments
    
    Returns:
        env: Wrapped environment
    """
    if env_type == "gym":
        return GymWrapper(env_name, seed=seed)
    elif env_type == "dmc_proprio":
        domain, task = env_name.split("-", 1)
        return DMCWrapper(domain, task, seed=seed, image_obs=False, **kwargs)
    elif env_type == "dmc_visual":
        domain, task = env_name.split("-", 1)
        return DMCWrapper(domain, task, seed=seed, image_obs=True, **kwargs)
    elif env_type == "atari":
        return AtariWrapper(env_name, seed=seed, **kwargs)
    else:
        raise ValueError(f"Unknown env_type: {env_type}")
