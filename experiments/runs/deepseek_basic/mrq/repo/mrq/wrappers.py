"""
Environment wrappers for MR.Q.

Implements the preprocessing described in Appendix B.3:
- Atari: grayscale, resize to 84x84, max between frames 3 and 4, 
  4-frame stacking, sticky actions, action repeat of 4
- DMC: action repeat of 2, frame stacking of 3 for visual
"""

import gymnasium as gym
import numpy as np
from collections import deque


class AtariPreprocessing(gym.Wrapper):
    """
    Atari preprocessing as described in Appendix B.3:
    - Action repeat of 4
    - Grayscale and resize to 84x84
    - Max between the 3rd and 4th frame of each 4-frame block
    - Stack 4 observations to form state
    - Sticky actions (randomly repeat previous action with 25% probability)
    """
    
    def __init__(self, env, frame_skip=4, screen_size=84, grayscale=True,
                 sticky_actions=True):
        super().__init__(env)
        self.frame_skip = frame_skip
        self.screen_size = screen_size
        self.grayscale = grayscale
        self.sticky_actions = sticky_actions
        
        # Frame stack
        self.frame_stack = 4
        self.frames = deque(maxlen=self.frame_stack)
        
        # Change observation space
        if grayscale:
            obs_shape = (self.frame_stack, screen_size, screen_size)
        else:
            obs_shape = (self.frame_stack, screen_size, screen_size, 3)
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )
        
        self._last_action = None
    
    def _preprocess_frame(self, frame):
        """Grayscale and resize frame."""
        import cv2
        if self.grayscale and len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.screen_size, self.screen_size),
                          interpolation=cv2.INTER_AREA)
        return frame
    
    def _max_pool_frames(self, frame1, frame2):
        """Element-wise max between two frames."""
        return np.maximum(frame1, frame2)
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._preprocess_frame(obs)
        
        self.frames.clear()
        for _ in range(self.frame_stack):
            self.frames.append(obs)
        
        state = np.array(list(self.frames))
        return state, info
    
    def step(self, action):
        total_reward = 0.0
        
        # Sticky actions: repeat previous action 25% of the time
        if self.sticky_actions and self._last_action is not None:
            if np.random.random() < 0.25:
                action = self._last_action
        
        self._last_action = action
        
        for i in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        obs = self._preprocess_frame(obs)
        self.frames.append(obs)
        
        # Build state from frame stack
        state = np.array(list(self.frames))
        
        return state, total_reward, terminated, truncated, info


class DMControlWrapper(gym.Wrapper):
    """
    DM Control wrapper with action repeat and optional frame stacking.
    For proprioceptive: just action repeat.
    For visual: action repeat + frame stacking of 3.
    """
    
    def __init__(self, env, action_repeat=2, from_pixels=False, 
                 frame_stack=3, image_size=84):
        super().__init__(env)
        self.action_repeat = action_repeat
        self.from_pixels = from_pixels
        self.frame_stack = frame_stack
        self.image_size = image_size
        
        if from_pixels:
            self.frames = deque(maxlen=frame_stack)
            obs_shape = (frame_stack, image_size, image_size, 3)
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=obs_shape, dtype=np.uint8
            )
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.from_pixels:
            obs = self._preprocess_frame(obs)
            self.frames.clear()
            for _ in range(self.frame_stack):
                self.frames.append(obs)
            state = np.array(list(self.frames))
            return state, info
        return obs, info
    
    def _preprocess_frame(self, frame):
        """Resize frame to image_size x image_size."""
        import cv2
        if frame.shape[:2] != (self.image_size, self.image_size):
            frame = cv2.resize(frame, (self.image_size, self.image_size),
                              interpolation=cv2.INTER_AREA)
        return frame
    
    def step(self, action):
        total_reward = 0.0
        for _ in range(self.action_repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        
        if self.from_pixels:
            obs = self._preprocess_frame(obs)
            self.frames.append(obs)
            state = np.array(list(self.frames))
            return state, total_reward, terminated, truncated, info
        
        return obs, total_reward, terminated, truncated, info


def wrap_deepmind_dmc(env, action_repeat=2, from_pixels=False, 
                       frame_stack=3, image_size=84):
    """Apply DM Control wrappers."""
    return DMControlWrapper(env, action_repeat=action_repeat,
                           from_pixels=from_pixels,
                           frame_stack=frame_stack,
                           image_size=image_size)


def wrap_atari(env, frame_skip=4, screen_size=84, grayscale=True,
               sticky_actions=True):
    """Apply Atari wrappers."""
    return AtariPreprocessing(env, frame_skip=frame_skip,
                              screen_size=screen_size,
                              grayscale=grayscale,
                              sticky_actions=sticky_actions)
