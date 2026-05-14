import gymnasium
import numpy as np
from collections import deque
from typing import Tuple, Dict, Any, Optional, Union
import random
import torch
from torchvision import transforms
from PIL import Image

from config import Config


class EnvironmentWrapper:
    """Provides a standardized interface for interacting with various Gymnasium
    environments, handling environment-specific preprocessing and frame stacking.
    """

    def __init__(self, config: Config, env_name: str, seed: int):
        """Initializes the EnvironmentWrapper.

        Args:
            config: Configuration object containing environment-related hyperparameters.
            env_name: The name of the Gymnasium environment to wrap.
            seed: The random seed for the environment.
        """
        self.config = config
        self.env_name = env_name
        self.seed = seed

        self.action_repeat = config.environment.action_repeat
        self.image_obs = config.environment.image_obs
        self.discrete_actions = config.environment.discrete_actions
        self.frame_stack = config.environment.frame_stack

        # Determine environment type for specific preprocessing logic
        self.is_atari = "ALE/" in env_name or "atari" in env_name.lower()
        # DMC visual is any image_obs that is not Atari
        self.is_dmc_visual = self.image_obs and not self.is_atari

        # 1. Environment Initialization
        if self.is_atari:
            # For Atari, frameskip=1 is crucial to obtain individual raw frames,
            # allowing for custom max-pooling and stacking logic described in the paper.
            self.env = gymnasium.make(
                f"ALE/{env_name}-v5",
                frameskip=1,
                repeat_action_probability=0.25,  # Handles sticky actions
                full_action_space=False,         # Standard for most Atari RL
                render_mode='rgb_array'          # Required if rendering is used
            )
        else:
            self.env = gymnasium.make(env_name, render_mode='rgb_array' if self.image_obs else None)

        # 2. Seeding
        # Set seeds for action and observation spaces
        self.env.action_space.seed(self.seed)
        self.env.observation_space.seed(self.seed)
        # Reset with seed to ensure environment internal state is consistently initialized
        # The returned obs is discarded here, actual first obs handled in self.reset()
        _ = self.env.reset(seed=self.seed) 

        # 3. Space Information Extraction (internal, then exposed by methods)
        self._action_space_info: Dict[str, Any] = self._get_action_space_info_internal()
        self._observation_space_info: Dict[str, Any] = self._get_observation_space_info_internal()

        # 4. Preprocessing Pipeline Setup
        self._setup_preprocessing_transforms()
        
        # Frame buffers
        self.atari_raw_frames_buffer: Optional[deque] = None
        self.visual_dmc_frame_stack_buffer: Optional[deque] = None

        if self.is_atari:
            # Stores 16 raw (grayscale, 84x84) frames for Atari state construction
            self.atari_raw_frames_buffer = deque(maxlen=16)
        elif self.is_dmc_visual:
            # Stores `frame_stack` (typically 3) processed (C, 84, 84) frames for DMC visual
            self.visual_dmc_frame_stack_buffer = deque(maxlen=self.frame_stack)

        # Flag for dynamically enabling terminal loss in agent
        self.has_terminal_transition_observed: bool = False 

    def _setup_preprocessing_transforms(self) -> None:
        """Sets up the image preprocessing transforms based on environment type."""
        # Common resize transform for image-based observations
        self.resize_transform = transforms.Resize((84, 84), interpolation=transforms.InterpolationMode.BILINEAR)
        self.to_tensor_transform = transforms.ToTensor()
        
        if self.is_atari:
            # Atari: Grayscale and Resize to 84x84
            self.grayscale_transform = transforms.Grayscale(num_output_channels=1)
        # For DMC Visual, we only need to resize, no grayscale.
        # Vector observations don't need these transforms.

    def _get_action_space_info_internal(self) -> Dict[str, Any]:
        """Extracts and returns information about the action space."""
        action_space = self.env.action_space
        info: Dict[str, Any] = {
            "is_discrete": isinstance(action_space, gymnasium.spaces.Discrete)
        }
        if info["is_discrete"]:
            info["action_dim"] = action_space.n
            info["low"] = 0
            info["high"] = action_space.n - 1
        else:
            info["action_dim"] = action_space.shape[0]
            info["low"] = action_space.low.min().item()
            info["high"] = action_space.high.max().item()
        return info

    def _get_observation_space_info_internal(self) -> Dict[str, Any]:
        """Calculates and returns the shape and dtype of the *processed* observation space."""
        info: Dict[str, Any] = {}
        if self.is_atari:
            # Processed Atari obs: (4, 84, 84) channels-first, uint8
            info["obs_shape"] = (4, 84, 84)
            info["obs_dim"] = None # Not a vector environment
            info["dtype"] = np.uint8
        elif self.is_dmc_visual:
            # DMC visual: frame_stack frames, each originally RGB (3 channels).
            # The processed single frame will be (C, 84, 84).
            # Concatenating `frame_stack` of these along the channel axis results in (frame_stack * C, 84, 84).
            num_channels_per_frame = self.env.observation_space.shape[-1] # e.g., 3 for RGB
            info["obs_shape"] = (self.frame_stack * num_channels_per_frame, 84, 84)
            info["obs_dim"] = None
            info["dtype"] = np.uint8
        else: # Vector observations
            info["obs_shape"] = self.env.observation_space.shape # e.g., (obs_dim,)
            info["obs_dim"] = self.env.observation_space.shape[0] if len(self.env.observation_space.shape) == 1 else None
            info["dtype"] = self.env.observation_space.dtype
        return info

    def _preprocess_raw_frame(self, raw_observation: np.ndarray) -> np.ndarray:
        """Processes a single raw observation from the environment.

        Args:
            raw_observation: A raw NumPy array observation (HxWxC for images).

        Returns:
            A processed NumPy array frame (HxW for Atari, CxHxW for DMC visual, or original for vector).
        """
        if self.image_obs:
            # Convert NumPy array (H, W, C) to PIL Image
            pil_image = Image.fromarray(raw_observation)
            
            if self.is_atari:
                # Apply Grayscale then Resize. Output: (1, 84, 84) Tensor
                transformed_tensor = self.to_tensor_transform(self.resize_transform(self.grayscale_transform(pil_image)))
                # Convert back to NumPy, remove channel dimension for Atari buffer (stores HxW frames)
                return (transformed_tensor.squeeze(0).numpy() * 255).astype(np.uint8)
            elif self.is_dmc_visual:
                # Only Resize. Output: (C, 84, 84) Tensor (e.g., (3, 84, 84) for RGB)
                transformed_tensor = self.to_tensor_transform(self.resize_transform(pil_image))
                # Return channels-first NumPy array (C, H, W)
                return (transformed_tensor.numpy() * 255).astype(np.uint8)
        
        # For vector observations, return as is.
        return raw_observation


    def _process_atari_stacked_observation(self) -> np.ndarray:
        """Constructs the 4-channel stacked observation for Atari based on the 16-frame buffer.

        Requires self.atari_raw_frames_buffer to contain 16 (84,84) grayscale frames.

        Returns:
            A (4, 84, 84) NumPy array representing the stacked observation, dtype=np.uint8.
        """
        if self.atari_raw_frames_buffer is None or len(self.atari_raw_frames_buffer) < 16:
            raise ValueError(f"Atari frame buffer not filled with 16 frames for stacking. Current size: {len(self.atari_raw_frames_buffer) if self.atari_raw_frames_buffer else 0}")

        # Access frames by converting deque to list for consistent indexing
        frames = list(self.atari_raw_frames_buffer) 

        # Apply max-pooling logic as per Appendix B.3 diagram
        o_0 = np.maximum(frames[2], frames[3])
        o_1 = np.maximum(frames[6], frames[7])
        o_2 = np.maximum(frames[10], frames[11])
        o_3 = np.maximum(frames[14], frames[15])

        # Stack the four 84x84 observations into a (4, 84, 84) array (channels-first)
        stacked_obs = np.stack([o_0, o_1, o_2, o_3], axis=0)
        return stacked_obs.astype(np.uint8)

    def reset(self) -> np.ndarray:
        """Resets the environment and returns the initial processed observation.

        Returns:
            The initial processed observation as a NumPy array.
        """
        raw_obs, info = self.env.reset(seed=self.seed)

        if self.is_atari:
            if self.atari_raw_frames_buffer is None:
                self.atari_raw_frames_buffer = deque(maxlen=16)
            self.atari_raw_frames_buffer.clear()

            # Collect 16 frames for the initial state as per Appendix B.3
            # f_0 is from reset(), f_1 to f_15 are from subsequent 15 no-op steps.
            current_raw_obs = raw_obs
            for i in range(16):
                self.atari_raw_frames_buffer.append(self._preprocess_raw_frame(current_raw_obs))
                if i < 15: # No-op for the next 15 frames
                    current_raw_obs, _, term, trunc, _ = self.env.step(0)
                    if term or trunc:
                        # If episode terminates prematurely during no-ops, reset and try again
                        # to ensure a full 16-frame sequence for the initial state.
                        return self.reset() # Recursive call to reset until a valid sequence is found
            
            return self._process_atari_stacked_observation()

        elif self.is_dmc_visual:
            if self.visual_dmc_frame_stack_buffer is None:
                self.visual_dmc_frame_stack_buffer = deque(maxlen=self.frame_stack)
            self.visual_dmc_frame_stack_buffer.clear()

            processed_frame = self._preprocess_raw_frame(raw_obs)
            # Initialize buffer by repeating the first processed frame `frame_stack` times
            for _ in range(self.frame_stack):
                self.visual_dmc_frame_stack_buffer.append(processed_frame)
            
            # Stack frames along channel dimension (channels-first)
            return np.concatenate(list(self.visual_dmc_frame_stack_buffer), axis=0)

        else: # Vector observations
            return raw_obs

    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Performs an action in the environment for `action_repeat` steps.

        Args:
            action: The action to take. Can be an int for discrete or np.ndarray for continuous.

        Returns:
            A tuple: (next_obs, total_reward, done, info).
        """
        total_reward = 0.0
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}
        last_raw_obs_in_sequence: Optional[np.ndarray] = None

        for _ in range(self.action_repeat):
            raw_obs, reward, term, trunc, current_info = self.env.step(action)
            total_reward += reward
            terminated = terminated or term
            truncated = truncated or trunc
            info.update(current_info)  # Simplistic info aggregation, last info overrides previous

            if self.is_atari:
                if self.atari_raw_frames_buffer is None:
                    raise ValueError("Atari frame buffer not initialized in step().")
                self.atari_raw_frames_buffer.append(self._preprocess_raw_frame(raw_obs))
            elif self.is_dmc_visual:
                # For DMC visual, we only care about the last raw_obs of the sequence
                # to form the next stacked observation for the agent
                last_raw_obs_in_sequence = raw_obs

            if terminated or truncated:
                # If episode ends early, break from action_repeat loop
                break

        done = terminated or truncated
        if done:
            self.has_terminal_transition_observed = True # Set flag when a terminal transition occurs

        next_obs: np.ndarray
        if self.is_atari:
            next_obs = self._process_atari_stacked_observation()
        elif self.is_dmc_visual:
            if self.visual_dmc_frame_stack_buffer is None:
                raise ValueError("DMC Visual frame buffer not initialized.")
            if last_raw_obs_in_sequence is None:
                # This case should ideally not happen if action_repeat >= 1.
                # If it does, it implies action_repeat loop never ran or raw_obs was never set.
                # For robustness, we can try to fall back to the raw_obs from the previous agent step
                # if the episode terminated immediately. But usually, last_raw_obs_in_sequence
                # will hold the observation just before termination.
                raise RuntimeError("last_raw_obs_in_sequence is None in DMC visual step after loop.")
            
            # Append the last processed raw observation from the sequence of repeated actions
            # This is the next observation after the agent's full action.
            self.visual_dmc_frame_stack_buffer.append(self._preprocess_raw_frame(last_raw_obs_in_sequence))
            next_obs = np.concatenate(list(self.visual_dmc_frame_stack_buffer), axis=0) # (frame_stack * C, H, W)

        else: # Vector observations
            next_obs = raw_obs # raw_obs from the *last* step in the action_repeat loop

        return next_obs, total_reward, done, info

    def render(self) -> Any:
        """Renders the environment."""
        return self.env.render()

    def close(self) -> None:
        """Closes the environment."""
        self.env.close()

    def get_action_space_info(self) -> Dict[str, Any]:
        """Returns information about the action space."""
        return self._action_space_info

    def get_observation_space_info(self) -> Dict[str, Any]:
        """Returns information about the processed observation space."""
        return self._observation_space_info

