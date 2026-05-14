import copy
from typing import Any, Dict, Optional, Tuple, Union

import gym
import torch
import torch.nn as nn

from config import Config
from environments import EnvironmentManager
from models.policy_nets import CNNActor, CNNCritic, PolicyNetwork, QNetwork
from rl_agents.redq_agent import REDQAgent # DRQV2 extends REDQ
from utils.data_processing import random_shifts_aug # Assumed to be in utils/data_processing.py

class DRQV2Agent(REDQAgent):
    """
    DRQ-V2 agent implementation, extending REDQAgent for pixel-based tasks
    with data augmentation and specific visual encoder updates.

    This agent applies random shifts data augmentation to observations before
    passing them to the core REDQ training logic. It also includes a separate
    soft-update mechanism for the visual encoders within the critic networks.
    """

    def __init__(self, config: Config, env_manager: EnvironmentManager, device: torch.device):
        """
        Initializes the DRQ-V2 agent.

        Args:
            config (Config): Configuration object.
            env_manager (EnvironmentManager): The environment manager.
            device (torch.device): The device (CPU/GPU) to run the agent on.
        """
        self.config: Config = config
        self.env_manager: EnvironmentManager = env_manager
        self.device: torch.device = device

        # Validate configuration: DRQV2 is specifically for pixel-based environments.
        if not self.config.get_hyperparam('environment.pixel_based'):
            raise ValueError(
                "DRQV2Agent is designed exclusively for pixel-based environments. "
                "Please set 'environment.pixel_based' to True in your configuration."
            )

        # Determine state and action dimensions from the environment
        state_space: gym.Space = self.env_manager.get_observation_space()
        action_space: gym.Space = self.env_manager.get_action_space()

        # DRQV2, like SAC and REDQ, is primarily used with continuous action spaces.
        is_continuous_action: bool = isinstance(action_space, gym.spaces.Box)
        if not is_continuous_action:
            # While possible to adapt for discrete, the paper's context (DMC, OpenAI Gym)
            # suggests continuous control for SAC/REDQ/DRQ-V2 baselines.
            print("Warning: DRQV2 is typically employed with continuous action spaces. "
                  "This implementation assumes a continuous action space for the actor's output structure. "
                  "Behavior with discrete action spaces might be undefined or require further adaptation.")

        state_dim: Tuple[int, ...] = state_space.shape # Expected (C, H, W) for pixel observations
        action_dim: int = action_space.shape[0] # Dimension of continuous action vector

        # Initialize CNN-based Policy and Q-networks for pixel observations
        actor: PolicyNetwork = CNNActor(
            config=self.config, state_dim=state_dim, action_dim=action_dim, is_continuous=is_continuous_action
        ).to(device)
        
        # A placeholder CNNCritic instance for the base class constructor.
        # The REDQAgent's __init__ will use this to build its ensemble of Q-networks.
        placeholder_q_net = CNNCritic(
            config=self.config, state_dim=state_dim, action_dim=action_dim
        ).to(device)

        # Initialize the base REDQAgent. This handles the Q-network ensemble creation,
        # optimizers, and general SAC/REDQ logic.
        super().__init__(self.config, self.env_manager, actor, placeholder_q_net, device)

        # --- DRQ-V2 specific parameters ---
        # Data augmentation: padding amount for random shifts
        self.aug_pad: int = 4 # Default value
        try:
            aug_pad_cfg: Union[int, str] = self.config.get_hyperparam('drqv2.aug_pad')
            if isinstance(aug_pad_cfg, int):
                self.aug_pad = aug_pad_cfg
            elif isinstance(aug_pad_cfg, str) and aug_pad_cfg.strip().upper() == "NOT_SPECIFIED":
                pass # Use default
            else:
                raise ValueError(f"Expected 'drqv2.aug_pad' to be an integer or 'NOT_SPECIFIED', got {aug_pad_cfg}")
        except KeyError:
            print("Warning: 'drqv2.aug_pad' not found in config. Using default value (4).")
            pass # Use default

        # Visual encoder update rate (tau_encoder), potentially different from general critic tau
        self.encoder_update_tau: float = self.config.get_hyperparam('rl_agent.encoder_update_tau')
        
    def _apply_augmentations(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Applies random shifts data augmentation to a batch of pixel observations.

        Args:
            observations (torch.Tensor): Input pixel observations (N, C, H, W) as float32.

        Returns:
            torch.Tensor: Augmented pixel observations.
        """
        # Ensure observations are float32 before augmentation.
        # This is typically handled by the replay buffer or environment manager,
        # but it's good practice to ensure.
        augmented_observations: torch.Tensor = random_shifts_aug(observations.float(), self.aug_pad)
        return augmented_observations

    def train_step(
        self,
        real_batch: Dict[str, torch.Tensor],
        synthetic_batch: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """
        Performs one training step for the DRQ-V2 agent, including data augmentation
        on observations and specific visual encoder updates.

        Args:
            real_batch (Dict[str, torch.Tensor]): A dictionary of transition components
                                                  sampled from the real replay buffer.
            synthetic_batch (Optional[Dict[str, torch.Tensor]]): An optional dictionary with the same structure,
                                                                 sampled from the synthetic replay buffer.

        Returns:
            Dict[str, float]: A dictionary of training metrics.
        """
        # Apply data augmentations to states and next_states in both real and synthetic batches.
        # These operations modify the dictionaries in place, preparing them for the superclass call.
        real_batch['state'] = self._apply_augmentations(real_batch['state'])
        real_batch['next_state'] = self._apply_augmentations(real_batch['next_state'])

        if synthetic_batch is not None:
            synthetic_batch['state'] = self._apply_augmentations(synthetic_batch['state'])
            synthetic_batch['next_state'] = self._apply_augmentations(synthetic_batch['next_state'])

        # Delegate to the base REDQAgent's train_step with the augmented data.
        # This performs all standard REDQ policy and Q-value updates.
        metrics: Dict[str, float] = super().train_step(real_batch, synthetic_batch)

        # --- DRQ-V2 Specific: Soft-update visual encoders of critics ---
        # The base REDQAgent's `sync_target_networks` method updates all Q-network parameters
        # using the general `self.tau`. DRQ-V2 typically specifies a potentially different
        # `encoder_update_tau` for soft-updating only the visual encoders within the critics.
        # This update is performed here after the main REDQ training step.
        with torch.no_grad():
            for i in range(self.num_q_networks): # Iterate through all Q-networks in the ensemble
                # Access the online and target visual encoders for the i-th critic
                online_encoder: nn.Module = self.q_networks[i].encoder # Assumes CNNCritic has an 'encoder' attribute
                target_encoder: nn.Module = self.target_q_networks[i].encoder # Same for target critic
                
                # Perform soft update for the encoder parameters
                for param, target_param in zip(online_encoder.parameters(), target_encoder.parameters()):
                    target_param.data.copy_(
                        self.encoder_update_tau * param.data + (1.0 - self.encoder_update_tau) * target_param.data
                    )
        
        return metrics

    # The `get_action`, `get_policy_nets`, and `save_checkpoint`/`load_checkpoint` methods
    # inherited from `REDQAgent` are suitable for `DRQV2Agent` without further modification.
    # `get_policy_nets` will return the `CNNActor` and the first `CNNCritic` instance,
    # from which their respective `encoder` attributes can be accessed if needed (e.g., for generative model).
