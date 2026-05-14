import collections
from typing import Dict, Optional

import torch

# Assuming these are available in the project structure
from config import Config
from replay_buffers import ReplayBuffer
from rl_agents.base_agent import RLBaseAgent
from utils.logging_utils import Logger


class RLTrainer:
    """
    Manages the training loop for the RL agent. It samples batches from real and synthetic
    replay buffers, performs policy updates, and logs relevant RL metrics. It handles the
    'update-to-data' (UTD) ratio and the synthetic data mixing ratio.
    """

    def __init__(
        self,
        config: Config,
        agent: RLBaseAgent,
        real_buffer: ReplayBuffer,
        synthetic_buffer: ReplayBuffer,
        logger: Logger,
    ):
        """
        Initializes the RLTrainer.

        Args:
            config (Config): Configuration object.
            agent (RLBaseAgent): The RL agent to be trained.
            real_buffer (ReplayBuffer): The real replay buffer (D_real).
            synthetic_buffer (ReplayBuffer): The synthetic replay buffer (D_syn).
            logger (Logger): The logger for recording metrics.
        """
        self.config: Config = config
        self.agent: RLBaseAgent = agent
        self.real_buffer: ReplayBuffer = real_buffer
        self.synthetic_buffer: ReplayBuffer = synthetic_buffer
        self.logger: Logger = logger

        self.device: torch.device = self.config.get_hyperparam("experiment.device")

        self.utd_ratio: int = self.config.get_hyperparam("rl_agent.utd_ratio")
        self.batch_size: int = self.config.get_hyperparam("rl_agent.batch_size")
        self.synthetic_data_ratio: float = self.config.get_hyperparam("pgr_loop.synthetic_data_ratio")

        # Calculate the number of real and synthetic samples to draw for each policy update
        # The actual mixing is handled by the agent's train_step.
        # Here we determine how many samples to *request* from each buffer.
        self.num_real_samples_to_request: int = int(self.batch_size * (1.0 - self.synthetic_data_ratio))
        self.num_synthetic_samples_to_request: int = self.batch_size - self.num_real_samples_to_request

        # Ensure that at least some real samples are requested if synthetic ratio is < 1.0 and batch_size > 0
        if self.synthetic_data_ratio < 1.0 and self.batch_size > 0:
            if self.num_real_samples_to_request == 0: # If it truncated to 0 but ratio isn't 1.0
                self.num_real_samples_to_request = 1 # Always request at least 1 real sample if ratio allows
                self.num_synthetic_samples_to_request = self.batch_size - 1
                if self.num_synthetic_samples_to_request < 0: # Should not happen if batch_size >= 1
                    self.num_synthetic_samples_to_request = 0
        
        # If synthetic_data_ratio is 1.0, num_real_samples_to_request can be 0.
        # If synthetic_data_ratio is 0.0, num_synthetic_samples_to_request can be 0.
        # If batch_size is 0, this might cause issues, but typically batch_size > 0.
        # The agent's train_step is designed to handle potential empty batches passed from here.


    def train_policy(self, current_env_step: int) -> Dict[str, float]:
        """
        Performs policy training for a number of steps defined by the UTD ratio.

        Args:
            current_env_step (int): The current global environment step, used for logging.

        Returns:
            Dict[str, float]: A dictionary of averaged training metrics.
        """
        metrics_accum: collections.defaultdict[str, float] = collections.defaultdict(float)
        update_count: int = 0 # Keep track of actual updates performed

        # Perform UTD_ratio policy gradient updates
        for _ in range(self.utd_ratio):
            # Check if real buffer has enough data to even attempt a training step
            if self.real_buffer.size() < self.num_real_samples_to_request:
                # If the real buffer is too small to fulfill the request for real samples,
                # we cannot proceed with this update. Break from the UTD loop.
                break # Break from UTD loop if we can't even get minimal real samples

            try:
                # Sample real batch
                real_batch: Dict[str, torch.Tensor] = self.real_buffer.sample(self.num_real_samples_to_request)

                synthetic_batch: Optional[Dict[str, torch.Tensor]] = None
                # Conditionally sample synthetic batch
                if self.num_synthetic_samples_to_request > 0 and \
                   self.synthetic_buffer.size() >= self.num_synthetic_samples_to_request:
                    
                    synthetic_batch = self.synthetic_buffer.sample(self.num_synthetic_samples_to_request)
                
                # Perform agent training step
                rl_metrics: Dict[str, float] = self.agent.train_step(real_batch, synthetic_batch)
                
                # Accumulate metrics only if training was successful (rl_metrics not empty)
                # An empty rl_metrics dict usually indicates the agent decided to skip the update
                # due to insufficient data internally, which should be rare if real_buffer check passes.
                if rl_metrics:
                    for key, value in rl_metrics.items():
                        metrics_accum[key] += value
                    update_count += 1

            except IndexError as e:
                # This typically means a replay buffer (either real or synthetic) is empty
                # when `sample` was called, or it received an invalid batch_size.
                self.logger.log_scalar("Warnings/RL_Buffer_Sample_Error", 1.0, current_env_step)
                print(f"Warning: Replay buffer sample error during RL training at env step {current_env_step}: {e}. Skipping update.")
                break # Break from UTD loop if a buffer is unexpectedly empty/problematic
            except Exception as e:
                # Catch any other unexpected errors during agent.train_step
                self.logger.log_scalar("Errors/RL_Training_Step_Failure", 1.0, current_env_step)
                print(f"Error: An unexpected error occurred during RL agent training at env step {current_env_step}: {e}. Skipping update.")
                break

        averaged_metrics: Dict[str, float] = {}
        if update_count > 0:
            # Average accumulated metrics
            for key, value in metrics_accum.items():
                averaged_metrics[key] = value / update_count
            
            # Log averaged metrics
            for key, value in averaged_metrics.items():
                self.logger.log_scalar(f"RL_Training/{key}", value, current_env_step)
        else:
            # Log that no updates were performed if update_count is 0,
            # which could happen if buffers were not ready or an error occurred.
            self.logger.log_scalar("Warnings/No_RL_Updates_Performed", 1.0, current_env_step)

        return averaged_metrics

