import collections
import math
from typing import Dict, Optional, Tuple

import torch
import torch.optim as optim

from config import Config
from models.generative_nets import DenoisingDiffusionModel
from models.policy_nets import PolicyNetwork, QNetwork
from relevance_functions import RelevanceFunction
from replay_buffers import ReplayBuffer
from utils.logging_utils import Logger
from utils.common_utils import get_optimizer


class GenerativeReplayTrainer:
    """
    Orchestrates the training of the DenoisingDiffusionModel and the generation of synthetic data.
    It samples from D_real, computes relevance scores using RelevanceFunction, trains G,
    and then uses G to sample and populate D_syn.
    """

    def __init__(
        self,
        config: Config,
        diffusion_model: DenoisingDiffusionModel,
        relevance_func: RelevanceFunction,
        device: torch.device,
        logger: Logger,
    ):
        """
        Initializes the GenerativeReplayTrainer.

        Args:
            config (Config): Configuration object.
            diffusion_model (DenoisingDiffusionModel): The generative model (G) to be trained.
            relevance_func (RelevanceFunction): The function (F) used to compute condition scores for G.
            device (torch.device): The computational device (CPU/GPU) to use.
            logger (Logger): An instance of the central logging utility.
        """
        self.config: Config = config
        self.diffusion_model: DenoisingDiffusionModel = diffusion_model
        self.relevance_func: RelevanceFunction = relevance_func
        self.device: torch.device = device
        self.logger: Logger = logger

        # Optimizer for the diffusion model (already initialized within DenoisingDiffusionModel)
        # self.diffusion_optimizer = self.diffusion_model.optimizer # Access directly if needed

        # Hyperparameters
        try:
            self.train_steps_per_inner_loop: int = self.config.get_hyperparam('generative_model.train_steps_per_inner_loop')
        except KeyError:
            self.train_steps_per_inner_loop: int = 50 # Default if not specified in config
            self.logger.log_scalar("Warnings/Generative_Train_Steps_Default", self.train_steps_per_inner_loop, 0)
            print(f"Warning: 'generative_model.train_steps_per_inner_loop' not specified. Using default: {self.train_steps_per_inner_loop}")

        self.generation_batch_size: int = self.config.get_hyperparam('rl_agent.batch_size') # Use RL batch size for generation batches
        self.unconditional_drop_prob: float = self.config.get_hyperparam('generative_model.unconditional_drop_prob')
        self.guidance_scale: float = self.config.get_hyperparam('generative_model.guidance_scale')
        self.diffusion_timesteps: int = self.config.get_hyperparam('generative_model.diffusion_steps')
        self.generation_samples_per_inner_loop: int = self.config.get_hyperparam('generative_model.generation_samples_per_inner_loop')

    def train_generative_model(self, real_buffer: ReplayBuffer,
                               policy_nets: Optional[Tuple[PolicyNetwork, QNetwork]] = None,
                               current_env_step: int = 0) -> Dict[str, float]:
        """
        Performs one training cycle for the DenoisingDiffusionModel using real transitions
        and their computed relevance scores.

        Args:
            real_buffer (ReplayBuffer): The real replay buffer (D_real) from which to sample transitions for training G.
            policy_nets (Optional[Tuple[PolicyNetwork, QNetwork]]): A tuple containing the current policy's actor
                                                                    and critic networks, needed for Q-based F.
            current_env_step (int): The current global environment step, used for logging.

        Returns:
            Dict[str, float]: A dictionary of averaged training metrics for the generative model.
        """
        metrics_accum: collections.defaultdict[str, float] = collections.defaultdict(float)
        update_count: int = 0

        self.diffusion_model.train() # Set diffusion model to training mode

        for _ in range(self.train_steps_per_inner_loop):
            # Ensure enough data in real buffer for sampling
            if real_buffer.size() < self.generation_batch_size:
                # print(f"Warning: Real buffer size ({real_buffer.size()}) is less than generation_batch_size ({self.generation_batch_size}). Skipping generative model training step.")
                self.logger.log_scalar("Warnings/Generative_Buffer_Too_Small", 1.0, current_env_step)
                break # Exit loop if not enough data

            try:
                # Sample a batch of transitions from D_real
                batch = real_buffer.sample(self.generation_batch_size)

                # Compute Relevance Scores (c) for the batch
                condition_scores = self.relevance_func.compute_score(batch, policy_nets=policy_nets)
                
                # Train the diffusion model
                loss = self.diffusion_model.train_step(batch, condition_scores, self.unconditional_drop_prob)

                metrics_accum['generative_loss'] += loss
                update_count += 1

            except Exception as e:
                self.logger.log_scalar("Errors/Generative_Training_Step_Failure", 1.0, current_env_step)
                print(f"Error during generative model training at env step {current_env_step}: {e}. Skipping update.")
                break

        averaged_metrics: Dict[str, float] = {}
        if update_count > 0:
            for key, value in metrics_accum.items():
                averaged_metrics[key] = value / update_count
            self.logger.log_scalar("Generative_Training/Loss", averaged_metrics['generative_loss'], current_env_step)
        else:
            self.logger.log_scalar("Warnings/No_Generative_Updates_Performed", 1.0, current_env_step)

        return averaged_metrics

    def generate_synthetic_data(self, real_buffer: ReplayBuffer, synthetic_buffer: ReplayBuffer,
                                num_samples: int, policy_nets: Optional[Tuple[PolicyNetwork, QNetwork]] = None,
                                current_env_step: int = 0) -> None:
        """
        Generates a specified number of new synthetic transitions using the trained
        DenoisingDiffusionModel and populates the synthetic replay buffer.

        Args:
            real_buffer (ReplayBuffer): The real replay buffer (D_real) used to sample conditions for generation.
            synthetic_buffer (ReplayBuffer): The synthetic replay buffer (D_syn) to be filled with generated data.
            num_samples (int): The total number of synthetic samples to generate.
            policy_nets (Optional[Tuple[PolicyNetwork, QNetwork]]): A tuple containing the current policy's actor
                                                                    and critic networks, needed for Q-based F.
            current_env_step (int): The current global environment step, used for logging.
        """
        self.diffusion_model.eval() # Set diffusion model to evaluation mode for generation
        generated_count: int = 0

        # Clear synthetic buffer if it's meant to be fully replaced by new generations
        # Assuming we replace the entire D_syn as per paper, but `add_batch` will handle capacity.
        # This implementation strategy is to fill it completely to its capacity,
        # so we iterate until synthetic_buffer.capacity is reached.
        # The paper states D_syn is kept at 1M transitions and new ones are added.
        # So we don't explicitly clear it, just add `num_samples` (which is often D_syn capacity)

        num_batches = math.ceil(num_samples / self.generation_batch_size)

        with torch.no_grad():
            for i in range(num_batches):
                current_batch_size = min(self.generation_batch_size, num_samples - generated_count)
                if current_batch_size == 0:
                    break

                # Sample conditions from real_buffer ("Prompting Strategy")
                # Ensure enough data in real buffer for condition sampling
                if real_buffer.size() < current_batch_size:
                    # If not enough real samples, we can't generate conditions.
                    # This might happen early in training.
                    print(f"Warning: Real buffer size ({real_buffer.size()}) is less than required for condition sampling ({current_batch_size}). "
                          "Cannot generate all synthetic data.")
                    self.logger.log_scalar("Warnings/Generative_Condition_Sampling_Failed", 1.0, current_env_step)
                    break # Stop generating if conditions cannot be sampled

                # Sample a batch from D_real to compute conditions from
                real_batch_for_cond = real_buffer.sample(current_batch_size)
                
                # Compute conditions (relevance scores)
                condition_scores = self.relevance_func.compute_score(real_batch_for_cond, policy_nets=policy_nets)
                
                # Generate synthetic data
                synthetic_data_batch = self.diffusion_model.sample(
                    current_batch_size,
                    condition_scores,
                    self.guidance_scale,
                    self.diffusion_timesteps
                )
                
                # Add generated data to synthetic_buffer
                # Check if synthetic_buffer has an `add_batch` method (as planned)
                if hasattr(synthetic_buffer, 'add_batch') and callable(getattr(synthetic_buffer, 'add_batch')):
                    synthetic_buffer.add_batch(synthetic_data_batch)
                else:
                    # Fallback to individual additions if add_batch is not implemented
                    # (This is less efficient but ensures compatibility if replay_buffers.py not updated yet)
                    for j in range(current_batch_size):
                        synthetic_buffer.add(
                            state=synthetic_data_batch['state'][j].cpu().numpy(),
                            action=synthetic_data_batch['action'][j].cpu().numpy(),
                            reward=synthetic_data_batch['reward'][j].item(),
                            next_state=synthetic_data_batch['next_state'][j].cpu().numpy(),
                            done=synthetic_data_batch['done'][j].item()
                        )
                generated_count += current_batch_size
        
        self.logger.log_scalar("Generative_Training/Generated_Samples", generated_count, current_env_step)
        self.logger.log_scalar("Generative_Training/D_syn_size", synthetic_buffer.size(), current_env_step)
        print(f"Synthetic data generation complete. Added {generated_count} samples to D_syn. D_syn size: {synthetic_buffer.size()}")

