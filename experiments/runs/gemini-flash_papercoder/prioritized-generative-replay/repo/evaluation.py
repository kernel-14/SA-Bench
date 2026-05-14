```python
import collections
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import gym

# Assuming these are available in the project structure
from config import Config
from environments import EnvironmentManager
from models.generative_nets import DenoisingDiffusionModel
from models.policy_nets import CNNActor, PolicyNetwork, QNetwork
from replay_buffers import ReplayBuffer
from relevance_functions import RelevanceFunction
from rl_agents.base_agent import RLBaseAgent
from utils.logging_utils import Logger


class Evaluator:
    """
    The Evaluator class is responsible for quantifying the performance and behavior
    of the trained RL agent and the generative model. It computes evaluation metrics
    such as average returns, generation fidelity, and dormant ratio.
    """

    def __init__(self, config: Config, agent: RLBaseAgent, env_manager: EnvironmentManager, logger: Logger):
        """
        Initializes the Evaluator.

        Args:
            config (Config): An instance of the Config class.
            agent (RLBaseAgent): The RL agent to be evaluated.
            env_manager (EnvironmentManager): The environment manager for policy evaluation.
            logger (Logger): The logger for recording evaluation metrics.
        """
        self.config: Config = config
        self.agent: RLBaseAgent = agent
        self.env_manager: EnvironmentManager = env_manager
        self.logger: Logger = logger
        self.device: torch.device = self.config.get_hyperparam('experiment.device')
        self.pixel_based: bool = self.config.get_hyperparam('environment.pixel_based')

    def evaluate_policy(self, num_episodes: Optional[int] = None, current_env_step: int = 0) -> Dict[str, float]:
        """
        Evaluates the current policy's performance by running it in the environment
        for a specified number of episodes and calculating the average return.

        Args:
            num_episodes (Optional[int]): The number of episodes to run for evaluation.
                                          If None, uses `config.evaluation.policy_eval_episodes`.
            current_env_step (int): The current global environment step for logging purposes.

        Returns:
            Dict[str, float]: A dictionary containing 'mean_return' and 'std_return'.
        """
        if num_episodes is None:
            num_episodes = self.config.get_hyperparam('evaluation.policy_eval_episodes')

        episode_returns: List[float] = []
        
        # Set agent to evaluation mode to disable exploration and use deterministic actions
        self.agent.policy_net.eval()
        # Handle Q-network ensemble if present (e.g., REDQ), otherwise just the single Q-net
        q_networks_to_eval = []
        if hasattr(self.agent, 'q_networks'):
            q_networks_to_eval.extend(self.agent.q_networks)
        else:
            q_networks_to_eval.append(self.agent.q_net)
        
        for q_net in q_networks_to_eval:
            q_net.eval()
        
        # If noisy networks are enabled for the actor, their `get_action` in eval mode
        # will automatically use the mean (deterministic) part.

        with torch.no_grad():
            for _ in range(num_episodes):
                state: np.ndarray = self.env_manager.reset()
                episode_return: float = 0.0
                done: bool = False

                while not done:
                    # Convert state to tensor for agent input, add batch dimension
                    state_tensor: torch.Tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                    action_tensor: torch.Tensor = self.agent.get_action(state_tensor, deterministic=True)
                    
                    # Convert action tensor back to numpy and remove batch dim for env step
                    action_np: Union[int, np.ndarray]
                    if isinstance(self.env_manager.get_action_space(), gym.spaces.Discrete):
                        action_np = action_tensor.squeeze(0).cpu().item() # Discrete action is a scalar integer
                    else: # Continuous action (gym.spaces.Box)
                        action_np = action_tensor.squeeze(0).cpu().numpy()

                    next_state, reward, done, _ = self.env_manager.step(action_np)
                    episode_return += reward
                    state = next_state
                episode_returns.append(episode_return)

        # Set agent back to training mode
        self.agent.policy_net.train()
        for q_net in q_networks_to_eval:
            q_net.train()

        mean_return: float = np.mean(episode_returns)
        std_return: float = np.std(episode_returns)

        self.logger.log_scalar("Evaluation/Mean_Return", mean_return, current_env_step)
        self.logger.log_scalar("Evaluation/Std_Return", std_return, current_env_step)
        self.logger.log_scalar("Evaluation/Max_Return", np.max(episode_returns), current_env_step)
        self.logger.log_scalar("Evaluation/Min_Return", np.min(episode_returns), current_env_step)
        print(f"Eval at step {current_env_step}: Mean Return = {mean_return:.2f} +/- {std_return:.2f}")

        return {'mean_return': float(mean_return), 'std_return': float(std_return)}


    def compute_generation_fidelity(
        self,
        diffusion_model: DenoisingDiffusionModel,
        real_buffer: ReplayBuffer,
        relevance_func: RelevanceFunction,
        policy_nets: Optional[Tuple[PolicyNetwork, QNetwork]] = None,
        current_env_step: int = 0
    ) -> Dict[str, float]:
        """
        Measures the fidelity of transitions generated by the diffusion_model by comparing
        generated next_states and rewards against the ground-truth values from the
        real replay buffer that were used to compute the conditioning scores.

        Note: The paper describes rolling out generated `(s, a)` in the environment simulator.
        However, the `EnvironmentManager` design does not universally support setting the
        environment to an arbitrary generated state `s`. As a practical compromise, this
        implementation evaluates how well the generative model can reproduce the `(s', r)`
        of *real* transitions when conditioned on their relevance scores. This is a common
        fidelity check, evaluating if the generative model faithfully captures the data distribution.

        Args:
            diffusion_model (DenoisingDiffusionModel): The generative model (G).
            real_buffer (ReplayBuffer): The real replay buffer to sample conditions from and get ground truth.
            relevance_func (RelevanceFunction): The relevance function (F) for computing conditions.
            policy_nets (Optional[Tuple[PolicyNetwork, QNetwork]]): Actor and Critic for relevance function (if needed)
                                                                    and visual encoder (if pixel-based).
            current_env_step (int): The current global environment step for logging purposes.

        Returns:
            Dict[str, float]: A dictionary containing 'avg_mse_next_state' and 'avg_mse_reward'.
        """
        if not self.config.get_hyperparam('evaluation.generation_fidelity.enabled'):
            return {}

        num_samples: int = self.config.get_hyperparam('evaluation.generation_fidelity.num_generated_transitions')
        
        # Ensure that the real buffer has enough data to sample conditions from
        if real_buffer.size() < num_samples:
            print(f"Warning: Real buffer size ({real_buffer.size()}) is less than required for generation fidelity evaluation ({num_samples}). Skipping.")
            self.logger.log_scalar("Warnings/Fidelity_Real_Buffer_Too_Small", 1.0, current_env_step)
            return {}

        diffusion_model.eval()

        # 1. Sample real transitions from D_real. These will serve as the source for conditions
        #    and their `next_state`/`reward` will be the ground truth for comparison.
        sampled_real_batch: Dict[str, torch.Tensor] = real_buffer.sample(num_samples)
        
        # 2. Compute Relevance Scores (conditions) for the sampled real batch
        condition_scores: torch.Tensor = relevance_func.compute_score(sampled_real_batch, policy_nets=policy_nets)
        
        # 3. Generate synthetic transitions using the diffusion model, conditioned on `condition_scores`
        generated_transitions: Dict[str, torch.Tensor] = diffusion_model.sample(
            num_samples=num_samples,
            condition_scores=condition_scores,
            guidance_scale=self.config.get_hyperparam('generative_model.guidance_scale'),
            timesteps=self.config.get_hyperparam('generative_model.diffusion_steps')
        )
        
        # Extract generated components
        s_gen_latent: torch.Tensor = generated_transitions['state'] # This is generated `s`
        a_gen: torch.Tensor = generated_transitions['action'] # This is generated `a`
        r_gen: torch.Tensor = generated_transitions['reward'] # This is generated `r`
        s_prime_gen_latent: torch.Tensor = generated_transitions['next_state'] # This is generated `s'`

        mse_next_state_list: List[float] = []
        mse_reward_list: List[float] = []

        visual_encoder: Optional[nn.Module] = None
        if self.pixel_based and policy_nets is not None:
            actor_net, _ = policy_nets
            # Assumes CNNActor has an 'encoder' attribute
            if isinstance(actor_net, CNNActor) and hasattr(actor_net, 'encoder'):
                visual_encoder = actor_net.encoder
                visual_encoder.eval()
            else:
                print("Warning: Pixel-based environment and actor lacks a 'encoder' attribute. "
                      "Cannot compute latent s'_gt for fidelity comparison. Skipping s' MSE.")
                self.logger.log_scalar("Warnings/Fidelity_Missing_Visual_Encoder_S_Prime", 1.0, current_env_step)
        
        with torch.no_grad():
            for i in range(num_samples):
                # --- Get Ground Truth from the sampled_real_batch ---
                # This interpretation aligns generated output with the real data that served as its 'prompt'
                # for the condition, rather than trying to roll out a generated `s` in the environment
                # which is generally not supported by `EnvironmentManager`'s current API.
                s_prime_gt_raw: torch.Tensor = sampled_real_batch['