## main.py
import argparse
import random
import os
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import gym # Import gym for type hints like gym.spaces.Box

# Import local modules
from config import Config
from environments import EnvironmentManager
from replay_buffers import ReplayBuffer, PrioritizedReplayBuffer
from utils.logging_utils import Logger
from utils.common_utils import init_weights # To apply weight init globally or in specific nets
from models.policy_nets import MLPActor, CNNActor, MLPCritic, CNNCritic, PolicyNetwork, QNetwork
from models.generative_nets import DenoisingDiffusionModel
from relevance_functions import (
    RelevanceFunction,
    ReturnRelevance,
    TDErrorRelevance,
    ICMCuriosity,
    RNDCuriosity,
    CTSDensity,
    EPICCuriosity,
)
from rl_agents.sac_agent import SACAgent
from rl_agents.redq_agent import REDQAgent
from rl_agents.drqv2_agent import DRQV2Agent
from rl_agents.ppo_agent import PPOAgent
from rl_agents.base_agent import RLBaseAgent
from trainers.rl_trainer import RLTrainer
from trainers.generative_trainer import GenerativeReplayTrainer
from evaluation import Evaluator


class Main:
    """
    Main class orchestrating the entire PGR reproduction system.
    Handles argument parsing, configuration loading, component initialization,
    and execution of the main training and evaluation loops.
    """

    def __init__(self, config: Config):
        """
        Initializes the Main class with the provided configuration.

        Args:
            config (Config): An instance of the Config class containing all hyperparameters.
        """
        self.config: Config = config
        self.device: torch.device = self.config.get_hyperparam('experiment.device')
        self.logger: Optional[Logger] = None

        # Components will be initialized in _setup_components
        self.env_manager: Optional[EnvironmentManager] = None
        self.d_real: Optional[ReplayBuffer] = None
        self.d_syn: Optional[ReplayBuffer] = None
        self.rl_agent: Optional[RLBaseAgent] = None
        self.relevance_func: Optional[RelevanceFunction] = None
        self.diffusion_model: Optional[DenoisingDiffusionModel] = None
        self.rl_trainer: Optional[RLTrainer] = None
        self.generative_trainer: Optional[GenerativeReplayTrainer] = None
        self.evaluator: Optional[Evaluator] = None

    def _setup_components(self) -> None:
        """
        Sets up all necessary components for the experiment, including:
        logging, random seeds, environment, replay buffers, RL agent,
        relevance function, generative model, and trainers/evaluators.
        """
        # 1. Logger Setup
        self.logger = Logger(self.config)
        
        # 2. Random Seed Management
        seed: int = self.config.get_hyperparam('experiment.seed')
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # 3. Environment Manager
        self.env_manager = EnvironmentManager(
            config=self.config,
            seed=seed
        )

        env_obs_space: gym.Space = self.env_manager.get_observation_space()
        env_act_space: gym.Space = self.env_manager.get_action_space()
        pixel_based: bool = self.config.get_hyperparam('environment.pixel_based')
        is_continuous_action: bool = isinstance(env_act_space, gym.spaces.Box)

        # Determine effective state and action dimensions for various modules
        if pixel_based:
            # Replay buffer stores raw pixel data
            buffer_state_dim: Tuple[int, ...] = env_obs_space.shape  # (C, H, W)
            # Networks operate on latent features derived from the visual encoder
            network_state_dim: Tuple[int, ...] = env_obs_space.shape # (C, H, W) for CNN input
            diffusion_state_dim_flat: Tuple[int, ...] = (self.config.get_hyperparam('environment.visual_encoder_output_dim'),)
        else:
            # State vector
            buffer_state_dim: Tuple[int, ...] = env_obs_space.shape
            network_state_dim: Union[int, Tuple[int, ...]] = env_obs_space.shape[0]
            diffusion_state_dim_flat: Tuple[int, ...] = (network_state_dim,) # Same for state-based

        if is_continuous_action:
            buffer_action_dim: Tuple[int, ...] = env_act_space.shape
            network_action_dim: int = env_act_space.shape[0]
            diffusion_action_dim: int = network_action_dim
        else:  # Discrete action space
            buffer_action_dim: Tuple[int, ...] = (1,)  # Store as a single integer
            network_action_dim: int = env_act_space.n # Number of possible actions
            diffusion_action_dim: int = 1 # Diffuse the action index

        # 4. Replay Buffers
        d_real_capacity: int = self.config.get_hyperparam('replay_buffers.d_real_capacity')
        d_syn_capacity: int = self.config.get_hyperparam('replay_buffers.d_syn_capacity')

        self.d_real = ReplayBuffer(
            capacity=d_real_capacity,
            state_dim=buffer_state_dim,
            action_dim=buffer_action_dim[0] if buffer_action_dim else 1, # Handle scalar action spaces if any
            pixel_based=pixel_based,
            device=self.device
        )
        self.d_syn = ReplayBuffer(
            capacity=d_syn_capacity,
            state_dim=diffusion_state_dim_flat, # Synthetic buffer stores latent states if pixel-based
            action_dim=diffusion_action_dim,
            pixel_based=False, # Synthetic buffer stores features, not raw pixels
            device=self.device
        )
        self.logger.log_scalar("ReplayBuffers/D_real_capacity", d_real_capacity, 0)
        self.logger.log_scalar("ReplayBuffers/D_syn_capacity", d_syn_capacity, 0)

        # 5. RL Agent (Policy and Q-Networks are instantiated within the agent)
        rl_algorithm: str = self.config.get_hyperparam('rl_agent.algorithm')
        
        agent_class: Type[RLBaseAgent]
        if rl_algorithm == "SAC":
            agent_class = SACAgent
        elif rl_algorithm == "REDQ":
            agent_class = REDQAgent
            # REDQ specific: number of Q networks in the ensemble
            num_q_networks: int = self.config.get_hyperparam('rl_agent.num_q_networks')
            if num_q_networks == "NOT_SPECIFIED":
                self.config.rl_agent.num_q_networks = 2 # Default for REDQ
                self.logger.log_scalar("Warnings/REDQ_num_q_networks_default", 2, 0)
            num_min_q_networks: int = self.config.get_hyperparam('rl_agent.num_min_q_networks')
            if num_min_q_networks == "NOT_SPECIFIED":
                self.config.rl_agent.num_min_q_networks = min(self.config.rl_agent.num_q_networks, 2) # Default for REDQ
                self.logger.log_scalar("Warnings/REDQ_num_min_q_networks_default", self.config.rl_agent.num_min_q_networks, 0)
        elif rl_algorithm == "DRQV2":
            if not pixel_based:
                raise ValueError("DRQV2 agent requires pixel-based environment. Set 'environment.pixel_based' to True.")
            agent_class = DRQV2Agent
        elif rl_algorithm == "PPO":
            if is_continuous_action:
                raise ValueError("PPOAgent in this context expects discrete action space, but got continuous.")
            agent_class = PPOAgent
            # PPO specific: clip epsilon, entropy_coef, value_loss_coef, ppo_epochs
            if self.config.get_hyperparam('rl_agent.ppo_clip_epsilon') == "NOT_SPECIFIED":
                self.config.rl_agent.ppo_clip_epsilon = 0.2
            if self.config.get_hyperparam('rl_agent.ppo_entropy_coef') == "NOT_SPECIFIED":
                self.config.rl_agent.ppo_entropy_coef = 0.01
            if self.config.get_hyperparam('rl_agent.ppo_value_loss_coef') == "NOT_SPECIFIED":
                self.config.rl_agent.ppo_value_loss_coef = 0.5
            if self.config.get_hyperparam('rl_agent.ppo_epochs') == "NOT_SPECIFIED":
                self.config.rl_agent.ppo_epochs = 4 # Common default for PPO training loops
            
        else:
            raise ValueError(f"Unknown RL algorithm: {rl_algorithm}")

        # Instantiate the RL agent
        self.rl_agent = agent_class(
            config=self.config,
            env_manager=self.env_manager,
            device=self.device
        )
        self.logger.log_scalar("RL_Agent/Algorithm", 0, 0) # Dummy log to indicate algorithm choice

        # 6. Relevance Function (F)
        relevance_type: str = self.config.get_hyperparam('relevance_function.type')
        
        relevance_func_class: Type[RelevanceFunction]
        if relevance_type == "Return":
            relevance_func_class = ReturnRelevance
        elif relevance_type == "TD_Error":
            relevance_func_class = TDErrorRelevance
        elif relevance_type == "Curiosity":
            relevance_func_class = ICMCuriosity
        elif relevance_type == "RND":
            relevance_func_class = RNDCuriosity
        elif relevance_type == "CTS":
            if not pixel_based:
                raise ValueError("CTS relevance function requires pixel-based environment. Set 'environment.pixel_based' to True.")
            relevance_func_class = CTSDensity
        elif relevance_type == "EPIC":
            if not pixel_based:
                raise ValueError("EPIC relevance function requires pixel-based environment. Set 'environment.pixel_based' to True.")
            relevance_func_class = EPICCuriosity
        else:
            raise ValueError(f"Unknown relevance function type: {relevance_type}")

        self.relevance_func = relevance_func_class(
            config=self.config,
            state_dim=network_state_dim,
            action_dim=network_action_dim,
            device=self.device
        )
        self.logger.log_scalar("RelevanceFunction/Type", 0, 0) # Dummy log to indicate type choice

        # 7. Generative Model (G)
        # Condition dimension is always 1 for a scalar relevance score
        diffusion_condition_dim: int = 1
        self.diffusion_model = DenoisingDiffusionModel(
            config=self.config,
            state_dim=diffusion_state_dim_flat,
            action_dim=diffusion_action_dim,
            reward_dim=1,
            condition_dim=diffusion_condition_dim
        )
        self.logger.log_scalar("GenerativeModel/Type", 0, 0) # Dummy log

        # 8. Trainers
        self.rl_trainer = RLTrainer(
            config=self.config,
            agent=self.rl_agent,
            real_buffer=self.d_real,
            synthetic_buffer=self.d_syn,
            logger=self.logger,
            relevance_func=self.relevance_func # Pass relevance_func for updating trainable F
        )
        self.generative_trainer = GenerativeReplayTrainer(
            config=self.config,
            diffusion_model=self.diffusion_model,
            relevance_func=self.relevance_func,
            device=self.device,
            logger=self.logger
        )

        # 9. Evaluator
        self.evaluator = Evaluator(
            config=self.config,
            agent=self.rl_agent,
            env_manager=self.env_manager,
            logger=self.logger
        )

        print(f"Setup complete for experiment: {self.config.get_hyperparam('experiment.name')}")

    def _prefill_buffers(self, min_samples: int) -> None:
        """
        Collects initial experiences to pre-fill the real replay buffer.
        """
        if self.d_real is None or self.env_manager is None:
            raise RuntimeError("Replay buffer or environment manager not initialized.")
        
        print(f"Prefilling real replay buffer with {min_samples} samples...")
        current_state: np.ndarray = self.env_manager.reset()
        current_env_step: int = 0
        while self.d_real.size() < min_samples:
            action_tensor: torch.Tensor = self.rl_agent.get_action(
                torch.tensor(current_state, dtype=torch.float32, device=self.device).unsqueeze(0),
                deterministic=False # Use stochastic policy for exploration
            )
            action_np: Union[int, np.ndarray]
            if isinstance(self.env_manager.get_action_space(), gym.spaces.Discrete):
                action_np = action_tensor.squeeze(0).cpu().item()
            else:
                action_np = action_tensor.squeeze(0).cpu().numpy()

            next_state, reward, done, _ = self.env_manager.step(action_np)
            self.d_real.add(current_state, action_np, float(reward), next_state, done)
            current_state = next_state
            if done:
                current_state = self.env_manager.reset()
            current_env_step += 1
            if current_env_step % 1000 == 0:
                print(f"Prefill: {self.d_real.size()}/{min_samples} samples collected.")
        print(f"Real replay buffer prefilled with {self.d_real.size()} samples.")
        self.logger.log_scalar("ReplayBuffers/D_real_initial_size", self.d_real.size(), 0)


    def run_experiment(self) -> None:
        """
        Executes the main training and evaluation loops of the PGR algorithm.
        """
        if not all([self.env_manager, self.d_real, self.d_syn, self.rl_agent,
                    self.relevance_func, self.diffusion_model, self.rl_trainer,
                    self.generative_trainer, self.evaluator, self.logger]):
            self._setup_components()

        total_env_steps: int = self.config.get_hyperparam('environment.total_env_steps')
        inner_loop_freq_env_steps: int = self.config.get_hyperparam('pgr_loop.inner_loop_freq_env_steps')
        policy_eval_freq_env_steps: int = self.config.get_hyperparam('evaluation.policy_eval_freq_env_steps')
        save_model_freq_env_steps: int = self.config.get_hyperparam('logging.save_model_freq_env_steps')
        min_prefill_samples: int = max(self.config.get_hyperparam('rl_agent.batch_size') * 5, 1000) # Ensure enough for first batch

        self._prefill_buffers(min_prefill_samples)

        current_env_step: int = 0
        episode: int = 0
        current_state: np.ndarray = self.env_manager.reset()

        print("Starting main training loop...")
        while current_env_step < total_env_steps:
            # 1. Environment Interaction (Outer Loop)
            with torch.no_grad():
                action_tensor: torch.Tensor = self.rl_agent.get_action(
                    torch.tensor(current_state, dtype=torch.float32, device=self.device).unsqueeze(0),
                    deterministic=False # Use stochastic policy for exploration
                )
                action_np: Union[int, np.ndarray]
                if isinstance(self.env_manager.get_action_space(), gym.spaces.Discrete):
                    action_np = action_tensor.squeeze(0).cpu().item()
                else:
                    action_np = action_tensor.squeeze(0).cpu().numpy()

            next_state, reward, done, _ = self.env_manager.step(action_np)
            self.d_real.add(current_state, action_np, float(reward), next_state, done)
            current_state = next_state
            
            if done:
                current_state = self.env_manager.reset()
                episode += 1
                self.logger.log_scalar("Environment/Episode", episode, current_env_step)

            current_env_step += 1
            self.logger.log_scalar("Environment/Total_Env_Steps", current_env_step, current_env_step)
            self.logger.log_scalar("ReplayBuffers/D_real_size", self.d_real.size(), current_env_step)


            # 2. RL Policy Training
            if self.d_real.size() >= self.config.get_hyperparam('rl_agent.batch_size'):
                self.rl_trainer.train_policy(
                    current_env_step=current_env_step,
                    rl_agent_policy_nets=self.rl_agent.get_policy_nets() # Pass for relevance_func updates
                )

            # 3. Generative Replay Inner Loop (Periodic updates)
            if current_env_step > min_prefill_samples and \
               current_env_step % inner_loop_freq_env_steps == 0:
                print(f"--- Generative Inner Loop at env step {current_env_step} ---")
                
                # Train generative model
                generative_metrics = self.generative_trainer.train_generative_model(
                    real_buffer=self.d_real,
                    policy_nets=self.rl_agent.get_policy_nets(),
                    current_env_step=current_env_step
                )
                if generative_metrics:
                    for k, v in generative_metrics.items():
                        self.logger.log_scalar(f"Generative_Training/{k}", v, current_env_step)
                
                # Generate synthetic data
                self.generative_trainer.generate_synthetic_data(
                    real_buffer=self.d_real,
                    synthetic_buffer=self.d_syn,
                    num_samples=self.config.get_hyperparam('generative_model.generation_samples_per_inner_loop'),
                    policy_nets=self.rl_agent.get_policy_nets(),
                    current_env_step=current_env_step
                )
                self.logger.log_scalar("ReplayBuffers/D_syn_size", self.d_syn.size(), current_env_step)


            # 4. Evaluation and Checkpointing
            if current_env_step % policy_eval_freq_env_steps == 0:
                print(f"--- Evaluation at env step {current_env_step} ---")
                
                # Policy Evaluation
                eval_results = self.evaluator.evaluate_policy(
                    current_env_step=current_env_step
                )
                if eval_results:
                    for k, v in eval_results.items():
                        self.logger.log_scalar(f"Evaluation/{k}", v, current_env_step)

                # Generation Fidelity Evaluation
                if self.config.get_hyperparam('evaluation.generation_fidelity.enabled') and \
                   current_env_step >= self.config.get_hyperparam('evaluation.generation_fidelity.eval_epoch') and \
                   self.d_real.size() >= self.config.get_hyperparam('evaluation.generation_fidelity.num_generated_transitions'):
                    
                    fidelity_metrics = self.evaluator.compute_generation_fidelity(
                        diffusion_model=self.diffusion_model,
                        real_buffer=self.d_real,
                        relevance_func=self.relevance_func,
                        policy_nets=self.rl_agent.get_policy_nets(),
                        current_env_step=current_env_step
                    )
                    if fidelity_metrics:
                        for k, v in fidelity_metrics.items():
                            self.logger.log_scalar(f"Fidelity/{k}", v, current_env_step)

                # Dormant Ratio Evaluation
                if self.config.get_hyperparam('evaluation.dormant_ratio.enabled'):
                    dormant_ratio_val = self.evaluator.compute_dormant_ratio(
                        policy_net=self.rl_agent.policy_net,
                        real_buffer=self.d_real
                    )
                    self.logger.log_scalar("Evaluation/Dormant_Ratio", dormant_ratio_val, current_env_step)

                # Save Checkpoint
                if current_env_step % save_model_freq_env_steps == 0:
                    checkpoint_path = os.path.join(self.logger.run_dir, f"checkpoint_env_step_{current_env_step}.pt")
                    self.rl_agent.save_checkpoint(checkpoint_path)
                    # For diffusion_model and relevance_func, we might need a separate save method
                    # or integrate them into agent checkpoint if they are tied.
                    # For now, let's just save agent.
                    self.logger.save_checkpoint(
                        {'rl_agent_state': self.rl_agent.state_dict()}, # Example, agent.save_checkpoint handles more
                        step=current_env_step,
                        filename=f"rl_agent_step_{current_env_step}.pt"
                    ) # Using logger's save_checkpoint to also log to wandb if enabled
                    # Need to save diffusion model too
                    self.logger.save_checkpoint(
                        {'diffusion_model_state': self.diffusion_model.state_dict()},
                        step=current_env_step,
                        filename=f"diffusion_model_step_{current_env_step}.pt"
                    )
                    # And relevance function if trainable
                    if self.relevance_func.get_params():
                        self.logger.save_checkpoint(
                            {'relevance_func_state': self.relevance_func.state_dict()},
                            step=current_env_step,
                            filename=f"relevance_func_step_{current_env_step}.pt"
                        )


        print(f"Training finished after {current_env_step} environment steps.")
        self.logger.close()
        self.env_manager.close()


def main() -> None:
    """
    Main entry point of the script. Parses arguments, loads config, and runs the experiment.
    """
    parser = argparse.ArgumentParser(description="Reproduce Prioritized Generative Replay (PGR) experiments.")
    parser.add_argument('--config_path', type=str, default='config.yaml',
                        help='Path to the YAML configuration file.')
    
    # Command-line overrides for common hyperparameters
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for the experiment (overrides config.yaml).')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (overrides config.yaml).')
    parser.add_argument('--device', type=str, default=None,
                        help='Computation device (e.g., "cuda", "cpu") (overrides config.yaml).')
    parser.add_argument('--env_name', type=str, default=None,
                        help='Environment name (overrides config.yaml).')
    parser.add_argument('--env_suite', type=str, default=None,
                        help='Environment suite (e.g., "DMC", "OpenAI_Gym", "DMLab") (overrides config.yaml).')
    parser.add_argument('--pixel_based', type=lambda x: (str(x).lower() == 'true'), default=None,
                        help='Whether environment observations are pixel-based (overrides config.yaml).')
    parser.add_argument('--rl_algo', type=str, default=None,
                        help='Reinforcement Learning algorithm to use (overrides config.yaml).')
    parser.add_argument('--relevance_type', type=str, default=None,
                        help='Type of relevance function for PGR (overrides config.yaml).')
    parser.add_argument('--utd_ratio', type=int, default=None,
                        help='Update-to-Data ratio for RL agent training (overrides config.yaml).')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for RL agent training (overrides config.yaml).')
    parser.add_argument('--synthetic_data_ratio', type=float, default=None,
                        help='Ratio of synthetic data in RL training batch (overrides config.yaml).')
    parser.add_argument('--inner_loop_freq', type=int, default=None,
                        help='Frequency of generative inner loop in environment steps (overrides config.yaml).')
    parser.add_argument('--total_env_steps', type=int, default=None,
                        help='Total environment steps for the experiment (overrides config.yaml).')
    parser.add_argument('--q_hidden_layers', type=int, default=None,
                        help='Number of hidden layers for Q-networks (overrides config.yaml).')
    parser.add_argument('--q_hidden_units', type=int, default=None,
                        help='Number of units in hidden layers for Q-networks (overrides config.yaml).')
    parser.add_argument('--policy_hidden_layers', type=int, default=None,
                        help='Number of hidden layers for policy network (overrides config.yaml).')
    parser.add_argument('--policy_hidden_units', type=int, default=None,
                        help='Number of units in hidden layers for policy network (overrides config.yaml).')
    parser.add_argument('--guidance_scale', type=float, default=None,
                        help='Guidance scale for conditional diffusion model generation (overrides config.yaml).')

    args = parser.parse_args()

    # Create Config object, which loads from YAML and applies CMD arguments
    config = Config(config_path=args.config_path, cmd_args=args)

    # Initialize and run the main experiment
    experiment_runner = Main(config)
    experiment_runner.run_experiment()


if __name__ == "__main__":
    main()

