```python
import os
import random
import torch
import numpy as np
from tqdm import tqdm
from typing import Any, Dict, List, Tuple, Callable, Optional, Union

# Import core modules
from config import Config
from utils.logger import Logger
from utils.level_generator import LevelGenerator

# Environment imports
from environments.base_environment import BaseEnvironment
from environments.sokoban import SokobanEnv
from environments.mini_pacman import MiniPacManEnv
from environments.sokoban import (
    NEVER, UP, DOWN, LEFT, RIGHT # For concept class names
)

# Agent imports
from agents.base_agent import BaseAgentModel
from agents.drc_agent import DRCAgent
from agents.resnet_agent import ResNetAgent

# RL core imports
from rl_core.trainer import IMPALATrainer
from rl_core.trajectory_collector import TrajectoryCollector

# Interpretability imports
from interpretability.concept_labeler import ConceptLabeler
from interpretability.probe_model import ProbeModel
from interpretability.probe_trainer import ProbeDataset, ProbeTrainer
from interpretability.intervention_module import InterventionModule

# Visualization imports
from utils.visualization import Visualization


class ExperimentRunner:
    """
    The main orchestrator for reproducing the paper's experiments.
    It initializes all necessary components, coordinates data flow,
    and calls the appropriate trainers, collectors, labelers, and intervention modules.
    """

    def __init__(self, config: Config, logger: Logger) -> None:
        """
        Initializes the ExperimentRunner.

        Args:
            config (Config): The configuration object for accessing settings.
            logger (Logger): The logger instance for logging metrics and checkpoints.
        """
        self.config: Config = config
        self.logger: Logger = logger
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Caches to avoid redundant training/instantiation
        self.agents_cache: Dict[str, BaseAgentModel] = {}
        self.envs_cache: Dict[str, BaseEnvironment] = {}
        # Stores probes in a nested dictionary:
        # probes_cache[concept_type][agent_type][env_name][layer_idx][tick_idx][kernel_size][seed_idx] -> ProbeModel
        self.probes_cache: Dict[str, Any] = {}
        self.visualization_util: Visualization = Visualization(config=self.config)
        self.level_generator: LevelGenerator = LevelGenerator(config=self.config)

        self.logger.log_info("ExperimentRunner initialized.")

    def _get_agent_and_env(self, agent_type: str, env_name: str,
                           specific_agent_config: Optional[Dict[str, Any]] = None) -> Tuple[BaseAgentModel, BaseEnvironment]:
        """
        Retrieves or creates an agent and environment instance, using caches to avoid redundancy.

        Args:
            agent_type (str): Type of agent to get (e.g., 'DRCAgent', 'ResNetAgent').
            env_name (str): Name of the environment (e.g., 'Sokoban', 'MiniPacMan').
            specific_agent_config (Optional[Dict[str, Any]]): Optional dictionary to override agent config parameters
                                                               for this specific instance (e.g., {'D':1, 'N':9}).

        Returns:
            Tuple[BaseAgentModel, BaseEnvironment]: The instantiated agent and environment.
        """
        # --- Environment Retrieval/Creation ---
        env_key: str = env_name
        if env_key not in self.envs_cache:
            self.logger.log_info(f"Creating new environment: {env_name}")
            if env_name == "Sokoban":
                env_instance = SokobanEnv(config=self.config)
            elif env_name == "MiniPacMan":
                env_instance = MiniPacManEnv(config=self.config)
            else:
                raise ValueError(f"Unknown environment type: {env_name}")
            self.envs_cache[env_key] = env_instance
        env: BaseEnvironment = self.envs_cache[env_key]

        # --- Agent Retrieval/Creation ---
        # Generate a unique key for the agent based on its type and specific config
        agent_config_key: str = f"{agent_type}_{env_name}"
        if specific_agent_config:
            # Sort keys for consistent hashing of config dict for cache key
            sorted_config_items = sorted(specific_agent_config.items())
            agent_config_key += "_" + "_".join([f"{k}-{v}" for k, v in sorted_config_items])

        if agent_config_key not in self.agents_cache:
            self.logger.log_info(f"Creating new agent: {agent_type} for environment {env_name} with config {specific_agent_config}")
            # Temporarily modify config to create agent with specific parameters
            original_config_agent_type_value = self.config.get('agent.type')
            self.config.set('agent.type', agent_type) # Ensure config reflects desired agent type

            agent_instance: BaseAgentModel
            if agent_type == "DRCAgent":
                original_drc_d = self.config.get('agent.drc_agent.D')
                original_drc_n = self.config.get('agent.drc_agent.N')
                if specific_agent_config:
                    self.config.set('agent.drc_agent.D', specific_agent_config.get('D', original_drc_d))
                    self.config.set('agent.drc_agent.N', specific_agent_config.get('N', original_drc_n))
                agent_instance = DRCAgent(config=self.config)
                # Restore original config values
                self.config.set('agent.drc_agent.D', original_drc_d)
                self.config.set('agent.drc_agent.N', original_drc_n)
            elif agent_type == "ResNetAgent":
                original_resnet_blocks = self.config.get('agent.resnet_agent.num_residual_blocks')
                if specific_agent_config:
                    self.config.set('agent.resnet_agent.num_residual_blocks', specific_agent_config.get('num_residual_blocks', original_resnet_blocks))
                agent_instance = ResNetAgent(config=self.config)
                # Restore original config values
                self.config.set('agent.resnet_agent.num_residual_blocks', original_resnet_blocks)
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")
            
            # Restore original agent type in config
            self.config.set('agent.type', original_config_agent_type_value)
            
            self.agents_cache[agent_config_key] = agent_instance
        agent: BaseAgentModel = self.agents_cache[agent_config_key]

        return agent, env

    def run_rl_training(self, agent_type: str, env_name: str = None, specific_agent_config: Optional[Dict[str, Any]] = None) -> BaseAgentModel:
        """
        Trains a specified RL agent using the IMPALA algorithm.

        Args:
            agent_type (str): Type of agent to train (e.g., 'DRCAgent', 'ResNetAgent').
            env_name (str, optional): Name of the environment. Defaults to config default.
            specific_agent_config (Optional[Dict[str, Any]]): Optional dictionary to override agent config parameters
                                                               (e.g., {'D':1, 'N':9}).

        Returns:
            BaseAgentModel: The trained agent model.
        """
        self.logger.log_info(f"--- Running RL Training for {agent_type} on {env_name if env_name else self.config.get('environment.name')} ---")
        
        env_name = env_name if env_name else self.config.get('environment.name')
        agent, env = self._get_agent_and_env(agent_type, env_name, specific_agent_config)

        num_transitions: int
        if specific_agent_config and (specific_agent_config.get('D') == 1 and specific_agent_config.get('N') == 9 or \
                                     specific_agent_config.get('D') == 9 and specific_agent_config.get('N') == 1):
            num_transitions = self.config.get('rl_training.num_transitions_appendix_f', 100_000_000)
        else:
            num_transitions = self.config.get('rl_training.num_transitions', 250_000_000)
        
        checkpoint_interval: int = self.config.get('rl_training.checkpoint_interval', 1_000_000) # Save every 1M transitions

        trainer = IMPALATrainer(agent_model=agent, env=env, config=self.config, logger=self.logger)
        trainer.train(num_transitions=num_transitions, checkpoint_interval=checkpoint_interval)

        return agent

    def run_probe_experiments(self, agent_model: BaseAgentModel, env_name: str = None) -> Dict[str, Any]:
        """
        Executes the probing for concept representations phase.

        Args:
            agent_model (BaseAgentModel): The agent model to probe.
            env_name (str, optional): Name of the environment. Defaults to config default.

        Returns:
            Dict[str, Any]: The updated probes cache.
        """
        self.logger.log_info(f"--- Running Probe Experiments for {type(agent_model).__name__} on {env_name if env_name else self.config.get('environment.name')} ---")

        env_name = env_name if env_name else self.config.get('environment.name')
        env = self.envs_cache.get(env_name)
        if env is None:
            raise ValueError(f"Environment '{env_name}' not found in cache. Ensure it's been initialized.")

        # --- Trajectory Data Collection ---
        self.logger.log_info("Collecting trajectories for probe training and validation...")
        collector = TrajectoryCollector(agent_model=agent_model, env=env, config=self.config)

        probe_data_collection_config = self.config.get('probing.probe_data_collection.full_agent')
        train_episodes = probe_data_collection_config.get('train_episodes', 3000)
        val_episodes = probe_data_collection_config.get('val_episodes', 1000)

        # For specific agents, need to ensure correct num_ticks for cell state collection
        store_cell_states = True # Always store for probing
        if isinstance(agent_model, DRCAgent):
            # For DRC, use the agent's internal forward pass for tick-by-tick state collection
            # This is handled internally by TrajectoryCollector.collect_trajectories
            # which will use agent_model's internal _h_states_per_tick_layer and _c_states_per_tick_layer.
            pass
        elif isinstance(agent_model, ResNetAgent):
            # For ResNet, num_ticks is 1, representing the state after each block.
            pass
        else:
            self.logger.log_warning(f"Unknown agent type {type(agent_model).__name__} for cell state collection logic.")


        raw_train_trajectories = collector.collect_trajectories(num_episodes=train_episodes, behavior_policy=agent_model.act, store_cell_states=store_cell_states)
        raw_val_trajectories = collector.collect_trajectories(num_episodes=val_episodes, behavior_policy=agent_model.act, store_cell_states=store_cell_states)
        self.logger.log_info(f"Collected {len(raw_train_trajectories)} train and {len(raw_val_trajectories)} validation trajectories.")

        # --- Concept Label Generation ---
        self.logger.log_info("Generating ground truth concept labels...")
        labeler = ConceptLabeler(env=env)
        labeled_train_data = []
        for traj in tqdm(raw_train_trajectories, desc="Labeling train data"):
            labeled_train_data.extend(labeler.label_trajectory(traj))
        
        labeled_val_data = []
        for traj in tqdm(raw_val_trajectories, desc="Labeling validation data"):
            labeled_val_data.extend(labeler.label_trajectory(traj))
        self.logger.log_info(f"Generated {len(labeled_train_data)} train and {len(labeled_val_data)} validation labeled data points (timesteps).")
        
        # --- Probe Training and Evaluation Loop ---
        concept_types: List[str] = self.config.get('probing.concept_types', ['CA', 'CB'])
        probe_kernel_sizes: List[int] = self.config.get('probing.probe_kernel_sizes', [1, 3])
        initialization_seeds: int = self.config.get('probing.initialization_seeds', 5)

        num_layers_agent: int
        if isinstance(agent_model, DRCAgent):
            num_layers_agent = agent_model.D
            num_ticks_agent = agent_model.N + 1 # from 0 to N
            agent_channels = agent_model.convlstm_channels
        elif isinstance(agent_model, ResNetAgent):
            num_layers_agent = agent_model.num_residual_blocks
            num_ticks_agent = 1 # ResNet has no internal ticks, we use block index as layer_idx and tick_idx=0
            agent_channels = agent_model.block_channels
        else:
            raise ValueError(f"Unsupported agent type for probing: {type(agent_model).__name__}")
        
        env_obs_shape = env.get_observation_space_shape()
        input_spatial_dims = env_obs_shape[:2] # (H, W)
        env_obs_channels = env_obs_shape[2]

        # Map integer concept labels to human-readable names for logging
        concept_class_names = {
            'CA': ['NEVER', 'UP', 'DOWN', 'LEFT', 'RIGHT'],
            'CB': ['NEVER', 'UP', 'DOWN', 'LEFT', 'RIGHT'],
            'AgentApproach': ['NEVER', 'AGAIN'],
            'BoxPush': ['NEVER', 'AGAIN'],
            'AgentExitDirection': ['NEVER', 'UP', 'DOWN', 'LEFT', 'RIGHT'],
            'BoxApproachDirection': ['NEVER', 'UP', 'DOWN', 'LEFT', 'RIGHT'],
            'ActionToTake_1': [str(i) for i in range(env.get_action_space_size())], # For global probe
            'AgentApproachDirection_MiniPacMan_16': ['NEVER', 'UP', 'DOWN', 'LEFT', 'RIGHT'],
            'AgentApproach_MiniPacMan_16': ['NEVER', 'AGAIN']
        }

        # Loop through layers, concepts, and kernel sizes
        for layer_idx in range(num_layers_agent):
            for tick_idx in range(num_ticks_agent): # For ResNet, this loop only runs once for tick_idx = 0
                for concept_type in concept_types:
                    # Determine if it's a global probe (e.g., ActionToTake_N_TimeSteps)
                    is_global_probe = concept_type.startswith("ActionToTake_")
                    
                    # Determine num_classes for the current concept
                    num_classes = len(concept_class_names.get(concept_type, []))
                    if num_classes == 0 and is_global_probe:
                         num_classes = env.get_action_space_size() # Default for global action prediction
                    if num_classes == 0:
                        self.logger.log_warning(f"No class names defined for concept {concept_type}, defaulting to 5 classes.")
                        num_classes = 5 # Fallback to 5 classes for directional concepts

                    for kernel_size in probe_kernel_sizes:
                        if is_global_probe and kernel_size != 1: # Global probes don't use kernels
                            continue

                        self.logger.log_info(f"Probing Layer {layer_idx}, Tick {tick_idx}, Concept '{concept_type}', Kernel {kernel_size}x{kernel_size}...")

                        # --- Train and Evaluate Agent-based Probes ---
                        probe_inputs_channels = agent_channels
                        if is_global_probe: # Flattened input for global probe
                            probe_inputs_channels = agent_channels * input_spatial_dims[0] * input_spatial_dims[1]

                        # Train probes multiple times with different seeds
                        for seed_idx in range(initialization_seeds):
                            random.seed(seed_idx + self.config.get('seed', 42)) # Use a unique seed for each probe
                            torch.manual_seed(seed_idx + self.config.get('seed', 42))
                            np.random.seed(seed_idx + self.config.get('seed', 42))

                            probe_model = ProbeModel(
                                in_channels=probe_inputs_channels,
                                num_classes=num_classes,
                                kernel_size=kernel_size,
                                is_global=is_global_probe
                            )
                            probe_trainer = ProbeTrainer(
                                probe_model=probe_model,
                                config=self.config,
                                logger=self.logger,
                                class_names=concept_class_names.get(concept_type, [])
                            )

                            train_dataset = ProbeDataset(
                                labeled_data=labeled_train_data, concept_key=concept_type, layer_idx=layer_idx,
                                tick_idx=tick_idx, is_baseline=False, input_spatial_dims=input_spatial_dims,
                                num_input_channels=probe_inputs_channels, is_global_probe=is_global_probe
                            )
                            val_dataset = ProbeDataset(
                                labeled_data=labeled_val_data, concept_key=concept_type, layer_idx=layer_idx,
                                tick_idx=tick_idx, is_baseline=False, input_spatial_dims=input_spatial_dims,
                                num_input_channels=probe_inputs_channels, is_global_probe=is_global_probe
                            )
                            
                            if len(train_dataset) == 0 or len(val_dataset) == 0:
                                self.logger.log_warning(f"Skipping probe training for {concept_type} L{layer_idx}T{tick_idx}K{kernel_size}S{seed_idx} due to empty dataset.")
                                continue

                            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.get('probing.batch_size'), shuffle=True)
                            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.get('probing.batch_size'), shuffle=False)

                            probe_trainer.train(train_dataloader, concept_type, layer_idx, tick_idx)
                            val_metrics = probe_trainer.evaluate(val_dataloader, concept_type, layer_idx, tick_idx, is_validation=True)
                            
                            # Store the probe in cache
                            if concept_type not in self.probes_cache: self.probes_cache[concept_type] = {}
                            if agent_model.__class__.__name__ not in self.probes_cache[concept_type]: self.probes_cache[concept_type][agent_model.__class__.__name__] = {}
                            if env_name not in self.probes_cache[concept_type][agent_model.__class__.__name__]: self.probes_cache[concept_type][agent_model.__class__.__name__][env_name] = {}
                            if layer_idx not in self.probes_cache[concept_type][agent_model.__class__.__name__][env_name]: self.probes_cache[concept_type][agent_model.__class__.__name__][env_name][layer_idx] = {}
                            if tick_idx not in self.probes_cache[concept_type][agent_model.__class__.__name__][env_name][layer_idx]: self.probes_cache[concept_type][agent_model.__class__.__name__][env_name][layer_idx][tick_idx] = {}
                            if kernel_size not in self.probes_cache[concept_type][agent_model.__class__.__name__][env_name][layer_idx][tick_idx]: self.probes_cache[concept_type][agent_model.__class__.__name__][env_name][layer_idx][tick_idx][kernel_size] = {}
                            self.probes_cache[concept_type][agent_model.__class__.__name__][env_name][layer_idx][tick_idx][kernel_size][seed_idx] = probe_model
                            self.logger.log_info(f"Probe stored for {concept_type} L{layer_idx}T{tick_idx}K{kernel_size}S{seed_idx}: Macro F1 {val_metrics['macro_f1']:.4f}")


                        # --- Train and Evaluate Baseline Probes ---
                        self.logger.log_info(f"Probing (Baseline) Layer {layer_idx}, Concept '{concept_type}', Kernel {kernel_size}x{kernel_size}...")
                        
                        # Baseline probes take raw observations as input
                        baseline_inputs_channels = env_obs_channels
                        if is_global_probe: # Flattened input for global probe
                            baseline_inputs_channels = env_obs_channels * input_spatial_dims[0] * input_spatial_dims[1]

                        for seed_idx in range(initialization_seeds):
                            random.seed(seed_idx + self.config.get('seed', 42))
                            torch.manual_seed(seed_idx + self.config.get('seed', 42))
                            np.random.seed(seed_idx + self.config.get('seed', 42))

                            baseline_probe_model = ProbeModel(
                                in_channels=baseline_inputs_channels,
                                num_classes=num_classes,
                                kernel_size=kernel_size,
                                is_global=is_global_probe
                            )
                            baseline_probe_trainer = ProbeTrainer(
                                probe_model=baseline_probe_model,
                                config=self.config,
                                logger=self.logger,
                                class_names=concept_class_names.get(concept_type, [])
                            )

                            baseline_train_dataset = ProbeDataset(
                                labeled_data=labeled_train_data, concept_key=concept_type, layer_idx=layer_idx,
                                tick_idx=tick_idx, is_baseline=True, input_spatial_dims=input_spatial_dims,
                                num_input_channels=baseline_inputs_channels, is_global_probe=is_global_probe
                            )
                            baseline_val_dataset = ProbeDataset(
                                labeled_data=labeled_val_data, concept_key=concept_type, layer_idx=layer_idx,
                                tick_idx=tick_idx, is_baseline=True, input_spatial_dims=input_spatial_dims,
                                num_input_channels=baseline_inputs_channels, is_global_probe=is_global_probe
                            )
                            
                            if len(baseline_train_dataset) == 0 or len(baseline_val_dataset) == 0:
                                self.logger.log_warning(f"Skipping baseline probe training for {concept_type} L{layer_idx}T{tick_idx}K{kernel_size}S{seed_idx} due to empty dataset.")
                                continue

                            baseline_train_dataloader = torch.utils.data.DataLoader(baseline_train_dataset, batch_size=self.config.get('probing.batch_size'), shuffle=True)
                            baseline_val_dataloader = torch.utils.data.DataLoader(baseline_val_dataset, batch_size=self.config.get('probing.batch_size'), shuffle=False)

                            baseline_probe_trainer.train(baseline_train_dataloader, f"Baseline_{concept_type}", layer_idx, tick_idx)
                            baseline_val_metrics = baseline_probe_trainer.evaluate(baseline_val_dataloader, f"Baseline_{concept_type}", layer_idx, tick_idx, is_validation=True)

                            # Store baseline probe in a separate cache or with a prefix
                            baseline_concept_key = f"Baseline_{concept_type}"
                            if baseline_concept_key not in self.probes_cache: self.probes_cache[baseline_concept_key] = {}
                            if agent_model.__class__.__name__ not in self.probes_cache[baseline_concept_key]: self.probes_cache[baseline_concept_key][agent_model.__class__.__name__] = {}
                            if env_name not in self.probes_cache[baseline_concept_key][agent_model.__class__.__name__]: self.probes_cache[baseline_concept_key][agent_model.__class__.__name__][env_name] = {}
                            if layer_idx not in self.probes_cache[baseline_concept_key][agent_model.__class__.__name__][env_name]: self.probes_cache[baseline_concept_key][agent_model.__class__.__name__][env_name][layer_idx] = {}
                            if tick_idx not in self.probes_cache[baseline_concept_key][agent_model.__class__.__name__][env_name][layer_idx]: self.probes_cache[baseline_concept_key][agent_model.__class__.__name__][env_name][layer_idx][tick_idx] = {}
                            if kernel_size not in self.probes_cache[baseline_concept_key][agent_model.__class__.__name__][env_name][layer_idx][tick_idx]: self.probes_cache[baseline_concept_key][agent_model.__class__.__name__][env_name][layer_idx][tick_idx][kernel_size] = {}
                            self.probes_cache[baseline_concept_key][agent_model.__class__.__name__][env_name][layer_idx][tick_idx][kernel_size][seed_idx] = baseline_probe_model
                            self.logger.log_info(f"Baseline probe stored for {concept_type} L{layer_idx}T{tick_idx}K{kernel_size}S{seed_idx}: Macro F1 {baseline_val_metrics['macro_f1']:.4f}")

        # After all probes are trained, generate Figures 4 and 40 (and 41, 42)
        self._plot_probe_results(agent_model, env_name)
        
        return self.probes_cache

    def _plot_probe_results(self, agent_model: BaseAgentModel, env_name: str) -> None:
        """Helper to plot and log probe results like Figure 4, 40, 41, 42."""
        agent_type_name = agent_model.__class__.__name__
        probe_kernel_sizes = self.config.get('probing.probe_kernel_sizes', [1, 3])
        num_layers_agent: int
        if isinstance(agent_model, DRCAgent):
            num_layers_agent = agent_model.D
            num_ticks_agent = agent_model.N + 1
            agent_channels = agent_model.convlstm_channels
        elif isinstance(agent_model, ResNetAgent):
            num_layers_agent = agent_model.num_residual_blocks
            num_ticks_agent = 1
            agent_channels = agent_model.block_channels
        else:
            raise ValueError(f"Unsupported agent type: {agent_type_name}")

        concepts_to_plot: List[str] = ['CA', 'CB'] # Main concepts for Figure 4
        
        # Collect F1 scores for plotting
        f1_data_fig4: Dict[str, Tuple[List[float], List[float]]] = {} # For Figure 4 (1x1, 3x3 for CA/CB)
        
        for concept_type in concepts_to_plot:
            for kernel_size in [1, 3]: # Figure 4 specifically for 1x1 and 3x3
                if kernel_size not in probe_kernel_sizes: continue # Skip if not trained

                for layer_idx in range(num_layers_agent):
                    # For Figure 4, assume we're looking at the final tick
                    tick_idx = num_ticks_agent - 1 # Corresponds to N for DRC or 0 for ResNet
                    
                    # Agent-based probe
                    probe_macro_f1s = [
                        self.probes_cache[concept_type][agent_type_name][env_name][layer_idx][tick_idx][kernel_size][seed_idx].best_val_macro_f1
                        for seed_idx in range(self.config.get('probing.initialization_seeds', 5))
                    ]
                    f1_data_fig4[f"{concept_type} L{layer_idx} K{kernel_size}"] = (
                        [np.mean(probe_macro_f1s)], [np.std(probe_macro_f1s)]
                    )

                    # Baseline probe
                    baseline_concept_key = f"Baseline_{concept_type}"
                    baseline_macro_f1s = [
                        self.probes_cache[baseline_concept_key][agent_type_name][env_name][layer_idx][tick_idx][kernel_size][seed_idx].best_val_macro_f1
                        for seed_idx in range(self.config.get('probing.initialization_seeds', 5))
                    ]
                    f1_data_fig4[f"Baseline {concept_type} L{layer_idx} K{kernel_size}"] = (
                        [np.mean(baseline_macro_f1s)], [np.std(baseline_macro_f1s)]
                    )
        
        # Plotting Figure 4
        fig = self.visualization_util.plot_macro_f1_curve(
            data=f1_data_fig4,
            x_label="Layer / Probe Type", # More dynamic label needed for this plot
            y_label="Macro F1 Score",
            title=f"Macro F1 for {agent_type_name} ({env_name}) - Fig 4"
        )
        self.logger.log_figure(f"fig4_probe_macro_f1_{agent_type_name}_{env_name}", fig)


        # Plotting Figure 40 (Larger Probes)
        if 5 in probe_kernel_sizes or 7 in probe_kernel_sizes:
            self.logger.log_info(f"Generating Figure 40 for {agent_type_name} ({env_name})...")
            f1_data_fig40: Dict[str, Tuple[List[float], List[float]]] = {}
            for concept_type in concepts_to_plot:
                for kernel_size in probe_kernel_sizes: # Use all trained kernel sizes
                    for layer_idx in range(num_layers_agent):
                        tick_idx = num_ticks_agent - 1
                        
                        probe_macro_f1s = [
                            self.probes_cache[concept_type][agent_type_name][env_name][layer_idx][tick_idx][kernel_size][seed_idx].best_val_macro_f1
                            for seed_idx in range(self.config.get('probing.initialization_seeds', 5))
                        ]
                        f1_data_fig40[f"{concept_type} L{layer_idx} K{kernel_size}"] = (
                            [np.mean(probe_macro_f1s)], [np.std(probe_macro_f1s)]
                        )
                        
                        baseline_concept_key = f"Baseline_{concept_type}"
                        baseline_macro_f1s = [
                            self.probes_cache[baseline_concept_key][agent_type_name][env_name][layer_idx][tick_idx][kernel_size][seed_idx].best_val_macro_f1
                            for seed_idx in range(self.config.get('probing.initialization_seeds', 5))
                        ]
                        f1_data_fig40[f"Baseline {concept_type} L{layer_idx} K{kernel_size}"] = (
                            [np.mean(baseline_macro_f1s)], [np.std(baseline_macro_f1s)]
                        )

            # This plotting function needs to be adapted to show kernel sizes on X-axis and layers as lines
            # For simplicity, will use the generic plot_macro_f1_curve and refine its usage.
            # A more specific plot function in visualization_util might be needed for Fig 4/40
            # or a custom plot in this runner.
            # Let's create a more specific plot for Figure 4/40 structure:
            self._plot_fig4_type(agent_type_name, env_name, concepts_to_plot, probe_kernel_sizes, num_layers_agent, num_ticks_agent-1)


    def _plot_fig4_type(self, agent_type_name: str, env_name: str, concepts: List[str], kernel_sizes: List[int], num_layers: int, tick_idx: int) -> None:
        """Helper to generate plots similar to Figure 4/40, showing F1 vs. layer for different kernel sizes."""
        fig, axes = plt.subplots(len(concepts), 2, figsize=(16, 6 * len(concepts)), squeeze=False)
        fig.suptitle(f"Macro F1 Scores for {agent_type_name} ({env_name}) - Probes", fontsize=16)

        for i, concept_type in enumerate(concepts):
            # Plot for agent's activations
            ax_agent = axes[i, 0]
            ax_agent.set_title(f"Agent Activations - {concept_type}")
            ax_agent.set_xlabel("Layer Index")
            ax_agent.set_ylabel("Macro F1")
            
            # Plot for baseline (raw observation)
            ax_baseline = axes[i, 1]
            ax_baseline.set_title(f"Baseline (Raw Obs) - {concept_type}")
            ax_baseline.set_xlabel("Layer Index")
            ax_baseline.set_ylabel("Macro F1")

            for k_size in kernel_sizes:
                means_agent_f1 = []
                stds_agent_f1 = []
                means_baseline_f1 = []
                stds_baseline_f1 = []

                for layer_idx in range(num_layers):
                    # Agent-based probes
                    try:
                        probe_macro_f1s = [
                            self.probes_cache[concept_type][agent_type_name][env_name][layer_idx][tick_idx][k_size][seed_idx].best_val_macro_f1
                            for seed_idx in range(self.config.get('probing.initialization_seeds', 5))
                        ]
                        means_agent_f1.append(np.mean(probe_macro_f1s))
                        stds_agent_f1.append(np.std(probe_macro_f1s))
                    except KeyError:
                        means_agent_f1.append(0.0)
                        stds_agent_f1.append(0.0)

                    # Baseline probes
                    try:
                        baseline_concept_key = f"Baseline_{concept_type}"
                        baseline_macro_f1s = [
                            self.probes_cache[baseline_concept_key][agent_type_name][env_name][layer_idx][tick_idx][k_size][seed_idx].best_val_macro_f1
                            for seed_idx in range(self.config.get('probing.initialization_seeds', 5))
                        ]
                        means_baseline_f1.append(np.mean(baseline_macro_f1s))
                        stds_baseline_f1.append(np.std(baseline_macro_f1s))
                    except KeyError:
                        means_baseline_f1.append(0.0)
                        stds_baseline_f1.append(0.0)
                
                # Plot for agent
                ax_agent.plot(range(num_layers), means_agent_f1, label=f'{k_size}x{k_size} Probe', marker='o')
                ax_agent.fill_between(range(num_layers), np.array(means_agent_f1) - np.array(stds_agent_f1),
                                      np.array(means_agent_f1) + np.array(stds_agent_f1), alpha=0.1)
                
                # Plot for baseline
                ax_baseline.plot(range(num_layers), means_baseline_f1, label=f'{k_size}x{k_size} Probe', marker='o')
                ax_baseline.fill_between(range(num_layers), np.array(means_baseline_f1) - np.array(stds_baseline_f1),
                                         np.array(means_baseline_f1) + np.array(stds_baseline_f1), alpha=0.1)
            
            ax_agent.legend()
            ax_agent.grid(True, linestyle='--', alpha=0.6)
            ax_baseline.legend()
            ax_baseline.grid(True, linestyle='--', alpha=0.6)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
        self.logger.log_figure(f"probe_f1_comparison_{agent_type_name}_{env_name}", fig)


    def run_plan_formation_experiments(self, agent_model: BaseAgentModel, env_name: str = None) -> None:
        """
        Investigates internal plan formation, iterative refinement, and behavioral evidence of search.
        Corresponds to Section 5 and Appendices A.1-A.3.

        Args:
            agent_model (BaseAgentModel): The agent model under investigation.
            env_name (str, optional): Name of the environment. Defaults to config default.
        """
        self.logger.log_info(f"--- Running Plan Formation Experiments for {type(agent_model).__name__} on {env_name if env_name else self.config.get('environment.name')} ---")
        
        env_name = env_name if env_name else self.config.get('environment.name')
        env = self.envs_cache.get(env_name)
        if env is None:
            raise ValueError(f"Environment '{env_name}' not found in cache. Ensure it's been initialized.")
        
        agent_type_name = agent_model.__class__.__name__

        num_layers_agent: int
        num_ticks_agent: int # Actual N, not N+1
        if isinstance(agent_model, DRCAgent):
            num_layers_agent = agent_model.D
            num_ticks_agent = agent_model.N
            agent_channels = agent_model.convlstm_channels
        elif isinstance(agent_model, ResNetAgent):
            num_layers_agent = agent_model.num_residual_blocks
            num_ticks_agent = 1 # ResNet always has 1 "tick" per block for concept extraction
            agent_channels = agent_model.block_channels
        else:
            raise ValueError(f"Unsupported agent type: {agent_type_name}")

        collector = TrajectoryCollector(agent_model=agent_model, env=env, config=self.config)
        labeler = ConceptLabeler(env=env)
        
        # We need 1x1 probes for CA and CB from the probes cache
        # For this experiment, we assume the probes have already been trained via run_probe_experiments
        concept_types_to_use = ['CA', 'CB']
        kernel_size_for_vis = 1 # Paper specifies 1x1 probes for visualization

        # --- 1. Iterative Plan Refinement (Quantitative - Figure 6, 22) ---
        self.logger.log_info("Investigating iterative plan refinement with thinking steps (Figure 6, 22)...")
        num_thinking_steps = 5
        eval_episodes = 100 # Reduced from 1000 for faster reproduction
        
        thinking_trajectories = collector.collect_trajectories(
            num_episodes=eval_episodes,
            behavior_policy=agent_model.act,
            store_cell_states=True,
            num_thinking_steps=num_thinking_steps,
            store_all_thinking_tick_states=True # This is crucial for collecting all N ticks
        )
        
        labeled_thinking_data = []
        for traj in tqdm(thinking_trajectories, desc="Labeling thinking step data"):
            labeled_thinking_data.extend(labeler.label_trajectory(traj))

        f1_data_per_layer_tick_concept: Dict[str, Dict[int, Dict[int, List[float]]]] = {} # concept -> layer -> tick -> [f1_scores]

        grid_h, grid_w = env.get_observation_space_shape()[:2]
        
        for concept_type in concept_types_to_use:
            f1_data_per_layer_tick_concept[concept_type] = {}
            for layer_idx in range(num_layers_agent):
                f1_data_per_layer_tick_concept[concept_type][layer_idx] = {}
                for tick_idx in range(num_ticks_agent + 1): # tick 0 to N
                    f1_data_per_layer_tick_concept[concept_type][layer_idx][tick_idx] = []

                    # For each tick, gather all collected cell states and labels
                    tick_data_points = []
                    for ts_data in labeled_thinking_data:
                        cell_states_at_layer = ts_data['cell_states_HWC_tensors'].get(f'layer_{layer_idx}')
                        if cell_states_at_layer:
                            cell_state_np = cell_states_at_layer.get(f'tick_{tick_idx}')
                            if cell_state_np is not None:
                                # Ensure only data points generated during thinking steps are used for this analysis
                                if ts_data['step_in_episode'] < num_thinking_steps: # Only first `num_thinking_steps` environment steps
                                    tick_data_points.append({
                                        'cell_states_HWC_tensors': {f'layer_{layer_idx}': {f'tick_{tick_idx}': cell_state_np}},
                                        'concept_labels': {concept_type: ts_data['concept_labels'][concept_type]},
                                        'observations': ts_data['observations'] # Placeholder, not used by ProbeDataset for cell states
                                    })
                        
                    if not tick_data_points:
                        self.logger.log_warning(f"No data points for {concept_type} L{layer_idx}T{tick_idx} in thinking steps. Skipping F1 calculation.")
                        continue
                    
                    # Need to train a dummy probe to get a fresh trainer for evaluation
                    dummy_probe_model = ProbeModel(
                        in_channels=agent_channels, num_classes=5, kernel_size=kernel_size_for_vis, is_global=False
                    )
                    dummy_probe_trainer = ProbeTrainer(
                        probe_model=dummy_probe_model, config=self.config, logger=self.logger,
                        class_names=['NEVER', 'UP', 'DOWN', 'LEFT', 'RIGHT'] # Default
                    )

                    # Now, for evaluation, we use the *pre-trained* probes from the cache
                    for seed_idx in range(self.config.get('probing.initialization_seeds', 5)):
                        try:
                            probe_model_for_eval = self.probes_cache[concept_type][agent_type_name][env_name][layer_idx][num_ticks_agent-1][kernel_size_for_vis][seed_idx]
                        except KeyError:
                            self.logger.log_warning(f"Probe not found in cache for {concept_type} L{layer_idx}T{num_ticks_agent-1}K{kernel_size_for_vis}S{seed_idx}. Skipping evaluation.")
                            continue

                        eval_dataset = ProbeDataset(
                            labeled_data=tick_data_points, concept_key=concept_type, layer_idx=layer_idx,
                            tick_idx=tick_idx, is_baseline=False, input_spatial_dims=(grid_h, grid_w),
                            num_input_channels=agent_channels, is_global_probe=False
                        )
                        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.config.get('probing.batch_size'), shuffle=False)
                        
                        # Use the pre-trained probe_model_for_eval in the dummy_probe_trainer
                        dummy_probe_trainer.probe_model = probe_model_for_eval
                        metrics = dummy_probe_trainer.evaluate(eval_dataloader, concept_type, layer_idx, tick_idx, is_validation=False)
                        f1_data_per_layer_tick_concept[concept_type][layer_idx][tick_idx].append(metrics['macro_f1'])
        
        # Plotting Figure 6 and 22
        # Data structure for visualization: Dict[str, Tuple[List[float], List[float]]]
        plot_data_fig6: Dict[str, Tuple[List[float], List[float]]] = {} # For final layer only
        plot_data_fig22: Dict[str, Tuple[List[float], List[float]]] = {} # For all layers

        for concept_type in concept_types_to_use:
            # Figure 6: Final layer (num_layers_agent-1)
            final_layer_data = f1_data_per_layer_tick_concept[concept_type][num_layers_agent-1]
            means = [np.mean(f1_scores) if f1_scores else 0.0 for f1_scores in final_layer_data.values()]
            stds = [np.std(f1_scores) if f1_scores else 0.0 for f1_scores in final_layer_data.values()]
            plot_data_fig6[f"{concept_type} (Final Layer)"] = (means, stds)

            # Figure 22: All layers
            for layer_idx in range(num_layers_agent):
                layer_data = f1_data_per_layer_tick_concept[concept_type][layer_idx]
                means = [np.mean(f1_scores) if f1_scores else 0.0 for f1_scores in layer_data.values()]
                stds = [np.std(f1_scores) if f1_scores else 0.0 for f1_scores in layer_data.values()]
                plot_data_fig22[f"{concept_type} L{layer_idx}"] = (means, stds)
        
        # Plot Figure 6
        fig_6 = self.visualization_util.plot_macro_f1_curve(
            data=plot_data_fig6,
            x_label="Internal Tick",
            y_label="Macro F1 Score",
            title=f"Iterative Plan Refinement (Fig 6) - {agent_type_name} {env_name}"
        )
        self.logger.log_figure(f"fig6_plan_refinement_{agent_type_name}_{env_name}", fig_6)

        # Plot Figure 22
        fig_22 = self.visualization_util.plot_macro_f1_curve(
            data=plot_data_fig22,
            x_label="Internal Tick",
            y_label="Macro F1 Score",
            title=f"Iterative Plan Refinement Across Layers (Fig 22) - {agent_type_name} {env_name}"
        )
        self.logger.log_figure(f"fig22_plan_refinement_layers_{agent_type_name}_{env_name}", fig_22)

        # --- 2. Behavioral Evidence of Search (Quantitative - Figure 24) ---
        self.logger.log_info("Investigating behavioral evidence of search (Figure 24)...")
        corridor_lengths = [2, 6, 10, 14] # As per Appendix A.3.2
        num_base_levels = 8
        num_transformations = len(self.level_generator.sokoban_transform_types) # 8 transformations
        
        all_corridor_levels: List[np.ndarray] = []
        # Generate all 8 * 4 * 8 = 256 levels
        for base_idx in range(num_base_levels):
            for length in corridor_lengths:
                for transform_type in self.level_generator.sokoban_transform_types:
                    corridor_level_config = {
                        'level_type': 'Corridor',
                        'seed': base_idx * 1000 + length, # Unique seed
                        'params': {'corridor_length': length, 'transform_type': transform_type}
                    }
                    all_corridor_levels.append(self.level_generator.generate_sokoban_level(**corridor_level_config))

        solved_percentages_data: Dict[str, Tuple[List[float], List[float]]] = {} # length -> ([percentages], [stds])
        
        impala_trainer = IMPALATrainer(agent_model=agent_model, env=env, config=self.config, logger=self.logger)

        for length in corridor_lengths:
            levels_for_length = [lvl for lvl in all_corridor_levels if self.level_generator.get_level_info(lvl).get('corridor_length') == length]
            
            percentages = []
            
            for thinking_steps_count in range(self.config.get('intervention.thinking_steps_for_cutoff', [0, 1, 2, 3, 4, 5])[-1] + 1):
                success_rate = impala_trainer.evaluate_behavior(levels_for_length, thinking_steps_count)
                percentages.append(success_rate * 100) # Convert to percentage
            
            # For plotting, combine all lengths into one dataset (different lines for different lengths)
            plot_data_fig24: Dict[str, Tuple[List[float], List[float]]] = {}
            for l_idx, current_length in enumerate(corridor_lengths):
                levels_of_this_length = [lvl for lvl in all_corridor_levels if self.level_generator.get_level_info(lvl).get('corridor_length') == current_length]
                percentages_this_length = []
                for thinking_steps_count in range(self.config.get('intervention.thinking_steps_for_cutoff', [0, 1, 2, 3, 4, 5])[-1] + 1):
                    success_rate = impala_trainer.evaluate_behavior(levels_of_this_length, thinking_steps_count)
                    percentages_this_length.append(success_rate * 100)
                plot_data_fig24[f"Length {current_length}"] = (percentages_this_length, [0.0] * len(percentages_this_length)) # No std dev collected for this

            fig_24 = self.visualization_util.plot_macro_f1_curve(
                data=plot_data_fig24,
                x_label="Number of Thinking Steps",
                y_label="Percentage of Levels Solved (%)",
                title=f"Behavioral Evidence of Search (Fig 24) - {agent_type_name} {env_name}"
            )
            self.logger.log_figure(f"fig24_behavioral_search_{agent_type_name}_{env_name}", fig_24)


        # --- 3. Qualitative Plan Visualization (Figures 5, 10-21, 25) ---
        # This part is highly qualitative and involves selecting specific examples.
        # We will implement a simplified version for a few examples.
        self.logger.log_info("Generating qualitative plan visualizations (Figures 5, 10-21, 25)...")

        example_levels_configs: List[Dict[str, Any]] = [
            # Example from Figure 5a-c
            {'level_type': 'Handcrafted_Fig5_A', 'seed': 0, 'description': 'Fig 5a'},
            {'level_type': 'Handcrafted_Fig5_B', 'seed': 0, 'description': 'Fig 5b'},
            {'level_type': 'Handcrafted_Fig5_C', 'seed': 0, 'description': 'Fig 5c'},
            # Example from Figure 1a (Evaluative Planning)
            {'level_type': 'Handcrafted_Fig1_A', 'seed': 0, 'description': 'Fig 1a - Evaluative'},
            # Example from Figure 25 (Corridor level with thinking steps)
            {'level_type': 'Corridor', 'seed': 0, 'params': {'corridor_length': 14, 'transform_type': 'none'}, 'description': 'Fig 25 - Corridor'}
        ]
        
        num_steps_to_visualize = 5 # Visualize first 5 environment steps
        num_ticks_to_visualize = num_ticks_agent + 1 # All internal ticks + initial state

        for lvl_config in example_levels_configs:
            self.logger.log_info(f"Visualizing for level: {lvl_config['description']}")
            
            # Generate the specific level for visualization
            initial_state_np = self.level_generator.generate_sokoban_level(lvl_config['level_type'], lvl_config['seed'], lvl_config.get('params', {}))

            # Collect a detailed trajectory for this single level
            # We need all cell states at all ticks and observations
            vis_traj_list = collector.collect_trajectories(
                num_episodes=1,
                behavior_policy=agent_model.act,
                store_cell_states=True,
                level_configs=[{'initial_state': initial_state_np, 'level_type': lvl_config['level_type'], 'seed': lvl_config['seed'], 'params': lvl_config.get('params',{})}],
                num_thinking_steps=num_thinking_steps if 'Fig 25' in lvl_config['description'] else 0, # Only apply thinking steps for relevant examples
                store_all_thinking_tick_states=True # Crucial
            )
            vis_traj = vis_traj_list[0] # Only one trajectory collected
            
            # Label the trajectory
            labeled_vis_data = labeler.label_trajectory(vis_traj)

            # Create a figure for this example
            fig, axes = plt.subplots(num_steps_to_visualize, num_ticks_to_visualize * num_layers_agent, figsize=(num_ticks_to_visualize * num_layers_agent * 2, num_steps_to_visualize * 2))
            if num_steps_to_visualize == 1: axes = np.expand_dims(axes, axis=0) # Handle single row case
            if num_ticks_to_visualize * num_layers_agent == 1: axes = np.expand_dims(axes, axis=1) # Handle single column case

            plot_title = f"Plan Visualization: {lvl_config['description']}"
            fig.suptitle(plot_title, fontsize=16)

            for step_idx in range(min(num_steps_to_visualize, len(labeled_vis_data))):
                timestep_data = labeled_vis_data[step_idx]
                current_obs = timestep_data['observations']
                
                # Check if it's during thinking steps and adjust tick_data
                is_thinking_step = (step_idx < num_thinking_steps) and (num_thinking_steps > 0)
                
                for tick_offset in range(num_ticks_to_visualize):
                    tick_to_extract = tick_offset # 0 is s_t-1, 1 to N are s_t,1 to s_t,N
                    if not is_thinking_step:
                         # If not in thinking steps, just visualize the final tick (tick N)
                         if tick_offset > 0: continue # Only show tick 0 and final tick
                         tick_to_extract = num_ticks_agent # Always show final tick

                    for layer_idx in range(num_layers_agent):
                        col_idx = (tick_offset * num_layers_agent) + layer_idx
                        if col_idx >= axes.shape[1]: continue # Ensure we don't go out of bounds

                        ax = axes[step_idx, col_idx]
                        self.visualization_util.plot_sokoban_board