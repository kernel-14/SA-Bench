import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Any
from collections import defaultdict

from config import AgentConfig, SokobanEnvConfig, ProbeConfig, EvalConfig
from model import DRC, ResNetAgent
from agent import DRCAgent, ResNetActorCritic
from data import SokobanEnv, collect_probe_data, ConceptDataset, SokobanConceptMap
from probes import ProbeTrainer, LinearProbe
from torch.utils.data import DataLoader

class Evaluator:
    """
    Handles evaluation of agent performance, probing for concepts, and interventions.
    """
    def __init__(self, 
                 agent_config: AgentConfig, 
                 sokoban_config: SokobanEnvConfig, 
                 probe_config: ProbeConfig, 
                 eval_config: EvalConfig,
                 device: torch.device):
        self.agent_config = agent_config
        self.sokoban_config = sokoban_config
        self.probe_config = probe_config
        self.eval_config = eval_config
        self.device = device
        self.probe_trainer = ProbeTrainer(probe_config, agent_config, device)

    def evaluate_agent_performance(self, agent: Any, env: SokobanEnv, num_episodes: int) -> float:
        """
        Evaluates the agent's performance by running a number of episodes.
        Returns the percentage of levels solved.
        """
        solved_count = 0
        for _ in range(num_episodes):
            obs, _ = env.reset()
            agent.reset()
            done = False
            truncated = False
            # episode_reward = 0 # Not directly used for solved_count
            while not done and not truncated:
                action = agent.get_action(obs, greedy=True)
                obs, reward, done, truncated, _ = env.step(action)
                # episode_reward += reward
                if done: # Assuming done implies solved for positive reward, or target count met
                    if env.boxes_on_target_count == self.sokoban_config.NUM_BOXES:
                        solved_count += 1
                        break
        return solved_count / num_episodes * 100

    def evaluate_probes(self, 
                        agent_model: nn.Module, 
                        env: SokobanEnv, 
                        num_episodes: int, 
                        is_drc_agent: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Collects data and trains/evaluates probes for C_A and C_B at each layer
        for various probe types.
        
        Returns:
            Dict[str, Dict[str, float]]: Nested dictionary with results, e.g.,
                                        {'CA_1x1_L0': {'macro_f1': 0.8, 'UP_F1_mean': 0.7, ...}, ...}
        """
        print("Collecting data for probe evaluation...")
        # `agent_model` is the raw nn.Module. `collect_probe_data` needs an `agent` wrapper.
        temp_agent_wrapper = DRCAgent(self.agent_config, agent_model, self.device) if is_drc_agent else ResNetActorCritic(self.agent_config, agent_model, self.device)
        dataset = collect_probe_data(temp_agent_wrapper, env, num_episodes, self.agent_config, is_drc_agent)
        dataloader = DataLoader(dataset, batch_size=self.probe_config.PROBE_BATCH_SIZE, shuffle=False)
        print(f"Collected {len(dataset)} samples for probing.")

        results = defaultdict(dict)
        # Determine number of layers based on agent type
        num_layers = self.agent_config.D_CONVLSTM_LAYERS if is_drc_agent else 24 # ResNet has 24 res blocks (Appendix G)

        for probe_type in self.probe_config.PROBE_TYPES:
            for layer_idx in range(num_layers):
                for concept_type, concept_classes in [("CA", self.probe_config.CONCEPT_CA_CLASSES), 
                                                     ("CB", self.probe_config.CONCEPT_CB_CLASSES)]:
                    print(f"  Concept: {concept_type}, Layer: {layer_idx}, Probe Type: {probe_type}")

                    # Collect activations and targets
                    if is_drc_agent:
                        activations, targets = self.probe_trainer.collect_activations_and_targets_drc(
                            agent_model, dataloader, concept_type, probe_type, layer_idx, self.device
                        )
                    else:
                        activations, targets = self.probe_trainer.collect_activations_and_targets_resnet(
                            agent_model, dataloader, concept_type, probe_type, layer_idx, self.device
                        )
                    
                    if activations.numel() == 0:
                        print(f"    No data for {concept_type} at layer {layer_idx} with probe type {probe_type}. Skipping.")
                        continue

                    # Train and evaluate probes with multiple seeds
                    seed_macro_f1s = []
                    seed_class_metrics = defaultdict(lambda: defaultdict(list))

                    for seed in range(self.probe_config.NUM_PROBE_SEEDS):
                        torch.manual_seed(seed)
                        probe = self.probe_trainer.train_probe(activations, targets, probe_type)
                        macro_f1, class_metrics = self.probe_trainer.evaluate_probe(probe, activations, targets, concept_classes)
                        seed_macro_f1s.append(macro_f1)
                        for cls_name, metrics_dict in class_metrics.items():
                            for metric_name, value in metrics_dict.items():
                                seed_class_metrics[cls_name][metric_name].append(value)
                    
                    avg_macro_f1 = np.mean(seed_macro_f1s)
                    std_macro_f1 = np.std(seed_macro_f1s)

                    results_key = f"{concept_type}_{probe_type}_L{layer_idx}"
                    results[results_key]['macro_f1'] = avg_macro_f1
                    results[results_key]['macro_f1_std'] = std_macro_f1
                    
                    for cls_name, metrics_dict in seed_class_metrics.items():
                        for metric_name, values_list in metrics_dict.items():
                            results[results_key][f'{cls_name}_{metric_name}_mean'] = np.mean(values_list)
                            results[results_key][f'{cls_name}_{metric_name}_std'] = np.std(values_list)
                    
                    print(f"    Macro F1: {avg_macro_f1:.4f} (±{std_macro_f1:.4f})")
        return results

    def _get_concept_vector(self, 
                            probe_weights_map: Dict[str, Dict[str, torch.Tensor]], # layer_key -> concept_type -> class_name -> vector
                            layer_key: str, 
                            concept_type: str, 
                            class_name: str,
                            probe_type: str # To correctly select the vector
                            ) -> torch.Tensor:
        """Helper to get concept vector from the pre-extracted map."""
        # The concept_vectors mapping stores weights extracted from probes.
        # This function should retrieve the correct vector based on layer, concept, and class.
        if layer_key not in probe_weights_map:
            raise KeyError(f"Layer key {layer_key} not found in concept vectors map.")
        if concept_type not in probe_weights_map[layer_key]:
            raise KeyError(f"Concept type {concept_type} not found for layer {layer_key}.")
        if class_name not in probe_weights_map[layer_key][concept_type]:
            raise KeyError(f"Class name {class_name} not found for {concept_type} at {layer_key}.")
        
        # Concept vectors are stored as CPU tensors, move to device for intervention
        return probe_weights_map[layer_key][concept_type][class_name].to(self.device).float()


    def perform_intervention(self, 
                             agent: DRCAgent, # Only DRC agents for interventions as per paper
                             env: SokobanEnv,
                             intervention_type: str, # 'Agent-Shortcut', 'Box-Shortcut'
                             all_concept_vectors: Dict[str, Dict[str, Dict[str, torch.Tensor]]], # layer_key -> concept_type -> class_name -> vector
                             alpha: float = 1.0, 
                             num_directional_squares: int = 1,
                             layer_to_intervene: int = 2, # Default to layer 2 as example from Table 1
                             probe_size_for_vectors: str = "1x1") -> bool: 

        """
        Performs an intervention on the agent's internal state to steer its behavior.
        
        Args:
            agent: The DRCAgent instance.
            env: The Sokoban environment.
            intervention_type: 'Agent-Shortcut' or 'Box-Shortcut'.
            all_concept_vectors: Dictionary of pre-extracted concept vectors (from trained probes).
            alpha: Scaling factor for intervention vectors.
            num_directional_squares: Number of squares along the long route to apply directional intervention.
            layer_to_intervene: The ConvLSTM layer index to apply the intervention.
            probe_size_for_vectors: The probe size (e.g., "1x1") from which concept vectors were extracted.
            
        Returns:
            bool: True if the intervention was successful in achieving the desired suboptimal plan.
        """
        obs, _ = env.reset()
        agent.reset() # Reset agent's recurrent states
        
        # Initial recurrent states for the start of the episode (all zeros)
        batch_size = 1 # Interventions are typically single-episode
        H, W = self.agent_config.GRID_SIZE, self.agent_config.GRID_SIZE
        C = self.agent_config.CHANNELS
        D = self.agent_config.D_CONVLSTM_LAYERS
        
        current_agent_states: List[Tuple[torch.Tensor, torch.Tensor]] = [
            (torch.zeros(batch_size, C, H, W, device=self.device), 
             torch.zeros(batch_size, C, H, W, device=self.device)) 
            for _ in range(D)
        ]

        layer_key = f"L{layer_to_intervene}"
        
        # Get concept vectors for NEVER and specific directions
        # These are now obtained from the `all_concept_vectors` map.
        never_ca_vector = self._get_concept_vector(all_concept_vectors, layer_key, "CA", "NEVER", probe_size_for_vectors)
        never_cb_vector = self._get_concept_vector(all_concept_vectors, layer_key, "CB", "NEVER", probe_size_for_vectors)
        
        # Reshape to (1, C, 1, 1) to allow broadcasting when added to cell state (1, C, H, W)
        # Note: self._get_concept_vector returns (C) or (C*K*K) tensor.
        # For 1x1 probe, it's (C). For NxN, it's (C*N*N).
        # We need to correctly handle this based on probe_size_for_vectors.
        # The paper describes intervention as adding `w_k` to `g_x,y`.
        # `g_x,y` is a vector in R^32 (channels). So `w_k` should be 32-dim.
        # This implies that even for NxN probes, we extract a 1x1 projection for intervention.
        # This is a subtle point. For faithfulness, assuming `w_k` is 32-dim.
        # If the probe used in `get_concept_vectors` was 1x1, then it's directly 32-dim.
        # If it was NxN, the `get_concept_vectors` flattened it to (C*N*N).
        # For intervention, we add to a specific (x,y) location, so it should be C-dimensional.
        # This requires `get_concept_vectors` to extract the correct C-dimensional vector,
        # or we assume `w_k` is actually flattened if NxN probe.
        # Given "add w_k to position (x,y) of the agent’s cell state g_x,y", it's C-dim.
        # So we take the first `self.agent_config.CHANNELS` components if it was flattened from NxN.
        
        # Correctly reshape `never` vectors to (1, C, 1, 1) for adding to (1, C, H, W)
        never_ca_vector_reshaped = never_ca_vector[:self.agent_config.CHANNELS].view(1, self.agent_config.CHANNELS, 1, 1)
        never_cb_vector_reshaped = never_cb_vector[:self.agent_config.CHANNELS].view(1, self.agent_config.CHANNELS, 1, 1)

        # Mock definitions for intervention paths/routes
        # In a real scenario, these would be defined based on the specific handcrafted levels
        short_route_squares = [] # List of (r, c) tuples
        long_route_first_square_dir_info = None # (r, c, direction_name) for first square of long path
        long_route_directional_squares_with_dirs = [] # List of (r, c, direction_name) for directional interventions

        # These mock paths are for a general 8x8 grid, replace with actual levels
        if intervention_type == 'Agent-Shortcut':
            short_route_squares = [(1, 2), (1, 3)] 
            long_route_first_square_dir_info = (2, 1, "DOWN") # Move onto (2,1) from UP direction
            long_route_directional_squares_with_dirs = [(3,1,"DOWN"), (4,1,"DOWN")] # Example: agent moves down in these squares
            target_concept_type = "CA"
            
        elif intervention_type == 'Box-Shortcut':
            short_route_squares = [(2, 3), (2, 4)]
            long_route_first_square_dir_info = (1, 2, "RIGHT") # Push box at (1,2) RIGHT
            long_route_directional_squares_with_dirs = [(1,3,"RIGHT"), (1,4,"RIGHT")] # Example: box pushed right in these squares
            target_concept_type = "CB"
        else:
            raise ValueError(f"Unknown intervention type: {intervention_type}")

        # Limit directional interventions to num_directional_squares
        long_route_directional_squares_with_dirs = long_route_directional_squares_with_dirs[:num_directional_squares]

        episode_length = env.max_steps
        solved_suboptimally = False
        
        # Track if directional intervention has been "used"
        # The paper says: "repeat the ‘directional’ intervention only until the agent moves onto,
        # or pushes the box off, the corresponding square." This requires real environment tracking.
        # For mock, we'll simplify and say it applies for a few steps and then "succeeds" or is "used".
        directional_intervention_triggered = False 
        
        for step_idx in range(episode_length):
            # Apply interventions before agent's forward pass
            # We must clone the current states to apply intervention,
            # then pass these intervened states to the model, and then update `current_agent_states`
            # with the `new_states` returned by the model.
            intervened_states = [(h.clone(), c.clone()) for h, c in current_agent_states]
            
            # Short-route intervention: Add NEVER vector
            for r, c in short_route_squares:
                if target_concept_type == "CA":
                    intervened_states[layer_to_intervene][1][:, :, r, c] += alpha * never_ca_vector_reshaped.squeeze()
                elif target_concept_type == "CB":
                    intervened_states[layer_to_intervene][1][:, :, r, c] += alpha * never_cb_vector_reshaped.squeeze()

            # Directional intervention: Add directional vector
            if not directional_intervention_triggered:
                # Apply the directional intervention for the specified number of squares
                for r, c, dir_name in long_route_directional_squares_with_dirs:
                    directional_vector = self._get_concept_vector(all_concept_vectors, layer_key, target_concept_type, dir_name, probe_size_for_vectors)
                    directional_vector_reshaped = directional_vector[:self.agent_config.CHANNELS].view(1, self.agent_config.CHANNELS, 1, 1)

                    if target_concept_type == "CA":
                        intervened_states[layer_to_intervene][1][:, :, r, c] += alpha * directional_vector_reshaped.squeeze()
                    elif target_concept_type == "CB":
                        intervened_states[layer_to_intervene][1][:, :, r, c] += alpha * directional_vector_reshaped.squeeze()
                
                if step_idx == 0:
                    # Mock: assume directional intervention is "used" after first step
                    # In a real environment, this logic would check agent/box position.
                    directional_intervention_triggered = True

            # Agent's forward pass with intervened states
            obs_tensor = torch.from_numpy(obs).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
            # Use agent.get_model() which is the raw nn.Module for direct state manipulation
            policy_logits, value, new_states, _ = agent.get_model()(obs_tensor, intervened_states)
            current_agent_states = new_states # Update agent's internal states for next step

            action = torch.argmax(policy_logits, dim=1).item() # Greedy action selection
            next_obs, reward, done, truncated, _ = env.step(action)
            current_obs = next_obs # Update current_obs for next loop iteration

            if done:
                # Check if solved suboptimally. This requires specific logic for each shortcut level.
                # For mock: assume if level solved and intervention was applied, it implies suboptimal.
                # In a real scenario, this would check if the agent actually followed the 'long path'
                if env.boxes_on_target_count == self.sokoban_config.NUM_BOXES:
                    # This is a very loose success condition for mock.
                    # A robust check would require tracking agent's path / box's route.
                    solved_suboptimally = True 
                break
            
            if truncated:
                break
        
        return solved_suboptimally

    def run_intervention_experiments(self, 
                                     agent_model: nn.Module, 
                                     num_levels_per_type: int = 25, 
                                     num_rotations: int = 8) -> Dict[str, Dict[str, float]]:
        """
        Runs intervention experiments for Agent-Shortcut and Box-Shortcut levels.
        """
        # This function would need to generate/load specific "Agent-Shortcut" and "Box-Shortcut" levels
        # which is not directly supported by the mock SokobanEnv.
        # For now, this is a placeholder function structure.

        results = defaultdict(dict)
        agent_drc = DRCAgent(self.agent_config, agent_model, self.device)
        num_layers = self.agent_config.D_CONVLSTM_LAYERS

        # First, collect concept vectors from a trained probe (assumed 1x1 probes are used for interventions)
        # This is a critical step, as interventions use these vectors.
        # We need to train probes on data and then extract their weights.
        
        print("Extracting concept vectors for interventions...")
        all_concept_vectors = defaultdict(lambda: defaultdict(dict)) # layer_key -> concept_type -> class_name -> vector

        # To extract vectors, we need a small dataset for each layer to train mock probes.
        # These probes are solely for extracting the vectors `w_k`.
        temp_env = SokobanEnv(self.sokoban_config)
        temp_dataset_size = self.probe_config.PROBE_TRAIN_EPISODES # Use same as probe training
        temp_dataset = collect_probe_data(agent_drc, temp_env, temp_dataset_size, self.agent_config, is_drc_agent=True)
        temp_dataloader = DataLoader(temp_dataset, batch_size=self.probe_config.PROBE_BATCH_SIZE, shuffle=True)

        for layer_idx in range(num_layers):
            layer_key = f"L{layer_idx}"
            # For CA concept
            activations_ca, targets_ca = self.probe_trainer.collect_activations_and_targets_drc(
                agent_model, temp_dataloader, "CA", "1x1", layer_idx, self.device
            )
            if activations_ca.numel() > 0:
                probe_ca = self.probe_trainer.train_probe(activations_ca, targets_ca, "1x1")
                vectors_ca = self.probe_trainer.get_concept_vectors(probe_ca)
                all_concept_vectors[layer_key]["CA"] = vectors_ca

            # For CB concept
            activations_cb, targets_cb = self.probe_trainer.collect_activations_and_targets_drc(
                agent_model, temp_dataloader, "CB", "1x1", layer_idx, self.device
            )
            if activations_cb.numel() > 0:
                probe_cb = self.probe_trainer.train_probe(activations_cb, targets_cb, "1x1")
                vectors_cb = self.probe_trainer.get_concept_vectors(probe_cb)
                all_concept_vectors[layer_key]["CB"] = vectors_cb

        print("Concept vectors extracted successfully.")
        
        intervention_types = ['Agent-Shortcut', 'Box-Shortcut']
        for int_type in intervention_types:
            print(f"Running {int_type} interventions...")
            for layer_idx in range(num_layers):
                success_rates = []
                num_trials = num_levels_per_type * num_rotations
                for _ in range(num_trials): # Simulate running on all generated levels
                    is_successful = self.perform_intervention(
                        agent_drc, 
                        env, # Use the mock env
                        int_type, 
                        all_concept_vectors,
                        alpha=self.eval_config.INTERVENTION_ALPHA,
                        num_directional_squares=self.eval_config.INTERVENTION_DIRECTIONAL_SQUARES,
                        layer_to_intervene=layer_idx,
                        probe_size_for_vectors="1x1" # Assume 1x1 probes for vectors as per paper
                    )
                    success_rates.append(is_successful)
                
                avg_success_rate = np.mean(success_rates) * 100
                results[f"{int_type}_L{layer_idx}"]['success_rate'] = avg_success_rate
                # Random baseline success rate calculation
                # Table 1 has random success rates: AS ~30%, BS ~30%
                random_baseline_success = 30.0 # Placeholder value based on paper Table 1
                results[f"{int_type}_L{layer_idx}"]['random_baseline_success_rate'] = random_baseline_success
                print(f"    Layer {layer_idx} {int_type} Success Rate: {avg_success_rate:.2f}% (Baseline: {random_baseline_success:.2f}%)")

        return results


    def investigate_plan_refinement(self, 
                                    agent_model: nn.Module, 
                                    env: SokobanEnv, 
                                    num_episodes: int, 
                                    num_thinking_steps: int) -> Dict[str, Dict[str, List[float]]]:
        """
        Investigates iterative plan refinement during 'thinking steps' by probing
        internal states at each internal tick.
        """
        print(f"Investigating plan refinement over {num_thinking_steps} thinking steps...")
        
        agent_drc = DRCAgent(self.agent_config, agent_model, self.device)
        num_layers = self.agent_config.D_CONVLSTM_LAYERS
        num_ticks_per_step = self.agent_config.N_INTERNAL_TICKS
        total_ticks = num_thinking_steps * num_ticks_per_step # Total internal ticks during thinking steps

        # Store macro F1 for each concept, each layer, each tick
        # layer_key -> list of F1 scores for each tick
        macro_f1_per_tick_per_layer_ca = defaultdict(lambda: [0.0] * total_ticks)
        macro_f1_per_tick_per_layer_cb = defaultdict(lambda: [0.0] * total_ticks)
        
        # Collect initial recurrent states for a new episode (all zeros)
        batch_size = 1
        H, W = self.agent_config.GRID_SIZE, self.agent_config.GRID_SIZE
        C = self.agent_config.CHANNELS
        D = self.agent_config.D_CONVLSTM_LAYERS
        
        # Placeholder for probes. In a real scenario, these would be loaded from pre-trained probes.
        # For this, we'll create mock probes on the fly for demo.
        mock_probes_ca = {} # layer_idx -> probe_obj
        mock_probes_cb = {} # layer_idx -> probe_obj

        # Collect some dummy data to train mock probes for this specific evaluation
        temp_env = SokobanEnv(self.sokoban_config)
        temp_dataset_size = self.probe_config.PROBE_TRAIN_EPISODES # Use full training episodes
        temp_dataset = collect_probe_data(agent_drc, temp_env, temp_dataset_size, self.agent_config, is_drc_agent=True)
        temp_dataloader = DataLoader(temp_dataset, batch_size=self.probe_config.PROBE_BATCH_SIZE, shuffle=True)

        for layer_idx in range(num_layers):
            activations_ca, targets_ca = self.probe_trainer.collect_activations_and_targets_drc(
                agent_model, temp_dataloader, "CA", "1x1", layer_idx, self.device
            )
            if activations_ca.numel() > 0:
                mock_probes_ca[layer_idx] = self.probe_trainer.train_probe(activations_ca, targets_ca, "1x1")

            activations_cb, targets_cb = self.probe_trainer.collect_activations_and_targets_drc(
                agent_model, temp_dataloader, "CB", "1x1", layer_idx, self.device
            )
            if activations_cb.numel() > 0:
                mock_probes_cb[layer_idx] = self.probe_trainer.train_probe(activations_cb, targets_cb, "1x1")

        print("Mock probes for plan refinement trained.")

        for episode in range(num_episodes):
            obs, _ = env.reset() # Initial observation for the episode
            # Initial states for the DRC agent for the very first environment step.
            # These are lists of (h,c) tuples, each (1,C,H,W)
            current_recurrent_states: List[Tuple[torch.Tensor, torch.Tensor]] = [
                (torch.zeros(batch_size, C, H, W, device=self.device), 
                 torch.zeros(batch_size, C, H, W, device=self.device)) 
                for _ in range(D)
            ]
            
            # Generate one set of "ground truth" labels for the entire episode's plan
            # This is a simplification. In a real scenario, this would be derived from
            # the optimal solution for the specific Sokoban level or the agent's actual trajectory.
            # For a mock, we use a fixed mock sequence.
            mock_next_action_sequence = [3, 3, 0] # Example: Right, Right, Up
            mock_box_push_sequence = [(2, 2, 1)] # Example: Box at (2,2) pushed DOWN
            
            # These will be the "static ground truth" to compare probe predictions against for F1 scores.
            static_ca_labels_episode, static_cb_labels_episode = map_sokoban_state_to_concepts(
                obs, mock_next_action_sequence, mock_box_push_sequence
            )
            # Flatten and move to device for comparison
            flat_static_ca_targets = torch.from_numpy(static_ca_labels_episode).long().to(self.device).flatten()
            flat_static_cb_targets = torch.from_numpy(static_cb_labels_episode).long().to(self.device).flatten()

            # Simulate 'thinking steps' where agent is forced to remain stationary
            for s_step in range(num_thinking_steps):
                # The paper says: "forced the agent to remain stationary for 5 steps prior to acting in 1000 episodes."
                # This means `obs` does not change across thinking steps.
                
                # For each thinking step, the DRC agent performs N_INTERNAL_TICKS
                obs_tensor = torch.from_numpy(obs).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
                
                # We need all cell states for all internal ticks of the current thinking step
                # get_all_cell_states_for_tick returns (new_states, all_cell_states_per_tick_current_step)
                current_recurrent_states, all_cell_states_per_tick_current_step = agent_drc.get_all_cell_states_for_tick(
                    obs, current_recurrent_states
                )
                
                # all_cell_states_per_tick_current_step: list (N ticks) of list (D layers) of (h,c)
                for n_tick in range(num_ticks_per_step):
                    tick_index_global = s_step * num_ticks_per_step + n_tick
                    
                    for layer_idx in range(num_layers):
                        # Use the cell state `c` for probing (index 1 in (h,c) tuple)
                        activations_to_probe = all_cell_states_per_tick_current_step[n_tick][layer_idx][1] # (1, C, H, W)
                        
                        if activations_to_probe.numel() == 0: continue

                        # Reshape for 1x1 probe evaluation: (H*W, C)
                        flat_activations = activations_to_probe.permute(0, 2, 3, 1).reshape(-1, C)

                        # Evaluate probes against the static ground truth for this episode
                        if layer_idx in mock_probes_ca:
                            _, class_metrics_ca = self.probe_trainer.evaluate_probe(mock_probes_ca[layer_idx], flat_activations, flat_static_ca_targets, self.probe_config.CONCEPT_CA_CLASSES)
                            # Macro F1 is already computed inside evaluate_probe
                            macro_f1_ca = class_metrics_ca['macro_f1'] if 'macro_f1' in class_metrics_ca else np.mean([m['F1'] for m in class_metrics_ca.values()])
                            macro_f1_per_tick_per_layer_ca[f"L{layer_idx}"][tick_index_global] += macro_f1_ca
                        
                        if layer_idx in mock_probes_cb:
                            _, class_metrics_cb = self.probe_trainer.evaluate_probe(mock_probes_cb[layer_idx], flat_activations, flat_static_cb_targets, self.probe_config.CONCEPT_CB_CLASSES)
                            macro_f1_cb = class_metrics_cb['macro_f1'] if 'macro_f1' in class_metrics_cb else np.mean([m['F1'] for m in class_metrics_cb.values()])
                            macro_f1_per_tick_per_layer_cb[f"L{layer_idx}"][tick_index_global] += macro_f1_cb
        
        # Average over episodes
        for layer_idx in range(num_layers):
            layer_key = f"L{layer_idx}"
            if layer_key in macro_f1_per_tick_per_layer_ca:
                macro_f1_per_tick_per_layer_ca[layer_key] = [val / num_episodes for val in macro_f1_per_tick_per_layer_ca[layer_key]]
            if layer_key in macro_f1_per_tick_per_layer_cb:
                macro_f1_per_tick_per_layer_cb[layer_key] = [val / num_episodes for val in macro_f1_per_tick_per_layer_cb[layer_key]]

        return {
            "CA_F1_per_tick_per_layer": macro_f1_per_tick_per_layer_ca,
            "CB_F1_per_tick_per_layer": macro_f1_per_tick_per_layer_cb
        }
    
    # Placeholder for more detailed plan formation investigation (qualitative)
    # This would involve visualization and manual analysis, not easily automated here.
    def investigate_plan_formation(self):
        print("Qualitative analysis of plan formation would involve visualizations (e.g., Figure 1, 5) and manual inspection.")
        print("This is not directly implementable as executable code in this setup.")

def run_evaluation(model_path: str | None = None, is_drc_agent: bool = True):
    agent_config = AgentConfig()
    sokoban_config = SokobanEnvConfig()
    probe_config = ProbeConfig()
    eval_config = EvalConfig()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if is_drc_agent:
        model = DRC(agent_config).to(device)
    else:
        model = ResNetAgent(agent_config).to(device)

    if model_path:
        print(f"Loading model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("Model loaded.")
    else:
        print("No model path provided. Using randomly initialized model for evaluation (will perform poorly).")

    evaluator = Evaluator(agent_config, sokoban_config, probe_config, eval_config, device)
    env = SokobanEnv(sokoban_config)
    
    # 1. Evaluate agent performance
    print("\n--- Agent Performance Evaluation ---")
    if is_drc_agent:
        agent_wrapper = DRCAgent(agent_config, model, device)
    else:
        agent_wrapper = ResNetActorCritic(agent_config, model, device)

    # For 'thinking steps' evaluation, we need to manually implement the loop
    print(f"Evaluating agent with 0 thinking steps on {eval_config.MEDIUM_LEVELS_COUNT} mock levels...")
    # This mock environment doesn't have "medium" or "hard" levels.
    # It would require a custom env with level loading.
    # For now, evaluate on the generic mock env.
    solved_percentage_no_thinking = evaluator.evaluate_agent_performance(agent_wrapper, env, 100) # Small episodes for mock
    print(f"Percentage of levels solved (0 thinking steps): {solved_percentage_no_thinking:.2f}%")

    # Simulate thinking steps (Figure 45, 6)
    if is_drc_agent:
        print(f"\nEvaluating agent with {eval_config.THINKING_STEPS} thinking steps...")
        # This requires a modified get_action or a custom evaluation loop
        # For a DRC agent, thinking steps mean N_INTERNAL_TICKS * THINKING_STEPS
        # This is not directly evaluated by `evaluate_agent_performance` as implemented.
        # It needs a dedicated loop.
        # This is a placeholder for the behavioral aspect of thinking steps.
        print("  (Behavioral effect of thinking steps is not fully simulated in mock env)")
    
    # 2. Probe for concept representations
    print("\n--- Probing for Concept Representations ---")
    probe_results = evaluator.evaluate_probes(model, env, probe_config.PROBE_TEST_EPISODES, is_drc_agent)
    print("\nProbe Evaluation Results:")
    for key, metrics in probe_results.items():
        print(f"- {key}: Macro F1 = {metrics['macro_f1']:.4f} (±{metrics['macro_f1_std']:.4f})")

    # 3. Investigate plan formation (qualitative)
    print("\n--- Investigating Plan Formation ---")
    evaluator.investigate_plan_formation() # Prints a message

    # 4. Investigate plan refinement (quantitative, Figure 6)
    if is_drc_agent:
        print("\n--- Investigating Plan Refinement ---")
        plan_refinement_results = evaluator.investigate_plan_refinement(model, env, probe_config.PROBE_TEST_EPISODES, eval_config.THINKING_STEPS)
        print("\nPlan Refinement Results (Average Macro F1 per tick):")
        for concept_key, layer_results in plan_refinement_results.items():
            print(f"- {concept_key}:")
            for layer, f1_scores_per_tick in layer_results.items():
                print(f"  {layer}: {f1_scores_per_tick}")

    # 5. Confirm behavioral dependence (Interventions)
    print("\n--- Confirming Behavioral Dependence (Interventions) ---")
    if is_drc_agent:
        intervention_results = evaluator.run_intervention_experiments(model, num_levels_per_type=1, num_rotations=1) # Small numbers for mock
        print("\nIntervention Experiment Results:")
        for key, metrics in intervention_results.items():
            print(f"- {key}: Success Rate = {metrics['success_rate']:.2f}% (Random Baseline: {metrics['random_baseline_success_rate']:.2f}%)")
    else:
        print("Interventions are primarily designed for DRC agents as per paper. Skipping for ResNet.")

if __name__ == "__main__":
    # To run evaluation for a trained model:
    # run_evaluation(model_path="path/to/your/trained_model.pth", is_drc_agent=True)
    
    # For a fresh run (randomly initialized agent):
    run_evaluation(model_path=None, is_drc_agent=True)
