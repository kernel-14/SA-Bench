import gymnasium as gym
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Any

from config import SokobanEnvConfig, AgentConfig

# Sokoban grid values (based on common Sokoban implementations and paper Figure 2)
# The paper states 7 channels in x_t in R^(8x8x7) for symbolic representation.
# Let's define the one-hot encoding order.
# The exact order is not specified, but common elements are:
# Wall, Empty, Box, Agent, Box on Target, Agent on Target, Target
WALL = 0
EMPTY = 1
BOX = 2
AGENT = 3
BOX_ON_TARGET = 4
AGENT_ON_TARGET = 5
TARGET = 6

class SokobanEnv:
    """
    A wrapper for the Sokoban environment.
    (Note: Gymnasium's 'Sokoban-v0' uses 10x10. Paper uses 8x8.
    This would typically require a custom Sokoban env or modification.
    For reproduction, we'll assume an 8x8 environment can be created or
    we adapt if using a standard gym env where 8x8 is not direct.)
    
    The paper mentions "Boxoban training set (Guez et al., 2018a)" which implies
    access to specific Boxoban levels. For this static reproduction, we'll simulate
    a generic Sokoban environment based on the description and assume a mechanism
    to generate/load Boxoban-like levels exists.
    """
    def __init__(self, config: SokobanEnvConfig, dim_maze=(8,8), num_boxes=4, num_targets=4, max_steps=120):
        # We need a custom Sokoban environment that matches the paper's description (8x8, 4 boxes, 4 targets, symbolic obs).
        # Standard Gym Sokoban is not directly 8x8 symbolic.
        # For now, we will use a placeholder or simplified representation if a custom env is not provided.
        # This is a key missing piece for direct execution, but we're reproducing the code structure.
        
        # Mock environment for structural completeness, assuming it behaves like specified.
        # In a real setup, this would integrate with Boxoban-levels or a custom Gym env.
        self.config = config
        self.dim_maze = dim_maze
        self.num_boxes = num_boxes
        self.num_targets = num_targets
        self.max_steps = max_steps
        self.observation_space = np.zeros((config.GRID_SIZE, config.GRID_SIZE, config.SYMBOLIC_CHANNELS))
        self.action_space_n = 5 # Up, Down, Left, Right, No-op

        self.state = None # Current symbolic state (H, W, C)
        self.current_step = 0
        self.boxes_on_target_count = 0

    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the environment.
        In a full implementation, this would load a new Sokoban level from Boxoban dataset.
        For now, returns a dummy initial state.
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Dummy initial state: agent at (1,1), boxes at (2,2), (3,3), etc., targets at (6,6), (7,7), etc.
        # This is a placeholder for symbolic representation R^(8x8x7)
        self.state = np.zeros((self.config.GRID_SIZE, self.config.GRID_SIZE, self.config.SYMBOLIC_CHANNELS), dtype=np.float32)
        
        # Fill with empty squares initially
        self.state[:, :, EMPTY] = 1.0 
        
        # Add walls (borders)
        self.state[0, :, EMPTY] = 0.0; self.state[0, :, WALL] = 1.0
        self.state[self.config.GRID_SIZE-1, :, EMPTY] = 0.0; self.state[self.config.GRID_SIZE-1, :, WALL] = 1.0
        self.state[:, 0, EMPTY] = 0.0; self.state[:, 0, WALL] = 1.0
        self.state[:, self.config.GRID_SIZE-1, EMPTY] = 0.0; self.state[self.config.GRID_SIZE-1, :, WALL] = 1.0
        
        # Add agent and some boxes/targets (placeholder logic)
        self.state[1, 1, EMPTY] = 0.0; self.state[1, 1, AGENT] = 1.0
        self.state[2, 2, EMPTY] = 0.0; self.state[2, 2, BOX] = 1.0
        self.state[3, 3, EMPTY] = 0.0; self.state[3, 3, TARGET] = 1.0

        self.current_step = 0
        self.boxes_on_target_count = 0
        
        info = {}
        return self.state, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Performs an action in the environment.
        (Placeholder logic for state transition and reward calculation)
        """
        self.current_step += 1
        reward = self.config.REWARD_STEP
        done = False
        truncated = False
        info = {}

        # Dummy logic for movement and box pushing
        # For actual Sokoban, this would involve complex state updates based on action, walls, boxes, targets.
        
        # Example: if agent moves to (1,2)
        # new_state = self.state.copy()
        # new_state[1,1, AGENT] = 0.0; new_state[1,1, EMPTY] = 1.0 # Agent leaves
        # new_state[1,2, EMPTY] = 0.0; new_state[1,2, AGENT] = 1.0 # Agent arrives
        # self.state = new_state

        # Check for episode termination
        if self.boxes_on_target_count == self.config.NUM_BOXES:
            reward += self.config.REWARD_LEVEL_SOLVED
            done = True
        elif self.current_step >= self.max_steps:
            truncated = True # Episode ends due to length limit

        return self.state, reward, done, truncated, info

def map_sokoban_state_to_concepts(
    observation: np.ndarray, 
    next_action_sequence: List[int], 
    box_push_sequence: List[Tuple[int, int, int]] # (r, c, direction)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Maps a Sokoban observation and hypothetical future actions to the concept labels (CA, CB).
    
    C_A (Agent Approach Direction): For a given square, if the agent will move onto it in the future,
                                    and from which direction. Classes: {UP, DOWN, LEFT, RIGHT, NEVER}.
    C_B (Box Push Direction): For a given square, if a box will be pushed off it in the future,
                              and in which direction. Classes: {UP, DOWN, LEFT, RIGHT, NEVER}.
                              
    This function will be highly simplified due to the lack of a real Sokoban simulator here.
    In a true implementation, one would simulate the environment with the agent's planned actions
    to derive these labels. For this reproduction, we will use mock logic for concept labels.
    
    Args:
        observation (np.ndarray): Current symbolic Sokoban observation (H, W, Channels).
        next_action_sequence (List[int]): A hypothetical sequence of actions the agent will take.
        box_push_sequence (List[Tuple[int, int, int]]): A hypothetical sequence of box pushes (r, c, direction).
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Two (H, W) arrays of integer labels for C_A and C_B.
    """
    H, W, _ = observation.shape
    ca_labels = np.full((H, W), fill_value=SokobanConceptMap.NEVER_IDX, dtype=np.int64)
    cb_labels = np.full((H, W), fill_value=SokobanConceptMap.NEVER_IDX, dtype=np.int64)

    # Mock logic:
    # C_A: Agent plans to move Right from (1,1) to (1,2)
    # If the first action is RIGHT and agent is at (1,1)
    agent_pos = np.argwhere(observation[:,:,AGENT] == 1)
    if agent_pos.size > 0:
        ar, ac = agent_pos[0]
        if len(next_action_sequence) > 0:
            first_action = next_action_sequence[0]
            if first_action == 0: # UP
                if ar > 0: ca_labels[ar-1, ac] = SokobanConceptMap.DOWN_IDX
            elif first_action == 1: # DOWN
                if ar < H-1: ca_labels[ar+1, ac] = SokobanConceptMap.UP_IDX
            elif first_action == 2: # LEFT
                if ac > 0: ca_labels[ar, ac-1] = SokobanConceptMap.RIGHT_IDX
            elif first_action == 3: # RIGHT
                if ac < W-1: ca_labels[ar, ac+1] = SokobanConceptMap.LEFT_IDX

    # C_B: Box at (2,2) will be pushed DOWN
    for r, c, direction in box_push_sequence:
        if observation[r, c, BOX] == 1 or observation[r,c,BOX_ON_TARGET]==1:
            if direction == 0: # UP
                cb_labels[r, c] = SokobanConceptMap.UP_IDX
            elif direction == 1: # DOWN
                cb_labels[r, c] = SokobanConceptMap.DOWN_IDX
            elif direction == 2: # LEFT
                cb_labels[r, c] = SokobanConceptMap.LEFT_IDX
            elif direction == 3: # RIGHT
                cb_labels[r, c] = SokobanConceptMap.RIGHT_IDX

    return ca_labels, cb_labels

class SokobanConceptMap:
    """Utility to map concept names to integer IDs."""
    CONCEPT_CA_CLASSES = ["UP", "DOWN", "LEFT", "RIGHT", "NEVER"]
    CONCEPT_CB_CLASSES = ["UP", "DOWN", "LEFT", "RIGHT", "NEVER"]

    UP_IDX = 0
    DOWN_IDX = 1
    LEFT_IDX = 2
    RIGHT_IDX = 3
    NEVER_IDX = 4 # Corresponds to agent not stepping/pushing box off square again

class ConceptDataset(Dataset):
    """
    Dataset for storing agent activations and corresponding concept labels.
    Used for training and evaluating probes.
    """
    def __init__(self, observations: List[np.ndarray], 
                 prev_states_h: List[List[np.ndarray]], # list of (D, C, H, W) for each transition
                 prev_states_c: List[List[np.ndarray]], # list of (D, C, H, W) for each transition
                 ca_labels: List[np.ndarray], 
                 cb_labels: List[np.ndarray]):
        self.observations = [torch.from_numpy(obs).float().permute(2,0,1) for obs in observations] # (C, H, W)
        # Convert list of lists of np.ndarray to single tensor for DataLoader
        # (num_transitions, D, C, H, W)
        self.prev_states_h = torch.stack([torch.from_numpy(np.stack(h_list)).float() for h_list in prev_states_h]) if prev_states_h else []
        self.prev_states_c = torch.stack([torch.from_numpy(np.stack(c_list)).float() for c_list in prev_states_c]) if prev_states_c else []
        
        self.ca_labels = [torch.from_numpy(labels).long() for labels in ca_labels] # (H, W)
        self.cb_labels = [torch.from_numpy(labels).long() for labels in cb_labels] # (H, W)

        assert len(self.observations) == len(self.ca_labels) == len(self.cb_labels), "Mismatched lengths in dataset"
        if len(self.prev_states_h) > 0: # Check if recurrent states are actually present
            assert len(self.observations) == len(self.prev_states_h), "Mismatched lengths for prev_states_h"
            assert len(self.observations) == len(self.prev_states_c), "Mismatched lengths for prev_states_c"

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        if len(self.prev_states_h) > 0:
            return (self.observations[idx], self.prev_states_h[idx], self.prev_states_c[idx], 
                    self.ca_labels[idx], self.cb_labels[idx])
        else:
            # For ResNet or cases without recurrent states, prev_states will be empty lists
            # Return dummy empty tensors for prev_states if they don't exist, to maintain signature
            dummy_h = torch.empty(0) 
            dummy_c = torch.empty(0)
            return (self.observations[idx], dummy_h, dummy_c, 
                    self.ca_labels[idx], self.cb_labels[idx])

def collect_probe_data(agent: Any, 
                       env: SokobanEnv, 
                       num_episodes: int, 
                       agent_config: AgentConfig,
                       is_drc_agent: bool = True) -> ConceptDataset:
    """
    Collects observations, agent internal states, and concept labels for probe training.
    
    Args:
        agent: The trained DRCAgent or ResNetActorCritic instance.
        env: The Sokoban environment.
        num_episodes: Number of episodes to collect data from.
        agent_config: Configuration of the agent.
        is_drc_agent: True if collecting data for DRC, False for ResNet.
        
    Returns:
        ConceptDataset: Dataset containing collected data.
    """
    all_observations = []
    all_prev_states_h = []
    all_prev_states_c = []
    all_ca_labels = []
    all_cb_labels = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        
        # Initial states for DRC agent for the beginning of an episode
        # These are NumPy arrays to be stored in all_prev_states_h/c
        initial_h_np = np.zeros((agent_config.D_CONVLSTM_LAYERS, agent_config.CHANNELS, agent_config.GRID_SIZE, agent_config.GRID_SIZE), dtype=np.float32)
        initial_c_np = np.zeros((agent_config.D_CONVLSTM_LAYERS, agent_config.CHANNELS, agent_config.GRID_SIZE, agent_config.GRID_SIZE), dtype=np.float32)
        prev_h_states_np = initial_h_np
        prev_c_states_np = initial_c_np

        done = False
        truncated = False
        while not done and not truncated:
            # For this simplified data collection, we'll generate *mock* next action/box push sequences.
            # In a real setup, these would come from the agent's actual behavior or an oracle.
            # For now, let's assume a dummy action (e.g., move right) and no box pushes for simplicity.
            mock_next_action_sequence = [3] # 3: RIGHT
            mock_box_push_sequence = []
            
            # This is a placeholder for generating ground truth concept labels based on a hypothetical plan.
            # The actual paper's concepts are "behavior-dependent" and require simulating agent's future.
            # Here, we're providing a simplified version.
            ca_labels, cb_labels = map_sokoban_state_to_concepts(obs, mock_next_action_sequence, mock_box_push_sequence)

            all_observations.append(obs)
            all_ca_labels.append(ca_labels)
            all_cb_labels.append(cb_labels)
            
            if is_drc_agent:
                all_prev_states_h.append(prev_h_states_np)
                all_prev_states_c.append(prev_c_states_np)
                
                # Convert numpy states to torch tensors for agent's forward pass
                prev_states_torch = [
                    (torch.from_numpy(prev_h_states_np[d]).unsqueeze(0).to(agent.device), 
                     torch.from_numpy(prev_c_states_np[d]).unsqueeze(0).to(agent.device)) 
                    for d in range(agent_config.D_CONVLSTM_LAYERS)
                ]
                
                # Update prev_states from agent's forward pass
                _, _, new_states_torch, _ = agent.get_forward_pass_data(obs, prev_states_torch)
                
                # Convert new_states back to numpy for storage
                prev_h_states_np = np.stack([h.cpu().numpy().squeeze(0) for h, _ in new_states_torch])
                prev_c_states_np = np.stack([c.cpu().numpy().squeeze(0) for _, c in new_states_torch])
            # For ResNet, there are no recurrent states to collect this way.
            # `agent.get_forward_pass_data` for ResNet only returns observations and hidden states.

            action = agent.get_action(obs, greedy=True)
            obs, reward, done, truncated, info = env.step(action)
    
    return ConceptDataset(all_observations, all_prev_states_h, all_prev_states_c, all_ca_labels, all_cb_labels)
