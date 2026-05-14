# intervention.py

from typing import List, Dict, Tuple
import numpy as np
import torch
from drc_model import DRCModel
from prober import Prober

class Intervention:
    """
    Intervention class to apply targeted changes to DRCModel's internal representations
    (e.g., hidden states or cell states) for testing the causal impact of planning-related
    concepts on agent behavior.
    """

    def __init__(self, model: DRCModel, prober: Prober, config: Dict):
        """
        Initialize the Intervention class.

        Args:
            model (DRCModel): The trained DRC model instance.
            prober (Prober): The Prober instance containing probe vectors for concepts.
            config (Dict): Configuration dictionary loaded from config.yaml.
        """
        self.model = model
        self.prober = prober
        self.alpha = config.get("intervention_strength", 1.0)  # Default scaling factor
        self.grid_size = config["environment"]["grid_size"]  # Assumes 8x8 grid by default
        
        # Validate configurations
        if self.grid_size <= 0 or self.alpha <= 0:
            raise ValueError("Invalid grid size or intervention strength in configuration.")
    
    def apply_intervention(
        self,
        layer: int,
        concept_vectors: Dict[str, torch.Tensor],
        target_positions: List[Tuple[int, int]],
        target_classes: Dict[Tuple[int, int], str]
    ) -> None:
        """
        Applies intervention to specific ConvLSTM layer by injecting concept vectors
        into the grid cell states.

        Args:
            layer (int): The ConvLSTM layer index to intervene upon.
            concept_vectors (Dict[str, torch.Tensor]): Mapping of concept classes to vectors (e.g., 'UP' to vector).
            target_positions (List[Tuple[int, int]]): List of grid coordinates for intervention (e.g., [(3, 4), ...]).
            target_classes (Dict[Tuple[int, int], str]): Map of target positions to concept class (e.g., {(3, 4): 'UP'}).
        """
        with torch.no_grad():
            # Step 1: Access the cell states of the specified layer
            cell_states = self.model.convlstm_layers_list[layer].cell_states

            if cell_states is None:
                raise RuntimeError(f"No cell state available for layer {layer}. Ensure the model has been forwarded.")

            # Step 2: Inject concept vectors into the specified spatial positions
            for position in target_positions:
                if position not in target_classes:
                    continue  # Skip if position doesn't have a designated target class

                class_name = target_classes[position]  # Get the target class for this position
                if class_name not in concept_vectors:
                    continue  # Skip if no vector exists for the designated class

                embed_vector = concept_vectors[class_name]  # Load the corresponding concept vector

                # Calculate the spatial coordinates
                x, y = position
                if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
                    raise ValueError(f"Invalid grid position ({x}, {y}). Grid size is {self.grid_size}x{self.grid_size}.")

                # Inject the scaled vector into the cell state at (x, y)
                cell_states[..., x, y] += self.alpha * embed_vector

    def steer_behavior(
        self,
        level_config: Dict,
        probe_vectors: Dict[str, torch.Tensor]
    ) -> None:
        """
        Automates experimental interventions to steer the agent's behavior based on 
        predefined configurations such as Agent-Shortcut or Box-Shortcut experiments.

        Args:
            level_config (Dict): Experiment-level configurations to determine intervention types.
            probe_vectors (Dict[str, torch.Tensor]): Map of concept classes to learned probe vectors.
        """
        intervention_type = level_config.get("intervention_type", "AgentShortcut")
        layer = level_config.get("layer", 0)  # Default to affecting layer 0 if unspecified
        grid_positions = level_config.get("positions", [])
        target_classes = level_config.get("target_classes", {})

        if intervention_type not in ["AgentShortcut", "BoxShortcut"]:
            raise ValueError(f"Invalid intervention type: {intervention_type}. Supported: AgentShortcut, BoxShortcut")

        # Apply intervention with predefined configuration
        self.apply_intervention(
            layer=layer,
            concept_vectors=probe_vectors,
            target_positions=grid_positions,
            target_classes=target_classes
        )

    def reset_interventions(self) -> None:
        """
        Clears any lingering interventions from the model to ensure future activations
        process normally without residual modifications.
        """
        for layer_idx, convlstm_layer in enumerate(self.model.convlstm_layers_list):
            if hasattr(convlstm_layer, "cell_states"):
                convlstm_layer.cell_states = None  # De-initialize cell states for clean usage
    
    def baseline_intervention(
        self,
        layer: int,
        random_seed: int = 42,
        scale: float = 0.25
    ) -> None:
        """
        Applies a baseline random intervention for comparison. Uses untrained, randomly
        initialized vectors to simulate the null hypothesis.

        Args:
            layer (int): The ConvLSTM layer index for random intervention.
            random_seed (int): Seed for reproducibility of random vector initialization.
            scale (float): Scaling strength for the random baseline intervention.
        """
        torch.manual_seed(random_seed)
        random_vectors = {
            class_name: torch.randn_like(vector) * scale
            for class_name, vector in self.prober.concept_vectors.items()
        }
        
        target_positions = [
            (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
            for _ in range(5)  # Randomly pick 5 positions, for example
        ]
        target_classes = {pos: np.random.choice(list(random_vectors.keys())) for pos in target_positions}

        self.apply_intervention(layer=layer, concept_vectors=random_vectors, target_positions=target_positions, target_classes=target_classes)
