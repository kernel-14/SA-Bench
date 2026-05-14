# prober.py

from typing import List, Dict
import numpy as np
import torch
from torch.nn.functional import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


class Prober:
    """
    Prober class to decode planning-relevant concepts (C_A, C_B) from
    the DRCModel's internal states using linear probes.
    """

    def __init__(self, model, probes_config: Dict, device: torch.device):
        """
        Initializes the Prober class.

        Args:
            model: A trained DRCModel instance to probe.
            probes_config (Dict): Configuration dictionary for probe training.
            device (torch.device): Torch device ('cuda' or 'cpu').
        """
        self.model = model
        self.probes_config = probes_config
        self.device = device
        self.num_classes = 5  # Classes: UP, DOWN, LEFT, RIGHT, NEVER
        self.probes = {}  # Dict to store trained probes for each layer
        self.results = {}  # Stores results of evaluation (F1-scores, metrics)

        # Hyperparameters
        self.learning_rate = probes_config.get("learning_rate", 0.001)
        self.max_epochs = probes_config.get("max_epochs", 10)
        self.batch_size = probes_config.get("batch_size", 16)

    def train_probes(self, dataset: List[Dict], concept_type: str, layer_ids: List[int]) -> None:
        """
        Trains linear probes for a given concept type (C_A or C_B) on specific layers.

        Args:
            dataset (List[Dict]): Preprocessed Sokoban transitions containing hidden states and labels.
            concept_type (str): The type of concept to decode ('C_A' or 'C_B').
            layer_ids (List[int]): ConvLSTM layers to probe.
        """
        assert concept_type in ["C_A", "C_B"], "Invalid concept_type. Must be 'C_A' or 'C_B'."

        print(f"Training probes for concept: {concept_type}...")

        for layer_id in layer_ids:
            self.probes[layer_id] = {}

            # Prepare datasets for the specific layer
            x_train, y_train = self._prepare_dataset(dataset, layer_id, concept_type)

            # Train a logistic regression probe for each grid position
            print(f"Training probes for layer {layer_id}...")
            for grid_x in range(8):  # For each row
                for grid_y in range(8):  # For each column
                    # Extract spatially-local or neighborhood data (1x1 grid)
                    sample_x = x_train[:, :, grid_x, grid_y] if concept_type == "C_A" else x_train[:, :, grid_x, grid_y].reshape(x_train.shape[0], -1)
                    sample_y = y_train[:, grid_x, grid_y]

                    # Train a Logistic Regression model
                    probe = LogisticRegression(max_iter=200)
                    probe.fit(sample_x, sample_y)  ###NEED PATCHــــRE-=NORMAL ALINUX(bound) 
                    
