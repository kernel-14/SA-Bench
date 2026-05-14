## synthetic_replay.py

import numpy as np
from typing import List, Dict, Any
from replay_buffer import ReplayBuffer
from generative_model import GenerativeModel
from curiosity_module import CuriosityModule

class SyntheticReplayManager:
    """
    Handles the generation and management of synthetic transitions using 
    relevance-driven conditioning on a generative model.
    """

    def __init__(self, generative_model: GenerativeModel, relevance_module: CuriosityModule, config: dict) -> None:
        """
        Initializes the synthetic replay manager with the given generative model, 
        relevance module, and configuration.

        Args:
            generative_model (GenerativeModel): The conditional diffusion-based generative model.
            relevance_module (CuriosityModule): Module for computing relevance scores.
            config (dict): Configuration dictionary parsed from `config.yaml`.
        """
        self.generative_model = generative_model
        self.relevance_module = relevance_module
        self.config = config

        # Initialize synthetic replay buffer with size from config
        self.buffer_size = config["replay_buffer"]["max_size"]
        self.synthetic_buffer = ReplayBuffer(size=self.buffer_size)

        # Set synthetic-to-real mixing ratio
        self.synthetic_to_real_ratio = config["training"].get("synthetic_to_real_ratio", 0.5)

    def regenerate_synthetic_buffer(self, real_buffer: ReplayBuffer) -> ReplayBuffer:
        """
        Regenerates/repopulates the synthetic replay buffer based on real transitions 
        and their computed relevance scores.

        Args:
            real_buffer (ReplayBuffer): Replay buffer containing real transitions.

        Returns:
            ReplayBuffer: Updated synthetic replay buffer.
        """
        # Sample a subset of transitions from the real buffer for relevance computation
        num_real_samples = min(real_buffer.current_size, self.buffer_size // 2)
        real_transitions = real_buffer.sample(num_real_samples)

        # Compute relevance scores using the curiosity module
        relevance_scores = self.relevance_module.compute_relevance_scores(real_transitions)

        # Generate synthetic transitions conditioned on relevance scores
        synthetic_transitions = self.generative_model.generate_synthetic_transitions(relevance_scores)

        # Clear and repopulate the synthetic buffer
        self.synthetic_buffer.clear()
        for transition in synthetic_transitions:
            self.synthetic_buffer.store(transition)

        return self.synthetic_buffer

    def get_mixed_batch(self, real_buffer: ReplayBuffer, batch_size: int) -> List[Dict[str, Any]]:
        """
        Creates a mixed batch of transitions by combining real and synthetic data 
        according to the synthetic-to-real ratio.

        Args:
            real_buffer (ReplayBuffer): Replay buffer containing real transitions.
            batch_size (int): Total number of transitions in the mixed batch.

        Returns:
            List[Dict[str, Any]]: Mixed batch of transitions combining real and synthetic data.
        """
        # Determine the number of real and synthetic transitions
        n_real = int(batch_size * (1.0 - self.synthetic_to_real_ratio))
        n_syn = batch_size - n_real

        # Sample transitions from both buffers
        real_transitions = real_buffer.sample(n_real)
        synthetic_transitions = self.synthetic_buffer.sample(n_syn)

        # Combine and shuffle the mixed batch
        mixed_batch = real_transitions + synthetic_transitions
        np.random.shuffle(mixed_batch)

        return mixed_batch

    def evaluate_synthetic_buffer(self) -> Dict[str, Any]:
        """
        Evaluates the synthetic replay buffer's quality and diversity statistics.

        Returns:
            Dict[str, Any]: Dictionary containing evaluation metrics such as average relevance score.
        """
        # Extract synthetic transitions
        synthetic_transitions = [t for t in self.synthetic_buffer.buffer if t is not None]

        if not synthetic_transitions:
            return {"average_relevance_score": 0.0, "buffer_size": 0}

        # Compute average relevance score
        relevance_scores = self.relevance_module.compute_relevance_scores(synthetic_transitions)
        average_relevance = np.mean(relevance_scores)

        return {
            "average_relevance_score": average_relevance,
            "buffer_size": len(synthetic_transitions)
        }

