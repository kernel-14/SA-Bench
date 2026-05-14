## adaptive_inference.py

import numpy as np
import torch
from torch import Tensor
from typing import Dict, Tuple


class AdaptiveInference:
    """
    Implements adaptive inference strategies for Masked Diffusion Models (MDMs). 
    Provides methods for Top Probability and Top Probability Margin strategies.
    """
    def __init__(self, model, config: Dict):
        """
        Initializes the AdaptiveInference class with the Masked Diffusion Model and configuration.

        Args:
            model: Pretrained Masked Diffusion Model (MDM).
            config (Dict): Configuration parameters loaded from `config.yaml`.
        """
        self.model = model
        self.sampling_steps = config["inference"].get("sampling_steps", 50)
        self.gumbel_noise_coefficient = config["inference"].get("gumbel_noise_coefficient", 0.5)
        self.oracle_temperature = config["inference"].get("oracle_temperature", 0.1)
        self.sequence_length = config["model"].get("max_sequence_length", 2048)

    def apply_top_probability(self, dataset: Tensor) -> Tensor:
        """
        Applies the "Top Probability" adaptive inference strategy to select masked positions based on max posterior probabilities.

        Args:
            dataset (Tensor): The current masked input sequence of shape [batch_size, sequence_length].

        Returns:
            Tensor: Updated sequence with selected tokens unmasked.
        """
        logits = self.model.forward(dataset)  # [batch_size, seq_length, vocab_size]
        batch_size, seq_length, vocab_size = logits.shape

        # Compute max probabilities for each token position
        max_probs, _ = logits.softmax(dim=-1).max(dim=-1)  # Shape: [batch_size, seq_length]

        # Add Gaussian noise for diversity during token selection
        noise = torch.normal(mean=0, std=self.gumbel_noise_coefficient, size=max_probs.shape, device=max_probs.device)
        perturbed_probs = max_probs + noise

        # Choose top K positions based on masking schedule
        masked_positions = (dataset == 0).float()  # Masked positions are marked with value 0
        K = int(masked_positions.sum(dim=1).max().item() * self.oracle_temperature)
        unmask_indices = torch.topk(perturbed_probs * masked_positions, K, dim=-1).indices  # Top-K indices per sequence

        # Update dataset with selected positions unmasked
        updated_dataset = self._unmask_tokens(dataset, logits, unmask_indices)
        return updated_dataset

    def apply_top_probability_margin(self, dataset: Tensor) -> Tensor:
        """
        Applies the "Top Probability Margin" adaptive inference strategy to select masked positions based on the largest margin between top-2 probabilities.

        Args:
            dataset (Tensor): The current masked input sequence of shape [batch_size, sequence_length].

        Returns:
            Tensor: Updated sequence with selected tokens unmasked.
        """
        logits = self.model.forward(dataset)  # [batch_size, seq_length, vocab_size]
        batch_size, seq_length, vocab_size = logits.shape

        # Compute probability margins (difference between top-1 and top-2 probabilities)
        probs = logits.softmax(dim=-1)  # Shape: [batch_size, seq_length, vocab_size]
        sorted_probs, _ = probs.sort(dim=-1, descending=True)  # Shape: [batch_size, seq_length, vocab_size]
        margin = sorted_probs[:, :, 0] - sorted_probs[:, :, 1]  # Top-1 minus Top-2 probabilities (Shape: [batch_size, seq_length])

        # Add Gaussian noise for diversity during token selection
        noise = torch.normal(mean=0, std=self.gumbel_noise_coefficient, size=margin.shape, device=margin.device)
        perturbed_margins = margin + noise

        # Choose top K positions based on masking schedule
        masked_positions = (dataset == 0).float()  # Masked positions are marked with value 0
        K = int(masked_positions.sum(dim=1).max().item() * self.oracle_temperature)
        unmask_indices = torch.topk(perturbed_margins * masked_positions, K, dim=-1).indices  # Top-K indices per sequence

        # Update dataset with selected positions unmasked
        updated_dataset = self._unmask_tokens(dataset, logits, unmask_indices)
        return updated_dataset

    def apply_adaptive_strategy(self, dataset: Tensor, strategy_name: str) -> Tensor:
        """
        Centralized handler for applying a specific adaptive inference strategy ("top_probability" or "top_probability_margin").

        Args:
            dataset (Tensor): The masked input sequence at noise level t, shape [batch_size, sequence_length].
            strategy_name (str): Selected adaptive strategy ("top_probability" or "top_probability_margin").

        Returns:
            Tensor: Updated sequence after applying the selected strategy.
        """
        if strategy_name == "top_probability":
            return self.apply_top_probability(dataset)
        elif strategy_name == "top_probability_margin":
            return self.apply_top_probability_margin(dataset)
        else:
            raise ValueError(f"Unknown strategy '{strategy_name}'. Supported strategies: 'top_probability', 'top_probability_margin'.")

    def _unmask_tokens(self, dataset: Tensor, logits: Tensor, unmask_indices: Tensor) -> Tensor:
        """
        Helper function to unmask tokens at selected positions and update the dataset with predictions.

        Args:
            dataset (Tensor): The current masked sequence, shape [batch_size, sequence_length].
            logits (Tensor): Model logits for the sequence, shape [batch_size, sequence_length, vocab_size].
            unmask_indices (Tensor): Indices of tokens to unmask, shape [batch_size, K].

        Returns:
            Tensor: Updated sequence with selected tokens unmasked.
        """
        for batch_idx in range(dataset.size(0)):
            for pos in unmask_indices[batch_idx]:
                token_pred = torch.argmax(logits[batch_idx, pos]).item()  # Most probable token
                dataset[batch_idx, pos] = token_pred  # Replace mask with predicted token

        return dataset

