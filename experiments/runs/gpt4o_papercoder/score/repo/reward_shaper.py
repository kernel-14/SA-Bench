# reward_shaper.py

import torch
from typing import Dict, Any
import numpy as np

class RewardShaper:
    """
    Implements reward shaping to compute progress-based rewards for Stage II RL training
    in the SCoRe framework. Encourages self-correction behavior by penalizing 
    behavior collapse and rewarding progress between turns.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes the RewardShaper with hyperparameters from the configuration.

        Args:
            config: Configuration dictionary extracted from `config.yaml`.
        """
        # Load reward shaping hyperparameters from config
        self.alpha = config['training']['stage2'].get('reward_shaping_alpha', 10)  # Progress multiplier
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.alpha <= 0:
            raise ValueError("Reward shaping alpha must be a positive value.")

    def compute_rewards(
        self,
        first_response: torch.Tensor,
        second_response: torch.Tensor,
        ground_truth: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes shaped rewards for reinforcement learning based on binary correctness and progress
        between first and second responses.

        Args:
            first_response (torch.Tensor): Batch of first-turn model outputs (token-level or decoded as required).
            second_response (torch.Tensor): Batch of second-turn model outputs (token-level or decoded as required).
            ground_truth (torch.Tensor): Batch of ground-truth outputs (e.g., expected solutions or programs).

        Returns:
            torch.Tensor: Shaped rewards for the batch, combining binary rewards, progress bonuses, and penalties.
        """
        # Ensure all inputs are moved to the expected device
        first_response = first_response.to(self.device)
        second_response = second_response.to(self.device)
        ground_truth = ground_truth.to(self.device)

        # Binary correctness calculation: compare each response to the ground truth
        reward_first = (first_response == ground_truth).all(dim=1).float()
        reward_second = (second_response == ground_truth).all(dim=1).float()

        # Compute progress between first and second responses
        progress = reward_second - reward_first

        # Calculate progress-based bonus with α multiplier
        progress_bonus = self.alpha * progress

        # Penalize degradation (behavioral collapse): correct response in first turn degraded in the second turn
        penalty = torch.where(
            (reward_first == 1.0) & (reward_second == 0.0),  # First turn correct but degraded in second turn
            -self.alpha,  # Apply a heavy penalty
            torch.tensor(0.0, device=self.device)  # No penalty otherwise
        )

        # Combine final shaped rewards
        shaped_rewards = reward_second + progress_bonus + penalty

        return shaped_rewards

    def _validate_inputs(
        self, first_response: torch.Tensor, second_response: torch.Tensor, ground_truth: torch.Tensor
    ) -> None:
        """
        Validates the shapes of input tensors to ensure they are compatible for batch processing.

        Args:
            first_response (torch.Tensor): First-turn model outputs.
            second_response (torch.Tensor): Second-turn model outputs.
            ground_truth (torch.Tensor): Ground-truth data.

        Raises:
            ValueError: If the input tensors do not have matching batch sizes.
        """
        if not (first_response.size(0) == second_response.size(0) == ground_truth.size(0)):
            raise ValueError(
                "Batch sizes of first response, second response, and ground truth must match. "
                f"Got: first_response ({first_response.size(0)}), "
                f"second_response ({second_response.size(0)}), "
                f"ground_truth ({ground_truth.size(0)})."
            )

        if not (first_response.size(1) == ground_truth.size(1)):
            raise ValueError(
                f"Sequence lengths of first_response ({first_response.size(1)}) "
                f"and ground_truth ({ground_truth.size(1)}) must match."
            )

    def debug_metrics(
        self,
        first_response: torch.Tensor,
        second_response: torch.Tensor,
        ground_truth: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Computes additional diagnostic metrics for debugging the reward shaping process.

        Args:
            first_response (torch.Tensor): Batch of first-turn model outputs.
            second_response (torch.Tensor): Batch of second-turn model outputs.
            ground_truth (torch.Tensor): Batch of ground-truth outputs.

        Returns:
            dict: Dictionary containing debugging stats such as penalty counts and progress counts.
        """
        # Binary correctness
        reward_first = (first_response == ground_truth).all(dim=1).float()
        reward_second = (second_response == ground_truth).all(dim=1).float()

        # Metrics
        total_examples = first_response.size(0)
        correct_to_incorrect = ((reward_first == 1.0) & (reward_second == 0.0)).sum().item()
        incorrect_to_correct = ((reward_first == 0.0) & (reward_second == 1.0)).sum().item()
        unchanged = ((reward_first == reward_second)).sum().item()

        metrics = {
            "total_examples": total_examples,
            "correct_to_incorrect": correct_to_incorrect,
            "incorrect_to_correct": incorrect_to_correct,
            "unchanged_responses": unchanged,
            "progress_positive": incorrect_to_correct,
            "collapses_penalized": correct_to_incorrect,
        }

        return metrics
