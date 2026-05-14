# leaderboard_simulation.py

import numpy as np
from typing import List, Dict, Tuple

class LeaderboardSimulation:
    """
    Simulates a voting-based leaderboard system for chatbot models,
    using Elo scoring to dynamically rank models based on voting outcomes.
    """

    def __init__(self, models: List[str], k_factor: int = 30) -> None:
        """
        Initializes the leaderboard simulation with models and Elo scoring parameters.

        Args:
            models (List[str]): List of model names to track.
            k_factor (int): Scaling factor for Elo score adjustments. Default is 30.
        """
        if not isinstance(models, list) or not models:
            raise ValueError("Models list must be non-empty and of type List[str].")
        
        self.models = {model: 1500 for model in models}  # Initialize Elo scores to 1500
        self.k_factor = k_factor  # Load the Elo scaling factor

    def update_elo_scores(self, model_a: str, model_b: str, winner: str) -> None:
        """
        Updates the Elo scores of two models based on the outcome of a vote.

        Args:
            model_a (str): Name of the first model in the matchup.
            model_b (str): Name of the second model.
            winner (str): The winner of the vote ("model_a", "model_b", or "tie").

        Raises:
            ValueError: If inputs are invalid (e.g., model names not in the leaderboard, invalid winner).
        """
        if model_a not in self.models or model_b not in self.models:
            raise ValueError(f"One or both models ('{model_a}', '{model_b}') are not in the leaderboard.")
        if winner not in ["model_a", "model_b", "tie"]:
            raise ValueError("Winner must be 'model_a', 'model_b', or 'tie'.")

        # Retrieve Elo scores
        ra = self.models[model_a]
        rb = self.models[model_b]

        # Calculate expected probabilities
        Ea = 1 / (1 + 10 ** ((rb - ra) / 400))  # Expected score for model_a
        Eb = 1 - Ea                            # Expected score for model_b

        # Calculate actual scores
        if winner == "model_a":
            Sa, Sb = 1, 0
        elif winner == "model_b":
            Sa, Sb = 0, 1
        else:  # winner == "tie"
            Sa, Sb = 0.5, 0.5

        # Update Elo scores
        ra_new = ra + self.k_factor * (Sa - Ea)
        rb_new = rb + self.k_factor * (Sb - Eb)

        # Store updated scores
        self.models[model_a] = ra_new
        self.models[model_b] = rb_new

    def get_rankings(self) -> List[Tuple[str, float]]:
        """
        Retrieves the current rankings of models based on Elo scores, in descending order.

        Returns:
            List[Tuple[str, float]]: A list of tuples, where each tuple contains:
                - Model name (str)
                - Elo score (float), sorted by descending order of scores.
        """
        # Sort models by Elo score in descending order
        rankings = sorted(self.models.items(), key=lambda x: x[1], reverse=True)
        return rankings
