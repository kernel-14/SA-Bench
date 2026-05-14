"""
Estimating Adversarial Votes (Section 3)

This module implements a simulation pipeline to estimate the number of
adversarial votes and interactions needed to shift model rankings on a
leaderboard using Bradley-Terry coefficients.

Key components:
- Bradley-Terry model for ranking
- Elo update mechanism (used by Chatbot Arena)
- Attacker simulation with de-anonymization detector
- Estimation of votes/interactions needed to achieve ranking objectives

Reference: Section 3.1, 3.2
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Bradley-Terry Model (Section 3.1)
# =============================================================================

class BradleyTerryModel:
    """Bradley-Terry model for ranking models based on pairwise comparisons.
    
    As described in the paper, Chatbot Arena ranks models using Bradley-Terry
    coefficients derived from user interactions.
    
    The probability that model i is preferred over model j is:
        Pr(i > j) = 1 / (1 + exp(-(Q_i - Q_j) / s))
    
    where Q_i, Q_j are Bradley-Terry coefficients and s is a scaling factor.
    """
    
    def __init__(self, model_names: List[str], scale: float = 400.0):
        """
        Args:
            model_names: List of model identifiers
            scale: Scaling factor s for probability computation
        """
        self.model_names = list(model_names)
        self.n_models = len(self.model_names)
        self.scale = scale
        
        # Initialize Bradley-Terry coefficients (ratings)
        self.coefficients = {m: 1000.0 for m in self.model_names}
        
        # Track statistics
        self.wins = defaultdict(int)
        self.losses = defaultdict(int)
        self.ties = defaultdict(int)
        self.total_votes = defaultdict(int)
        
    def get_probability(self, model_i: str, model_j: str) -> float:
        """Compute Pr(model_i preferred over model_j)."""
        q_i = self.coefficients[model_i]
        q_j = self.coefficients[model_j]
        return 1.0 / (1.0 + np.exp(-(q_i - q_j) / self.scale))
    
    def update(self, winner: str, loser: str, is_tie: bool = False):
        """Update Bradley-Terry coefficients after a single vote.
        
        Uses an Elo-like update mechanism.
        
        Args:
            winner: The preferred model
            loser: The other model
            is_tie: Whether the vote was a tie
        """
        # Elo K-factor
        K = 32
        
        expected_win = self.get_probability(winner, loser)
        
        if is_tie:
            actual_win = 0.5
            self.ties[winner] += 1
            self.ties[loser] += 1
        else:
            actual_win = 1.0
            self.wins[winner] += 1
            self.losses[loser] += 1
        
        # Update both models
        delta = K * (actual_win - expected_win)
        self.coefficients[winner] += delta
        self.coefficients[loser] -= delta
        
        self.total_votes[winner] += 1
        self.total_votes[loser] += 1
    
    def get_ranking(self) -> List[Tuple[str, float, int]]:
        """Get current model rankings sorted by coefficient (descending)."""
        ranked = sorted(self.coefficients.items(), key=lambda x: x[1], reverse=True)
        return [(name, score, self.total_votes[name]) for name, score in ranked]
    
    def get_rank(self, model: str) -> int:
        """Get the current rank (1-indexed) of a model."""
        ranking = self.get_ranking()
        for i, (name, _, _) in enumerate(ranking):
            if name == model:
                return i + 1
        return len(ranking) + 1
    
    def simulate_vote(self, model_a: str, model_b: str) -> str:
        """Simulate a fair vote between two models based on current ratings."""
        prob_a = self.get_probability(model_a, model_b)
        if np.random.random() < prob_a:
            return model_a
        else:
            return model_b


# =============================================================================
# Attacker Simulation (Section 3)
# =============================================================================

@dataclass
class AttackerConfig:
    """Configuration for the adversarial attacker."""
    # Target model to manipulate
    target_model: str
    
    # Attack direction: "up" for promoting, "down" for demoting
    direction: str = "up"
    
    # De-anonymization detector accuracy
    detector_accuracy: float = 0.95
    
    # False positive rate (identify other as target)
    false_positive_rate: float = 0.05
    
    # False negative rate (miss target)
    false_negative_rate: float = 0.05
    
    # Strategy when target not detected: 
    # "abstain", "random_upvote", "vote_tie", "vote_both_bad"
    non_target_strategy: str = "abstain"


class AttackerSimulator:
    """Simulates an adversarial user attempting to manipulate rankings.
    
    The attack consists of two steps:
    1. Re-identification: Determine which model generated a given reply
    2. Reranking: Cast malicious vote for/against the target model
    """
    
    def __init__(self, config: AttackerConfig):
        self.config = config
        self.votes_cast = 0
        self.interactions = 0
        self.correct_detections = 0
        self.incorrect_detections = 0
        
    def _simulate_detection(self, model: str) -> bool:
        """Simulate the de-anonymization detector.
        
        With probability detector_accuracy, correctly identifies target model.
        With false_negative_rate, misses the target.
        With false_positive_rate, falsely identifies other model as target.
        
        Returns True if attacker thinks the model is the target.
        """
        is_target = (model == self.config.target_model)
        
        if is_target:
            # True positive with probability (1 - false_negative_rate)
            return np.random.random() > self.config.false_negative_rate
        else:
            # False positive with probability false_positive_rate
            return np.random.random() < self.config.false_positive_rate
    
    def decide_vote(
        self, 
        model_a: str, 
        model_b: str
    ) -> Optional[Tuple[str, str]]:
        """
        Decide how to vote given two anonymous models.
        
        The attacker first tries to de-anonymize. If the target is detected in
        one of the two responses, they vote accordingly. Otherwise, they
        follow the non_target_strategy.
        
        Returns:
            Tuple (preferred_model, other_model) or None if abstaining
        """
        self.interactions += 1
        
        detected_a = self._simulate_detection(model_a)
        detected_b = self._simulate_detection(model_b)
        
        # Track detection stats
        is_a_target = (model_a == self.config.target_model)
        is_b_target = (model_b == self.config.target_model)
        
        if detected_a == is_a_target:
            self.correct_detections += 1
        else:
            self.incorrect_detections += 1
        
        if detected_b == is_b_target:
            self.correct_detections += 1
        else:
            self.incorrect_detections += 1
        
        # Determine action based on detection
        if detected_a and not detected_b:
            # Think model_a is target (and model_b is not)
            if self.config.direction == "up":
                return (model_a, model_b)  # Vote FOR target
            else:
                return (model_b, model_a)  # Vote AGAINST target
        elif detected_b and not detected_a:
            if self.config.direction == "up":
                return (model_b, model_a)  # Vote FOR target
            else:
                return (model_a, model_b)  # Vote AGAINST target
        elif detected_a and detected_b:
            # Both detected as target (unlikely but possible due to FP)
            # Random choice
            if np.random.random() < 0.5:
                return (model_a, model_b)
            else:
                return (model_b, model_a)
        else:
            # Neither detected as target
            return self._non_target_action(model_a, model_b)
    
    def _non_target_action(
        self, model_a: str, model_b: str
    ) -> Optional[Tuple[str, str]]:
        """Action when target model is not detected."""
        strategy = self.config.non_target_strategy
        
        if strategy == "abstain":
            return None
        elif strategy == "random_upvote":
            if np.random.random() < 0.5:
                return (model_a, model_b)
            else:
                return (model_b, model_a)
        elif strategy == "vote_tie":
            return (model_a, model_b)  # Will be treated as tie
        elif strategy == "vote_both_bad":
            return (model_a, model_b)  # Both bad - treated as tie
        else:
            return None


# =============================================================================
# Simulation Runner (Section 3.1, 3.2)
# =============================================================================

class LeaderboardSimulation:
    """Simulate the full leaderboard with attacker interactions.
    
    As described in Section 3.1:
    - Uses historical voting data for initialization
    - Iteratively simulates attacker interactions and adversarial votes
    - Calculates Bradley-Terry coefficient and model ranking after every N interactions
    - Tracks cumulative interactions and votes required to achieve objectives
    """
    
    def __init__(
        self,
        model_names: List[str],
        initial_votes: Optional[List[Tuple[str, str, bool]]] = None,
        scale: float = 400.0,
    ):
        """
        Args:
            model_names: List of model identifiers
            initial_votes: List of (model_a, model_b, is_tie) for historical votes
            scale: Bradley-Terry scale parameter
        """
        self.model_names = list(model_names)
        self.n_models = len(model_names)
        self.bt_model = BradleyTerryModel(model_names, scale=scale)
        
        # Initialize with historical votes if provided
        if initial_votes:
            for model_a, model_b, is_tie in initial_votes:
                self.bt_model.update(model_a, model_b, is_tie)
        
        # Track history
        self.rank_history: List[Dict[str, int]] = []
        self.rating_history: List[Dict[str, float]] = []
        self.vote_counts: Dict[str, int] = defaultdict(int)
        
        # Record initial state
        self._record_state()
    
    def _record_state(self):
        """Record current ranking state."""
        ranks = {}
        ratings = {}
        for name, score, votes in self.bt_model.get_ranking():
            ranks[name] = self.bt_model.get_rank(name)
            ratings[name] = score
        self.rank_history.append(ranks)
        self.rating_history.append(ratings)
    
    def simulate_benign_votes(
        self, 
        n_votes: int,
        model_distribution: Optional[np.ndarray] = None,
    ):
        """Simulate benign user votes.
        
        Two models are randomly selected and a fair vote is simulated.
        
        Args:
            n_votes: Number of benign votes to simulate
            model_distribution: Probability distribution over models for selection
        """
        if model_distribution is None:
            model_distribution = np.ones(self.n_models) / self.n_models
        
        for _ in range(n_votes):
            # Randomly select two models
            i, j = np.random.choice(
                self.n_models, size=2, replace=False, p=model_distribution
            )
            model_a, model_b = self.model_names[i], self.model_names[j]
            
            # Simulate fair vote
            winner = self.bt_model.simulate_vote(model_a, model_b)
            loser = model_b if winner == model_a else model_a
            
            self.bt_model.update(winner, loser, is_tie=False)
            self.vote_counts[model_a] += 1
            self.vote_counts[model_b] += 1
    
    def run_attack(
        self,
        attacker_config: AttackerConfig,
        max_interactions: int = 100000,
        target_rank: Optional[int] = None,
        target_position_change: Optional[int] = None,
        checkpoint_interval: int = 1000,
        verbose: bool = True,
    ) -> Dict:
        """
        Run the full attack simulation.
        
        As described in Section 3.1:
        - Attack objectives: Up(M, x) or Down(M, x)
        - Tracks interactions and adversarial votes required
        
        Args:
            attacker_config: Attacker configuration
            max_interactions: Maximum number of interactions to simulate
            target_rank: Desired absolute rank for target model
            target_position_change: Desired change in rank position
            checkpoint_interval: How often to record state
            verbose: Whether to log progress
            
        Returns:
            Dict with attack statistics
        """
        attacker = AttackerSimulator(attacker_config)
        
        # Determine target objective
        start_rank = self.bt_model.get_rank(attacker_config.target_model)
        
        if target_rank is not None:
            if attacker_config.direction == "up":
                objective = f"Up to rank {target_rank}"
            else:
                objective = f"Down to rank {target_rank}"
        elif target_position_change is not None:
            if attacker_config.direction == "up":
                target_rank = max(1, start_rank - target_position_change)
                objective = f"Up {target_position_change} positions"
            else:
                target_rank = min(self.n_models, start_rank + target_position_change)
                objective = f"Down {target_position_change} positions"
        else:
            target_rank = start_rank
            objective = "No objective"
        
        if verbose:
            logger.info(f"Starting attack: {objective}")
            logger.info(f"Target model: {attacker_config.target_model}")
            logger.info(f"Start rank: {start_rank}, Target rank: {target_rank}")
        
        objective_achieved = False
        interactions_at_achievement = None
        votes_at_achievement = None
        
        for step in range(max_interactions + 1):
            # Randomly select two models for comparison
            i, j = np.random.choice(self.n_models, size=2, replace=False)
            model_a, model_b = self.model_names[i], self.model_names[j]
            
            # Attacker decides vote
            vote_result = attacker.decide_vote(model_a, model_b)
            
            if vote_result is not None:
                winner, loser = vote_result
                self.bt_model.update(winner, loser, is_tie=False)
                attacker.votes_cast += 1
                self.vote_counts[model_a] += 1
                self.vote_counts[model_b] += 1
            
            # Checkpoint
            if step > 0 and step % checkpoint_interval == 0:
                self._record_state()
                current_rank = self.bt_model.get_rank(attacker_config.target_model)
                
                if verbose and step % (checkpoint_interval * 5) == 0:
                    logger.info(
                        f"Step {step}: interactions={attacker.interactions}, "
                        f"votes={attacker.votes_cast}, rank={current_rank}"
                    )
                
                # Check if objective achieved
                if not objective_achieved:
                    if attacker_config.direction == "up" and current_rank <= target_rank:
                        objective_achieved = True
                        interactions_at_achievement = attacker.interactions
                        votes_at_achievement = attacker.votes_cast
                        if verbose:
                            logger.info(
                                f"OBJECTIVE ACHIEVED at step {step}: "
                                f"interactions={interactions_at_achievement}, "
                                f"votes={votes_at_achievement}"
                            )
                    elif attacker_config.direction == "down" and current_rank >= target_rank:
                        objective_achieved = True
                        interactions_at_achievement = attacker.interactions
                        votes_at_achievement = attacker.votes_cast
                        if verbose:
                            logger.info(
                                f"OBJECTIVE ACHIEVED at step {step}: "
                                f"interactions={interactions_at_achievement}, "
                                f"votes={votes_at_achievement}"
                            )
        
        # Final record
        self._record_state()
        
        end_rank = self.bt_model.get_rank(attacker_config.target_model)
        
        return {
            "target_model": attacker_config.target_model,
            "direction": attacker_config.direction,
            "start_rank": start_rank,
            "end_rank": end_rank,
            "target_rank": target_rank,
            "objective_achieved": objective_achieved,
            "total_interactions": attacker.interactions,
            "total_votes": attacker.votes_cast,
            "interactions_at_achievement": interactions_at_achievement,
            "votes_at_achievement": votes_at_achievement,
            "detection_accuracy": attacker_config.detector_accuracy,
            "non_target_strategy": attacker_config.non_target_strategy,
        }
    
    def get_rank_history_array(self) -> np.ndarray:
        """Get rank history as numpy array for analysis."""
        n_checkpoints = len(self.rank_history)
        arr = np.zeros((n_checkpoints, self.n_models))
        for t, ranks in enumerate(self.rank_history):
            for j, model in enumerate(self.model_names):
                arr[t, j] = ranks.get(model, self.n_models + 1)
        return arr
    
    def get_rating_history_array(self) -> np.ndarray:
        """Get rating history as numpy array."""
        n_checkpoints = len(self.rating_history)
        arr = np.zeros((n_checkpoints, self.n_models))
        for t, ratings in enumerate(self.rating_history):
            for j, model in enumerate(self.model_names):
                arr[t, j] = ratings.get(model, 0.0)
        return arr


# =============================================================================
# Utility functions for running experiments (as in paper)
# =============================================================================

def estimate_votes_for_rank_change(
    simulation: LeaderboardSimulation,
    target_model: str,
    position_change: int,
    direction: str = "up",
    detector_accuracy: float = 0.95,
    max_interactions: int = 250000,
    n_trials: int = 3,
    **kwargs
) -> Dict:
    """
    Estimate the number of votes needed to change a model's rank.
    
    Runs multiple trials and returns average results.
    
    This corresponds to the Up(M, x) and Down(M, x) objectives in Section 3.1.
    """
    results = []
    
    for trial in range(n_trials):
        # Create a copy of the simulation by resetting
        bt = BradleyTerryModel(simulation.model_names, scale=simulation.bt_model.scale)
        # Copy coefficients from original
        for model in simulation.model_names:
            bt.coefficients[model] = simulation.bt_model.coefficients[model]
            bt.total_votes[model] = simulation.bt_model.total_votes[model]
        
        # New simulation with copied state
        sim = LeaderboardSimulation(simulation.model_names)
        sim.bt_model = bt
        
        config = AttackerConfig(
            target_model=target_model,
            direction=direction,
            detector_accuracy=detector_accuracy,
            false_positive_rate=(1 - detector_accuracy) / 2,
            false_negative_rate=(1 - detector_accuracy) / 2,
            **kwargs
        )
        
        result = sim.run_attack(
            config,
            max_interactions=max_interactions,
            target_position_change=position_change,
            verbose=False,
        )
        results.append(result)
    
    # Aggregate
    avg_votes = np.mean([r["votes_at_achievement"] or r["total_votes"] for r in results])
    avg_interactions = np.mean([r["interactions_at_achievement"] or r["total_interactions"] for r in results])
    
    return {
        "target_model": target_model,
        "position_change": position_change,
        "direction": direction,
        "avg_votes_required": avg_votes,
        "avg_interactions_required": avg_interactions,
        "n_trials": n_trials,
        "trial_results": results,
    }


def generate_rank_table(
    simulation: LeaderboardSimulation,
    target_models: List[str],
    rank_changes: List[int],
    direction: str = "up",
    detector_accuracy: float = 0.95,
) -> List[List]:
    """
    Generate a table similar to Table 4/5 in the paper.
    
    For each target model and each desired rank change, estimate
    the number of votes and interactions required.
    """
    results = []
    for model in target_models:
        current_rank = simulation.bt_model.get_rank(model)
        current_votes = simulation.bt_model.total_votes[model]
        row = [model, current_rank, current_votes]
        
        for change in rank_changes:
            if direction == "up":
                target_r = max(1, current_rank - change)
            else:
                target_r = min(simulation.n_models, current_rank + change)
            
            if target_r == current_rank:
                row.append("N/A")
            else:
                est = estimate_votes_for_rank_change(
                    simulation, model, change, direction, 
                    detector_accuracy, max_interactions=250000, n_trials=1
                )
                row.append(int(est["avg_votes_required"]))
        
        results.append(row)
    
    return results
