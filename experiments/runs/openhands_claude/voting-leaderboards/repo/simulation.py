"""
Adversarial voting simulation for estimating the cost of leaderboard manipulation.

Implements the simulation pipeline from Section 3.1:
  - Attacker interacts with the arena, using a detector to identify the target model
  - Attacker votes for (or against) the target model when detected
  - Bradley-Terry coefficients are recalculated every 1000 interactions
  - Tracks cumulative interactions and votes to achieve ranking objectives

Reproduces Tables 4, 5, and 8 from the paper.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

from bradley_terry import (
    fit_bradley_terry,
    get_rank,
    get_rankings,
)
from config import BradleyTerryConfig, SimulationConfig
from data import VoteRecord


# ---------------------------------------------------------------------------
# Attack objectives
# ---------------------------------------------------------------------------

AttackObjective = Literal["up", "down"]
NonTargetStrategy = Literal["nothing", "random_upvote", "tie", "both_bad"]


@dataclass
class AttackResult:
    """Result of a single simulation run."""
    target_model: str
    objective: AttackObjective
    target_rank: int
    initial_rank: int
    achieved: bool
    num_votes: int
    num_interactions: int
    final_rank: int
    rank_history: List[int] = field(default_factory=list)
    votes_history: List[int] = field(default_factory=list)
    interactions_history: List[int] = field(default_factory=list)


@dataclass
class SimulationState:
    """Mutable state of the simulation."""
    votes: List[VoteRecord]
    strengths: Dict[str, float]
    models: List[str]
    total_interactions: int = 0
    total_votes: int = 0


# ---------------------------------------------------------------------------
# Attacker behavior model
# ---------------------------------------------------------------------------

class AdversarialAttacker:
    """
    Models the behavior of an adversarial user in Chatbot Arena.

    The attacker:
    1. Submits a prompt and receives two anonymous model responses
    2. Uses a detector to identify if the target model is present
    3. Votes for (upvote) or against (downvote) the target model if detected
    4. Takes a non-target action if the target model is not detected

    Detection is modeled probabilistically with the specified accuracy,
    symmetric false positive and false negative rates (Section 3.1).
    """

    def __init__(
        self,
        target_model: str,
        objective: AttackObjective,
        detection_accuracy: float = 0.95,
        false_positive_rate: float = 0.05,
        false_negative_rate: float = 0.05,
        non_target_strategy: NonTargetStrategy = "nothing",
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.target_model = target_model
        self.objective = objective
        self.detection_accuracy = detection_accuracy
        self.false_positive_rate = false_positive_rate
        self.false_negative_rate = false_negative_rate
        self.non_target_strategy = non_target_strategy
        self.rng = rng or np.random.default_rng(42)

    def detect_target(self, model_a: str, model_b: str) -> Tuple[bool, bool]:
        """
        Simulate detection of the target model in a pair of responses.

        Returns:
            (target_in_a, target_in_b): Whether the attacker believes the
            target model is model_a or model_b.
        """
        actual_a = (model_a == self.target_model)
        actual_b = (model_b == self.target_model)

        # Apply detection errors
        if actual_a:
            detected_a = self.rng.random() > self.false_negative_rate
        else:
            detected_a = self.rng.random() < self.false_positive_rate

        if actual_b:
            detected_b = self.rng.random() > self.false_negative_rate
        else:
            detected_b = self.rng.random() < self.false_positive_rate

        return detected_a, detected_b

    def cast_vote(
        self,
        model_a: str,
        model_b: str,
        all_models: List[str],
    ) -> Optional[VoteRecord]:
        """
        Simulate the attacker casting a vote for a given model pair.

        Returns:
            VoteRecord if a vote is cast, None if the attacker abstains.
        """
        detected_a, detected_b = self.detect_target(model_a, model_b)

        if detected_a or detected_b:
            # Target model detected: cast adversarial vote
            if self.objective == "up":
                if detected_a:
                    winner = "model_a"
                else:
                    winner = "model_b"
            else:  # "down"
                if detected_a:
                    winner = "model_b"
                else:
                    winner = "model_a"

            return VoteRecord(model_a=model_a, model_b=model_b, winner=winner)

        # Target not detected: apply non-target strategy
        return self._non_target_action(model_a, model_b, all_models)

    def _non_target_action(
        self,
        model_a: str,
        model_b: str,
        all_models: List[str],
    ) -> Optional[VoteRecord]:
        """Handle the case where the target model was not detected."""
        if self.non_target_strategy == "nothing":
            return None
        elif self.non_target_strategy == "random_upvote":
            winner = "model_a" if self.rng.random() < 0.5 else "model_b"
            return VoteRecord(model_a=model_a, model_b=model_b, winner=winner)
        elif self.non_target_strategy == "tie":
            return VoteRecord(model_a=model_a, model_b=model_b, winner="tie")
        elif self.non_target_strategy == "both_bad":
            # Treated as a tie in the BT model
            return VoteRecord(model_a=model_a, model_b=model_b, winner="tie")
        else:
            return None


# ---------------------------------------------------------------------------
# Model sampling (uniform random pair selection)
# ---------------------------------------------------------------------------

def sample_model_pair(
    models: List[str],
    rng: np.random.Generator,
) -> Tuple[str, str]:
    """
    Sample two distinct models uniformly at random (as in Chatbot Arena).
    """
    idx_a, idx_b = rng.choice(len(models), size=2, replace=False)
    return models[idx_a], models[idx_b]


# ---------------------------------------------------------------------------
# Core simulation loop
# ---------------------------------------------------------------------------

def run_attack_simulation(
    target_model: str,
    objective: AttackObjective,
    target_rank: int,
    initial_votes: List[VoteRecord],
    models: List[str],
    bt_config: Optional[BradleyTerryConfig] = None,
    sim_config: Optional[SimulationConfig] = None,
    run_id: int = 0,
) -> AttackResult:
    """
    Simulate an adversarial attack on the Chatbot Arena leaderboard.

    The simulation follows Section 3.1:
    1. Start with historical voting data to establish initial rankings
    2. Iteratively simulate attacker interactions
    3. Recalculate BT coefficients every 1000 interactions
    4. Track when the target rank is achieved

    Args:
        target_model: The model the attacker wants to promote/demote.
        objective: "up" to promote, "down" to demote.
        target_rank: The desired rank position.
        initial_votes: Historical voting data to initialize the leaderboard.
        models: List of all models in the arena.
        bt_config: Bradley-Terry configuration.
        sim_config: Simulation configuration.
        run_id: Run identifier for seeding.

    Returns:
        AttackResult with vote/interaction counts and rank history.
    """
    if bt_config is None:
        bt_config = BradleyTerryConfig()
    if sim_config is None:
        sim_config = SimulationConfig()

    rng = np.random.default_rng(sim_config.random_seed + run_id)

    # Initialize state with historical votes
    current_votes = list(initial_votes)
    strengths = fit_bradley_terry(models, current_votes, bt_config)
    initial_rank = get_rank(target_model, strengths)

    attacker = AdversarialAttacker(
        target_model=target_model,
        objective=objective,
        detection_accuracy=sim_config.detection_accuracy,
        false_positive_rate=sim_config.false_positive_rate,
        false_negative_rate=sim_config.false_negative_rate,
        non_target_strategy=sim_config.non_target_strategy,
        rng=rng,
    )

    total_interactions = 0
    total_votes = 0
    rank_history = [initial_rank]
    votes_history = [0]
    interactions_history = [0]
    achieved = False
    final_rank = initial_rank

    while total_interactions < sim_config.max_interactions:
        # Simulate one batch of interactions
        for _ in range(sim_config.recalc_interval):
            model_a, model_b = sample_model_pair(models, rng)
            total_interactions += 1

            vote = attacker.cast_vote(model_a, model_b, models)
            if vote is not None:
                current_votes.append(vote)
                total_votes += 1

        # Recalculate BT coefficients
        strengths = fit_bradley_terry(models, current_votes, bt_config)
        current_rank = get_rank(target_model, strengths)
        final_rank = current_rank

        rank_history.append(current_rank)
        votes_history.append(total_votes)
        interactions_history.append(total_interactions)

        # Check if objective achieved
        if objective == "up" and current_rank <= target_rank:
            achieved = True
            break
        elif objective == "down" and current_rank >= target_rank:
            achieved = True
            break

    return AttackResult(
        target_model=target_model,
        objective=objective,
        target_rank=target_rank,
        initial_rank=initial_rank,
        achieved=achieved,
        num_votes=total_votes,
        num_interactions=total_interactions,
        final_rank=final_rank,
        rank_history=rank_history,
        votes_history=votes_history,
        interactions_history=interactions_history,
    )


def run_attack_simulation_averaged(
    target_model: str,
    objective: AttackObjective,
    target_rank: int,
    initial_votes: List[VoteRecord],
    models: List[str],
    bt_config: Optional[BradleyTerryConfig] = None,
    sim_config: Optional[SimulationConfig] = None,
) -> Tuple[float, float]:
    """
    Run multiple simulation runs and return average votes and interactions.

    Args:
        target_model: Target model.
        objective: Attack objective.
        target_rank: Desired rank.
        initial_votes: Historical votes.
        models: All models.
        bt_config: BT configuration.
        sim_config: Simulation configuration.

    Returns:
        Tuple of (mean_votes, mean_interactions) across successful runs.
    """
    if sim_config is None:
        sim_config = SimulationConfig()

    all_votes = []
    all_interactions = []

    for run_id in range(sim_config.num_runs):
        result = run_attack_simulation(
            target_model=target_model,
            objective=objective,
            target_rank=target_rank,
            initial_votes=initial_votes,
            models=models,
            bt_config=bt_config,
            sim_config=sim_config,
            run_id=run_id,
        )
        if result.achieved:
            all_votes.append(result.num_votes)
            all_interactions.append(result.num_interactions)

    if not all_votes:
        return float("inf"), float("inf")

    return float(np.mean(all_votes)), float(np.mean(all_interactions))


# ---------------------------------------------------------------------------
# Reproduce Tables 4 and 5
# ---------------------------------------------------------------------------

def simulate_high_ranked_models(
    initial_votes: List[VoteRecord],
    models: List[str],
    high_ranked_models: Dict[str, Dict],
    target_ranks: List[int],
    bt_config: Optional[BradleyTerryConfig] = None,
    sim_config: Optional[SimulationConfig] = None,
) -> Dict[str, Dict[int, Tuple[float, float]]]:
    """
    Reproduce Table 4: votes and interactions for high-ranked models.

    Args:
        initial_votes: Historical voting data.
        models: All models in the arena.
        high_ranked_models: Dict of {model_name: {"rank": int, "votes": int}}.
        target_ranks: List of target rank positions to evaluate.
        bt_config: BT configuration.
        sim_config: Simulation configuration.

    Returns:
        {model_name -> {target_rank -> (mean_votes, mean_interactions)}}
    """
    results = {}

    for model_name, info in high_ranked_models.items():
        current_rank = info["rank"]
        results[model_name] = {}

        for target_rank in target_ranks:
            if target_rank == current_rank:
                results[model_name][target_rank] = (0.0, 0.0)
                continue

            objective: AttackObjective = "up" if target_rank < current_rank else "down"

            mean_votes, mean_interactions = run_attack_simulation_averaged(
                target_model=model_name,
                objective=objective,
                target_rank=target_rank,
                initial_votes=initial_votes,
                models=models,
                bt_config=bt_config,
                sim_config=sim_config,
            )
            results[model_name][target_rank] = (mean_votes, mean_interactions)

    return results


def simulate_varying_detector_accuracy(
    target_model: str,
    initial_votes: List[VoteRecord],
    models: List[str],
    target_ranks: List[int],
    detector_accuracies: List[float],
    bt_config: Optional[BradleyTerryConfig] = None,
    sim_config: Optional[SimulationConfig] = None,
) -> Dict[float, Dict[int, Tuple[float, float]]]:
    """
    Reproduce Table 8: ablation over detector accuracy.

    Args:
        target_model: Target model (e.g., "llama-13b").
        initial_votes: Historical voting data.
        models: All models.
        target_ranks: Target rank positions.
        detector_accuracies: List of accuracy values to evaluate.
        bt_config: BT configuration.
        sim_config: Base simulation configuration.

    Returns:
        {accuracy -> {target_rank -> (mean_votes, mean_interactions)}}
    """
    if sim_config is None:
        sim_config = SimulationConfig()

    results = {}

    for accuracy in detector_accuracies:
        error_rate = 1.0 - accuracy
        current_config = SimulationConfig(
            detection_accuracy=accuracy,
            false_positive_rate=error_rate / 2,
            false_negative_rate=error_rate / 2,
            recalc_interval=sim_config.recalc_interval,
            max_interactions=sim_config.max_interactions,
            random_seed=sim_config.random_seed,
            non_target_strategy=sim_config.non_target_strategy,
            num_runs=sim_config.num_runs,
        )
        results[accuracy] = {}

        for target_rank in target_ranks:
            mean_votes, mean_interactions = run_attack_simulation_averaged(
                target_model=target_model,
                objective="up",
                target_rank=target_rank,
                initial_votes=initial_votes,
                models=models,
                bt_config=bt_config,
                sim_config=current_config,
            )
            results[accuracy][target_rank] = (mean_votes, mean_interactions)

    return results


def simulate_varying_non_target_strategy(
    target_model: str,
    initial_votes: List[VoteRecord],
    models: List[str],
    target_ranks: List[int],
    strategies: List[NonTargetStrategy],
    bt_config: Optional[BradleyTerryConfig] = None,
    sim_config: Optional[SimulationConfig] = None,
) -> Dict[str, Dict[int, float]]:
    """
    Reproduce Table 9: ablation over non-target strategies.

    Returns:
        {strategy -> {target_rank -> mean_interactions}}
    """
    if sim_config is None:
        sim_config = SimulationConfig()

    results = {}

    for strategy in strategies:
        current_config = SimulationConfig(
            detection_accuracy=sim_config.detection_accuracy,
            false_positive_rate=sim_config.false_positive_rate,
            false_negative_rate=sim_config.false_negative_rate,
            recalc_interval=sim_config.recalc_interval,
            max_interactions=sim_config.max_interactions,
            random_seed=sim_config.random_seed,
            non_target_strategy=strategy,
            num_runs=sim_config.num_runs,
        )
        results[strategy] = {}

        for target_rank in target_ranks:
            _, mean_interactions = run_attack_simulation_averaged(
                target_model=target_model,
                objective="up",
                target_rank=target_rank,
                initial_votes=initial_votes,
                models=models,
                bt_config=bt_config,
                sim_config=current_config,
            )
            results[strategy][target_rank] = mean_interactions

    return results
