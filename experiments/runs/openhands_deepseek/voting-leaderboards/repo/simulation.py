"""Leaderboard manipulation simulation (Section 3).

Estimates the number of adversarial votes and interactions needed to
significantly shift a model's ranking on Chatbot Arena.

Key concepts:
  - Vote: When a user submits a preference for one model over another.
    Attacker only votes if they have identified the target model.
  - Interaction: All prompts/queries submitted by a user, even if no
    vote was cast (e.g., attacker abstains when target model not detected).

Attack objectives (Section 3.1):
  - Up(M, x): Manipulate model M to rise x positions
  - Down(M, x): Manipulate model M to fall x positions

Uses Bradley-Terry coefficient ratings derived from user interactions
for ranking models (Hunter, 2004).
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict

from data import VotingDataSimulator


@dataclass
class SimulationResult:
    """Tracks simulation results for a single attack configuration."""
    target_model: str
    target_initial_rank: int
    attack_type: str  # "up" or "down"
    target_rank_delta: int
    votes_required: int = 0
    interactions_required: int = 0
    success: bool = False
    vote_history: List[int] = field(default_factory=list)
    rank_history: List[int] = field(default_factory=list)


@dataclass
class AttackOutcome:
    """Tabular result matching the paper's Tables 4 and 5."""
    target_model: str
    current_rank: int
    total_votes: int
    votes_to_ranks: Dict[int, int] = field(default_factory=dict)  # target_rank -> #votes
    interactions_to_ranks: Dict[int, int] = field(default_factory=dict)  # target_rank -> #interactions


class AdversarialVoter:
    """Simulates an adversarial user trying to manipulate the leaderboard.

    The attacker:
    1. Submits a prompt to Chatbot Arena
    2. Two models are randomly selected (Section 1, Figure 1 step 1)
    3. Adversary uses detector to identify if target model is present (Section 2)
    4. If target detected, casts malicious vote for/against target (Section 3)
    5. If target not detected, chooses a non-detection strategy

    Default assumptions (Section 3.1):
      - Detector accuracy: 95%, symmetric 5% FPR/FNR
      - Non-detection strategy: "do_nothing" (passive abstention)
    """

    def __init__(
        self,
        target_model_idx: int,
        detector: object,
        attack_type: str = "up",
        detector_accuracy: float = 0.95,
        false_positive_rate: float = 0.05,
        false_negative_rate: float = 0.05,
        non_detection_strategy: str = "do_nothing",
        seed: int = 42,
    ):
        self.target_model_idx = target_model_idx
        self.detector = detector
        self.attack_type = attack_type
        self.detector_accuracy = detector_accuracy
        self.false_positive_rate = false_positive_rate
        self.false_negative_rate = false_negative_rate
        self.non_detection_strategy = non_detection_strategy
        self.rng = np.random.RandomState(seed)

        self.total_votes_cast = 0
        self.total_interactions = 0

    def should_vote(self, model_a: int, model_b: int, response_a: str, response_b: str) -> bool:
        """Determine if the attacker should cast a vote in this interaction.

        Returns True if the target model was detected in either response
        (with errors accounting for detector accuracy).
        """
        target_present = (model_a == self.target_model_idx) or (model_b == self.target_model_idx)
        return target_present

    def detect_target(self, response: str, is_target: bool) -> bool:
        """Simulate detector with specified false positive/negative rates.

        When is_target=True and detector says yes -> True Negative rate already handled.
        The paper assumes symmetric 5% FPR and FNR.
        """
        if is_target:
            return self.rng.random() < self.detector_accuracy  # True positive rate = accuracy
        else:
            return self.rng.random() < self.false_positive_rate

    def cast_vote(
        self,
        model_a: int,
        model_b: int,
        response_a: str,
        response_b: str,
    ) -> int:
        """Cast an adversarial vote.

        For attack_type="up": votes for the target model.
        For attack_type="down": votes against the target model.

        Returns: 1 if model_a wins, 0 if tie, -1 if model_b wins.
        """
        self.total_interactions += 1

        target_is_a = (model_a == self.target_model_idx)
        target_is_b = (model_b == self.target_model_idx)

        if not target_is_a and not target_is_b:
            return self._handle_non_target()

        if target_is_a:
            is_detected = self.detect_target(response_a, True)
        else:
            is_detected = self.detect_target(response_b, True)

        if is_detected:
            self.total_votes_cast += 1
            if self.attack_type == "up":
                return 1 if target_is_a else -1
            else:
                return -1 if target_is_a else 1
        else:
            return self._handle_non_target()

    def _handle_non_target(self) -> int:
        """Apply non-detection strategy when target not detected.

        Four strategies (Appendix B.2):
          - "do_nothing": abstain (return 0 = tie, not counted as vote)
          - "random_upvote": randomly choose one model
          - "vote_tie": vote for a tie
          - "vote_both_bad": vote both as bad (tie)
        """
        if self.non_detection_strategy == "do_nothing":
            return 0  # Tie, not counted as an adversarial vote
        elif self.non_detection_strategy == "random_upvote":
            return 1 if self.rng.random() < 0.5 else -1
        elif self.non_detection_strategy == "vote_tie":
            return 0
        elif self.non_detection_strategy == "vote_both_bad":
            return 0
        return 0


def compute_bradley_terry_ratings(
    win_matrix: np.ndarray,
    total_comparisons: np.ndarray,
    scale: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """Compute Bradley-Terry coefficient ratings from pairwise win/loss data.

    Uses the iterative MM algorithm (Hunter, 2004) to estimate maximum
    likelihood ratings.

    Args:
        win_matrix: win_matrix[i, j] = number of times i beat j
        total_comparisons: total_comparisons[i, j] = total matches between i and j
        scale: scaling factor s (default 1.0)
        max_iter: maximum number of iterations
        tol: convergence tolerance

    Returns:
        ratings: array of Bradley-Terry coefficients
    """
    n = win_matrix.shape[0]
    ratings = np.zeros(n)

    for _ in range(max_iter):
        old_ratings = ratings.copy()
        exp_ratings = np.exp(ratings / scale)

        for i in range(n):
            numerator = win_matrix[i, :].sum()
            denominator = 0.0
            for j in range(n):
                if i != j and total_comparisons[i, j] > 0:
                    denominator += total_comparisons[i, j] / (exp_ratings[i] + exp_ratings[j])
            denominator *= exp_ratings[i]
            if denominator > 0:
                ratings[i] = scale * np.log(numerator / denominator)

        if np.max(np.abs(ratings - old_ratings)) < tol:
            break

    ratings -= ratings.mean()
    return ratings


class LeaderboardSimulation:
    """Simulates the Chatbot Arena leaderboard under adversarial voting.

    Tracks model rankings using Bradley-Terry coefficients and simulates
    the effect of adversarial votes as described in Section 3.

    The simulation iteratively adds adversarial interactions and computes
    rankings after every steps_per_check interactions. It tracks cumulative
    interactions and votes required to achieve attack objectives.
    """

    def __init__(
        self,
        model_names: List[str],
        initial_ratings: Optional[np.ndarray] = None,
        detector_accuracy: float = 0.95,
        false_positive_rate: float = 0.05,
        false_negative_rate: float = 0.05,
        bradley_terry_scale: float = 1.0,
        steps_per_check: int = 1000,
        seed: int = 42,
    ):
        self.model_names = list(model_names)
        self.n_models = len(self.model_names)
        self.detector_accuracy = detector_accuracy
        self.false_positive_rate = false_positive_rate
        self.false_negative_rate = false_negative_rate
        self.bradley_terry_scale = bradley_terry_scale
        self.steps_per_check = steps_per_check
        self.seed = seed

        self.simulator = VotingDataSimulator(model_names, seed=seed)

        self.win_matrix = np.zeros((self.n_models, self.n_models))
        self.total_comparisons = np.zeros((self.n_models, self.n_models))

        if initial_ratings is not None:
            self.simulator.ratings = initial_ratings.copy()

    def _initialize_historical_votes(self, num_preexisting_votes: int = 10000):
        """Initialize with some pre-existing historical votes to establish rankings."""
        for _ in range(num_preexisting_votes):
            a, b = self.simulator.sample_pair()
            result = self.simulator.simulate_vote(a, b)
            self.total_comparisons[a, b] += 1
            self.total_comparisons[b, a] += 1
            if result == 1:
                self.win_matrix[a, b] += 1
            elif result == -1:
                self.win_matrix[b, a] += 1

        self.simulator.ratings = compute_bradley_terry_ratings(
            self.win_matrix, self.total_comparisons, self.bradley_terry_scale
        )

    def run_attack(
        self,
        target_model_name: str,
        attack_type: str = "up",
        target_deltas: Optional[List[int]] = None,
        non_detection_strategy: str = "do_nothing",
        response_generator: Optional[Callable] = None,
        max_interactions: int = 500000,
    ) -> AttackOutcome:
        """Run an adversarial attack to shift a model's ranking.

        Simulates attacker interactions with Chatbot Arena, tracking how many
        votes and interactions are needed to achieve position changes.

        Args:
            target_model_name: Name of the model to manipulate
            attack_type: "up" (promote) or "down" (demote)
            target_deltas: List of position changes to track.
                Default: [1, 2, 3, 4, 5] for high-ranked, [1, 2, 5, 10, 20, 50] for low-ranked
            non_detection_strategy: Strategy when target not detected
            response_generator: Function(model_name, prompt) -> response string
            max_interactions: Maximum simulations to run before giving up

        Returns:
            AttackOutcome with votes and interactions needed for each target rank
        """
        self._initialize_historical_votes()

        target_idx = self.model_names.index(target_model_name)
        initial_rank = self.simulator.get_rank(target_idx)
        initial_total_votes = int(self.win_matrix.sum())

        if target_deltas is None:
            if initial_rank <= 5:
                target_deltas = [1, 2, 3, 4, 5]
            else:
                target_deltas = [1, 2, 5, 10, 20, 50]

        target_deltas = sorted(target_deltas)
        target_ranks = {}
        for d in target_deltas:
            if attack_type == "up":
                target_ranks[d] = max(1, initial_rank - d)
            else:
                target_ranks[d] = min(self.n_models, initial_rank + d)

        remaining_deltas = set(target_deltas)
        votes_to_ranks = {}
        interactions_to_ranks = {}

        adversarial_votes = 0
        total_interactions = 0

        rng = np.random.RandomState(self.seed + 1)

        for step in range(max_interactions):
            a, b = self.simulator.sample_pair()
            total_interactions += 1

            is_target_present = (a == target_idx) or (b == target_idx)
            target_in_a = (a == target_idx)

            if not is_target_present:
                result = self.simulator.simulate_vote(a, b)
                self.total_comparisons[a, b] += 1
                self.total_comparisons[b, a] += 1
                if result == 1:
                    self.win_matrix[a, b] += 1
                elif result == -1:
                    self.win_matrix[b, a] += 1
                self.simulator.update_ratings(
                    a if result == 1 else (b if result == -1 else None),
                    b if result == 1 else (a if result == -1 else None),
                )
                continue

            detected = rng.random() < self.detector_accuracy if is_target_present else (rng.random() < self.false_positive_rate)

            if detected:
                adversarial_votes += 1
                self.total_comparisons[a, b] += 1
                self.total_comparisons[b, a] += 1
                if attack_type == "up":
                    if target_in_a:
                        self.win_matrix[a, b] += 1
                        self.simulator.update_ratings(a, b)
                    else:
                        self.win_matrix[b, a] += 1
                        self.simulator.update_ratings(b, a)
                else:
                    if target_in_a:
                        self.win_matrix[b, a] += 1
                        self.simulator.update_ratings(b, a)
                    else:
                        self.win_matrix[a, b] += 1
                        self.simulator.update_ratings(a, b)
            else:
                if non_detection_strategy == "do_nothing":
                    pass

            if (step + 1) % self.steps_per_check == 0:
                self.simulator.ratings = compute_bradley_terry_ratings(
                    self.win_matrix, self.total_comparisons, self.bradley_terry_scale
                )
                current_rank = self.simulator.get_rank(target_idx)

                for delta in list(remaining_deltas):
                    tr = target_ranks[delta]
                    if (attack_type == "up" and current_rank <= tr) or \
                       (attack_type == "down" and current_rank >= tr):
                        votes_to_ranks[tr] = adversarial_votes
                        interactions_to_ranks[tr] = total_interactions
                        remaining_deltas.remove(delta)

                if not remaining_deltas:
                    break

        self.simulator.ratings = compute_bradley_terry_ratings(
            self.win_matrix, self.total_comparisons, self.bradley_terry_scale
        )

        return AttackOutcome(
            target_model=target_model_name,
            current_rank=initial_rank,
            total_votes=initial_total_votes,
            votes_to_ranks=votes_to_ranks,
            interactions_to_ranks=interactions_to_ranks,
        )


def run_vote_simulation(
    model_names: List[str],
    target_models: List[str],
    attack_type: str = "up",
    target_deltas: Optional[List[int]] = None,
    detector_accuracy: float = 0.95,
    false_positive_rate: float = 0.05,
    false_negative_rate: float = 0.05,
    non_detection_strategy: str = "do_nothing",
    bradley_terry_scale: float = 1.0,
    steps_per_check: int = 1000,
    max_interactions: int = 500000,
    seed: int = 42,
) -> List[AttackOutcome]:
    """Run voting simulation for multiple target models.

    Reproduces Tables 4 and 5 from Section 3.2.
    """
    results = []

    for target in target_models:
        sim = LeaderboardSimulation(
            model_names=model_names,
            detector_accuracy=detector_accuracy,
            false_positive_rate=false_positive_rate,
            false_negative_rate=false_negative_rate,
            bradley_terry_scale=bradley_terry_scale,
            steps_per_check=steps_per_check,
            seed=seed + model_names.index(target),
        )

        outcome = sim.run_attack(
            target_model_name=target,
            attack_type=attack_type,
            target_deltas=target_deltas,
            non_detection_strategy=non_detection_strategy,
            max_interactions=max_interactions,
        )
        results.append(outcome)

    return results


def ablation_detector_accuracy(
    model_names: List[str],
    target_model: str,
    attack_type: str = "up",
    target_deltas: List[int] = None,
    accuracies: List[float] = None,
    **kwargs,
) -> Dict[float, AttackOutcome]:
    """Ablation study for detector accuracy (Appendix B.2, Table 8)."""
    if accuracies is None:
        accuracies = [1.0, 0.95, 0.9]
    if target_deltas is None:
        target_deltas = [1, 2, 5, 10, 20, 50]

    results = {}
    for acc in accuracies:
        fnr = fpr = 1.0 - acc
        sim = LeaderboardSimulation(
            model_names=model_names,
            detector_accuracy=acc,
            false_positive_rate=fpr,
            false_negative_rate=fnr,
            **kwargs,
        )
        outcome = sim.run_attack(
            target_model_name=target_model,
            attack_type=attack_type,
            target_deltas=target_deltas,
        )
        results[acc] = outcome
    return results


def ablation_non_detection_strategies(
    model_names: List[str],
    target_model: str,
    attack_type: str = "up",
    target_deltas: Optional[List[int]] = None,
    strategies: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, AttackOutcome]:
    """Ablation study for non-detection strategies (Appendix B.2, Table 9)."""
    if strategies is None:
        strategies = ["do_nothing", "random_upvote", "vote_tie", "vote_both_bad"]
    if target_deltas is None:
        target_deltas = [1, 2, 5, 10, 20, 50]

    results = {}
    for strategy in strategies:
        sim = LeaderboardSimulation(model_names=model_names, **kwargs)
        outcome = sim.run_attack(
            target_model_name=target_model,
            attack_type=attack_type,
            target_deltas=target_deltas,
            non_detection_strategy=strategy,
        )
        results[strategy] = outcome
    return results
