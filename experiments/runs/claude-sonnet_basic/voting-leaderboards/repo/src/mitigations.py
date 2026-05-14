"""
Mitigations against adversarial manipulation of voting-based leaderboards.

This module implements the mitigations described in Section 4 of the paper:
1. Authentication (Section 4.2.1)
2. Rate limiting (Section 4.2.2)
3. Malicious user identification (Section 4.2.3)
   - Scenario 1: Known benign distribution (likelihood test)
   - Scenario 2: Known benign and malicious distributions (Neyman-Pearson)
4. Increasing action cost (Section 4.2.4)

The module also implements the cost model from Section 4.1.
"""

import numpy as np
import logging
from typing import Optional
from scipy import stats

logger = logging.getLogger(__name__)


# ============================================================================
# Section 4.1: Cost Model
# ============================================================================

def compute_attack_cost(
    n_actions: int,
    max_actions_per_account: int,
    cost_per_account: float,
    cost_per_action: float,
    detector_cost: float,
) -> float:
    """
    Compute the total cost of an adversarial attack.

    From Section 4.1:
        Total cost = ceil(N/m) * c_account + N * c_action + c_detector

    where:
        N = total number of actions
        m = maximum actions per account
        c_account = cost per account
        c_action = cost per action
        c_detector = one-time detector training cost

    Args:
        n_actions: Total number of actions (N)
        max_actions_per_account: Maximum actions per account (m)
        cost_per_account: Cost per account (c_account)
        cost_per_action: Cost per action (c_action)
        detector_cost: One-time detector training cost (c_detector)

    Returns:
        Total attack cost
    """
    import math

    n_accounts = math.ceil(n_actions / max_actions_per_account)
    account_cost = n_accounts * cost_per_account
    action_cost = n_actions * cost_per_action

    total_cost = account_cost + action_cost + detector_cost
    return total_cost


def estimate_detector_training_cost(
    n_prompts: int = 200,
    n_proprietary_models: int = 10,
    n_opensource_models: int = 20,
    responses_per_model: int = 50,
    max_output_tokens: int = 512,
    proprietary_cost_per_million: float = 5.00,
    opensource_cost_per_million: float = 1.80,
) -> float:
    """
    Estimate the cost of training the target model detector.

    From Appendix A.3:
    - Proprietary model: $5.00 per 1M output tokens
    - Open-source model: $1.80 per 1M output tokens
    - 50 responses per model, 512 tokens each
    - 200 prompts total

    Args:
        n_prompts: Number of prompts used for training
        n_proprietary_models: Number of proprietary models
        n_opensource_models: Number of open-source models
        responses_per_model: Responses per model per prompt
        max_output_tokens: Maximum tokens per response
        proprietary_cost_per_million: Cost per million tokens for proprietary models
        opensource_cost_per_million: Cost per million tokens for open-source models

    Returns:
        Estimated training cost in USD
    """
    # Cost per prompt
    proprietary_cost_per_prompt = (
        proprietary_cost_per_million
        * (max_output_tokens * responses_per_model)
        / 1e6
        * n_proprietary_models
    )
    opensource_cost_per_prompt = (
        opensource_cost_per_million
        * (max_output_tokens * responses_per_model)
        / 1e6
        * n_opensource_models
    )
    cost_per_prompt = proprietary_cost_per_prompt + opensource_cost_per_prompt

    total_cost = cost_per_prompt * n_prompts
    return total_cost


# ============================================================================
# Section 4.2.3: Malicious User Identification
# ============================================================================

class MaliciousUserDetector:
    """
    Detects malicious users based on their voting patterns.

    Implements the anomaly detection approach described in Section 4.2.3.
    """

    def __init__(
        self,
        benign_vote_distribution: np.ndarray,
        model_names: list,
        significance_level: float = 0.01,
        n_simulations: int = 10000,
        random_seed: int = 42,
    ):
        """
        Initialize the detector.

        Args:
            benign_vote_distribution: Expected vote distribution for benign users
                                      (probability of voting for each model)
            model_names: List of model names
            significance_level: Significance level alpha for hypothesis test (default: 0.01)
            n_simulations: Number of simulations for p-value estimation
            random_seed: Random seed
        """
        self.benign_dist = benign_vote_distribution / benign_vote_distribution.sum()
        self.model_names = model_names
        self.alpha = significance_level
        self.n_simulations = n_simulations
        self.rng = np.random.RandomState(random_seed)

    def compute_test_statistic(self, vote_sequence: list) -> float:
        """
        Compute the test statistic T(x) = -2 * ln(L(x | H_benign)).

        From Section 4.2.3:
            L(x | H_benign) = prod_i Pr(x_i | H_benign)
            T(x) = -2 * ln(L(x | H_benign))

        Args:
            vote_sequence: List of model indices voted for

        Returns:
            Test statistic value
        """
        log_likelihood = 0.0
        for vote in vote_sequence:
            prob = self.benign_dist[vote]
            if prob > 0:
                log_likelihood += np.log(prob)
            else:
                log_likelihood += -1e10  # Very small probability

        return -2 * log_likelihood

    def compute_empirical_pvalue(self, vote_sequence: list) -> float:
        """
        Compute the empirical p-value for a vote sequence.

        From Section 4.2.3:
            p = (1/m) * sum_j I{T(s^j) >= T(x)}

        where s^j are simulated sequences under H_benign.

        Args:
            vote_sequence: List of model indices voted for

        Returns:
            Empirical p-value
        """
        observed_stat = self.compute_test_statistic(vote_sequence)
        n = len(vote_sequence)

        # Simulate sequences under null hypothesis
        simulated_stats = []
        for _ in range(self.n_simulations):
            simulated_seq = self.rng.choice(
                len(self.benign_dist), size=n, p=self.benign_dist
            )
            simulated_stats.append(self.compute_test_statistic(simulated_seq))

        # Compute empirical p-value
        p_value = np.mean(np.array(simulated_stats) >= observed_stat)
        return p_value

    def is_malicious(self, vote_sequence: list) -> tuple:
        """
        Determine if a user is malicious based on their vote sequence.

        Rejects H_benign if p-value < alpha.

        Args:
            vote_sequence: List of model indices voted for

        Returns:
            Tuple of (is_malicious, p_value)
        """
        p_value = self.compute_empirical_pvalue(vote_sequence)
        return p_value < self.alpha, p_value

    def evaluate_detection(
        self,
        benign_sequences: list,
        malicious_sequences: list,
    ) -> dict:
        """
        Evaluate the detector on benign and malicious vote sequences.

        Args:
            benign_sequences: List of vote sequences from benign users
            malicious_sequences: List of vote sequences from malicious users

        Returns:
            Dictionary with detection metrics
        """
        # True negatives (benign correctly identified as benign)
        tn = sum(
            1 for seq in benign_sequences
            if not self.is_malicious(seq)[0]
        )
        # False positives (benign incorrectly identified as malicious)
        fp = len(benign_sequences) - tn

        # True positives (malicious correctly identified as malicious)
        tp = sum(
            1 for seq in malicious_sequences
            if self.is_malicious(seq)[0]
        )
        # False negatives (malicious incorrectly identified as benign)
        fn = len(malicious_sequences) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "false_positive_rate": fp / len(benign_sequences) if benign_sequences else 0.0,
            "true_positive_rate": tp / len(malicious_sequences) if malicious_sequences else 0.0,
        }


class NeymanPearsonDetector:
    """
    Malicious user detector using the Neyman-Pearson Lemma.

    Implements Scenario 2 from Section 4.2.3, where the defender knows both
    the benign and malicious distributions.

    The defender releases perturbed rankings to reduce the attacker's knowledge
    of the true values.
    """

    def __init__(
        self,
        true_ratings: np.ndarray,
        model_names: list,
        noise_scale: float = 0.1,
        significance_level: float = 0.01,
        scale: float = 1.0,
        random_seed: int = 42,
    ):
        """
        Initialize the detector.

        Args:
            true_ratings: True Bradley-Terry ratings
            model_names: List of model names
            noise_scale: Scale of Gaussian noise added to ratings before release
            significance_level: Significance level for hypothesis test
            scale: Scaling factor for win probability computation
            random_seed: Random seed
        """
        self.true_ratings = true_ratings
        self.model_names = model_names
        self.noise_scale = noise_scale
        self.alpha = significance_level
        self.scale = scale
        self.rng = np.random.RandomState(random_seed)

        # Compute benign distribution from true ratings
        self.benign_dist = self._compute_vote_distribution(true_ratings)

        # Generate perturbed ratings for release
        self.perturbed_ratings = self._perturb_ratings(true_ratings, noise_scale)

        # Compute adversarial distribution from perturbed ratings
        # (attacker uses perturbed ratings to mimic benign behavior)
        self.adversarial_dist = self._compute_vote_distribution(self.perturbed_ratings)

    def _compute_vote_distribution(self, ratings: np.ndarray) -> np.ndarray:
        """
        Compute vote distribution from Bradley-Terry ratings.

        Pr_B(i) = prod_j Pr_B(i preferred over j | ratings)

        Args:
            ratings: Bradley-Terry ratings

        Returns:
            Vote probability distribution
        """
        n = len(ratings)
        vote_probs = np.ones(n)

        for i in range(n):
            for j in range(n):
                if i != j:
                    p_ij = 1.0 / (1.0 + np.exp(-(ratings[i] - ratings[j]) / self.scale))
                    vote_probs[i] *= p_ij

        # Normalize
        total = vote_probs.sum()
        if total > 0:
            vote_probs = vote_probs / total

        return vote_probs

    def _perturb_ratings(
        self, ratings: np.ndarray, noise_scale: float
    ) -> np.ndarray:
        """
        Add Gaussian noise to ratings before releasing to users.

        From Section 4.2.3: "we add scaled Gaussian noise to Bradley-Terry
        coefficient ratings before releasing the rating."

        Args:
            ratings: True Bradley-Terry ratings
            noise_scale: Scale of Gaussian noise

        Returns:
            Perturbed ratings
        """
        noise = self.rng.randn(len(ratings)) * noise_scale
        perturbed = ratings + noise
        # Ensure positive ratings
        perturbed = np.maximum(perturbed, 1e-6)
        return perturbed

    def compute_likelihood_ratio(self, vote_sequence: list) -> float:
        """
        Compute the likelihood ratio Lambda(x) = Pr_M(x) / Pr_B(x).

        From Section 4.2.3 (Neyman-Pearson Lemma).

        Args:
            vote_sequence: List of model indices voted for

        Returns:
            Log likelihood ratio
        """
        log_ratio = 0.0
        for vote in vote_sequence:
            p_malicious = self.adversarial_dist[vote]
            p_benign = self.benign_dist[vote]

            if p_benign > 0 and p_malicious > 0:
                log_ratio += np.log(p_malicious / p_benign)
            elif p_benign == 0:
                log_ratio += 1e10  # Definitely malicious
            elif p_malicious == 0:
                log_ratio += -1e10  # Definitely benign

        return log_ratio

    def is_malicious(
        self, vote_sequence: list, threshold: float = 0.0
    ) -> tuple:
        """
        Determine if a user is malicious using the likelihood ratio test.

        Args:
            vote_sequence: List of model indices voted for
            threshold: Decision threshold for log likelihood ratio

        Returns:
            Tuple of (is_malicious, log_likelihood_ratio)
        """
        log_ratio = self.compute_likelihood_ratio(vote_sequence)
        return log_ratio > threshold, log_ratio

    def evaluate_detection(
        self,
        benign_sequences: list,
        malicious_sequences: list,
        threshold: float = 0.0,
    ) -> dict:
        """
        Evaluate the Neyman-Pearson detector.

        Args:
            benign_sequences: List of vote sequences from benign users
            malicious_sequences: List of vote sequences from malicious users
            threshold: Decision threshold

        Returns:
            Dictionary with detection metrics
        """
        tn = sum(
            1 for seq in benign_sequences
            if not self.is_malicious(seq, threshold)[0]
        )
        fp = len(benign_sequences) - tn
        tp = sum(
            1 for seq in malicious_sequences
            if self.is_malicious(seq, threshold)[0]
        )
        fn = len(malicious_sequences) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "false_positive_rate": fp / len(benign_sequences) if benign_sequences else 0.0,
            "true_positive_rate": tp / len(malicious_sequences) if malicious_sequences else 0.0,
        }

    def compute_utility_impact(self) -> float:
        """
        Compute the utility impact of releasing perturbed ratings.

        From Section 4.2.3: "we measure utility as the average absolute change
        in the ranking of any item."

        Returns:
            Average absolute rank change due to perturbation
        """
        from bradley_terry import get_rankings

        true_rankings = get_rankings(self.true_ratings, self.model_names)
        perturbed_rankings = get_rankings(self.perturbed_ratings, self.model_names)

        true_rank_dict = {name: rank for rank, name, _ in true_rankings}
        perturbed_rank_dict = {name: rank for rank, name, _ in perturbed_rankings}

        rank_changes = [
            abs(true_rank_dict[name] - perturbed_rank_dict[name])
            for name in self.model_names
        ]

        return np.mean(rank_changes)


def generate_adversarial_vote_sequence(
    target_model_idx: int,
    n_votes: int,
    n_models: int,
    benign_dist: np.ndarray,
    attack_direction: str = "up",
    use_public_rankings: bool = False,
    rng: Optional[np.random.RandomState] = None,
) -> list:
    """
    Generate an adversarial vote sequence for a target model.

    Args:
        target_model_idx: Index of the target model
        n_votes: Number of votes to generate
        n_models: Total number of models
        benign_dist: Benign vote distribution
        attack_direction: "up" (promote) or "down" (demote)
        use_public_rankings: Whether the attacker uses public rankings to mimic benign behavior
        rng: Random number generator

    Returns:
        List of model indices voted for
    """
    if rng is None:
        rng = np.random.RandomState(42)

    votes = []
    for _ in range(n_votes):
        if use_public_rankings:
            # Attacker mimics benign behavior for non-target votes
            # but always votes for/against target when detected
            # For simplicity, use benign distribution with target model boosted
            modified_dist = benign_dist.copy()
            if attack_direction == "up":
                modified_dist[target_model_idx] = 1.0  # Always vote for target
            else:
                modified_dist[target_model_idx] = 0.0  # Never vote for target
            modified_dist = modified_dist / modified_dist.sum()
            vote = rng.choice(n_models, p=modified_dist)
        else:
            # Naive adversary: randomly choose between non-target models
            # but always vote for/against target
            if attack_direction == "up":
                vote = target_model_idx
            else:
                # Vote for a random non-target model
                non_target = [i for i in range(n_models) if i != target_model_idx]
                vote = rng.choice(non_target)

        votes.append(vote)

    return votes


def run_mitigation_experiment(
    true_ratings: np.ndarray,
    model_names: list,
    target_model_idx: int,
    n_benign_users: int = 100,
    n_malicious_users: int = 100,
    votes_per_user: int = 50,
    noise_scales: list = None,
    random_seed: int = 42,
) -> dict:
    """
    Run the full mitigation experiment from Section 4.3.

    Evaluates both Scenario 1 (known benign distribution) and
    Scenario 2 (known benign and malicious distributions with perturbed leaderboard).

    Args:
        true_ratings: True Bradley-Terry ratings
        model_names: List of model names
        target_model_idx: Index of the target model
        n_benign_users: Number of benign users to simulate
        n_malicious_users: Number of malicious users to simulate
        votes_per_user: Number of votes per user
        noise_scales: List of noise scales to evaluate for Scenario 2
        random_seed: Random seed

    Returns:
        Dictionary with results for both scenarios
    """
    if noise_scales is None:
        noise_scales = [0.0, 0.1, 0.2, 0.5, 1.0]

    rng = np.random.RandomState(random_seed)
    n_models = len(model_names)

    # Compute benign vote distribution
    from bradley_terry import compute_benign_vote_distribution
    benign_dist = compute_benign_vote_distribution(true_ratings)

    # Generate benign vote sequences
    benign_sequences = []
    for _ in range(n_benign_users):
        seq = list(rng.choice(n_models, size=votes_per_user, p=benign_dist))
        benign_sequences.append(seq)

    # Generate naive adversarial vote sequences (random non-target)
    naive_adversarial_sequences = []
    for _ in range(n_malicious_users):
        seq = generate_adversarial_vote_sequence(
            target_model_idx=target_model_idx,
            n_votes=votes_per_user,
            n_models=n_models,
            benign_dist=benign_dist,
            attack_direction="up",
            use_public_rankings=False,
            rng=rng,
        )
        naive_adversarial_sequences.append(seq)

    # Generate sophisticated adversarial sequences (using public rankings)
    sophisticated_adversarial_sequences = []
    for _ in range(n_malicious_users):
        seq = generate_adversarial_vote_sequence(
            target_model_idx=target_model_idx,
            n_votes=votes_per_user,
            n_models=n_models,
            benign_dist=benign_dist,
            attack_direction="up",
            use_public_rankings=True,
            rng=rng,
        )
        sophisticated_adversarial_sequences.append(seq)

    results = {}

    # Scenario 1: Known benign distribution
    detector_s1 = MaliciousUserDetector(
        benign_vote_distribution=benign_dist,
        model_names=model_names,
        significance_level=0.01,
        random_seed=random_seed,
    )

    results["scenario1"] = {
        "naive_adversary": detector_s1.evaluate_detection(
            benign_sequences, naive_adversarial_sequences
        ),
        "sophisticated_adversary": detector_s1.evaluate_detection(
            benign_sequences, sophisticated_adversarial_sequences
        ),
    }

    # Scenario 2: Known benign and malicious distributions with noise
    results["scenario2"] = {}
    for noise_scale in noise_scales:
        detector_s2 = NeymanPearsonDetector(
            true_ratings=true_ratings,
            model_names=model_names,
            noise_scale=noise_scale,
            significance_level=0.01,
            random_seed=random_seed,
        )

        # Adversary uses perturbed rankings
        perturbed_adversarial_sequences = []
        for _ in range(n_malicious_users):
            seq = generate_adversarial_vote_sequence(
                target_model_idx=target_model_idx,
                n_votes=votes_per_user,
                n_models=n_models,
                benign_dist=detector_s2.adversarial_dist,
                attack_direction="up",
                use_public_rankings=True,
                rng=rng,
            )
            perturbed_adversarial_sequences.append(seq)

        detection_results = detector_s2.evaluate_detection(
            benign_sequences, perturbed_adversarial_sequences
        )
        utility_impact = detector_s2.compute_utility_impact()

        results["scenario2"][noise_scale] = {
            "detection": detection_results,
            "utility_impact": utility_impact,
            "noise_scale": noise_scale,
        }

    return results
