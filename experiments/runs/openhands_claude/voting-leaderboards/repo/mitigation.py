"""
Defense mechanisms against adversarial manipulation of voting-based leaderboards.

Implements the mitigations from Section 4:
  - Section 4.2.3: Malicious user identification via likelihood tests
    - Scenario 1: Known benign distribution (likelihood ratio test, Eq. 1-3)
    - Scenario 2: Known benign and malicious distributions (Neyman-Pearson, Eq. 4-6)
  - Section 4.2.4: Perturbed leaderboard defense (Gaussian noise on BT ratings)
  - Section 4.1: Attack cost model

Reproduces Figures 4, 5, and 6 from the paper.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from bradley_terry import (
    compute_vote_distribution,
    fit_bradley_terry,
    get_rank,
    get_rankings,
    strengths_to_elo,
)
from config import BradleyTerryConfig, MitigationConfig
from data import VoteRecord
from simulation import AdversarialAttacker, sample_model_pair


# ---------------------------------------------------------------------------
# User voting history
# ---------------------------------------------------------------------------

@dataclass
class UserVotingHistory:
    """Records a user's voting behavior for anomaly detection."""
    user_id: str
    voted_for: List[str]  # model names the user voted for
    model_pairs_seen: List[Tuple[str, str]]  # (model_a, model_b) pairs shown


# ---------------------------------------------------------------------------
# Scenario 1: Known benign distribution (Section 4.2.3)
# ---------------------------------------------------------------------------

def compute_log_likelihood_benign(
    voted_for: List[str],
    benign_probs: Dict[str, float],
) -> float:
    """
    Compute the log-likelihood of a user's voting sequence under the benign
    distribution (Eq. 1 in the paper).

    L(x | H_benign) = prod_{i=1}^{n} Pr(x_i | H_benign)

    Args:
        voted_for: Sequence of models the user voted for.
        benign_probs: Pr(model | H_benign) for each model.

    Returns:
        Log-likelihood value.
    """
    log_likelihood = 0.0
    for model in voted_for:
        p = benign_probs.get(model, 1e-10)
        log_likelihood += np.log(max(p, 1e-300))
    return log_likelihood


def compute_test_statistic(log_likelihood: float) -> float:
    """
    Compute the test statistic T(x) = -2 * ln(L(x | H_benign)) (Eq. 2).
    """
    return -2.0 * log_likelihood


def compute_empirical_pvalue(
    observed_statistic: float,
    benign_probs: Dict[str, float],
    sequence_length: int,
    num_simulations: int = 10000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Compute the empirical p-value by simulating sequences under H_benign (Eq. 3).

    p = (1/m) * sum_{j=1}^{m} I{T(s^j) >= T(x)}

    Args:
        observed_statistic: T(x) for the observed sequence.
        benign_probs: Benign model vote probabilities.
        sequence_length: Length of the voting sequence.
        num_simulations: Number of simulated sequences (m in the paper).
        rng: Random number generator.

    Returns:
        Empirical p-value.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    models = list(benign_probs.keys())
    probs = np.array([benign_probs[m] for m in models])
    probs = probs / probs.sum()  # Normalize

    count_extreme = 0
    for _ in range(num_simulations):
        # Sample a sequence under H_benign
        simulated_votes = rng.choice(models, size=sequence_length, p=probs)
        sim_log_ll = sum(np.log(max(benign_probs.get(m, 1e-10), 1e-300)) for m in simulated_votes)
        sim_statistic = compute_test_statistic(sim_log_ll)
        if sim_statistic >= observed_statistic:
            count_extreme += 1

    return count_extreme / num_simulations


def detect_malicious_user_scenario1(
    user_history: UserVotingHistory,
    benign_probs: Dict[str, float],
    alpha: float = 0.01,
    num_simulations: int = 10000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[bool, float]:
    """
    Detect a malicious user using the likelihood test (Scenario 1, Section 4.2.3).

    Rejects H_benign if p-value < alpha (paper uses alpha=0.01).

    Args:
        user_history: The user's voting history.
        benign_probs: Expected benign vote distribution.
        alpha: Significance level.
        num_simulations: Number of simulations for p-value estimation.
        rng: Random number generator.

    Returns:
        Tuple of (is_malicious, p_value).
    """
    if not user_history.voted_for:
        return False, 1.0

    log_ll = compute_log_likelihood_benign(user_history.voted_for, benign_probs)
    test_stat = compute_test_statistic(log_ll)

    p_value = compute_empirical_pvalue(
        observed_statistic=test_stat,
        benign_probs=benign_probs,
        sequence_length=len(user_history.voted_for),
        num_simulations=num_simulations,
        rng=rng,
    )

    return p_value < alpha, p_value


# ---------------------------------------------------------------------------
# Scenario 2: Known benign and malicious distributions (Section 4.2.3)
# ---------------------------------------------------------------------------

def compute_likelihood_ratio(
    voted_for: List[str],
    benign_probs: Dict[str, float],
    malicious_probs: Dict[str, float],
) -> float:
    """
    Compute the Neyman-Pearson likelihood ratio (Eq. 4):

    Lambda(x) = Pr_M(x) / Pr_B(x)

    Args:
        voted_for: Sequence of models the user voted for.
        benign_probs: Pr(model | H_benign).
        malicious_probs: Pr(model | H_malicious).

    Returns:
        Log likelihood ratio.
    """
    log_ratio = 0.0
    for model in voted_for:
        p_b = max(benign_probs.get(model, 1e-10), 1e-300)
        p_m = max(malicious_probs.get(model, 1e-10), 1e-300)
        log_ratio += np.log(p_m) - np.log(p_b)
    return log_ratio


def detect_malicious_user_scenario2(
    user_history: UserVotingHistory,
    benign_probs: Dict[str, float],
    malicious_probs: Dict[str, float],
    threshold: float = 0.0,
) -> Tuple[bool, float]:
    """
    Detect a malicious user using the Neyman-Pearson likelihood ratio test
    (Scenario 2, Section 4.2.3).

    The optimal decision rule is based on the likelihood ratio (NP Lemma).
    Classify as malicious if log Lambda(x) > threshold.

    Args:
        user_history: The user's voting history.
        benign_probs: Benign vote distribution.
        malicious_probs: Malicious vote distribution (based on perturbed ratings).
        threshold: Decision threshold for log likelihood ratio.

    Returns:
        Tuple of (is_malicious, log_likelihood_ratio).
    """
    if not user_history.voted_for:
        return False, 0.0

    log_lr = compute_likelihood_ratio(user_history.voted_for, benign_probs, malicious_probs)
    return log_lr > threshold, log_lr


# ---------------------------------------------------------------------------
# Perturbed leaderboard defense (Section 4.2.3, Scenario 2)
# ---------------------------------------------------------------------------

def perturb_bt_ratings(
    strengths: Dict[str, float],
    noise_scale: float,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """
    Add scaled Gaussian noise to Bradley-Terry ratings before releasing them.

    The defender releases perturbed rankings to reduce the attacker's knowledge
    of true values (Section 4.2.3, Scenario 2).

    Args:
        strengths: True Bradley-Terry strength parameters.
        noise_scale: Standard deviation of Gaussian noise (sigma).
        rng: Random number generator.

    Returns:
        Perturbed strength parameters.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    perturbed = {}
    for model, strength in strengths.items():
        noise = rng.normal(0, noise_scale * strength)
        perturbed[model] = max(strength + noise, 1e-10)

    return perturbed


def compute_rank_change_from_perturbation(
    true_strengths: Dict[str, float],
    perturbed_strengths: Dict[str, float],
) -> float:
    """
    Compute the average absolute rank change due to perturbation.

    Used to measure the utility cost of the perturbed leaderboard defense
    (Figure 6 in the paper).

    Args:
        true_strengths: True BT strength parameters.
        perturbed_strengths: Perturbed BT strength parameters.

    Returns:
        Mean absolute rank change across all models.
    """
    true_rankings = {model: rank for rank, model, _ in get_rankings(true_strengths)}
    perturbed_rankings = {model: rank for rank, model, _ in get_rankings(perturbed_strengths)}

    rank_changes = []
    for model in true_rankings:
        if model in perturbed_rankings:
            rank_changes.append(abs(true_rankings[model] - perturbed_rankings[model]))

    return float(np.mean(rank_changes)) if rank_changes else 0.0


# ---------------------------------------------------------------------------
# Simulation of malicious user detection (Figures 4 and 5)
# ---------------------------------------------------------------------------

def simulate_detection_scenario1(
    target_model: str,
    models: List[str],
    true_strengths: Dict[str, float],
    num_votes_per_user: int,
    num_benign_users: int = 1000,
    num_malicious_users: int = 100,
    alpha: float = 0.01,
    num_simulations: int = 1000,
    attacker_uses_public_ranking: bool = False,
    random_seed: int = 42,
) -> Tuple[float, float]:
    """
    Simulate Scenario 1 detection: defender uses historical benign distribution.

    Reproduces Figure 4 from the paper.

    Args:
        target_model: The model the attacker is promoting.
        models: All models in the arena.
        true_strengths: True BT strength parameters.
        num_votes_per_user: Number of votes per user session.
        num_benign_users: Number of benign users to simulate.
        num_malicious_users: Number of malicious users to simulate.
        alpha: Significance level for detection.
        num_simulations: Simulations for p-value estimation.
        attacker_uses_public_ranking: If True, attacker uses public rankings
                                       to mimic benign behavior (harder to detect).
        random_seed: Random seed.

    Returns:
        Tuple of (false_positive_rate, true_positive_rate).
    """
    rng = np.random.default_rng(random_seed)

    # Compute benign vote distribution from true BT ratings
    benign_probs = compute_vote_distribution(true_strengths)

    # Simulate benign users
    false_positives = 0
    for _ in range(num_benign_users):
        voted_for = _simulate_benign_votes(models, benign_probs, num_votes_per_user, rng)
        history = UserVotingHistory(user_id="benign", voted_for=voted_for, model_pairs_seen=[])
        is_malicious, _ = detect_malicious_user_scenario1(
            history, benign_probs, alpha, num_simulations, rng
        )
        if is_malicious:
            false_positives += 1

    fpr = false_positives / num_benign_users

    # Simulate malicious users
    true_positives = 0
    for _ in range(num_malicious_users):
        if attacker_uses_public_ranking:
            # Attacker mimics benign distribution using public rankings
            voted_for = _simulate_benign_votes(models, benign_probs, num_votes_per_user, rng)
        else:
            # Naive attacker: always votes for target when detected, random otherwise
            voted_for = _simulate_malicious_votes_naive(
                target_model, models, benign_probs, num_votes_per_user, rng
            )

        history = UserVotingHistory(user_id="malicious", voted_for=voted_for, model_pairs_seen=[])
        is_malicious, _ = detect_malicious_user_scenario1(
            history, benign_probs, alpha, num_simulations, rng
        )
        if is_malicious:
            true_positives += 1

    tpr = true_positives / num_malicious_users

    return fpr, tpr


def simulate_detection_scenario2(
    target_model: str,
    models: List[str],
    true_strengths: Dict[str, float],
    noise_scale: float,
    num_votes_per_user: int,
    num_benign_users: int = 1000,
    num_malicious_users: int = 100,
    alpha: float = 0.01,
    random_seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Simulate Scenario 2 detection: defender releases perturbed leaderboard.

    Reproduces Figure 5 from the paper.

    Args:
        target_model: The model the attacker is promoting.
        models: All models.
        true_strengths: True BT strength parameters.
        noise_scale: Gaussian noise scale for perturbation.
        num_votes_per_user: Votes per user session.
        num_benign_users: Number of benign users.
        num_malicious_users: Number of malicious users.
        alpha: Significance level.
        random_seed: Random seed.

    Returns:
        Tuple of (false_positive_rate, true_positive_rate, mean_rank_change).
    """
    rng = np.random.default_rng(random_seed)

    # Compute true benign distribution
    benign_probs = compute_vote_distribution(true_strengths)

    # Perturb the leaderboard
    perturbed_strengths = perturb_bt_ratings(true_strengths, noise_scale, rng)
    malicious_probs = compute_vote_distribution(perturbed_strengths)

    # Compute utility cost
    mean_rank_change = compute_rank_change_from_perturbation(true_strengths, perturbed_strengths)

    # Simulate benign users (use true distribution)
    false_positives = 0
    for _ in range(num_benign_users):
        voted_for = _simulate_benign_votes(models, benign_probs, num_votes_per_user, rng)
        history = UserVotingHistory(user_id="benign", voted_for=voted_for, model_pairs_seen=[])
        is_malicious, _ = detect_malicious_user_scenario2(
            history, benign_probs, malicious_probs
        )
        if is_malicious:
            false_positives += 1

    fpr = false_positives / num_benign_users

    # Simulate malicious users (use perturbed distribution to choose non-target votes)
    true_positives = 0
    for _ in range(num_malicious_users):
        voted_for = _simulate_malicious_votes_with_perturbed(
            target_model, models, malicious_probs, num_votes_per_user, rng
        )
        history = UserVotingHistory(user_id="malicious", voted_for=voted_for, model_pairs_seen=[])
        is_malicious, _ = detect_malicious_user_scenario2(
            history, benign_probs, malicious_probs
        )
        if is_malicious:
            true_positives += 1

    tpr = true_positives / num_malicious_users

    return fpr, tpr, mean_rank_change


def evaluate_noise_scales(
    target_model: str,
    models: List[str],
    true_strengths: Dict[str, float],
    noise_scales: List[float],
    num_votes_per_user: int = 50,
    num_benign_users: int = 500,
    num_malicious_users: int = 100,
    random_seed: int = 42,
) -> Dict[float, Dict[str, float]]:
    """
    Evaluate detection performance across different noise scales.

    Reproduces Figures 5 and 6 from the paper.

    Returns:
        {noise_scale -> {"fpr": float, "tpr": float, "mean_rank_change": float}}
    """
    results = {}
    for noise_scale in noise_scales:
        fpr, tpr, mean_rank_change = simulate_detection_scenario2(
            target_model=target_model,
            models=models,
            true_strengths=true_strengths,
            noise_scale=noise_scale,
            num_votes_per_user=num_votes_per_user,
            num_benign_users=num_benign_users,
            num_malicious_users=num_malicious_users,
            random_seed=random_seed,
        )
        results[noise_scale] = {
            "fpr": fpr,
            "tpr": tpr,
            "mean_rank_change": mean_rank_change,
        }
    return results


# ---------------------------------------------------------------------------
# Helper functions for vote simulation
# ---------------------------------------------------------------------------

def _simulate_benign_votes(
    models: List[str],
    benign_probs: Dict[str, float],
    num_votes: int,
    rng: np.random.Generator,
) -> List[str]:
    """Simulate a benign user's votes according to the benign distribution."""
    model_list = list(benign_probs.keys())
    probs = np.array([benign_probs[m] for m in model_list])
    probs = probs / probs.sum()
    voted_for = rng.choice(model_list, size=num_votes, p=probs)
    return list(voted_for)


def _simulate_malicious_votes_naive(
    target_model: str,
    models: List[str],
    benign_probs: Dict[str, float],
    num_votes: int,
    rng: np.random.Generator,
    detection_accuracy: float = 0.95,
) -> List[str]:
    """
    Simulate a naive malicious user who always votes for the target when detected,
    and randomly votes otherwise (Figure 4, naive adversary).
    """
    voted_for = []
    model_list = list(benign_probs.keys())
    probs = np.array([benign_probs[m] for m in model_list])
    probs = probs / probs.sum()

    for _ in range(num_votes):
        # Sample a pair
        idx_a, idx_b = rng.choice(len(models), size=2, replace=False)
        model_a, model_b = models[idx_a], models[idx_b]

        # Check if target is in the pair
        target_present = (model_a == target_model or model_b == target_model)

        if target_present and rng.random() < detection_accuracy:
            voted_for.append(target_model)
        else:
            # Random vote from the pair
            voted_for.append(rng.choice([model_a, model_b]))

    return voted_for


def _simulate_malicious_votes_with_perturbed(
    target_model: str,
    models: List[str],
    perturbed_probs: Dict[str, float],
    num_votes: int,
    rng: np.random.Generator,
    detection_accuracy: float = 0.95,
) -> List[str]:
    """
    Simulate a sophisticated malicious user who uses the perturbed leaderboard
    to choose non-target votes (Figure 5, adversary using perturbed ranking).
    """
    voted_for = []
    model_list = list(perturbed_probs.keys())
    probs = np.array([perturbed_probs[m] for m in model_list])
    probs = probs / probs.sum()

    for _ in range(num_votes):
        idx_a, idx_b = rng.choice(len(models), size=2, replace=False)
        model_a, model_b = models[idx_a], models[idx_b]

        target_present = (model_a == target_model or model_b == target_model)

        if target_present and rng.random() < detection_accuracy:
            voted_for.append(target_model)
        else:
            # Use perturbed distribution to choose (mimics public ranking)
            pair_probs = np.array([
                perturbed_probs.get(model_a, 1e-10),
                perturbed_probs.get(model_b, 1e-10),
            ])
            pair_probs = pair_probs / pair_probs.sum()
            voted_for.append(rng.choice([model_a, model_b], p=pair_probs))

    return voted_for


# ---------------------------------------------------------------------------
# Attack cost model (Section 4.1)
# ---------------------------------------------------------------------------

@dataclass
class AttackCost:
    """Breakdown of attack cost components (Section 4.1)."""
    detector_training_cost: float
    account_maintenance_cost: float
    action_cost: float
    total_cost: float
    num_accounts_needed: int


def compute_attack_cost(
    num_actions: int,
    max_actions_per_account: int,
    cost_per_account: float,
    cost_per_action: float,
    detector_training_cost: float = 440.0,
) -> AttackCost:
    """
    Compute the total attack cost using the cost model from Section 4.1.

    Total cost = ceil(N/m) * c_account + N * c_action + c_detector

    Args:
        num_actions: Total number of actions N required.
        max_actions_per_account: Maximum actions per account m.
        cost_per_account: Cost per account c_account.
        cost_per_action: Cost per action c_action.
        detector_training_cost: One-time detector training cost c_detector.

    Returns:
        AttackCost breakdown.
    """
    import math
    num_accounts = math.ceil(num_actions / max_actions_per_account)
    account_cost = num_accounts * cost_per_account
    action_cost = num_actions * cost_per_action
    total = detector_training_cost + account_cost + action_cost

    return AttackCost(
        detector_training_cost=detector_training_cost,
        account_maintenance_cost=account_cost,
        action_cost=action_cost,
        total_cost=total,
        num_accounts_needed=num_accounts,
    )


def compute_detector_training_cost(
    num_prompts: int = 200,
    num_proprietary_models: int = 10,
    num_opensource_models: int = 20,
    tokens_per_response: int = 512,
    responses_per_model: int = 50,
    proprietary_cost_per_million: float = 5.00,
    opensource_cost_per_million: float = 1.80,
) -> float:
    """
    Estimate the cost of building the training-based detector (Appendix A.3).

    Upper bound cost per prompt:
      Proprietary: $5.00 × (512 × 50 / 10^6) = $0.128 per model
      Open-source: $1.80 × (512 × 50 / 10^6) = $0.046 per model

    Total per prompt ≈ $2.2 for 10 proprietary + 20 open-source models.
    Total for 200 prompts ≈ $440.

    Returns:
        Estimated total cost in USD.
    """
    tokens_per_model = tokens_per_response * responses_per_model
    cost_proprietary = proprietary_cost_per_million * (tokens_per_model / 1e6)
    cost_opensource = opensource_cost_per_million * (tokens_per_model / 1e6)

    cost_per_prompt = (
        num_proprietary_models * cost_proprietary
        + num_opensource_models * cost_opensource
    )
    return cost_per_prompt * num_prompts
